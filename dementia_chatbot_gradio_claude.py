"""
Dementia Guidelines Chatbot with Gradio Interface
"""

import os
import getpass
from pathlib import Path
from typing import List, Dict, Tuple, Set
import json
import gradio as gr
import base64
import pickle
import tempfile
from PIL import Image

# LangChain imports
from langchain_anthropic import ChatAnthropic
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import HumanMessage

# Graph imports
import networkx as nx
import spacy

# Load environment
if not os.environ.get("ANTHROPIC_API_KEY"):
    os.environ["ANTHROPIC_API_KEY"] = getpass.getpass("Enter API key for Anthropic: ")

# Initialize LLM and embeddings
llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)
vision_llm = ChatAnthropic(model="claude-3-5-sonnet-20241022", temperature=0)  # Claude supports vision
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Load spaCy model
try:
    nlp = spacy.load("en_core_web_sm")
except:
    print("Downloading spaCy model...")
    os.system("python -m spacy download en_core_web_sm")
    nlp = spacy.load("en_core_web_sm")


class KnowledgeGraph:
    """Knowledge Graph for storing entities and relationships."""

    def __init__(self):
        self.graph = nx.MultiDiGraph()
        self.entity_to_docs = {}

    def add_triplet(self, subject: str, predicate: str, object_: str, doc_id: str = None):
        """Add a subject-predicate-object triplet to the graph."""
        self.graph.add_edge(subject, object_, relation=predicate, doc_id=doc_id)

        if doc_id:
            for entity in [subject, object_]:
                if entity not in self.entity_to_docs:
                    self.entity_to_docs[entity] = set()
                self.entity_to_docs[entity].add(doc_id)

    def get_neighbors(self, entity: str, depth: int = 1) -> Set[str]:
        """Get neighboring entities up to specified depth."""
        if entity not in self.graph:
            return set()

        neighbors = set([entity])
        current_level = {entity}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                next_level.update(self.graph.predecessors(node))
                next_level.update(self.graph.successors(node))
            neighbors.update(next_level)
            current_level = next_level

        return neighbors

    def get_related_docs(self, entities: List[str]) -> Set[str]:
        """Get document IDs related to given entities."""
        related_docs = set()
        for entity in entities:
            if entity in self.entity_to_docs:
                related_docs.update(self.entity_to_docs[entity])
        return related_docs

    def get_entity_context(self, entity: str) -> str:
        """Get textual context about an entity from the graph."""
        if entity not in self.graph:
            return ""

        context_parts = []

        for successor in self.graph.successors(entity):
            edges = self.graph.get_edge_data(entity, successor)
            for edge_data in edges.values():
                relation = edge_data.get('relation', 'related_to')
                context_parts.append(f"{entity} {relation} {successor}")

        for predecessor in self.graph.predecessors(entity):
            edges = self.graph.get_edge_data(predecessor, entity)
            for edge_data in edges.values():
                relation = edge_data.get('relation', 'related_to')
                context_parts.append(f"{predecessor} {relation} {entity}")

        return "; ".join(context_parts) if context_parts else ""


class EntityRelationExtractor:
    """Extract entities and relationships from text."""

    def __init__(self, llm):
        self.llm = llm
        self.extraction_prompt = PromptTemplate(
            input_variables=["text"],
            template="""Extract entities and relationships from the following medical/healthcare text about dementia.
Return a JSON array of triplets in the format: [["subject", "predicate", "object"], ...]

Focus on:
- Medical conditions, symptoms, treatments
- Healthcare guidelines and recommendations
- Patient care instructions
- Relationships between conditions and treatments
- Risk factors and preventive measures

Text: {text}

JSON (only the array, no additional text):"""
        )

    def extract_with_spacy(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract entities using spaCy NER."""
        doc = nlp(text)
        triplets = []

        entities = [(ent.text, ent.label_) for ent in doc.ents]

        for ent_text, ent_type in entities:
            triplets.append((ent_text, "is_a", ent_type))

        return triplets

    def extract_with_llm(self, text: str) -> List[Tuple[str, str, str]]:
        """Extract entities and relationships using LLM."""
        try:
            prompt = self.extraction_prompt.format(text=text[:2000])
            response = self.llm.invoke(prompt)

            content = response.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            triplets_raw = json.loads(content)
            triplets = [(s, p, o) for s, p, o in triplets_raw]
            return triplets
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []

    def extract(self, text: str, use_llm: bool = True) -> List[Tuple[str, str, str]]:
        """Extract entities and relationships using both methods."""
        triplets = self.extract_with_spacy(text)

        if use_llm:
            llm_triplets = self.extract_with_llm(text)
            triplets.extend(llm_triplets)

        return triplets


class GraphRAG:
    """Hybrid RAG system combining vector search with knowledge graph."""

    def __init__(self, llm, embeddings, persist_directory="./chroma_db"):
        self.llm = llm
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embeddings
        )
        self.knowledge_graph = KnowledgeGraph()
        self.extractor = EntityRelationExtractor(llm)
        self.documents = {}
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

    def add_documents(self, documents: List[Document], use_llm_extraction: bool = True):
        """Add documents to both vector store and knowledge graph."""
        print(f"Processing {len(documents)} documents...")

        chunks = self.text_splitter.split_documents(documents)

        # Clean metadata - remove any lists or unhashable types
        for i, chunk in enumerate(chunks):
            clean_metadata = {}
            for key, value in chunk.metadata.items():
                # Only keep simple hashable types
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                elif isinstance(value, list):
                    # Convert lists to strings
                    clean_metadata[key] = str(value)
            clean_metadata["chunk_id"] = f"doc_{i}"
            chunk.metadata = clean_metadata

        # Add to vector store
        print("Adding documents to vector store...")
        try:
            self.vector_store.add_documents(chunks)
        except Exception as e:
            print(f"Vector store error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: add one by one
            for idx, chunk in enumerate(chunks):
                try:
                    self.vector_store.add_documents([chunk])
                except Exception as e2:
                    print(f"Failed to add chunk {idx}: {e2}")

        print("Building knowledge graph...")
        for i, chunk in enumerate(chunks):
            doc_id = f"doc_{i}"
            self.documents[doc_id] = chunk

            print(f"Extracting entities from chunk {i+1}/{len(chunks)}...", end='\r')

            triplets = self.extractor.extract(chunk.page_content, use_llm=use_llm_extraction)

            for subject, predicate, obj in triplets:
                self.knowledge_graph.add_triplet(subject, predicate, obj, doc_id)

        print(f"\nProcessed {len(chunks)} chunks. Graph has {len(self.knowledge_graph.graph.nodes())} entities.")

    def save_graph_data(self, filepath: str = None):
        """Save knowledge graph and documents to disk."""
        if filepath is None:
            filepath = os.path.join(self.persist_directory, "graph_data_claude.pkl")

        data = {
            "knowledge_graph": self.knowledge_graph,
            "documents": self.documents
        }

        with open(filepath, 'wb') as f:
            pickle.dump(data, f)

        print(f"Saved knowledge graph data to {filepath}")

    def load_graph_data(self, filepath: str = None) -> bool:
        """Load knowledge graph and documents from disk."""
        if filepath is None:
            filepath = os.path.join(self.persist_directory, "graph_data_claude.pkl")

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.knowledge_graph = data["knowledge_graph"]
            self.documents = data["documents"]

            print(f"Loaded knowledge graph data from {filepath}")
            print(f"Graph has {len(self.knowledge_graph.graph.nodes())} entities and {len(self.documents)} documents")
            return True
        except Exception as e:
            print(f"Error loading graph data: {e}")
            return False

    def retrieve(self, query: str, k: int = 4, use_graph: bool = True) -> List[Document]:
        """Hybrid retrieval using vector search and knowledge graph."""
        vector_results = self.vector_store.similarity_search(query, k=k)

        if not use_graph:
            return vector_results

        query_entities = self.extractor.extract_with_spacy(query)
        query_entity_names = set([ent[0] for ent in query_entities])

        graph_entities = set()
        for entity in query_entity_names:
            graph_entities.update(self.knowledge_graph.get_neighbors(entity, depth=2))

        graph_doc_ids = self.knowledge_graph.get_related_docs(graph_entities)
        graph_results = [self.documents[doc_id] for doc_id in graph_doc_ids if doc_id in self.documents]

        all_results = vector_results + graph_results

        seen = set()
        unique_results = []
        for doc in all_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_results.append(doc)

        return unique_results[:k*2]

    def query(self, question: str, use_graph: bool = True) -> Dict[str, any]:
        """Query the GraphRAG system."""
        retrieved_docs = self.retrieve(question, use_graph=use_graph)

        question_entities = self.extractor.extract_with_spacy(question)
        entity_names = [ent[0] for ent in question_entities]

        graph_context = []
        for entity in entity_names:
            context = self.knowledge_graph.get_entity_context(entity)
            if context:
                graph_context.append(context)

        doc_context = "\n\n".join([doc.page_content for doc in retrieved_docs])
        kg_context = "\n".join(graph_context) if graph_context else "No direct graph relationships found."

        prompt = f"""Answer the question based on the following context from Dementia Guidelines:

DOCUMENT CONTEXT:
{doc_context}

KNOWLEDGE GRAPH CONTEXT:
{kg_context}

QUESTION: {question}

Provide a comprehensive, medically accurate answer using both the document context and knowledge graph relationships.
If the information is not in the context, say so clearly:"""

        response = self.llm.invoke(prompt)

        return {
            "answer": response.content,
            "retrieved_docs": retrieved_docs,
            "graph_context": graph_context,
            "entities": entity_names
        }


class DementiaChatbot:
    """Interactive chatbot for Dementia Guidelines."""

    def __init__(self, pdf_folder: str, persist_directory: str = "./chroma_db"):
        self.pdf_folder = pdf_folder
        self.persist_directory = persist_directory
        self.graph_rag = None
        self.conversation_history = []
        self.is_loaded = False

    def load_pdfs(self, use_llm_extraction: bool = False, force_reload: bool = False):
        """Load PDFs from the Dementia Guidelines folder."""
        print(f"\nLoading PDFs from: {self.pdf_folder}")

        pdf_path = Path(self.pdf_folder)
        if not pdf_path.exists():
            return f"❌ Error: Folder '{self.pdf_folder}' does not exist!"

        try:
            # Initialize GraphRAG with persistence
            self.graph_rag = GraphRAG(llm=llm, embeddings=embeddings, persist_directory=self.persist_directory)

            # Check if vector store already has data
            persist_path = Path(self.persist_directory)
            has_existing_data = persist_path.exists() and any(persist_path.iterdir())

            if has_existing_data and not force_reload:
                print("Found existing Chroma database. Checking for saved graph data...")

                # Try to load saved knowledge graph and documents
                if self.graph_rag.load_graph_data():
                    self.is_loaded = True
                    return f"✅ Loaded all data from disk (vector store + knowledge graph)!\n\nKnowledge Graph Stats:\n- Entities: {len(self.graph_rag.knowledge_graph.graph.nodes())}\n- Relationships: {len(self.graph_rag.knowledge_graph.graph.edges())}\n- Documents: {len(self.graph_rag.documents)}\n\n⚡ Fast load - no PDF processing needed!"
                else:
                    print("No saved graph data found. Rebuilding knowledge graph from PDFs...")
                    self.is_loaded = True
                    # Need to rebuild knowledge graph
                    loader = PyPDFDirectoryLoader(self.pdf_folder)
                    documents = loader.load()
                    chunks = self.graph_rag.text_splitter.split_documents(documents)

                    print("Rebuilding knowledge graph...")
                    for i, chunk in enumerate(chunks):
                        doc_id = f"doc_{i}"
                        self.graph_rag.documents[doc_id] = chunk

                        print(f"Extracting entities from chunk {i+1}/{len(chunks)}...", end='\r')

                        triplets = self.graph_rag.extractor.extract(chunk.page_content, use_llm=use_llm_extraction)
                        for subject, predicate, obj in triplets:
                            self.graph_rag.knowledge_graph.add_triplet(subject, predicate, obj, doc_id)

                    # Save the graph data for next time
                    self.graph_rag.save_graph_data()

                    return f"✅ Loaded existing vector database from disk and rebuilt knowledge graph!\n\nKnowledge Graph Stats:\n- Entities: {len(self.graph_rag.knowledge_graph.graph.nodes())}\n- Relationships: {len(self.graph_rag.knowledge_graph.graph.edges())}\n\n💾 Knowledge graph saved for faster loading next time!"

            # Load PDFs and process
            loader = PyPDFDirectoryLoader(self.pdf_folder)
            documents = loader.load()

            if not documents:
                return f"❌ No PDF files found in '{self.pdf_folder}'"

            print(f"Loaded {len(documents)} pages from PDFs")

            self.graph_rag.add_documents(documents, use_llm_extraction=use_llm_extraction)

            # Save the knowledge graph data to disk
            self.graph_rag.save_graph_data()

            self.is_loaded = True
            return f"✅ Successfully loaded {len(documents)} pages from PDFs!\n\nKnowledge Graph Stats:\n- Entities: {len(self.graph_rag.knowledge_graph.graph.nodes())}\n- Relationships: {len(self.graph_rag.knowledge_graph.graph.edges())}\n\n💾 All data saved to: {self.persist_directory}\n⚡ Next time you run, everything will load instantly!"

        except Exception as e:
            return f"❌ Error loading PDFs: {str(e)}"

    def chat(self, message: str) -> str:
        """Send a message and get a response."""
        if not self.graph_rag or not self.is_loaded:
            return "⚠️ Please load the PDFs first using the 'Load PDFs' button."

        try:
            result = self.graph_rag.query(message, use_graph=True)

            self.conversation_history.append({
                "question": message,
                "answer": result["answer"],
                "entities": result["entities"]
            })

            return result["answer"]
        except Exception as e:
            return f"❌ Error: {str(e)}"

    def get_stats(self) -> str:
        """Get knowledge graph statistics."""
        if not self.graph_rag or not self.is_loaded:
            return "⚠️ Please load the PDFs first."

        kg = self.graph_rag.knowledge_graph
        stats = f"""📊 **Knowledge Graph Statistics**

**Total Entities:** {len(kg.graph.nodes())}
**Total Relationships:** {len(kg.graph.edges())}
**Conversation History:** {len(self.conversation_history)} messages

**Top 10 Most Connected Entities:**
"""
        degrees = dict(kg.graph.degree())
        top_entities = sorted(degrees.items(), key=lambda x: x[1], reverse=True)[:10]

        for i, (entity, degree) in enumerate(top_entities, 1):
            stats += f"\n{i}. {entity}: {degree} connections"

        return stats

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []
        return "✅ Conversation history cleared!"

    def analyze_image(self, image_path: str) -> str:
        """Analyze a home image against dementia guidelines."""
        if not self.graph_rag or not self.is_loaded:
            return "⚠️ Please load the PDFs first using the 'Load PDFs' button."

        try:
            # Verify image file exists and is readable
            if not os.path.exists(image_path):
                return f"❌ Image file not found: {image_path}"

            print(f"Reading image from: {image_path}")
            file_size = os.path.getsize(image_path)
            print(f"Image file size: {file_size} bytes")

            # Load and encode image
            with open(image_path, "rb") as image_file:
                image_bytes = image_file.read()
                image_data = base64.b64encode(image_bytes).decode('utf-8')

            print(f"Image encoded successfully. Base64 length: {len(image_data)}")

            # Get relevant guidelines from the knowledge base
            print("Retrieving dementia guidelines from knowledge base...")
            guidelines_query = "What are the key design principles, lighting, color schemes, flooring, furniture, and safety features for dementia-friendly spaces?"
            guidelines_result = self.graph_rag.retrieve(guidelines_query, k=6, use_graph=True)

            guidelines_context = "\n\n".join([doc.page_content for doc in guidelines_result])
            print(f"Retrieved {len(guidelines_result)} guideline documents")

            # Create prompt for vision model
            prompt_text = f"""You are an expert occupational therapist specializing in dementia-friendly home design. Carefully analyze this image of a home space and identify specific issues based on evidence-based dementia care guidelines.

REFERENCE GUIDELINES:
{guidelines_context}

HOW TO INTERPRET CONTRAST REQUIREMENTS FROM THE GUIDELINES:
The guidelines above mention "contrast," "visual contrast," "colour contrast," and "tonal contrast" - these all refer to the SAME principle:

✓ GOOD CONTRAST (desired) = Different/distinct colors or tones that make objects easily distinguishable
   - Examples: Dark door against light wall, contrasting handrails, distinct furniture colors
   - The guidelines say to "provide good visual contrast," "use contrasting colours," "ensure contrast"

✗ POOR/INSUFFICIENT CONTRAST (problem) = Similar colors or tones that blend together
   - Examples: White door on white wall, beige furniture on beige floor, light fixtures on light walls
   - Makes it hard for people with dementia to distinguish objects and boundaries

INSTRUCTIONS:
1. Look at EVERY visible element in the image
2. Compare each element's color/tone with its background (wall, floor, etc.)
3. Identify items that LACK sufficient contrast (similar colors that blend together)
4. Output your findings in BOTH formats:
   a) Human-readable markdown format
   b) JSON format

OUTPUT FORMAT:

First, provide a markdown analysis using this format for each item:

**Item:** [Name the specific item only - e.g., "White door", "Beige sofa"]
**Issue:** [Explain why this LACKS sufficient contrast according to the guidelines - mention the specific similar colors]
**Guideline Reference:** [Quote or paraphrase the relevant principle from the guidelines above]
**Recommendation:** [Give a DEFINITE, specific action - use "Change to...", "Replace with...", "Paint/Install..." format with HIGH CONTRAST colors to match guideline requirements]

---

Then, after all items, provide a JSON summary:

```json
{{
  "analysis_summary": {{
    "total_issues": <number>
  }},
  "issues": [
    {{
      "item": "Item name",
      "issue": "Description of the issue",
      "guideline_reference": "Referenced guideline",
      "recommendation": "Specific recommendation"
    }}
  ]
}}
```

CRITICAL ELEMENTS TO ASSESS:
✓ Doors & Frames: Do they have GOOD CONTRAST against walls? (Similar colors = problem)
✓ Furniture: Does it have GOOD CONTRAST against walls/floors? (Similar tones = problem)
✓ Fixtures: Do switches, outlets, handles have GOOD CONTRAST? (Blending in = problem)
✓ Floor-to-Wall transitions: Is there CLEAR CONTRAST between floor and skirting/wall? (Similar = problem)
✓ Lighting: Type, brightness, glare, shadows, natural light
✓ Flooring: Pattern, color, material, reflectivity, trip hazards
✓ Décor: Visual clutter, confusing patterns, mirrors
✓ Safety: Grab bars, clear pathways, hazards

IMPORTANT:
- ONLY flag items that LACK sufficient contrast as problems (don't flag good contrast as a problem!)
- Reference specific guideline principles from the context above in your analysis
- Be concrete about the exact colors/tones you see and why they lack contrast
- Analyze what you SEE in the image, not generic advice
- Provide BOTH the markdown analysis AND the JSON summary

Begin your analysis now:"""

            # Create message with image
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            )

            # Get analysis from vision model
            print("Sending request to Claude vision model...")
            response = vision_llm.invoke([message])
            print("Received response from vision model")

            return response.content

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"❌ Error analyzing image: {str(e)}\n\nDetails:\n{error_details}"


# Global chatbot instance
pdf_folder = r"Dementia Guidelines"
# Use separate database directory for Claude to avoid embedding dimension conflicts with OpenAI
chatbot = DementiaChatbot(pdf_folder, persist_directory="./chroma_db_claude")


def load_pdfs_handler(use_llm_extraction):
    """Handle PDF loading."""
    return chatbot.load_pdfs(use_llm_extraction=use_llm_extraction)


def chat_handler(message, history):
    """Handle chat messages."""
    if not message.strip():
        return history

    response = chatbot.chat(message)
    history.append((message, response))
    return history


def get_stats_handler():
    """Handle stats request."""
    return chatbot.get_stats()


def clear_history_handler():
    """Handle clear history."""
    msg = chatbot.clear_history()
    return [], msg


def analyze_image_handler(image):
    """Handle image analysis."""
    if image is None:
        return "⚠️ Please upload an image first."

    temp_path = None
    try:
        print("Starting image analysis...")

        # Create a temporary file to save the image
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
            temp_path = tmp.name

        # Handle different image types
        if isinstance(image, Image.Image):
            print("Processing PIL Image...")
            # PIL Image
            image.save(temp_path, format='JPEG')
        elif isinstance(image, str):
            print("Processing filepath...")
            # File path - copy to temp location
            img = Image.open(image)
            img.save(temp_path, format='JPEG')
        else:
            return f"❌ Unexpected image format: {type(image)}"

        print(f"Image saved to temp file: {temp_path}")

        # Analyze the image
        print("Analyzing image against dementia guidelines...")
        result = chatbot.analyze_image(temp_path)

        # Clean up
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
            print("Cleaned up temp file")

        return result

    except Exception as e:
        # Clean up on error
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except:
                pass

        import traceback
        error_details = traceback.format_exc()
        print(f"Error in image handler: {error_details}")
        return f"❌ Error analyzing image: {str(e)}\n\nDetails:\n{error_details}"


# Create Gradio Interface
def create_interface():
    with gr.Blocks(title="🧠 Dementia Guidelines Chatbot", theme=gr.themes.Soft()) as demo:
        gr.Markdown("""
        # 🧠 Dementia Guidelines Chatbot
        ### Powered by Claude 3.5 Sonnet + GraphRAG (Knowledge Graph + Vector Search)

        Ask questions about dementia care guidelines, design recommendations, and best practices.
        **NEW:** Upload images of your home for dementia-friendly design analysis!
        """)

        with gr.Row():
            with gr.Column(scale=2):
                # Chat Interface
                chatbot_ui = gr.Chatbot(
                    label="Chat",
                    height=500,
                    show_label=True,
                    bubble_full_width=False
                )

                with gr.Row():
                    msg_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Ask about dementia guidelines (e.g., 'What are key design principles for dementia-friendly spaces?')",
                        scale=4
                    )
                    submit_btn = gr.Button("Send", variant="primary", scale=1)

                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear Chat", size="sm")

            with gr.Column(scale=1):
                # Control Panel
                gr.Markdown("### 📋 Control Panel")

                with gr.Group():
                    gr.Markdown("**Step 1: Load PDFs**")
                    use_llm_check = gr.Checkbox(
                        label="Use LLM extraction (slower, more accurate)",
                        value=False
                    )
                    load_btn = gr.Button("📁 Load PDFs", variant="secondary")
                    load_status = gr.Textbox(
                        label="Status",
                        lines=8,
                        interactive=False
                    )

                with gr.Group():
                    gr.Markdown("**Step 2: Ask Questions**")
                    gr.Markdown("Use the chat interface to ask questions after loading PDFs.")

                with gr.Group():
                    gr.Markdown("**📊 Statistics**")
                    stats_btn = gr.Button("View Stats", size="sm")
                    stats_output = gr.Markdown("")

        # Image Analysis Section
        gr.Markdown("---")
        gr.Markdown("## 📸 Home Image Analysis")
        gr.Markdown("Upload an image of your home space to get dementia-friendly design recommendations.")

        with gr.Row():
            with gr.Column(scale=1):
                image_input = gr.Image(
                    label="Upload Home Image",
                    type="pil",
                    height=400
                )
                analyze_btn = gr.Button("🔍 Analyze Image", variant="primary", size="lg")

            with gr.Column(scale=1):
                image_analysis_output = gr.Markdown(
                    label="Analysis Results",
                    value="Upload an image and click 'Analyze Image' to see recommendations."
                )

        # Event handlers
        load_btn.click(
            fn=load_pdfs_handler,
            inputs=[use_llm_check],
            outputs=[load_status]
        )

        submit_btn.click(
            fn=chat_handler,
            inputs=[msg_input, chatbot_ui],
            outputs=[chatbot_ui]
        ).then(
            lambda: "",
            outputs=[msg_input]
        )

        msg_input.submit(
            fn=chat_handler,
            inputs=[msg_input, chatbot_ui],
            outputs=[chatbot_ui]
        ).then(
            lambda: "",
            outputs=[msg_input]
        )

        clear_btn.click(
            fn=clear_history_handler,
            outputs=[chatbot_ui, load_status]
        )

        stats_btn.click(
            fn=get_stats_handler,
            outputs=[stats_output]
        )

        analyze_btn.click(
            fn=analyze_image_handler,
            inputs=[image_input],
            outputs=[image_analysis_output],
            show_progress=True,
            api_name="analyze_image"
        )

        # Example questions
        gr.Markdown("""
        ### 💡 Example Questions:
        - What are the key principles for dementia-friendly design?
        - How should lighting be designed for people with dementia?
        - What are recommendations for outdoor spaces?
        - How can technology support dementia care?
        - What color schemes are recommended for dementia-friendly environments?
        """)

    return demo


if __name__ == "__main__":
    demo = create_interface()
    demo.queue()  # Enable queue to fix ASGI response issues
    demo.launch(
        server_name="127.0.0.1",
        server_port=7860,
        share=False,
        show_error=True
    )
