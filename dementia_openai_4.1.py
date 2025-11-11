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
from langchain.chat_models import init_chat_model
from langchain_openai import OpenAIEmbeddings
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
if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")

# Initialize LLM and embeddings
llm = init_chat_model("gpt-4.1", model_provider="openai")
# Use ChatOpenAI directly for vision to ensure proper image handling
from langchain_openai import ChatOpenAI
vision_llm = ChatOpenAI(model="gpt-4o", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")

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
        # Validate inputs are hashable (strings, not lists or other unhashable types)
        if not isinstance(subject, str):
            subject = str(subject) if subject else "unknown"
        if not isinstance(predicate, str):
            predicate = str(predicate) if predicate else "related_to"
        if not isinstance(object_, str):
            object_ = str(object_) if object_ else "unknown"

        # Skip empty or invalid triplets
        if not subject or not predicate or not object_:
            return

        try:
            self.graph.add_edge(subject, object_, relation=predicate, doc_id=doc_id)

            if doc_id:
                for entity in [subject, object_]:
                    if entity not in self.entity_to_docs:
                        self.entity_to_docs[entity] = set()
                    self.entity_to_docs[entity].add(doc_id)
        except TypeError as e:
            print(f"[WARNING] Skipping invalid triplet: ({type(subject).__name__}, {type(predicate).__name__}, {type(object_).__name__})")
            print(f"[WARNING] Values: ({subject}, {predicate}, {object_})")
            print(f"[WARNING] Error: {e}")

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
            print(f"[DEBUG] LLM response (first 200 chars): {content[:200]}")

            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]

            triplets_raw = json.loads(content)
            print(f"[DEBUG] Parsed {len(triplets_raw)} raw triplets from LLM")

            # Validate and sanitize triplets
            triplets = []
            for item in triplets_raw:
                if not isinstance(item, (list, tuple)) or len(item) != 3:
                    continue  # Skip malformed triplets

                s, p, o = item

                # Ensure all parts are strings (not lists or other types)
                if isinstance(s, list):
                    s = str(s) if s else "unknown"
                elif not isinstance(s, str):
                    s = str(s)

                if isinstance(p, list):
                    p = str(p) if p else "related_to"
                elif not isinstance(p, str):
                    p = str(p)

                if isinstance(o, list):
                    o = str(o) if o else "unknown"
                elif not isinstance(o, str):
                    o = str(o)

                # Only add if all parts are non-empty strings
                if s and p and o and isinstance(s, str) and isinstance(p, str) and isinstance(o, str):
                    triplets.append((s.strip(), p.strip(), o.strip()))

            return triplets
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []

    def extract(self, text: str, use_llm: bool = True) -> List[Tuple[str, str, str]]:
        """Extract entities and relationships using both methods."""
        triplets = self.extract_with_spacy(text)
        print(f"[DEBUG] spaCy extracted {len(triplets)} triplets")

        if use_llm:
            llm_triplets = self.extract_with_llm(text)
            print(f"[DEBUG] LLM extracted {len(llm_triplets)} triplets")
            triplets.extend(llm_triplets)

        print(f"[DEBUG] Total extracted {len(triplets)} triplets")
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

        # Debug: Check vector store contents
        try:
            collection = self.vector_store._collection
            count = collection.count()
            print(f"[DEBUG] Vector store initialized. Document count: {count}")
        except Exception as e:
            print(f"[DEBUG] Could not check vector store count: {e}")

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
            if i == 0:  # Debug first chunk
                print(f"[DEBUG] Original metadata keys: {chunk.metadata.keys()}")
                print(f"[DEBUG] Original metadata: {chunk.metadata}")

            for key, value in chunk.metadata.items():
                # Only keep simple hashable types
                if isinstance(value, (str, int, float, bool)):
                    clean_metadata[key] = value
                elif isinstance(value, list):
                    # Convert lists to strings
                    if i == 0:
                        print(f"[DEBUG] Converting list metadata '{key}': {value}")
                    clean_metadata[key] = str(value)
                elif isinstance(value, dict):
                    # Convert dicts to strings
                    if i == 0:
                        print(f"[DEBUG] Converting dict metadata '{key}': {value}")
                    clean_metadata[key] = str(value)
                elif value is None:
                    clean_metadata[key] = ""
                else:
                    # Convert any other types to string
                    if i == 0:
                        print(f"[DEBUG] Converting other type '{key}' ({type(value)}): {value}")
                    clean_metadata[key] = str(value)
            clean_metadata["chunk_id"] = f"doc_{i}"
            chunk.metadata = clean_metadata

            if i == 0:  # Debug first chunk
                print(f"[DEBUG] Cleaned metadata: {chunk.metadata}")

        # Add to vector store
        print("Adding documents to vector store...")
        try:
            # Generate explicit IDs for each chunk to avoid Chroma auto-generation issues
            chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
            self.vector_store.add_documents(chunks, ids=chunk_ids)
        except Exception as e:
            print(f"Vector store error: {e}")
            import traceback
            traceback.print_exc()
            # Fallback: add one by one
            for idx, chunk in enumerate(chunks):
                try:
                    self.vector_store.add_documents([chunk], ids=[f"chunk_{idx}"])
                except Exception as e2:
                    print(f"Failed to add chunk {idx}: {e2}")
                    print(f"Chunk metadata: {chunk.metadata}")
                    print(f"Metadata types: {[(k, type(v)) for k, v in chunk.metadata.items()]}")

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
            filepath = os.path.join(self.persist_directory, "graph_data.pkl")

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
            filepath = os.path.join(self.persist_directory, "graph_data.pkl")

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
        try:
            print(f"[DEBUG] Attempting vector search with query: {query[:100]}...")
            vector_results = self.vector_store.similarity_search(query, k=k)
            print(f"[DEBUG] Vector search returned {len(vector_results)} results")
        except Exception as e:
            print(f"[ERROR] Vector search failed: {e}")
            import traceback
            traceback.print_exc()
            vector_results = []

        if not use_graph:
            return vector_results

        try:
            # Use both spaCy and LLM extraction for better entity detection
            print(f"[DEBUG] Extracting entities from query...")
            query_entities = self.extractor.extract(query, use_llm=True)  # Set to True to enable LLM extraction
            print(f"[DEBUG] Raw query entities: {query_entities}")
            query_entity_names = set([ent[0] for ent in query_entities])
            print(f"[DEBUG] Extracted entity names from query: {query_entity_names}")

            graph_entities = set()
            for entity in query_entity_names:
                neighbors = self.knowledge_graph.get_neighbors(entity, depth=2)
                if neighbors:
                    print(f"[DEBUG] Entity '{entity}' has {len(neighbors)} neighbors in graph")
                    graph_entities.update(neighbors)
                else:
                    print(f"[DEBUG] Entity '{entity}' not found in graph")

            # If no entities found, try keyword-based matching as fallback
            if len(query_entity_names) == 0:
                print(f"[DEBUG] No entities extracted from query. Trying keyword-based fallback...")
                # Extract important keywords from query
                keywords = [word.lower().strip() for word in query.split()
                           if len(word) > 3 and word.lower() not in {'what', 'how', 'when', 'where', 'which', 'who', 'are', 'the', 'for', 'and', 'with', 'about'}]
                print(f"[DEBUG] Keywords extracted: {keywords}")

                # Look for exact or partial matches in graph nodes
                graph_nodes = list(self.knowledge_graph.graph.nodes())
                for keyword in keywords:
                    for node in graph_nodes:
                        if keyword in node.lower():
                            neighbors = self.knowledge_graph.get_neighbors(node, depth=2)
                            if neighbors:
                                print(f"[DEBUG] Keyword '{keyword}' matched node '{node}' with {len(neighbors)} neighbors")
                                graph_entities.update(neighbors)
                                break  # Found a match for this keyword

            print(f"[DEBUG] Total graph entities found: {len(graph_entities)}")

            graph_doc_ids = self.knowledge_graph.get_related_docs(graph_entities)
            print(f"[DEBUG] Graph doc IDs: {len(graph_doc_ids)}")

            graph_results = [self.documents[doc_id] for doc_id in graph_doc_ids if doc_id in self.documents]
            print(f"[DEBUG] Graph search returned {len(graph_results)} results")
        except Exception as e:
            print(f"[ERROR] Graph search failed: {e}")
            import traceback
            traceback.print_exc()
            graph_results = []

        all_results = vector_results + graph_results

        seen = set()
        unique_results = []
        for doc in all_results:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_results.append(doc)

        print(f"[DEBUG] Total unique results after deduplication: {len(unique_results)}")
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

                # Check if vector store actually has documents
                try:
                    vector_count = self.graph_rag.vector_store._collection.count()
                    print(f"Vector store has {vector_count} embeddings")
                except Exception as e:
                    print(f"Could not check vector store: {e}")
                    vector_count = 0

                # Try to load saved knowledge graph and documents
                if self.graph_rag.load_graph_data():
                    # Verify vector store has data
                    if vector_count == 0:
                        print("\n⚠️ WARNING: Vector store is empty! Need to reprocess PDFs to create embeddings.")
                        print("Processing PDFs to populate vector store...")

                        # Load and process PDFs to populate vector store
                        loader = PyPDFDirectoryLoader(self.pdf_folder)
                        documents = loader.load()

                        if documents:
                            # Only add to vector store, knowledge graph already loaded
                            chunks = self.graph_rag.text_splitter.split_documents(documents)

                            # Clean metadata - remove unhashable types
                            for i, chunk in enumerate(chunks):
                                clean_metadata = {}
                                for key, value in chunk.metadata.items():
                                    if isinstance(value, (str, int, float, bool)):
                                        clean_metadata[key] = value
                                    elif isinstance(value, list):
                                        clean_metadata[key] = str(value)
                                    elif value is None:
                                        clean_metadata[key] = ""
                                    else:
                                        # Convert any other types to string
                                        clean_metadata[key] = str(value)
                                clean_metadata["chunk_id"] = f"doc_{i}"
                                chunk.metadata = clean_metadata

                            print(f"Adding {len(chunks)} chunks to vector store...")
                            try:
                                # Generate explicit IDs for each chunk
                                chunk_ids = [f"chunk_{i}" for i in range(len(chunks))]
                                self.graph_rag.vector_store.add_documents(chunks, ids=chunk_ids)
                            except Exception as e:
                                print(f"Error adding documents: {e}")
                                import traceback
                                traceback.print_exc()
                                # Try adding one by one to identify problematic chunks
                                for idx, chunk in enumerate(chunks):
                                    try:
                                        self.graph_rag.vector_store.add_documents([chunk], ids=[f"chunk_{idx}"])
                                    except Exception as e2:
                                        print(f"Failed to add chunk {idx}: {e2}")
                                        print(f"Problematic metadata: {chunk.metadata}")
                                        print(f"Metadata types: {[(k, type(v)) for k, v in chunk.metadata.items()]}")
                            print("✅ Vector store populated!")

                        self.is_loaded = True
                        return f"✅ Loaded all data from disk and populated vector store!\n\nKnowledge Graph Stats:\n- Entities: {len(self.graph_rag.knowledge_graph.graph.nodes())}\n- Relationships: {len(self.graph_rag.knowledge_graph.graph.edges())}\n- Documents: {len(self.graph_rag.documents)}\n- Vector Embeddings: {len(chunks)}\n\n⚡ Ready for queries!"

                    self.is_loaded = True
                    return f"✅ Loaded all data from disk (vector store + knowledge graph)!\n\nKnowledge Graph Stats:\n- Entities: {len(self.graph_rag.knowledge_graph.graph.nodes())}\n- Relationships: {len(self.graph_rag.knowledge_graph.graph.edges())}\n- Documents: {len(self.graph_rag.documents)}\n- Vector Embeddings: {vector_count}\n\n⚡ Fast load - no PDF processing needed!"
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
            import traceback
            error_details = traceback.format_exc()
            print(f"[ERROR] Full traceback:\n{error_details}")

            # If it's a hashable type error, suggest clearing the database
            if "unhashable type" in str(e):
                return f"""❌ Error loading PDFs: {str(e)}

This error is likely caused by corrupted data in the existing Chroma database.

SOLUTION: Delete the '{self.persist_directory}' folder and try again.

This will force a fresh reload of all PDFs with the corrected metadata handling.

Full error details:
{error_details}"""

            return f"❌ Error loading PDFs: {str(e)}\n\nFull error:\n{error_details}"

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

            # Load image with PIL to ensure it's valid and convert to RGB
            img = Image.open(image_path)
            # Convert to RGB if necessary (handles RGBA, P, etc.)
            if img.mode != 'RGB':
                img = img.convert('RGB')

            # Resize if too large (max 2000px on longest side)
            max_size = 2000
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"Resized image to: {new_size}")

            # Save to bytes and encode
            import io
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format='JPEG', quality=95)
            image_bytes = img_byte_arr.getvalue()
            image_data = base64.b64encode(image_bytes).decode('utf-8')

            print(f"Image processed successfully. Base64 length: {len(image_data)}")

            # Get relevant guidelines from the knowledge base
            print("Retrieving dementia guidelines from knowledge base...")
            guidelines_queries = [
                "What are the key design principles, lighting, color schemes, flooring, furniture, and safety features for dementia-friendly spaces?",
                "What are specific color recommendations and contrast requirements for dementia care?",
                "What are safety features and accessibility requirements for dementia-friendly homes?"
            ]

            all_guidelines = []
            for query in guidelines_queries:
                results = self.graph_rag.retrieve(query, k=4, use_graph=True)
                all_guidelines.extend(results)

            # Deduplicate based on content
            seen_content = set()
            unique_guidelines = []
            for doc in all_guidelines:
                if doc.page_content not in seen_content:
                    seen_content.add(doc.page_content)
                    unique_guidelines.append(doc)

            guidelines_context = "\n\n".join([doc.page_content for doc in unique_guidelines])
            print(f"Retrieved {len(unique_guidelines)} unique guideline documents")

            # Create prompt for vision model
            prompt_text = f"""TASK: Analyze the provided image of a home interior space comprehensively for dementia-friendly design compliance.

You are an expert occupational therapist specializing in dementia care environments. You MUST examine the actual image provided and give specific, actionable feedback about what you observe.

REFERENCE GUIDELINES FROM KNOWLEDGE BASE:
{guidelines_context}

COMPREHENSIVE ASSESSMENT AREAS:

1. **CONTRAST & VISUAL CLARITY**
   - Color contrast between walls, floors, doors, furniture, fixtures
   - Ability to distinguish boundaries and key elements
   - Visual cues for navigation and safety

2. **LIGHTING**
   - Natural light sources and quality
   - Artificial lighting (type, placement, adequacy)
   - Glare from reflective surfaces or windows
   - Shadows that could cause confusion
   - Even lighting distribution

3. **FLOORING & SURFACES**
   - Patterns that might be confusing or disorienting
   - Reflective surfaces that could appear wet
   - Color changes that might look like steps
   - Slip resistance and trip hazards
   - Transitions between different floor types

4. **WAYFINDING & ORIENTATION**
   - Clear pathways and circulation
   - Visual landmarks and signage
   - Ability to identify room purpose
   - Memory cues and familiar elements

5. **SAFETY FEATURES**
   - Grab bars and handrails (presence, contrast, accessibility)
   - Sharp corners or hazardous edges
   - Clutter or obstacles in pathways
   - Secure furniture that won't tip
   - Window and door safety

6. **COLOR SCHEME**
   - Calming vs. stimulating colors
   - Appropriate color choices per guidelines
   - Avoidance of problematic colors (dark floors, busy patterns)
   - Use of color for wayfinding

7. **SPATIAL DESIGN**
   - Room layout and furniture arrangement
   - Clear sightlines and open spaces
   - Accessibility and ease of movement
   - Reduction of visual clutter

8. **ENVIRONMENTAL CUES**
   - Mirrors that could cause confusion
   - Reflective or glass surfaces
   - Patterns that create visual disturbance
   - Decor that supports orientation

INSTRUCTIONS:
1. DO NOT include any disclaimers about your capabilities or inability to analyze images - you can and will analyze this image
2. Start immediately with a brief description of what you see in the image (room type, main features)
3. Systematically assess ALL 8 areas above based on what's visible in the image
4. For each issue found, provide specific details
5. Output findings in BOTH markdown and JSON formats

OUTPUT FORMAT:

**ROOM DESCRIPTION:**
[Brief description of what you see - room type, main elements, overall impression]

---

**ISSUES IDENTIFIED:**

For each issue:

**Item:** [Name the PHYSICAL ITEM/ELEMENT only - e.g., "Sofa", "Floor tiles", "Ceiling light", "Glass partition door", "Wall paint", "Window blinds", "TV cabinet". DO NOT list abstract concepts like "sofa color" or "lighting level" - list the actual object]
**Category:** [One of: Contrast, Lighting, Flooring, Wayfinding, Safety, Color, Spatial Design, Environmental Cues]
**Issue:** [Detailed description of the problem with this specific item - be specific about colors, materials observed in the image]
**Guideline Reference:** [Quote the SPECIFIC, EXACT text from the reference guidelines above that addresses this issue]
**Recommendation:** [You are the dementia design expert. Give ONE SPECIFIC directive with exact specifications. Format: "ACTION the ITEM with SPECIFIC-MATERIAL/COLOR/SPECS." Examples:
  - "Replace sofa with one upholstered in solid burgundy fabric (LRV 15-25)."
  - "Repaint walls in warm cream (LRV 70-75) with matte finish."
  - "Install LED recessed lights providing 450 lux at 3000K color temperature."
  Make the decision - do NOT offer choices or say "consider" or "could". State exactly what should be done. NO explanations in this field - just the directive.]
**Explanation:** [Explain WHY this change is needed for people with dementia - reference the guideline principle and explain the benefit, e.g., "This creates sufficient visual contrast making the sofa clearly visible, reducing fall risk and supporting independent movement as per dementia-friendly design principles."]

---

**JSON SUMMARY:**

```json
{{
  "room_type": "description",
  "analysis_summary": {{
    "total_issues": <number>,
    "by_category": {{
      "contrast": <number>,
      "lighting": <number>,
      "flooring": <number>,
      "wayfinding": <number>,
      "safety": <number>,
      "color": <number>,
      "spatial_design": <number>,
      "environmental_cues": <number>
    }}
  }},
  "issues": [
    {{
      "item": "Physical item name only (e.g., 'Sofa', 'Floor tiles', 'Door')",
      "category": "Category name",
      "issue": "Description of the issue with this item",
      "guideline_reference": "Exact quote from reference guidelines",
      "recommendation": "Direct instruction only (e.g., 'Replace sofa with one upholstered in burgundy fabric (LRV 15-25).')",
      "explanation": "Why this change benefits people with dementia (e.g., 'Creates visual contrast for fall prevention and supports independent movement.')"
    }}
  ]
}}
```

IMPORTANT GUIDELINES:
- Analyze the ACTUAL image provided - describe specific colors, materials, and features you see
- Consider ALL aspects of dementia-friendly design, not just contrast
- Be specific about what you observe (e.g., "dark grey sofa on dark grey floor" not just "poor contrast")
- **CRITICAL: Quote EXACT text from the REFERENCE GUIDELINES above for each issue** - don't paraphrase
- **CRITICAL: Recommendations must be SPECIFIC, AUTHORITATIVE, and EVIDENCE-BASED:**
  * You are the dementia design expert - make definitive decisions
  * Give ONE specific value, not ranges (e.g., "LRV 45" not "LRV 40-50")
  * Give ONE specific color, not choices (e.g., "burgundy" not "burgundy or navy")
  * Give ONE specific action, not suggestions (e.g., "Replace" not "Consider replacing")
  * Use specific color names, materials, product types from the guidelines
  * Include exact technical specifications (e.g., LRV values, lux levels, color temperatures)
  * Provide measurable, actionable directives (e.g., "Install LED lights at 450 lux" not "add more light")
  * NO explanatory text in the Recommendation field - save explanations for the Explanation field
- Read through ALL the reference guidelines carefully and apply them to what you see
- Include both positive observations and areas for improvement
- If guidelines mention specific products, materials, or standards, use them in recommendations

CRITICAL RULES FOR ITEM NAMING:
✅ CORRECT: "Sofa", "Floor tiles", "Walls", "Door", "Light fixture", "Window blinds", "TV stand", "Glass partition"
❌ WRONG: "Sofa color", "Flooring color", "Wall color", "Lighting level", "Contrast between sofa and floor"

EXAMPLES OF SPECIFIC RECOMMENDATIONS:

**Example 1:**
❌ BAD (wrong item name): Item: "Lighting level"
❌ BAD (has explanation, gives range): Recommendation: "Replace existing ceiling lights with LED downlights providing 300-500 lux as per guidelines for living spaces, with warm white color temperature (2700-3000K) to reduce agitation"
✅ GOOD:
**Item:** Ceiling light fixtures
**Recommendation:** Install LED recessed downlights providing 450 lux at 3000K color temperature.

**Example 2:**
❌ BAD (abstract concept): Item: "Door-wall contrast"
✅ GOOD:
**Item:** Bedroom door
**Recommendation:** Paint door in dark navy (LRV 25).

**Example 3:**
❌ BAD (mentions color not item): Item: "Sofa color"
❌ BAD (gives choices, has explanations): Recommendation: "Replace with a sofa upholstered in solid burgundy or navy fabric (LRV 15-25) to create strong contrast against the light grey floor (LRV 65), making the seating clearly visible"
✅ GOOD (complete example with all fields):
**Item:** Sofa
**Category:** Contrast
**Issue:** The grey sofa (LRV 45) has insufficient contrast against the grey floor (LRV 40), making it difficult to distinguish the seating area.
**Guideline Reference:** "Furniture should contrast with flooring to ensure clear visibility and prevent falls."
**Recommendation:** Replace sofa with one upholstered in solid burgundy fabric (LRV 18).
**Explanation:** This creates a 22 LRV point contrast difference against the floor, making the sofa clearly visible and reducing fall risk when sitting or standing, supporting safe independent movement.

**Example 4:**
❌ BAD (too vague): Recommendation: "Change the flooring"
❌ BAD (gives range): Recommendation: "Replace floor tiles with matte-finish vinyl flooring in solid mid-tone beige (LRV 40-50) without patterns"
✅ GOOD:
**Item:** Floor tiles
**Recommendation:** Replace floor tiles with matte-finish vinyl flooring in solid beige (LRV 45) without patterns.

REMEMBER: Do not say you cannot analyze images or provide disclaimers. You will analyze this image directly and provide specific findings.

Begin your comprehensive analysis now:"""

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
            print("Sending request to GPT-4o vision model...")
            response = vision_llm.invoke([message])
            print("Received response from vision model")

            return response.content

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            return f"❌ Error analyzing image: {str(e)}\n\nDetails:\n{error_details}"


# Global chatbot instance
pdf_folder = r"Dementia Guidelines"
chatbot = DementiaChatbot(pdf_folder)


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
        ### Powered by GraphRAG (Knowledge Graph + Vector Search)

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
