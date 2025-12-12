"""
Dementia Home Safety Image Analysis Pipeline
===========================================

This pipeline analyzes home images for dementia safety using:
1. Vector Store + Knowledge Graph (RAG) - Loaded from dementia care guidelines PDFs
2. Vision Language Model (GPT-4o) - Analyzes the image
3. Combined Analysis - VLM results enhanced with RAG-retrieved guidelines

Pipeline Flow:
Image Input → VLM Analysis → RAG Enhancement → Final Output (Analysis Report)

Usage:
    python dementia_image_analysis_pipeline.py --image "path/to/image.jpg"
    python dementia_image_analysis_pipeline.py --image "path/to/image.jpg" --output "analysis.txt"
"""

import os
import sys
import argparse
import pickle
from typing import List, Dict, Tuple, Set
import base64
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage

# Knowledge Graph
import networkx as nx

# NLP for entity extraction
import spacy

# Image processing
from PIL import Image

# Environment setup
from dotenv import load_dotenv
load_dotenv()


# ============================================================================
# KNOWLEDGE GRAPH CLASS
# ============================================================================

class KnowledgeGraph:
    """
    Stores entities and their relationships as a directed graph.

    Example:
        Graph node: "dementia patient"
        Edge: "needs" → "consistent lighting"
        Document: PDF page reference
    """

    def __init__(self):
        # MultiDiGraph allows multiple edges between same nodes
        self.graph = nx.MultiDiGraph()
        # Track which documents mention which entities
        self.entity_docs = {}

    def add_triplet(self, subject: str, predicate: str, object_: str, doc_id: str = None):
        """
        Add a relationship triplet to the graph.

        Args:
            subject: The entity (e.g., "patient")
            predicate: The relationship (e.g., "needs")
            object_: The related entity (e.g., "lighting")
            doc_id: Source document reference
        """
        # Normalize to lowercase for consistency
        subject = subject.lower().strip()
        object_ = object_.lower().strip()

        # Add edge to graph with relationship label
        self.graph.add_edge(subject, object_, label=predicate, doc_id=doc_id)

        # Track document associations
        if doc_id:
            if subject not in self.entity_docs:
                self.entity_docs[subject] = set()
            if object_ not in self.entity_docs:
                self.entity_docs[object_] = set()
            self.entity_docs[subject].add(doc_id)
            self.entity_docs[object_].add(doc_id)

    def get_neighbors(self, entity: str, depth: int = 1) -> Set[str]:
        """
        Get all entities connected to this entity within specified depth.

        Args:
            entity: The entity to search from
            depth: How many hops to traverse (1 = direct neighbors)

        Returns:
            Set of related entity names
        """
        entity = entity.lower().strip()
        if entity not in self.graph:
            return set()

        # Use breadth-first search to find neighbors
        neighbors = set()
        current_level = {entity}

        for _ in range(depth):
            next_level = set()
            for node in current_level:
                # Add both successors and predecessors
                next_level.update(self.graph.successors(node))
                next_level.update(self.graph.predecessors(node))
            neighbors.update(next_level)
            current_level = next_level

        neighbors.discard(entity)  # Remove the original entity
        return neighbors

    def get_related_docs(self, entities: List[str]) -> Set[str]:
        """
        Get all document IDs that mention any of these entities.

        Args:
            entities: List of entity names to search for

        Returns:
            Set of document IDs
        """
        doc_ids = set()
        for entity in entities:
            entity = entity.lower().strip()
            if entity in self.entity_docs:
                doc_ids.update(self.entity_docs[entity])
        return doc_ids

    def get_entity_context(self, entity: str) -> str:
        """
        Get a text summary of an entity's relationships.

        Args:
            entity: The entity name

        Returns:
            Formatted string describing relationships
        """
        entity = entity.lower().strip()
        if entity not in self.graph:
            return f"No information found for '{entity}'"

        # Get outgoing relationships (what this entity does/has)
        outgoing = []
        for _, target, data in self.graph.out_edges(entity, data=True):
            label = data.get('label', 'related to')
            outgoing.append(f"{entity} {label} {target}")

        # Get incoming relationships (what relates to this entity)
        incoming = []
        for source, _, data in self.graph.in_edges(entity, data=True):
            label = data.get('label', 'related to')
            incoming.append(f"{source} {label} {entity}")

        context_parts = []
        if outgoing:
            context_parts.append("Relationships: " + "; ".join(outgoing[:5]))
        if incoming:
            context_parts.append("Referenced by: " + "; ".join(incoming[:5]))

        return " | ".join(context_parts) if context_parts else f"Entity '{entity}' exists but has no relationships"


# ============================================================================
# ENTITY EXTRACTION CLASS
# ============================================================================

class EntityRelationExtractor:
    """
    Extracts entities and relationships from text using spaCy NER and LLM.

    Two methods:
        1. spaCy NER: Fast, rule-based (PERSON, ORG, GPE, etc.)
        2. LLM: Slower, more comprehensive, domain-aware
    """

    def __init__(self, llm):
        """
        Initialize extractor with language models.

        Args:
            llm: The LLM for intelligent extraction
        """
        self.llm = llm
        try:
            # Load English language model for NER
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            print("Downloading spaCy model 'en_core_web_sm'...")
            os.system("python -m spacy download en_core_web_sm")
            self.nlp = spacy.load("en_core_web_sm")

    def extract_with_spacy(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Fast entity extraction using spaCy NER.

        Args:
            text: Input text to analyze

        Returns:
            List of (subject, predicate, object) triplets
        """
        doc = self.nlp(text)
        triplets = []

        # Extract named entities
        entities = [ent.text for ent in doc.ents]

        # Create simple relationships based on proximity
        for i in range(len(entities) - 1):
            subject = entities[i]
            object_ = entities[i + 1]
            # Generic predicate
            predicate = "related_to"
            triplets.append((subject, predicate, object_))

        return triplets

    def extract_with_llm(self, text: str) -> List[Tuple[str, str, str]]:
        """
        Intelligent entity extraction using LLM - understands domain context.

        Args:
            text: Input text to analyze

        Returns:
            List of (subject, predicate, object) triplets
        """
        # Prompt for structured extraction
        extraction_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert at extracting entities and relationships from dementia care guidelines.
Extract key concepts, safety features, design principles, and their relationships.

Output format (one per line):
subject | predicate | object

Example:
patient | needs | consistent lighting
corridor | should have | handrails
color contrast | helps with | navigation

Extract 5-10 most important relationships from the text."""),
            ("user", "{text}")
        ])

        try:
            # Get LLM response
            chain = extraction_prompt | self.llm
            response = chain.invoke({"text": text[:2000]})  # Limit text length

            # Parse response into triplets
            triplets = []
            for line in response.content.split('\n'):
                line = line.strip()
                if '|' in line:
                    parts = [p.strip() for p in line.split('|')]
                    if len(parts) == 3:
                        triplets.append(tuple(parts))

            return triplets
        except Exception as e:
            print(f"LLM extraction error: {e}")
            return []

    def extract(self, text: str, use_llm: bool = True) -> List[Tuple[str, str, str]]:
        """
        Extract entities using chosen method.

        Args:
            text: Input text
            use_llm: If True, use LLM (slower, better). If False, use spaCy (faster)

        Returns:
            List of (subject, predicate, object) triplets
        """
        if use_llm:
            return self.extract_with_llm(text)
        else:
            return self.extract_with_spacy(text)


# ============================================================================
# HYBRID RAG CLASS (Vector Store + Knowledge Graph)
# ============================================================================

class GraphRAG:
    """
    Hybrid Retrieval-Augmented Generation combining:
        1. Vector Search: Semantic similarity in embedding space
        2. Knowledge Graph: Entity relationship traversal

    This provides both broad context (vector) and specific relationships (graph).
    """

    def __init__(self, llm, embeddings, persist_directory="./chroma_db"):
        """
        Initialize the hybrid RAG system.

        Args:
            llm: Language model for generation and extraction
            embeddings: Embedding model for vector search
            persist_directory: Where to cache the vector store
        """
        self.llm = llm
        self.embeddings = embeddings
        self.persist_directory = persist_directory
        self.vectorstore = None
        self.knowledge_graph = KnowledgeGraph()
        self.extractor = EntityRelationExtractor(llm)
        self.documents_loaded = False

    def add_documents(self, documents: List[Document], use_llm_extraction: bool = True):
        """
        Process documents: create embeddings and build knowledge graph.

        Args:
            documents: List of LangChain Document objects
            use_llm_extraction: Use LLM for entity extraction (slower but better)
        """
        print(f"Processing {len(documents)} documents...")

        # Create vector store with embeddings
        print("Creating embeddings and vector store...")
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings,
            persist_directory=self.persist_directory
        )

        # Build knowledge graph from documents
        print("Building knowledge graph...")
        for i, doc in enumerate(documents):
            # Extract entities and relationships
            triplets = self.extractor.extract(doc.page_content, use_llm=use_llm_extraction)

            # Add to graph
            doc_id = f"doc_{i}"
            for subject, predicate, object_ in triplets:
                self.knowledge_graph.add_triplet(subject, predicate, object_, doc_id)

            if (i + 1) % 10 == 0:
                print(f"   Processed {i + 1}/{len(documents)} documents")

        # Count entities and relationships
        num_entities = self.knowledge_graph.graph.number_of_nodes()
        num_relations = self.knowledge_graph.graph.number_of_edges()
        print(f"Knowledge graph built:")
        print(f"   - {num_entities:,} entities")
        print(f"   - {num_relations:,} relationships")

        self.documents_loaded = True

    def save_graph_data(self, filepath: str = None):
        """
        Save knowledge graph to disk for fast reloading.

        Args:
            filepath: Where to save (default: ./knowledge_graph.pkl)
        """
        if filepath is None:
            filepath = os.path.join(os.path.dirname(self.persist_directory), "knowledge_graph.pkl")

        with open(filepath, 'wb') as f:
            pickle.dump({
                'graph': self.knowledge_graph.graph,
                'entity_docs': self.knowledge_graph.entity_docs
            }, f)
        print(f"Knowledge graph saved to {filepath}")

    def load_graph_data(self, filepath: str = None) -> bool:
        """
        Load knowledge graph from disk.

        Args:
            filepath: Where to load from (default: ./knowledge_graph.pkl)

        Returns:
            True if successful, False otherwise
        """
        if filepath is None:
            filepath = os.path.join(os.path.dirname(self.persist_directory), "knowledge_graph.pkl")

        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, 'rb') as f:
                data = pickle.load(f)

            self.knowledge_graph.graph = data['graph']
            self.knowledge_graph.entity_docs = data['entity_docs']

            num_entities = self.knowledge_graph.graph.number_of_nodes()
            num_relations = self.knowledge_graph.graph.number_of_edges()
            print(f"Knowledge graph loaded:")
            print(f"   - {num_entities:,} entities")
            print(f"   - {num_relations:,} relationships")

            return True
        except Exception as e:
            print(f"Error loading graph: {e}")
            return False

    def retrieve(self, query: str, k: int = 4, use_graph: bool = True) -> List[Document]:
        """
        Hybrid retrieval combining vector search and knowledge graph.

        Args:
            query: Search query
            k: Number of documents to retrieve
            use_graph: Whether to enhance with graph traversal

        Returns:
            List of relevant documents
        """
        if not self.vectorstore:
            return []

        # Step 1: Vector similarity search
        vector_docs = self.vectorstore.similarity_search(query, k=k)

        if not use_graph:
            return vector_docs

        # Step 2: Extract entities from query
        query_entities = self.extractor.extract(query, use_llm=False)
        query_entity_names = set()
        for subj, _, obj in query_entities:
            query_entity_names.add(subj)
            query_entity_names.add(obj)

        # Step 3: Expand with graph neighbors
        expanded_entities = set(query_entity_names)
        for entity in query_entity_names:
            neighbors = self.knowledge_graph.get_neighbors(entity, depth=1)
            expanded_entities.update(neighbors)

        # Step 4: Get documents related to expanded entities
        related_doc_ids = self.knowledge_graph.get_related_docs(list(expanded_entities))

        # Combine results (deduplicate)
        all_docs = vector_docs.copy()

        return all_docs

    def query(self, question: str, use_graph: bool = True) -> Dict[str, any]:
        """
        Answer a question using hybrid RAG.

        Args:
            question: The question to answer
            use_graph: Whether to use knowledge graph enhancement

        Returns:
            Dictionary with answer and source documents
        """
        if not self.vectorstore:
            return {
                "answer": "No documents loaded. Please load PDFs first.",
                "sources": []
            }

        # Retrieve relevant documents
        docs = self.retrieve(question, k=4, use_graph=use_graph)

        if not docs:
            return {
                "answer": "No relevant information found in the documents.",
                "sources": []
            }

        # Build context from retrieved documents
        context = "\n\n".join([doc.page_content for doc in docs])

        # Generate answer using LLM
        qa_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are an expert on dementia care and home safety design.
Answer questions based on the provided context from dementia care guidelines.
Be specific, practical, and cite key principles from the guidelines."""),
            ("user", """Context from dementia care guidelines:
{context}

Question: {question}

Answer:""")
        ])

        chain = qa_prompt | self.llm
        response = chain.invoke({"context": context, "question": question})

        return {
            "answer": response.content,
            "sources": docs
        }


# ============================================================================
# MAIN PIPELINE CLASS
# ============================================================================

class DementiaImageAnalysisPipeline:
    """
    Main pipeline for analyzing home images for dementia safety.

    Pipeline stages:
        1. Load dementia care guidelines (PDFs) → RAG system
        2. Analyze image with Vision Language Model
        3. Enhance VLM analysis with RAG-retrieved guidelines
        4. Output comprehensive safety report
    """

    def __init__(self, pdf_folder: str = "./pdfs", persist_dir: str = "./chroma_db"):
        """
        Initialize the pipeline.

        Args:
            pdf_folder: Folder containing dementia care guideline PDFs
            persist_dir: Directory to cache vector store and knowledge graph
        """
        print("\n" + "="*70)
        print("DEMENTIA HOME SAFETY IMAGE ANALYSIS PIPELINE")
        print("="*70)

        self.pdf_folder = pdf_folder
        self.persist_dir = persist_dir

        # Initialize OpenAI models
        print("Initializing AI models...")
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
        self.vision_model = ChatOpenAI(model="gpt-4o", temperature=0)

        # Initialize RAG system
        self.rag = GraphRAG(
            llm=self.llm,
            embeddings=self.embeddings,
            persist_directory=persist_dir
        )

        print("Models initialized")

    def load_guidelines(self, use_llm_extraction: bool = False, force_reload: bool = False):
        """
        Load dementia care guidelines from PDFs.

        This method implements smart caching:
            1. Try to load from cache (fast)
            2. If cache missing or force_reload, process PDFs (slow)

        Args:
            use_llm_extraction: Use LLM for entity extraction (better but slower)
            force_reload: Ignore cache and reprocess PDFs
        """
        print("\n" + "-"*70)
        print("STAGE 1: LOADING DEMENTIA CARE GUIDELINES")
        print("-"*70)

        graph_path = os.path.join(os.path.dirname(self.persist_dir), "knowledge_graph.pkl")

        # Check if we can load from cache
        cache_exists = os.path.exists(self.persist_dir) and os.path.exists(graph_path)

        if cache_exists and not force_reload:
            print("Loading from cache...")

            # Load vector store
            try:
                self.rag.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                num_embeddings = self.rag.vectorstore._collection.count()
                print(f"Vector store loaded: {num_embeddings:,} embeddings")
            except Exception as e:
                print(f"Error loading vector store: {e}")
                force_reload = True

            # Load knowledge graph
            if self.rag.load_graph_data(graph_path):
                self.rag.documents_loaded = True
                print("Guidelines loaded successfully from cache!")
                return
            else:
                force_reload = True

        # Need to process PDFs
        print(f"Processing PDFs from: {self.pdf_folder}")

        if not os.path.exists(self.pdf_folder):
            raise FileNotFoundError(f"PDF folder not found: {self.pdf_folder}")

        # Load all PDFs from directory
        print("   Loading PDFs...")
        loader = PyPDFDirectoryLoader(self.pdf_folder)
        all_documents = loader.load()

        if not all_documents:
            raise FileNotFoundError(f"No PDF files found or loaded from {self.pdf_folder}")

        print(f"Loaded {len(all_documents)} pages from PDFs")

        # Split into chunks
        print("Splitting into chunks...")
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(all_documents)
        print(f"   Created {len(splits)} text chunks")

        # Clean metadata for Chroma compatibility
        for doc in splits:
            if doc.metadata:
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    # Chroma only accepts str, int, float, or bool
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_metadata[key] = value
                    else:
                        cleaned_metadata[key] = str(value)
                doc.metadata = cleaned_metadata

        # Process documents through RAG
        self.rag.add_documents(splits, use_llm_extraction=use_llm_extraction)

        # Save to cache
        print("Saving to cache...")
        self.rag.save_graph_data(graph_path)

        print("\n Guidelines loaded and cached successfully!")

    def analyze_image(self, image_path: str, user_context_data: dict = None) -> str:
        """
        Analyze a home image for dementia safety against guidelines.

        This matches the approach from dementia_chatbot_gradio_openai.py:
        1. Retrieve relevant guidelines from RAG system
        2. Include guidelines in vision prompt
        3. Include user's personal context (conversation + assessment) if available
        4. Vision model outputs complete formatted analysis

        Args:
            image_path: Path to the image file
            user_context_data: Optional dict with 'conversation' and 'assessment' data

        Returns:
            Analysis report with Item/Guideline/Recommendation format
        """
        print("\n" + "-"*70)
        print("STAGE 2: IMAGE ANALYSIS")
        print("-"*70)

        if not self.rag.documents_loaded:
            raise RuntimeError("Guidelines not loaded. Call load_guidelines() first.")

        # Format user context if provided
        user_context_text = ""
        if user_context_data:
            user_context_text = self._format_user_context(user_context_data)
            print("   Including personalized user context in analysis")
            # DEBUG: Log what's being included
            print(f"   DEBUG - User context data keys: {user_context_data.keys()}")
            if 'preferenceSummary' in user_context_data:
                print(f"   DEBUG - Preference summary present: {user_context_data['preferenceSummary'][:100]}...")
            print(f"   DEBUG - Formatted context length: {len(user_context_text)} chars")

        # Validate image path
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        print(f"Loading image: {image_path}")

        # Load and validate image
        try:
            img = Image.open(image_path)
            print(f"   Image size: {img.size}")
            print(f"   Image mode: {img.mode}")

            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
                print(f"   Converted to RGB mode")

            # Resize if too large (max 2000px on longest side)
            max_size = 2000
            if max(img.size) > max_size:
                ratio = max_size / max(img.size)
                new_size = tuple(int(dim * ratio) for dim in img.size)
                img = img.resize(new_size, Image.Resampling.LANCZOS)
                print(f"   Resized to: {new_size}")

        except Exception as e:
            raise ValueError(f"Error loading image: {e}")

        # Encode image to base64
        import io
        img_byte_arr = io.BytesIO()
        img.save(img_byte_arr, format='JPEG', quality=95)
        img_byte_arr.seek(0)
        image_data = base64.b64encode(img_byte_arr.read()).decode('utf-8')
        print(f"   Encoded image: {len(image_data)} bytes")

        # Step 1: Retrieve relevant guidelines from RAG
        print("Retrieving dementia guidelines from knowledge base...")

        guidelines_queries = [
            "What are the key design principles, lighting, color schemes, flooring, furniture, and safety features for dementia-friendly spaces?",
            "What are specific color recommendations and contrast requirements for dementia care?",
            "What are safety features and accessibility requirements for dementia-friendly homes?"
        ]

        all_guidelines = []
        for query in guidelines_queries:
            results = self.rag.retrieve(query, k=4, use_graph=True)
            all_guidelines.extend(results)

        # Deduplicate based on content
        seen_content = set()
        unique_guidelines = []
        for doc in all_guidelines:
            if doc.page_content not in seen_content:
                seen_content.add(doc.page_content)
                unique_guidelines.append(doc)

        guidelines_context = "\n\n".join([doc.page_content for doc in unique_guidelines])
        print(f"   Retrieved {len(unique_guidelines)} unique guideline documents")

        # Step 2: Create comprehensive prompt with guidelines embedded
        print("Analyzing image with Vision Language Model...")

        prompt_text = f"""TASK: Analyze the provided image of a home interior space comprehensively for dementia-friendly design compliance.

You are an expert occupational therapist specializing in dementia care environments. You MUST examine the actual image provided and give specific, actionable feedback about what you observe.

{user_context_text}
REFERENCE GUIDELINES FROM KNOWLEDGE BASE:
{guidelines_context}

**BALANCING SAFETY AND PERSONALIZATION**
Your recommendations must EQUALLY prioritize:
1. Dementia design safety guidelines (non-negotiable - safety first)
2. User's personal preferences and identity (listed above in "EXTRACTED USER DESIGN PREFERENCES" if present)

When making recommendations:
- ALWAYS ensure dementia safety standards are met (this is the baseline)
- WITHIN the safety constraints, incorporate user preferences wherever possible
- If user preferences align with safety guidelines, explicitly mention this positive alignment
- If user preferences conflict with safety, find creative compromises (e.g., if user likes dark colors but safety requires contrast, use dark colors for accent pieces against light backgrounds)
- If no user preferences are provided, proceed with standard evidence-based recommendations

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
**Recommendation:** [You are the dementia design expert. Give ONE SPECIFIC directive with exact specifications. Format: "ACTION the ITEM with SPECIFIC-MATERIAL/COLOR/SPECS." Make the decision - do NOT offer choices or say "consider" or "could". State exactly what should be done. NO explanations in this field - just the directive. Choose colors and materials appropriate for the ACTUAL room you are analyzing.]
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
      "item": <item_name>,
      "category": <category>,
      "issue": <issue_description>,
      "guideline_reference": <quoted_guideline>,
      "recommendation": <actionable_instruction>,
      "explanation": <dementia_benefit>
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
 CORRECT: "Sofa", "Floor tiles", "Walls", "Door", "Light fixture", "Window blinds", "TV stand", "Glass partition"
 WRONG: "Sofa color", "Flooring color", "Wall color", "Lighting level", "Contrast between sofa and floor"

RECOMMENDATION FORMAT RULES:
- Use the format: "ACTION the ITEM with SPECIFIC-MATERIAL/COLOR/SPECS."
- State ONE specific value (not ranges): Use "LRV 45" not "LRV 40-50"
- State ONE specific color (not choices): Use "navy blue" not "navy or burgundy"
- Use imperative verbs: "Replace", "Install", "Repaint" (NOT "Consider" or "Could")
- Include technical specifications: LRV values, lux levels, color temperatures, dimensions
- Be concise: No explanations in this field
- Base colors/materials on what would work with the ACTUAL room you observe

Begin your analysis now."""

        # Step 3: Send to vision model
        try:
            message = HumanMessage(
                content=[
                    {"type": "text", "text": prompt_text},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image_data}"}
                    }
                ]
            )

            print("   Sending request to GPT-4o vision model...")
            response = self.vision_model.invoke([message])
            print("Received response from vision model")

            return response.content

        except Exception as e:
            import traceback
            error_details = traceback.format_exc()
            raise RuntimeError(f"Error analyzing image: {str(e)}\n\nDetails:\n{error_details}")

    def _format_user_context(self, user_context_data: dict) -> str:
        """
        Format user conversation and assessment data for inclusion in the analysis prompt.

        Args:
            user_context_data: Dict with 'conversation' and/or 'assessment' keys

        Returns:
            Formatted string to include in the prompt
        """
        if not user_context_data:
            return ""

        context_parts = []
        context_parts.append("="*70)
        context_parts.append("USER'S PERSONAL CONTEXT & PREFERENCES")
        context_parts.append("="*70)
        context_parts.append("")
        context_parts.append("IMPORTANT: Use this personalized information to create recommendations that:")
        context_parts.append("1. Respect the user's specific needs and memories")
        context_parts.append("2. Avoid modifying things they explicitly want to keep unchanged")
        context_parts.append("3. Support the activities and environments they value")
        context_parts.append("")

        # Format conversation data (from Memory Bot)
        if 'conversation' in user_context_data:
            conv = user_context_data['conversation']
            context_parts.append("--- USER'S MEMORIES & VALUED ACTIVITIES ---")
            context_parts.append("")

            topics = conv.get('selectedTopics', [])
            if topics:
                context_parts.append(f"Important activities to the user: {', '.join(topics)}")
                context_parts.append("")

            topic_convos = conv.get('topicConversations', {})
            if topic_convos:
                context_parts.append("Detailed memories shared by the user:")
                context_parts.append("")
                for topic, details in topic_convos.items():
                    context_parts.append(f"**{topic}:**")

                    if details.get('fixedAnswer'):
                        q = details.get('fixedQuestion', '')
                        context_parts.append(f"  Q: {q}")
                        context_parts.append(f"  A: {details['fixedAnswer']}")

                    if details.get('firstAIAnswer'):
                        q = details.get('firstAIQuestion', '')
                        context_parts.append(f"  Q: {q}")
                        context_parts.append(f"  A: {details['firstAIAnswer']}")

                    if details.get('secondAIAnswer'):
                        q = details.get('secondAIQuestion', '')
                        context_parts.append(f"  Q: {q}")
                        context_parts.append(f"  A: {details['secondAIAnswer']}")

                    context_parts.append("")

        # Format preference summary (from Memory Bot conversation analysis)
        if 'preferenceSummary' in user_context_data:
            pref_summary = user_context_data['preferenceSummary'].strip()
            if pref_summary:
                context_parts.append("--- EXTRACTED USER DESIGN PREFERENCES ---")
                context_parts.append("")
                context_parts.append("The following preferences were extracted from the user's conversation with Memory Bot")
                context_parts.append("and balanced against dementia design guidelines:")
                context_parts.append("")
                context_parts.append(pref_summary)
                context_parts.append("")
                context_parts.append("IMPORTANT: Incorporate these preferences into your recommendations wherever possible,")
                context_parts.append("while maintaining dementia design safety standards.")
                context_parts.append("")

        # Format assessment data (from Fix My Home)
        if 'assessment' in user_context_data:
            assess = user_context_data['assessment']
            context_parts.append("--- USER'S SPECIFIC CONCERNS & CONSTRAINTS ---")
            context_parts.append("")

            issues = assess.get('selectedIssues', [])
            if issues:
                context_parts.append(f"Specific safety concerns: {', '.join(issues)}")
                context_parts.append("")

            comments = assess.get('comments', '').strip()
            if comments:
                context_parts.append("Additional comments from user:")
                context_parts.append(f'  "{comments}"')
                context_parts.append("")

            no_change = assess.get('noChangeComments', '').strip()
            if no_change:
                context_parts.append("⚠️ CRITICAL - DO NOT MODIFY: ")
                context_parts.append(f'  "{no_change}"')
                context_parts.append("")
                context_parts.append("  → Your recommendations MUST NOT suggest changes to items or features")
                context_parts.append("     mentioned in this constraint. Find alternative solutions that respect")
                context_parts.append("     this preference while still addressing safety concerns.")
                context_parts.append("")

        context_parts.append("="*70)
        context_parts.append("")

        return "\n".join(context_parts)


def main():
    """
    Command-line interface for the pipeline.

    Required arguments:
        --image: Path to the home image to analyze

    Optional arguments:
        --pdf-folder: Folder with guideline PDFs (default: ./pdfs)
        --persist-dir: Cache directory (default: ./chroma_db)
        --use-llm-extraction: Use LLM for entity extraction (slower, better)
        --force-reload: Ignore cache and reprocess PDFs
        --output: Save analysis to file (if not specified, prints to console)
    """
    parser = argparse.ArgumentParser(
        description="Dementia Home Safety Image Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic usage
    python dementia_image_analysis_pipeline.py --image "living_room.jpg"

    # Save output to file
    python dementia_image_analysis_pipeline.py --image "bedroom.jpg" --output "analysis.txt"

    # Force reload PDFs (don't use cache)
    python dementia_image_analysis_pipeline.py --image "kitchen.jpg" --force-reload

    # Use LLM for better entity extraction (slower)
    python dementia_image_analysis_pipeline.py --image "hallway.jpg" --use-llm-extraction
        """
    )

    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to the home image to analyze"
    )

    parser.add_argument(
        "--pdf-folder",
        type=str,
        default="./Dementia Guidelines",
        help="Folder containing dementia care guideline PDFs (default: ./pdfs)"
    )

    parser.add_argument(
        "--persist-dir",
        type=str,
        default="./chroma_db",
        help="Directory to cache vector store and knowledge graph (default: ./chroma_db)"
    )

    parser.add_argument(
        "--use-llm-extraction",
        action="store_true",
        help="Use LLM for entity extraction (slower but more comprehensive)"
    )

    parser.add_argument(
        "--force-reload",
        action="store_true",
        help="Force reload PDFs, ignoring cache"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Save analysis to file (if not specified, prints to console)"
    )

    args = parser.parse_args()

    try:
        # Initialize pipeline
        pipeline = DementiaImageAnalysisPipeline(
            pdf_folder=args.pdf_folder,
            persist_dir=args.persist_dir
        )

        # Load guidelines
        pipeline.load_guidelines(
            use_llm_extraction=args.use_llm_extraction,
            force_reload=args.force_reload
        )

        # Analyze image
        analysis = pipeline.analyze_image(args.image)

        # Generate auto-save filename if not specified
        if not args.output:
            # Create timestamped filename based on image name
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            image_basename = os.path.splitext(os.path.basename(args.image))[0]
            args.output = f"analysis_{image_basename}_{timestamp}.txt"

        # Output results
        print("\n" + "="*70)
        print("ANALYSIS RESULTS")
        print("="*70 + "\n")

        # Always save to file (auto-generated or user-specified)
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(analysis)
        print(f"Analysis saved to: {args.output}\n")

        # Also print to console
        print(analysis)

        print("\n" + "="*70)
        print("PIPELINE COMPLETE")
        print("="*70 + "\n")

    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()