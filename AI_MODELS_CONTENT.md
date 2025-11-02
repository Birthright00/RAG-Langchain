# AI MODELS - Content for Slide

## LLM (Large Language Model)

**Primary Model:** GPT-4o-mini
- Used for conversational responses and query processing
- Handles natural language understanding of user questions
- Generates comprehensive answers based on retrieved context
- Processes entity extraction from text chunks
- Fast and cost-effective for general text tasks

**Vision Model:** GPT-4o
- Analyzes uploaded home images
- Identifies dementia-unfriendly design elements
- Provides specific recommendations based on guidelines
- Multi-modal capabilities for image understanding


## EMBEDDING

**Model:** text-embedding-3-large (OpenAI)
- Converts text chunks into high-dimensional vectors (3072 dimensions)
- Enables semantic similarity search across document chunks
- Creates vector representations of dementia guideline documents
- Supports efficient retrieval of relevant context
- State-of-the-art embedding quality for accurate matching


## RETRIEVAL

**Hybrid RAG System:** Vector Search + Knowledge Graph

**Vector Store:** ChromaDB
- Stores embedded chunks with persistent SQLite backend
- Performs semantic similarity search (cosine/L2 distance)
- Returns top-k most relevant document chunks
- Persistent storage in `./chroma_db/` directory

**Knowledge Graph:** NetworkX MultiDiGraph
- Extracts entities and relationships using spaCy NER
- Creates triplet structures (subject-predicate-object)
- Enables graph-based contextual retrieval
- Finds related entities within specified depth
- Combines with vector results for enhanced retrieval

**Entity Extraction:**
- spaCy (en_core_web_sm) for fast NER
- Optional LLM-based extraction for medical domain entities
- Identifies medical conditions, symptoms, treatments, guidelines


## RETRIEVAL PROCESS

1. **Query Processing:** Extract entities from user question
2. **Vector Search:** Find semantically similar chunks (k=4)
3. **Graph Expansion:** Get neighboring entities (depth=2)
4. **Hybrid Merge:** Combine vector + graph results
5. **Deduplication:** Return unique documents (up to k*2)
6. **Context Building:** Assemble retrieved chunks + graph relationships
7. **Response Generation:** LLM generates answer with full context
