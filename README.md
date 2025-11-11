# Dementia Home Safety Image Analysis Pipeline

An AI-powered pipeline that analyzes home environment images for dementia safety compliance using Vision Language Models (VLM) and Retrieval-Augmented Generation (RAG).

## Overview

This pipeline:
1. Loads dementia care guidelines from PDF documents into a RAG system (Vector Store + Knowledge Graph)
2. Analyzes home images using GPT-4o Vision Model
3. Provides detailed safety assessments with specific, actionable recommendations
4. Outputs structured reports with Item/Guideline/Recommendation format

## Features

- **Hybrid RAG System**: Combines vector search (Chroma DB) with knowledge graph (NetworkX) for comprehensive guideline retrieval
- **Vision Analysis**: Uses GPT-4o to analyze home images for safety issues
- **Evidence-Based**: All recommendations are backed by exact quotes from dementia care guidelines
- **Structured Output**: Clear format with physical items, issues, guideline references, specific recommendations, and explanations
- **Auto-Save**: Automatically saves analysis reports with timestamped filenames
- **Smart Caching**: Saves processed data for fast subsequent runs

---

## Setup Instructions

### 1. Prerequisites

- **Python**: 3.10 or higher
- **OpenAI API Key**: Required for GPT-4o and embeddings

### 2. Clone or Download the Project

```bash
cd D:\virtual-gf-with-reminders-rolled-back-04-dec\RAG-Langchain
```

### 3. Create Virtual Environment (Recommended)

**Option A: Using venv**
```bash
python -m venv .venv
.venv\Scripts\activate
```

**Option B: Using conda**
```bash
conda create -n dementia python=3.10
conda activate dementia
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

**Core dependencies:**
- `langchain-openai` - OpenAI integration
- `langchain-chroma` - Vector store
- `langchain-community` - PDF loaders and utilities
- `langchain-text-splitters` - Text chunking
- `chromadb` - Vector database
- `networkx` - Knowledge graph
- `spacy` - NLP for entity extraction
- `Pillow` - Image processing
- `pypdf` - PDF parsing
- `python-dotenv` - Environment variables

### 5. Download spaCy Language Model

```bash
python -m spacy download en_core_web_sm
```

### 6. Set Up OpenAI API Key

**Option A: Create `.env` file (Recommended)**

Create a file named `.env` in the project directory:

```
OPENAI_API_KEY=sk-your-actual-api-key-here
```

**Option B: Set Environment Variable in PowerShell**

Temporary (current session only):
```powershell
$env:OPENAI_API_KEY="sk-your-actual-api-key-here"
```

Permanent:
```powershell
[System.Environment]::SetEnvironmentVariable('OPENAI_API_KEY', 'sk-your-actual-api-key-here', 'User')
```

### 7. Prepare Dementia Guidelines (PDFs)

Place your dementia care guideline PDF files in:
```
D:\virtual-gf-with-reminders-rolled-back-04-dec\RAG-Langchain\Dementia Guidelines\
```

The pipeline will automatically load all `.pdf` files from this folder.

---

## Usage

### Basic Usage

Analyze an image and auto-save results:

```bash
python dementia_pipeline.py --image "path/to/your/image.jpg"
```

**Example:**
```bash
python dementia_pipeline.py --image "Yishun_full-533x400.jpg"
```

**Output:**
- Prints analysis to console
- Auto-saves to file: `analysis_Yishun_full-533x400_20251111_143052.txt`

### Specify Custom Output File

```bash
python dementia_pipeline.py --image "bedroom.jpg" --output "bedroom_analysis.txt"
```

### Force Reload PDFs (Ignore Cache)

If you've updated the PDF guidelines and want to reprocess:

```bash
python dementia_pipeline.py --image "living_room.jpg" --force-reload
```

### Use Better Entity Extraction (Slower)

For more comprehensive knowledge graph building:

```bash
python dementia_pipeline.py --image "kitchen.jpg" --use-llm-extraction
```

### Custom PDF Folder

If your PDFs are in a different location:

```bash
python dementia_pipeline.py --image "hallway.jpg" --pdf-folder "D:\path\to\pdfs"
```

---

## Command-Line Arguments

| Argument | Required | Default | Description |
|----------|----------|---------|-------------|
| `--image` | **Yes** | - | Path to the home image to analyze |
| `--pdf-folder` | No | `./Dementia Guidelines` | Folder containing guideline PDFs |
| `--persist-dir` | No | `./chroma_db` | Directory to cache vector store and knowledge graph |
| `--output` | No | Auto-generated | Output file path (auto-saves with timestamp if not specified) |
| `--use-llm-extraction` | No | False | Use LLM for entity extraction (slower but more comprehensive) |
| `--force-reload` | No | False | Force reload PDFs, ignoring cache |

---

## How It Works

### Pipeline Flow

```
┌─────────────────────────────────────────────────────────────┐
│  1. LOAD GUIDELINES                                         │
│     • Load PDFs from "Dementia Guidelines" folder           │
│     • Create embeddings (text-embedding-3-large)            │
│     • Build knowledge graph (entities & relationships)      │
│     • Cache to disk for fast reloading                      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  2. RETRIEVE RELEVANT GUIDELINES                            │
│     • Query RAG system for relevant guidelines              │
│     • Hybrid retrieval: Vector search + Knowledge graph     │
│     • Get specific guidelines about lighting, contrast,     │
│       safety features, etc.                                 │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  3. ANALYZE IMAGE WITH VISION MODEL                         │
│     • Load and validate image                               │
│     • Encode to base64                                      │
│     • Send to GPT-4o Vision with guidelines embedded        │
│     • Vision model analyzes against 8 assessment areas      │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│  4. OUTPUT STRUCTURED ANALYSIS                              │
│     • Room description                                      │
│     • Issues identified with:                               │
│       - Item (physical object)                              │
│       - Category                                            │
│       - Issue description                                   │
│       - Guideline reference (exact quote)                   │
│       - Specific recommendation                             │
│       - Explanation                                         │
│     • JSON summary                                          │
└─────────────────────────────────────────────────────────────┘
```

### Assessment Areas

The pipeline evaluates 8 key areas:

1. **Contrast & Visual Clarity** - Color contrast between surfaces, boundaries, visual cues
2. **Lighting** - Natural light, artificial lighting, glare, shadows, distribution
3. **Flooring & Surfaces** - Patterns, reflections, color changes, slip resistance
4. **Wayfinding & Orientation** - Clear pathways, landmarks, signage, memory cues
5. **Safety Features** - Grab bars, handrails, sharp corners, obstacles, furniture stability
6. **Color Scheme** - Calming vs. stimulating colors, appropriate choices, wayfinding use
7. **Spatial Design** - Layout, furniture arrangement, sightlines, accessibility
8. **Environmental Cues** - Mirrors, reflective surfaces, confusing patterns, orientation aids

---

## Output Format

The analysis report includes:

### 1. Room Description
Brief description of the space and main features

### 2. Issues Identified
For each issue:

```
**Item:** [Physical object name - e.g., "Sofa", "Floor tiles", "Ceiling light"]
**Category:** [One of the 8 assessment areas]
**Issue:** [Detailed description of the problem observed]
**Guideline Reference:** [Exact quote from dementia care guidelines]
**Recommendation:** [Specific, actionable directive with exact specifications]
**Explanation:** [Why this change benefits people with dementia]
```

**Example:**

```
**Item:** Sofa
**Category:** Contrast
**Issue:** Dark grey sofa placed on dark grey floor with insufficient contrast (LRV difference < 10)
**Guideline Reference:** "Furniture should contrast with flooring by at least 30 LRV points to support visual perception and reduce fall risk."
**Recommendation:** Replace sofa with one upholstered in burgundy fabric (LRV 20).
**Explanation:** This creates sufficient visual contrast making the sofa clearly visible, reducing fall risk and supporting independent movement as per dementia-friendly design principles.
```

### 3. JSON Summary
Machine-readable summary with issue counts by category

---

## First Run vs. Subsequent Runs

### First Run (Slow - ~2-3 minutes)
1. Loads all PDFs from `Dementia Guidelines` folder
2. Splits documents into chunks
3. Creates embeddings for all chunks (~1768 embeddings)
4. Builds knowledge graph (~18,000 entities, ~30,000 relationships)
5. Saves everything to `chroma_db/` and `knowledge_graph.pkl`
6. Analyzes image
7. Generates report

### Subsequent Runs (Fast - ~30 seconds)
1. Loads vector store from `chroma_db/` (instant)
2. Loads knowledge graph from `knowledge_graph.pkl` (instant)
3. Analyzes image
4. Generates report

---

## Troubleshooting

### Error: "No module named 'langchain'"

**Solution:**
```bash
pip install -r requirements.txt
```

### Error: "Can't find model 'en_core_web_sm'"

**Solution:**
```bash
python -m spacy download en_core_web_sm
```

### Error: "OPENAI_API_KEY not set"

**Solution:** Create a `.env` file with your API key:
```
OPENAI_API_KEY=sk-your-key-here
```

### Error: "PDF folder not found: ./pdfs"

**Solution:** Make sure your PDFs are in the correct folder:
```
D:\virtual-gf-with-reminders-rolled-back-04-dec\RAG-Langchain\Dementia Guidelines\
```

Or specify the folder:
```bash
python dementia_pipeline.py --image "image.jpg" --pdf-folder "path/to/pdfs"
```

### Error: "Collection expecting embedding dimension of 3072, got 1536"

**Solution:** Your cached vector store uses a different embedding model. Delete the cache and regenerate:
```bash
rm -rf chroma_db
rm knowledge_graph.pkl
python dementia_pipeline.py --image "image.jpg"
```

### Image Not Loading or Analysis Fails

**Check:**
- Image file exists and path is correct
- Image is a valid format (JPG, PNG, etc.)
- Image file is not corrupted
- You have sufficient OpenAI API credits

---

## Technical Details

### Models Used

- **Vision Model:** `gpt-4o` (for image analysis)
- **Text Model:** `gpt-4o-mini` (for text processing)
- **Embeddings:** `text-embedding-3-large` (3072 dimensions)

### Technologies

- **LangChain:** Framework for RAG and LLM integration
- **Chroma DB:** Vector database for semantic search
- **NetworkX:** Graph library for knowledge graph
- **spaCy:** NLP for entity extraction
- **OpenAI API:** GPT-4o Vision and embeddings

### File Structure

```
RAG-Langchain/
├── dementia_pipeline.py          # Main pipeline script
├── requirements.txt              # Python dependencies
├── .env                          # OpenAI API key (create this)
├── Dementia Guidelines/          # PDF guidelines (your PDFs here)
├── chroma_db/                    # Vector store cache (auto-generated)
├── knowledge_graph.pkl           # Knowledge graph cache (auto-generated)
└── analysis_*.txt                # Generated analysis reports
```

---

## API Costs

Approximate costs per analysis (as of 2025):

- **Embeddings (first run):** ~$0.05 (1768 chunks × text-embedding-3-large)
- **Embeddings (cached):** $0 (uses cache)
- **Vision Analysis:** ~$0.01-0.02 per image (GPT-4o Vision)
- **Guideline Retrieval:** Negligible (retrieval only, no generation)

**Total per image:**
- First run: ~$0.06-0.07
- Subsequent runs: ~$0.01-0.02

---

## Support

For issues or questions:
1. Check the Troubleshooting section above
2. Verify all dependencies are installed correctly
3. Ensure your OpenAI API key is valid and has credits
4. Make sure PDF guidelines are in the correct folder

---

## License

This project uses OpenAI API and requires an OpenAI API key with access to GPT-4o Vision.

---

**Last Updated:** November 11, 2025
