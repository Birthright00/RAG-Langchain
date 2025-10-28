# Dementia Guidelines Chatbot

An intelligent chatbot that uses GraphRAG (Retrieval-Augmented Generation with Knowledge Graphs) to answer questions from your Dementia Guidelines PDFs.

## Features

- 🧠 **Knowledge Graph Integration**: Extracts entities and relationships from medical documents
- 📚 **PDF Support**: Automatically loads and processes all PDFs in a folder
- 💬 **Interactive Chat**: Natural conversation interface
- 📊 **Statistics**: View knowledge graph insights
- 📝 **History**: Track your conversation history
- 🔍 **Hybrid Search**: Combines vector similarity search with graph-based retrieval

## Setup

### 1. Install Dependencies

```bash
pip install langchain langchain-openai langchain-community langchain-core langchain-text-splitters
pip install networkx spacy pypdf
python -m spacy download en_core_web_sm
```

### 2. Set OpenAI API Key

The chatbot will prompt you for your API key when you run it, or you can set it as an environment variable:

```bash
export OPENAI_API_KEY="your-api-key-here"
```

### 3. Prepare Your PDFs

Create a folder called "Dementia Guidelines" in the same directory as the script and add your PDF files:

```
RAG-Langchain/
├── dementia_chatbot.py
├── Dementia Guidelines/
│   ├── guideline1.pdf
│   ├── guideline2.pdf
│   └── guideline3.pdf
```

## Usage

### Run the Chatbot

```bash
python dementia_chatbot.py
```

### First Time Setup

1. The chatbot will ask for the PDF folder path (default: `./Dementia Guidelines`)
2. Choose whether to use LLM extraction:
   - **Yes (y)**: Slower but extracts more accurate relationships
   - **No (N)**: Faster, uses only spaCy NER

3. Wait for PDFs to be processed (this may take a few minutes)

### Chat Commands

Once loaded, you can:

- **Ask questions**: Just type your question about dementia care
  ```
  💬 You: What are the early signs of dementia?
  ```

- **View history**: Type `history` to see past conversations
  ```
  💬 You: history
  ```

- **View statistics**: Type `stats` to see knowledge graph info
  ```
  💬 You: stats
  ```

- **Exit**: Type `quit`, `exit`, or `q` to end the session

## Example Session

```
🧠 DEMENTIA GUIDELINES CHATBOT
================================================================

Commands:
  - Type your question to get answers from the guidelines
  - Type 'history' to see conversation history
  - Type 'stats' to see knowledge graph statistics
  - Type 'quit' or 'exit' to end the session

💬 You: What are the recommended treatments for early-stage dementia?

🤖 Assistant: Based on the Dementia Guidelines, recommended treatments
for early-stage dementia include:

1. Pharmacological interventions:
   - Acetylcholinesterase inhibitors (donepezil, rivastigmine, galantamine)
   - These medications can help manage cognitive symptoms...

2. Non-pharmacological approaches:
   - Cognitive stimulation therapy
   - Regular physical exercise...

[Additional detailed response based on your PDFs]
```

## How It Works

1. **Document Loading**: Loads all PDFs from the specified folder
2. **Text Splitting**: Breaks documents into manageable chunks (1000 chars with 200 overlap)
3. **Entity Extraction**: Identifies medical entities, conditions, treatments using:
   - spaCy NER (fast, rule-based)
   - GPT-4o-mini (optional, more accurate)
4. **Knowledge Graph**: Builds a graph of relationships between entities
5. **Vector Store**: Creates embeddings for semantic search
6. **Hybrid Retrieval**: Combines:
   - Semantic similarity search
   - Graph-based entity relationships
7. **Answer Generation**: Uses GPT-4o-mini to generate comprehensive answers

## Benefits Over Standard RAG

- **Better Context**: Captures relationships between medical concepts
- **Multi-hop Reasoning**: Finds connections across multiple documents
- **Entity-Centric**: Retrieves based on entity relationships, not just text similarity
- **Explainable**: Shows which entities and relationships were used

## Customization

### Change PDF Folder

Edit the default path in `dementia_chatbot.py`:

```python
pdf_folder = "./Your Custom Folder Name"
```

### Adjust Chunk Size

Modify the text splitter settings:

```python
self.text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,  # Increase for larger chunks
    chunk_overlap=200  # Adjust overlap
)
```

### Change LLM Model

Update the model in the script:

```python
llm = ChatOpenAI(model="gpt-4o", temperature=0)  # Use GPT-4o instead
```

## Troubleshooting

### "No PDF files found"
- Ensure your PDFs are in the correct folder
- Check folder path is correct

### "spaCy model not found"
```bash
python -m spacy download en_core_web_sm
```

### Slow processing
- Use fewer PDFs for testing
- Set `use_llm_extraction=False` for faster processing
- Reduce chunk size

### Out of memory
- Process PDFs in batches
- Reduce chunk size
- Use a smaller embedding model

## License

MIT License - feel free to modify and use for your projects!