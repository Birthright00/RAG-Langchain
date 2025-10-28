# 🧠 Dementia Guidelines Chatbot - Gradio Interface

A beautiful web interface for the Dementia Guidelines Chatbot using Gradio.

## Features

- 🎨 **Modern UI**: Clean, user-friendly interface
- 💬 **Chat Interface**: Interactive conversation with the chatbot
- 📊 **Statistics Panel**: View knowledge graph insights
- 📁 **PDF Loader**: Load PDFs directly from the interface
- 🗑️ **Clear History**: Reset conversation anytime
- 💡 **Example Questions**: Get started quickly with suggestions

## Installation

### 1. Install Gradio

```bash
pip install gradio
```

### 2. Ensure Other Dependencies are Installed

```bash
pip install langchain langchain-openai langchain-community langchain-core langchain-text-splitters
pip install networkx spacy pypdf
python -m spacy download en_core_web_sm
```

## Usage

### Run the Gradio Interface

```bash
python dementia_chatbot_gradio.py
```

The interface will open automatically in your browser at: `http://127.0.0.1:7860`

### How to Use

1. **Load PDFs**
   - Click the "Load PDFs" button in the Control Panel
   - Optionally check "Use LLM extraction" for better accuracy (slower)
   - Wait for loading to complete (status will show ✅ when done)

2. **Ask Questions**
   - Type your question in the text box
   - Press Enter or click "Send"
   - View the response in the chat window

3. **View Statistics**
   - Click "View Stats" to see knowledge graph information
   - Shows entity counts, relationships, and top entities

4. **Clear Chat**
   - Click "🗑️ Clear Chat" to reset the conversation

## Interface Layout

```
┌─────────────────────────────────────────────────────────┐
│  🧠 Dementia Guidelines Chatbot                         │
├──────────────────────────┬──────────────────────────────┤
│                          │  📋 Control Panel            │
│  Chat Window             │  ┌────────────────────────┐  │
│  ┌────────────────────┐  │  │ Step 1: Load PDFs      │  │
│  │ User: Question     │  │  │ [Load PDFs Button]     │  │
│  │ Bot: Response      │  │  └────────────────────────┘  │
│  └────────────────────┘  │                              │
│                          │  ┌────────────────────────┐  │
│  [Your Question...]      │  │ 📊 Statistics          │  │
│  [Send]                  │  │ [View Stats Button]    │  │
│  [Clear Chat]            │  └────────────────────────┘  │
└──────────────────────────┴──────────────────────────────┘
```

## Example Questions

Try asking:
- "What are the key principles for dementia-friendly design?"
- "How should lighting be designed for people with dementia?"
- "What are recommendations for outdoor spaces?"
- "How can technology support dementia care?"
- "What color schemes are recommended for dementia-friendly environments?"

## Configuration

### Change Port

Edit the launch settings in `dementia_chatbot_gradio.py`:

```python
demo.launch(
    server_port=8080,  # Change port here
    share=False
)
```

### Enable Public Sharing

Set `share=True` to get a public URL (requires internet):

```python
demo.launch(share=True)
```

### Change PDF Folder Path

Edit line 312 in the script:

```python
pdf_folder = r"Your\Custom\Path\Here"
```

## Features Explained

### 📁 PDF Loading
- Loads all PDFs from the Dementia Guidelines folder
- Option for LLM-based entity extraction (more accurate but slower)
- Shows loading status and statistics

### 💬 Chat Interface
- Natural conversation flow
- Maintains chat history within session
- Can clear and restart conversation

### 📊 Statistics Panel
- Shows total entities and relationships in knowledge graph
- Displays most connected entities
- Tracks conversation history count

### 🎨 UI Theme
- Soft, professional theme
- Responsive design
- Easy-to-read typography

## Troubleshooting

### Gradio not installed
```bash
pip install gradio
```

### Port already in use
Change the port in `demo.launch(server_port=7860)` to another number (e.g., 7861)

### PDFs not loading
- Check the folder path is correct
- Ensure PDFs exist in the folder
- Check console for error messages

### Slow loading
- Uncheck "Use LLM extraction" for faster loading
- Large PDFs take longer to process
- First load always takes the longest

## Keyboard Shortcuts

- **Enter**: Send message
- **Ctrl+C**: Stop server (in terminal)

## Advanced

### Running on Network

To make the chatbot accessible on your local network:

```python
demo.launch(
    server_name="0.0.0.0",  # Listen on all interfaces
    server_port=7860,
    share=False
)
```

Then access from other devices using: `http://YOUR_IP:7860`

### Auto-reload on Changes

```bash
gradio dementia_chatbot_gradio.py
```

## Requirements

- Python 3.8+
- Gradio 4.0+
- OpenAI API key
- Internet connection (for LLM calls)

## License

MIT License - Free to use and modify!
