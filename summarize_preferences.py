"""
User Preference Summarizer for Dementia-Friendly Design
=======================================================

This script analyzes conversation logs between users and the chatbot to extract
and summarize user preferences balanced against dementia design guidelines.

The script focuses on two key categories:
1. Color & Contrast - Color preferences, visual patterns, contrast needs
2. Familiarity & Personal Identity - Personal themes, hobbies, cultural elements

The LLM evaluates user preferences through the lens of dementia design guidelines
retrieved from a RAG system (vector store + knowledge graph) to produce a balanced
summary that maintains design safety while incorporating personal preferences.

Usage:
    python summarize_preferences.py --input conversation.json --output preferences_summary.json
    python summarize_preferences.py --input conversation.json --pdf-folder ./pdfs --persist-dir ./chroma_db

    Or as a module:
    from summarize_preferences import PreferenceSummarizer
    summarizer = PreferenceSummarizer()
    result = summarizer.summarize(conversation_data)
"""

import os
import json
import argparse
from typing import Dict, List, Any
from datetime import datetime

# LangChain imports
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, SystemMessage

# Import the GraphRAG system from dementia_pipeline
from dementia_pipeline import GraphRAG

# Environment setup
from dotenv import load_dotenv
load_dotenv()


class PreferenceSummarizer:
    """
    Analyzes conversation logs to extract user preferences balanced with
    dementia design guidelines retrieved from a RAG system.
    """

    def __init__(self, model: str = "gpt-4o", temperature: float = 0.3,
                 pdf_folder: str = "./pdfs", persist_dir: str = "./chroma_db",
                 load_rag: bool = True):
        """
        Initialize the preference summarizer with RAG system.

        Args:
            model: OpenAI model to use (default: gpt-4o)
            temperature: Model temperature for creativity vs consistency (default: 0.3)
            pdf_folder: Folder containing dementia guideline PDFs (default: ./pdfs)
            persist_dir: Directory to cache vector store and knowledge graph (default: ./chroma_db)
            load_rag: Whether to load RAG system (default: True)
        """
        print("\n" + "="*70)
        print("PREFERENCE SUMMARIZER WITH RAG")
        print("="*70)

        self.llm = ChatOpenAI(model=model, temperature=temperature)
        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
        self.pdf_folder = pdf_folder
        self.persist_dir = persist_dir

        # Initialize RAG system
        if load_rag:
            print("Initializing RAG system...")
            self.rag = GraphRAG(
                llm=self.llm,
                embeddings=self.embeddings,
                persist_directory=persist_dir
            )
            self._load_guidelines()
        else:
            self.rag = None
            print("RAG system not loaded (load_rag=False)")

    def _load_guidelines(self):
        """
        Load dementia design guidelines from PDFs into RAG system.
        Uses caching for fast subsequent loads.
        """
        import pickle
        from langchain_community.document_loaders import PyPDFDirectoryLoader
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        print("\nLoading dementia design guidelines...")

        graph_path = os.path.join(os.path.dirname(self.persist_dir), "knowledge_graph.pkl")

        # Check if we can load from cache
        cache_exists = os.path.exists(self.persist_dir) and os.path.exists(graph_path)

        if cache_exists:
            print("Loading from cache...")
            try:
                from langchain_chroma import Chroma
                self.rag.vectorstore = Chroma(
                    persist_directory=self.persist_dir,
                    embedding_function=self.embeddings
                )
                num_embeddings = self.rag.vectorstore._collection.count()
                print(f"Vector store loaded: {num_embeddings:,} embeddings")

                if self.rag.load_graph_data(graph_path):
                    self.rag.documents_loaded = True
                    print("Guidelines loaded successfully from cache!\n")
                    return
            except Exception as e:
                print(f"Cache load failed: {e}. Will reload from PDFs...")

        # Load from PDFs
        print(f"Processing PDFs from: {self.pdf_folder}")

        if not os.path.exists(self.pdf_folder):
            print(f"WARNING: PDF folder not found: {self.pdf_folder}")
            print("RAG system will operate without guideline documents.")
            return

        # Load all PDFs
        loader = PyPDFDirectoryLoader(self.pdf_folder)
        all_documents = loader.load()

        if not all_documents:
            print(f"WARNING: No PDF files found in {self.pdf_folder}")
            print("RAG system will operate without guideline documents.")
            return

        print(f"Loaded {len(all_documents)} pages from PDFs")

        # Split into chunks
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
        )
        splits = text_splitter.split_documents(all_documents)
        print(f"Created {len(splits)} text chunks")

        # Clean metadata for Chroma
        for doc in splits:
            if doc.metadata:
                cleaned_metadata = {}
                for key, value in doc.metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        cleaned_metadata[key] = value
                    else:
                        cleaned_metadata[key] = str(value)
                doc.metadata = cleaned_metadata

        # Process through RAG
        self.rag.add_documents(splits, use_llm_extraction=False)

        # Save to cache
        self.rag.save_graph_data(graph_path)
        print("Guidelines loaded and cached successfully!\n")

    def _retrieve_guidelines(self, conversation_text: str) -> Dict[str, List[str]]:
        """
        Retrieve relevant dementia design guidelines from RAG system.

        Args:
            conversation_text: The formatted conversation to analyze

        Returns:
            Dictionary with guideline categories and retrieved guidelines
        """
        if not self.rag or not self.rag.documents_loaded:
            # Fallback to basic guidelines if RAG not available
            return {
                "color_contrast": [
                    "High contrast between surfaces helps with depth perception",
                    "Avoid busy patterns that may cause visual confusion",
                    "Use color to differentiate spaces and aid wayfinding"
                ],
                "familiarity_identity": [
                    "Familiar objects and themes provide comfort and orientation",
                    "Personal items from the past can trigger positive memories",
                    "Balance between personal preferences and design safety is essential"
                ]
            }

        # Retrieve guidelines for each category using RAG
        color_query = f"dementia design guidelines for color, contrast, visual patterns, and color schemes in interior spaces. Context: {conversation_text[:500]}"
        identity_query = f"dementia design guidelines for familiarity, personal identity, cultural elements, and meaningful personal items. Context: {conversation_text[:500]}"

        try:
            # Retrieve documents
            color_docs = self.rag.retrieve(color_query, k=5, use_graph=True)
            identity_docs = self.rag.retrieve(identity_query, k=5, use_graph=True)

            # Extract guideline text
            color_guidelines = []
            for doc in color_docs:
                content = doc.page_content.strip()
                if content:
                    # Split into sentences and take first few relevant ones
                    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
                    color_guidelines.extend(sentences[:2])

            identity_guidelines = []
            for doc in identity_docs:
                content = doc.page_content.strip()
                if content:
                    sentences = [s.strip() for s in content.split('.') if len(s.strip()) > 20]
                    identity_guidelines.extend(sentences[:2])

            # Remove duplicates while preserving order
            color_guidelines = list(dict.fromkeys(color_guidelines))[:8]
            identity_guidelines = list(dict.fromkeys(identity_guidelines))[:8]

            return {
                "color_contrast": color_guidelines if color_guidelines else ["Use appropriate color contrast for safety"],
                "familiarity_identity": identity_guidelines if identity_guidelines else ["Incorporate familiar elements"]
            }

        except Exception as e:
            print(f"Warning: RAG retrieval failed: {e}")
            return {
                "color_contrast": ["Use appropriate color contrast for dementia safety"],
                "familiarity_identity": ["Incorporate familiar and meaningful personal elements"]
            }

    def _build_analysis_prompt(self, conversation_data: Dict[str, Any]) -> List:
        """
        Build the prompt for GPT-4o to analyze preferences with RAG-retrieved guidelines.

        Args:
            conversation_data: The conversation JSON containing messages

        Returns:
            List of messages for the LLM
        """

        # Extract conversation messages
        all_messages = conversation_data.get('allMessages', [])

        # Format conversation for analysis
        conversation_text = self._format_conversation(all_messages)

        # Retrieve relevant guidelines from RAG
        print("Retrieving relevant dementia design guidelines from RAG system...")
        retrieved_guidelines = self._retrieve_guidelines(conversation_text)

        system_prompt = """You are an expert in dementia-friendly interior design and user preference analysis.
Your task is to analyze conversations between users and a chatbot to extract user preferences, then balance
those preferences against dementia design guidelines to create a safe, personalized design summary.

You will focus on two categories:
1. **Color & Contrast**: Color preferences, visual patterns, contrast needs
2. **Familiarity & Personal Identity**: Personal themes, hobbies, cultural elements, meaningful items

CRITICAL GUIDELINES:
- Dementia design safety is paramount - never compromise safety for preference
- When user preferences conflict with guidelines, find creative compromises
- Extract implicit preferences (e.g., mentioning Manchester United implies red/yellow color preference)
- Consider dementia-specific implications (e.g., red/yellow can be stimulating - assess appropriateness)
- Be specific and actionable in your summaries
- If no clear preferences are found in a category, state that explicitly

TRANSLATING INTANGIBLE TO TANGIBLE DESIGN ELEMENTS:
When users mention intangible preferences (music, songs, smells, emotions, memories, feelings), you MUST translate
them into tangible, implementable design elements for spatial design:

**Music & Songs**:
- Extract associated colors (album covers, music video aesthetics, cultural associations)
- Identify emotional qualities → translate to textures (soft/smooth for calm, rough/textured for energetic)
- Consider lyrics/themes → suggest symbolic representations or patterns
- Example: "甜蜜蜜" (Chinese love song) → soft pink/warm red tones from era, smooth rounded textures for tenderness,
  heart or floral motifs representing love and sweetness

**Smells & Scents**:
- Translate to visual color palettes (lavender → purple/soft blue, coffee → warm browns, ocean → blues/teals)
- Convert to textures/materials (fresh linen → smooth cotton fabrics, wood → natural wood grains)
- Consider source objects as decor elements (citrus → fruit bowl, flowers → floral patterns)
- Example: "Smell of grandmother's kitchen" → warm yellows/oranges, wood textures, vintage ceramic patterns

**Emotions & Feelings**:
- Map to color psychology (joy → yellows, calm → blues, comfort → earth tones)
- Translate to spatial qualities (openness → spacious layouts, coziness → enclosed nooks)
- Convert to tactile elements (security → soft textiles, energy → varied textures)

**Memories & Places**:
- Extract visual elements (beach → sandy beiges, ocean blues, natural textures)
- Identify cultural/regional design elements (traditional patterns, architectural features)
- Suggest representative objects or imagery

**Abstract Concepts**:
- Always ground in concrete design recommendations
- Use metaphorical translations (freedom → open spaces + light colors, warmth → reds/oranges + soft fabrics)

For EVERY intangible preference mentioned, you must provide:
1. The tangible translation (colors, textures, patterns, materials)
2. Rationale connecting the intangible to the tangible
3. Specific implementation suggestions compatible with dementia design guidelines

OUTPUT FORMAT:
Return a JSON object with this exact structure:
{
  "color_and_contrast": {
    "user_preferences": ["specific preference 1 (with intangible→tangible translations)", "specific preference 2"],
    "guideline_considerations": ["relevant guideline 1", "relevant guideline 2"],
    "balanced_recommendations": ["recommendation 1", "recommendation 2"],
    "confidence_level": "high|medium|low"
  },
  "familiarity_and_identity": {
    "user_preferences": ["specific preference 1 (with intangible→tangible translations)", "specific preference 2"],
    "guideline_considerations": ["relevant guideline 1", "relevant guideline 2"],
    "balanced_recommendations": ["recommendation 1", "recommendation 2"],
    "confidence_level": "high|medium|low"
  },
  "overall_summary": "Brief 2-3 sentence summary of key findings including how intangible preferences were translated to tangible design elements"
}
"""

        user_prompt = f"""DEMENTIA DESIGN GUIDELINES (Retrieved from Expert Documents):

COLOR & CONTRAST GUIDELINES:
{chr(10).join(f"- {g}" for g in retrieved_guidelines['color_contrast'])}

FAMILIARITY & IDENTITY GUIDELINES:
{chr(10).join(f"- {g}" for g in retrieved_guidelines['familiarity_identity'])}

---

CONVERSATION TO ANALYZE:
{conversation_text}

---

Please analyze this conversation and extract user preferences in the two categories (Color & Contrast, and Familiarity & Personal Identity).
Balance each preference against the dementia design guidelines provided above and provide actionable recommendations.

Return ONLY the JSON object with no additional text."""

        return [
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt)
        ]

    def _format_conversation(self, messages: List[Dict[str, Any]]) -> str:
        """
        Format conversation messages into readable text.

        Args:
            messages: List of message objects from the conversation

        Returns:
            Formatted conversation string
        """
        formatted = []
        for msg in messages:
            sender = "User" if msg.get('isUser', False) else "Chatbot"
            text = msg.get('text', '')
            timestamp = msg.get('timestamp', '')

            formatted.append(f"[{sender}] {text}")

        return "\n\n".join(formatted)

    def summarize(self, conversation_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze conversation and generate preference summary using RAG-enhanced guidelines.

        Args:
            conversation_data: The conversation JSON from the frontend

        Returns:
            Dictionary containing the summarized preferences
        """
        try:
            print("\n" + "-"*70)
            print("ANALYZING CONVERSATION")
            print("-"*70)

            # Build the analysis prompt with RAG-retrieved guidelines
            messages = self._build_analysis_prompt(conversation_data)

            # Call GPT-4o
            print("Generating preference summary with GPT-4o...")
            response = self.llm.invoke(messages)

            # Parse the JSON response - handle markdown-wrapped JSON
            response_text = response.content.strip()

            # Try to extract JSON from markdown code blocks if present
            if '```json' in response_text:
                # Extract JSON from markdown code fence
                import re
                json_match = re.search(r'```json\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1).strip()
            elif '```' in response_text:
                # Extract from generic code fence
                import re
                json_match = re.search(r'```\s*(.*?)\s*```', response_text, re.DOTALL)
                if json_match:
                    response_text = json_match.group(1).strip()

            # Parse JSON
            result = json.loads(response_text)

            # Add metadata
            result['metadata'] = {
                'timestamp': datetime.now().isoformat(),
                'model': self.llm.model_name,
                'conversation_id': conversation_data.get('_id', 'unknown'),
                'message_count': len(conversation_data.get('allMessages', [])),
                'rag_enabled': self.rag is not None and self.rag.documents_loaded,
                'vector_store': self.persist_dir if self.rag else None
            }

            print("Summary generated successfully!\n")
            return result

        except json.JSONDecodeError as e:
            print(f"Error parsing LLM response as JSON: {e}")
            print(f"Raw response: {response.content}")
            raise
        except Exception as e:
            print(f"Error during preference summarization: {e}")
            raise

    def summarize_from_file(self, input_path: str, output_path: str = None) -> Dict[str, Any]:
        """
        Load conversation from file, analyze, and optionally save result.

        Args:
            input_path: Path to conversation JSON file
            output_path: Optional path to save the summary (if None, only returns result)

        Returns:
            Dictionary containing the summarized preferences
        """
        # Load conversation
        with open(input_path, 'r', encoding='utf-8') as f:
            conversation_data = json.load(f)

        # Generate summary
        result = self.summarize(conversation_data)

        # Save if output path provided
        if output_path:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"Summary saved to: {output_path}")

        return result


def main():
    """
    Command-line interface for the preference summarizer.
    """
    parser = argparse.ArgumentParser(
        description='Analyze conversation logs to extract user preferences for dementia-friendly design using RAG'
    )
    parser.add_argument(
        '--input',
        required=True,
        help='Path to conversation JSON file'
    )
    parser.add_argument(
        '--output',
        help='Path to save the summary JSON (optional)'
    )
    parser.add_argument(
        '--pdf-folder',
        default='./pdfs',
        help='Folder containing dementia guideline PDFs (default: ./pdfs)'
    )
    parser.add_argument(
        '--persist-dir',
        default='./chroma_db',
        help='Directory to cache vector store and knowledge graph (default: ./chroma_db)'
    )
    parser.add_argument(
        '--no-rag',
        action='store_true',
        help='Disable RAG and use basic guidelines only'
    )
    parser.add_argument(
        '--model',
        default='gpt-4o',
        help='OpenAI model to use (default: gpt-4o)'
    )
    parser.add_argument(
        '--temperature',
        type=float,
        default=0.3,
        help='Model temperature (default: 0.3)'
    )

    args = parser.parse_args()

    # Check for API key
    if not os.getenv('OPENAI_API_KEY'):
        print("Error: OPENAI_API_KEY environment variable not set")
        print("Please set your OpenAI API key in .env file or environment")
        return 1

    # Create summarizer with RAG
    summarizer = PreferenceSummarizer(
        model=args.model,
        temperature=args.temperature,
        pdf_folder=args.pdf_folder,
        persist_dir=args.persist_dir,
        load_rag=not args.no_rag
    )

    # Generate default output path if not provided
    output_path = args.output
    if not output_path:
        input_base = os.path.splitext(args.input)[0]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = f"{input_base}_preferences_summary_{timestamp}.json"

    try:
        # Run summarization
        result = summarizer.summarize_from_file(args.input, output_path)

        # Print summary to console
        print("\n" + "="*60)
        print("PREFERENCE SUMMARY")
        print("="*60)
        print(f"\nOverall: {result.get('overall_summary', 'N/A')}")

        print("\n--- Color & Contrast ---")
        color_prefs = result.get('color_and_contrast', {})
        print(f"Confidence: {color_prefs.get('confidence_level', 'N/A')}")
        print("Recommendations:")
        for rec in color_prefs.get('balanced_recommendations', []):
            print(f"  • {rec}")

        print("\n--- Familiarity & Identity ---")
        identity_prefs = result.get('familiarity_and_identity', {})
        print(f"Confidence: {identity_prefs.get('confidence_level', 'N/A')}")
        print("Recommendations:")
        for rec in identity_prefs.get('balanced_recommendations', []):
            print(f"  • {rec}")

        print("\n" + "="*60)
        print(f"Full results saved to: {output_path}")

        return 0

    except Exception as e:
        print(f"\nError: {e}")
        return 1


if __name__ == "__main__":
    exit(main())
