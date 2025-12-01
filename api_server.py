"""
RAG-Langchain Microservice API
==============================
FastAPI server that wraps dementia_pipeline.py for microservice architecture.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
import json
import re
from datetime import datetime

from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Load shared environment variables
from dotenv import load_dotenv
parent_dir = Path(__file__).parent.parent
env_path = parent_dir / ".env"
if env_path.exists():
    load_dotenv(env_path)

# Import the dementia pipeline and preference summarizer
from dementia_pipeline import DementiaImageAnalysisPipeline
from summarize_preferences import PreferenceSummarizer

# Initialize FastAPI app
app = FastAPI(
    title="RAG-Langchain Service",
    description="Dementia home safety image analysis using RAG + Vision LLM",
    version="1.0.0"
)

# Global instances
pipeline: Optional[DementiaImageAnalysisPipeline] = None
preference_summarizer: Optional[PreferenceSummarizer] = None


class AnalysisResponse(BaseModel):
    """Response model for analysis results"""
    success: bool
    analysis_text: Optional[str] = None
    analysis_json: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    service: str
    version: str


class ConversationData(BaseModel):
    """Request model for conversation data"""
    allMessages: list
    selectedTopics: Optional[list] = None
    topicConversations: Optional[Dict[str, Any]] = None
    timestamp: Optional[str] = None
    _id: Optional[str] = None


class PreferenceSummaryResponse(BaseModel):
    """Response model for preference summarization"""
    success: bool
    summary: Optional[Dict[str, Any]] = None
    error: Optional[str] = None


def extract_json_from_analysis(analysis_text: str) -> Optional[Dict]:
    """Extract JSON summary from analysis text"""
    try:
        # Find JSON block in markdown code fence
        json_match = re.search(r'```json\s*(\{.*?\})\s*```', analysis_text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))
    except Exception as e:
        print(f"Error extracting JSON: {e}")
    return None


def get_pipeline():
    """Lazy-load pipeline on first use"""
    global pipeline
    if pipeline is None:
        print("Initializing RAG-Langchain pipeline (first request)...")
        pipeline = DementiaImageAnalysisPipeline(
            pdf_folder="./Dementia Guidelines",
            persist_dir="./chroma_db"
        )
        # Load guidelines (will use cache if available)
        pipeline.load_guidelines(use_llm_extraction=False, force_reload=False)
        print("RAG-Langchain pipeline ready!")
    return pipeline


def get_preference_summarizer():
    """Lazy-load preference summarizer on first use"""
    global preference_summarizer
    if preference_summarizer is None:
        print("Initializing Preference Summarizer (first request)...")
        preference_summarizer = PreferenceSummarizer(
            model="gpt-4o",
            temperature=0.3,
            pdf_folder="./Dementia Guidelines",
            persist_dir="./chroma_db",
            load_rag=True
        )
        print("Preference Summarizer ready!")
    return preference_summarizer


@app.on_event("startup")
async def startup_event():
    """Service startup - quick initialization"""
    print("RAG-Langchain service starting...")
    print("Note: Pipeline will be initialized on first request")
    print("RAG-Langchain service ready!")


@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    pass


@app.get("/", response_model=HealthResponse)
async def root():
    """Root endpoint - service info"""
    return {
        "status": "running",
        "service": "RAG-Langchain",
        "version": "1.0.0"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    # Service is healthy as long as it's running
    # Pipeline loads lazily on first request
    return {
        "status": "healthy",
        "service": "RAG-Langchain",
        "version": "1.0.0"
    }


@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_image(
    file: UploadFile = File(...),
    user_context: Optional[str] = Form(None)
):
    """
    Analyze an uploaded image for dementia safety issues.

    Args:
        file: Image file (JPEG, PNG, etc.)
        user_context: Optional JSON string with user conversation and assessment data

    Returns:
        Analysis results with both text and JSON
    """
    temp_path = None
    try:
        # Lazy-load pipeline on first request
        current_pipeline = get_pipeline()

        # Parse user context if provided
        context_data = None
        if user_context:
            try:
                context_data = json.loads(user_context)
            except json.JSONDecodeError as e:
                print(f"[API] Warning: Failed to parse user_context JSON: {e}")

        # Save uploaded file temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S%f")
        temp_path = Path(f"/tmp/analysis_{timestamp}_{file.filename}")
        temp_path.parent.mkdir(parents=True, exist_ok=True)

        with open(temp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        # Run analysis with user context
        print(f"Analyzing image: {temp_path}")
        analysis_text = current_pipeline.analyze_image(str(temp_path), user_context_data=context_data)

        # Extract JSON from analysis
        analysis_json = extract_json_from_analysis(analysis_text)

        return {
            "success": True,
            "analysis_text": analysis_text,
            "analysis_json": analysis_json
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        return {
            "success": False,
            "error": error_detail
        }

    finally:
        # Clean up temp file
        if temp_path and temp_path.exists():
            try:
                temp_path.unlink()
            except:
                pass


@app.post("/summarize-preferences", response_model=PreferenceSummaryResponse)
async def summarize_preferences(conversation: ConversationData):
    """
    Analyze conversation logs to extract user preferences for dementia-friendly design.

    Args:
        conversation: Conversation data with messages, topics, etc.

    Returns:
        Preference summary with color/contrast and familiarity/identity recommendations
    """
    try:
        # Lazy-load preference summarizer on first request
        summarizer = get_preference_summarizer()

        # Convert Pydantic model to dict for processing
        conversation_dict = conversation.model_dump()

        print(f"[API] Summarizing preferences for conversation with {len(conversation_dict.get('allMessages', []))} messages")

        # Run preference summarization
        summary = summarizer.summarize(conversation_dict)

        return {
            "success": True,
            "summary": summary
        }

    except Exception as e:
        import traceback
        error_detail = f"{str(e)}\n{traceback.format_exc()}"
        print(f"[API] Error in preference summarization: {error_detail}")
        return {
            "success": False,
            "error": error_detail
        }


if __name__ == "__main__":
    # Get configuration from environment
    host = os.getenv("RAG_SERVICE_HOST", "127.0.0.1")
    port = int(os.getenv("RAG_SERVICE_PORT", "8001"))

    print(f"Starting RAG-Langchain service on {host}:{port}")

    uvicorn.run(
        app,
        host=host,
        port=port,
        log_level="info"
    )