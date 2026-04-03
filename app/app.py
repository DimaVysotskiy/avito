import json
import httpx
from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from typing import List

from .core.settings import settings
from .core.mc_reference import McReference
from .core.detector import McCandidateDetector
from .schemas import SplitPredictionRequest, SplitPredictionResponse, DraftSchema, CandidateMc

from google import genai







@asynccontextmanager
async def lifespan(app: FastAPI):
    mc_dataset = McReference(
        path=settings.path_to_mc_search_dataset,
        encoding=settings.encoding_to_mc_search_dataset_csv,
    )
    app.state.detector = McCandidateDetector(mc_dataset.get_data())
    app.state.llm = genai.Client(api_key=settings.gemini_api_key)
    yield
    await app.state.http_client.aclose()


app = FastAPI(lifespan=lifespan)


@app.post("/predict", response_model=SplitPredictionResponse)
async def predict(request: SplitPredictionRequest):
    # Первичная работа детектора над текстом
    detector: McCandidateDetector = app.state.detector
    detector_response = detector.detect(
        raw_text=request.description,
        source_mc_id=request.mcId,
    )

    # Нет кандидатов — ранний выход без LLM
    if not detector_response:
        return SplitPredictionResponse(
            detectedMcIds=[],
            shouldSplit=False,
            drafts=[],
        )

    user_prompt = _build_user_prompt(request, detector_response.detected_mc)

    

    

    
    

    return 