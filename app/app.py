from fastapi import FastAPI
from contextlib import asynccontextmanager
from .core.settings import settings
from .core.mc_reference import McReference
from .core.detector import McCandidateDetector
from .schemas import SplitPredictionRequest





@asynccontextmanager
async def lifespan(app: FastAPI):
    mc_dataset = McReference(
        path=settings.path_to_mc_search_dataset,
        encoding=settings.encoding_to_mc_search_dataset_csv,
    )
    app.state.detector = McCandidateDetector(mc_dataset.get_data())
    yield





app = FastAPI(lifespan=lifespan)





@app.post("/detect")
async def detect(request: SplitPredictionRequest):
    
    detector: McCandidateDetector = app.state.detector
    candidates = detector.detect(
        raw_text=request.description,
        source_mc_id=request.mcId,
    )

    return {
        "detectedMcIds": [c.mc_id for c in candidates],
        "candidates": [
            {
                "mcId": c.mc_id,
                "mcTitle": c.mc_title,
                "matchedPhrases": c.matched_phrases,
            }
            for c in candidates
        ],
    }