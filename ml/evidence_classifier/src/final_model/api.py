
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from final_model.predictor import FeverPredictor


predictor = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global predictor
    predictor = FeverPredictor()
    yield
    predictor = None


app = FastAPI(
    title="Digital Mirror ML API",
    version="1.0.0",
    lifespan=lifespan
)


class PredictionRequest(BaseModel):
    claim: str = Field(min_length=1)
    evidence: str = Field(min_length=1)
    wiki_title: str = ""


class PredictionResponse(BaseModel):
    claim: str
    evidence: str
    wiki_title: str
    prediction: Literal["SUPPORTS", "REFUTES"]
    p_supports: float
    p_refutes: float
    confidence: float
    model_version: str


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": predictor is not None,
        "model_version": "e06b"
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    if predictor is None:
        raise HTTPException(
            status_code=503,
            detail="ML model is not loaded"
        )

    try:
        return predictor.predict(
            claim=request.claim,
            evidence=request.evidence,
            wiki_title=request.wiki_title
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=422,
            detail=str(exc)
        ) from exc
