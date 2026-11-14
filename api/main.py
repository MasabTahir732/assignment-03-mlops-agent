"""Inference is ready only when a quality-approved, provenance-bearing model loads."""

import os
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from training.train import FEATURES


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.bundle = None
    model_path = Path(os.getenv("MODEL_PATH", "models/model.joblib"))
    if model_path.exists():
        try:
            bundle = joblib.load(model_path)
            if bundle["metadata"]["quality_passed"]:
                app.state.bundle = bundle
        except (OSError, ValueError, KeyError, TypeError):
            pass
    yield


app = FastAPI(title="Tiny Iris inference service", version="1.0.0", lifespan=lifespan)


class PredictionInput(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    sepal_length: float = Field(gt=0, le=20)
    sepal_width: float = Field(gt=0, le=20)
    petal_length: float = Field(gt=0, le=20)
    petal_width: float = Field(gt=0, le=20)


@app.get("/health/live")
def live():
    return {"status": "alive"}


@app.get("/health/ready")
def ready():
    if getattr(app.state, "bundle", None) is None:
        raise HTTPException(503, "No quality-approved model is loaded")
    return {"status": "ready"}


@app.get("/metadata")
def metadata():
    ready()
    return {"app_version": os.getenv("APP_VERSION", "1.0.0"), **app.state.bundle["metadata"]}


@app.post("/predict")
def predict(payload: PredictionInput):
    ready()
    bundle = app.state.bundle
    frame = pd.DataFrame([payload.model_dump()], columns=FEATURES)
    index = int(bundle["model"].predict(frame)[0])
    return {
        "class_id": index,
        "class_name": bundle["metadata"]["classes"][index],
        "run_id": bundle["metadata"]["run_id"],
    }
