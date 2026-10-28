"""FastAPI service for BRD4 pIC50 prediction and applicability-domain reporting."""
from __future__ import annotations

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .inference import AssetError, FeatureError, load_assets, parse_mol2, parse_sdf, parse_smiles_text, predict_records

logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO"))
LOGGER = logging.getLogger("brd4.api")
MAX_UPLOAD_BYTES = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
MAX_MOLECULES = int(os.getenv("MAX_MOLECULES", "250"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Warm model, descriptor, and applicability-domain objects once at process start.
    load_assets()
    yield


app = FastAPI(title="BRD4 QSAR Prediction API", version="2.0.0", lifespan=lifespan)
# The API uses no cookies or credentials. Public browser clients may call it from any origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)


class SmilesRequest(BaseModel):
    smiles_text: str = Field(min_length=1, max_length=2_000_000)


def _check_limit(records, failures):
    if len(records) > MAX_MOLECULES:
        raise HTTPException(
            status_code=413,
            detail=f"This request contains {len(records)} valid molecules; the limit is {MAX_MOLECULES}. Split the input into smaller batches.",
        )


@app.get("/")
def root():
    return {"service": "BRD4 QSAR Prediction API", "health": "/api/health", "docs": "/docs"}


@app.get("/api/health")
def health():
    try:
        assets = load_assets()
    except Exception as exc:
        LOGGER.exception("Health check failed while loading model assets")
        raise HTTPException(status_code=503, detail=f"Prediction assets are unavailable: {exc}") from exc
    return {
        "status": "ok",
        "models": list(assets["models"]),
        "n_features": len(assets["selected"]),
        "n_training_reference": assets["n_training"],
        "applicability_methods": ["standardized_5nn", "training_bounding_box", "leverage"],
    }


@app.post("/api/predict/smiles")
def predict_smiles(payload: SmilesRequest):
    try:
        assets = load_assets()
        records, failures = parse_smiles_text(payload.smiles_text, assets["config"])
        _check_limit(records, failures)
        return predict_records(records, failures, assets)
    except HTTPException:
        raise
    except (AssetError, FeatureError, ValueError, ImportError) as exc:
        LOGGER.exception("SMILES prediction request failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected SMILES prediction failure")
        raise HTTPException(status_code=500, detail=f"Prediction failed ({type(exc).__name__}). Check backend logs for details.") from exc


@app.post("/api/predict/file")
async def predict_file(file: UploadFile = File(...)):
    filename = file.filename or "molecules"
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if suffix not in {"sdf", "mol2"}:
        raise HTTPException(status_code=415, detail="Upload an .sdf or .mol2 file.")
    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="Upload exceeds the 20 MB file limit.")
    try:
        assets = load_assets()
        records, failures = parse_sdf(data, assets["config"]) if suffix == "sdf" else parse_mol2(data, assets["config"])
        _check_limit(records, failures)
        return predict_records(records, failures, assets)
    except HTTPException:
        raise
    except (AssetError, FeatureError, ValueError, ImportError) as exc:
        LOGGER.exception("Molecule file prediction request failed")
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected file prediction failure")
        raise HTTPException(status_code=500, detail=f"Prediction failed ({type(exc).__name__}). Check backend logs for details.") from exc
    finally:
        await file.close()
