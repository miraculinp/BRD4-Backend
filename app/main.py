"""FastAPI endpoints for BRD4 molecule prediction."""
from __future__ import annotations

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .inference import (
    load_assets,
    parse_mol2,
    parse_sdf,
    parse_smiles_text,
    predict_records,
)

MAX_UPLOAD_BYTES = 20 * 1024 * 1024

app = FastAPI(title="BRD4 QSAR Prediction API", version="1.0.0")

# Allow browser requests from any origin. Credentials remain disabled.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


class SmilesRequest(BaseModel):
    smiles_text: str = Field(min_length=1, max_length=2_000_000)


@app.get("/api/health")
def health():
    try:
        assets = load_assets()
    except Exception as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Prediction assets are unavailable: {exc}",
        ) from exc

    return {"status": "ok", "models": list(assets["models"])}


@app.post("/api/predict/smiles")
def predict_smiles(payload: SmilesRequest):
    try:
        assets = load_assets()
        records, failures = parse_smiles_text(
            payload.smiles_text,
            assets["config"],
        )
        return predict_records(records, failures, assets)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model assets are not installed: {exc}",
        ) from exc
    except (ValueError, ImportError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc


@app.post("/api/predict/file")
async def predict_file(file: UploadFile = File(...)):
    filename = file.filename or "molecules"
    suffix = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""

    if suffix not in {"sdf", "mol2"}:
        raise HTTPException(
            status_code=415,
            detail="Upload a .sdf or .mol2 file.",
        )

    data = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(
            status_code=413,
            detail="File exceeds the 20 MB upload limit.",
        )

    try:
        assets = load_assets()

        if suffix == "sdf":
            records, failures = parse_sdf(data, assets["config"])
        else:
            records, failures = parse_mol2(data, assets["config"])

        return predict_records(records, failures, assets)
    except FileNotFoundError as exc:
        raise HTTPException(
            status_code=503,
            detail=f"Model assets are not installed: {exc}",
        ) from exc
    except (ValueError, ImportError) as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {type(exc).__name__}: {exc}",
        ) from exc