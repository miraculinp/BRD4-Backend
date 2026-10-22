"""Preflight saved assets under the Vercel Python runtime before deployment."""
from __future__ import annotations

import joblib
import numpy as np

from app.inference import ASSET_DIR, MODELS, load_assets


def main() -> None:
    assets = load_assets()
    reference = joblib.load(ASSET_DIR / "training_reference.joblib")
    row = reference["X_train"].loc[:, assets["selected"]].iloc[[0]]
    print(f"Loaded {len(assets['selected'])} selected features and {len(reference['X_train']):,} training reference rows.")
    for name in MODELS:
        prediction = np.asarray(assets["models"][name].predict(row), dtype=float)
        if prediction.shape != (1,) or not np.isfinite(prediction).all():
            raise RuntimeError(f"{name} returned an invalid prediction: {prediction!r}")
        print(f"{name}: compatible; sample pIC50 prediction={prediction[0]:.4f}")
    print("Asset preflight passed.")


if __name__ == "__main__":
    main()
