"""Copy the six Notebook 13 inference assets into this backend's app_assets folder."""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

ASSET_NAMES = (
    "app_config.json",
    "selected_features.csv",
    "training_reference.joblib",
    "final_ExtraTreesRegressor.joblib",
    "final_CatBoostRegressor.joblib",
    "final_SVR.joblib",
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Folder containing the six Notebook 13 inference assets")
    args = parser.parse_args()
    source = args.source.expanduser().resolve()
    destination = Path(__file__).resolve().parent / "app_assets"
    missing = [name for name in ASSET_NAMES if not (source / name).is_file()]
    if missing:
        raise SystemExit(f"Missing assets in {source}:\n" + "\n".join(missing))
    destination.mkdir(parents=True, exist_ok=True)
    for name in ASSET_NAMES:
        shutil.copy2(source / name, destination / name)
        print(f"Copied {name} ({(destination / name).stat().st_size / (1024**2):.1f} MB)")
    print(f"\nAssets ready in: {destination}")
    print("Next, install requirements and run: python validate_assets.py")


if __name__ == "__main__":
    main()
