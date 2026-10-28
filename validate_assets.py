"""Run the same saved-asset and model compatibility checks used by the API startup."""
from __future__ import annotations

from app.inference import load_assets


def main() -> None:
    assets = load_assets()
    print(f"PASS: {len(assets['selected'])} fixed features")
    print(f"PASS: {assets['n_training']:,} training reference molecules")
    print("PASS: " + ", ".join(assets["models"].keys()))
    print(f"PASS: 5-NN threshold = {assets['knn_threshold']:.6f}")
    print(f"PASS: leverage threshold = {assets['leverage_threshold']:.6f}")


if __name__ == "__main__":
    main()
