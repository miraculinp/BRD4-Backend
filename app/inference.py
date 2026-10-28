"""Cached BRD4 inference and applicability-domain calculations."""
from __future__ import annotations

import io
import json
import logging
import os
import threading
from collections import OrderedDict
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

from .standardize import standardize_mol, standardize_smiles

LOGGER = logging.getLogger("brd4.inference")
MODEL_NAMES = ("ExtraTreesRegressor", "CatBoostRegressor", "SVR")
ASSET_DIR = Path(os.getenv("BRD4_ASSET_DIR", Path(__file__).resolve().parents[1] / "app_assets"))
CACHE_LIMIT = 1024


class AssetError(RuntimeError):
    pass


class FeatureError(ValueError):
    pass


@lru_cache(maxsize=1)
def load_assets() -> dict:
    """Load estimators and build training-only AD references once per worker."""
    required = [
        ASSET_DIR / "app_config.json",
        ASSET_DIR / "selected_features.csv",
        ASSET_DIR / "training_reference.joblib",
        *[ASSET_DIR / f"final_{name}.joblib" for name in MODEL_NAMES],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise AssetError("Backend model assets are missing: " + "; ".join(missing))

    config = json.loads((ASSET_DIR / "app_config.json").read_text(encoding="utf-8"))
    features_df = pd.read_csv(ASSET_DIR / "selected_features.csv")
    if "feature" not in features_df:
        raise AssetError("selected_features.csv must contain a 'feature' column.")
    selected = features_df["feature"].astype(str).tolist()
    expected = (250, 250, 103, 146, 1, 0)
    counts = (
        len(selected), len(set(selected)),
        sum(x.startswith("ecfp4_") for x in selected),
        sum(x.startswith("mordred2d_") for x in selected),
        selected.count("phys_MolLogP"),
        sum(x.startswith("mordred3d_") for x in selected),
    )
    if counts != expected:
        raise AssetError(
            "Expected 250 unique features (103 Morgan, 146 Mordred 2D, "
            f"phys_MolLogP, zero 3D); received {counts}."
        )

    reference = joblib.load(ASSET_DIR / "training_reference.joblib")
    if not isinstance(reference, dict) or "X_train" not in reference:
        raise AssetError("training_reference.joblib must contain X_train and feature_names.")
    X_train = reference["X_train"]
    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train, columns=reference.get("feature_names", selected))
    if list(map(str, reference.get("feature_names", []))) != selected:
        raise AssetError("Training reference feature order differs from selected_features.csv.")
    if list(map(str, X_train.columns)) != selected:
        X_train = X_train.loc[:, selected]
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    train_array = X_train.to_numpy(dtype=float)
    if train_array.ndim != 2 or train_array.shape[1] != 250 or not np.isfinite(train_array).all():
        raise AssetError("Training reference has missing/nonfinite values or wrong shape; no imputation is used.")

    models = {}
    for name in MODEL_NAMES:
        bundle = joblib.load(ASSET_DIR / f"final_{name}.joblib")
        if isinstance(bundle, dict) and "model" in bundle:
            model = bundle["model"]
            model_features = bundle.get("feature_names", selected)
            if list(map(str, model_features)) != selected:
                raise AssetError(f"{name} feature order differs from the fixed Stage 05 order.")
        else:
            # Accept an estimator-only joblib if its input width matches; assets from Stage 13
            # normally use the bundle form with the feature list.
            model = bundle
        models[name] = model

    scaler = StandardScaler().fit(train_array)
    train_scaled = scaler.transform(train_array)
    k = min(5, len(train_array) - 1)
    if k < 1:
        raise AssetError("At least two training reference rows are required for AD.")
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean", n_jobs=1).fit(train_scaled)
    train_knn = neighbors.kneighbors(return_distance=True)[0].mean(axis=1)
    knn_threshold = float(train_knn.mean() + 2.0 * train_knn.std(ddof=0))

    train_min = train_array.min(axis=0)
    train_max = train_array.max(axis=0)
    design_train = np.column_stack([np.ones(len(train_scaled)), train_scaled])
    gram_inverse = np.linalg.pinv(design_train.T @ design_train, hermitian=True)
    leverage_threshold = float(3 * len(selected) / len(train_array))

    generator = rdFingerprintGenerator.GetMorganGenerator(
        radius=int(config.get("fingerprint_radius", 2)),
        fpSize=int(config.get("fingerprint_bits", 2048)),
        includeChirality=bool(config.get("use_chirality", True)),
    )

    try:
        from mordred import Calculator, descriptors
    except ImportError as exc:
        raise AssetError("Mordred Community is required; install mordredcommunity.") from exc
    all_2d = Calculator(descriptors, ignore_3D=True)
    wanted = {name[len("mordred2d_"):] for name in selected if name.startswith("mordred2d_")}
    descriptor_objects = [d for d in all_2d.descriptors if str(d) in wanted]
    found = {str(d) for d in descriptor_objects}
    if found != wanted:
        raise AssetError(f"Mordred descriptor mismatch; missing: {sorted(wanted-found)[:10]}")
    mordred_calculator = Calculator(descriptor_objects, ignore_3D=True)

    assets = {
        "config": config,
        "selected": selected,
        "models": models,
        "X_train": X_train,
        "scaler": scaler,
        "neighbors": neighbors,
        "knn_threshold": knn_threshold,
        "train_min": train_min,
        "train_max": train_max,
        "gram_inverse": gram_inverse,
        "leverage_threshold": leverage_threshold,
        "morgan_generator": generator,
        "mordred_calculator": mordred_calculator,
        "mordred_names": {str(d): f"mordred2d_{d}" for d in descriptor_objects},
        "feature_cache": OrderedDict(),
        "cache_lock": threading.Lock(),
        "n_training": len(train_array),
    }
    # Fail at startup if saved estimators cannot accept the declared fixed 250 inputs.
    sample = train_array[:1]
    with threadpool_limits(limits=1):
        for name, model in models.items():
            pred = np.asarray(model.predict(sample), dtype=float).reshape(-1)
            if pred.size != 1 or not np.isfinite(pred[0]):
                raise AssetError(f"{name} failed the startup compatibility prediction.")
    LOGGER.info("Loaded three BRD4 models, %d fixed features and %d training references", len(selected), len(train_array))
    return assets


def _to_record(name: str, mol: Chem.Mol, source: str, assets: dict, input_smiles: str | None = None):
    try:
        standardized, canonical = standardize_mol(
            mol, remove_salts=bool(assets["config"].get("remove_salts", True))
        )
    except ValueError as exc:
        return None, str(exc)
    record = {
        "input_name": str(name or "Molecule"),
        "input_format": source,
        "canonical_smiles": canonical,
        "mol": standardized,
    }
    if input_smiles:
        record["input_smiles"] = input_smiles
    return record, None


def parse_smiles_text(text: str, config: dict):
    records, failures = [], []
    for line_no, raw in enumerate((text or "").splitlines(), 1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        pieces = line.split(maxsplit=1)
        smiles = pieces[0]
        name = pieces[1].strip() if len(pieces) > 1 else f"Molecule_{line_no}"
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            failures.append({"input_name": name, "record": line_no, "reason": "RDKit could not parse the SMILES."})
            continue
        record, error = _to_record(name, mol, "SMILES", {"config": config}, smiles)
        if error:
            failures.append({"input_name": name, "record": line_no, "reason": error})
        else:
            records.append(record)
    return records, failures


def parse_sdf(data: bytes, config: dict):
    supplier = Chem.ForwardSDMolSupplier(io.BytesIO(data), sanitize=True, removeHs=True, strictParsing=False)
    records, failures = [], []
    for number, mol in enumerate(supplier, 1):
        if mol is None:
            failures.append({"input_name": f"SDF_record_{number}", "record": number, "reason": "RDKit could not parse or sanitize this SDF record."})
            continue
        name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"SDF_record_{number}"
        record, error = _to_record(name, mol, "SDF", {"config": config})
        if error:
            failures.append({"input_name": name, "record": number, "reason": error})
        else:
            records.append(record)
    return records, failures


def parse_mol2(data: bytes, config: dict):
    text = data.decode("utf-8", errors="replace")
    marker = "@<TRIPOS>MOLECULE"
    chunks = [x for x in text.split(marker) if x.strip()]
    records, failures = [], []
    if not chunks:
        return [], [{"input_name": "MOL2_upload", "record": 1, "reason": "No @<TRIPOS>MOLECULE block was found."}]
    for number, chunk in enumerate(chunks, 1):
        block = marker + chunk
        lines = chunk.splitlines()
        name = lines[0].strip() if lines and lines[0].strip() else f"MOL2_record_{number}"
        mol = Chem.MolFromMol2Block(block, sanitize=True, removeHs=True)
        if mol is None:
            failures.append({"input_name": name, "record": number, "reason": "RDKit could not parse or sanitize this MOL2 record."})
            continue
        record, error = _to_record(name, mol, "MOL2", {"config": config})
        if error:
            failures.append({"input_name": name, "record": number, "reason": error})
        else:
            records.append(record)
    return records, failures


def _molecule_key(record: dict) -> str:
    return record["canonical_smiles"]


def calculate_fixed_features(records: list[dict], assets: dict) -> pd.DataFrame:
    """Compute the selected feature vector once per unique molecule; never impute."""
    selected = assets["selected"]
    cache: OrderedDict = assets["feature_cache"]
    lock = assets["cache_lock"]
    unique = OrderedDict()
    for record in records:
        unique.setdefault(_molecule_key(record), record["mol"])

    with lock:
        rows = {key: cache[key].copy() for key in unique if key in cache}
    missing_keys = [key for key in unique if key not in rows]
    if missing_keys:
        mols = [unique[key] for key in missing_keys]
        bit_features = [(name, int(name.rsplit("_", 1)[1])) for name in selected if name.startswith("ecfp4_")]
        generated = []
        for mol in mols:
            fp = assets["morgan_generator"].GetFingerprint(mol)
            on_bits = set(fp.GetOnBits())
            values = {name: float(bit in on_bits) for name, bit in bit_features}
            values["phys_MolLogP"] = float(Descriptors.MolLogP(mol))
            generated.append(values)

        mordred_calc = assets["mordred_calculator"]
        try:
            mordred_df = mordred_calc.pandas(mols, nproc=1).apply(pd.to_numeric, errors="coerce")
        except Exception as exc:
            raise FeatureError(f"Mordred descriptor calculation failed: {type(exc).__name__}: {exc}") from exc
        mordred_df.columns = [f"mordred2d_{str(c)}" for c in mordred_calc.descriptors]
        for i, row in enumerate(generated):
            for column in mordred_df.columns:
                row[column] = mordred_df.iloc[i][column]

        for key, row in zip(missing_keys, generated):
            vector = np.asarray([row.get(name, np.nan) for name in selected], dtype=float)
            rows[key] = vector
            with lock:
                cache[key] = vector.copy()
                cache.move_to_end(key)
                while len(cache) > CACHE_LIMIT:
                    cache.popitem(last=False)

    matrix = np.vstack([rows[_molecule_key(record)] for record in records])
    return pd.DataFrame(matrix, columns=selected)


def calculate_ad(X: pd.DataFrame, assets: dict) -> list[dict]:
    values = X.loc[:, assets["selected"]].to_numpy(dtype=float)
    scaled = assets["scaler"].transform(values)
    knn_distance = assets["neighbors"].kneighbors(scaled, return_distance=True)[0].mean(axis=1)
    outside = (values < assets["train_min"]) | (values > assets["train_max"])
    outside_count = outside.sum(axis=1)
    design = np.column_stack([np.ones(len(values)), scaled])
    leverage = np.einsum("ij,jk,ik->i", design, assets["gram_inverse"], design)
    result = []
    for i in range(len(values)):
        knn_inside = bool(knn_distance[i] <= assets["knn_threshold"])
        bbox_inside = bool(outside_count[i] == 0)
        leverage_inside = bool(leverage[i] <= assets["leverage_threshold"])
        result.append({
            "knn": {"distance": float(knn_distance[i]), "threshold": assets["knn_threshold"], "inside": knn_inside},
            "bounding_box": {"outside_feature_count": int(outside_count[i]), "n_features": len(assets["selected"]), "inside": bbox_inside},
            "leverage": {"value": float(leverage[i]), "threshold": assets["leverage_threshold"], "inside": leverage_inside},
            "inside_count": int(knn_inside) + int(bbox_inside) + int(leverage_inside),
            "inside_all_three": bool(knn_inside and bbox_inside and leverage_inside),
        })
    return result


def _ic50_nm(pic50: float) -> float:
    return float(10.0 ** (9.0 - pic50))


def predict_records(records: list[dict], failures: list[dict], assets: dict) -> dict:
    if not records:
        return {"results": [], "failures": failures, "n_parsed": 0, "n_scored": 0}
    try:
        X = calculate_fixed_features(records, assets)
    except FeatureError as exc:
        return {
            "results": [],
            "failures": failures + [{"input_name": "Feature calculation", "record": None, "reason": str(exc)}],
            "n_parsed": len(records), "n_scored": 0,
        }
    finite = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    results, good_positions = [], np.flatnonzero(finite)
    for i in np.flatnonzero(~finite):
        missing = X.columns[~np.isfinite(X.iloc[i].to_numpy(dtype=float))].tolist()
        failures.append({"input_name": records[i]["input_name"], "record": int(i+1),
                         "reason": "Selected model feature(s) are missing/nonfinite; no imputation is used.",
                         "features": missing[:30]})
    if len(good_positions):
        X_good = X.iloc[good_positions].loc[:, assets["selected"]]
        with threadpool_limits(limits=2):
            predictions = {
                name: np.asarray(model.predict(X_good), dtype=float).reshape(-1)
                for name, model in assets["models"].items()
            }
        if any(not np.isfinite(p).all() for p in predictions.values()):
            raise ValueError("A saved model returned a nonfinite prediction.")
        ad_rows = calculate_ad(X_good, assets)
        for local_i, source_i in enumerate(good_positions):
            row = {
                "input_name": records[source_i]["input_name"],
                "input_format": records[source_i]["input_format"],
                "canonical_smiles": records[source_i]["canonical_smiles"],
                "status": "scored",
                "predictions": {
                    name: {"pIC50": float(pred[local_i]), "IC50_nM": _ic50_nm(float(pred[local_i]))}
                    for name, pred in predictions.items()
                },
                "applicability_domain": ad_rows[local_i],
            }
            results.append(row)
    return {"results": results, "failures": failures, "n_parsed": len(records), "n_scored": len(results)}
