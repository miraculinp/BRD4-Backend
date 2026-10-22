"""Inference and applicability-domain calculations for the standalone API."""
from __future__ import annotations

import io
import json
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import Descriptors, rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

RDLogger.DisableLog("rdApp.warning")
MODELS = ("ExtraTreesRegressor", "CatBoostRegressor", "SVR")
ASSET_DIR = Path(__file__).resolve().parents[1] / "app_assets"
MAX_RECORDS = 100


@lru_cache(maxsize=1)
def load_assets() -> dict:
    required = [
        ASSET_DIR / "app_config.json",
        ASSET_DIR / "selected_features.csv",
        ASSET_DIR / "training_reference.joblib",
        *[ASSET_DIR / f"final_{name}.joblib" for name in MODELS],
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing prediction assets:\n" + "\n".join(missing))

    config = json.loads((ASSET_DIR / "app_config.json").read_text(encoding="utf-8"))
    selected = pd.read_csv(ASSET_DIR / "selected_features.csv")["feature"].astype(str).tolist()
    if len(selected) != 250 or len(set(selected)) != 250:
        raise ValueError(f"Expected 250 unique fixed features; found {len(selected)}.")
    counts = (
        sum(name.startswith("ecfp4_") for name in selected),
        sum(name.startswith("mordred2d_") for name in selected),
        sum(name.startswith("phys_") for name in selected),
        sum(name.startswith("mordred3d_") for name in selected),
    )
    if counts != (103, 146, 1, 0) or selected.count("phys_MolLogP") != 1:
        raise ValueError(f"Unexpected fixed-feature composition: {counts}.")

    reference = joblib.load(ASSET_DIR / "training_reference.joblib")
    X_train = reference["X_train"].loc[:, selected].apply(pd.to_numeric, errors="coerce")
    if list(map(str, reference["feature_names"])) != selected:
        raise ValueError("Training reference feature order does not match the fixed list.")
    if X_train.empty or not np.isfinite(X_train.to_numpy(dtype=float)).all():
        raise ValueError("Training reference contains missing or infinite feature values.")

    models = {}
    for model_name in MODELS:
        bundle = joblib.load(ASSET_DIR / f"final_{model_name}.joblib")
        if not isinstance(bundle, dict) or "model" not in bundle:
            raise ValueError(f"Unexpected model bundle for {model_name}.")
        if list(map(str, bundle.get("feature_names", []))) != selected:
            raise ValueError(f"{model_name} feature order differs from the fixed list.")
        models[model_name] = bundle["model"]

    scaler = StandardScaler().fit(X_train)
    train_scaled = scaler.transform(X_train)
    k = min(5, len(X_train) - 1)
    if k < 1:
        raise ValueError("At least two training reference rows are required.")
    neighbors = NearestNeighbors(n_neighbors=k, metric="euclidean").fit(train_scaled)
    train_knn = neighbors.kneighbors(return_distance=True)[0].mean(axis=1)
    knn_threshold = float(train_knn.mean() + 2 * train_knn.std(ddof=0))

    design_train = np.column_stack([np.ones(len(train_scaled)), train_scaled])
    return {
        "config": config,
        "selected": selected,
        "models": models,
        "scaler": scaler,
        "neighbors": neighbors,
        "knn_threshold": knn_threshold,
        "train_min": X_train.min(axis=0).to_numpy(dtype=float),
        "train_max": X_train.max(axis=0).to_numpy(dtype=float),
        "gram_inverse": np.linalg.pinv(design_train.T @ design_train, hermitian=True),
        "leverage_threshold": float(3 * len(selected) / len(X_train)),
        "morgan_generator": rdFingerprintGenerator.GetMorganGenerator(
            radius=int(config["fingerprint_radius"]),
            fpSize=int(config["fingerprint_bits"]),
            includeChirality=bool(config["use_chirality"]),
        ),
    }


def _standardize_mol(mol: Chem.Mol, config: dict):
    if mol is None:
        return None, None
    try:
        if bool(config.get("remove_salts", True)):
            mol = rdMolStandardize.LargestFragmentChooser().choose(mol)
        mol = rdMolStandardize.Cleanup(mol)
        mol = rdMolStandardize.Uncharge(mol)
        if mol is None or mol.GetNumAtoms() == 0:
            return None, None
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return mol, smiles
    except Exception:
        return None, None


def _record(name: str, mol: Chem.Mol, input_format: str, config: dict):
    standardized, smiles = _standardize_mol(mol, config)
    if standardized is None:
        return None, "Invalid structure or failed standardization."
    return {
        "input_name": str(name or "Molecule"),
        "input_format": input_format,
        "canonical_smiles": smiles,
        "mol": standardized,
    }, None


def parse_smiles_text(text: str, config: dict):
    records, failures = [], []
    submitted = 0
    for line_no, line in enumerate(text.splitlines(), start=1):
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        submitted += 1
        if submitted > MAX_RECORDS:
            raise ValueError(f"Limit input to {MAX_RECORDS} molecules per request.")
        parts = line.split(maxsplit=1)
        name = parts[1].strip() if len(parts) > 1 else f"Molecule_{line_no}"
        record, error = _record(name, Chem.MolFromSmiles(parts[0]), "SMILES", config)
        if error:
            failures.append({"input_name": name, "record": line_no, "reason": error})
        else:
            records.append(record)
    return records, failures


def parse_sdf(data: bytes, config: dict):
    records, failures = [], []
    supplier = Chem.ForwardSDMolSupplier(
        io.BytesIO(data), sanitize=True, removeHs=True, strictParsing=False
    )
    for number, mol in enumerate(supplier, start=1):
        if number > MAX_RECORDS:
            raise ValueError(f"Limit input to {MAX_RECORDS} molecules per request.")
        name = mol.GetProp("_Name") if mol is not None and mol.HasProp("_Name") else f"SDF_record_{number}"
        if mol is None:
            failures.append({"input_name": name, "record": number, "reason": "RDKit could not parse this SDF record."})
            continue
        record, error = _record(name, mol, "SDF", config)
        if error:
            failures.append({"input_name": name, "record": number, "reason": error})
        else:
            records.append(record)
    return records, failures


def parse_mol2(data: bytes, config: dict):
    records, failures = [], []
    text = data.decode("utf-8", errors="replace")
    marker = "@<TRIPOS>MOLECULE"
    chunks = [chunk for chunk in text.split(marker) if chunk.strip()]
    if not chunks:
        return [], [{"input_name": "MOL2 upload", "record": 1, "reason": "No MOL2 molecule block was found."}]
    if len(chunks) > MAX_RECORDS:
        raise ValueError(f"Limit input to {MAX_RECORDS} molecules per request.")
    for number, chunk in enumerate(chunks, start=1):
        lines = chunk.splitlines()
        name = lines[0].strip() if lines and lines[0].strip() else f"MOL2_record_{number}"
        mol = Chem.MolFromMol2Block(marker + chunk, sanitize=True, removeHs=True)
        record, error = _record(name, mol, "MOL2", config)
        if error:
            failures.append({"input_name": name, "record": number, "reason": error})
        else:
            records.append(record)
    return records, failures


@lru_cache(maxsize=1)
def _mordred_calculator(selected_names: tuple[str, ...]):
    try:
        from mordred import Calculator, descriptors
    except ImportError as exc:
        raise ImportError("Install Mordred Community with `pip install mordredcommunity`.") from exc
    all_calculator = Calculator(descriptors, ignore_3D=True)
    wanted = {name.removeprefix("mordred2d_") for name in selected_names if name.startswith("mordred2d_")}
    chosen = [descriptor for descriptor in all_calculator.descriptors if str(descriptor) in wanted]
    found = {str(descriptor) for descriptor in chosen}
    absent = sorted(wanted - found)
    if absent:
        raise ValueError(f"Installed Mordred Community lacks selected descriptors: {absent[:10]}")
    return Calculator(chosen, ignore_3D=True)


def calculate_features(records: list[dict], assets: dict) -> pd.DataFrame:
    selected = assets["selected"]
    rows = [dict() for _ in records]
    morgan_features = [name for name in selected if name.startswith("ecfp4_")]
    for i, record in enumerate(records):
        fp = assets["morgan_generator"].GetFingerprint(record["mol"])
        bits = np.zeros(int(assets["config"]["fingerprint_bits"]), dtype=np.uint8)
        DataStructs.ConvertToNumpyArray(fp, bits)
        for feature in morgan_features:
            rows[i][feature] = float(bits[int(feature.rsplit("_", 1)[1])])
        rows[i]["phys_MolLogP"] = float(Descriptors.MolLogP(record["mol"]))

    calculator = _mordred_calculator(tuple(selected))
    mordred = calculator.pandas([record["mol"] for record in records])
    mordred.columns = [f"mordred2d_{descriptor}" for descriptor in calculator.descriptors]
    mordred = mordred.apply(pd.to_numeric, errors="coerce")
    for feature in selected:
        if feature.startswith("mordred2d_"):
            for i, value in enumerate(mordred[feature].to_numpy(dtype=float)):
                rows[i][feature] = value
    return pd.DataFrame(rows).reindex(columns=selected).apply(pd.to_numeric, errors="coerce")


def applicability_domain(X: pd.DataFrame, assets: dict) -> pd.DataFrame:
    values = X.loc[:, assets["selected"]].to_numpy(dtype=float)
    scaled = assets["scaler"].transform(values)
    distances = assets["neighbors"].kneighbors(scaled, return_distance=True)[0].mean(axis=1)
    inside_knn = distances <= assets["knn_threshold"]
    outside = (values < assets["train_min"]) | (values > assets["train_max"])
    outside_count = outside.sum(axis=1)
    design = np.column_stack([np.ones(len(scaled)), scaled])
    leverage = np.einsum("ij,jk,ik->i", design, assets["gram_inverse"], design)
    inside_lev = leverage <= assets["leverage_threshold"]
    return pd.DataFrame({
        "mean_5nn_distance": distances,
        "5nn_threshold": assets["knn_threshold"],
        "5nn_distance_over_threshold": distances / assets["knn_threshold"],
        "inside_5nn_AD": inside_knn,
        "features_outside_training_range": outside_count,
        "fraction_features_outside_training_range": outside_count / len(assets["selected"]),
        "inside_bounding_box_AD": outside_count == 0,
        "leverage": leverage,
        "leverage_threshold_h_star": assets["leverage_threshold"],
        "leverage_over_threshold": leverage / assets["leverage_threshold"],
        "inside_leverage_AD": inside_lev,
        "inside_all_three_AD": inside_knn & (outside_count == 0) & inside_lev,
    }, index=X.index)


def predict_records(records: list[dict], initial_failures: list[dict], assets: dict):
    if not records:
        return {"results": [], "failures": initial_failures, "n_parsed": 0, "n_scored": 0}
    X = calculate_features(records, assets)
    finite = np.isfinite(X.to_numpy(dtype=float)).all(axis=1)
    results = []
    for i, ok in enumerate(finite):
        if not ok:
            bad = ~np.isfinite(X.iloc[i].to_numpy(dtype=float))
            initial_failures.append({
                "input_name": records[i]["input_name"],
                "record": i + 1,
                "reason": "Not scored: selected feature(s) are missing/nonfinite; no imputation was applied.",
                "feature_issue": ", ".join(X.columns[bad].astype(str).tolist()[:25]),
            })
    positions = np.flatnonzero(finite)
    if len(positions):
        X_valid = X.iloc[positions].loc[:, assets["selected"]]
        ad = applicability_domain(X_valid, assets)
        with threadpool_limits(limits=2):
            predictions = {name: np.asarray(model.predict(X_valid), dtype=float) for name, model in assets["models"].items()}
        if any(not np.isfinite(values).all() for values in predictions.values()):
            raise ValueError("A saved model returned a non-finite prediction.")
        for local_i, source_i in enumerate(positions):
            row = {key: records[source_i][key] for key in ("input_name", "input_format", "canonical_smiles")}
            row["status"] = "Scored"
            for name, values in predictions.items():
                row[f"predicted_pIC50_{name}"] = float(values[local_i])
            row.update({key: (value.item() if isinstance(value, np.generic) else value) for key, value in ad.iloc[local_i].to_dict().items()})
            row["applicability_status"] = "Inside all three AD methods" if row["inside_all_three_AD"] else "Outside at least one AD method"
            results.append(row)
    return {"results": results, "failures": initial_failures, "n_parsed": len(records), "n_scored": len(results)}
