"""RDKit standardization matching the project's salt/charge parent workflow."""
from __future__ import annotations

from rdkit import Chem
from rdkit.Chem.MolStandardize import rdMolStandardize


def standardize_mol(mol: Chem.Mol, remove_salts: bool = True):
    if mol is None:
        raise ValueError("RDKit could not parse this structure.")
    try:
        params = rdMolStandardize.CleanupParameters()
        mol = rdMolStandardize.Cleanup(mol, params)
        if remove_salts:
            mol = rdMolStandardize.FragmentParent(mol, params)
        mol = rdMolStandardize.Uncharger().uncharge(mol)
        Chem.SanitizeMol(mol)
        smiles = Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        if not smiles:
            raise ValueError("Structure is empty after standardization.")
        return mol, smiles
    except Exception as exc:
        raise ValueError(f"Structure standardization failed: {type(exc).__name__}: {exc}") from exc


def standardize_smiles(smiles: str, remove_salts: bool = True):
    mol = Chem.MolFromSmiles(str(smiles))
    return standardize_mol(mol, remove_salts=remove_salts)
