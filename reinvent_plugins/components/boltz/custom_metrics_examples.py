"""Example custom metric functions for Boltz2 component

This module provides template and example custom metric functions that can be used
with the Boltz2 scoring component. Each function should follow the signature:

    def metric_function(prediction_dir: Path, mol_id: str, smiles: str) -> float

Custom functions have access to all Boltz2 output files:
- Structure files: predictions/{mol_id}/{mol_id}_model_0.cif
- Confidence JSON: predictions/{mol_id}/confidence_{mol_id}.json
- Affinity JSON: predictions/{mol_id}/affinity_{mol_id}.json
"""

from pathlib import Path
from typing import Dict, List
import json
import numpy as np


def binding_pocket_plddt(prediction_dir: Path, mol_id: str, smiles: str) -> float:
    """Calculate average pLDDT for residues near the ligand binding site

    This metric focuses on the confidence of the binding pocket region,
    which is often more relevant than the overall protein confidence.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        smiles: Input SMILES string

    Returns:
        Average pLDDT for binding pocket residues (0-100)
    """
    try:
        # Load confidence data
        confidence_file = prediction_dir / "predictions" / mol_id / f"confidence_{mol_id}.json"

        if not confidence_file.exists():
            return np.nan

        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)

        # Get per-residue pLDDT values
        if "plddt" not in confidence_data or not isinstance(confidence_data["plddt"], list):
            return np.nan

        plddt_values = confidence_data["plddt"]

        # TODO: Implement logic to identify binding pocket residues
        # For now, use a simple approach: focus on residues with lower indices
        # In a real implementation, you would:
        # 1. Parse the structure file to get ligand coordinates
        # 2. Calculate distances from each protein residue to ligand
        # 3. Select residues within a cutoff distance (e.g., 5-8 Å)

        # Simple example: use first 50 residues as proxy for binding site
        pocket_residues = min(50, len(plddt_values))
        pocket_plddt = plddt_values[:pocket_residues]

        return float(np.mean(pocket_plddt))

    except Exception as e:
        print(f"Error calculating binding_pocket_plddt: {e}")
        return np.nan


def combined_affinity_confidence(prediction_dir: Path, mol_id: str, smiles: str) -> float:
    """Combined score: affinity probability weighted by structure confidence

    This metric multiplies the binding affinity probability by the structure
    quality metrics to get a combined confidence score.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        smiles: Input SMILES string

    Returns:
        Combined score (0-1)
    """
    try:
        # Load affinity data
        affinity_file = prediction_dir / "predictions" / mol_id / f"affinity_{mol_id}.json"
        confidence_file = prediction_dir / "predictions" / mol_id / f"confidence_{mol_id}.json"

        if not affinity_file.exists() or not confidence_file.exists():
            return np.nan

        with open(affinity_file, 'r') as f:
            affinity_data = json.load(f)

        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)

        # Extract metrics
        affinity_prob = affinity_data.get("affinity_probability_binary", np.nan)
        ptm = confidence_data.get("ptm", np.nan)
        iptm = confidence_data.get("iptm", np.nan)

        if np.isnan(affinity_prob) or np.isnan(ptm) or np.isnan(iptm):
            return np.nan

        # Combined score: affinity * structure_confidence
        # Weight interface confidence more heavily
        structure_conf = (ptm + 2 * iptm) / 3
        combined = affinity_prob * structure_conf

        return float(combined)

    except Exception as e:
        print(f"Error calculating combined_affinity_confidence: {e}")
        return np.nan


def ligand_efficiency_proxy(prediction_dir: Path, mol_id: str, smiles: str) -> float:
    """Estimate ligand efficiency using affinity prediction and molecular weight

    Ligand efficiency = affinity / molecular_weight
    Higher values indicate better binding per unit mass.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        smiles: Input SMILES string

    Returns:
        Ligand efficiency proxy value
    """
    try:
        # Load affinity data
        affinity_file = prediction_dir / "predictions" / mol_id / f"affinity_{mol_id}.json"

        if not affinity_file.exists():
            return np.nan

        with open(affinity_file, 'r') as f:
            affinity_data = json.load(f)

        affinity_value = affinity_data.get("affinity_pred_value", np.nan)

        if np.isnan(affinity_value):
            return np.nan

        # Calculate molecular weight from SMILES
        from rdkit import Chem
        from rdkit.Chem import Descriptors

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return np.nan

        mol_weight = Descriptors.MolWt(mol)

        # Ligand efficiency: higher is better
        # affinity_pred_value is log10(IC50), so negate it
        ligand_eff = -affinity_value / (mol_weight / 1000.0)  # per kDa

        return float(ligand_eff)

    except Exception as e:
        print(f"Error calculating ligand_efficiency_proxy: {e}")
        return np.nan


def interface_quality_score(prediction_dir: Path, mol_id: str, smiles: str) -> float:
    """Calculate interface quality score from ipTM and pLDDT

    This combines interface confidence (ipTM) with local structure quality
    to assess the quality of the predicted protein-ligand interface.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        smiles: Input SMILES string

    Returns:
        Interface quality score (0-1)
    """
    try:
        # Load confidence data
        confidence_file = prediction_dir / "predictions" / mol_id / f"confidence_{mol_id}.json"

        if not confidence_file.exists():
            return np.nan

        with open(confidence_file, 'r') as f:
            confidence_data = json.load(f)

        # Extract metrics
        iptm = confidence_data.get("iptm", np.nan)
        plddt = confidence_data.get("plddt", np.nan)

        if np.isnan(iptm):
            return np.nan

        # Calculate mean pLDDT if it's a list
        if isinstance(plddt, list):
            mean_plddt = np.mean(plddt) / 100.0  # Normalize to 0-1
        elif isinstance(plddt, (int, float)):
            mean_plddt = float(plddt) / 100.0
        else:
            mean_plddt = 0.7  # Default if not available

        # Weighted combination: interface TM-score and structure quality
        interface_quality = 0.7 * iptm + 0.3 * mean_plddt

        return float(interface_quality)

    except Exception as e:
        print(f"Error calculating interface_quality_score: {e}")
        return np.nan


# Template for creating your own custom metric
def custom_metric_template(prediction_dir: Path, mol_id: str, smiles: str) -> float:
    """Template for creating custom metrics

    Args:
        prediction_dir: Boltz2 output directory (Path object)
        mol_id: Molecule identifier (string, InChiKey or hash)
        smiles: Input SMILES string

    Returns:
        Metric value as float (return np.nan on failure)
    """
    try:
        # Step 1: Construct paths to Boltz2 output files
        prediction_subdir = prediction_dir / "predictions" / mol_id

        # Available files:
        # - Structure: {mol_id}_model_0.cif (or .pdb depending on output_format)
        # - Confidence: confidence_{mol_id}.json
        # - Affinity: affinity_{mol_id}.json

        # Step 2: Load data
        confidence_file = prediction_subdir / f"confidence_{mol_id}.json"

        if not confidence_file.exists():
            return np.nan

        with open(confidence_file, 'r') as f:
            data = json.load(f)

        # Step 3: Extract and compute your metric
        # Example: extract a value and apply transformation
        metric_value = data.get("some_field", 0.0)

        # Step 4: Return as float
        return float(metric_value)

    except Exception as e:
        # Always catch exceptions and return NaN
        print(f"Error in custom_metric_template: {e}")
        return np.nan
