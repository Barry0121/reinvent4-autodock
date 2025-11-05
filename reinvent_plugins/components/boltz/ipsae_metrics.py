"""ipSAE and related interface quality metrics for Boltz2 predictions

This module implements custom metrics for evaluating protein-protein and protein-ligand
interfaces from Boltz2 structure predictions, including:

- ipSAE (interface Predicted Structural Alignment Error): Dunbrack 2025
- pDockQ: Bryant, Pozzati, and Elofsson 2022
- pDockQ2: Zhu, Shenoy, Kundrotas, Elofsson 2023
- LIS (Local Interaction Score): Kim et al. 2024

Original ipSAE script by Roland Dunbrack, Fox Chase Cancer Center
https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1
MIT License

Adapted for REINVENT4 Boltz2 component by integration into custom metrics framework.
Maintains core calculation logic while adapting to Boltz2 output format.

References:
-----------
1. Dunbrack RJ Jr. (2025). ipSAE: Interface Predicted Structural Alignment Error
   for scoring pairwise protein-protein interactions. bioRxiv.
   https://www.biorxiv.org/content/10.1101/2025.02.10.637595v1

2. Bryant P, Pozzati G, Elofsson A. (2022). Improved prediction of protein-protein
   interactions using AlphaFold2. Nature Communications, 13(1):1265.
   https://www.nature.com/articles/s41467-022-28865-w

3. Zhu W, Shenoy A, Kundrotas P, Elofsson A. (2023). Evaluation of AlphaFold-Multimer
   prediction on multi-chain protein complexes. Bioinformatics, 39(7):btad424.
   https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714

4. Kim H, Hu Z, Comjean A, Rodiger J, Mohr SE, Perrimon N. (2024). LIS: a local
   interaction score for assessing AF2-multimer models. bioRxiv.
   https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

Usage in REINVENT4 config:
--------------------------
```toml
# Use ipSAE as primary optimization target
params.primary_metric = "ipsae_d0res"
params.custom_metric_functions = [
    "reinvent_plugins.components.boltz.ipsae_metrics.ipsae_d0res",
    "reinvent_plugins.components.boltz.ipsae_metrics.pdockq2"
]
params.custom_metric_names = ["ipsae_d0res", "pDockQ2"]
params.pae_cutoff = 10.0
params.dist_cutoff = 10.0
```

All metric functions follow the standard signature:
    def metric_name(prediction_dir: Path, mol_id: str, smiles: str) -> float

And have access to Boltz2 output files:
- Structure: predictions/{mol_id}/{mol_id}_model_0.cif
- PAE matrix: predictions/{mol_id}/pae_{mol_id}_model_0.npz
- Confidence: predictions/{mol_id}/confidence_{mol_id}_model_0.json
"""

from __future__ import annotations

from click import option

__all__ = [
    "ipsae_d0res",
    "ipsae_d0chn",
    "ipsae_d0dom",
    "pdockq",
    "pdockq2",
    "lis_score",
    "iptm_from_pae",
]

import math
import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import warnings
import numpy as np
from rdkit import Chem

# Suppress numpy warnings for cleaner output
np.seterr(divide='ignore', invalid='ignore')

# Helper to sanitize numeric outputs: convert None/NaN/Inf to 0.0
def sanitize_metric_value(v: Optional[float]) -> float:
    """Return a safe float: convert None, NaN or infinite to 0.0, else return float(v).

    We explicitly convert to 0.0 to ensure custom metrics cannot increase a score
    by returning NaN/Inf. Returning 0.0 reduces the score (safe fallback).
    """
    try:
        if v is None:
            return 0.0
        val = float(v)
        if not np.isfinite(val):
            return 0.0
        return val
    except Exception:
        return 0.0

# Default cutoffs (can be overridden via environment variables)
DEFAULT_PAE_CUTOFF = 10.0
DEFAULT_DIST_CUTOFF = 10.0


def get_cutoffs() -> Tuple[float, float]:
    """Get PAE and distance cutoffs from environment or use defaults

    Environment variables:
        BOLTZ_IPSAE_PAE_CUTOFF: PAE cutoff in Ångströms (default 10.0)
        BOLTZ_IPSAE_DIST_CUTOFF: Distance cutoff in Ångströms (default 10.0)

    Returns:
        Tuple of (pae_cutoff, dist_cutoff)
    """
    pae_cutoff = float(os.getenv('BOLTZ_IPSAE_PAE_CUTOFF', DEFAULT_PAE_CUTOFF))
    dist_cutoff = float(os.getenv('BOLTZ_IPSAE_DIST_CUTOFF', DEFAULT_DIST_CUTOFF))
    return pae_cutoff, dist_cutoff


# =============================================================================
# Core Helper Functions
# =============================================================================

def extract_heavy_atom(smiles: str) -> Optional[List[int]]:
    """Extract heavy atom indices from SMILES.

    Args:
        smiles (str): SMILES string of the ligand.

    Returns:
        Optional[List[int]]: List of heavy atom indices.
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError("Not a valid SMILES.")
        heavy_indices = [atom.GetIdx() for atom in mol.GetAtoms() if atom.GetAtomicNum() > 1]
        return heavy_indices
    except Exception as e:
        warnings.warn(f"Failed to parse SMILES {smiles}: {e}")
        return None

def ptm_func(x: float, d0: float) -> float:
    """Calculate PTM (Predicted TM-score) function value

    Args:
        x: Distance or error value (Å)
        d0: Normalization factor (Å)

    Returns:
        PTM score contribution (0-1, higher is better)
    """
    return 1.0 / (1 + (x / d0) ** 2.0)


def ptm_func_vec(x: np.ndarray, d0: float) -> np.ndarray:
    """Vectorized PTM function for arrays

    Args:
        x: Array of distance or error values (Å)
        d0: Normalization factor (Å)

    Returns:
        Array of PTM score contributions (0-1)
    """
    return 1.0 / (1 + (x / d0) ** 2.0)


def calc_d0(L: float, pair_type: str = 'protein') -> float:
    """Calculate d0 normalization factor based on sequence length

    Based on Yang and Skolnick, PROTEINS 57:702–710 (2004)

    Args:
        L: Sequence length (number of residues)
        pair_type: 'protein' or 'nucleic_acid' (determines minimum d0)

    Returns:
        d0 value in Ångströms (minimum 1.0 for protein, 2.0 for nucleic acid)
    """
    L = float(L)
    if L < 27:
        L = 27

    min_value = 2.0 if pair_type == 'nucleic_acid' else 1.0
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8

    return max(min_value, d0)


def calc_d0_array(L: np.ndarray, pair_type: str = 'protein') -> np.ndarray:
    """Vectorized d0 calculation for array of lengths

    Args:
        L: Array of sequence lengths
        pair_type: 'protein' or 'nucleic_acid'

    Returns:
        Array of d0 values
    """
    L = np.array(L, dtype=float)
    L = np.maximum(27, L)

    min_value = 2.0 if pair_type == 'nucleic_acid' else 1.0
    d0 = 1.24 * (L - 15) ** (1.0 / 3.0) - 1.8

    return np.maximum(min_value, d0)

def _calculate_protein_ligand_metrics(
    ca_residues: List[Dict],
    cb_residues: List[Dict],
    ligand_atoms: List[Dict],
    chains: np.ndarray,
    pae_matrix: np.ndarray,
    protein_ligand_distances: np.ndarray,
    protein_chains: List[str],
    pae_cutoff: float,
    dist_cutoff: float
) -> Dict[str, float]:
    """Calculate interface metrics for protein-ligand complexes

    Supports multi-chain proteins (e.g., heterodimers) binding to ligands.
    All protein chains are pooled together as the "protein" entity.

    Args:
        ca_residues: List of CA atoms
        cb_residues: List of CB atoms
        ligand_atoms: List of ligand heavy atoms
        chains: Array of chain IDs
        pae_matrix: PAE matrix (protein_residues + ligand_atoms)
        protein_ligand_distances: Distance matrix (n_protein_residues, n_ligand_atoms)
        protein_chains: List of protein chain IDs (can be multiple for multi-chain proteins)
        pae_cutoff: PAE cutoff in Angstroms
        dist_cutoff: Distance cutoff in Angstroms

    Returns:
        Dictionary with calculated metrics
    """
    if len(ligand_atoms) == 0 or len(protein_chains) == 0:
        return {}

    numres = len(ca_residues)
    num_ligand_atoms = len(ligand_atoms)

    # Protein residue mask - ALL protein chains together
    protein_mask = np.isin(chains, protein_chains)
    n_protein = np.sum(protein_mask)

    # Ligand atoms are at indices [numres:numres+num_ligand_atoms] in PAE matrix
    ligand_start_idx = numres
    ligand_end_idx = numres + num_ligand_atoms

    # Extract protein-ligand PAE submatrix
    if pae_matrix.shape[0] < ligand_end_idx:
        warnings.warn(f"PAE matrix too small for ligand atoms: {pae_matrix.shape[0]} < {ligand_end_idx}")
        return {}

    # Get the protein residue indices (to index into PAE matrix)
    protein_indices = np.where(protein_mask)[0]

    # PAE from protein residues to ligand atoms
    # Shape: (n_protein_residues, n_ligand_atoms)
    protein_ligand_pae = pae_matrix[protein_indices, ligand_start_idx:ligand_end_idx]

    # Find interface residues (protein residues with good PAE to any ligand atom)
    interface_protein_residues = set()
    for i, res_idx in enumerate(protein_indices):
        if np.any(protein_ligand_pae[i] < pae_cutoff):
            interface_protein_residues.add(res_idx)

    n0_interface = len(interface_protein_residues)

    if n0_interface == 0:
        # No good interface
        return {
            'ipsae_d0res': 0.0,
            'ipsae_d0chn': 0.0,
            'ipsae_d0dom': 0.0,
            'pdockq': 0.0,
            'pdockq2': 0.0,
            'lis_score': 0.0,
            'iptm_from_pae': 0.0
        }

    # Calculate d0 values
    d0chn = calc_d0(float(n_protein + num_ligand_atoms), 'protein')
    d0dom = calc_d0(float(n0_interface + num_ligand_atoms), 'protein')

    # Calculate ipSAE variants
    ipsae_d0chn_values = []
    ipsae_d0dom_values = []
    ipsae_d0res_values = []

    for i, res_idx in enumerate(protein_indices):
        pae_to_ligand = protein_ligand_pae[i]
        valid_ligand_atoms = pae_to_ligand < pae_cutoff
        n_valid = np.sum(valid_ligand_atoms)

        if n_valid > 0:
            # ipSAE_d0chn
            ptm_values_chn = ptm_func_vec(pae_to_ligand[valid_ligand_atoms], d0chn)
            ipsae_d0chn_values.append(np.mean(ptm_values_chn))

            # ipSAE_d0dom
            ptm_values_dom = ptm_func_vec(pae_to_ligand[valid_ligand_atoms], d0dom)
            ipsae_d0dom_values.append(np.mean(ptm_values_dom))

            # ipSAE_d0res (per-residue d0)
            d0res = calc_d0(n_valid, 'protein')
            ptm_values_res = ptm_func_vec(pae_to_ligand[valid_ligand_atoms], d0res)
            ipsae_d0res_values.append(np.mean(ptm_values_res))

    ipsae_d0res = float(np.max(ipsae_d0res_values)) if ipsae_d0res_values else 0.0
    ipsae_d0chn = float(np.max(ipsae_d0chn_values)) if ipsae_d0chn_values else 0.0
    ipsae_d0dom = float(np.max(ipsae_d0dom_values)) if ipsae_d0dom_values else 0.0

    # Calculate LIS for protein-ligand
    valid_pae = protein_ligand_pae[protein_ligand_pae <= 12]
    if valid_pae.size > 0:
        scores = (12 - valid_pae) / 12
        lis_score = float(np.mean(scores))
    else:
        lis_score = 0.0

    # Calculate ipTM-like score for protein-ligand
    ptm_matrix_chn = ptm_func_vec(protein_ligand_pae, d0chn)
    iptm_values = []
    for i in range(len(protein_ligand_pae)):
        iptm_values.append(np.mean(ptm_matrix_chn[i]))
    iptm_from_pae = float(np.mean(iptm_values)) if iptm_values else 0.0

    # pDockQ and pDockQ2 calculations for protein-ligand
    # Note: These were originally designed for protein-protein interfaces
    # Adapting them for protein-ligand (may not be as well validated)

    if protein_ligand_distances is not None:
        # Filter protein CB atoms that correspond to our protein_mask
        protein_cb_mask = protein_mask[:len(cb_residues)]  # In case cb_residues is shorter

        # Find interface residues by distance (within dist_cutoff)
        interface_by_distance = set()
        npairs_distance = 0

        for i, res_idx in enumerate(protein_indices):
            if res_idx >= len(protein_ligand_distances):
                continue
            close_ligand_atoms = protein_ligand_distances[res_idx] <= dist_cutoff
            if np.any(close_ligand_atoms):
                interface_by_distance.add(res_idx)
                npairs_distance += np.sum(close_ligand_atoms)

        # Simplified pDockQ-like score
        # Original pDockQ uses pLDDT * log10(npairs), but we don't have pLDDT easily
        # We'll use a simplified version with just the PAE-based score
        if npairs_distance > 0:
            # Use mean PAE quality as a proxy
            mean_pae_score = np.mean(ptm_matrix_chn)
            pdockq = float(mean_pae_score)  # Simplified
            pdockq2 = float(mean_pae_score)  # Simplified
        else:
            pdockq = 0.0
            pdockq2 = 0.0
    else:
        pdockq = 0.0
        pdockq2 = 0.0

    return {
        'ipsae_d0res': ipsae_d0res,
        'ipsae_d0chn': ipsae_d0chn,
        'ipsae_d0dom': ipsae_d0dom,
        'pdockq': pdockq,
        'pdockq2': pdockq2,
        'lis_score': lis_score,
        'iptm_from_pae': iptm_from_pae
    }

def _calculate_protein_protein_metrics(
        ca_residues: List[Dict],
        cb_residues: List[Dict],
        chains: np.ndarray,
        pae_matrix: np.ndarray,
        distances: np.ndarray,
        chain1: str,
        chain2: str,
        pae_cutoff: float,
        dist_cutoff: float
    ) -> Dict[str, float]:
    """Calculate interface metrics for protein-protein complexes

    Args:
        ca_residues: List of CA atoms
        cb_residues: List of CB atoms
        chains: Array of chain IDs
        pae_matrix: PAE matrix
        distances: CB-CB distance matrix
        chain1: First protein chain ID
        chain2: Second protein chain ID
        pae_cutoff: PAE cutoff in Angstroms
        dist_cutoff: Distance cutoff in Angstroms

    Returns:
        Dictionary with calculated metrics
    """
    numres = len(ca_residues)

    # Get residue masks for each chain
    chain1_mask = (chains == chain1)
    chain2_mask = (chains == chain2)

    # Calculate chain pair properties
    n_chain1 = np.sum(chain1_mask)
    n_chain2 = np.sum(chain2_mask)
    n0chn = n_chain1 + n_chain2

    pair_type = 'protein'  # Both chains are protein
    d0chn = calc_d0(n0chn, pair_type)

    # Calculate interface PAE matrix (chain1 -> chain2)
    pae_valid_mask = (pae_matrix < pae_cutoff)

    # Find residues with good interface PAE values
    interface_residues_chain1 = set()
    interface_residues_chain2 = set()

    for i in range(numres):
        if not chain1_mask[i]:
            continue
        for j in range(numres):
            if not chain2_mask[j]:
                continue
            if pae_matrix[i, j] < pae_cutoff:
                interface_residues_chain1.add(i)
                interface_residues_chain2.add(j)

    n0dom = len(interface_residues_chain1) + len(interface_residues_chain2)

    if n0dom == 0:
        # No good interface, return zeros
        return {
            'ipsae_d0res': 0.0,
            'ipsae_d0chn': 0.0,
            'ipsae_d0dom': 0.0,
            'pdockq': 0.0,
            'pdockq2': 0.0,
            'lis_score': 0.0,
            'iptm_from_pae': 0.0
        }

    d0dom = calc_d0(n0dom, pair_type)

    # Calculate ipSAE variants
    # ipSAE_d0chn: Use d0 based on total chain pair length
    ptm_matrix_d0chn = ptm_func_vec(pae_matrix, d0chn)
    ipsae_d0chn_values = []

    for i in range(numres):
        if not chain1_mask[i]:
            continue
        valid_pairs = chain2_mask & (pae_matrix[i] < pae_cutoff)
        if valid_pairs.any():
            ipsae_d0chn_values.append(ptm_matrix_d0chn[i, valid_pairs].mean())

    ipsae_d0chn = float(np.max(ipsae_d0chn_values)) if ipsae_d0chn_values else 0.0

    # ipSAE_d0dom: Use d0 based on interface domain size
    ptm_matrix_d0dom = ptm_func_vec(pae_matrix, d0dom)
    ipsae_d0dom_values = []

    for i in range(numres):
        if not chain1_mask[i]:
            continue
        valid_pairs = chain2_mask & (pae_matrix[i] < pae_cutoff)
        if valid_pairs.any():
            ipsae_d0dom_values.append(ptm_matrix_d0dom[i, valid_pairs].mean())

    ipsae_d0dom = float(np.max(ipsae_d0dom_values)) if ipsae_d0dom_values else 0.0

    # ipSAE_d0res: Use per-residue d0 (RECOMMENDED)
    ipsae_d0res_values = []

    for i in range(numres):
        if not chain1_mask[i]:
            continue
        valid_pairs = chain2_mask & (pae_matrix[i] < pae_cutoff)
        n0res_i = np.sum(valid_pairs)
        if n0res_i > 0:
            d0res_i = calc_d0(n0res_i, pair_type)
            ptm_row = ptm_func_vec(pae_matrix[i], d0res_i)
            ipsae_d0res_values.append(ptm_row[valid_pairs].mean())

    ipsae_d0res = float(np.max(ipsae_d0res_values)) if ipsae_d0res_values else 0.0

    # Calculate pDockQ (Bryant et al. 2022)
    pdockq_cutoff = 8.0
    interface_residues_pdockq = set()

    for i in range(numres):
        if not chain1_mask[i]:
            continue
        close_contacts = chain2_mask & (distances[i] <= pdockq_cutoff)
        if close_contacts.any():
            interface_residues_pdockq.add(i)
            for j in np.where(close_contacts)[0]:
                interface_residues_pdockq.add(j)

    npairs_pdockq = 0
    for i in range(numres):
        if not chain1_mask[i]:
            continue
        npairs_pdockq += np.sum(chain2_mask & (distances[i] <= pdockq_cutoff))

    if npairs_pdockq > 0 and len(interface_residues_pdockq) > 0:
        # Use uniform pLDDT estimate (we don't have per-atom pLDDT readily available)
        # Alternative: extract from confidence JSON if available
        mean_plddt = 70.0  # Conservative estimate
        x = mean_plddt * math.log10(npairs_pdockq)
        pdockq = 0.724 / (1 + math.exp(-0.052 * (x - 152.611))) + 0.018
    else:
        pdockq = 0.0

    # Calculate pDockQ2 (Zhu et al. 2023)
    sum_ptm = 0.0
    npairs_pdockq2 = 0

    for i in range(numres):
        if not chain1_mask[i]:
            continue
        valid_pairs = chain2_mask & (distances[i] <= pdockq_cutoff)
        if valid_pairs.any():
            pae_list = pae_matrix[i][valid_pairs]
            pae_list_ptm = ptm_func_vec(pae_list, 10.0)
            sum_ptm += pae_list_ptm.sum()
            npairs_pdockq2 += np.sum(valid_pairs)

    if npairs_pdockq2 > 0 and len(interface_residues_pdockq) > 0:
        mean_ptm = sum_ptm / npairs_pdockq2
        x = mean_plddt * mean_ptm
        pdockq2 = 1.31 / (1 + math.exp(-0.075 * (x - 84.733))) + 0.005
    else:
        pdockq2 = 0.0

    # Calculate LIS (Kim et al. 2024)
    interface_pae = pae_matrix[np.ix_(chain1_mask, chain2_mask)]
    valid_pae = interface_pae[interface_pae <= 12]

    if valid_pae.size > 0:
        scores = (12 - valid_pae) / 12
        lis_score = float(np.mean(scores))
    else:
        lis_score = 0.0

    # Calculate ipTM from PAE (for comparison with Boltz's built-in ipTM)
    iptm_values = []
    for i in range(numres):
        if not chain1_mask[i]:
            continue
        ptm_row = ptm_matrix_d0chn[i, chain2_mask]
        if ptm_row.size > 0:
            iptm_values.append(ptm_row.mean())

    iptm_from_pae = float(np.mean(iptm_values)) if iptm_values else 0.0

    # Return all metrics
    return {
        'ipsae_d0res': ipsae_d0res,
        'ipsae_d0chn': ipsae_d0chn,
        'ipsae_d0dom': ipsae_d0dom,
        'pdockq': pdockq,
        'pdockq2': pdockq2,
        'lis_score': lis_score,
        'iptm_from_pae': iptm_from_pae
    }

def parse_cif_atom_line(line: str, field_dict: Dict[str, int]) -> Optional[Dict[str, Any]]:
    """Parse a single ATOM/HETATM line from mmCIF file

    Args:
        line: Line from CIF file starting with ATOM or HETATM
        field_dict: Mapping of field names to column indices

    Returns:
        Dictionary with atom information, or None for ligands without residue numbers
    """
    try:
        parts = line.split()

        atom_num = int(parts[field_dict['id']])
        atom_name = parts[field_dict['label_atom_id']]
        residue_name = parts[field_dict['label_comp_id']]
        chain_id = parts[field_dict['label_asym_id']]
        residue_seq_num = parts[field_dict['label_seq_id']]
        x = float(parts[field_dict['Cartn_x']])
        y = float(parts[field_dict['Cartn_y']])
        z = float(parts[field_dict['Cartn_z']])

        # Skip ligand atoms (no residue number)
        if residue_seq_num == '.':
            # use atom number as identifier
            return {
                'atom_num': atom_num,
                'atom_name': atom_name,
                'residue_name': residue_name,
                'chain_id': chain_id,
                'residue_seq_num': -atom_num,
                'is_ligand': True,
                'x': x,
                'y': y,
                'z': z,
                'residue': f"{residue_name:3}   {chain_id:3} {residue_seq_num:4}"
            }
        residue_seq_num = int(residue_seq_num)

        return {
            'atom_num': atom_num,
            'atom_name': atom_name,
            'residue_name': residue_name,
            'chain_id': chain_id,
            'residue_seq_num': residue_seq_num,
            'is_ligand': False,
            'x': x,
            'y': y,
            'z': z,
            'residue': f"{residue_name:3}   {chain_id:3} {residue_seq_num:4}"
        }
    except (ValueError, IndexError, KeyError):
        return None


def load_structure_from_cif(
        cif_path: Path,
        ligand_chain_id: Optional[str] = None  # Changed from ligand_heavy_atom_indices
    ) -> Tuple[List[Dict], List[Dict], List[Dict], np.ndarray]:
    """Load structure data from Boltz2 mmCIF file

    Args:
        cif_path: Path to mmCIF structure file
        ligand_chain_id: Chain ID of the ligand (e.g., 'C'). If provided,
                        only heavy atoms from this chain will be included.

    Returns:
        Tuple of (CA_residues, CB_residues, ligand_atoms, chains_array)
    """
    ca_residues = []
    cb_residues = []
    ligand_atoms = []
    chains = []

    field_dict = {}
    field_num = 0

    with open(cif_path, 'r') as f:
        for line in f:
            # Parse field definitions
            if line.startswith("_atom_site."):
                field_name = line.strip().split('.')[1]
                field_dict[field_name] = field_num
                field_num += 1
                continue

            # Parse atom lines
            if not (line.startswith("ATOM") or line.startswith("HETATM")):
                continue

            atom = parse_cif_atom_line(line, field_dict)
            if atom is None:
                continue

            # Handle ligand atoms - filter by chain ID
            if atom.get('is_ligand', False):
                # If ligand_chain_id specified, only include atoms from that chain
                if ligand_chain_id is None or atom['chain_id'] == ligand_chain_id:
                    # Only keep heavy atoms (non-hydrogen)
                    if not atom['atom_name'].startswith('H'):
                        ligand_atoms.append({
                            'atom_num': atom['atom_num'] - 1,  # 0-indexed
                            'coor': np.array([atom['x'], atom['y'], atom['z']]),
                            'res': atom['residue_name'],
                            'chainid': atom['chain_id'],
                            'resnum': atom['residue_seq_num'],
                            'residue': atom['residue']
                        })

            # Collect CA atoms
            if atom['atom_name'] == 'CA' or 'C1' in atom['atom_name']:
                ca_residues.append({
                    'atom_num': atom['atom_num'] - 1,
                    'coor': np.array([atom['x'], atom['y'], atom['z']]),
                    'res': atom['residue_name'],
                    'chainid': atom['chain_id'],
                    'resnum': atom['residue_seq_num'],
                    'residue': atom['residue']
                })
                chains.append(atom['chain_id'])

            # Collect CB atoms
            is_cb = atom['atom_name'] == 'CB' or 'C3' in atom['atom_name']
            is_gly_ca = atom['residue_name'] == 'GLY' and atom['atom_name'] == 'CA'

            if is_cb or is_gly_ca:
                cb_residues.append({
                    'atom_num': atom['atom_num'] - 1,
                    'coor': np.array([atom['x'], atom['y'], atom['z']]),
                    'res': atom['residue_name'],
                    'chainid': atom['chain_id'],
                    'resnum': atom['residue_seq_num'],
                    'residue': atom['residue']
                })

    return ca_residues, cb_residues, ligand_atoms, np.array(chains)


def classify_chains(chains: np.ndarray, residue_types: np.ndarray) -> Dict[str, str]:
    """Classify each chain as protein or nucleic acid

    Args:
        chains: Array of chain IDs
        residue_types: Array of residue type names

    Returns:
        Dictionary mapping chain ID to 'protein' or 'nucleic_acid'
    """
    nuc_residues = {"DA", "DC", "DT", "DG", "A", "C", "U", "G"}
    chain_types = {}

    for chain in np.unique(chains):
        indices = np.where(chains == chain)[0]
        chain_residues = residue_types[indices]
        nuc_count = sum(1 for res in chain_residues if res in nuc_residues)
        chain_types[chain] = 'nucleic_acid' if nuc_count > 0 else 'protein'

    return chain_types


def calculate_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """Calculate pairwise distance matrix between coordinates

    Args:
        coordinates: Nx3 array of (x, y, z) coordinates

    Returns:
        NxN symmetric distance matrix in Ångströms
    """
    diff = coordinates[:, np.newaxis, :] - coordinates[np.newaxis, :, :]
    distances = np.sqrt((diff ** 2).sum(axis=2))
    return distances


def load_pae_matrix(pae_path: Path) -> Optional[np.ndarray]:
    """Load PAE matrix from Boltz2 NPZ file

    Args:
        pae_path: Path to pae_{mol_id}_model_0.npz file

    Returns:
        NxN PAE matrix in Ångströms, or None if file not found
    """
    try:
        pae_data = np.load(pae_path, allow_pickle=False)
    except Exception:
        return None

    # Preferred key
    try:
        if 'pae' in pae_data.files:
            arr = np.array(pae_data['pae'], dtype=float)
            return arr
    except Exception:
        pass

    # Fallback: try any 2D square array in the NPZ
    for key in pae_data.files:
        try:
            arr = np.array(pae_data[key], dtype=float)
            if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
                return arr
        except Exception:
            continue

    return None


def load_plddt_from_confidence(confidence_path: Path, ca_atom_indices: np.ndarray) -> Optional[np.ndarray]:
    """Load per-residue pLDDT values from confidence JSON

    Boltz2 outputs per-atom pLDDT values. We extract only CA atoms.

    Args:
        confidence_path: Path to confidence JSON file
        ca_atom_indices: Indices of CA atoms to extract

    Returns:
        Array of pLDDT values (0-100) for each residue
    """
    try:
        import json
        with open(confidence_path, 'r') as f:
            data = json.load(f)

        # Boltz2 doesn't output per-atom pLDDT in the JSON
        # We'll compute from complex_plddt or return uniform values
        # For now, return None to indicate we need alternative approach
        return None

    except (FileNotFoundError, KeyError, json.JSONDecodeError):
        return None


# =============================================================================
# ipSAE Metric Functions
# =============================================================================

def _calculate_ipsae_metrics(
        prediction_dir: Path,
        mol_id: str,
        ligand_chain_id: Optional[str] = None,  # NEW: specify ligand chain
        pae_cutoff: Optional[float] = None,
        dist_cutoff: Optional[float] = None,
        verbose: Optional[bool] = False,
    ) -> Dict[str, float]:
    """Core ipSAE calculation for all variants

    This is the main calculation function that computes all ipSAE variants,
    pDockQ, pDockQ2, and LIS scores from Boltz2 output.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        ligand_chain: Ligand chain IDs
        pae_cutoff: PAE cutoff in Ångströms (None = use from env/defaults)
        dist_cutoff: Distance cutoff in Ångströms (None = use from env/defaults)

    Returns:
        Dictionary with all computed metrics, or empty dict on failure
    """
    try:
        # Get cutoffs from environment or defaults if not provided
        if pae_cutoff is None or dist_cutoff is None:
            env_pae, env_dist = get_cutoffs()
            if pae_cutoff is None:
                pae_cutoff = env_pae
            if dist_cutoff is None:
                dist_cutoff = env_dist
        # Construct paths
        # Note: prediction_dir from Boltz2 component is already at boltz_pose level
        # Files are in prediction_dir/{mol_id}/ directly (no "predictions" subdirectory)
        pred_subdir = prediction_dir / mol_id
        structure_file = pred_subdir / f"{mol_id}_model_0.cif"
        pae_file = pred_subdir / f"pae_{mol_id}_model_0.npz"
        confidence_file = pred_subdir / f"confidence_{mol_id}_model_0.json"

        # Check required files exist
        if not structure_file.exists() or not pae_file.exists():
            return {}

        # Load structure
        if ligand_chain_id:
            ca_residues, cb_residues, ligand_heavy_atoms, chains = load_structure_from_cif(
                structure_file,
                ligand_chain_id=ligand_chain_id
            )
        else:
            ca_residues, cb_residues, _, chains = load_structure_from_cif(structure_file)

        if len(ca_residues) == 0 or len(cb_residues) == 0:
            return {}

        numres = len(ca_residues)
        unique_chains = np.unique(chains)

        # If only one chain (ligand only), cannot compute interface metrics
        if len(unique_chains) < 2:
            return {}

        # Load PAE matrix
        pae_matrix = load_pae_matrix(pae_file)
        if pae_matrix is None:
            return {}

        # update the PASE matrix size with true protein-ligand atoms
        expected_pae_size = numres + len(ligand_heavy_atoms)
        if pae_matrix.shape[0] != expected_pae_size and pae_matrix.shape[0] != numres:
            warnings.warn(f"PAE matrix size mismatch: {pae_matrix.shape[0]} vs expected {numres} (protein-protein) / {expected_pae_size} (protein-ligand)")
            return {}

        # Get coordinates and calculate distances
        ca_coordinates = np.array([res['coor'] for res in ca_residues])
        cb_coordinates = np.array([res['coor'] for res in cb_residues])
        distances = calculate_distance_matrix(cb_coordinates)

        # Get coordinate from the ligand atoms for interface distance calculation later
        ligand_coordinate = None
        if len(ligand_heavy_atoms) > 0:
            ligand_coordinate = np.array([res['coor'] for res in ligand_heavy_atoms])

        # Classify chains
        residue_types = np.array([res['res'] for res in ca_residues])
        chain_types = classify_chains(chains, residue_types)

        # Initialize result storage
        results = {}

        # For each chain pair, calculate metrics
        # We'll focus on the first two protein chains (ignore ligand chain)
        protein_chains = [ch for ch in unique_chains if chain_types[ch] == 'protein']

        # Determine what interface to calculate
        has_ligand = len(ligand_heavy_atoms) > 0
        has_multiple_protein_chains = len(protein_chains) >= 2

        if has_ligand and (ligand_coordinate is not None):
            if verbose:
                print(f"Calculating interface scores between a pair or more protein chains with a ligand chain.")

            # Only one protein chain - compute protein-ligand metrics if applicable
            if len(ligand_heavy_atoms) == 0:
                return {}

            # Calculate protein-ligand distances
            diff = cb_coordinates[:,np.newaxis,:] - ligand_coordinate[np.newaxis,:,:]
            protein_ligand_distances = np.sqrt((diff ** 2).sum(axis=2))

            return _calculate_protein_ligand_metrics(
                ca_residues, cb_residues, ligand_heavy_atoms, chains,
                pae_matrix, protein_ligand_distances, protein_chains,  # ← FIXED: pass the whole list
                pae_cutoff, dist_cutoff
            )

        elif has_multiple_protein_chains:
            if verbose:
                print(f"Calculating interface scores for a pair or more protein chains.")
            if len(protein_chains) == 2:
                # Simple case: one interface
                if verbose:
                    print(f"Calculating interface scores between protein chains {protein_chains[0]} and {protein_chains[1]}.")

                chain1, chain2 = protein_chains[0], protein_chains[1]
                return _calculate_protein_protein_metrics(
                    ca_residues, cb_residues, chains,
                    pae_matrix, distances, chain1, chain2,
                    pae_cutoff, dist_cutoff
                )

            else:
                # Multiple chains (3+): calculate all pairwise interfaces
                if verbose:
                    print(f"Calculating interface scores for {len(protein_chains)} protein chains (all pairwise combinations).")

                all_metrics = []
                interface_pairs = []

                # Calculate all pairwise interfaces
                for i in range(len(protein_chains)):
                    for j in range(i + 1, len(protein_chains)):
                        chain1, chain2 = protein_chains[i], protein_chains[j]

                        metrics = _calculate_protein_protein_metrics(
                            ca_residues, cb_residues, chains,
                            pae_matrix, distances, chain1, chain2,
                            pae_cutoff, dist_cutoff
                        )

                        # Only keep interfaces with some signal (ipsae_d0res > 0)
                        if metrics and metrics.get('ipsae_d0res', 0.0) > 0:
                            all_metrics.append(metrics)
                            interface_pairs.append((chain1, chain2))

                if not all_metrics:
                    # No valid interfaces found
                    if verbose:
                        print(f"Warning: No valid protein-protein interfaces found among {len(protein_chains)} chains.")
                    return {
                        'ipsae_d0res': 0.0,
                        'ipsae_d0chn': 0.0,
                        'ipsae_d0dom': 0.0,
                        'pdockq': 0.0,
                        'pdockq2': 0.0,
                        'lis_score': 0.0,
                        'iptm_from_pae': 0.0
                    }

                # Aggregate across all interfaces using MAX (best interface)
                if verbose:
                    print(f"Found {len(all_metrics)} valid interfaces. Aggregating using MAX.")

                best_metrics = {
                    'ipsae_d0res': max(m['ipsae_d0res'] for m in all_metrics),
                    'ipsae_d0chn': max(m['ipsae_d0chn'] for m in all_metrics),
                    'ipsae_d0dom': max(m['ipsae_d0dom'] for m in all_metrics),
                    'pdockq': max(m['pdockq'] for m in all_metrics),
                    'pdockq2': max(m['pdockq2'] for m in all_metrics),
                    'lis_score': max(m['lis_score'] for m in all_metrics),
                    'iptm_from_pae': max(m['iptm_from_pae'] for m in all_metrics)
                }

                # Print which interface was best for primary metric
                best_idx = np.argmax([m['ipsae_d0res'] for m in all_metrics])
                best_pair = interface_pairs[best_idx]
                print(f"Best interface: {best_pair[0]}-{best_pair[1]} (ipSAE_d0res={best_metrics['ipsae_d0res']:.3f})")

                return best_metrics

        else:
            # Only one protein chain without ligand, nothing to calculation
            raise ValueError(f"There's only one protein chain without any ligand, nothing will be calculated.")

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        warnings.warn(f"Error calculating ipSAE metrics for {mol_id}: {e}\n{traceback.format_exc()}")
        # Return zeroed metrics so downstream code receives numeric values
        return {
            'ipsae_d0res': 0.0,
            'ipsae_d0chn': 0.0,
            'ipsae_d0dom': 0.0,
            'pdockq': 0.0,
            'pdockq2': 0.0,
            'lis_score': 0.0,
            'iptm_from_pae': 0.0
        }


# =============================================================================
# Public Metric Functions
# =============================================================================

def ipsae_d0res(prediction_dir: Path, mol_id: str, ligand_chain_id: str, verbose: bool = False) -> float:
    """Calculate ipSAE with per-residue d0 normalization (RECOMMENDED)

    This is the primary ipSAE metric recommended by Dunbrack 2025.
    Uses d0 calculated from the number of residues in chain2 that have
    good PAE values (< cutoff) for each residue in chain1.

    Most sensitive to interface quality and recommended for optimization.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        ligand_chain_id: Input ligand_chain_id (not used but required by interface)

    Returns:
        ipSAE score (0-1, higher = better interface quality)
    """
    metrics = _calculate_ipsae_metrics(prediction_dir, mol_id, ligand_chain_id, verbose)
    return sanitize_metric_value(metrics.get('ipsae_d0res', 0.0))


def ipsae_d0chn(prediction_dir: Path, mol_id: str, ligand_chain_id: str, verbose: bool = False) -> float:
    """Calculate ipSAE with chain-pair d0 normalization

    Uses d0 based on total chain pair length (len(chain1) + len(chain2)).
    More stable across different complex sizes than per-residue d0.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        ligand_chain_id: Input ligand_chain_id (not used)

    Returns:
        ipSAE score (0-1, higher = better)
    """
    metrics = _calculate_ipsae_metrics(prediction_dir, mol_id, ligand_chain_id, verbose)
    return sanitize_metric_value(metrics.get('ipsae_d0chn', 0.0))


def ipsae_d0dom(prediction_dir: Path, mol_id: str, ligand_chain_id: str, verbose: bool = False) -> float:
    """Calculate ipSAE with domain-based d0 normalization

    Uses d0 based on the number of residues in the interface domain
    (residues with PAE < cutoff). Focuses on well-defined interface regions.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        ligand_chain_id: Input ligand_chain_id (not used)

    Returns:
        ipSAE score (0-1, higher = better)
    """
    metrics = _calculate_ipsae_metrics(prediction_dir, mol_id, ligand_chain_id, verbose)
    return sanitize_metric_value(metrics.get('ipsae_d0dom', 0.0))


def pdockq(prediction_dir: Path, mol_id: str, ligand_chain_id: str, verbose: bool = False) -> float:
    """Calculate pDockQ score (Bryant et al. 2022)

    Predicts DockQ score from pLDDT and number of interface contacts.
    Original method for AlphaFold2 protein complex quality assessment.

    Reference:
        Bryant P, Pozzati G, Elofsson A. Nature Communications 2022.
        https://www.nature.com/articles/s41467-022-28865-w

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        ligand_chain_id: Input ligand_chain_id (not used)

    Returns:
        pDockQ score (0-1, higher = better)
    """
    metrics = _calculate_ipsae_metrics(prediction_dir, mol_id, ligand_chain_id, verbose)
    return sanitize_metric_value(metrics.get('pdockq', 0.0))


def pdockq2(prediction_dir: Path, mol_id: str, ligand_chain_id: str, verbose: bool = False) -> float:
    """Calculate pDockQ2 score (Zhu et al. 2023)

    Improved version incorporating PAE information alongside pLDDT.
    Better performance than original pDockQ for AF2-Multimer predictions.

    Reference:
        Zhu W, Shenoy A, Kundrotas P, Elofsson A. Bioinformatics 2023.
        https://academic.oup.com/bioinformatics/article/39/7/btad424/7219714

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        ligand_chain_id: Input ligand_chain_id (not used)

    Returns:
        pDockQ2 score (0-1, higher = better)
    """
    metrics = _calculate_ipsae_metrics(prediction_dir, mol_id, ligand_chain_id, verbose)
    return sanitize_metric_value(metrics.get('pdockq2', 0.0))


def lis_score(prediction_dir: Path, mol_id: str, ligand_chain_id: str, verbose: bool = False) -> float:
    """Calculate LIS (Local Interaction Score) (Kim et al. 2024)

    Evaluates local interface quality from PAE values.
    Simpler metric focusing on average PAE quality across interface.

    Reference:
        Kim H et al. bioRxiv 2024.
        https://www.biorxiv.org/content/10.1101/2024.02.19.580970v1

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        ligand_chain_id: Input chain (not used)

    Returns:
        LIS score (0-1, higher = better)
    """
    metrics = _calculate_ipsae_metrics(prediction_dir, mol_id, ligand_chain_id, verbose)
    return sanitize_metric_value(metrics.get('lis_score', 0.0))


def iptm_from_pae(prediction_dir: Path, mol_id: str, ligand_chain_id: str, verbose: bool = False) -> float:
    """Calculate ipTM from PAE matrix for comparison

    Calculates interface predicted TM-score from the PAE matrix.
    Useful for comparing with Boltz2's built-in ipTM metric to
    validate PAE matrix loading and calculations.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        ligand chain id: Input ligand chain id

    Returns:
        ipTM score (0-1, higher = better)
    """
    metrics = _calculate_ipsae_metrics(prediction_dir, mol_id, ligand_chain_id, verbose)
    return sanitize_metric_value(metrics.get('iptm_from_pae', 0.0))

def get_all_score(prediction_dir: Path, mol_id: str, ligand_chain_id: str, verbose: bool = False) -> Optional[Dict]:
    """Calculate ipSAE with per-residue d0 normalization (RECOMMENDED)

    This is the primary ipSAE metric recommended by Dunbrack 2025.
    Uses d0 calculated from the number of residues in chain2 that have
    good PAE values (< cutoff) for each residue in chain1.

    Most sensitive to interface quality and recommended for optimization.

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        ligand_chain_id: Input ligand chain id

    Returns:
        ipSAE score (0-1, higher = better interface quality)
    """
    metrics = _calculate_ipsae_metrics(prediction_dir, mol_id, ligand_chain_id, verbose)
    if not metrics:
        return None
    # Sanitize all numeric values
    sanitized = {k: sanitize_metric_value(v) for k, v in metrics.items()}
    return sanitized