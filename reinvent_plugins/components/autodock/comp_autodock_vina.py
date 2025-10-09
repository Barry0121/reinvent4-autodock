"""AutoDock Vina docking scoring component for REINVENT 4

This component performs molecular docking using AutoDock Vina to estimate
binding affinity for generated molecules against a target receptor.
"""

from __future__ import annotations

__all__ = ["AutoDockVina"]

import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pydantic import Field
from pydantic.dataclasses import dataclass
from vina import Vina

from ..component_results import ComponentResults
from ..add_tag import add_tag

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for AutoDock Vina scoring component

    Note that all parameters are lists because components can have
    multiple endpoints, even if only one is used.
    """

    receptor_pdbqt_path: List[str] = Field(
        default_factory=lambda: ["/mnt/data/gpu-server/NR4A1-small-molecule/nr4a1_reinvent/data_cache/6kz5_receptor.pdbqt"]
    )
    output_dir: List[str] = Field(default_factory=lambda: ["./output/docking_poses"])
    run_id: List[str] = Field(default_factory=lambda: ["default_run"])
    center_x: List[float] = Field(default_factory=lambda: [1.910])
    center_y: List[float] = Field(default_factory=lambda: [3.831])
    center_z: List[float] = Field(default_factory=lambda: [2.31])
    size_x: List[float] = Field(default_factory=lambda: [20.0])
    size_y: List[float] = Field(default_factory=lambda: [20.0])
    size_z: List[float] = Field(default_factory=lambda: [20.0])
    exhaustiveness: List[int] = Field(default_factory=lambda: [8])
    num_modes: List[int] = Field(default_factory=lambda: [9])
    energy_range: List[float] = Field(default_factory=lambda: [3.0])
    enable_pocket_detection: List[bool] = Field(default_factory=lambda: [False])


"""
NOTE: output directory structure
outputs/
└── docking_poses/
    └── {run_id}/
        └── vina_cpu/
            ├── {smiles_hash}_ligand.pdbqt
            ├── {smiles_hash}_docked.pdbqt
            └── {smiles_hash}_minimized.pdbqt
        └── other_scoring_artifacts/
        └── {run_id}_summary.csv
"""

@add_tag("__component")
class AutoDockVina:
    """AutoDock Vina docking scoring component

    This component:
    1. Converts SMILES to 3D SDF format
    2. Converts SDF to PDBQT format for Vina
    3. Runs AutoDock Vina docking
    4. Extracts binding affinity from log output

    Returns binding affinity in kcal/mol (more negative = better binding)
    """

    def __init__(self, params: Parameters):
        self.receptor_path = params.receptor_pdbqt_path[0]
        self.output_dir = params.output_dir[0]
        self.run_id = params.run_id[0]
        self.center_x = params.center_x[0]
        self.center_y = params.center_y[0]
        self.center_z = params.center_z[0]
        self.size_x = params.size_x[0]
        self.size_y = params.size_y[0]
        self.size_z = params.size_z[0]
        self.exhaustiveness = params.exhaustiveness[0]
        self.num_modes = params.num_modes[0]
        self.energy_range = params.energy_range[0]
        self.enable_pocket_detection = params.enable_pocket_detection[0]

        # Verify receptor file exists
        if not Path(self.receptor_path).exists():
            raise FileNotFoundError(f"Receptor PDBQT file not found: {self.receptor_path}")

    def _smiles_to_sdf(self, smiles: str, sdf_path: Path) -> bool:
        """Convert SMILES to 3D SDF file"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return False

            # Add hydrogens and generate 3D coordinates
            mol = Chem.AddHs(mol)

            # Generate conformer
            result = AllChem.EmbedMolecule(mol, randomSeed=42)
            if result != 0:
                # Try with different parameters if embedding fails
                result = AllChem.EmbedMolecule(mol, useRandomCoords=True, randomSeed=42)
                if result != 0:
                    return False

            # Optimize geometry
            AllChem.UFFOptimizeMolecule(mol)

            # Write to SDF
            writer = Chem.SDWriter(str(sdf_path))
            mol.SetProp("_Name", smiles)
            writer.write(mol)
            writer.close()

            return True

        except Exception:
            return False

    def _prepare_ligand_pdbqt(self, sdf_path: Path, pdbqt_path: Path) -> bool:
        """Convert SDF to PDBQT using Meeko or prepare_ligand4.py"""
        try:
            # Try using Meeko (modern approach)
            cmd = [
                "uv", "run", "mk_prepare_ligand.py",
                "-i", str(sdf_path),
                "-o", str(pdbqt_path)  # Use full path, Meeko will handle it correctly
            ]

            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            return pdbqt_path.exists()

        except (subprocess.CalledProcessError, FileNotFoundError):
            # Fallback: try prepare_ligand4.py if available
            try:
                cmd = [
                    "prepare_ligand4.py",
                    "-l", str(sdf_path),
                    "-o", str(pdbqt_path),
                    "-A", "hydrogens"
                ]

                subprocess.run(cmd, capture_output=True, text=True, check=True)
                return pdbqt_path.exists()

            except (subprocess.CalledProcessError, FileNotFoundError):
                return False

    def _prepare_ligand_pdbqt_parallel(self, sdf_paths: List[Path], pdbqt_paths: List[Path], max_workers: int = 8) -> List[bool]:
        """Run ligand preparation in parallel to speedup processing"""
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._prepare_ligand_pdbqt, sdf, pdbqt): i
                for i, (sdf, pdbqt) in enumerate(zip(sdf_paths, pdbqt_paths))
            }
            results = [False] * len(sdf_paths)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def _run_vina_docking(self, ligand_pdbqt: Path, output_pdbqt: Path) -> tuple[bool, Optional[Path]]:
        """Run AutoDock Vina docking

        Returns:
            (success: bool, stdout: str) - success status and stdout content for log parsing
        """
        try:
            v = Vina(sf_name="vina")
            v.set_receptor(self.receptor_path)
            v.set_ligand_from_file(str(ligand_pdbqt))
            v.compute_vina_maps(
                center=[self.center_x, self.center_y, self.center_z],
                box_size=[self.size_x, self.size_y, self.size_z]
            )

            v.optimize()
            v.dock(exhaustiveness=self.exhaustiveness, n_poses=self.num_modes)
            v.write_poses(str(output_pdbqt), n_poses=self.num_modes, overwrite=True)

            return True, output_pdbqt

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False, None

    def _extract_affinity(self, pdbqt_path: Path, aggregate_method: str = 'mean') -> Optional[float]:
        """Extract best binding affinity from Vina prediction (.pdbqt).

        Args:
            log_content: The stdout content from Vina
        """
        try:
            affinities = self._extract_all_affinities(pdbqt_path=pdbqt_path)

            if aggregate_method == 'mean':
                return float(np.mean(affinities))
            else:
                raise NotImplementedError("Only mean aggregation is implemented.")


        except Exception:
            return None

    def _extract_all_affinities(self, pdbqt_path: Path) -> List[float]:
        """Extract all pose affinities from PDBQT file"""
        affinities = []
        try:
            with open(pdbqt_path, 'r') as f:
                for line in f:
                    if line.startswith('REMARK VINA RESULT:'):
                        parts = line.split()
                        if len(parts) >= 4:
                            affinities.append(float(parts[3]))
            return affinities
        except Exception:
            return []

    def _transform_to_score(self, affinity: float, min_affinity: float = -20.0, max_affinity: float = 0.0) -> float:
        clamped = max(min_affinity, min(max_affinity, affinity))
        score = (max_affinity - clamped) / (max_affinity - min_affinity)
        return score

    def _score_single_molecule(self, smiles: str, temp_dir: Path) -> float:
        """Score a single molecule using Vina docking"""
        try:
            # Generate unique filenames
            ligand_sdf = temp_dir / f"ligand_{hash(smiles) % 10000}.sdf"
            ligand_pdbqt = temp_dir / f"ligand_{hash(smiles) % 10000}.pdbqt"
            output_pdbqt = temp_dir / f"output_{hash(smiles) % 10000}.pdbqt"

            # Step 1: Convert SMILES to SDF
            if not self._smiles_to_sdf(smiles, ligand_sdf):
                return np.nan

            # Step 2: Convert SDF to PDBQT
            if not self._prepare_ligand_pdbqt(ligand_sdf, ligand_pdbqt):
                return np.nan

            # Step 3: Run Vina docking
            success, output_path = self._run_vina_docking(ligand_pdbqt, output_pdbqt)
            if not success:
                return np.nan

            # Step 4: Extract affinity from log content
            affinity = self._extract_affinity(output_path)

            # Step 5: Transform affinity into a score
            # score = self._transform_to_score(affinity=affinity) if affinity is not None else np.nan

            return affinity

        except Exception:
            return np.nan

    def __call__(self, smilies: List[str]) -> ComponentResults:
        """Score a list of SMILES using Vina docking

        Args:
            smilies: List of SMILES strings to score

        Returns:
            ComponentResults containing binding affinities (kcal/mol)
            More negative values indicate better binding
        """
        pose_dir = Path(self.output_dir) / self.run_id / "vina_cpu"
        pose_dir.mkdir(parents=True, exist_ok=True)
        scores = []

        for smiles in smilies:
            score = self._score_single_molecule(smiles, pose_dir)
            scores.append(score)

        return ComponentResults([np.array(scores, dtype=float)])