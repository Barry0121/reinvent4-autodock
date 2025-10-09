"""AutoDock GPU docking scoring component for REINVENT 4

This component performs molecular high throughput docking using AutoDock GPU to estimate
binding affinity for generated molecules against a target receptor.
"""
from __future__ import annotations

__all__ = ["VinaGPU"]

import shutil
import subprocess
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, MolToInchiKey
from pydantic import Field
from pydantic.dataclasses import dataclass

from ..component_results import ComponentResults
from ..add_tag import add_tag

@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for Vina GPU scoring component

    Note that all parameters are lists because components can have
    multiple endpoints, even if only one is used.
    """

    receptor_pdbqt_path: List[str] = Field(
        default_factory=lambda: ["/mnt/data/gpu-server/NR4A1-small-molecule/nr4a1_reinvent/data_cache/6kz5_receptor.pdbqt"]
    )
    # Output
    output_dir: List[str] = Field(default_factory=lambda: ["./output/docking_poses"])
    run_id: List[str] = Field(default_factory=lambda: ["default_run"])
    # Binding Box
    center_x: List[float] = Field(default_factory=lambda: [1.910])
    center_y: List[float] = Field(default_factory=lambda: [3.831])
    center_z: List[float] = Field(default_factory=lambda: [2.31])
    size_x: List[float] = Field(default_factory=lambda: [20.0])
    size_y: List[float] = Field(default_factory=lambda: [20.0])
    size_z: List[float] = Field(default_factory=lambda: [20.0])
    # Run setting
    num_modes: List[int] = Field(default_factory=lambda: [9])
    energy_range: List[float] = Field(default_factory=lambda: [3.0])
    # Script setting
    opencl_binary_path: List[str] = Field(default_factory=lambda: ["/mnt/data/gpu-server/NR4A1-small-molecule/src/Vina-GPU-2.1/AutoDock-Vina-GPU-2.1"])
    threads: List[int] = Field(default_factory=lambda: [8000])

@add_tag("__component")
class VinaGPU:
    """Vina-GPU docking scoring component

    This component:
    1. Converts SMILES to 3D SDF format
    2. Converts SDF to PDBQT format for Vina
    3. Runs Vina-GPU docking
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
        self.num_modes = params.num_modes[0]
        self.energy_range = params.energy_range[0]
        self.opencl_binary_path = params.opencl_binary_path[0]
        self.threads = params.threads[0]


        # Verify receptor file exists
        if not Path(self.receptor_path).exists():
            raise FileNotFoundError(f"Receptor PDBQT file not found: {self.receptor_path}")

    def _get_molecule_id(self, smiles: str) -> str:
        try:
            mol = Chem.MolFromSmiles(smiles)
            return MolToInchiKey(mol) if mol else f"invalid_{hash(smiles)}"
        except:
            return f"invalid_{hash(smiles)}"

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

    def _prepare_ligand_pdbqt(self, smiles: str, sdf_path: Path, pdbqt_path: Path) -> bool:
        """Convert SDF to PDBQT using Meeko or prepare_ligand4.py"""
        try:
            out = self._smiles_to_sdf(smiles, sdf_path)
            if not out:
                raise RuntimeError
        except (RuntimeError, FileNotFoundError):
            return False

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

    def _prepare_ligand_pdbqt_parallel(self, smiles_lst: List[str], ligand_dir: Path, max_workers: int = 8) -> List[bool]:
        """Run ligand preparation in parallel to speedup processing"""
        mol_ids = [self._get_molecule_id(s) for s in smiles_lst]
        ligand_sdfs = [ligand_dir / "sdf" / f"{mol_id}.sdf" for mol_id in mol_ids]
        ligand_pdbqts = [ligand_dir / "pdbqt" / f"{mol_id}.pdbqt" for mol_id in mol_ids]
        (ligand_dir / "sdf").mkdir(parents=True, exist_ok=True)
        (ligand_dir / "pdbqt").mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._prepare_ligand_pdbqt, smiles, sdf, pdbqt): i
                for i, (smiles, sdf, pdbqt) in enumerate(zip(smiles_lst, ligand_sdfs, ligand_pdbqts))
            }
            results = [False] * len(ligand_sdfs)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
        return results

    def _run_batched_vina_docking(self, ligand_dir: Path, pose_dir: Path, verbose: bool = False) -> tuple[bool, Optional[Path]]:
        """Run AutoDock Vina docking

        Returns:
            (success: bool, stdout: str) - success status and stdout content for log parsing
        """
        try:
            ligand_input = ligand_dir / "pdbqt"
            cmd = [
                "./AutoDock-Vina-GPU-2-1",
                "--receptor", str(self.receptor_path),
                "--ligand_directory", str(ligand_input),
                "--output_directory", str(pose_dir),
                "--opencl_binary_path", str(self.opencl_binary_path),
                "--thread", str(self.threads),
                "--center_x", str(self.center_x),
                "--center_y", str(self.center_y),
                "--center_z", str(self.center_z),
                "--size_x", str(self.size_x),
                "--size_y", str(self.size_y),
                "--size_z", str(self.size_z),
                "--num_modes", str(self.num_modes),
                "--energy_range", str(self.energy_range),
            ]

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,
                cwd=str(self.opencl_binary_path)
            )
            if process.stdout is not None and verbose:
                for line in process.stdout:
                    print(line, end=" ")
            process.wait()

            return True, pose_dir

        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False, None


    def _extract_all_affinities(self, pdbqt_path: Path) -> List[float]:
        """Extract all pose affinities from PDBQT file"""
        affinities = []
        try:
            with open(str(pdbqt_path.resolve()), 'r') as f:
                for line in f:
                    if line.startswith('REMARK VINA RESULT'):
                        parts = line.split()
                        if len(parts) >= 4:
                            affinities.append(float(parts[3]))

            return affinities
        except Exception:
            return []

    def _extract_affinity_batch_aligned(self, pdbqt_dir: Path, smilies: List[str],
                                       preparation_results: List[bool],
                                       aggregate_method: str = 'mean') -> List[float]:
        """Extract affinities aligned with input SMILES list

        Args:
            pdbqt_dir: Directory containing PDBQT output files
            smilies: Original list of SMILES in order
            preparation_results: Boolean list indicating which molecules were successfully prepared
            aggregate_method: Method to aggregate multiple poses ('mean')

        Returns:
            List of affinities matching input SMILES order (NaN for failures)
        """
        # Initialize all results as NaN
        results = [np.nan] * len(smilies)

        # Get molecule IDs for all input SMILES
        mol_ids = [self._get_molecule_id(s) for s in smilies]

        # Process each molecule sequentially
        for i, (mol_id, prepared) in enumerate(zip(mol_ids, preparation_results)):
            if prepared:
                # Vina-GPU appends "_out" to output filenames
                pdbqt_path = pdbqt_dir / f"{mol_id}_out.pdbqt"
                if pdbqt_path.exists():
                    affinities = self._extract_all_affinities(pdbqt_path)
                    if affinities:
                        if aggregate_method == 'mean':
                            results[i] = float(np.mean(affinities))
                        else:
                            raise NotImplementedError("Only mean aggregation is implemented.")

        return results


    def _score_molecules(self, smilies: List[str], out_dir: Path, temp_dir: Path):
        """Score a single molecule using Vina docking

        Args:
            smilies: List of SMILES to score
            out_dir: Permanent output directory for latest poses
            temp_dir: Temporary directory for docking operations

        Returns:
            List of affinities matching the input SMILES order (NaN for failures)
        """
        try:
            # Use temporary directory for docking operations
            ligand_dir = temp_dir / "ligands"
            temp_pose_dir = temp_dir / "vina_gpu_pose"

            # Step 1: Convert SMILES to PDBQT - track which ones succeeded
            preparation_results = self._prepare_ligand_pdbqt_parallel(smilies, ligand_dir)

            # If no molecules were successfully prepared, return all NaN
            if not any(preparation_results):
                return [np.nan] * len(smilies)

            # Step 2: Run Batched Vina-GPU docking
            success, _ = self._run_batched_vina_docking(ligand_dir, temp_pose_dir, verbose=False)

            if not success:
                return [np.nan] * len(smilies)

            # Step 3: Extract affinity from temp pose directory, aligned with input SMILES
            affinity = self._extract_affinity_batch_aligned(temp_pose_dir, smilies, preparation_results)


            # Step 4: Copy successful poses to permanent output location
            self._copy_latest_poses(temp_pose_dir, out_dir / "vina_gpu_pose", smilies, affinity)

            return affinity

        except Exception:
            return [np.nan] * len(smilies)

    def _copy_latest_poses(self, temp_pose_dir: Path, output_pose_dir: Path, smilies: List[str], affinities: Optional[List[float]]):
        """Copy only the latest successful docking poses to the output directory

        Clears the output directory first to ensure only the current batch's poses are stored.

        Args:
            temp_pose_dir: Temporary directory containing all poses
            output_pose_dir: Permanent output directory
            smilies: List of SMILES that were scored
            affinities: List of affinities (None for failed dockings)
        """
        try:
            if affinities is None:
                return

            # Clear and recreate output directory to only keep latest batch
            if output_pose_dir.exists():
                shutil.rmtree(output_pose_dir)
            output_pose_dir.mkdir(parents=True, exist_ok=True)

            # Get molecule IDs for successful dockings
            mol_ids = [self._get_molecule_id(s) for s in smilies]

            # Copy only successful poses (where affinity is not None/NaN)
            for mol_id, affinity in zip(mol_ids, affinities):
                if affinity is not None and not (isinstance(affinity, float) and np.isnan(affinity)):
                    # Vina-GPU appends "_out" to output filenames
                    pose_file = temp_pose_dir / f"{mol_id}_out.pdbqt"
                    if pose_file.exists():
                        shutil.copy2(pose_file, output_pose_dir / f"{mol_id}.pdbqt")
        except Exception as e:
            # Don't fail the whole scoring if copying fails
            print(f"Warning: Failed to copy poses to output directory: {e}")

    def __call__(self, smilies: List[str]) -> ComponentResults:
        """Score a list of SMILES using Vina docking

        Args:
            smilies: List of SMILES strings to score

        Returns:
            ComponentResults containing binding affinities (kcal/mol)
            More negative values indicate better binding
        """
        # Permanent storage location for latest poses
        out_dir = Path(self.output_dir) / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for docking operations
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            affinities = self._score_molecules(smilies, out_dir, temp_dir)

        return ComponentResults([np.array(affinities, dtype=float)])