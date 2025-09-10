"""AutoDock Vina docking scoring component for REINVENT 4

This component performs molecular docking using AutoDock Vina to estimate 
binding affinity for generated molecules against a target receptor.
"""

from __future__ import annotations

__all__ = ["AutoDockVina"]

import subprocess
import tempfile
from pathlib import Path
from typing import List, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from pydantic import Field
from pydantic.dataclasses import dataclass

from .component_results import ComponentResults
from .add_tag import add_tag


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
    center_x: List[float] = Field(default_factory=lambda: [1.910])
    center_y: List[float] = Field(default_factory=lambda: [3.831]) 
    center_z: List[float] = Field(default_factory=lambda: [2.31])
    size_x: List[float] = Field(default_factory=lambda: [20.0])
    size_y: List[float] = Field(default_factory=lambda: [20.0])
    size_z: List[float] = Field(default_factory=lambda: [20.0])
    exhaustiveness: List[int] = Field(default_factory=lambda: [8])
    num_modes: List[int] = Field(default_factory=lambda: [9])
    energy_range: List[float] = Field(default_factory=lambda: [3.0])


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
        self.center_x = params.center_x[0]
        self.center_y = params.center_y[0] 
        self.center_z = params.center_z[0]
        self.size_x = params.size_x[0]
        self.size_y = params.size_y[0]
        self.size_z = params.size_z[0]
        self.exhaustiveness = params.exhaustiveness[0]
        self.num_modes = params.num_modes[0]
        self.energy_range = params.energy_range[0]
        
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
    
    def _run_vina_docking(self, ligand_pdbqt: Path, output_pdbqt: Path, log_path: Path) -> tuple[bool, str]:
        """Run AutoDock Vina docking
        
        Returns:
            (success: bool, stdout: str) - success status and stdout content for log parsing
        """
        try:
            cmd = [
                "vina",
                "--receptor", str(self.receptor_path),
                "--ligand", str(ligand_pdbqt),
                "--center_x", str(self.center_x),
                "--center_y", str(self.center_y), 
                "--center_z", str(self.center_z),
                "--size_x", str(self.size_x),
                "--size_y", str(self.size_y),
                "--size_z", str(self.size_z),
                "--exhaustiveness", str(self.exhaustiveness),
                "--num_modes", str(self.num_modes),
                "--energy_range", str(self.energy_range),
                "--out", str(output_pdbqt)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)  # 5 min timeout
            
            if result.returncode == 0:
                # Save output to log file for debugging
                with open(log_path, 'w') as f:
                    f.write(result.stdout)
                return True, result.stdout
            else:
                return False, result.stderr
            
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired, FileNotFoundError):
            return False, ""
    
    def _extract_affinity(self, log_content: str) -> Optional[float]:
        """Extract best binding affinity from Vina output
        
        Args:
            log_content: The stdout content from Vina
        """
        try:
            lines = log_content.split('\n')
            for line in lines:
                stripped = line.strip()
                # Look for the first scoring line (mode 1)
                if stripped.startswith('1') and len(stripped.split()) >= 2:
                    parts = stripped.split()
                    try:
                        # Second column should be the affinity
                        affinity = float(parts[1])
                        return affinity
                    except (ValueError, IndexError):
                        continue
            return None
            
        except Exception:
            return None
    
    def _score_single_molecule(self, smiles: str, temp_dir: Path) -> float:
        """Score a single molecule using Vina docking"""
        try:
            # Generate unique filenames
            ligand_sdf = temp_dir / f"ligand_{hash(smiles) % 10000}.sdf"
            ligand_pdbqt = temp_dir / f"ligand_{hash(smiles) % 10000}.pdbqt" 
            output_pdbqt = temp_dir / f"output_{hash(smiles) % 10000}.pdbqt"
            log_file = temp_dir / f"docking_{hash(smiles) % 10000}.log"
            
            # Step 1: Convert SMILES to SDF
            if not self._smiles_to_sdf(smiles, ligand_sdf):
                return np.nan
            
            # Step 2: Convert SDF to PDBQT
            if not self._prepare_ligand_pdbqt(ligand_sdf, ligand_pdbqt):
                return np.nan
            
            # Step 3: Run Vina docking
            success, log_content = self._run_vina_docking(ligand_pdbqt, output_pdbqt, log_file)
            if not success:
                return np.nan
            
            # Step 4: Extract affinity from log content
            affinity = self._extract_affinity(log_content)
            
            return affinity if affinity is not None else np.nan
            
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
        scores = []
        
        # Use temporary directory for all docking operations
        with tempfile.TemporaryDirectory(prefix="vina_scoring_") as temp_dir:
            temp_path = Path(temp_dir)
            
            for smiles in smilies:
                score = self._score_single_molecule(smiles, temp_path)
                scores.append(score)
        
        return ComponentResults([np.array(scores, dtype=float)])