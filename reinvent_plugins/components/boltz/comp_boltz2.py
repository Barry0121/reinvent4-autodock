"""Boltz2 structure prediction and affinity scoring component for REINVENT 4

This component performs structure prediction and binding affinity estimation using Boltz2.
"""
from __future__ import annotations

__all__ = ["Boltz2"]

import importlib
import json
import shutil
import subprocess
import tempfile
import yaml
from pathlib import Path
from typing import List, Optional, Dict, Any, Callable
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from rdkit import Chem
from rdkit.Chem import MolToInchiKey
from pydantic import Field
from pydantic.dataclasses import dataclass

from ..component_results import ComponentResults
from ..add_tag import add_tag


@add_tag("__parameters")
@dataclass
class Parameters:
    """Parameters for Boltz2 scoring component

    Note that all parameters are lists because components can have
    multiple endpoints, even if only one is used.
    """

    # Receptor configuration - supports multiple protein chains
    # Each endpoint is a list, and each protein chain within an endpoint is specified
    # Example: [["SEQUENCE1", "SEQUENCE2"]] for two chains in one endpoint
    receptor_sequences: List[List[str]] = Field(
        default_factory=lambda: [["MVTPEGNVSLVDESLLVGVTDEDRAVRSAHQFYERLIGLWAPAVMEAAHELGVFAALAEAPADSGELARRLDCDARAMRVLLDALYAYDVIDRIHDTNGFRYLLSAEARECLLPGTLFSLVGKFMHDINVAWPAWRNLAEVVRHGARDTSGAESPNGIAQEDYESLVGGINFWAPPIVTTLSRKLRASGRSGDATASVLDVGCGTGLYSQLLLREFPRWTATGLDVERIATLANAQALRLGVEERFATRAGDFWRGGWGTGYDLVLFANIFHLQTPASAVRLMRHAAACLAPDGLVAVVDQIVDADREPKTPQDRFALLFAASMTNTGGGDAYTFQEYEEWFTAAGLQRIETLDTPMHRILLARRATEPSAVPEGQASENLYFQ"]]
    )
    receptor_chain_ids: List[List[str]] = Field(default_factory=lambda: [["A"]])
    receptor_msa_paths: List[List[Optional[str]]] = Field(default_factory=lambda: [[None]])
    ligand_chain_id: List[str] = Field(default_factory=lambda: ["B"])

    # Output configuration
    output_dir: List[str] = Field(default_factory=lambda: ["./output/boltz_predictions"])
    run_id: List[str] = Field(default_factory=lambda: ["default_run"])

    # Boltz2 execution parameters
    cache: List[str] = Field(default_factory=lambda: ["~/.boltz"])
    checkpoint: List[Optional[str]] = Field(default_factory=lambda: [None])
    devices: List[int] = Field(default_factory=lambda: [1])
    accelerator: List[str] = Field(default_factory=lambda: ["gpu"])

    # Prediction parameters
    recycling_steps: List[int] = Field(default_factory=lambda: [3])
    sampling_steps: List[int] = Field(default_factory=lambda: [200])
    diffusion_samples: List[int] = Field(default_factory=lambda: [1])
    max_parallel_samples: List[int] = Field(default_factory=lambda: [5])
    step_scale: List[float] = Field(default_factory=lambda: [1.638])

    # MSA parameters
    use_msa_server: List[bool] = Field(default_factory=lambda: [False])
    msa_server_url: List[str] = Field(default_factory=lambda: ["https://api.colabfold.com"])
    msa_pairing_strategy: List[str] = Field(default_factory=lambda: ["greedy"])
    max_msa_seqs: List[int] = Field(default_factory=lambda: [8192])
    subsample_msa: List[bool] = Field(default_factory=lambda: [False])
    num_subsampled_msa: List[int] = Field(default_factory=lambda: [1024])

    # Affinity parameters
    affinity_mw_correction: List[bool] = Field(default_factory=lambda: [False])
    sampling_steps_affinity: List[int] = Field(default_factory=lambda: [200])
    diffusion_samples_affinity: List[int] = Field(default_factory=lambda: [5])
    affinity_checkpoint: List[Optional[str]] = Field(default_factory=lambda: [None])

    # Other parameters
    output_format: List[str] = Field(default_factory=lambda: ["mmcif"])
    num_workers: List[int] = Field(default_factory=lambda: [2])
    use_potentials: List[bool] = Field(default_factory=lambda: [False])
    no_kernels: List[bool] = Field(default_factory=lambda: [False])
    override: List[bool] = Field(default_factory=lambda: [True])

    # Metrics to extract and return
    # Available metrics: affinity_probability_binary, affinity_pred_value, plddt, ptm, iptm
    primary_metric: List[str] = Field(default_factory=lambda: ["affinity_probability_binary"])
    additional_metrics: List[List[str]] = Field(
        default_factory=lambda: [["affinity_pred_value", "plddt", "ptm"]]
    )

    # Custom metric functions
    # List of Python module paths to custom metric functions
    # Example: ["mypackage.metrics.custom_rmsd", "mypackage.metrics.contact_count"]
    custom_metric_functions: List[List[str]] = Field(default_factory=lambda: [[]])
    # Names for custom metrics (must match length of custom_metric_functions)
    custom_metric_names: List[List[str]] = Field(default_factory=lambda: [[]])


@add_tag("__component")
class Boltz2:
    """Boltz2 structure prediction and affinity scoring component

    This component supports multi-chain protein complexes and:
    1. Creates YAML input files for each SMILES with multiple protein chains
    2. Runs Boltz2 prediction in batch mode
    3. Extracts binding affinity predictions
    4. Returns affinity probability binary as the score

    Supports multiple protein chains per complex (e.g., heterodimers, trimers, etc.)
    Returns affinity probability binary (0-1, higher = better binding)
    """

    def __init__(self, params: Parameters):
        # Support multiple protein chains
        self.receptor_sequences = params.receptor_sequences[0]  # List of sequences
        self.receptor_chain_ids = params.receptor_chain_ids[0]  # List of chain IDs
        self.receptor_msa_paths = params.receptor_msa_paths[0]  # List of MSA paths (can be None)
        self.ligand_chain_id = params.ligand_chain_id[0]

        self.output_dir = params.output_dir[0]
        self.run_id = params.run_id[0]

        # Boltz2 execution parameters
        self.cache = params.cache[0]
        self.checkpoint = params.checkpoint[0]
        self.devices = params.devices[0]
        self.accelerator = params.accelerator[0]

        # Prediction parameters
        self.recycling_steps = params.recycling_steps[0]
        self.sampling_steps = params.sampling_steps[0]
        self.diffusion_samples = params.diffusion_samples[0]
        self.max_parallel_samples = params.max_parallel_samples[0]
        self.step_scale = params.step_scale[0]

        # MSA parameters
        self.use_msa_server = params.use_msa_server[0]
        self.msa_server_url = params.msa_server_url[0]
        self.msa_pairing_strategy = params.msa_pairing_strategy[0]
        self.max_msa_seqs = params.max_msa_seqs[0]
        self.subsample_msa = params.subsample_msa[0]
        self.num_subsampled_msa = params.num_subsampled_msa[0]

        # Affinity parameters
        self.affinity_mw_correction = params.affinity_mw_correction[0]
        self.sampling_steps_affinity = params.sampling_steps_affinity[0]
        self.diffusion_samples_affinity = params.diffusion_samples_affinity[0]
        self.affinity_checkpoint = params.affinity_checkpoint[0]

        # Other parameters
        self.output_format = params.output_format[0]
        self.num_workers = params.num_workers[0]
        self.use_potentials = params.use_potentials[0]
        self.no_kernels = params.no_kernels[0]
        self.override = params.override[0]

        # Metrics configuration
        self.primary_metric = params.primary_metric[0]
        self.additional_metrics = params.additional_metrics[0]

        # Custom metrics configuration
        self.custom_metric_functions = params.custom_metric_functions[0]
        self.custom_metric_names = params.custom_metric_names[0]

        # Validate custom metrics configuration
        if len(self.custom_metric_functions) != len(self.custom_metric_names):
            raise ValueError(
                f"custom_metric_functions and custom_metric_names must have the same length. "
                f"Got {len(self.custom_metric_functions)} functions and {len(self.custom_metric_names)} names"
            )

    def _get_molecule_id(self, smiles: str) -> str:
        """Generate unique molecule ID from SMILES"""
        try:
            mol = Chem.MolFromSmiles(smiles)
            return MolToInchiKey(mol) if mol else f"invalid_{hash(smiles)}"
        except:
            return f"invalid_{hash(smiles)}"

    def _create_yaml_input(self, smiles: str, yaml_path: Path) -> bool:
        """Create Boltz2 YAML input file for a ligand-protein complex

        Args:
            smiles: SMILES string of the ligand
            yaml_path: Path where to save the YAML file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Build the YAML structure with multiple protein chains
            sequences = []

            # Add all protein chains
            for i, (chain_id, sequence) in enumerate(zip(self.receptor_chain_ids, self.receptor_sequences)):
                protein_entry = {
                    "protein": {
                        "id": chain_id,
                        "sequence": sequence,
                    }
                }

                # Add MSA path if provided for this chain
                msa_path = self.receptor_msa_paths[i]
                if msa_path is not None:
                    protein_entry["protein"]["msa"] = msa_path

                sequences.append(protein_entry)

            # Add ligand
            sequences.append({
                "ligand": {
                    "id": self.ligand_chain_id,
                    "smiles": smiles
                }
            })

            yaml_data = {
                "version": 1,
                "sequences": sequences,
                "properties": [
                    {
                        "affinity": {
                            "binder": self.ligand_chain_id
                        }
                    }
                ]
            }

            # Write YAML file
            with open(yaml_path, 'w') as f:
                yaml.dump(yaml_data, f, default_flow_style=False, sort_keys=False)

            return True

        except Exception:
            return False

    def _create_yaml_inputs_parallel(self, smilies: List[str], input_dir: Path,
                                     max_workers: int = 8) -> List[bool]:
        """Create YAML input files in parallel

        Args:
            smilies: List of SMILES strings
            input_dir: Directory to save YAML files
            max_workers: Number of parallel workers

        Returns:
            List of booleans indicating success for each SMILES
        """
        mol_ids = [self._get_molecule_id(s) for s in smilies]
        yaml_paths = [input_dir / f"{mol_id}.yaml" for mol_id in mol_ids]
        input_dir.mkdir(parents=True, exist_ok=True)

        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self._create_yaml_input, smiles, yaml_path): i
                for i, (smiles, yaml_path) in enumerate(zip(smilies, yaml_paths))
            }
            results = [False] * len(smilies)
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()

        return results

    def _run_boltz2_prediction(self, input_dir: Path, temp_out_dir: Path) -> bool:
        """Run Boltz2 prediction on all YAML files in input directory

        Args:
            input_dir: Directory containing YAML input files
            temp_out_dir: Temporary output directory for predictions

        Returns:
            True if successful, False otherwise
        """
        try:
            cmd = [
                "boltz", "predict", str(input_dir),
                "--out_dir", str(temp_out_dir),
                "--cache", self.cache,
                "--devices", str(self.devices),
                "--accelerator", self.accelerator,
                "--recycling_steps", str(self.recycling_steps),
                "--sampling_steps", str(self.sampling_steps),
                "--diffusion_samples", str(self.diffusion_samples),
                "--max_parallel_samples", str(self.max_parallel_samples),
                "--step_scale", str(self.step_scale),
                "--output_format", self.output_format,
                "--num_workers", str(self.num_workers),
                "--max_msa_seqs", str(self.max_msa_seqs),
                "--sampling_steps_affinity", str(self.sampling_steps_affinity),
                "--diffusion_samples_affinity", str(self.diffusion_samples_affinity),
            ]

            # Add optional checkpoint
            if self.checkpoint:
                cmd.extend(["--checkpoint", self.checkpoint])

            if self.affinity_checkpoint:
                cmd.extend(["--affinity_checkpoint", self.affinity_checkpoint])

            # Add boolean flags
            if self.use_msa_server:
                cmd.append("--use_msa_server")
                cmd.extend(["--msa_server_url", self.msa_server_url])
                cmd.extend(["--msa_pairing_strategy", self.msa_pairing_strategy])

            if self.subsample_msa:
                cmd.append("--subsample_msa")
                cmd.extend(["--num_subsampled_msa", str(self.num_subsampled_msa)])

            if self.affinity_mw_correction:
                cmd.append("--affinity_mw_correction")

            if self.use_potentials:
                cmd.append("--use_potentials")

            if self.no_kernels:
                cmd.append("--no_kernels")

            if self.override:
                cmd.append("--override")

            # Run Boltz2
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True
            )

            return True

        except (subprocess.CalledProcessError, FileNotFoundError):
            return False

    def _extract_structure_metrics(self, prediction_dir: Path, mol_id: str) -> Optional[Dict[str, float]]:
        """Extract structure confidence metrics from Boltz2 output

        Args:
            prediction_dir: Directory containing Boltz2 predictions
            mol_id: Molecule ID

        Returns:
            Dictionary with structure metrics (plddt, ptm, iptm), or None if not found
        """
        try:
            # Boltz2 saves confidence metrics in predictions/{input_name}/confidence_{input_name}.json
            confidence_file = prediction_dir / "predictions" / mol_id / f"confidence_{mol_id}.json"

            if not confidence_file.exists():
                return {}

            with open(confidence_file, 'r') as f:
                confidence_data = json.load(f)

            metrics = {}
            # Extract mean pLDDT (per-residue confidence)
            if "plddt" in confidence_data:
                plddt_values = confidence_data["plddt"]
                if isinstance(plddt_values, list):
                    metrics["plddt"] = float(np.mean(plddt_values))
                else:
                    metrics["plddt"] = float(plddt_values)

            # Extract pTM (predicted TM-score)
            if "ptm" in confidence_data:
                metrics["ptm"] = float(confidence_data["ptm"])

            # Extract ipTM (interface predicted TM-score)
            if "iptm" in confidence_data:
                metrics["iptm"] = float(confidence_data["iptm"])

            return metrics

        except Exception:
            return {}

    def _extract_affinity(self, prediction_dir: Path, mol_id: str) -> Optional[Dict[str, float]]:
        """Extract affinity predictions from Boltz2 output

        Args:
            prediction_dir: Directory containing Boltz2 predictions
            mol_id: Molecule ID

        Returns:
            Dictionary with affinity_probability_binary and affinity_pred_value, or None if not found
        """
        try:
            # Boltz2 saves predictions in predictions/{input_name}/affinity_{input_name}.json
            affinity_file = prediction_dir / "predictions" / mol_id / f"affinity_{mol_id}.json"

            if not affinity_file.exists():
                return {}

            with open(affinity_file, 'r') as f:
                affinity_data = json.load(f)

            return {
                "affinity_probability_binary": affinity_data.get("affinity_probability_binary", np.nan),
                "affinity_pred_value": affinity_data.get("affinity_pred_value", np.nan)
            }

        except Exception:
            return {}

    def _load_custom_function(self, function_path: str) -> Optional[Callable]:
        """Dynamically load a custom metric function from a module path

        Args:
            function_path: Full module path to function (e.g., "mypackage.metrics.my_function")

        Returns:
            Callable function or None if loading fails
        """
        try:
            # Split module path and function name
            module_path, function_name = function_path.rsplit('.', 1)

            # Import module and get function
            module = importlib.import_module(module_path)
            return getattr(module, function_name)

        except (ImportError, AttributeError, ValueError) as e:
            print(f"Warning: Failed to load custom metric function '{function_path}': {e}")
            return None

    def _extract_custom_metrics(self, prediction_dir: Path, mol_id: str, smiles: str) -> Dict[str, float]:
        """Extract custom metrics using user-provided functions

        Args:
            prediction_dir: Directory containing Boltz2 predictions
            mol_id: Molecule ID
            smiles: Input SMILES string

        Returns:
            Dictionary with custom metric values
        """
        custom_metrics = {}

        for func_path, metric_name in zip(self.custom_metric_functions, self.custom_metric_names):
            try:
                # Load the custom function
                custom_func = self._load_custom_function(func_path)

                if custom_func is None:
                    custom_metrics[metric_name] = np.nan
                    continue

                # Call the custom function
                # Function signature: (prediction_dir: Path, mol_id: str, smiles: str) -> float
                metric_value = custom_func(prediction_dir, mol_id, smiles)

                # Validate and store
                if isinstance(metric_value, (int, float)):
                    custom_metrics[metric_name] = float(metric_value)
                else:
                    print(f"Warning: Custom metric '{metric_name}' returned non-numeric value: {type(metric_value)}")
                    custom_metrics[metric_name] = np.nan

            except Exception as e:
                print(f"Warning: Error computing custom metric '{metric_name}': {e}")
                custom_metrics[metric_name] = np.nan

        return custom_metrics

    def _extract_all_metrics(self, prediction_dir: Path, mol_id: str, smiles: str = "") -> Optional[Dict[str, float]]:
        """Extract all available metrics from Boltz2 output

        Args:
            prediction_dir: Directory containing Boltz2 predictions
            mol_id: Molecule ID
            smiles: Input SMILES string (for custom metrics)

        Returns:
            Dictionary with all available metrics, or None if no metrics found
        """
        all_metrics = {}

        # Extract affinity metrics
        affinity_metrics = self._extract_affinity(prediction_dir, mol_id)
        if affinity_metrics:
            all_metrics.update(affinity_metrics)

        # Extract structure metrics
        structure_metrics = self._extract_structure_metrics(prediction_dir, mol_id)
        if structure_metrics:
            all_metrics.update(structure_metrics)

        # Extract custom metrics
        if self.custom_metric_functions:
            custom_metrics = self._extract_custom_metrics(prediction_dir, mol_id, smiles)
            if custom_metrics:
                all_metrics.update(custom_metrics)

        return all_metrics if all_metrics else None

    def _extract_metrics_aligned(self, prediction_dir: Path, smilies: List[str],
                                preparation_results: List[bool]) -> Dict[str, List[float]]:
        """Extract all metrics aligned with input SMILES list

        Args:
            prediction_dir: Directory containing Boltz2 predictions
            smilies: Original list of SMILES in order
            preparation_results: Boolean list indicating which molecules were successfully prepared

        Returns:
            Dictionary mapping metric names to lists of values
        """
        # Initialize built-in metrics
        all_metric_names = ["affinity_probability_binary", "affinity_pred_value", "plddt", "ptm", "iptm"]

        # Add custom metric names
        if self.custom_metric_names:
            all_metric_names.extend(self.custom_metric_names)

        # Initialize all metrics with NaN
        metrics_dict = {metric: [np.nan] * len(smilies) for metric in all_metric_names}

        mol_ids = [self._get_molecule_id(s) for s in smilies]

        for i, (mol_id, smiles, prepared) in enumerate(zip(mol_ids, smilies, preparation_results)):
            if prepared:
                # Pass SMILES to _extract_all_metrics for custom metrics
                all_metrics = self._extract_all_metrics(prediction_dir, mol_id, smiles)
                if all_metrics:
                    for metric_name, value in all_metrics.items():
                        if metric_name in metrics_dict:
                            metrics_dict[metric_name][i] = value

        return metrics_dict

    def _copy_latest_predictions(self, temp_pred_dir: Path, output_pred_dir: Path,
                                smilies: List[str], binary_scores: List[float]):
        """Copy successful predictions to permanent output directory

        Args:
            temp_pred_dir: Temporary prediction directory
            output_pred_dir: Permanent output directory
            smilies: List of SMILES that were predicted
            binary_scores: List of binary scores (None/NaN for failed predictions)
        """
        try:
            # Clear and recreate output directory
            if output_pred_dir.exists():
                shutil.rmtree(output_pred_dir)
            output_pred_dir.mkdir(parents=True, exist_ok=True)

            mol_ids = [self._get_molecule_id(s) for s in smilies]

            # Copy only successful predictions
            for mol_id, score in zip(mol_ids, binary_scores):
                if not (np.isnan(score) or score is None):
                    src_dir = temp_pred_dir / "predictions" / mol_id
                    dst_dir = output_pred_dir / mol_id

                    if src_dir.exists():
                        shutil.copytree(src_dir, dst_dir, dirs_exist_ok=True)

        except Exception as e:
            print(f"Warning: Failed to copy predictions to output directory: {e}")

    def _score_molecules(self, smilies: List[str], out_dir: Path, temp_dir: Path) -> Dict[str, List[float]]:
        """Score molecules using Boltz2 prediction

        Args:
            smilies: List of SMILES to score
            out_dir: Permanent output directory
            temp_dir: Temporary directory for operations

        Returns:
            Dictionary mapping metric names to lists of values
        """
        try:
            # Create input directory for YAML files
            input_dir = temp_dir / "inputs"
            temp_out_dir = temp_dir / "boltz_output"

            # Step 1: Create YAML input files
            preparation_results = self._create_yaml_inputs_parallel(smilies, input_dir)

            if not any(preparation_results):
                return self._get_empty_metrics(len(smilies))

            # Step 2: Run Boltz2 prediction
            success = self._run_boltz2_prediction(input_dir, temp_out_dir)

            if not success:
                return self._get_empty_metrics(len(smilies))

            # Step 3: Extract all metrics
            metrics_dict = self._extract_metrics_aligned(
                temp_out_dir, smilies, preparation_results
            )

            # Step 4: Copy successful predictions to permanent output
            # Use primary metric for determining success
            primary_scores = metrics_dict.get(self.primary_metric, [np.nan] * len(smilies))
            self._copy_latest_predictions(
                temp_out_dir,
                out_dir / "boltz_pose",
                smilies,
                primary_scores
            )

            return metrics_dict

        except Exception:
            return self._get_empty_metrics(len(smilies))

    def _get_empty_metrics(self, n_molecules: int) -> Dict[str, List[float]]:
        """Get empty metrics dictionary for failed predictions

        Args:
            n_molecules: Number of molecules

        Returns:
            Dictionary with NaN values for all metrics
        """
        # Built-in metrics
        all_metric_names = ["affinity_probability_binary", "affinity_pred_value", "plddt", "ptm", "iptm"]

        # Add custom metrics
        if self.custom_metric_names:
            all_metric_names.extend(self.custom_metric_names)

        return {metric: [np.nan] * n_molecules for metric in all_metric_names}

    def __call__(self, smilies: List[str]) -> ComponentResults:
        """Score a list of SMILES using Boltz2 prediction

        Args:
            smilies: List of SMILES strings to score

        Returns:
            ComponentResults containing:
            - Primary score: configured primary metric (default: affinity_probability_binary)
            - Additional scores: can be configured to include other metrics as separate scores
            - Metadata: all additional configured metrics

        Example metrics:
            - affinity_probability_binary: 0-1, higher = better binding
            - affinity_pred_value: log10(IC50)
            - plddt: mean per-residue confidence (0-100)
            - ptm: predicted TM-score (0-1)
            - iptm: interface predicted TM-score (0-1)
        """
        # Permanent storage location
        out_dir = Path(self.output_dir) / self.run_id
        out_dir.mkdir(parents=True, exist_ok=True)

        # Create temporary directory for operations
        with tempfile.TemporaryDirectory() as temp_dir_str:
            temp_dir = Path(temp_dir_str)
            metrics_dict = self._score_molecules(smilies, out_dir, temp_dir)

        # Get primary metric scores
        primary_scores = metrics_dict.get(self.primary_metric, [np.nan] * len(smilies))

        # Build scores list - primary metric is always first
        scores = [np.array(primary_scores, dtype=float)]

        # Build metadata dictionary with additional metrics
        metadata = {}
        for metric_name in self.additional_metrics:
            if metric_name in metrics_dict:
                metadata[f"boltz_{metric_name}"] = metrics_dict[metric_name]

        # Create ComponentResults with primary metric and metadata
        results = ComponentResults(
            scores=scores,
            metadata=metadata if metadata else None
        )

        return results
