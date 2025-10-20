"""Unit tests for Boltz2 scoring component"""
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import numpy as np
import pytest

from reinvent_plugins.components.boltz.comp_boltz2 import Boltz2, Parameters


# Simple test SMILES
TEST_SMILES = ["CCO", "c1ccccc1"]

# NR4A1 receptor sequences from the real config (2-chain heterodimer)
NR4A1_SEQUENCES = [
    "PANLLTSLVRAHLDSGPSTAKLDYSKFQELVLPHFGKEDAGDVQQFYDLLSGSLEVIRKWAEKIPGFAELSPADQDLLLESAFLELFILRLAYRSKPGEGKLIFCSGLVLHRLQCARGFGDWIDSILAFSRSLHSLLVDVPAFACLSALVLITDRHGLQEPRRVEELQNRIASCLKEHVAAVAGASCLSRLLGKLPELRTLCTQGLQRIFYLKLEDLVPPPPIIDKIFMDTLPF",
    "ANLLTSLVRAHLDSGPSTAKLDYSKFQELVLPHFGKEDAGDVQQFYDLLSGSLEVIRKWAEKIPGFAELSPADQDLLLESAFLELFILRLAYRSKPGEGKLIFCSGLVLHRLQCARGFGDWIDSILAFSRSLHSLLVDVPAFACLSALVLITDRHGLQEPRRVEELQNRIASCLKEHVAAVASCLSRLLGKLPELRTLCTQGLQRIFYLKLEDLVPPPPIIDKIFMDTLPF"
]
NR4A1_CHAIN_IDS = ["A", "B"]


class TestBoltz2Parameters:
    """Test parameter validation and initialization"""

    def test_basic_parameter_initialization(self):
        """Test that basic parameters initialize correctly"""
        params = Parameters(
            receptor_sequences=NR4A1_SEQUENCES,
            receptor_chain_ids=NR4A1_CHAIN_IDS,
            receptor_msa_paths=[None, None],
            ligand_chain_id=["C"]
        )

        assert len(params.receptor_sequences) == 2
        assert params.receptor_chain_ids == ["A", "B"]
        assert params.ligand_chain_id[0] == "C"

    def test_minimal_parameters(self):
        """Test creation with minimal parameters using defaults"""
        params = Parameters()

        # Check defaults are set
        assert params.primary_metric[0] == "affinity_probability_binary"
        assert params.devices[0] == 1
        assert params.accelerator[0] == "gpu"
        assert params.sample_aggregation_method[0] == "best"

    def test_custom_metrics_parameters(self):
        """Test custom metrics configuration"""
        params = Parameters(
            custom_metric_functions=[["mypackage.metrics.func1", "mypackage.metrics.func2"]],
            custom_metric_names=[["metric1", "metric2"]]
        )

        assert len(params.custom_metric_functions[0]) == 2
        assert len(params.custom_metric_names[0]) == 2


class TestBoltz2Initialization:
    """Test Boltz2 component initialization"""

    def test_single_chain_initialization(self):
        """Test initialization with single protein chain"""
        params = Parameters(
            receptor_sequences=["MVHLTPEEK"],
            receptor_chain_ids=["A"],
            receptor_msa_paths=[None],
            ligand_chain_id=["B"]
        )

        component = Boltz2(params)

        assert len(component.receptor_sequences) == 1
        assert component.receptor_chain_ids == ["A"]
        assert component.ligand_chain_id == "B"

    def test_multi_chain_initialization(self):
        """Test initialization with NR4A1 multi-chain complex"""
        params = Parameters(
            receptor_sequences=NR4A1_SEQUENCES,
            receptor_chain_ids=NR4A1_CHAIN_IDS,
            receptor_msa_paths=[None, None],
            ligand_chain_id=["C"]
        )

        component = Boltz2(params)

        assert len(component.receptor_sequences) == 2
        assert component.receptor_chain_ids == ["A", "B"]
        assert component.receptor_sequences[0] == NR4A1_SEQUENCES[0]
        assert component.receptor_sequences[1] == NR4A1_SEQUENCES[1]

    def test_custom_metrics_initialization(self):
        """Test initialization with custom metrics"""
        params = Parameters(
            custom_metric_functions=[["pkg.metrics.func1"]],
            custom_metric_names=[["custom1"]]
        )

        component = Boltz2(params)

        assert component.custom_metric_functions == ["pkg.metrics.func1"]
        assert component.custom_metric_names == ["custom1"]

    def test_sample_aggregation_validation(self):
        """Test that invalid sample aggregation method raises error"""
        params = Parameters(
            sample_aggregation_method=["invalid_method"]
        )

        with pytest.raises(ValueError, match="sample_aggregation_method must be one of"):
            Boltz2(params)

    def test_valid_aggregation_methods(self):
        """Test that all valid aggregation methods work"""
        valid_methods = ["best", "mean", "max", "min"]

        for method in valid_methods:
            params = Parameters(sample_aggregation_method=[method])
            component = Boltz2(params)
            assert component.sample_aggregation_method == method

    def test_custom_metrics_length_mismatch(self):
        """Test that mismatched custom metrics raise error"""
        params = Parameters(
            custom_metric_functions=[["func1", "func2"]],
            custom_metric_names=[["name1"]]  # Mismatch: 2 functions, 1 name
        )

        with pytest.raises(ValueError, match="must have the same length"):
            Boltz2(params)


class TestYAMLCreation:
    """Test YAML input file creation"""

    def test_create_yaml_single_chain(self):
        """Test YAML creation for single protein chain"""
        params = Parameters(
            receptor_sequences=["MVHLTPEEK"],
            receptor_chain_ids=["A"],
            receptor_msa_paths=[None],
            ligand_chain_id=["B"]
        )
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            success = component._create_yaml_input("CCO", yaml_path)

            assert success
            assert yaml_path.exists()

            # Read and validate YAML structure
            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            assert data["version"] == 1
            assert len(data["sequences"]) == 2  # 1 protein + 1 ligand
            assert data["sequences"][0]["protein"]["id"] == "A"
            assert data["sequences"][0]["protein"]["sequence"] == "MVHLTPEEK"
            assert data["sequences"][1]["ligand"]["id"] == "B"
            assert data["sequences"][1]["ligand"]["smiles"] == "CCO"

    def test_create_yaml_multi_chain_nr4a1(self):
        """Test YAML creation for NR4A1 multi-chain complex"""
        params = Parameters(
            receptor_sequences=NR4A1_SEQUENCES,
            receptor_chain_ids=NR4A1_CHAIN_IDS,
            receptor_msa_paths=[None, None],
            ligand_chain_id=["C"]
        )
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            success = component._create_yaml_input("c1ccccc1", yaml_path)

            assert success

            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            # Should have 2 proteins + 1 ligand = 3 sequences
            assert len(data["sequences"]) == 3
            assert data["sequences"][0]["protein"]["id"] == "A"
            assert data["sequences"][0]["protein"]["sequence"] == NR4A1_SEQUENCES[0]
            assert data["sequences"][1]["protein"]["id"] == "B"
            assert data["sequences"][1]["protein"]["sequence"] == NR4A1_SEQUENCES[1]
            assert data["sequences"][2]["ligand"]["id"] == "C"
            assert data["sequences"][2]["ligand"]["smiles"] == "c1ccccc1"

    def test_create_yaml_with_msa(self):
        """Test YAML creation with MSA paths"""
        params = Parameters(
            receptor_sequences=["MVHLTPEEK"],
            receptor_chain_ids=["A"],
            receptor_msa_paths=["/path/to/msa.a3m"],
            ligand_chain_id=["B"]
        )
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            success = component._create_yaml_input("CCO", yaml_path)

            assert success

            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            assert data["sequences"][0]["protein"]["msa"] == "/path/to/msa.a3m"

    def test_create_yaml_affinity_property(self):
        """Test that YAML includes affinity property"""
        params = Parameters(ligand_chain_id=["C"])
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            yaml_path = Path(tmpdir) / "test.yaml"
            success = component._create_yaml_input("CCO", yaml_path)

            assert success

            import yaml
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)

            assert "properties" in data
            assert len(data["properties"]) == 1
            assert "affinity" in data["properties"][0]
            assert data["properties"][0]["affinity"]["binder"] == "C"


class TestMetricExtraction:
    """Test metric extraction from Boltz2 output files"""

    def test_extract_structure_metrics_single_model(self):
        """Test extracting plddt, ptm, iptm from single model"""
        params = Parameters()
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock Boltz output directory structure
            pred_dir = Path(tmpdir)
            mol_id = "test_mol"
            confidence_dir = pred_dir / "boltz_results_inputs" / "predictions" / mol_id
            confidence_dir.mkdir(parents=True)

            # Create mock confidence file
            confidence_data = {
                "plddt": [85.5, 90.2, 88.7, 92.1],
                "ptm": 0.85,
                "iptm": 0.78
            }
            confidence_file = confidence_dir / f"confidence_{mol_id}.json"
            with open(confidence_file, 'w') as f:
                json.dump(confidence_data, f)

            # Extract metrics
            metrics = component._extract_structure_metrics(pred_dir, mol_id)

            assert "plddt" in metrics
            assert "ptm" in metrics
            assert "iptm" in metrics
            assert np.isclose(metrics["plddt"], np.mean([85.5, 90.2, 88.7, 92.1]))
            assert metrics["ptm"] == 0.85
            assert metrics["iptm"] == 0.78

    def test_extract_structure_metrics_multiple_models(self):
        """Test extracting metrics from multiple diffusion samples"""
        params = Parameters(sample_aggregation_method=["mean"])
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)
            mol_id = "test_mol"
            confidence_dir = pred_dir / "boltz_results_inputs" / "predictions" / mol_id
            confidence_dir.mkdir(parents=True)

            # Create 3 model files with different metrics
            model_data = [
                {"plddt": [90.0, 88.0], "ptm": 0.9, "iptm": 0.85},
                {"plddt": [85.0, 83.0], "ptm": 0.8, "iptm": 0.75},
                {"plddt": [88.0, 86.0], "ptm": 0.85, "iptm": 0.80}
            ]

            for i, data in enumerate(model_data):
                confidence_file = confidence_dir / f"confidence_{mol_id}_model_{i}.json"
                with open(confidence_file, 'w') as f:
                    json.dump(data, f)

            # Extract with mean aggregation
            metrics = component._extract_structure_metrics(pred_dir, mol_id)

            # Check mean aggregation
            expected_ptm = np.mean([0.9, 0.8, 0.85])
            expected_iptm = np.mean([0.85, 0.75, 0.80])

            assert np.isclose(metrics["ptm"], expected_ptm)
            assert np.isclose(metrics["iptm"], expected_iptm)

    def test_extract_affinity_single_model(self):
        """Test extracting affinity metrics from single model"""
        params = Parameters()
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)
            mol_id = "test_mol"
            affinity_dir = pred_dir / "boltz_results_inputs" / "predictions" / mol_id
            affinity_dir.mkdir(parents=True)

            # Create mock affinity file
            affinity_data = {
                "affinity_probability_binary": 0.92,
                "affinity_pred_value": -6.5
            }
            affinity_file = affinity_dir / f"affinity_{mol_id}.json"
            with open(affinity_file, 'w') as f:
                json.dump(affinity_data, f)

            # Extract metrics
            metrics = component._extract_affinity(pred_dir, mol_id)

            assert metrics["affinity_probability_binary"] == 0.92
            assert metrics["affinity_pred_value"] == -6.5

    def test_extract_affinity_multiple_models(self):
        """Test extracting affinity with multiple samples and max aggregation"""
        params = Parameters(sample_aggregation_method=["max"])
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)
            mol_id = "test_mol"
            affinity_dir = pred_dir / "boltz_results_inputs" / "predictions" / mol_id
            affinity_dir.mkdir(parents=True)

            # Create 3 model files
            affinity_values = [
                {"affinity_probability_binary": 0.85, "affinity_pred_value": -5.5},
                {"affinity_probability_binary": 0.92, "affinity_pred_value": -6.5},
                {"affinity_probability_binary": 0.88, "affinity_pred_value": -6.0}
            ]

            for i, data in enumerate(affinity_values):
                affinity_file = affinity_dir / f"affinity_{mol_id}_model_{i}.json"
                with open(affinity_file, 'w') as f:
                    json.dump(data, f)

            # Extract with max aggregation
            metrics = component._extract_affinity(pred_dir, mol_id)

            # Max should select highest values
            assert metrics["affinity_probability_binary"] == 0.92
            assert metrics["affinity_pred_value"] == -5.5  # Closer to 0 is "higher"

    def test_extract_metrics_missing_files(self):
        """Test that missing files return empty dict"""
        params = Parameters()
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)

            # Extract from non-existent directory
            metrics = component._extract_structure_metrics(pred_dir, "nonexistent")
            assert metrics == {}

            metrics = component._extract_affinity(pred_dir, "nonexistent")
            assert metrics == {}


class TestAggregationMethods:
    """Test different sample aggregation methods"""

    def test_aggregate_best(self):
        """Test 'best' aggregation (first value)"""
        params = Parameters(sample_aggregation_method=["best"])
        component = Boltz2(params)

        values = [0.9, 0.8, 0.85, 0.7]
        result = component._aggregate_values(values, "best")
        assert result == 0.9

    def test_aggregate_mean(self):
        """Test mean aggregation"""
        params = Parameters(sample_aggregation_method=["mean"])
        component = Boltz2(params)

        values = [0.8, 0.9, 0.7, 0.6]
        result = component._aggregate_values(values, "mean")
        assert np.isclose(result, 0.75)

    def test_aggregate_max(self):
        """Test max aggregation"""
        params = Parameters(sample_aggregation_method=["max"])
        component = Boltz2(params)

        values = [0.8, 0.9, 0.7, 0.95]
        result = component._aggregate_values(values, "max")
        assert result == 0.95

    def test_aggregate_min(self):
        """Test min aggregation"""
        params = Parameters(sample_aggregation_method=["min"])
        component = Boltz2(params)

        values = [0.8, 0.9, 0.6, 0.7]
        result = component._aggregate_values(values, "min")
        assert result == 0.6

    def test_aggregate_with_nan_values(self):
        """Test that NaN values are filtered before aggregation"""
        params = Parameters()
        component = Boltz2(params)

        values = [0.8, np.nan, 0.9, np.nan, 0.7]
        result = component._aggregate_values(values, "mean")
        assert np.isclose(result, np.mean([0.8, 0.9, 0.7]))

    def test_aggregate_empty_list(self):
        """Test that empty list returns NaN"""
        params = Parameters()
        component = Boltz2(params)

        result = component._aggregate_values([], "mean")
        assert np.isnan(result)

    def test_aggregate_all_nan(self):
        """Test that all NaN values returns NaN"""
        params = Parameters()
        component = Boltz2(params)

        values = [np.nan, np.nan, np.nan]
        result = component._aggregate_values(values, "mean")
        assert np.isnan(result)


class TestCustomMetrics:
    """Test custom metric functionality"""

    def test_load_custom_function_success(self):
        """Test loading a valid custom function"""
        params = Parameters()
        component = Boltz2(params)

        # Load numpy.mean as a test (it exists and is callable)
        func = component._load_custom_function("numpy.mean")
        assert func is not None
        assert callable(func)

    def test_load_custom_function_invalid_path(self):
        """Test loading invalid module path returns None"""
        params = Parameters()
        component = Boltz2(params)

        func = component._load_custom_function("nonexistent.module.function")
        assert func is None

    def test_extract_custom_metrics_with_mock_function(self):
        """Test custom metric extraction with mocked function"""
        params = Parameters(
            custom_metric_functions=[["test.metrics.custom_score"]],
            custom_metric_names=[["custom_score"]]
        )
        component = Boltz2(params)

        # Mock the custom function to return a fixed value
        mock_function = Mock(return_value=0.75)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)

            # Mock _load_custom_function to return our mock
            with patch.object(component, '_load_custom_function', return_value=mock_function):
                metrics = component._extract_custom_metrics(pred_dir, "mol123", "CCO")

            assert "custom_score" in metrics
            assert metrics["custom_score"] == 0.75
            # Verify function was called with correct args
            mock_function.assert_called_once_with(pred_dir, "mol123", "CCO")

    def test_extract_custom_metrics_function_returns_nan(self):
        """Test that custom function returning NaN is handled"""
        params = Parameters(
            custom_metric_functions=[["test.metrics.failing_metric"]],
            custom_metric_names=[["failing_metric"]]
        )
        component = Boltz2(params)

        mock_function = Mock(return_value=np.nan)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(component, '_load_custom_function', return_value=mock_function):
                metrics = component._extract_custom_metrics(Path(tmpdir), "mol123", "CCO")

            assert metrics["failing_metric"] is np.nan or np.isnan(metrics["failing_metric"])

    def test_extract_custom_metrics_function_raises_exception(self):
        """Test that exceptions in custom functions are caught"""
        params = Parameters(
            custom_metric_functions=[["test.metrics.broken_metric"]],
            custom_metric_names=[["broken_metric"]]
        )
        component = Boltz2(params)

        mock_function = Mock(side_effect=RuntimeError("Computation failed"))

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(component, '_load_custom_function', return_value=mock_function):
                metrics = component._extract_custom_metrics(Path(tmpdir), "mol123", "CCO")

            # Should return NaN instead of raising
            assert np.isnan(metrics["broken_metric"])


class TestMoleculeScoring:
    """Test end-to-end molecule scoring"""

    @pytest.mark.slow
    def test_scoring_with_real_boltz_integration(self):
        """Integration test with real Boltz2 execution on NR4A1 complex

        This test actually runs Boltz2 prediction with:
        - 2 diffusion samples
        - 1 recycling step
        - Mean aggregation
        - Simple SMILES: CCO (ethanol) and c1ccccc1 (benzene)
        - Real NR4A1 2-chain heterodimer sequences

        NOTE: Requires GPU and Boltz2 installation. May take several minutes.
        """
        # Use persistent directory for debugging
        import os
        test_output_dir = "/tmp/boltz_integration_test_output"
        os.makedirs(test_output_dir, exist_ok=True)

        print(f"\n[TEST] Output will be saved to: {test_output_dir}")
        print(f"[TEST] Testing with SMILES: {TEST_SMILES}")

        params = Parameters(
            receptor_sequences=NR4A1_SEQUENCES,  # Just a list, not nested
            receptor_chain_ids=NR4A1_CHAIN_IDS,  # Just a list, not nested
            receptor_msa_paths=[None, None],  # Just a list, not nested - No MSAs provided
            ligand_chain_id=["C"],
            primary_metric=["affinity_probability_binary"],
            additional_metrics=[["iptm", "plddt", "ptm", "affinity_pred_value"]],
            sampling_steps=[10],  # Minimal for faster testing
            diffusion_samples=[2],  # 2 samples as requested
            recycling_steps=[1],  # 1 recycling step as requested
            sample_aggregation_method=["mean"],  # Test mean aggregation
            output_dir=[test_output_dir],
            run_id=["boltz_integration_test"],
            verbose=[True]  # Enable verbose output for debugging
        )
        component = Boltz2(params)

        # Run real Boltz2 prediction
        print(f"[TEST] Running Boltz2 prediction...")
        results = component(TEST_SMILES)
        print(f"[TEST] Prediction complete!")

        # Print where structures are saved
        structure_dir = Path(test_output_dir) / "boltz_integration_test" / "boltz_pose"
        print(f"[TEST] Structures saved to: {structure_dir}")
        if structure_dir.exists():
            print(f"[TEST] Structure directories:")
            for d in structure_dir.iterdir():
                if d.is_dir():
                    files = list(d.glob("*"))
                    print(f"  {d.name}/: {len(files)} files")
                    for f in files:
                        print(f"    - {f.name}")

        # Verify results structure
        assert len(results.scores) == 1, "Should have 1 score array (primary metric)"
        assert len(results.scores[0]) == 2, "Should have 2 scores (one per SMILES)"

        # Verify scores are valid (not NaN) - real predictions should succeed
        print(f"[TEST] CCO affinity score: {results.scores[0][0]}")
        print(f"[TEST] Benzene affinity score: {results.scores[0][1]}")

        assert not np.isnan(results.scores[0][0]), "CCO prediction should not be NaN"
        assert not np.isnan(results.scores[0][1]), "Benzene prediction should not be NaN"

        # Verify scores are in valid range [0, 1] for affinity_probability_binary
        assert 0.0 <= results.scores[0][0] <= 1.0, "Affinity score should be between 0 and 1"
        assert 0.0 <= results.scores[0][1] <= 1.0, "Affinity score should be between 0 and 1"

        # Verify metadata exists and contains expected metrics
        assert results.metadata is not None, "Metadata should be present"
        assert "boltz_iptm" in results.metadata, "Should have iptm in metadata"
        assert "boltz_plddt" in results.metadata, "Should have plddt in metadata"
        assert "boltz_ptm" in results.metadata, "Should have ptm in metadata"
        assert "boltz_affinity_pred_value" in results.metadata, "Should have affinity_pred_value in metadata"

        # Verify metadata values are valid
        assert len(results.metadata["boltz_iptm"]) == 2, "Should have 2 iptm values"
        assert len(results.metadata["boltz_plddt"]) == 2, "Should have 2 plddt values"

        # iptm and ptm should be in [0, 1] range
        for i in range(2):
            if not np.isnan(results.metadata["boltz_iptm"][i]):
                assert 0.0 <= results.metadata["boltz_iptm"][i] <= 1.0, "iptm should be in [0,1]"
            if not np.isnan(results.metadata["boltz_ptm"][i]):
                assert 0.0 <= results.metadata["boltz_ptm"][i] <= 1.0, "ptm should be in [0,1]"

        # plddt should be in [0, 100] range
        for i in range(2):
            if not np.isnan(results.metadata["boltz_plddt"][i]):
                assert 0.0 <= results.metadata["boltz_plddt"][i] <= 100.0, "plddt should be in [0,100]"

        print(f"\nIntegration test results:")
        print(f"CCO affinity: {results.scores[0][0]:.3f}")
        print(f"Benzene affinity: {results.scores[0][1]:.3f}")
        print(f"CCO iptm: {results.metadata['boltz_iptm'][0]:.3f}")
        print(f"Benzene iptm: {results.metadata['boltz_iptm'][1]:.3f}")

    def test_scoring_with_failed_preparation(self):
        """Test that molecules with failed YAML preparation get NaN scores"""
        params = Parameters()
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock YAML creation to fail
            with patch.object(component, '_create_yaml_inputs_parallel', return_value=[False, False]):
                results = component(TEST_SMILES)

            # All scores should be NaN
            assert np.isnan(results.scores[0][0])
            assert np.isnan(results.scores[0][1])

    @patch('reinvent_plugins.components.boltz.comp_boltz2.subprocess.run')
    def test_scoring_with_failed_boltz_execution(self, mock_subprocess):
        """Test that failed Boltz2 execution returns NaN scores"""
        params = Parameters()
        component = Boltz2(params)

        # Mock failed subprocess run
        mock_subprocess.side_effect = Exception("Boltz2 execution failed")

        results = component(TEST_SMILES)

        # All scores should be NaN
        assert np.isnan(results.scores[0][0])
        assert np.isnan(results.scores[0][1])


class TestGetMoleculeId:
    """Test molecule ID generation"""

    def test_molecule_id_from_valid_smiles(self):
        """Test that valid SMILES generate InChI keys"""
        params = Parameters()
        component = Boltz2(params)

        mol_id = component._get_molecule_id("CCO")
        assert mol_id == "LFQSCWFLJHTTHZ-UHFFFAOYSA-N"  # InChI key for ethanol

        mol_id = component._get_molecule_id("c1ccccc1")
        assert mol_id == "UHOVQNZJYSORNB-UHFFFAOYSA-N"  # InChI key for benzene

    def test_molecule_id_from_invalid_smiles(self):
        """Test that invalid SMILES generate hash-based IDs"""
        params = Parameters()
        component = Boltz2(params)

        mol_id = component._get_molecule_id("invalid_smiles_xyz")
        assert mol_id.startswith("invalid_")

    def test_molecule_id_consistency(self):
        """Test that same SMILES always generates same ID"""
        params = Parameters()
        component = Boltz2(params)

        mol_id1 = component._get_molecule_id("CCO")
        mol_id2 = component._get_molecule_id("CCO")
        assert mol_id1 == mol_id2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
