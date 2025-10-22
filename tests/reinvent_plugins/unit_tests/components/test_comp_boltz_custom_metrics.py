"""Integration tests for Boltz2 component with custom metrics

This module tests the custom metrics functionality of the Boltz2 component,
including:
- Loading and executing custom metric functions
- Integration with ipSAE metrics
- Combining built-in and custom metrics
- Error handling for custom metrics
- End-to-end workflow with custom metrics
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest

from reinvent_plugins.components.boltz.comp_boltz2 import Boltz2, Parameters


# Simple test SMILES
TEST_SMILES = ["CCO", "c1ccccc1"]


# =============================================================================
# Test Custom Metric Loading
# =============================================================================

class TestCustomMetricLoading:
    """Test loading and caching of custom metric functions"""

    def test_load_valid_custom_function(self):
        """Test loading a valid custom metric function"""
        params = Parameters()
        component = Boltz2(params)

        # Load a real function (numpy.mean)
        func = component._load_custom_function("numpy.mean")

        assert func is not None
        assert callable(func)
        # Test it works
        assert func([1, 2, 3, 4]) == 2.5

    def test_load_invalid_module_path(self):
        """Test that invalid module path returns None"""
        params = Parameters()
        component = Boltz2(params)

        func = component._load_custom_function("nonexistent.module.function")
        assert func is None

    def test_load_invalid_function_name(self):
        """Test that invalid function name in valid module returns None"""
        params = Parameters()
        component = Boltz2(params)

        func = component._load_custom_function("numpy.nonexistent_function")
        assert func is None

    def test_custom_function_caching(self):
        """Test that custom functions are cached after first load"""
        params = Parameters()
        component = Boltz2(params)

        # Load function twice
        func1 = component._load_custom_function("numpy.mean")
        func2 = component._load_custom_function("numpy.mean")

        # Should be the same object (cached)
        assert func1 is func2


# =============================================================================
# Test Custom Metric Extraction
# =============================================================================

class TestCustomMetricExtraction:
    """Test extraction of custom metrics from predictions"""

    def test_extract_single_custom_metric(self):
        """Test extracting a single custom metric"""
        params = Parameters(
            custom_metric_functions=[["test.custom_func"]],
            custom_metric_names=[["custom_score"]]
        )
        component = Boltz2(params)

        # Mock the custom function
        mock_func = Mock(return_value=0.85)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)
            mol_id = "test_mol"

            with patch.object(component, '_load_custom_function', return_value=mock_func):
                metrics = component._extract_custom_metrics(pred_dir, mol_id, "CCO")

            assert "custom_score" in metrics
            assert metrics["custom_score"] == 0.85
            mock_func.assert_called_once_with(pred_dir, mol_id, "CCO")

    def test_extract_multiple_custom_metrics(self):
        """Test extracting multiple custom metrics"""
        params = Parameters(
            custom_metric_functions=[["func1", "func2", "func3"]],
            custom_metric_names=[["metric1", "metric2", "metric3"]]
        )
        component = Boltz2(params)

        # Mock multiple functions with different return values
        mock_func1 = Mock(return_value=0.7)
        mock_func2 = Mock(return_value=0.8)
        mock_func3 = Mock(return_value=0.9)

        def mock_loader(path):
            if "func1" in path:
                return mock_func1
            elif "func2" in path:
                return mock_func2
            elif "func3" in path:
                return mock_func3
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)
            mol_id = "test_mol"

            with patch.object(component, '_load_custom_function', side_effect=mock_loader):
                metrics = component._extract_custom_metrics(pred_dir, mol_id, "CCO")

            assert len(metrics) == 3
            assert metrics["metric1"] == 0.7
            assert metrics["metric2"] == 0.8
            assert metrics["metric3"] == 0.9

    def test_custom_metric_returns_nan(self):
        """Test that NaN from custom function is handled correctly"""
        params = Parameters(
            custom_metric_functions=[["failing_func"]],
            custom_metric_names=[["failing_metric"]]
        )
        component = Boltz2(params)

        mock_func = Mock(return_value=np.nan)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)
            mol_id = "test_mol"

            with patch.object(component, '_load_custom_function', return_value=mock_func):
                metrics = component._extract_custom_metrics(pred_dir, mol_id, "CCO")

            assert "failing_metric" in metrics
            assert np.isnan(metrics["failing_metric"])

    def test_custom_metric_raises_exception(self):
        """Test that exceptions in custom functions are caught and return NaN"""
        params = Parameters(
            custom_metric_functions=[["error_func"]],
            custom_metric_names=[["error_metric"]]
        )
        component = Boltz2(params)

        mock_func = Mock(side_effect=RuntimeError("Computation failed"))

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)
            mol_id = "test_mol"

            with patch.object(component, '_load_custom_function', return_value=mock_func):
                metrics = component._extract_custom_metrics(pred_dir, mol_id, "CCO")

            # Should return NaN instead of raising
            assert "error_metric" in metrics
            assert np.isnan(metrics["error_metric"])

    def test_custom_metric_function_not_loaded(self):
        """Test behavior when custom function fails to load"""
        params = Parameters(
            custom_metric_functions=[["nonexistent.func"]],
            custom_metric_names=[["missing_metric"]]
        )
        component = Boltz2(params)

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir)
            mol_id = "test_mol"

            # _load_custom_function will return None for nonexistent module
            metrics = component._extract_custom_metrics(pred_dir, mol_id, "CCO")

            # Should return NaN for missing function
            assert "missing_metric" in metrics
            assert np.isnan(metrics["missing_metric"])


# =============================================================================
# Test ipSAE Integration
# =============================================================================

class TestIpSAEIntegration:
    """Test integration of ipSAE metrics with Boltz2 component"""

    def test_load_ipsae_d0res(self):
        """Test loading ipSAE_d0res metric"""
        params = Parameters()
        component = Boltz2(params)

        func = component._load_custom_function(
            "reinvent_plugins.components.boltz.ipsae_metrics.ipsae_d0res"
        )

        assert func is not None
        assert callable(func)

    def test_load_all_ipsae_metrics(self):
        """Test loading all ipSAE metric variants"""
        params = Parameters()
        component = Boltz2(params)

        ipsae_functions = [
            "reinvent_plugins.components.boltz.ipsae_metrics.ipsae_d0res",
            "reinvent_plugins.components.boltz.ipsae_metrics.ipsae_d0chn",
            "reinvent_plugins.components.boltz.ipsae_metrics.ipsae_d0dom",
            "reinvent_plugins.components.boltz.ipsae_metrics.pdockq",
            "reinvent_plugins.components.boltz.ipsae_metrics.pdockq2",
            "reinvent_plugins.components.boltz.ipsae_metrics.lis_score",
            "reinvent_plugins.components.boltz.ipsae_metrics.iptm_from_pae",
        ]

        for func_path in ipsae_functions:
            func = component._load_custom_function(func_path)
            assert func is not None, f"Failed to load {func_path}"
            assert callable(func)

    def test_ipsae_metrics_configuration(self):
        """Test Boltz2 component configured with ipSAE metrics"""
        params = Parameters(
            custom_metric_functions=[[
                "reinvent_plugins.components.boltz.ipsae_metrics.ipsae_d0res",
                "reinvent_plugins.components.boltz.ipsae_metrics.pdockq2"
            ]],
            custom_metric_names=[["ipsae", "pdockq2"]],
            primary_metric=["ipsae"],  # Use ipSAE as optimization target
            additional_metrics=[["pdockq2", "affinity_probability_binary"]]
        )

        component = Boltz2(params)

        assert component.custom_metric_functions == [
            "reinvent_plugins.components.boltz.ipsae_metrics.ipsae_d0res",
            "reinvent_plugins.components.boltz.ipsae_metrics.pdockq2"
        ]
        assert component.custom_metric_names == ["ipsae", "pdockq2"]
        assert component.primary_metric == "ipsae"


# =============================================================================
# Test Custom Metrics Examples
# =============================================================================

class TestCustomMetricsExamples:
    """Test the example custom metrics from custom_metrics_examples.py"""

    def test_load_combined_affinity_confidence(self):
        """Test loading combined_affinity_confidence example"""
        params = Parameters()
        component = Boltz2(params)

        func = component._load_custom_function(
            "reinvent_plugins.components.boltz.custom_metrics_examples.combined_affinity_confidence"
        )

        assert func is not None
        assert callable(func)

    def test_load_ligand_efficiency_proxy(self):
        """Test loading ligand_efficiency_proxy example"""
        params = Parameters()
        component = Boltz2(params)

        func = component._load_custom_function(
            "reinvent_plugins.components.boltz.custom_metrics_examples.ligand_efficiency_proxy"
        )

        assert func is not None
        assert callable(func)

    def test_load_interface_quality_score(self):
        """Test loading interface_quality_score example"""
        params = Parameters()
        component = Boltz2(params)

        func = component._load_custom_function(
            "reinvent_plugins.components.boltz.custom_metrics_examples.interface_quality_score"
        )

        assert func is not None
        assert callable(func)

    def test_combined_affinity_confidence_with_mock_data(self):
        """Test combined_affinity_confidence with mock Boltz2 output"""
        from reinvent_plugins.components.boltz.custom_metrics_examples import (
            combined_affinity_confidence
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir) / "predictions"
            mol_id = "test_mol"
            mol_dir = pred_dir / mol_id
            mol_dir.mkdir(parents=True)

            # Create mock affinity file
            affinity_data = {
                "affinity_probability_binary": 0.92,
                "affinity_pred_value": -6.5
            }
            with open(mol_dir / f"affinity_{mol_id}.json", 'w') as f:
                json.dump(affinity_data, f)

            # Create mock confidence file
            confidence_data = {
                "ptm": 0.85,
                "iptm": 0.78,
                "plddt": [85.0, 90.0, 88.0]
            }
            with open(mol_dir / f"confidence_{mol_id}.json", 'w') as f:
                json.dump(confidence_data, f)

            # Calculate metric
            score = combined_affinity_confidence(pred_dir, mol_id, "CCO")

            # Should be affinity * ((ptm + 2*iptm) / 3)
            expected_structure_conf = (0.85 + 2 * 0.78) / 3
            expected_score = 0.92 * expected_structure_conf

            assert not np.isnan(score)
            assert np.isclose(score, expected_score, rtol=0.01)

    def test_ligand_efficiency_proxy_with_mock_data(self):
        """Test ligand_efficiency_proxy with mock Boltz2 output"""
        from reinvent_plugins.components.boltz.custom_metrics_examples import (
            ligand_efficiency_proxy
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir = Path(tmpdir) / "predictions"
            mol_id = "test_mol"
            mol_dir = pred_dir / mol_id
            mol_dir.mkdir(parents=True)

            # Create mock affinity file
            affinity_data = {
                "affinity_probability_binary": 0.85,
                "affinity_pred_value": -6.5  # log10(IC50)
            }
            with open(mol_dir / f"affinity_{mol_id}.json", 'w') as f:
                json.dump(affinity_data, f)

            # Calculate metric for ethanol (CCO)
            score = ligand_efficiency_proxy(pred_dir, mol_id, "CCO")

            assert not np.isnan(score)
            # Should be -affinity_value / (mol_weight / 1000)
            # Ethanol MW ≈ 46 Da, so expect positive value
            assert score > 0


# =============================================================================
# Test Parameter Validation
# =============================================================================

class TestCustomMetricsParameterValidation:
    """Test validation of custom metrics parameters"""

    def test_custom_metrics_length_mismatch_raises_error(self):
        """Test that mismatched function and name lists raise error"""
        params = Parameters(
            custom_metric_functions=[["func1", "func2"]],
            custom_metric_names=[["name1"]]  # Mismatch: 2 functions, 1 name
        )

        with pytest.raises(ValueError, match="must have the same length"):
            Boltz2(params)

    def test_custom_metrics_empty_lists(self):
        """Test that empty custom metrics lists work"""
        params = Parameters(
            custom_metric_functions=[[]],
            custom_metric_names=[[]]
        )

        component = Boltz2(params)
        assert component.custom_metric_functions == []
        assert component.custom_metric_names == []

    def test_custom_metrics_none_defaults(self):
        """Test default behavior with no custom metrics"""
        params = Parameters()
        component = Boltz2(params)

        assert component.custom_metric_functions == []
        assert component.custom_metric_names == []


# =============================================================================
# Test End-to-End with Custom Metrics
# =============================================================================

class TestEndToEndCustomMetrics:
    """Test end-to-end workflow with custom metrics"""

    def test_scoring_with_mocked_custom_metrics(self):
        """Test molecule scoring with mocked custom metric functions"""
        params = Parameters(
            custom_metric_functions=[["custom.metric1", "custom.metric2"]],
            custom_metric_names=[["custom1", "custom2"]],
            primary_metric=["affinity_probability_binary"],
            additional_metrics=[["custom1", "custom2"]]
        )

        component = Boltz2(params)

        # Mock custom functions
        mock_func1 = Mock(return_value=0.75)
        mock_func2 = Mock(return_value=0.82)

        def mock_loader(path):
            if "metric1" in path:
                return mock_func1
            elif "metric2" in path:
                return mock_func2
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            # Mock the entire prediction workflow
            with patch.object(component, '_create_yaml_inputs_parallel', return_value=[True, True]):
                with patch.object(component, '_run_boltz_prediction'):
                    with patch.object(component, '_load_custom_function', side_effect=mock_loader):
                        # Mock extraction methods to return valid data
                        with patch.object(component, '_extract_affinity', return_value={
                            'affinity_probability_binary': 0.90,
                            'affinity_pred_value': -6.0
                        }):
                            with patch.object(component, '_extract_structure_metrics', return_value={
                                'plddt': 85.0,
                                'ptm': 0.80,
                                'iptm': 0.75
                            }):
                                results = component(TEST_SMILES)

            # Verify results structure
            assert len(results.scores) == 1  # Primary metric
            assert len(results.scores[0]) == 2  # Two SMILES

            # Verify metadata includes custom metrics
            assert results.metadata is not None
            assert "boltz_custom1" in results.metadata
            assert "boltz_custom2" in results.metadata

            # Verify custom functions were called
            assert mock_func1.call_count == 2  # Once per SMILES
            assert mock_func2.call_count == 2

    def test_primary_metric_is_custom_metric(self):
        """Test using a custom metric as the primary optimization target"""
        params = Parameters(
            custom_metric_functions=[["custom.my_score"]],
            custom_metric_names=[["my_score"]],
            primary_metric=["my_score"],  # Custom metric as primary
            additional_metrics=[["affinity_probability_binary"]]
        )

        component = Boltz2(params)

        mock_custom_func = Mock(return_value=0.88)

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(component, '_create_yaml_inputs_parallel', return_value=[True]):
                with patch.object(component, '_run_boltz_prediction'):
                    with patch.object(component, '_load_custom_function', return_value=mock_custom_func):
                        with patch.object(component, '_extract_affinity', return_value={
                            'affinity_probability_binary': 0.85
                        }):
                            with patch.object(component, '_extract_structure_metrics', return_value={}):
                                results = component(["CCO"])

            # Primary score should be custom metric value
            assert len(results.scores[0]) == 1
            assert results.scores[0][0] == 0.88

            # Affinity should be in metadata
            assert "boltz_affinity_probability_binary" in results.metadata

    def test_multiple_custom_metrics_in_additional_metrics(self):
        """Test tracking multiple custom metrics in additional_metrics"""
        params = Parameters(
            custom_metric_functions=[[
                "custom.metric_a",
                "custom.metric_b",
                "custom.metric_c"
            ]],
            custom_metric_names=[["metric_a", "metric_b", "metric_c"]],
            primary_metric=["affinity_probability_binary"],
            additional_metrics=[["metric_a", "metric_b", "metric_c", "plddt"]]
        )

        component = Boltz2(params)

        mock_a = Mock(return_value=0.70)
        mock_b = Mock(return_value=0.80)
        mock_c = Mock(return_value=0.90)

        def mock_loader(path):
            if "metric_a" in path:
                return mock_a
            elif "metric_b" in path:
                return mock_b
            elif "metric_c" in path:
                return mock_c
            return None

        with tempfile.TemporaryDirectory() as tmpdir:
            with patch.object(component, '_create_yaml_inputs_parallel', return_value=[True]):
                with patch.object(component, '_run_boltz_prediction'):
                    with patch.object(component, '_load_custom_function', side_effect=mock_loader):
                        with patch.object(component, '_extract_affinity', return_value={
                            'affinity_probability_binary': 0.92
                        }):
                            with patch.object(component, '_extract_structure_metrics', return_value={
                                'plddt': 88.5
                            }):
                                results = component(["CCO"])

            # Verify all custom metrics in metadata
            assert "boltz_metric_a" in results.metadata
            assert "boltz_metric_b" in results.metadata
            assert "boltz_metric_c" in results.metadata
            assert results.metadata["boltz_metric_a"][0] == 0.70
            assert results.metadata["boltz_metric_b"][0] == 0.80
            assert results.metadata["boltz_metric_c"][0] == 0.90


# =============================================================================
# Test Real-World Configuration Examples
# =============================================================================

class TestRealWorldConfigurations:
    """Test realistic configuration patterns from CUSTOM_METRICS_README.md"""

    def test_ipsae_as_primary_target_configuration(self):
        """Test configuration using ipSAE as primary optimization target"""
        params = Parameters(
            receptor_sequences=[["MVHLTPEEK"]],
            receptor_chain_ids=[["A"]],
            receptor_msa_paths=[[None]],
            ligand_chain_id=["B"],
            primary_metric=["ipsae_d0res"],
            custom_metric_functions=[[
                "reinvent_plugins.components.boltz.ipsae_metrics.ipsae_d0res",
                "reinvent_plugins.components.boltz.ipsae_metrics.pdockq2"
            ]],
            custom_metric_names=[["ipsae_d0res", "pdockq2"]],
            additional_metrics=[["iptm", "plddt", "pdockq2"]]
        )

        component = Boltz2(params)

        assert component.primary_metric == "ipsae_d0res"
        assert "ipsae_d0res" in component.custom_metric_names
        assert "pdockq2" in component.custom_metric_names

    def test_combined_builtin_and_custom_metrics(self):
        """Test combining built-in Boltz2 metrics with custom metrics"""
        params = Parameters(
            primary_metric=["affinity_probability_binary"],
            custom_metric_functions=[[
                "reinvent_plugins.components.boltz.custom_metrics_examples.combined_affinity_confidence",
                "reinvent_plugins.components.boltz.custom_metrics_examples.ligand_efficiency_proxy"
            ]],
            custom_metric_names=[["combined_score", "ligand_eff"]],
            additional_metrics=[[
                "affinity_pred_value",
                "plddt",
                "ptm",
                "iptm",
                "combined_score",
                "ligand_eff"
            ]]
        )

        component = Boltz2(params)

        # Primary is built-in
        assert component.primary_metric == "affinity_probability_binary"

        # Additional includes both built-in and custom
        assert component.additional_metrics == [
            "affinity_pred_value",
            "plddt",
            "ptm",
            "iptm",
            "combined_score",
            "ligand_eff"
        ]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
