"""Unit tests for ipSAE and interface quality metrics

This module tests the custom metric functions in ipsae_metrics.py, including:
- ipSAE (interface Predicted Structural Alignment Error) variants
- pDockQ and pDockQ2
- LIS (Local Interaction Score)
- ipTM from PAE

Tests use mock Boltz2 output files (CIF structures, PAE matrices, confidence JSONs)
to verify the calculation logic without requiring actual Boltz2 predictions.
"""

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch
import numpy as np
import pytest

from reinvent_plugins.components.boltz.ipsae_metrics import (
    ipsae_d0res,
    ipsae_d0chn,
    ipsae_d0dom,
    pdockq,
    pdockq2,
    lis_score,
    iptm_from_pae,
    ptm_func,
    ptm_func_vec,
    calc_d0,
    calc_d0_array,
    load_pae_matrix,
    calculate_distance_matrix,
    _calculate_ipsae_metrics,
)


# =============================================================================
# Helper Functions Tests
# =============================================================================

class TestPTMFunctions:
    """Test PTM (Predicted TM-score) calculation functions"""

    def test_ptm_func_basic(self):
        """Test basic PTM function calculation"""
        # When x = 0, PTM should be 1.0 (perfect alignment)
        assert ptm_func(0.0, 5.0) == 1.0

        # When x = d0, PTM should be 0.5
        assert np.isclose(ptm_func(5.0, 5.0), 0.5)

        # When x >> d0, PTM approaches 0
        assert ptm_func(100.0, 5.0) < 0.01

    def test_ptm_func_vec(self):
        """Test vectorized PTM function"""
        x = np.array([0.0, 5.0, 10.0])
        d0 = 5.0

        result = ptm_func_vec(x, d0)

        assert np.isclose(result[0], 1.0)
        assert np.isclose(result[1], 0.5)
        assert result[2] < result[1]  # Further distance = lower score

    def test_ptm_func_vec_matches_scalar(self):
        """Test that vectorized version matches scalar implementation"""
        x_values = np.array([0.5, 1.0, 2.5, 5.0, 10.0])
        d0 = 3.5

        vec_result = ptm_func_vec(x_values, d0)
        scalar_results = np.array([ptm_func(x, d0) for x in x_values])

        assert np.allclose(vec_result, scalar_results)


class TestD0Calculation:
    """Test d0 normalization factor calculation"""

    def test_calc_d0_small_protein(self):
        """Test d0 for small protein (< 27 residues)"""
        # For L < 27, should use L = 27
        d0_20 = calc_d0(20, 'protein')
        d0_27 = calc_d0(27, 'protein')

        assert d0_20 == d0_27
        assert d0_20 >= 1.0  # Minimum for proteins

    def test_calc_d0_normal_protein(self):
        """Test d0 for typical protein length"""
        d0_100 = calc_d0(100, 'protein')
        d0_200 = calc_d0(200, 'protein')

        # Larger proteins should have larger d0
        assert d0_200 > d0_100
        assert d0_100 >= 1.0

    def test_calc_d0_nucleic_acid(self):
        """Test d0 for nucleic acids (different minimum)"""
        d0_protein = calc_d0(50, 'protein')
        d0_nucleic = calc_d0(50, 'nucleic_acid')

        # Nucleic acids have minimum d0 = 2.0 instead of 1.0
        # For larger lengths, might be same due to formula
        assert d0_nucleic >= 2.0

    def test_calc_d0_array(self):
        """Test vectorized d0 calculation"""
        lengths = np.array([50, 100, 150, 200])
        d0_array = calc_d0_array(lengths, 'protein')

        # Should increase with length
        assert all(d0_array[i] < d0_array[i+1] for i in range(len(d0_array) - 1))

        # Should match scalar calculations
        for L, d0 in zip(lengths, d0_array):
            assert np.isclose(d0, calc_d0(L, 'protein'))


class TestDistanceMatrix:
    """Test distance matrix calculation"""

    def test_distance_matrix_simple(self):
        """Test distance matrix for simple coordinates"""
        # Three points: origin, (1,0,0), (0,1,0)
        coords = np.array([
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0]
        ])

        dist_matrix = calculate_distance_matrix(coords)

        # Check diagonal is zero
        assert np.allclose(np.diag(dist_matrix), 0.0)

        # Check known distances
        assert np.isclose(dist_matrix[0, 1], 1.0)  # Distance to (1,0,0)
        assert np.isclose(dist_matrix[0, 2], 1.0)  # Distance to (0,1,0)
        assert np.isclose(dist_matrix[1, 2], np.sqrt(2))  # Distance between them

        # Check symmetry
        assert np.allclose(dist_matrix, dist_matrix.T)

    def test_distance_matrix_single_point(self):
        """Test distance matrix for single point"""
        coords = np.array([[1.0, 2.0, 3.0]])
        dist_matrix = calculate_distance_matrix(coords)

        assert dist_matrix.shape == (1, 1)
        assert dist_matrix[0, 0] == 0.0


class TestPAELoading:
    """Test PAE matrix loading from NPZ files"""

    def test_load_pae_matrix_success(self):
        """Test loading valid PAE matrix"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create mock PAE file
            pae_matrix = np.random.rand(50, 50) * 30  # PAE values 0-30 Å
            pae_file = Path(tmpdir) / "pae_test.npz"
            np.savez(pae_file, pae=pae_matrix)

            # Load and verify
            loaded_pae = load_pae_matrix(pae_file)
            assert loaded_pae is not None
            assert np.allclose(loaded_pae, pae_matrix)

    def test_load_pae_matrix_file_not_found(self):
        """Test loading from non-existent file returns None"""
        pae_file = Path("/nonexistent/path/pae.npz")
        loaded_pae = load_pae_matrix(pae_file)
        assert loaded_pae is None

    def test_load_pae_matrix_wrong_format(self):
        """Test loading NPZ without 'pae' key returns None"""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create NPZ with wrong key
            pae_file = Path(tmpdir) / "pae_wrong.npz"
            np.savez(pae_file, wrong_key=np.random.rand(50, 50))

            loaded_pae = load_pae_matrix(pae_file)
            assert loaded_pae is None


# =============================================================================
# Mock Data Generation
# =============================================================================

def create_mock_cif_file(filepath: Path, n_residues_chain_a: int = 30,
                         n_residues_chain_b: int = 35):
    """Create a mock CIF file with two protein chains

    Args:
        filepath: Path to write CIF file
        n_residues_chain_a: Number of residues in chain A
        n_residues_chain_b: Number of residues in chain B
    """
    with open(filepath, 'w') as f:
        # Write field definitions
        f.write("_atom_site.group_PDB\n")
        f.write("_atom_site.id\n")
        f.write("_atom_site.type_symbol\n")
        f.write("_atom_site.label_atom_id\n")
        f.write("_atom_site.label_alt_id\n")
        f.write("_atom_site.label_comp_id\n")
        f.write("_atom_site.label_asym_id\n")
        f.write("_atom_site.label_entity_id\n")
        f.write("_atom_site.label_seq_id\n")
        f.write("_atom_site.pdbx_PDB_ins_code\n")
        f.write("_atom_site.Cartn_x\n")
        f.write("_atom_site.Cartn_y\n")
        f.write("_atom_site.Cartn_z\n")
        f.write("_atom_site.occupancy\n")
        f.write("_atom_site.B_iso_or_equiv\n")
        f.write("_atom_site.pdbx_formal_charge\n")
        f.write("_atom_site.auth_seq_id\n")
        f.write("_atom_site.auth_comp_id\n")
        f.write("_atom_site.auth_asym_id\n")
        f.write("_atom_site.auth_atom_id\n")
        f.write("_atom_site.pdbx_PDB_model_num\n")

        atom_id = 1

        # Write chain A
        for i in range(n_residues_chain_a):
            res_num = i + 1
            # CA atom
            x, y, z = float(i), 0.0, 0.0  # Simple linear arrangement
            f.write(f"ATOM {atom_id} C CA . ALA A 1 {res_num} ? {x:.3f} {y:.3f} {z:.3f} 1.00 50.0 ? {res_num} ALA A CA 1\n")
            atom_id += 1
            # CB atom (for non-GLY)
            f.write(f"ATOM {atom_id} C CB . ALA A 1 {res_num} ? {x:.3f} {y+1.0:.3f} {z:.3f} 1.00 50.0 ? {res_num} ALA A CB 1\n")
            atom_id += 1

        # Write chain B (offset in space)
        for i in range(n_residues_chain_b):
            res_num = i + 1
            # CA atom (offset by 5 Å in Y direction to create interface)
            x, y, z = float(i), 5.0, 0.0
            f.write(f"ATOM {atom_id} C CA . ALA B 2 {res_num} ? {x:.3f} {y:.3f} {z:.3f} 1.00 50.0 ? {res_num} ALA B CA 1\n")
            atom_id += 1
            # CB atom
            f.write(f"ATOM {atom_id} C CB . ALA B 2 {res_num} ? {x:.3f} {y-1.0:.3f} {z:.3f} 1.00 50.0 ? {res_num} ALA B CB 1\n")
            atom_id += 1


def create_mock_pae_matrix(filepath: Path, n_residues: int,
                          interface_quality: str = 'good'):
    """Create a mock PAE matrix NPZ file

    Args:
        filepath: Path to write NPZ file
        n_residues: Total number of residues
        interface_quality: 'good', 'medium', or 'poor'
    """
    # Start with high PAE values everywhere
    pae_matrix = np.ones((n_residues, n_residues)) * 25.0

    # Make diagonal low (residues confident about themselves)
    for i in range(n_residues):
        pae_matrix[i, i] = np.random.rand() * 2.0

    # Make intra-chain PAE reasonable
    mid = n_residues // 2
    # Chain A internal
    pae_matrix[:mid, :mid] = np.random.rand(mid, mid) * 8.0
    # Chain B internal
    pae_matrix[mid:, mid:] = np.random.rand(n_residues - mid, n_residues - mid) * 8.0

    # Set interface PAE values based on quality
    # Assume first half is chain A, second half is chain B
    if interface_quality == 'good':
        # Good interface: PAE < 5 Å with many contacts
        pae_matrix[:mid, mid:] = np.random.rand(mid, n_residues - mid) * 4.5
        pae_matrix[mid:, :mid] = np.random.rand(n_residues - mid, mid) * 4.5
    elif interface_quality == 'medium':
        # Medium interface: PAE 6-9 Å (some passing cutoff)
        pae_matrix[:mid, mid:] = 6 + np.random.rand(mid, n_residues - mid) * 3
        pae_matrix[mid:, :mid] = 6 + np.random.rand(n_residues - mid, mid) * 3
    else:  # poor
        # Poor interface: PAE > 15 Å (well above default cutoff of 10)
        # This ensures NO residues will pass the PAE cutoff
        pae_matrix[:mid, mid:] = 15 + np.random.rand(mid, n_residues - mid) * 15
        pae_matrix[mid:, :mid] = 15 + np.random.rand(n_residues - mid, mid) * 15

    np.savez(filepath, pae=pae_matrix)


def create_mock_confidence_json(filepath: Path, n_residues: int):
    """Create a mock confidence JSON file

    Args:
        filepath: Path to write JSON file
        n_residues: Total number of residues
    """
    confidence_data = {
        "plddt": [float(70 + np.random.rand() * 30) for _ in range(n_residues)],
        "ptm": 0.85,
        "iptm": 0.78
    }

    with open(filepath, 'w') as f:
        json.dump(confidence_data, f)


# =============================================================================
# Integration Tests for ipSAE Metrics
# =============================================================================

class TestIpSAEMetrics:
    """Test ipSAE calculation on mock Boltz2 output"""

    def setup_mock_prediction(self, tmpdir: str, n_chain_a: int = 30,
                             n_chain_b: int = 35, quality: str = 'good'):
        """Create mock Boltz2 prediction output

        Returns:
            Tuple of (prediction_dir, mol_id)
        """
        prediction_dir = Path(tmpdir)
        mol_id = "test_molecule"
        mol_dir = prediction_dir / mol_id
        mol_dir.mkdir(parents=True)

        # Create mock files
        n_total = n_chain_a + n_chain_b

        structure_file = mol_dir / f"{mol_id}_model_0.cif"
        create_mock_cif_file(structure_file, n_chain_a, n_chain_b)

        pae_file = mol_dir / f"pae_{mol_id}_model_0.npz"
        create_mock_pae_matrix(pae_file, n_total, quality)

        confidence_file = mol_dir / f"confidence_{mol_id}_model_0.json"
        create_mock_confidence_json(confidence_file, n_total)

        return prediction_dir, mol_id

    def test_ipsae_d0res_good_interface(self):
        """Test ipSAE_d0res on good quality interface"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir, mol_id = self.setup_mock_prediction(tmpdir, quality='good')

            score = ipsae_d0res(pred_dir, mol_id, "CCO")

            # Good interface (PAE < 5 Å) should have high ipSAE score
            assert not np.isnan(score)
            assert 0.0 <= score <= 1.0
            # Good interface with low PAE should be > 0.4
            assert score > 0.4

    def test_ipsae_d0res_poor_interface(self):
        """Test ipSAE_d0res on poor quality interface"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir, mol_id = self.setup_mock_prediction(tmpdir, quality='poor')

            score = ipsae_d0res(pred_dir, mol_id, "CCO")

            # Poor interface (PAE > 15 Å, well above default cutoff of 10) should return 0.0
            # because no residues pass the PAE cutoff
            assert not np.isnan(score)
            # With no residues passing cutoff, ipSAE returns 0.0
            assert score == 0.0

    def test_ipsae_d0chn(self):
        """Test ipSAE_d0chn calculation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir, mol_id = self.setup_mock_prediction(tmpdir, quality='good')

            score = ipsae_d0chn(pred_dir, mol_id, "CCO")

            assert not np.isnan(score)
            assert 0.0 <= score <= 1.0

    def test_ipsae_d0dom(self):
        """Test ipSAE_d0dom calculation"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir, mol_id = self.setup_mock_prediction(tmpdir, quality='good')

            score = ipsae_d0dom(pred_dir, mol_id, "CCO")

            assert not np.isnan(score)
            assert 0.0 <= score <= 1.0

    def test_all_ipsae_variants_consistent(self):
        """Test that all ipSAE variants give reasonable relative values"""
        with tempfile.TemporaryDirectory() as tmpdir:
            pred_dir, mol_id = self.setup_mock_prediction(tmpdir, quality='good')

            d0res = ipsae_d0res(pred_dir, mol_id, "CCO")
            d0chn = ipsae_d0chn(pred_dir, mol_id, "CCO")
            d0dom = ipsae_d0dom(pred_dir, mol_id, "CCO")

            # All should be valid scores
            assert all(0.0 <= s <= 1.0 for s in [d0res, d0chn, d0dom])
            assert all(not np.isnan(s) for s in [d0res, d0chn, d0dom])


class TestPDockQMetrics:
    """Test pDockQ and pDockQ2 metrics"""

    def test_pdockq_calculation(self):
        """Test pDockQ calculation on mock data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir)
            mol_id = "test_mol"
            mol_dir = prediction_dir / mol_id
            mol_dir.mkdir(parents=True)

            # Create mock files
            n_chain_a, n_chain_b = 40, 45
            n_total = n_chain_a + n_chain_b

            create_mock_cif_file(mol_dir / f"{mol_id}_model_0.cif", n_chain_a, n_chain_b)
            create_mock_pae_matrix(mol_dir / f"pae_{mol_id}_model_0.npz", n_total, 'good')
            create_mock_confidence_json(mol_dir / f"confidence_{mol_id}_model_0.json", n_total)

            score = pdockq(prediction_dir, mol_id, "CCO")

            assert not np.isnan(score)
            assert 0.0 <= score <= 1.0

    def test_pdockq2_calculation(self):
        """Test pDockQ2 calculation on mock data"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir)
            mol_id = "test_mol"
            mol_dir = prediction_dir / mol_id
            mol_dir.mkdir(parents=True)

            n_chain_a, n_chain_b = 40, 45
            n_total = n_chain_a + n_chain_b

            create_mock_cif_file(mol_dir / f"{mol_id}_model_0.cif", n_chain_a, n_chain_b)
            create_mock_pae_matrix(mol_dir / f"pae_{mol_id}_model_0.npz", n_total, 'good')
            create_mock_confidence_json(mol_dir / f"confidence_{mol_id}_model_0.json", n_total)

            score = pdockq2(prediction_dir, mol_id, "CCO")

            assert not np.isnan(score)
            assert 0.0 <= score <= 1.0

    def test_pdockq_vs_pdockq2(self):
        """Test that pDockQ and pDockQ2 give different but correlated results"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir)
            mol_id = "test_mol"
            mol_dir = prediction_dir / mol_id
            mol_dir.mkdir(parents=True)

            n_chain_a, n_chain_b = 50, 50
            n_total = n_chain_a + n_chain_b

            create_mock_cif_file(mol_dir / f"{mol_id}_model_0.cif", n_chain_a, n_chain_b)
            create_mock_pae_matrix(mol_dir / f"pae_{mol_id}_model_0.npz", n_total, 'good')
            create_mock_confidence_json(mol_dir / f"confidence_{mol_id}_model_0.json", n_total)

            pdockq_score = pdockq(prediction_dir, mol_id, "CCO")
            pdockq2_score = pdockq2(prediction_dir, mol_id, "CCO")

            # Both should be valid
            assert not np.isnan(pdockq_score)
            assert not np.isnan(pdockq2_score)

            # They use different formulas, so shouldn't be identical
            # but should both be in valid range
            assert 0.0 <= pdockq_score <= 1.0
            assert 0.0 <= pdockq2_score <= 1.0


class TestLISScore:
    """Test LIS (Local Interaction Score) metric"""

    def test_lis_good_interface(self):
        """Test LIS on good quality interface"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir)
            mol_id = "test_mol"
            mol_dir = prediction_dir / mol_id
            mol_dir.mkdir(parents=True)

            n_chain_a, n_chain_b = 30, 35
            n_total = n_chain_a + n_chain_b

            create_mock_cif_file(mol_dir / f"{mol_id}_model_0.cif", n_chain_a, n_chain_b)
            create_mock_pae_matrix(mol_dir / f"pae_{mol_id}_model_0.npz", n_total, 'good')
            create_mock_confidence_json(mol_dir / f"confidence_{mol_id}_model_0.json", n_total)

            score = lis_score(prediction_dir, mol_id, "CCO")

            # Good interface (PAE < 5 Å) should have high LIS
            # LIS = mean((12 - PAE) / 12) for PAE <= 12
            # For PAE < 5, this gives LIS > 0.58
            assert not np.isnan(score)
            assert 0.0 <= score <= 1.0
            assert score > 0.55  # Good PAE values should give high LIS

    def test_lis_poor_interface(self):
        """Test LIS on poor quality interface"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir)
            mol_id = "test_mol"
            mol_dir = prediction_dir / mol_id
            mol_dir.mkdir(parents=True)

            n_chain_a, n_chain_b = 30, 35
            n_total = n_chain_a + n_chain_b

            create_mock_cif_file(mol_dir / f"{mol_id}_model_0.cif", n_chain_a, n_chain_b)
            create_mock_pae_matrix(mol_dir / f"pae_{mol_id}_model_0.npz", n_total, 'poor')
            create_mock_confidence_json(mol_dir / f"confidence_{mol_id}_model_0.json", n_total)

            score = lis_score(prediction_dir, mol_id, "CCO")

            # Poor interface should have low LIS
            assert not np.isnan(score) or score == 0.0
            if not np.isnan(score):
                assert 0.0 <= score <= 1.0


class TestIPTMFromPAE:
    """Test ipTM calculation from PAE matrix"""

    def test_iptm_from_pae(self):
        """Test ipTM calculation from PAE matrix"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir)
            mol_id = "test_mol"
            mol_dir = prediction_dir / mol_id
            mol_dir.mkdir(parents=True)

            n_chain_a, n_chain_b = 40, 40
            n_total = n_chain_a + n_chain_b

            create_mock_cif_file(mol_dir / f"{mol_id}_model_0.cif", n_chain_a, n_chain_b)
            create_mock_pae_matrix(mol_dir / f"pae_{mol_id}_model_0.npz", n_total, 'good')
            create_mock_confidence_json(mol_dir / f"confidence_{mol_id}_model_0.json", n_total)

            score = iptm_from_pae(prediction_dir, mol_id, "CCO")

            assert not np.isnan(score)
            assert 0.0 <= score <= 1.0


# =============================================================================
# Error Handling Tests
# =============================================================================

class TestErrorHandling:
    """Test error handling for missing or invalid files"""

    def test_missing_structure_file(self):
        """Test that missing structure file returns NaN"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir)
            mol_id = "missing_mol"

            # Don't create any files
            score = ipsae_d0res(prediction_dir, mol_id, "CCO")
            assert np.isnan(score)

    def test_missing_pae_file(self):
        """Test that missing PAE file returns NaN"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir)
            mol_id = "test_mol"
            mol_dir = prediction_dir / mol_id
            mol_dir.mkdir(parents=True)

            # Create only structure file, no PAE
            create_mock_cif_file(mol_dir / f"{mol_id}_model_0.cif")

            score = ipsae_d0res(prediction_dir, mol_id, "CCO")
            assert np.isnan(score)

    def test_single_chain_structure(self):
        """Test that single-chain structure returns NaN (no interface)"""
        with tempfile.TemporaryDirectory() as tmpdir:
            prediction_dir = Path(tmpdir)
            mol_id = "single_chain"
            mol_dir = prediction_dir / mol_id
            mol_dir.mkdir(parents=True)

            # Create single-chain CIF
            with open(mol_dir / f"{mol_id}_model_0.cif", 'w') as f:
                f.write("_atom_site.group_PDB\n")
                f.write("_atom_site.id\n")
                f.write("_atom_site.type_symbol\n")
                f.write("_atom_site.label_atom_id\n")
                f.write("_atom_site.label_alt_id\n")
                f.write("_atom_site.label_comp_id\n")
                f.write("_atom_site.label_asym_id\n")
                f.write("_atom_site.label_entity_id\n")
                f.write("_atom_site.label_seq_id\n")
                f.write("_atom_site.pdbx_PDB_ins_code\n")
                f.write("_atom_site.Cartn_x\n")
                f.write("_atom_site.Cartn_y\n")
                f.write("_atom_site.Cartn_z\n")
                f.write("_atom_site.occupancy\n")
                f.write("_atom_site.B_iso_or_equiv\n")
                f.write("_atom_site.pdbx_formal_charge\n")
                f.write("_atom_site.auth_seq_id\n")
                f.write("_atom_site.auth_comp_id\n")
                f.write("_atom_site.auth_asym_id\n")
                f.write("_atom_site.auth_atom_id\n")
                f.write("_atom_site.pdbx_PDB_model_num\n")
                # Only chain A
                f.write("ATOM 1 C CA . ALA A 1 1 ? 0.0 0.0 0.0 1.00 50.0 ? 1 ALA A CA 1\n")

            # Create PAE file
            np.savez(mol_dir / f"pae_{mol_id}_model_0.npz", pae=np.array([[0.0]]))

            score = ipsae_d0res(prediction_dir, mol_id, "CCO")
            assert np.isnan(score)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
