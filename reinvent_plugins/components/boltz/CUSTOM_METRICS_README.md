# Custom Metrics for Boltz2 Component

This guide explains how to add custom metric functions to the Boltz2 scoring component for REINVENT 4.

## Overview

The Boltz2 component now supports custom metric functions that can compute additional scores from Boltz2 prediction outputs. This allows you to:

- Create domain-specific scoring functions
- Combine multiple metrics in custom ways
- Extract information from structure files
- Implement custom analysis without modifying the core component

## Quick Start

### 1. Create Your Custom Metric Function

Create a Python file with your custom metric function(s):

```python
# myproject/boltz_metrics.py
from pathlib import Path
import json
import numpy as np

def my_custom_score(prediction_dir: Path, mol_id: str, smiles: str) -> float:
    """Calculate a custom score from Boltz2 outputs

    Args:
        prediction_dir: Boltz2 output directory
        mol_id: Molecule identifier
        smiles: Input SMILES string

    Returns:
        Score as float (return np.nan on failure)
    """
    try:
        # Load Boltz2 output files
        confidence_file = prediction_dir / "predictions" / mol_id / f"confidence_{mol_id}.json"

        with open(confidence_file, 'r') as f:
            data = json.load(f)

        # Compute your metric
        score = data.get("ptm", 0.0) * data.get("iptm", 0.0)

        return float(score)

    except Exception as e:
        print(f"Error in my_custom_score: {e}")
        return np.nan
```

### 2. Configure in TOML

Add your custom metrics to the Boltz2 component configuration:

```toml
[[stage.scoring.component]]
[stage.scoring.component.Boltz2]

[[stage.scoring.component.Boltz2.endpoint]]
name = "Boltz2 NR4A1 with Custom Metrics"
weight = 0.5

# ... other Boltz2 parameters ...

# Custom metrics configuration
params.custom_metric_functions = [
    "myproject.boltz_metrics.my_custom_score",
    "myproject.boltz_metrics.another_metric"
]
params.custom_metric_names = ["custom_score", "another_metric"]

# Include custom metrics in additional_metrics to track them
params.additional_metrics = ["affinity_pred_value", "plddt", "ptm", "custom_score", "another_metric"]
```

### 3. Run REINVENT

Your custom metrics will be computed automatically and available in the output CSV:

```bash
reinvent your_config.toml
```

Output columns will include:
- `boltz_custom_score`
- `boltz_another_metric`

## Function Signature

All custom metric functions must follow this signature:

```python
def metric_function(prediction_dir: Path, mol_id: str, smiles: str) -> float:
    """
    Args:
        prediction_dir: Path to Boltz2 output directory
        mol_id: Molecule identifier (InChiKey or hash)
        smiles: Input SMILES string

    Returns:
        Metric value as float (np.nan for failures)
    """
```

## Available Boltz2 Output Files

Your custom functions have access to all Boltz2 output files:

```
prediction_dir/
└── predictions/
    └── {mol_id}/
        ├── {mol_id}_model_0.cif          # Structure file (or .pdb)
        ├── confidence_{mol_id}.json       # Confidence metrics
        └── affinity_{mol_id}.json         # Affinity predictions
```

### confidence_{mol_id}.json
```json
{
  "plddt": [87.5, 89.2, ...],  // Per-residue confidence (0-100)
  "ptm": 0.85,                  // Predicted TM-score (0-1)
  "iptm": 0.78                  // Interface predicted TM-score (0-1)
}
```

### affinity_{mol_id}.json
```json
{
  "affinity_probability_binary": 0.92,  // Binding probability (0-1)
  "affinity_pred_value": -6.5           // log10(IC50)
}
```

## Example Custom Metrics

See `custom_metrics_examples.py` for complete implementations:

### 1. Binding Pocket pLDDT
Average pLDDT for residues near the binding site:

```python
from reinvent_plugins.components.boltz.custom_metrics_examples import binding_pocket_plddt

# In TOML:
params.custom_metric_functions = [
    "reinvent_plugins.components.boltz.custom_metrics_examples.binding_pocket_plddt"
]
params.custom_metric_names = ["pocket_confidence"]
```

### 2. Combined Affinity-Confidence Score
Multiply affinity probability by structure quality:

```python
from reinvent_plugins.components.boltz.custom_metrics_examples import combined_affinity_confidence

# In TOML:
params.custom_metric_functions = [
    "reinvent_plugins.components.boltz.custom_metrics_examples.combined_affinity_confidence"
]
params.custom_metric_names = ["combined_score"]
params.primary_metric = "combined_score"  # Use as optimization target
```

### 3. Ligand Efficiency Proxy
Binding affinity normalized by molecular weight:

```python
from reinvent_plugins.components.boltz.custom_metrics_examples import ligand_efficiency_proxy

# In TOML:
params.custom_metric_functions = [
    "reinvent_plugins.components.boltz.custom_metrics_examples.ligand_efficiency_proxy"
]
params.custom_metric_names = ["ligand_eff"]
```

### 4. Interface Quality Score
Combines ipTM with local structure quality:

```python
from reinvent_plugins.components.boltz.custom_metrics_examples import interface_quality_score

# In TOML:
params.custom_metric_functions = [
    "reinvent_plugins.components.boltz.custom_metrics_examples.interface_quality_score"
]
params.custom_metric_names = ["interface_quality"]
```

## Advanced Usage

### Using Multiple Custom Metrics

```toml
params.custom_metric_functions = [
    "myproject.metrics.score1",
    "myproject.metrics.score2",
    "myproject.metrics.score3"
]
params.custom_metric_names = ["metric1", "metric2", "metric3"]
params.additional_metrics = ["plddt", "ptm", "metric1", "metric2", "metric3"]
```

### Using Custom Metric as Primary Score

```toml
params.primary_metric = "my_custom_score"
params.custom_metric_functions = ["myproject.metrics.my_custom_score"]
params.custom_metric_names = ["my_custom_score"]
```

### Parsing Structure Files

If you need to analyze the structure:

```python
def analyze_structure(prediction_dir: Path, mol_id: str, smiles: str) -> float:
    """Example: parse CIF structure file"""
    try:
        from Bio.PDB import MMCIFParser

        structure_file = prediction_dir / "predictions" / mol_id / f"{mol_id}_model_0.cif"

        parser = MMCIFParser()
        structure = parser.get_structure(mol_id, structure_file)

        # Analyze structure...
        # Example: count atoms
        n_atoms = sum(1 for atom in structure.get_atoms())

        return float(n_atoms)

    except Exception as e:
        print(f"Error: {e}")
        return np.nan
```

## Error Handling

Custom functions should always:

1. **Catch all exceptions** and return `np.nan` on failure
2. **Return float values** (not int, list, etc.)
3. **Print informative error messages** for debugging

```python
def safe_custom_metric(prediction_dir: Path, mol_id: str, smiles: str) -> float:
    try:
        # Your metric computation
        value = compute_something()

        # Validate output
        if not isinstance(value, (int, float)):
            print(f"Warning: Expected numeric value, got {type(value)}")
            return np.nan

        return float(value)

    except FileNotFoundError as e:
        print(f"File not found: {e}")
        return np.nan
    except Exception as e:
        print(f"Unexpected error in safe_custom_metric: {e}")
        return np.nan
```

## Performance Considerations

1. **Cache loaded functions**: The component caches loaded custom functions
2. **Parallel-safe**: Custom functions are called once per molecule
3. **Failed predictions**: Return `np.nan` - don't raise exceptions
4. **Heavy computations**: Consider vectorizing if possible

## Troubleshooting

### Import Errors

If you get `ImportError`:
- Ensure your module is in Python path
- Use absolute imports: `myproject.metrics.func` not `metrics.func`
- Check for typos in function path

### NaN Values in Output

If all custom metrics are NaN:
- Check file paths are correct
- Verify Boltz2 predictions succeeded
- Add print statements to debug
- Check exception messages in logs

### Type Errors

If you get type-related errors:
- Ensure function returns `float`, not `int`, `list`, etc.
- Use `float()` to convert before returning
- Return `np.nan` for invalid cases

## Complete Example Configuration

```toml
[[stage.scoring.component]]
[stage.scoring.component.Boltz2]

[[stage.scoring.component.Boltz2.endpoint]]
name = "Boltz2 NR4A1 with Custom Scoring"
weight = 0.5

# Protein configuration
params.receptor_sequences = ["SEQUENCE1", "SEQUENCE2"]
params.receptor_chain_ids = ["A", "B"]
params.receptor_msa_paths = ["/path/to/msa1.a3m", "/path/to/msa2.a3m"]
params.ligand_chain_id = "C"

# Output
params.output_dir = "./boltz_output"
params.run_id = "custom_metrics_run"

# Boltz2 parameters
params.devices = 1
params.accelerator = "gpu"
params.sampling_steps_affinity = 200

# Metrics configuration
params.primary_metric = "combined_score"
params.additional_metrics = ["plddt", "ptm", "iptm", "ligand_eff"]

# Custom metrics
params.custom_metric_functions = [
    "reinvent_plugins.components.boltz.custom_metrics_examples.combined_affinity_confidence",
    "reinvent_plugins.components.boltz.custom_metrics_examples.ligand_efficiency_proxy"
]
params.custom_metric_names = ["combined_score", "ligand_eff"]
```

## Best Practices

1. **Start simple**: Test with basic metrics first
2. **Validate outputs**: Check that metrics make sense
3. **Handle edge cases**: Empty files, missing fields, etc.
4. **Document your functions**: Explain what they compute
5. **Version control**: Keep custom metrics in source control
6. **Share metrics**: Create a library of useful custom metrics

## See Also

- `custom_metrics_examples.py` - Example implementations
- `comp_boltz2.py` - Main component code
- REINVENT 4 documentation
