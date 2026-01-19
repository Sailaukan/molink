# Molink Evaluation Framework

A comprehensive evaluation framework for assessing Molink model performance across different molecular generation tasks.

## Overview

This framework provides standardized evaluation tools for five molecular generation tasks:

1. **De Novo Generation**: Unconditional molecule generation from scratch
2. **Scaffold Decoration**: Adding chemical groups to scaffold structures
3. **Linker Design**: Designing linkers between molecular fragments
4. **Scaffold Morphing**: Scaffold hopping while preserving side chains

Each task has its own evaluation module with task-specific metrics while sharing core functionality for molecular property analysis.

## Directory Structure

```
exps/
├── README.md                          # This file
├── core/                              # Shared evaluation utilities
│   ├── metrics.py                     # Molecular metrics (validity, QED, SA, etc.)
│   ├── evaluator.py                   # BaseEvaluator class
│   └── utils.py                       # Helper functions
│
├── denovo/                            # De novo generation
│   ├── eval.py                        # DeNovoEvaluator
│   ├── run.py                         # CLI script
│   └── hparams.yaml                   # Default hyperparameters
│
├── scaffold_decoration/               # Scaffold decoration
│   ├── eval.py                        # ScaffoldDecorationEvaluator
│   ├── run.py                         # CLI script
│   └── hparams.yaml                   # Default hyperparameters
│
├── linker_design/                     # Linker design
│   ├── eval.py                        # LinkerDesignEvaluator
│   ├── run.py                         # CLI script
│   └── hparams.yaml                   # Default hyperparameters
│
├── scaffold_morphing/                 # Scaffold morphing
│   ├── eval.py                        # ScaffoldMorphingEvaluator
│   ├── run.py                         # CLI script
│   └── hparams.yaml                   # Default hyperparameters
│
└── results/                           # Output directory (gitignored)
```

## Installation

### Dependencies

Install required packages:

```bash
pip install rdkit-pypi datamol safe-mol tdc scipy matplotlib seaborn pandas pyyaml
```

Or add to your requirements.txt:

```
rdkit-pypi>=2022.9.5
datamol>=0.11.0
safe-mol>=1.0.0
tdc>=0.4.0
scipy>=1.10.0
matplotlib>=3.7.0
seaborn>=0.12.0
pandas>=2.0.0
pyyaml>=6.0
```

## Usage

### De Novo Generation

Generate and evaluate molecules from scratch:

```bash
python exps/denovo/run.py \
    --checkpoint ckpt/tmp/checkpoints/10000.ckpt \
    --num-samples 10000 \
    --temperature 1.0 \
    --top-p 0.9
```

With training set for novelty calculation:

```bash
python exps/denovo/run.py \
    --checkpoint ckpt/tmp/checkpoints/10000.ckpt \
    --num-samples 10000 \
    --train-set-path data/train.txt
```

### Scaffold Decoration

Evaluate scaffold decoration with a single scaffold:

```bash
python exps/scaffold_decoration/run.py \
    --checkpoint ckpt/tmp/checkpoints/10000.ckpt \
    --scaffold "c1ccccc1" \
    --num-samples-per-scaffold 100
```

With multiple scaffolds from a file:

```bash
python exps/scaffold_decoration/run.py \
    --checkpoint ckpt/tmp/checkpoints/10000.ckpt \
    --scaffolds-file data/test_scaffolds.txt \
    --num-samples-per-scaffold 100
```

### Linker Design

Evaluate linker design with a single fragment pair:

```bash
python exps/linker_design/run.py \
    --checkpoint ckpt/tmp/checkpoints/10000.ckpt \
    --fragment-a "c1ccccc1" \
    --fragment-b "CC(C)C" \
    --num-samples-per-pair 100
```

With multiple fragment pairs from a file:

```bash
python exps/linker_design/run.py \
    --checkpoint ckpt/tmp/checkpoints/10000.ckpt \
    --fragments-file data/test_fragment_pairs.txt \
    --num-samples-per-pair 100
```

Fragment file format (one pair per line):
```
c1ccccc1,CC(C)C
c1ccncc1,c1ccc(O)cc1
...
```

### Scaffold Morphing

Evaluate scaffold morphing with side chains:

```bash
python exps/scaffold_morphing/run.py \
    --checkpoint ckpt/tmp/checkpoints/10000.ckpt \
    --side-chains "C*" "CC*" \
    --num-samples-per-example 100
```

With morphing examples from a JSON file:

```bash
python exps/scaffold_morphing/run.py \
    --checkpoint ckpt/tmp/checkpoints/10000.ckpt \
    --examples-file data/morphing_examples.json \
    --num-samples-per-example 100
```

Morphing examples file format:
```json
[
  {
    "mol": "CCc1ccccc1CC",
    "core": "c1ccccc1",
    "side_chains": ["C*", "CC*"]
  },
  ...
]
```

## Common Command-Line Arguments

All evaluation scripts support the following arguments:

### Required
- `--checkpoint`: Path to model checkpoint

### Generation Parameters
- `--num-samples`: Number of molecules to generate (de novo)
- `--num-samples-per-scaffold`: Samples per scaffold (scaffold decoration)
- `--num-samples-per-pair`: Samples per fragment pair (linker design)
- `--num-samples-per-example`: Samples per example (scaffold morphing)
- `--batch-size`: Batch size for generation (default: 128, de novo only)
- `--max-new-tokens`: Maximum tokens per molecule (default: 200)
- `--temperature`: Sampling temperature (default: 1.0)
- `--top-p`: Nucleus sampling parameter (default: 0.9)
- `--top-k`: Top-k sampling parameter (default: 0)

### Output Parameters
- `--output-dir`: Base output directory (default: exps/results)
- `--experiment-name`: Custom experiment name (default: timestamp)
- `--no-save-molecules`: Don't save generated molecules
- `--no-save-plots`: Don't generate property distribution plots

### Other
- `--device`: Device to use (cuda/cpu, default: auto-detect)
- `--hparams`: Path to hyperparameters YAML file

## Metrics

### Basic Metrics

- **Validity**: Fraction of valid SMILES strings
- **Uniqueness**: Fraction of unique molecules among valid ones
- **Diversity**: Average pairwise Tanimoto distance (Morgan fingerprints)
- **Novelty**: Fraction of molecules not in training set (if provided)

### Chemical Properties

Statistics (mean, std, median, min, max, percentiles) for:

- **QED**: Quantitative Estimate of Drug-likeness (0-1, higher is better)
- **SA**: Synthetic Accessibility (1-10, lower is better)
- **LogP**: Partition coefficient (typically -2 to 5 for drug-like molecules)
- **MW**: Molecular weight (typically 200-500 for drug-like molecules)
- **TPSA**: Topological Polar Surface Area
- **Rotatable Bonds**: Number of rotatable bonds
- **H-Bond Donors**: Number of hydrogen bond donors
- **H-Bond Acceptors**: Number of hydrogen bond acceptors

### Molecular Filters

Pass rates for:

- **Lipinski's Rule of Five**: MW ≤ 500, LogP ≤ 5, HBD ≤ 5, HBA ≤ 10
- **PAINS**: Pan-Assay Interference compounds filter
- **Drug-like**: Passes both Lipinski and PAINS filters

### Task-Specific Metrics

#### Scaffold Decoration
- **Preservation Rate**: Fraction of molecules containing the scaffold

#### Linker Design
- **Fragment Preservation**: Fraction containing both fragments

#### Scaffold Morphing
- **Side Chain Preservation**: Fraction preserving side chains
- **Scaffold Divergence**: Tanimoto distance from original scaffold

## Output Format

Each evaluation creates a timestamped directory:

```
exps/results/{task}/{timestamp}/
├── config.yaml              # Experiment configuration
├── molecules.txt            # Generated SMILES (one per line)
├── molecules.csv            # SMILES with properties
├── metrics.json             # All computed metrics (JSON)
├── summary.txt              # Human-readable summary
├── run.log                  # Execution log
└── plots/                   # Property distribution plots
    ├── qed_distribution.png
    ├── sa_distribution.png
    ├── logp_distribution.png
    ├── mw_distribution.png
    └── tpsa_distribution.png
```

### metrics.json Structure

```json
{
  "metadata": {
    "timestamp": "2026-01-19T15:30:45",
    "checkpoint": "ckpt/tmp/checkpoints/10000.ckpt",
    "task": "denovo",
    "num_generated": 9850
  },
  "generation_params": {
    "num_samples": 10000,
    "temperature": 1.0,
    "top_p": 0.9
  },
  "basic_metrics": {
    "validity": 0.985,
    "uniqueness": 0.963,
    "diversity": 0.872,
    "novelty": 0.945
  },
  "chemical_properties": {
    "qed": {
      "mean": 0.653,
      "std": 0.182,
      "median": 0.671,
      "min": 0.102,
      "max": 0.948,
      "p25": 0.534,
      "p75": 0.782
    },
    "sa": {
      "mean": 3.21,
      "std": 0.89,
      "median": 3.15,
      ...
    },
    ...
  },
  "filters": {
    "lipinski": 0.823,
    "pains": 0.971,
    "drug_like": 0.815
  }
}
```

## Benchmark Comparison

Compare multiple checkpoints:

```bash
for step in 5000 10000 15000 20000; do
    python exps/denovo/run.py \
        --checkpoint ckpt/tmp/checkpoints/${step}.ckpt \
        --num-samples 5000 \
        --experiment-name step${step}
done
```

Then compare results from:
- `exps/results/denovo/step5000/metrics.json`
- `exps/results/denovo/step10000/metrics.json`
- `exps/results/denovo/step15000/metrics.json`
- `exps/results/denovo/step20000/metrics.json`

## Using Hyperparameter Files

Create a custom hyperparameter file:

```yaml
# my_config.yaml
generation:
  num_samples: 5000
  temperature: 0.8
  top_p: 0.95

evaluation:
  save_molecules: true
  save_plots: true
```

Run with custom config:

```bash
python exps/denovo/run.py \
    --checkpoint ckpt/tmp/checkpoints/10000.ckpt \
    --hparams my_config.yaml
```

## Extending the Framework

### Adding Custom Metrics

Edit `exps/core/metrics.py`:

```python
def my_custom_metric(smiles: str) -> float:
    """Calculate a custom molecular property."""
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None
    # Your calculation here
    return value
```

### Creating Custom Evaluators

Subclass `BaseEvaluator` in `exps/core/evaluator.py`:

```python
from exps.core.evaluator import BaseEvaluator

class MyCustomEvaluator(BaseEvaluator):
    def run_evaluation(self, **kwargs):
        # Your custom evaluation logic
        molecules = self.generate(...)
        metrics = self.evaluate_basic_metrics(molecules)
        # Add custom metrics
        return metrics
```

## Troubleshooting

### SA Score Not Available

If `rdkit.Chem.SA_Score` is not available, the framework falls back to a simple heuristic. Install the full RDKit package for accurate SA scores.

### Memory Issues

For large-scale generation:
- Reduce `--batch-size`
- Generate in smaller chunks
- Use `--no-save-plots` to skip plot generation

### CUDA Out of Memory

- Reduce `--batch-size`
- Use `--device cpu` to run on CPU (slower but no memory limit)

## Citation

If you use this evaluation framework, please cite:

```bibtex
@software{molink_eval_2026,
  title={Molink Evaluation Framework},
  author={Your Name},
  year={2026},
  url={https://github.com/yourrepo/molink}
}
```

## License

Same license as the main Molink project.
