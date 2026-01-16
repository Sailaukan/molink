# Molink: RWKV-7 + SAFE Molecular Generator

Molink is a PyTorch implementation of a molecular generative model that combines the RWKV-7 "Goose" architecture with the SAFE (Sequential Attachment-based Fragment Embedding) molecular representation. It mirrors the GenMol project layout while switching the backbone to RWKV-7 for autoregressive molecule generation.

This repo focuses on the following molecular design tasks:
- De novo generation
- Scaffold decoration
- Linker design
- Motif extension
- Scaffold morphing (scaffold hopping)

## Project layout

- `configs/` Hydra configs for training
- `scripts/` CLI entry points for preprocessing, training, and sampling
- `src/molink/` model, data, and sampling code
- `data/` local datasets (optional)

## Setup

```bash
pip install -e .
```

You will need RDKit and SAFE. The `safe-mol` package provides both the SAFE tokenizer and the SMILES to SAFE converter.

## Data

Recommended dataset: `datamol-io/safe-drugs` from Hugging Face.

To convert your own SMILES file to SAFE:

```bash
python scripts/preprocess_data.py --input smiles.txt --output data/custom.safe
```

## Training

```bash
python scripts/train.py
```

The default config points at `datamol-io/safe-drugs`. For larger training, use `datamol-io/safe-gpt` or your own SAFE file.

## Sampling

```bash
python scripts/sample.py --checkpoint checkpoints/10000.ckpt --task de_novo --num-samples 8
python scripts/sample.py --checkpoint checkpoints/10000.ckpt --task scaffold_decoration --scaffold "c1ccccc1[*:1]" --num-samples 8
python scripts/sample.py --checkpoint checkpoints/10000.ckpt --task linker_design --fragment-a "[*:1]c1ccccc1" --fragment-b "[*:1]C(=O)O" --num-samples 8
```

## Notes on RWKV-7 and SAFE

- The RWKV-7 TimeMix and ChannelMix blocks follow the RWKV-LM reference (x070), with SAFE tokens as the vocabulary.
- The implementation includes a pure PyTorch WKV fallback. For speed, you can plug in the CUDA kernels from RWKV-LM.
- SAFE strings are tokenized with `SAFETokenizer.from_pretrained("datamol-io/safe-gpt")` as in GenMol.
- Optional bracket SAFE support is included to distinguish inter-fragment attachment points.

## References

- RWKV-LM: https://github.com/BlinkDL/RWKV-LM
- SAFE / SafeGPT: https://github.com/datamol-io/safe
- GenMol: https://github.com/NVIDIA-Digital-Bio/genmol
- SAFE dataset: https://huggingface.co/datasets/datamol-io/safe-drugs
# molink
