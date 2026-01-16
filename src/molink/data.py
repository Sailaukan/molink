import logging
import os
from typing import Dict, Iterable, Optional

import datasets
import torch
from rdkit import RDLogger
from safe.tokenizer import SAFETokenizer
import safe as sf

from molink.utils.bracket_safe import safe2bracketsafe

RDLogger.DisableLog("rdApp.*")
logger = logging.getLogger(__name__)


def get_tokenizer(config) -> "transformers.PreTrainedTokenizerFast":
    tk = SAFETokenizer.from_pretrained(config.data.tokenizer_name).get_pretrained()
    if config.training.get("use_bracket_safe"):
        tk.add_tokens(["<", ">"])
    return tk


def _extract_safe(example: Dict, safe_field: Optional[str], smiles_field: Optional[str], encoder):
    """Extract SAFE string from example, trying multiple common field names."""
    try:
        # Try explicit safe field first
        if safe_field and safe_field in example:
            return example[safe_field]

        # Try common SAFE field names
        for key in ["safe", "SAFE", "safe_str", "safe_string", "sf"]:
            if key in example:
                return example[key]

        # Try to convert from SMILES
        if smiles_field and smiles_field in example:
            smiles = example[smiles_field]
            if smiles:
                return encoder.encoder(smiles, allow_empty=True)

        # Try common SMILES field names
        for key in ["smiles", "SMILES", "canonical_smiles"]:
            if key in example:
                smiles = example[key]
                if smiles:
                    return encoder.encoder(smiles, allow_empty=True)

    except Exception as e:
        logger.warning(f"Failed to extract SAFE from example: {e}")

    return None


class SafeIterableDataset(torch.utils.data.IterableDataset):
    def __init__(self, dataset: Iterable, safe_field: Optional[str], smiles_field: Optional[str]):
        super().__init__()
        self.dataset = dataset
        self.safe_field = safe_field
        self.smiles_field = smiles_field
        self.encoder = sf.SAFEConverter(ignore_stereo=True)

    def __iter__(self):
        for example in self.dataset:
            safe_str = _extract_safe(example, self.safe_field, self.smiles_field, self.encoder)
            if safe_str is None:
                continue
            yield {"safe": safe_str}


class SafeMapDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, safe_field: Optional[str], smiles_field: Optional[str]):
        self.dataset = dataset
        self.safe_field = safe_field
        self.smiles_field = smiles_field
        self.encoder = sf.SAFEConverter(ignore_stereo=True)

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        example = self.dataset[idx]
        safe_str = _extract_safe(example, self.safe_field, self.smiles_field, self.encoder)
        if safe_str is None:
            return {"safe": ""}
        return {"safe": safe_str}


class SafeFileDataset(torch.utils.data.Dataset):
    def __init__(self, path: str, file_format: str = "safe"):
        logger.info(f"Loading data from file: {path} (format: {file_format})")
        try:
            with open(path, 'r', encoding='utf-8') as f:
                self.items = [line.strip() for line in f if line.strip()]
            logger.info(f"Loaded {len(self.items)} examples from {path}")
        except FileNotFoundError:
            logger.error(f"File not found: {path}")
            raise
        except Exception as e:
            logger.error(f"Error reading file {path}: {e}")
            raise

        self.file_format = file_format
        self.encoder = sf.SAFEConverter(ignore_stereo=True)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        item = self.items[idx]
        try:
            if self.file_format == "safe":
                return {"safe": item}
            else:
                safe_str = self.encoder.encoder(item, allow_empty=True)
                if safe_str:
                    return {"safe": safe_str}
                else:
                    logger.warning(f"Failed to encode item at index {idx}: {item[:50]}...")
                    return {"safe": ""}
        except Exception as e:
            logger.warning(f"Error processing item at index {idx}: {e}")
            return {"safe": ""}


class SafeCollator:
    def __init__(self, tokenizer, max_length: int, use_bracket_safe: bool = False):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.use_bracket_safe = use_bracket_safe

    def __call__(self, examples):
        # Filter out empty or None examples
        safe_list = [example["safe"] for example in examples if example.get("safe")]

        # If all examples are invalid, return a dummy batch
        if not safe_list:
            logger.warning("Batch contains no valid examples, returning dummy batch")
            dummy_batch = self.tokenizer(
                [""],
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
            if "token_type_ids" in dummy_batch:
                del dummy_batch["token_type_ids"]
            dummy_batch["labels"] = dummy_batch["input_ids"].clone()
            return dummy_batch

        # Convert to bracket SAFE if needed
        if self.use_bracket_safe:
            try:
                safe_list = [safe2bracketsafe(s) for s in safe_list]
            except Exception as e:
                logger.warning(f"Error converting to bracket SAFE: {e}")

        try:
            batch = self.tokenizer(
                safe_list,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            )
        except Exception as e:
            logger.error(f"Tokenization error: {e}")
            raise

        if "token_type_ids" in batch:
            del batch["token_type_ids"]

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return batch


def get_dataloader(config, tokenizer=None):
    if tokenizer is None:
        tokenizer = get_tokenizer(config)

    collator = SafeCollator(
        tokenizer,
        max_length=config.model.max_seq_len,
        use_bracket_safe=config.training.get("use_bracket_safe", False),
    )

    # Handle file-based datasets
    if config.data.source == "file":
        if not config.data.file_path:
            raise ValueError("data.file_path must be specified when data.source is 'file'")
        if not os.path.exists(config.data.file_path):
            raise FileNotFoundError(f"Data file not found: {config.data.file_path}")

        dataset = SafeFileDataset(config.data.file_path, config.data.get("file_format", "safe"))
        return torch.utils.data.DataLoader(
            dataset,
            batch_size=config.loader.batch_size,
            collate_fn=collator,
            num_workers=config.loader.num_workers,
            pin_memory=config.loader.get("pin_memory", True),
            shuffle=True,
            persistent_workers=config.loader.num_workers > 0,
        )

    # Handle HuggingFace datasets
    logger.info(f"Loading HuggingFace dataset: {config.data.dataset_name}")
    try:
        dataset = datasets.load_dataset(
            config.data.dataset_name,
            split=config.data.get("split", "train"),
            streaming=config.data.get("streaming", False),
        )
    except Exception as e:
        logger.error(f"Failed to load dataset {config.data.dataset_name}: {e}")
        raise

    if config.data.get("streaming", False):
        logger.info("Using streaming dataset")
        dataset = SafeIterableDataset(
            dataset,
            safe_field=config.data.get("safe_field"),
            smiles_field=config.data.get("smiles_field"),
        )
        shuffle = False
    else:
        logger.info("Using map-style dataset")
        dataset = SafeMapDataset(
            dataset,
            safe_field=config.data.get("safe_field"),
            smiles_field=config.data.get("smiles_field"),
        )
        shuffle = True

    return torch.utils.data.DataLoader(
        dataset,
        batch_size=config.loader.batch_size,
        collate_fn=collator,
        num_workers=config.loader.num_workers,
        pin_memory=config.loader.get("pin_memory", True),
        shuffle=shuffle,
        persistent_workers=config.loader.num_workers > 0,
    )
