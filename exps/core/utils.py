"""
Shared utility functions for the evaluation framework.
Includes file I/O, logging, experiment setup, and helper functions.
"""

import os
import json
import yaml
import logging
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional, Set
import pandas as pd
from .metrics import canonicalize_smiles, is_valid


def setup_logging(log_file: Optional[str] = None, level: int = logging.INFO):
    """
    Set up logging configuration.

    Args:
        log_file: Path to log file. If None, only logs to console.
        level: Logging level (default: INFO)
    """
    handlers = [logging.StreamHandler()]

    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )


def load_molecules_from_file(file_path: str) -> List[str]:
    """
    Load SMILES strings from a text file (one per line).

    Args:
        file_path: Path to file containing SMILES

    Returns:
        List of SMILES strings
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    with open(file_path, 'r') as f:
        smiles_list = [line.strip() for line in f if line.strip()]

    return smiles_list


def save_molecules_to_file(smiles_list: List[str], file_path: str):
    """
    Save SMILES strings to a text file (one per line).

    Args:
        smiles_list: List of SMILES strings
        file_path: Path to output file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        for smiles in smiles_list:
            f.write(f"{smiles}\n")


def save_molecules_to_csv(smiles_list: List[str], properties: Dict[str, List], file_path: str):
    """
    Save SMILES with properties to a CSV file.

    Args:
        smiles_list: List of SMILES strings
        properties: Dict with property names as keys and lists of values
        file_path: Path to output CSV file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Create DataFrame
    data = {"smiles": smiles_list}

    # Add properties (ensure all lists have same length as smiles_list)
    for prop_name, prop_values in properties.items():
        if len(prop_values) == len(smiles_list):
            data[prop_name] = prop_values

    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)


def canonicalize_smiles_list(smiles_list: List[str]) -> List[str]:
    """
    Canonicalize a list of SMILES strings, filtering out invalid ones.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of canonical SMILES strings (only valid ones)
    """
    canonical = []
    for s in smiles_list:
        c = canonicalize_smiles(s)
        if c is not None:
            canonical.append(c)
    return canonical


def remove_duplicates(smiles_list: List[str]) -> List[str]:
    """
    Remove duplicate molecules from a list (based on canonical SMILES).

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of unique SMILES strings
    """
    canonical_set = set()
    unique_smiles = []

    for s in smiles_list:
        c = canonicalize_smiles(s)
        if c is not None and c not in canonical_set:
            canonical_set.add(c)
            unique_smiles.append(s)

    return unique_smiles


def filter_valid_molecules(smiles_list: List[str]) -> List[str]:
    """
    Filter out invalid molecules from a list.

    Args:
        smiles_list: List of SMILES strings

    Returns:
        List of valid SMILES strings
    """
    return [s for s in smiles_list if is_valid(s)]


def create_experiment_dir(base_dir: str, task_name: str, experiment_name: Optional[str] = None) -> str:
    """
    Create a timestamped experiment directory.

    Args:
        base_dir: Base directory (e.g., "exps/results")
        task_name: Name of the task (e.g., "denovo")
        experiment_name: Optional experiment name (default: timestamp)

    Returns:
        Path to created experiment directory
    """
    if experiment_name is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        experiment_name = timestamp

    exp_dir = os.path.join(base_dir, task_name, experiment_name)
    os.makedirs(exp_dir, exist_ok=True)

    # Create plots subdirectory
    plots_dir = os.path.join(exp_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    return exp_dir


def load_hparams(hparams_file: str) -> Dict[str, Any]:
    """
    Load hyperparameters from a YAML file.

    Args:
        hparams_file: Path to YAML file

    Returns:
        Dictionary of hyperparameters
    """
    if not os.path.exists(hparams_file):
        raise FileNotFoundError(f"Hyperparameters file not found: {hparams_file}")

    with open(hparams_file, 'r') as f:
        hparams = yaml.safe_load(f)

    return hparams


def save_hparams(hparams: Dict[str, Any], file_path: str):
    """
    Save hyperparameters to a YAML file.

    Args:
        hparams: Dictionary of hyperparameters
        file_path: Path to output YAML file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        yaml.dump(hparams, f, default_flow_style=False)


def save_json(data: Dict[str, Any], file_path: str):
    """
    Save dictionary to a JSON file.

    Args:
        data: Dictionary to save
        file_path: Path to output JSON file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        json.dump(data, f, indent=2)


def load_json(file_path: str) -> Dict[str, Any]:
    """
    Load dictionary from a JSON file.

    Args:
        file_path: Path to JSON file

    Returns:
        Dictionary loaded from file
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"JSON file not found: {file_path}")

    with open(file_path, 'r') as f:
        data = json.load(f)

    return data


def save_summary(metrics: Dict[str, Any], file_path: str):
    """
    Save a human-readable summary of metrics to a text file.

    Args:
        metrics: Dictionary of metrics
        file_path: Path to output text file
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    with open(file_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("EVALUATION SUMMARY\n")
        f.write("=" * 80 + "\n\n")

        # Metadata
        if "metadata" in metrics:
            f.write("Metadata:\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics["metadata"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        # Generation parameters
        if "generation_params" in metrics:
            f.write("Generation Parameters:\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics["generation_params"].items():
                f.write(f"  {key}: {value}\n")
            f.write("\n")

        # Basic metrics
        if "basic_metrics" in metrics:
            f.write("Basic Metrics:\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics["basic_metrics"].items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

        # Chemical properties
        if "chemical_properties" in metrics:
            f.write("Chemical Properties:\n")
            f.write("-" * 80 + "\n")
            for prop_name, prop_stats in metrics["chemical_properties"].items():
                if isinstance(prop_stats, dict):
                    f.write(f"  {prop_name}:\n")
                    for stat_name, stat_value in prop_stats.items():
                        f.write(f"    {stat_name}: {stat_value:.4f}\n")
                else:
                    f.write(f"  {prop_name}: {prop_stats}\n")
            f.write("\n")

        # Filters
        if "filters" in metrics:
            f.write("Filter Pass Rates:\n")
            f.write("-" * 80 + "\n")
            for key, value in metrics["filters"].items():
                if isinstance(value, float):
                    f.write(f"  {key}: {value:.4f}\n")
                else:
                    f.write(f"  {key}: {value}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")


def compute_statistics(values: List[float]) -> Dict[str, float]:
    """
    Compute basic statistics for a list of values.
    This is a wrapper around metrics.compute_property_statistics for convenience.

    Args:
        values: List of numeric values

    Returns:
        Dictionary with statistics (mean, std, median, min, max, percentiles)
    """
    from .metrics import compute_property_statistics
    return compute_property_statistics(values)
