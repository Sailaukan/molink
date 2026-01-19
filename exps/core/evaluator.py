"""
Base evaluator class for molecular generation tasks.
Provides common functionality for evaluation across all tasks.
"""

import logging
import os
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Any, Optional

import torch

from molink.sampler import MolinkSampler
from . import metrics
from . import utils

logger = logging.getLogger(__name__)


class BaseEvaluator(ABC):
    """Base class for task-specific evaluators."""

    def __init__(self, checkpoint_path: str, device: Optional[str] = None):
        """
        Initialize the evaluator.

        Args:
            checkpoint_path: Path to model checkpoint
            device: Device to use for inference (cuda/cpu). If None, auto-detect.
        """
        self.checkpoint_path = checkpoint_path
        self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

        logger.info(f"Initializing evaluator with checkpoint: {checkpoint_path}")
        logger.info(f"Using device: {self.device}")

        # Load sampler
        self.sampler = MolinkSampler(checkpoint_path, device=self.device)

    def evaluate_basic_metrics(self, molecules: List[str], train_set: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Evaluate basic metrics: validity, uniqueness, diversity, novelty.

        Args:
            molecules: List of generated SMILES strings
            train_set: Optional training set for novelty calculation

        Returns:
            Dictionary with basic metrics
        """
        logger.info("Evaluating basic metrics...")

        results = {}

        # Validity
        validity = metrics.validity_rate(molecules)
        results["validity"] = validity
        logger.info(f"  Validity: {validity:.4f}")

        # Uniqueness
        uniqueness, unique_smiles = metrics.uniqueness_rate(molecules)
        results["uniqueness"] = uniqueness
        results["num_unique"] = len(unique_smiles)
        logger.info(f"  Uniqueness: {uniqueness:.4f} ({len(unique_smiles)} unique)")

        # Diversity
        if unique_smiles:
            diversity = metrics.diversity_rate(unique_smiles)
            results["diversity"] = diversity
            logger.info(f"  Diversity: {diversity:.4f}")
        else:
            results["diversity"] = 0.0

        # Novelty (if training set provided)
        if train_set is not None:
            novelty = metrics.novelty_rate(molecules, train_set)
            results["novelty"] = novelty
            logger.info(f"  Novelty: {novelty:.4f}")

        return results

    def evaluate_chemical_properties(self, molecules: List[str]) -> Dict[str, Dict[str, float]]:
        """
        Evaluate chemical properties: QED, SA, LogP, MW, TPSA, etc.

        Args:
            molecules: List of SMILES strings

        Returns:
            Dictionary with property statistics
        """
        logger.info("Evaluating chemical properties...")

        # Compute all properties
        properties = metrics.compute_all_properties(molecules)

        # Compute statistics for each property
        results = {}
        for prop_name, prop_values in properties.items():
            if prop_values:
                stats = metrics.compute_property_statistics(prop_values)
                results[prop_name] = stats
                logger.info(f"  {prop_name}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

        return results

    def evaluate_filters(self, molecules: List[str]) -> Dict[str, float]:
        """
        Evaluate filter pass rates: Lipinski, PAINS, drug-like.

        Args:
            molecules: List of SMILES strings

        Returns:
            Dictionary with filter pass rates
        """
        logger.info("Evaluating molecular filters...")

        valid_molecules = [m for m in molecules if metrics.is_valid(m)]
        if not valid_molecules:
            return {
                "lipinski": 0.0,
                "pains": 0.0,
                "drug_like": 0.0,
            }

        lipinski_pass = sum(1 for m in valid_molecules if metrics.lipinski_filter(m))
        pains_pass = sum(1 for m in valid_molecules if metrics.pains_filter(m))
        drug_like_pass = sum(1 for m in valid_molecules if metrics.drug_like_filter(m))

        results = {
            "lipinski": lipinski_pass / len(valid_molecules),
            "pains": pains_pass / len(valid_molecules),
            "drug_like": drug_like_pass / len(valid_molecules),
        }

        logger.info(f"  Lipinski pass rate: {results['lipinski']:.4f}")
        logger.info(f"  PAINS pass rate: {results['pains']:.4f}")
        logger.info(f"  Drug-like pass rate: {results['drug_like']:.4f}")

        return results

    def save_results(
        self,
        molecules: List[str],
        metrics_dict: Dict[str, Any],
        output_dir: str,
        save_molecules: bool = True,
        save_plots: bool = True,
    ):
        """
        Save evaluation results to disk.

        Args:
            molecules: Generated molecules (SMILES)
            metrics_dict: Dictionary of all metrics
            output_dir: Directory to save results
            save_molecules: Whether to save molecules to file
            save_plots: Whether to generate and save plots
        """
        logger.info(f"Saving results to: {output_dir}")

        # Save metrics as JSON
        metrics_file = os.path.join(output_dir, "metrics.json")
        utils.save_json(metrics_dict, metrics_file)
        logger.info(f"  Saved metrics to {metrics_file}")

        # Save human-readable summary
        summary_file = os.path.join(output_dir, "summary.txt")
        utils.save_summary(metrics_dict, summary_file)
        logger.info(f"  Saved summary to {summary_file}")

        # Save molecules
        if save_molecules:
            # Save SMILES to text file
            molecules_file = os.path.join(output_dir, "molecules.txt")
            utils.save_molecules_to_file(molecules, molecules_file)
            logger.info(f"  Saved {len(molecules)} molecules to {molecules_file}")

            # Save SMILES with properties to CSV
            if "chemical_properties" in metrics_dict:
                properties = metrics.compute_all_properties(molecules)
                csv_file = os.path.join(output_dir, "molecules.csv")
                utils.save_molecules_to_csv(molecules, properties, csv_file)
                logger.info(f"  Saved molecules with properties to {csv_file}")

        # Save plots
        if save_plots and "chemical_properties" in metrics_dict:
            self._save_plots(molecules, output_dir)

    def _save_plots(self, molecules: List[str], output_dir: str):
        """
        Generate and save property distribution plots.

        Args:
            molecules: List of SMILES strings
            output_dir: Directory to save plots
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns

            sns.set_style("whitegrid")
            plots_dir = os.path.join(output_dir, "plots")
            os.makedirs(plots_dir, exist_ok=True)

            properties = metrics.compute_all_properties(molecules)

            # Define properties to plot with their labels and ranges
            plot_configs = {
                "qed": {"label": "QED", "range": (0, 1)},
                "sa": {"label": "Synthetic Accessibility", "range": (1, 10)},
                "logp": {"label": "LogP", "range": (-5, 10)},
                "mw": {"label": "Molecular Weight", "range": (0, 1000)},
                "tpsa": {"label": "TPSA", "range": (0, 200)},
            }

            for prop_name, config in plot_configs.items():
                if prop_name in properties and properties[prop_name]:
                    fig, ax = plt.subplots(figsize=(8, 6))
                    ax.hist(properties[prop_name], bins=50, edgecolor="black", alpha=0.7)
                    ax.set_xlabel(config["label"], fontsize=12)
                    ax.set_ylabel("Frequency", fontsize=12)
                    ax.set_title(f"{config['label']} Distribution", fontsize=14)

                    # Add statistics text
                    stats = metrics.compute_property_statistics(properties[prop_name])
                    stats_text = f"Mean: {stats['mean']:.2f}\nStd: {stats['std']:.2f}\nMedian: {stats['median']:.2f}"
                    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
                           verticalalignment='top', horizontalalignment='right',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

                    plt.tight_layout()
                    plot_file = os.path.join(plots_dir, f"{prop_name}_distribution.png")
                    plt.savefig(plot_file, dpi=150)
                    plt.close()
                    logger.info(f"  Saved plot: {plot_file}")

        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")

    @abstractmethod
    def run_evaluation(self, **kwargs) -> Dict[str, Any]:
        """
        Run the full evaluation pipeline.
        Must be implemented by subclasses.

        Returns:
            Dictionary with all evaluation metrics
        """
        pass
