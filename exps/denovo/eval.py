"""
De novo generation evaluator.
Evaluates unconditional molecule generation from scratch.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from exps.core.evaluator import BaseEvaluator
from exps.core import utils

logger = logging.getLogger(__name__)


class DeNovoEvaluator(BaseEvaluator):
    """Evaluator for de novo molecular generation."""

    def generate(
        self,
        num_samples: int,
        batch_size: int = 128,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
    ) -> List[str]:
        """
        Generate molecules de novo.

        Args:
            num_samples: Total number of molecules to generate
            batch_size: Batch size for generation
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter

        Returns:
            List of generated SMILES strings
        """
        logger.info(f"Generating {num_samples} molecules de novo...")
        logger.info(f"  Batch size: {batch_size}")
        logger.info(f"  Max tokens: {max_new_tokens}")
        logger.info(f"  Temperature: {temperature}")
        logger.info(f"  Top-p: {top_p}")
        logger.info(f"  Top-k: {top_k}")

        all_molecules = []
        num_batches = (num_samples + batch_size - 1) // batch_size

        for i in range(num_batches):
            current_batch_size = min(batch_size, num_samples - len(all_molecules))
            logger.info(f"  Batch {i+1}/{num_batches} (size: {current_batch_size})")

            try:
                batch_molecules = self.sampler.de_novo_generation(
                    num_samples=current_batch_size,
                    max_new_tokens=max_new_tokens,
                    temperature=temperature,
                    top_p=top_p,
                    top_k=top_k,
                )
                all_molecules.extend(batch_molecules)
                logger.info(f"    Generated {len(batch_molecules)} valid molecules")
            except Exception as e:
                logger.error(f"    Batch {i+1} failed: {e}")

        logger.info(f"Total molecules generated: {len(all_molecules)}/{num_samples}")
        return all_molecules

    def run_evaluation(
        self,
        num_samples: int = 10000,
        batch_size: int = 128,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        train_set_path: Optional[str] = None,
        output_dir: str = "exps/results/denovo",
        experiment_name: Optional[str] = None,
        save_molecules: bool = True,
        save_plots: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run full de novo generation evaluation.

        Args:
            num_samples: Number of molecules to generate
            batch_size: Batch size for generation
            max_new_tokens: Maximum tokens per molecule
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            train_set_path: Optional path to training set for novelty calculation
            output_dir: Base output directory
            experiment_name: Optional name for this experiment
            save_molecules: Whether to save generated molecules
            save_plots: Whether to generate plots

        Returns:
            Dictionary with all evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("Starting de novo generation evaluation")
        logger.info("=" * 80)

        # Create experiment directory
        exp_dir = utils.create_experiment_dir(output_dir, "denovo", experiment_name)
        logger.info(f"Results will be saved to: {exp_dir}")

        # Generate molecules
        molecules = self.generate(
            num_samples=num_samples,
            batch_size=batch_size,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
        )

        if not molecules:
            logger.error("No molecules generated! Evaluation failed.")
            return {}

        # Load training set if provided
        train_set = None
        if train_set_path and os.path.exists(train_set_path):
            logger.info(f"Loading training set from: {train_set_path}")
            try:
                train_set = utils.load_molecules_from_file(train_set_path)
                logger.info(f"  Loaded {len(train_set)} training molecules")
            except Exception as e:
                logger.warning(f"Failed to load training set: {e}")

        # Evaluate basic metrics
        basic_metrics = self.evaluate_basic_metrics(molecules, train_set=train_set)

        # Evaluate chemical properties
        chemical_properties = self.evaluate_chemical_properties(molecules)

        # Evaluate filters
        filter_metrics = self.evaluate_filters(molecules)

        # Compile all metrics
        all_metrics = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "checkpoint": self.checkpoint_path,
                "task": "denovo",
                "num_generated": len(molecules),
            },
            "generation_params": {
                "num_samples": num_samples,
                "batch_size": batch_size,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
            },
            "basic_metrics": basic_metrics,
            "chemical_properties": chemical_properties,
            "filters": filter_metrics,
        }

        # Save results
        self.save_results(
            molecules=molecules,
            metrics_dict=all_metrics,
            output_dir=exp_dir,
            save_molecules=save_molecules,
            save_plots=save_plots,
        )

        logger.info("=" * 80)
        logger.info("Evaluation complete!")
        logger.info(f"Results saved to: {exp_dir}")
        logger.info("=" * 80)

        return all_metrics
