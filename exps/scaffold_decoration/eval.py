"""
Scaffold decoration evaluator.
Evaluates the model's ability to decorate scaffolds with chemical groups.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from exps.core.evaluator import BaseEvaluator
from exps.core import utils, metrics

logger = logging.getLogger(__name__)


class ScaffoldDecorationEvaluator(BaseEvaluator):
    """Evaluator for scaffold decoration task."""

    def evaluate_scaffold_preservation(self, molecules: List[str], scaffold: str) -> Dict[str, Any]:
        """
        Evaluate how well molecules preserve the scaffold.

        Args:
            molecules: List of generated SMILES
            scaffold: Original scaffold SMILES

        Returns:
            Dictionary with preservation metrics
        """
        from rdkit import Chem

        valid_molecules = [m for m in molecules if metrics.is_valid(m)]
        if not valid_molecules:
            return {"preservation_rate": 0.0, "num_preserved": 0}

        scaffold_mol = Chem.MolFromSmiles(scaffold)
        if scaffold_mol is None:
            logger.warning("Invalid scaffold SMILES")
            return {"preservation_rate": 0.0, "num_preserved": 0}

        preserved_count = 0
        for mol_smiles in valid_molecules:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is not None and mol.HasSubstructMatch(scaffold_mol):
                preserved_count += 1

        preservation_rate = preserved_count / len(valid_molecules)

        return {
            "preservation_rate": preservation_rate,
            "num_preserved": preserved_count,
            "num_valid": len(valid_molecules),
        }

    def generate_for_scaffold(
        self,
        scaffold: str,
        num_samples: int,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        apply_filter: bool = True,
    ) -> List[str]:
        """
        Generate decorated molecules for a single scaffold.

        Args:
            scaffold: Scaffold SMILES
            num_samples: Number of molecules to generate
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            apply_filter: Whether to filter by substructure

        Returns:
            List of generated SMILES
        """
        try:
            molecules = self.sampler.scaffold_decoration(
                scaffold=scaffold,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                apply_filter=apply_filter,
            )
            return molecules
        except Exception as e:
            logger.error(f"Generation failed for scaffold '{scaffold}': {e}")
            return []

    def run_evaluation(
        self,
        scaffolds: Optional[List[str]] = None,
        scaffolds_file: Optional[str] = None,
        num_samples_per_scaffold: int = 100,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        apply_filter: bool = True,
        output_dir: str = "exps/results/scaffold_decoration",
        experiment_name: Optional[str] = None,
        save_molecules: bool = True,
        save_plots: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run scaffold decoration evaluation.

        Args:
            scaffolds: List of scaffold SMILES (optional)
            scaffolds_file: Path to file with scaffolds (one per line)
            num_samples_per_scaffold: Number of molecules to generate per scaffold
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            apply_filter: Whether to apply substructure filter
            output_dir: Base output directory
            experiment_name: Optional experiment name
            save_molecules: Whether to save molecules
            save_plots: Whether to generate plots

        Returns:
            Dictionary with evaluation metrics
        """
        logger.info("=" * 80)
        logger.info("Starting scaffold decoration evaluation")
        logger.info("=" * 80)

        # Load scaffolds
        if scaffolds is None:
            if scaffolds_file is None:
                raise ValueError("Either scaffolds or scaffolds_file must be provided")
            logger.info(f"Loading scaffolds from: {scaffolds_file}")
            scaffolds = utils.load_molecules_from_file(scaffolds_file)
            logger.info(f"  Loaded {len(scaffolds)} scaffolds")
        else:
            logger.info(f"Using {len(scaffolds)} provided scaffolds")

        # Create experiment directory
        exp_dir = utils.create_experiment_dir(output_dir, "scaffold_decoration", experiment_name)
        logger.info(f"Results will be saved to: {exp_dir}")

        # Generate and evaluate for each scaffold
        all_molecules = []
        scaffold_results = []

        for i, scaffold in enumerate(scaffolds):
            logger.info(f"\nProcessing scaffold {i+1}/{len(scaffolds)}: {scaffold}")

            # Generate molecules
            molecules = self.generate_for_scaffold(
                scaffold=scaffold,
                num_samples=num_samples_per_scaffold,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                apply_filter=apply_filter,
            )

            if not molecules:
                logger.warning(f"  No molecules generated for scaffold {i+1}")
                continue

            # Evaluate preservation
            preservation = self.evaluate_scaffold_preservation(molecules, scaffold)
            logger.info(f"  Generated {len(molecules)} molecules")
            logger.info(f"  Preservation rate: {preservation['preservation_rate']:.4f}")

            # Store results
            scaffold_results.append({
                "scaffold": scaffold,
                "num_generated": len(molecules),
                "preservation": preservation,
            })

            all_molecules.extend(molecules)

        if not all_molecules:
            logger.error("No molecules generated! Evaluation failed.")
            return {}

        logger.info(f"\nTotal molecules generated: {len(all_molecules)}")

        # Evaluate overall metrics
        basic_metrics = self.evaluate_basic_metrics(all_molecules)
        chemical_properties = self.evaluate_chemical_properties(all_molecules)
        filter_metrics = self.evaluate_filters(all_molecules)

        # Aggregate scaffold-specific metrics
        avg_preservation = sum(r["preservation"]["preservation_rate"] for r in scaffold_results) / len(scaffold_results)

        # Compile all metrics
        all_metrics = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "checkpoint": self.checkpoint_path,
                "task": "scaffold_decoration",
                "num_scaffolds": len(scaffolds),
                "total_generated": len(all_molecules),
            },
            "generation_params": {
                "num_samples_per_scaffold": num_samples_per_scaffold,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "apply_filter": apply_filter,
            },
            "basic_metrics": basic_metrics,
            "chemical_properties": chemical_properties,
            "filters": filter_metrics,
            "scaffold_metrics": {
                "avg_preservation_rate": avg_preservation,
                "num_scaffolds": len(scaffolds),
                "per_scaffold_results": scaffold_results,
            },
        }

        # Save results
        self.save_results(
            molecules=all_molecules,
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
