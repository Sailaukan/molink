"""
Linker design evaluator.
Evaluates the model's ability to design linkers between molecular fragments.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List, Tuple

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from exps.core.evaluator import BaseEvaluator
from exps.core import utils, metrics

logger = logging.getLogger(__name__)


class LinkerDesignEvaluator(BaseEvaluator):
    """Evaluator for linker design task."""

    def evaluate_fragment_preservation(
        self, molecules: List[str], fragment_a: str, fragment_b: str
    ) -> Dict[str, Any]:
        """
        Evaluate how well molecules preserve both fragments.

        Args:
            molecules: List of generated SMILES
            fragment_a: First fragment SMILES
            fragment_b: Second fragment SMILES

        Returns:
            Dictionary with preservation metrics
        """
        from rdkit import Chem

        valid_molecules = [m for m in molecules if metrics.is_valid(m)]
        if not valid_molecules:
            return {
                "both_preserved_rate": 0.0,
                "fragment_a_preserved_rate": 0.0,
                "fragment_b_preserved_rate": 0.0,
                "num_both_preserved": 0,
            }

        frag_a_mol = Chem.MolFromSmiles(fragment_a)
        frag_b_mol = Chem.MolFromSmiles(fragment_b)

        if frag_a_mol is None or frag_b_mol is None:
            logger.warning("Invalid fragment SMILES")
            return {
                "both_preserved_rate": 0.0,
                "fragment_a_preserved_rate": 0.0,
                "fragment_b_preserved_rate": 0.0,
                "num_both_preserved": 0,
            }

        frag_a_count = 0
        frag_b_count = 0
        both_count = 0

        for mol_smiles in valid_molecules:
            mol = Chem.MolFromSmiles(mol_smiles)
            if mol is None:
                continue

            has_a = mol.HasSubstructMatch(frag_a_mol)
            has_b = mol.HasSubstructMatch(frag_b_mol)

            if has_a:
                frag_a_count += 1
            if has_b:
                frag_b_count += 1
            if has_a and has_b:
                both_count += 1

        return {
            "both_preserved_rate": both_count / len(valid_molecules),
            "fragment_a_preserved_rate": frag_a_count / len(valid_molecules),
            "fragment_b_preserved_rate": frag_b_count / len(valid_molecules),
            "num_both_preserved": both_count,
            "num_valid": len(valid_molecules),
        }

    def generate_for_fragments(
        self,
        fragment_a: str,
        fragment_b: str,
        num_samples: int,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        apply_filter: bool = True,
    ) -> List[str]:
        """
        Generate molecules linking two fragments.

        Args:
            fragment_a: First fragment SMILES
            fragment_b: Second fragment SMILES
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
            molecules = self.sampler.linker_design(
                fragment_a=fragment_a,
                fragment_b=fragment_b,
                num_samples=num_samples,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                apply_filter=apply_filter,
            )
            return molecules
        except Exception as e:
            logger.error(f"Generation failed for fragments '{fragment_a}', '{fragment_b}': {e}")
            return []

    def run_evaluation(
        self,
        fragment_pairs: Optional[List[Tuple[str, str]]] = None,
        fragments_file: Optional[str] = None,
        num_samples_per_pair: int = 100,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_p: float = 0.9,
        top_k: int = 0,
        apply_filter: bool = True,
        output_dir: str = "exps/results/linker_design",
        experiment_name: Optional[str] = None,
        save_molecules: bool = True,
        save_plots: bool = True,
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Run linker design evaluation.

        Args:
            fragment_pairs: List of (fragment_a, fragment_b) tuples
            fragments_file: Path to file with fragment pairs (format: "frag_a,frag_b" per line)
            num_samples_per_pair: Number of molecules to generate per fragment pair
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
        logger.info("Starting linker design evaluation")
        logger.info("=" * 80)

        # Load fragment pairs
        if fragment_pairs is None:
            if fragments_file is None:
                raise ValueError("Either fragment_pairs or fragments_file must be provided")

            logger.info(f"Loading fragment pairs from: {fragments_file}")
            fragment_pairs = []
            with open(fragments_file, 'r') as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith('#'):
                        continue
                    parts = line.split(',')
                    if len(parts) >= 2:
                        fragment_pairs.append((parts[0].strip(), parts[1].strip()))

            logger.info(f"  Loaded {len(fragment_pairs)} fragment pairs")
        else:
            logger.info(f"Using {len(fragment_pairs)} provided fragment pairs")

        # Create experiment directory
        exp_dir = utils.create_experiment_dir(output_dir, "linker_design", experiment_name)
        logger.info(f"Results will be saved to: {exp_dir}")

        # Generate and evaluate for each fragment pair
        all_molecules = []
        pair_results = []

        for i, (frag_a, frag_b) in enumerate(fragment_pairs):
            logger.info(f"\nProcessing pair {i+1}/{len(fragment_pairs)}: {frag_a} + {frag_b}")

            # Generate molecules
            molecules = self.generate_for_fragments(
                fragment_a=frag_a,
                fragment_b=frag_b,
                num_samples=num_samples_per_pair,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                top_k=top_k,
                apply_filter=apply_filter,
            )

            if not molecules:
                logger.warning(f"  No molecules generated for pair {i+1}")
                continue

            # Evaluate fragment preservation
            preservation = self.evaluate_fragment_preservation(molecules, frag_a, frag_b)
            logger.info(f"  Generated {len(molecules)} molecules")
            logger.info(f"  Both fragments preserved: {preservation['both_preserved_rate']:.4f}")

            # Store results
            pair_results.append({
                "fragment_a": frag_a,
                "fragment_b": frag_b,
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

        # Aggregate fragment-specific metrics
        avg_both_preserved = sum(r["preservation"]["both_preserved_rate"] for r in pair_results) / len(pair_results)

        # Compile all metrics
        all_metrics = {
            "metadata": {
                "timestamp": datetime.now().isoformat(),
                "checkpoint": self.checkpoint_path,
                "task": "linker_design",
                "num_fragment_pairs": len(fragment_pairs),
                "total_generated": len(all_molecules),
            },
            "generation_params": {
                "num_samples_per_pair": num_samples_per_pair,
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "apply_filter": apply_filter,
            },
            "basic_metrics": basic_metrics,
            "chemical_properties": chemical_properties,
            "filters": filter_metrics,
            "linker_metrics": {
                "avg_both_preserved_rate": avg_both_preserved,
                "num_fragment_pairs": len(fragment_pairs),
                "per_pair_results": pair_results,
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
