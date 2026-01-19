#!/usr/bin/env python
"""
CLI script for scaffold decoration evaluation.

Example usage:
    python exps/scaffold_decoration/run.py \\
        --checkpoint ckpt/tmp/checkpoints/10000.ckpt \\
        --scaffolds-file data/test_scaffolds.txt \\
        --num-samples-per-scaffold 100
"""

import argparse
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from exps.scaffold_decoration.eval import ScaffoldDecorationEvaluator
from exps.core.utils import setup_logging, load_hparams


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate scaffold decoration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Scaffold input
    parser.add_argument(
        "--scaffolds-file",
        type=str,
        default=None,
        help="Path to file with scaffold SMILES (one per line)",
    )
    parser.add_argument(
        "--scaffold",
        type=str,
        default=None,
        help="Single scaffold SMILES (alternative to --scaffolds-file)",
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples-per-scaffold",
        type=int,
        default=100,
        help="Number of molecules to generate per scaffold",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.9,
        help="Nucleus sampling parameter",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=0,
        help="Top-k sampling parameter (0 = disabled)",
    )
    parser.add_argument(
        "--no-filter",
        action="store_true",
        help="Disable substructure filtering",
    )

    # Output parameters
    parser.add_argument(
        "--output-dir",
        type=str,
        default="exps/results",
        help="Base output directory",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Name for this experiment (default: timestamp)",
    )
    parser.add_argument(
        "--no-save-molecules",
        action="store_true",
        help="Don't save generated molecules",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Don't generate plots",
    )

    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default=None,
        help="Path to hyperparameters YAML file",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Validate scaffold input
    if args.scaffolds_file is None and args.scaffold is None:
        print("Error: Either --scaffolds-file or --scaffold must be provided")
        return 1

    # Load hyperparameters
    hparams = {}
    if args.hparams:
        hparams = load_hparams(args.hparams)
        print(f"Loaded hyperparameters from: {args.hparams}")

    # Prepare scaffolds
    scaffolds = None
    scaffolds_file = args.scaffolds_file
    if args.scaffold:
        scaffolds = [args.scaffold]

    # Setup logging
    log_dir = os.path.join(args.output_dir, "scaffold_decoration",
                           args.experiment_name or "latest")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, "run.log")
    setup_logging(log_file=log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("Scaffold Decoration Evaluation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Scaffolds: {scaffolds_file or scaffolds}")
    logger.info("=" * 80)

    # Initialize evaluator
    try:
        evaluator = ScaffoldDecorationEvaluator(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return 1

    # Run evaluation
    try:
        results = evaluator.run_evaluation(
            scaffolds=scaffolds,
            scaffolds_file=scaffolds_file,
            num_samples_per_scaffold=args.num_samples_per_scaffold,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            apply_filter=not args.no_filter,
            output_dir=args.output_dir,
            experiment_name=args.experiment_name,
            save_molecules=not args.no_save_molecules,
            save_plots=not args.no_save_plots,
        )

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)

        if "scaffold_metrics" in results:
            logger.info("\nScaffold Metrics:")
            logger.info(f"  Average preservation rate: {results['scaffold_metrics']['avg_preservation_rate']:.4f}")

        if "basic_metrics" in results:
            logger.info("\nBasic Metrics:")
            for key, value in results["basic_metrics"].items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")

        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
