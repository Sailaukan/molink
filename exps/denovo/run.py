#!/usr/bin/env python
"""
CLI script for de novo generation evaluation.

Example usage:
    python exps/denovo/run.py --checkpoint ckpt/tmp/checkpoints/10000.ckpt --num-samples 1000
"""

import argparse
import logging
import os
import sys

# Add parent directory to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

from exps.denovo.eval import DeNovoEvaluator
from exps.core.utils import setup_logging, load_hparams


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate de novo molecular generation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required arguments
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10000,
        help="Number of molecules to generate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=128,
        help="Batch size for generation",
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=200,
        help="Maximum number of tokens to generate per molecule",
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

    # Evaluation parameters
    parser.add_argument(
        "--train-set-path",
        type=str,
        default=None,
        help="Path to training set SMILES file (for novelty calculation)",
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
        help="Don't save generated molecules to file",
    )
    parser.add_argument(
        "--no-save-plots",
        action="store_true",
        help="Don't generate property distribution plots",
    )

    # Other parameters
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). If not specified, auto-detect.",
    )
    parser.add_argument(
        "--hparams",
        type=str,
        default=None,
        help="Path to hyperparameters YAML file (overrides command-line args)",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=None,
        help="Path to log file (default: output_dir/experiment/run.log)",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Load hyperparameters from file if provided
    hparams = {}
    if args.hparams:
        hparams = load_hparams(args.hparams)
        print(f"Loaded hyperparameters from: {args.hparams}")

    # Merge command-line args with hparams (command-line takes precedence)
    config = {
        "num_samples": args.num_samples,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "train_set_path": args.train_set_path,
        "output_dir": args.output_dir,
        "experiment_name": args.experiment_name,
        "save_molecules": not args.no_save_molecules,
        "save_plots": not args.no_save_plots,
    }

    # Override with hparams if available
    if "generation" in hparams:
        for key in ["num_samples", "batch_size", "max_new_tokens", "temperature", "top_p", "top_k"]:
            if key in hparams["generation"]:
                # Only override if not explicitly set via command line
                if getattr(args, key.replace("_", "-"), None) is None:
                    config[key] = hparams["generation"][key]

    if "evaluation" in hparams:
        if "save_molecules" in hparams["evaluation"] and not args.no_save_molecules:
            config["save_molecules"] = hparams["evaluation"]["save_molecules"]
        if "save_plots" in hparams["evaluation"] and not args.no_save_plots:
            config["save_plots"] = hparams["evaluation"]["save_plots"]

    # Set up logging
    log_file = args.log_file
    if log_file is None and args.experiment_name:
        log_dir = os.path.join(args.output_dir, "denovo", args.experiment_name)
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "run.log")

    setup_logging(log_file=log_file, level=logging.INFO)
    logger = logging.getLogger(__name__)

    logger.info("=" * 80)
    logger.info("De Novo Generation Evaluation")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {args.device or 'auto-detect'}")
    logger.info(f"Configuration: {config}")
    logger.info("=" * 80)

    # Initialize evaluator
    try:
        evaluator = DeNovoEvaluator(
            checkpoint_path=args.checkpoint,
            device=args.device,
        )
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        return 1

    # Run evaluation
    try:
        results = evaluator.run_evaluation(**config)

        # Print summary
        logger.info("\n" + "=" * 80)
        logger.info("EVALUATION SUMMARY")
        logger.info("=" * 80)
        if "basic_metrics" in results:
            logger.info("\nBasic Metrics:")
            for key, value in results["basic_metrics"].items():
                if isinstance(value, float):
                    logger.info(f"  {key}: {value:.4f}")
                else:
                    logger.info(f"  {key}: {value}")

        if "chemical_properties" in results:
            logger.info("\nChemical Properties:")
            for prop, stats in results["chemical_properties"].items():
                if isinstance(stats, dict) and "mean" in stats:
                    logger.info(f"  {prop}: mean={stats['mean']:.3f}, std={stats['std']:.3f}")

        if "filters" in results:
            logger.info("\nFilter Pass Rates:")
            for key, value in results["filters"].items():
                logger.info(f"  {key}: {value:.4f}")

        logger.info("=" * 80)

        return 0

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
