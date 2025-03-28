#!/usr/bin/env python
"""
Command-line interface for running multi-agent monitoring experiments.
"""

import os
import sys
import argparse
from pathlib import Path
from typing import List, Optional

from utils.experiment_runner import ExperimentRunner


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Run multi-agent monitoring experiments"
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        required=True,
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default=None,
        help="Directory to save experiment results (default: auto-generated)"
    )
    
    parser.add_argument(
        "--tracking",
        type=str,
        choices=["mlflow", "wandb", "none"],
        default="mlflow",
        help="Experiment tracking platform to use (default: mlflow)"
    )
    
    parser.add_argument(
        "--ray",
        action="store_true",
        help="Use Ray for parallel execution"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode"
    )
    
    return parser.parse_args()


def main():
    """Main entry point for running experiments."""
    args = parse_args()
    
    # Resolve configuration path
    config_path = os.path.abspath(args.config)
    if not os.path.exists(config_path):
        print(f"Error: Configuration file not found: {config_path}")
        sys.exit(1)
        
    # Resolve output directory
    output_dir = args.output_dir
    if output_dir:
        output_dir = os.path.abspath(output_dir)
        os.makedirs(output_dir, exist_ok=True)
    
    # Set up tracking options
    use_mlflow = args.tracking == "mlflow"
    use_wandb = args.tracking == "wandb"
    
    # Create and run the experiment
    try:
        experiment_runner = ExperimentRunner(
            config_path=config_path,
            output_dir=output_dir,
            use_mlflow=use_mlflow,
            use_wandb=use_wandb,
            use_ray=args.ray
        )
        
        # Run the experiment
        results = experiment_runner.run_experiment()
        
        # Print summary
        successful_trials = [r for r in results if r.get('success', False)]
        print(f"\nExperiment complete: {len(successful_trials)}/{len(results)} trials successful")
        print(f"Results saved to: {experiment_runner.output_dir}")
        
        return 0
        
    except Exception as e:
        if args.debug:
            import traceback
            traceback.print_exc()
        else:
            print(f"Error: {str(e)}")
            print("Run with --debug for more information")
        return 1


if __name__ == "__main__":
    sys.exit(main()) 