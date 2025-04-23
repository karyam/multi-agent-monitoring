"""
Experiment runner for multi-agent simulations.
Handles experiment execution, tracking, and result management.
"""

import os
import time
import json
import random
import logging
import importlib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional, Union, Callable, Tuple
from datetime import datetime
import concurrent.futures
from pathlib import Path
import mlflow
import ray
from tqdm import tqdm

from utils.config_manager import ConfigManager
from utils.logging import setup_experiment_logger, log_experiment_config, log_trial_results, export_results


class ExperimentRunner:
    """
    Class for running multi-agent simulation experiments.
    Handles configuration, execution, tracking, and result management.
    """
    
    def __init__(self, 
        config_path: str,
        output_dir: Optional[str] = None,
        use_mlflow: bool = True,
        use_wandb: bool = False,
        use_ray: bool = False
    ):
        """
        Initialize an experiment runner.
        
        Args:
            config_path: Path to the experiment configuration file
            output_dir: Directory to save experiment results
            use_mlflow: Whether to use MLflow for tracking
            use_wandb: Whether to use Weights & Biases for tracking
            use_ray: Whether to use Ray for parallel execution
        """
        self.config_manager = ConfigManager(config_path)
        self.config = self.config_manager.get_merged_config()
        
        # Override tracking settings from constructor
        if 'tracking' not in self.config['experiment']:
            self.config['experiment']['tracking'] = {}
            
        if use_mlflow:
            self.config['experiment']['tracking']['platform'] = 'mlflow'
            self.config['experiment']['tracking']['enabled'] = True
        elif use_wandb:
            self.config['experiment']['tracking']['platform'] = 'wandb'
            self.config['experiment']['tracking']['enabled'] = True
        
        # Override execution settings
        if use_ray:
            self.config['execution']['use_ray'] = True
        
        # Set up output directory
        self.output_dir = output_dir or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "experiments/results",
            self.config['experiment']['name'],
            datetime.now().strftime("%Y%m%d_%H%M%S")
        )
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save the merged configuration
        config_save_path = os.path.join(self.output_dir, "experiment_config.yaml")
        self.config_manager.save_config(self.config, config_save_path)
        
        # Set up logging
        self.logger = setup_experiment_logger(
            experiment_name=self.config['experiment']['name'],
            output_dir=self.output_dir
        )
        
        # Generate sweep configurations
        self.sweep_configs = self.config_manager.generate_sweep_configs()
        self.logger.info(f"Generated {len(self.sweep_configs)} configurations for parameter sweep")
        
        # Initialize Ray if using it
        if self.config['execution']['use_ray']:
            if not ray.is_initialized():
                ray.init()
            self.use_ray = True
        else:
            self.use_ray = False
        
        # Initialize tracking
        self._initialize_tracking()
        
        # Register a signal handler to clean up resources
        try:
            import signal
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
        except:
            pass
    
    def _signal_handler(self, sig, frame):
        """Handle signals to clean up resources properly."""
        self.logger.info("Received signal to terminate. Cleaning up...")
        self._cleanup()
        import sys
        sys.exit(0)
        
    def _cleanup(self):
        """Clean up resources."""
        if self.use_ray and ray.is_initialized():
            ray.shutdown()
        
        if hasattr(self, 'tracking_client') and self.tracking_client:
            if self.config['experiment']['tracking']['platform'] == 'mlflow':
                mlflow.end_run()
            elif self.config['experiment']['tracking']['platform'] == 'wandb':
                import wandb
                wandb.finish()
    
    def _initialize_tracking(self):
        """Initialize experiment tracking based on configuration."""
        self.tracking_client = None
        
        if not self.config['experiment']['tracking']['enabled']:
            self.logger.info("Experiment tracking is disabled")
            return
        
        platform = self.config['experiment']['tracking']['platform']
        project_name = self.config['experiment']['tracking']['project_name']
        
        if platform == 'mlflow':
            # Initialize MLflow
            mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "")
            if not mlflow_tracking_uri:
                mlflow_dir = os.path.join(self.output_dir, "mlruns")
                os.makedirs(mlflow_dir, exist_ok=True)
                mlflow.set_tracking_uri(f"file:{mlflow_dir}")
            
            mlflow.set_experiment(project_name)
            mlflow.start_run(run_name=self.config['experiment']['name'])
            
            # Log experiment configuration
            mlflow.log_params({
                "config_path": os.path.abspath(self.config_manager.base_config_path),
                "output_dir": os.path.abspath(self.output_dir),
                "num_sweep_configs": len(self.sweep_configs),
                **self._flatten_dict(self.config, parent_key="config")
            })
            
            self.tracking_client = mlflow
            self.logger.info(f"Initialized MLflow tracking with experiment '{project_name}'")
            
        elif platform == 'wandb':
            # Initialize Weights & Biases
            try:
                import wandb
                wandb.init(
                    project=project_name,
                    name=self.config['experiment']['name'],
                    config=self.config
                )
                self.tracking_client = wandb
                self.logger.info(f"Initialized W&B tracking with project '{project_name}'")
            except ImportError:
                self.logger.warning("wandb package not found. Install with 'pip install wandb'")
        else:
            self.logger.warning(f"Unknown tracking platform: {platform}")
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '') -> Dict[str, Any]:
        """
        Flatten a nested dictionary for logging parameters.
        
        Args:
            d: The dictionary to flatten
            parent_key: The parent key for nested dictionaries
            
        Returns:
            Flattened dictionary with keys in the format "parent.child"
        """
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}.{k}" if parent_key else k
            
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key).items())
            elif isinstance(v, (list, tuple)):
                # Convert lists to JSON strings for parameter logging
                items.append((new_key, json.dumps(v)))
            else:
                items.append((new_key, v))
                
        return dict(items)
    
    def _load_environment_class(self, environment_type: str) -> type:
        """
        Dynamically load the environment class based on the environment type.
        
        Args:
            environment_type: The type of environment to load
            
        Returns:
            The environment class
        """
        # Map environment types to module paths
        environment_modules = {
            'prisoners_dilemma': 'prisoners_dilemma.environment',
            # Add more environments here
        }
        
        if environment_type not in environment_modules:
            raise ValueError(f"Unknown environment type: {environment_type}")
        
        module_path = environment_modules[environment_type]
        module = importlib.import_module(module_path)
        
        # Assume the environment class is named "Simulation"
        return getattr(module, "Simulation")
    
    def _load_model_and_embedder(self, model_config: Dict[str, Any]) -> Tuple[Any, Any]:
        """
        Load the language model and embedder based on configuration.
        
        Args:
            model_config: Configuration for the model
            
        Returns:
            Tuple of (model, embedder)
        """
        # Import model class from the models directory
        from models.concordia_models import GoodfireModel
        
        # Create agent model
        agent_model = GoodfireModel(
            model_name=model_config['agent_model']['name'],
            system_prompt=model_config['agent_model']['system_prompt'],
            max_tokens=model_config['agent_model']['max_tokens']
        )
        
        # Create game master model (might be the same as agent model)
        gm_model = GoodfireModel(
            model_name=model_config['gm_model']['name'],
            system_prompt=model_config['gm_model']['system_prompt'],
            max_tokens=model_config['gm_model']['max_tokens']
        )
        
        # For now, use a simple embedder that returns random vectors
        # In a real implementation, you would use a proper embedding model
        def simple_embedder(text: str) -> np.ndarray:
            np.random.seed(hash(text) % 2**32)
            return np.random.rand(384)  # 384-dimensional embeddings
        
        return {
            'agent_model': agent_model,
            'gm_model': gm_model
        }, simple_embedder
    
    def _run_single_trial(self, 
                         config: Dict[str, Any], 
                         trial_index: int,
                         run_id: str = None) -> Dict[str, Any]:
        """
        Run a single trial with the given configuration.
        
        Args:
            config: The configuration for this trial
            trial_index: The index of this trial
            run_id: An optional run ID for tracking
            
        Returns:
            Dict containing the results of the trial
        """
        # Set random seed based on trial index
        random_seed = config['execution']['random_seed'] + trial_index
        random.seed(random_seed)
        np.random.seed(random_seed)
        
        self.logger.info(f"Starting trial {trial_index} with seed {random_seed}")
        
        # Load environment class
        environment_class = self._load_environment_class(config['environment']['type'])
        
        # Load model and embedder
        models, embedder = self._load_model_and_embedder(config['model'])
        
        # Create measurements
        from concordia.utils.measurements import Measurements
        measurements = Measurements()
        
        # Initialize environment
        try:
            environment = environment_class(
                gm_model=models['gm_model'],
                agent_model=models['agent_model'],
                embedder=embedder,
                measurements=measurements,
                seed=random_seed
            )
            
            # Run the simulation
            start_time = time.time()
            simulation_outcome, html_log = environment()
            end_time = time.time()
            
            # Process results
            results = {
                'trial_index': trial_index,
                'seed': random_seed,
                'runtime_seconds': end_time - start_time,
                'outcome': simulation_outcome,
                'html_log': html_log,
                'success': True,
                'error': None
            }
            
            # Save HTML log to file
            html_log_path = os.path.join(
                self.output_dir, 
                f"trial_{trial_index}_config_{config['sweep_metadata']['combination_name'] if 'sweep_metadata' in config else 'default'}.html"
            )
            with open(html_log_path, 'w') as f:
                f.write(html_log)
                
            # If tracking is enabled, log metrics
            if self.tracking_client and self.config['experiment']['tracking']['enabled']:
                self._log_trial_metrics(results, config, trial_index, run_id)
                
            return results
                
        except Exception as e:
            self.logger.error(f"Error in trial {trial_index}: {str(e)}", exc_info=True)
            return {
                'trial_index': trial_index,
                'seed': random_seed,
                'success': False,
                'error': str(e)
            }
    
    def _log_trial_metrics(self, 
                          results: Dict[str, Any], 
                          config: Dict[str, Any],
                          trial_index: int,
                          run_id: str = None) -> None:
        """
        Log metrics from a trial to the tracking system.
        
        Args:
            results: The results from the trial
            config: The configuration used for the trial
            trial_index: The index of the trial
            run_id: An optional run ID for tracking
        """
        if not self.tracking_client or not self.config['experiment']['tracking']['enabled']:
            return
            
        platform = self.config['experiment']['tracking']['platform']
        
        # Extract metrics from the results
        metrics = {}
        
        if results['success'] and 'outcome' in results:
            # Add agent scores
            if hasattr(results['outcome'], 'agent_scores'):
                for agent_name, score in results['outcome'].agent_scores.items():
                    metrics[f"score_{agent_name}"] = score
                    
                # Calculate average score
                metrics["avg_score"] = sum(results['outcome'].agent_scores.values()) / len(results['outcome'].agent_scores)
            
            # Add runtime
            metrics["runtime_seconds"] = results["runtime_seconds"]
            
            # Add custom metrics from the metadata if they exist
            if hasattr(results['outcome'], 'metadata'):
                for key, value in results['outcome'].metadata.items():
                    if isinstance(value, (int, float)):
                        metrics[key] = value
        
        # Log metrics to the appropriate platform
        if platform == 'mlflow':
            for key, value in metrics.items():
                self.tracking_client.log_metric(key, value, step=trial_index)
                
            # Log artifacts
            if 'html_log' in results and results['success']:
                html_log_path = os.path.join(
                    self.output_dir, 
                    f"trial_{trial_index}_config_{config['sweep_metadata']['combination_name'] if 'sweep_metadata' in config else 'default'}.html"
                )
                self.tracking_client.log_artifact(html_log_path)
                
        elif platform == 'wandb':
            # For W&B, we can log all metrics at once
            metrics["trial_index"] = trial_index
            self.tracking_client.log(metrics)
            
            # Log artifacts
            if 'html_log' in results and results['success']:
                html_log_path = os.path.join(
                    self.output_dir, 
                    f"trial_{trial_index}_config_{config['sweep_metadata']['combination_name'] if 'sweep_metadata' in config else 'default'}.html"
                )
                self.tracking_client.save(html_log_path)
    
    def run_experiment(self) -> List[Dict[str, Any]]:
        """
        Run the experiment with all sweep configurations.
        
        Returns:
            List of results from all trials
        """
        self.logger.info(f"Starting experiment '{self.config['experiment']['name']}'")
        self.logger.info(f"Running {len(self.sweep_configs)} configuration(s) with {self.config['execution']['num_trials']} trial(s) each")
        
        all_results = []
        
        # Save a copy of the sweep configurations
        for i, sweep_config in enumerate(self.sweep_configs):
            config_save_path = os.path.join(self.output_dir, f"sweep_config_{i}.yaml")
            self.config_manager.save_config(sweep_config, config_save_path)
        
        # Run each sweep configuration
        for sweep_idx, sweep_config in enumerate(self.sweep_configs):
            config_name = sweep_config['sweep_metadata']['combination_name'] if 'sweep_metadata' in sweep_config else f"config_{sweep_idx}"
            self.logger.info(f"Running configuration {sweep_idx+1}/{len(self.sweep_configs)}: {config_name}")
            
            # Get the number of trials to run
            num_trials = sweep_config['execution']['num_trials']
            
            # Create a new run in MLflow for this sweep configuration
            run_id = None
            if self.tracking_client and self.config['experiment']['tracking']['platform'] == 'mlflow':
                mlflow.end_run()
                run = mlflow.start_run(run_name=f"{self.config['experiment']['name']}_{config_name}")
                run_id = run.info.run_id
                
                # Log the sweep configuration
                mlflow.log_params(self._flatten_dict(sweep_config, parent_key="sweep"))
            
            # Run the trials
            if self.use_ray and sweep_config['execution']['parallel']:
                # Use Ray for parallel execution
                @ray.remote
                def ray_run_trial(config, trial_idx, run_id):
                    return self._run_single_trial(config, trial_idx, run_id)
                
                # Submit all trials to Ray
                ray_futures = [ray_run_trial.remote(sweep_config, i, run_id) for i in range(num_trials)]
                
                # Get results as they complete
                config_results = []
                for _ in tqdm(range(len(ray_futures)), desc=f"Running trials for {config_name}"):
                    done_id, ray_futures = ray.wait(ray_futures, num_returns=1)
                    result = ray.get(done_id[0])
                    config_results.append(result)
                    
            else:
                # Use concurrent.futures for parallel execution
                config_results = []
                
                if sweep_config['execution']['parallel'] and num_trials > 1:
                    with concurrent.futures.ProcessPoolExecutor(
                        max_workers=min(sweep_config['execution']['num_workers'], num_trials)
                    ) as executor:
                        futures = {
                            executor.submit(self._run_single_trial, sweep_config, i, run_id): i 
                            for i in range(num_trials)
                        }
                        
                        for future in tqdm(
                            concurrent.futures.as_completed(futures), 
                            total=len(futures),
                            desc=f"Running trials for {config_name}"
                        ):
                            result = future.result()
                            config_results.append(result)
                else:
                    # Run trials sequentially
                    for i in tqdm(range(num_trials), desc=f"Running trials for {config_name}"):
                        result = self._run_single_trial(sweep_config, i, run_id)
                        config_results.append(result)
            
            # Process and log results for this configuration
            self._process_config_results(config_results, sweep_config, config_name)
            
            # Add results to the overall results list
            all_results.extend(config_results)
        
        # End the MLflow run
        if self.tracking_client and self.config['experiment']['tracking']['platform'] == 'mlflow':
            mlflow.end_run()
        
        # Process all results together
        self._process_all_results(all_results)
        
        return all_results
    
    def _process_config_results(self, 
                               results: List[Dict[str, Any]], 
                               config: Dict[str, Any],
                               config_name: str) -> None:
        """
        Process and log the results from a single configuration.
        
        Args:
            results: The results from all trials of the configuration
            config: The configuration used
            config_name: The name of the configuration
        """
        # Count successful trials
        successful_trials = [r for r in results if r['success']]
        success_rate = len(successful_trials) / len(results) if results else 0
        
        self.logger.info(f"Configuration {config_name} complete: {len(successful_trials)}/{len(results)} trials successful ({success_rate:.1%})")
        
        # Skip further processing if no successful trials
        if not successful_trials:
            self.logger.warning(f"No successful trials for configuration {config_name}")
            return
        
        # Convert results to DataFrame for analysis
        results_df = self._results_to_dataframe(successful_trials, config)
        
        # Save results to CSV
        results_path = os.path.join(self.output_dir, f"results_{config_name}.csv")
        results_df.to_csv(results_path, index=False)
        
        # Log to tracking system
        if self.tracking_client and self.config['experiment']['tracking']['enabled']:
            platform = self.config['experiment']['tracking']['platform']
            
            if platform == 'mlflow':
                # Log the summary statistics
                for col in results_df.select_dtypes(include=['number']).columns:
                    if col != 'trial_index':
                        self.tracking_client.log_metric(f"mean_{col}", results_df[col].mean())
                        self.tracking_client.log_metric(f"std_{col}", results_df[col].std())
                        self.tracking_client.log_metric(f"min_{col}", results_df[col].min())
                        self.tracking_client.log_metric(f"max_{col}", results_df[col].max())
                
                # Log the results CSV
                self.tracking_client.log_artifact(results_path)
                
            elif platform == 'wandb':
                # Log summary statistics
                summary_metrics = {}
                for col in results_df.select_dtypes(include=['number']).columns:
                    if col != 'trial_index':
                        summary_metrics[f"mean_{col}"] = results_df[col].mean()
                        summary_metrics[f"std_{col}"] = results_df[col].std()
                        summary_metrics[f"min_{col}"] = results_df[col].min()
                        summary_metrics[f"max_{col}"] = results_df[col].max()
                
                self.tracking_client.log(summary_metrics)
                self.tracking_client.save(results_path)
    
    def _process_all_results(self, all_results: List[Dict[str, Any]]) -> None:
        """
        Process and analyze the results from all configurations.
        
        Args:
            all_results: The results from all trials and configurations
        """
        # Count successful trials
        successful_trials = [r for r in all_results if r['success']]
        success_rate = len(successful_trials) / len(all_results) if all_results else 0
        
        self.logger.info(f"Experiment complete: {len(successful_trials)}/{len(all_results)} trials successful ({success_rate:.1%})")
        
        # Skip further processing if no successful trials
        if not successful_trials:
            self.logger.warning("No successful trials in the entire experiment")
            return
        
        # Convert all results to a single DataFrame
        all_results_df = pd.concat([
            self._results_to_dataframe([r], None) for r in successful_trials
        ], ignore_index=True)
        
        # Save combined results
        all_results_path = os.path.join(self.output_dir, "all_results.csv")
        all_results_df.to_csv(all_results_path, index=False)
        
        # Create visualizations if enabled
        if self.config['visualization']['enabled']:
            self._create_visualizations(all_results_df)
        
        # Log combined results to tracking
        if self.tracking_client and self.config['experiment']['tracking']['enabled']:
            platform = self.config['experiment']['tracking']['platform']
            
            if platform == 'mlflow':
                # Log the all_results.csv file
                mlflow.log_artifact(all_results_path)
                
            elif platform == 'wandb':
                # Log the all_results.csv file
                self.tracking_client.save(all_results_path)
    
    def _results_to_dataframe(self, 
                             results: List[Dict[str, Any]], 
                             config: Optional[Dict[str, Any]]) -> pd.DataFrame:
        """
        Convert experiment results to a pandas DataFrame.
        
        Args:
            results: The results to convert
            config: The configuration used (optional)
            
        Returns:
            DataFrame containing the results
        """
        rows = []
        
        for result in results:
            if not result['success']:
                continue
                
            row = {
                'trial_index': result['trial_index'],
                'seed': result['seed'],
                'runtime_seconds': result['runtime_seconds'],
            }
            
            # Add configuration information if provided
            if config:
                if 'sweep_metadata' in config:
                    row['config_name'] = config['sweep_metadata']['combination_name']
                
                # Add key configuration parameters
                if 'environment' in config:
                    for key, value in config['environment'].items():
                        if not isinstance(value, dict):
                            row[f'env_{key}'] = value
                    
                    if 'settings' in config['environment']:
                        for key, value in config['environment']['settings'].items():
                            if not isinstance(value, dict):
                                row[f'env_setting_{key}'] = value
            
            # Add agent scores
            if 'outcome' in result and hasattr(result['outcome'], 'agent_scores'):
                for agent_name, score in result['outcome'].agent_scores.items():
                    row[f'score_{agent_name}'] = score
                
                # Calculate average score
                row['avg_score'] = sum(result['outcome'].agent_scores.values()) / len(result['outcome'].agent_scores)
            
            # Add metadata from outcome if it exists
            if 'outcome' in result and hasattr(result['outcome'], 'metadata'):
                for key, value in result['outcome'].metadata.items():
                    if not isinstance(value, (dict, list)):
                        row[f'metadata_{key}'] = value
            
            rows.append(row)
        
        return pd.DataFrame(rows)
    
    def _create_visualizations(self, results_df: pd.DataFrame) -> None:
        """
        Create visualizations from experiment results.
        
        Args:
            results_df: DataFrame containing experiment results
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        # Create visualization directory
        vis_dir = os.path.join(self.output_dir, "visualizations")
        os.makedirs(vis_dir, exist_ok=True)
        
        # Set up plotting style
        sns.set_theme(style="whitegrid")
        plt.rcParams.update({'font.size': 12})
        
        # Create visualization for agent scores
        score_cols = [col for col in results_df.columns if col.startswith('score_')]
        
        if score_cols:
            plt.figure(figsize=(12, 8))
            
            # Create a box plot of scores by agent
            score_data = results_df.melt(
                id_vars=['trial_index', 'config_name'] if 'config_name' in results_df.columns else ['trial_index'],
                value_vars=score_cols,
                var_name='agent',
                value_name='score'
            )
            score_data['agent'] = score_data['agent'].str.replace('score_', '')
            
            if 'config_name' in score_data.columns:
                # Group by configuration
                g = sns.boxplot(x='agent', y='score', hue='config_name', data=score_data)
                plt.legend(title='Configuration', bbox_to_anchor=(1.05, 1), loc='upper left')
            else:
                # Simple box plot by agent
                g = sns.boxplot(x='agent', y='score', data=score_data)
            
            plt.title('Score Distribution by Agent')
            plt.xlabel('Agent')
            plt.ylabel('Score')
            plt.tight_layout()
            plt.savefig(os.path.join(vis_dir, 'agent_scores.png'), dpi=300)
            plt.savefig(os.path.join(vis_dir, 'agent_scores.pdf'))
            plt.close()
            
        # Create other visualizations as needed
        # ...
        
        # Log visualizations to tracking platform
        if self.tracking_client and self.config['experiment']['tracking']['enabled']:
            platform = self.config['experiment']['tracking']['platform']
            
            if platform == 'mlflow':
                # Log the visualization directory
                mlflow.log_artifacts(vis_dir, artifact_path="visualizations")
                
            elif platform == 'wandb':
                # Log individual visualizations
                for filename in os.listdir(vis_dir):
                    if filename.endswith(('.png', '.pdf')):
                        self.tracking_client.log({
                            filename: wandb.Image(os.path.join(vis_dir, filename))
                        })
    
    def __del__(self):
        """Clean up resources when the object is destroyed."""
        self._cleanup() 