"""
Configuration manager for multi-agent experiment framework.
Provides functionality to load, validate, and merge configuration files.
"""

import os
import yaml
import copy
from typing import Dict, Any, List, Optional, Union
from dataclasses import dataclass
import itertools


@dataclass
class ConfigEntry:
    """Data class for storing configuration entries with metadata."""
    name: str
    value: Any
    source: str  # Which config file this entry came from
    is_override: bool = False


class ConfigManager:
    """
    Manages configuration for experiments, supporting layered configs and parameter sweeps.
    """
    
    BASE_CONFIG_PATH = "experiments/configs/base_config.yaml"
    
    def __init__(self, config_path: str = None, base_config_path: Optional[str] = None):
        """
        Initialize a configuration manager.
        
        Args:
            config_path: Path to the experiment-specific config file
            base_config_path: Optional path to the base config file. If not provided, 
                             will use the default base config path.
        """
        self.base_config_path = base_config_path or os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 
            self.BASE_CONFIG_PATH
        )
        
        # Load the base configuration
        self.base_config = self._load_yaml_config(self.base_config_path)
        
        # If a specific config is provided, load and merge it
        self.experiment_config = {}
        if config_path:
            self.load_experiment_config(config_path)
            
        # Generated configurations for parameter sweeps
        self.sweep_configs = []
            
    def _load_yaml_config(self, config_path: str) -> Dict[str, Any]:
        """Load a YAML configuration file."""
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
            
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
            
    def load_experiment_config(self, config_path: str) -> None:
        """
        Load an experiment-specific configuration and merge it with the base config.
        
        Args:
            config_path: Path to the experiment config file
        """
        self.experiment_config = self._load_yaml_config(config_path)
        
    def get_merged_config(self) -> Dict[str, Any]:
        """
        Get the merged configuration (base + experiment).
        
        Returns:
            Dict containing the merged configuration
        """
        merged = copy.deepcopy(self.base_config)
        self._deep_update(merged, self.experiment_config)
        return merged
    
    def _deep_update(self, base_dict: Dict[str, Any], update_dict: Dict[str, Any]) -> None:
        """
        Recursively update a dictionary with values from another dictionary.
        
        Args:
            base_dict: The dictionary to update
            update_dict: The dictionary containing values to update with
        """
        for key, value in update_dict.items():
            if isinstance(value, dict) and key in base_dict and isinstance(base_dict[key], dict):
                self._deep_update(base_dict[key], value)
            else:
                base_dict[key] = value
                
    def generate_sweep_configs(self) -> List[Dict[str, Any]]:
        """
        Generate a list of configurations for parameter sweeps.
        
        Returns:
            List of configurations with all parameter combinations
        """
        merged_config = self.get_merged_config()
        
        # If no sweep parameters are defined, just return the merged config
        if 'sweep_params' not in merged_config:
            self.sweep_configs = [merged_config]
            return self.sweep_configs
            
        sweep_params = merged_config.pop('sweep_params', {})
        
        # Generate all combinations of parameters
        parameter_combinations = self._generate_parameter_combinations(sweep_params)
        
        # Create a config for each combination
        self.sweep_configs = []
        for params in parameter_combinations:
            config = copy.deepcopy(merged_config)
            
            # Apply the parameter values to the config
            for param_name, param_value in params.items():
                # If param_name is a nested parameter (e.g. 'environment.settings.rounds')
                if '.' in param_name:
                    self._set_nested_param(config, param_name, param_value)
                else:
                    config[param_name] = param_value
                    
            # Add metadata about the parameter combination
            config['sweep_metadata'] = {
                'combination_name': '_'.join([f"{k}_{v['name']}" for k, v in params.items() if isinstance(v, dict) and 'name' in v])
            }
            
            self.sweep_configs.append(config)
            
        return self.sweep_configs
    
    def _generate_parameter_combinations(self, sweep_params: Dict[str, List[Any]]) -> List[Dict[str, Any]]:
        """
        Generate all combinations of sweep parameters.
        
        Args:
            sweep_params: Dictionary of parameter names to lists of values
            
        Returns:
            List of dictionaries, each containing one combination of parameters
        """
        param_names = list(sweep_params.keys())
        param_values = [sweep_params[name] for name in param_names]
        
        combinations = []
        for values in itertools.product(*param_values):
            combination = {}
            for name, value in zip(param_names, values):
                combination[name] = value
            combinations.append(combination)
            
        return combinations
    
    def _set_nested_param(self, config: Dict[str, Any], param_path: str, value: Any) -> None:
        """
        Set a nested parameter in the config dictionary.
        
        Args:
            config: The config dictionary to modify
            param_path: The path to the parameter (e.g. 'environment.settings.rounds')
            value: The value to set
        """
        parts = param_path.split('.')
        current = config
        
        # Navigate to the nested location
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
            
        # Set the value
        current[parts[-1]] = value
    
    def save_config(self, config: Dict[str, Any], output_path: str) -> None:
        """
        Save a configuration to a YAML file.
        
        Args:
            config: The configuration to save
            output_path: The path to save the configuration to
        """
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'w') as f:
            yaml.dump(config, f, default_flow_style=False)
            
    def validate_config(self, config: Dict[str, Any]) -> List[str]:
        """
        Validate a configuration against a schema.
        
        Args:
            config: The configuration to validate
            
        Returns:
            List of validation errors, empty if valid
        """
        # TODO: Implement schema validation
        # This could use jsonschema or pydantic for validation
        return [] 