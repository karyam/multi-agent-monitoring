import logging
import os
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime


def setup_experiment_logger(
    experiment_name: str,
    output_dir: str,
    log_filename: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger for experiment results.
    
    Args:
        experiment_name: Name of the experiment (used as logger name)
        output_dir: Directory where logs will be saved
        log_filename: Optional custom filename for the log file
        
    Returns:
        Configured logger object
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Configure logger
    logger = logging.getLogger(experiment_name)
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if any
    if logger.handlers:
        logger.handlers.clear()
    
    # Create file handler with a default filename if not provided
    if log_filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"{experiment_name}_{timestamp}.log"
    
    file_handler = logging.FileHandler(os.path.join(output_dir, log_filename))
    
    # Set formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # Add handler to logger
    logger.addHandler(file_handler)
    
    return logger


def log_experiment_config(
    logger: logging.Logger,
    config: Dict[str, Any]
) -> None:
    """
    Log the experiment configuration parameters.
    
    Args:
        logger: The logger to use
        config: Dictionary containing experiment configuration
    """
    logger.info("Experiment Configuration:")
    for key, value in config.items():
        logger.info(f"  {key}: {value}")


def log_trial_results(
    logger: logging.Logger,
    trial_results: List[Dict[str, Any]],
    score_attribute_path: str = "outcome.agent_scores",
    trial_index_key: str = "trial"
) -> None:
    """
    Log individual trial results.
    
    Args:
        logger: The logger to use
        trial_results: List of trial result dictionaries
        score_attribute_path: Path to scores in the result dict (using dot notation)
        trial_index_key: Key to use for trial indexing
    """
    logger.info("Individual Trial Results:")
    
    for trial_data in trial_results:
        # Get trial number
        trial_num = trial_data.get(trial_index_key, "Unknown")
        logger.info(f"Trial {trial_num} results:")
        
        # Extract scores using the provided attribute path
        current = trial_data
        for attr in score_attribute_path.split('.'):
            if attr in current:
                current = current[attr]
            else:
                logger.warning(f"Could not find {attr} in trial data structure")
                current = {}
                break
        
        # Log scores
        if isinstance(current, dict):
            for entity, score in current.items():
                logger.info(f"  {entity}: {score}")
        else:
            logger.warning(f"Expected dict of scores, got {type(current)}")


def log_summary_statistics(
    logger: logging.Logger,
    data_frame: pd.DataFrame,
    groupby_column: str,
    score_column: str,
    agg_functions: List[str] = ['mean', 'std', 'min', 'max']
) -> None:
    """
    Log summary statistics from a DataFrame.
    
    Args:
        logger: The logger to use
        data_frame: Pandas DataFrame containing the experiment results
        groupby_column: Column to group by (e.g., 'agent_type')
        score_column: Column containing score values to analyze
        agg_functions: List of aggregation functions to apply
    """
    if data_frame.empty:
        logger.warning("Empty DataFrame provided, cannot calculate statistics")
        return
        
    try:
        stats = data_frame.groupby(groupby_column)[score_column].agg(agg_functions)
        logger.info(f"Summary Statistics (grouped by {groupby_column}):")
        logger.info(f"\n{stats.to_string()}")
    except Exception as e:
        logger.error(f"Error calculating statistics: {str(e)}")


def export_results(
    data_frame: pd.DataFrame,
    output_dir: str,
    filename_prefix: str,
    formats: List[str] = ['csv']
) -> Dict[str, str]:
    """
    Export results to various formats.
    
    Args:
        data_frame: Pandas DataFrame containing the results
        output_dir: Directory where files will be saved
        filename_prefix: Prefix for output filenames
        formats: List of formats to export to ('csv', 'json', 'excel', 'pickle')
        
    Returns:
        Dictionary mapping format to file path
    """
    os.makedirs(output_dir, exist_ok=True)
    exported_files = {}
    
    for fmt in formats:
        if fmt.lower() == 'csv':
            path = os.path.join(output_dir, f"{filename_prefix}.csv")
            data_frame.to_csv(path, index=False)
            exported_files['csv'] = path
            
        elif fmt.lower() == 'json':
            path = os.path.join(output_dir, f"{filename_prefix}.json")
            data_frame.to_json(path, orient='records', indent=2)
            exported_files['json'] = path
            
        elif fmt.lower() == 'excel':
            path = os.path.join(output_dir, f"{filename_prefix}.xlsx")
            data_frame.to_excel(path, index=False)
            exported_files['excel'] = path
            
        elif fmt.lower() == 'pickle':
            path = os.path.join(output_dir, f"{filename_prefix}.pkl")
            data_frame.to_pickle(path)
            exported_files['pickle'] = path
    
    return exported_files


# Example usage function showing how the above functions would be used together
def log_experiment(
    experiment_name: str,
    config: Dict[str, Any],
    trial_results: List[Dict[str, Any]],
    results_df: pd.DataFrame,
    output_dir: str,
    score_attribute_path: str = "outcome.agent_scores",
    groupby_column: str = "agent_type",
    score_column: str = "final_score",
    export_formats: List[str] = ['csv']
) -> logging.Logger:
    """
    Comprehensive logging function for experiment results.
    
    Args:
        experiment_name: Name of the experiment
        config: Experiment configuration dictionary
        trial_results: List of trial result dictionaries
        results_df: DataFrame containing processed results
        output_dir: Directory where logs and exports will be saved
        score_attribute_path: Path to scores in the trial result dictionaries
        groupby_column: Column to group by for statistics
        score_column: Column containing score values
        export_formats: Formats to export results to
        
    Returns:
        Configured logger object
    """
    # Setup logger
    logger = setup_experiment_logger(experiment_name, output_dir)
    
    # Log configuration
    log_experiment_config(logger, config)
    
    # Log trial results
    log_trial_results(logger, trial_results, score_attribute_path)
    
    # Log summary statistics
    log_summary_statistics(logger, results_df, groupby_column, score_column)
    
    # Export results
    exported_files = export_results(
        results_df, 
        output_dir, 
        f"{experiment_name}_results", 
        export_formats
    )
    
    # Log export locations
    logger.info("Results exported to:")
    for fmt, path in exported_files.items():
        logger.info(f"  {fmt}: {path}")
    
    return logger