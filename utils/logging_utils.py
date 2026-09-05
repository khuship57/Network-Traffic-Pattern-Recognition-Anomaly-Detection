"""
Logging utilities for comprehensive model performance tracking
"""
import logging
import os
from datetime import datetime
from typing import Optional
import json

def setup_logging(
    log_level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs"
) -> None:
    """Setup comprehensive logging configuration"""
    
    # Create logs directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Set log level
    level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
    )
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)
    
    # Clear existing handlers
    root_logger.handlers.clear()
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler
    if log_file is None:
        log_file = f"network_anomaly_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    
    file_path = os.path.join(log_dir, log_file)
    file_handler = logging.FileHandler(file_path)
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    root_logger.addHandler(file_handler)
    
    # Performance logger for model metrics
    perf_logger = logging.getLogger('performance')
    perf_handler = logging.FileHandler(os.path.join(log_dir, 'performance.log'))
    perf_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
    perf_logger.addHandler(perf_handler)
    perf_logger.setLevel(logging.INFO)
    perf_logger.propagate = False

def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module"""
    return logging.getLogger(name)

class PerformanceLogger:
    """Specialized logger for model performance metrics"""
    
    def __init__(self):
        self.logger = logging.getLogger('performance')
    
    def log_model_performance(
        self,
        model_name: str,
        metrics: dict,
        training_time: float,
        inference_time: float,
        memory_usage: dict
    ) -> None:
        """Log comprehensive model performance metrics"""
        
        performance_data = {
            'timestamp': datetime.now().isoformat(),
            'model_name': model_name,
            'metrics': metrics,
            'training_time_seconds': training_time,
            'inference_time_seconds': inference_time,
            'memory_usage_mb': memory_usage,
            'precision': metrics.get('precision', 0),
            'recall': metrics.get('recall', 0),
            'f1_score': metrics.get('f1_score', 0),
            'accuracy': metrics.get('accuracy', 0)
        }
        
        self.logger.info(json.dumps(performance_data, indent=2))
    
    def log_data_info(
        self,
        dataset_size: int,
        features_count: int,
        attack_types: int,
        class_distribution: dict
    ) -> None:
        """Log dataset information"""
        
        data_info = {
            'timestamp': datetime.now().isoformat(),
            'dataset_size': dataset_size,
            'features_count': features_count,
            'attack_types_count': attack_types,
            'class_distribution': class_distribution
        }
        
        self.logger.info(f"Dataset Info: {json.dumps(data_info, indent=2)}")
    
    def log_preprocessing_info(
        self,
        preprocessing_steps: list,
        feature_engineering_time: float,
        encoding_time: float
    ) -> None:
        """Log preprocessing information"""
        
        prep_info = {
            'timestamp': datetime.now().isoformat(),
            'preprocessing_steps': preprocessing_steps,
            'feature_engineering_time_seconds': feature_engineering_time,
            'encoding_time_seconds': encoding_time
        }
        
        self.logger.info(f"Preprocessing Info: {json.dumps(prep_info, indent=2)}")

# Global performance logger instance
performance_logger = PerformanceLogger()
