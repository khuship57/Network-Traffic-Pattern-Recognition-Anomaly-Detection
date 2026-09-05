"""
Hardware optimization utilities for automatic GPU/CPU detection and management
"""
import os
import platform
from typing import Dict, Any, Optional
import logging

try:
    import psutil
    PSUTIL_AVAILABLE = True
except ImportError:
    PSUTIL_AVAILABLE = False

try:
    import GPUtil
    GPUTIL_AVAILABLE = True
except ImportError:
    GPUTIL_AVAILABLE = False

logger = logging.getLogger(__name__)


class HardwareManager:
    """Manages hardware resources and optimization"""

    def __init__(self):
        self.system_info = self._get_system_info()
        self.gpu_available = self._check_gpu_availability()
        self.optimal_device = self._determine_optimal_device()

    def _get_system_info(self) -> Dict[str, Any]:
        """Get comprehensive system information"""
        info = {
            'platform': platform.platform(),
            'processor': platform.processor(),
            'python_version': platform.python_version()
        }
        if PSUTIL_AVAILABLE:
            info['cpu_count'] = psutil.cpu_count()
            info['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
            info['memory_total'] = psutil.virtual_memory().total
            info['memory_available'] = psutil.virtual_memory().available
        else:
            info['cpu_count'] = os.cpu_count() or 4
        return info

    def _check_gpu_availability(self) -> bool:
        """Check if GPU is available and accessible"""
        try:
            import tensorflow as tf
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                logger.info(f"Found {len(gpus)} GPU(s) via TensorFlow")
                return True
        except Exception:
            pass

        if GPUTIL_AVAILABLE:
            try:
                gpus = GPUtil.getGPUs()
                if gpus:
                    logger.info(f"Found {len(gpus)} GPU(s): {[gpu.name for gpu in gpus]}")
                    return True
            except Exception:
                pass

        logger.info("Using CPU for computation")
        return False

    def _determine_optimal_device(self) -> str:
        """Determine the optimal device for computation"""
        if self.gpu_available:
            return "gpu"
        return "cpu"

    def get_optimal_n_jobs(self) -> int:
        """Get optimal number of parallel jobs based on hardware"""
        cpu_count = self.system_info.get('cpu_count', 4)
        return max(1, cpu_count - 1)

    def get_memory_usage(self) -> Dict[str, float]:
        """Get current memory usage statistics"""
        if PSUTIL_AVAILABLE:
            memory = psutil.virtual_memory()
            return {
                'total_gb': memory.total / (1024**3),
                'available_gb': memory.available / (1024**3),
                'used_percent': memory.percent,
                'free_gb': memory.free / (1024**3)
            }
        return {'total_gb': 0.0, 'available_gb': 0.0, 'used_percent': 0.0, 'free_gb': 0.0}


hardware_manager = HardwareManager()
