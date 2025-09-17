"""
Configuration for performance tests.
"""
import pytest
import time
import psutil
import sys
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class PerformanceMetrics:
    """Class to track performance metrics during tests."""
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all metrics."""
        self.start_time = None
        self.end_time = None
        self.start_memory = None
        self.end_memory = None
        self.start_cpu = None
        self.end_cpu = None
    
    def start_measurement(self):
        """Start measuring performance."""
        self.start_time = time.perf_counter()
        self.start_memory = psutil.virtual_memory().used
        self.start_cpu = psutil.cpu_percent()
    
    def end_measurement(self):
        """End measuring performance."""
        self.end_time = time.perf_counter()
        self.end_memory = psutil.virtual_memory().used
        self.end_cpu = psutil.cpu_percent()
    
    def get_results(self) -> Dict[str, Any]:
        """Get measurement results."""
        if self.start_time is None or self.end_time is None:
            return {}
        
        return {
            'duration_seconds': self.end_time - self.start_time,
            'memory_usage_mb': (self.end_memory - self.start_memory) / 1024 / 1024,
            'cpu_usage_percent': self.end_cpu - self.start_cpu if self.start_cpu else None
        }


@pytest.fixture
def performance_metrics():
    """Fixture to provide performance metrics tracking."""
    return PerformanceMetrics()


@pytest.fixture
def benchmark_config():
    """Configuration for benchmarks."""
    return {
        'min_rounds': 3,
        'max_time': 30.0,  # Maximum time in seconds for benchmarks
        'warmup_rounds': 1
    }