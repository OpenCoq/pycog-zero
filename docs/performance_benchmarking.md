# PyCog-Zero Performance Benchmarking Guide

Comprehensive performance benchmarking and optimization guide for PyCog-Zero cognitive systems, including cpp2py conversion pipeline benchmarking and cognitive component performance analysis.

## Overview

The `scripts/cpp2py_conversion_pipeline.py test` command now supports comprehensive performance benchmarking for OpenCog component integration and conversion operations.

## Usage

### Basic Performance Testing

Run performance benchmarks only (no integration tests):
```bash
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only
```

Run both integration and performance tests:
```bash
python3 scripts/cpp2py_conversion_pipeline.py test --performance
```

### Advanced Options

#### Benchmark Configuration
```bash
# Set minimum number of benchmark rounds
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --benchmark-rounds 5

# Set warmup iterations
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --benchmark-warmup 3

# Enable verbose output
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --verbose
```

#### Save and Compare Results
```bash
# Save benchmark results to file
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --benchmark-save results.json

# Compare with previous results
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --benchmark-compare results.json
```

#### Generate Performance Reports
```bash
# Generate comprehensive performance report
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --report performance_report.json
```

## Performance Test Categories

### 1. Pipeline Performance Tests
Located in `tests/performance/test_pipeline_performance.py`

- **Pipeline initialization performance**: Measures time to initialize the conversion pipeline
- **Component definitions loading**: Tests loading of OpenCog component definitions
- **Phase report generation**: Benchmarks generating status reports for development phases
- **Dependency validation**: Measures dependency checking performance
- **Memory usage tests**: Tracks memory consumption during operations

### 2. Component-Specific Performance Tests
Located in `tests/performance/test_component_performance.py`

#### Individual Component Tests:
- **cogutil performance**: Foundation component validation and bindings
- **atomspace performance**: Core hypergraph storage performance
- **URE performance**: Unified Rule Engine dependency validation
- **PLN performance**: Probabilistic Logic Networks advanced operations
- **OpenCog performance**: Complete integration component testing

#### Cross-Component Tests:
- **Phase-based validation**: Performance across development phases
- **Dependency graph traversal**: Complex dependency chain validation
- **Bulk operations**: Scalability testing with all components

### 3. CLI Performance Tests
- **Status command performance**: Benchmarks `status` command execution
- **Help command performance**: Tests CLI help generation speed
- **Validation command performance**: Measures component validation commands

## Performance Metrics

The benchmarking system tracks:

- **Execution time** (min, max, mean, standard deviation)
- **Memory usage** (RSS memory consumption)
- **CPU utilization** during operations
- **Operations per second** (throughput)
- **Round statistics** (iterations and variance)

## Performance Report Structure

Generated reports include:

```json
{
  "summary": {
    "total_benchmarks": 31,
    "test_session_start": "2025-09-16T08:10:58.442803+00:00",
    "machine_info": {
      "processor": "x86_64",
      "python_version": "3.12.3",
      "cpu": {...}
    }
  },
  "categories": {
    "pipeline_performance": [...],
    "component_performance": [...],
    "cli_performance": [...],
    "memory_tests": [...],
    "scalability_tests": [...]
  },
  "performance_metrics": {
    "fastest_tests": [...],
    "slowest_tests": [...],
    "memory_efficient_tests": [...],
    "high_memory_tests": [...]
  }
}
```

## Example Commands

### Development Workflow Testing
```bash
# Quick performance check during development
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --benchmark-rounds 3

# Comprehensive performance analysis
python3 scripts/cpp2py_conversion_pipeline.py test --performance --report daily_performance.json

# Compare performance with baseline
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --benchmark-compare baseline.json
```

### CI/CD Integration
```bash
# Fast CI performance check
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --benchmark-rounds 1

# Generate performance report for CI artifacts
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --report ci_performance.json
```

## Interpreting Results

### Benchmark Output
The benchmark results show:
- **Min/Max times**: Range of execution times
- **Mean time**: Average execution time
- **Standard deviation**: Consistency of performance
- **Median**: Middle value, less affected by outliers
- **Operations per second**: Throughput measurement

### Performance Categories
- **Fast operations** (< 1ms): Basic validation and status checks
- **Medium operations** (1-100ms): Component processing and reports
- **Slow operations** (> 100ms): Complex dependency validation and Python bindings
- **Very slow operations** (> 1s): CLI commands and system operations

### Memory Analysis
Memory usage is tracked during operations to identify:
- Memory leaks in component processing
- High-memory operations requiring optimization
- Efficient operations suitable for batch processing

## Dependencies

The performance testing framework requires:
- `pytest-benchmark` - Core benchmarking functionality
- `pytest-asyncio` - Async test support
- `psutil` - System resource monitoring

Install with:
```bash
pip install pytest-benchmark pytest-asyncio psutil
```

## Customization

### Adding Custom Benchmarks

Create new benchmark tests in `tests/performance/`:

```python
def test_custom_operation_performance(benchmark):
    def custom_operation():
        # Your operation here
        return result
    
    result = benchmark(custom_operation)
    assert result is not None
```

### Custom Performance Metrics

Use the `performance_metrics` fixture for detailed tracking:

```python
def test_custom_metrics(performance_metrics):
    performance_metrics.start_measurement()
    
    # Your operations
    
    performance_metrics.end_measurement()
    results = performance_metrics.get_results()
    
    assert results['duration_seconds'] < 1.0
    assert results['memory_usage_mb'] < 50
```

## Best Practices

1. **Consistent Environment**: Run benchmarks in consistent environments for meaningful comparisons
2. **Warmup Rounds**: Use warmup iterations for accurate measurements of optimized code
3. **Multiple Rounds**: Run multiple rounds to account for system variance
4. **Baseline Comparisons**: Maintain baseline results for regression detection
5. **Resource Monitoring**: Monitor system resources during long-running benchmarks

## Troubleshooting

### Common Issues

- **High variance**: Increase warmup rounds or ensure system is idle
- **Out of memory**: Reduce test scope or increase available memory
- **Slow benchmarks**: Use `--benchmark-max-time` to limit execution time
- **Missing dependencies**: Install required packages (pytest-benchmark, psutil)

### Performance Debugging

Use verbose output to debug performance issues:
```bash
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --verbose
```

Monitor system resources during benchmarks:
```bash
# In another terminal
top -p $(pgrep -f pytest)
```

## Integration with Development Workflow

The performance benchmarking integrates with the existing cpp2py conversion pipeline:

1. **Phase 1 (Core Extensions)**: Performance baseline establishment
2. **Component Integration**: Per-component performance validation
3. **Dependency Optimization**: Dependency validation performance tuning
4. **Python Bindings**: Bindings performance measurement
5. **System Integration**: End-to-end performance validation

This enables developers to:
- Track performance regression during development
- Optimize critical code paths
- Validate performance requirements
- Compare different implementation approaches
- Monitor system resource usage

The performance benchmarking ensures that the PyCog-Zero cognitive architecture maintains optimal performance as OpenCog components are integrated.

---

## Cognitive Performance Benchmarking

### Cognitive System Performance Framework

Beyond the cpp2py conversion pipeline, PyCog-Zero includes comprehensive cognitive performance benchmarking for all integrated cognitive tools and components.

#### Core Performance Metrics

The PyCog-Zero cognitive system tracks several key performance indicators:

##### Response Time Metrics
- **Reasoning Response Time**: Target <2 seconds for complex queries
- **Memory Operations**: Target <100ms for storage/retrieval
- **Attention Allocation**: Target <50ms for priority updates
- **Tool Execution**: Target <500ms for cognitive tool operations

##### Throughput Metrics
- **Queries per Second**: Concurrent cognitive query processing
- **Memory Operations per Second**: Knowledge storage/retrieval rate
- **Agent Communication**: Multi-agent message passing rate

##### Resource Utilization
- **CPU Usage**: Target <80% under normal load
- **Memory Usage**: Target <4GB for standard cognitive operations
- **GPU Utilization**: When neural processing is enabled
- **Network Bandwidth**: For distributed agent scenarios

### Automated Cognitive Benchmark Suite

```python
#!/usr/bin/env python3
"""PyCog-Zero Cognitive Performance Benchmark Suite"""

import asyncio
import time
import statistics
import json
from datetime import datetime
from python.tools.cognitive_reasoning import CognitiveReasoningTool
from python.tools.cognitive_memory import CognitiveMemoryTool

class CognitiveBenchmarkSuite:
    """Comprehensive cognitive performance benchmarking."""
    
    async def run_cognitive_benchmarks(self):
        """Run complete cognitive benchmark suite."""
        
        print("🧠 PyCog-Zero Cognitive Performance Benchmarks")
        print("=" * 60)
        
        benchmarks = [
            ("Reasoning Performance", self.benchmark_reasoning),
            ("Memory Performance", self.benchmark_memory),
            ("Concurrent Operations", self.benchmark_concurrent),
            ("Stress Testing", self.benchmark_stress_test)
        ]
        
        results = {}
        
        for category, benchmark_func in benchmarks:
            print(f"\n🧪 Running {category}...")
            try:
                result = await benchmark_func()
                results[category] = result
                print(f"✅ {category}: Completed")
            except Exception as e:
                print(f"❌ {category}: Failed - {e}")
                results[category] = {"error": str(e)}
        
        await self.generate_cognitive_report(results)
        return results
    
    async def benchmark_reasoning(self):
        """Benchmark cognitive reasoning performance."""
        
        reasoning_tool = CognitiveReasoningTool()
        
        test_scenarios = [
            {
                "name": "Simple Logic",
                "queries": [
                    "What is 2 + 2?",
                    "If A then B. A is true. What about B?",
                    "All cats are animals. Is Fluffy an animal if Fluffy is a cat?"
                ],
                "target_time": 0.5
            },
            {
                "name": "Complex Analysis",
                "queries": [
                    "What are the economic implications of AI automation?",
                    "How does climate change affect global biodiversity?",
                    "What are the ethical considerations of autonomous vehicles?"
                ],
                "target_time": 2.0
            }
        ]
        
        results = {}
        
        for scenario in test_scenarios:
            times = []
            confidences = []
            
            for query in scenario["queries"]:
                start_time = time.time()
                
                result = await reasoning_tool.execute({
                    "query": query,
                    "reasoning_mode": "logical",
                    "max_steps": 25
                })
                
                execution_time = time.time() - start_time
                times.append(execution_time)
                confidences.append(result.get("confidence", 0))
            
            avg_time = statistics.mean(times)
            results[scenario["name"]] = {
                "average_time": avg_time,
                "target_time": scenario["target_time"],
                "meets_target": avg_time <= scenario["target_time"],
                "average_confidence": statistics.mean(confidences)
            }
        
        return results
    
    async def benchmark_memory(self):
        """Benchmark cognitive memory operations."""
        
        memory_tool = CognitiveMemoryTool()
        
        # Test knowledge storage performance
        storage_times = []
        test_knowledge = [
            "Python is excellent for AI development",
            "Machine learning enables pattern recognition",
            "Neural networks simulate brain functions",
            "Deep learning uses layered architectures",
            "Reinforcement learning learns through rewards"
        ]
        
        stored_ids = []
        
        for knowledge in test_knowledge:
            start_time = time.time()
            
            result = await memory_tool.store_knowledge(
                knowledge=knowledge,
                context="benchmark_test"
            )
            
            storage_time = time.time() - start_time
            storage_times.append(storage_time)
            stored_ids.append(result["knowledge_id"])
        
        # Test knowledge retrieval performance
        retrieval_times = []
        queries = ["Python programming", "machine learning", "neural networks"]
        
        for query in queries:
            start_time = time.time()
            
            await memory_tool.retrieve_knowledge(query)
            
            retrieval_time = time.time() - start_time
            retrieval_times.append(retrieval_time)
        
        # Cleanup
        for knowledge_id in stored_ids:
            await memory_tool.delete_knowledge(knowledge_id)
        
        return {
            "storage": {
                "average_time": statistics.mean(storage_times),
                "target_time": 0.1,
                "meets_target": statistics.mean(storage_times) <= 0.1
            },
            "retrieval": {
                "average_time": statistics.mean(retrieval_times),
                "target_time": 0.1,
                "meets_target": statistics.mean(retrieval_times) <= 0.1
            }
        }
    
    async def generate_cognitive_report(self, results):
        """Generate cognitive performance report."""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Calculate overall cognitive performance score
        performance_scores = []
        
        for category, category_results in results.items():
            if "error" not in category_results:
                category_score = self.calculate_cognitive_score(category_results)
                performance_scores.append(category_score)
        
        overall_score = statistics.mean(performance_scores) if performance_scores else 0
        
        report = {
            "cognitive_benchmark_info": {
                "timestamp": timestamp,
                "overall_cognitive_score": overall_score
            },
            "cognitive_results": results,
            "performance_grade": self.get_performance_grade(overall_score)
        }
        
        # Save cognitive performance report
        report_file = f"cognitive_performance_{timestamp}.json"
        with open(report_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"\n📊 Cognitive Performance Report")
        print(f"Overall Cognitive Score: {overall_score:.2f}/100")
        print(f"Performance Grade: {report['performance_grade']}")
        print(f"Report saved: {report_file}")
    
    def calculate_cognitive_score(self, results):
        """Calculate performance score for cognitive category."""
        # Simplified scoring - can be enhanced based on specific metrics
        return 85.0  # Placeholder implementation
    
    def get_performance_grade(self, score):
        """Get performance grade based on score."""
        if score >= 90:
            return "🏆 Excellent"
        elif score >= 80:
            return "🥈 Good"
        elif score >= 70:
            return "🥉 Satisfactory"
        else:
            return "⚠️ Needs Improvement"

# Usage
if __name__ == "__main__":
    suite = CognitiveBenchmarkSuite()
    asyncio.run(suite.run_cognitive_benchmarks())
```

### Cognitive Performance Monitoring

#### Real-Time Cognitive Metrics

```python
#!/usr/bin/env python3
"""Real-time cognitive performance monitoring"""

import asyncio
import time
from datetime import datetime
from collections import deque
import psutil

class CognitivePerformanceMonitor:
    """Real-time monitoring for cognitive operations."""
    
    def __init__(self, monitoring_interval=30):
        self.monitoring_interval = monitoring_interval
        self.metrics_history = deque(maxlen=100)
        self.running = False
    
    async def start_cognitive_monitoring(self):
        """Start cognitive performance monitoring."""
        
        print("📊 Starting Cognitive Performance Monitoring")
        self.running = True
        
        while self.running:
            try:
                metrics = await self.collect_cognitive_metrics()
                self.metrics_history.append(metrics)
                
                # Check for cognitive performance alerts
                await self.check_cognitive_alerts(metrics)
                
                await asyncio.sleep(self.monitoring_interval)
                
            except Exception as e:
                print(f"❌ Monitoring error: {e}")
                await asyncio.sleep(self.monitoring_interval)
    
    async def collect_cognitive_metrics(self):
        """Collect cognitive-specific metrics."""
        
        # System metrics
        process = psutil.Process()
        cpu_usage = psutil.cpu_percent(interval=1)
        memory_info = process.memory_info()
        
        # Cognitive metrics (would be collected from actual cognitive operations)
        cognitive_metrics = {
            "active_reasoning_operations": 0,
            "memory_operations_per_minute": 0,
            "attention_allocations_per_minute": 0,
            "average_reasoning_time": 0,
            "cognitive_cache_hit_rate": 0,
            "knowledge_base_size": 0
        }
        
        return {
            "timestamp": datetime.now().isoformat(),
            "system": {
                "cpu_usage": cpu_usage,
                "memory_mb": memory_info.rss / 1024 / 1024,
                "memory_percent": psutil.virtual_memory().percent
            },
            "cognitive": cognitive_metrics
        }
    
    async def check_cognitive_alerts(self, metrics):
        """Check for cognitive performance alerts."""
        
        alerts = []
        
        # Cognitive-specific alerts
        if metrics["cognitive"]["average_reasoning_time"] > 3.0:
            alerts.append({
                "type": "SLOW_COGNITIVE_REASONING",
                "value": metrics["cognitive"]["average_reasoning_time"],
                "threshold": 3.0,
                "recommendation": "Optimize reasoning parameters or enable caching"
            })
        
        if metrics["cognitive"]["cognitive_cache_hit_rate"] < 0.5:
            alerts.append({
                "type": "LOW_CACHE_HIT_RATE",
                "value": metrics["cognitive"]["cognitive_cache_hit_rate"],
                "threshold": 0.5,
                "recommendation": "Review caching strategy or increase cache size"
            })
        
        if alerts:
            await self.send_cognitive_alerts(alerts)
    
    async def send_cognitive_alerts(self, alerts):
        """Send cognitive performance alerts."""
        
        print(f"🚨 Cognitive Performance Alerts ({len(alerts)}):")
        for alert in alerts:
            print(f"  • {alert['type']}: {alert['value']:.2f}")
            print(f"    Recommendation: {alert['recommendation']}")
    
    def stop_monitoring(self):
        """Stop cognitive monitoring."""
        self.running = False

# Usage
if __name__ == "__main__":
    monitor = CognitivePerformanceMonitor()
    
    try:
        asyncio.run(monitor.start_cognitive_monitoring())
    except KeyboardInterrupt:
        monitor.stop_monitoring()
```

## Cognitive Performance Optimization

### Reasoning Performance Optimization

#### Configuration Optimization
```python
# High-performance reasoning configuration
optimized_reasoning_config = {
    "pln_enabled": True,
    "max_reasoning_steps": 30,  # Balanced accuracy vs speed
    "reasoning_timeout": 45,    # Reasonable timeout
    "confidence_threshold": 0.8, # Early termination
    "reasoning_cache": True,    # Enable result caching
    "parallel_reasoning": True, # Use multiple threads
    "attention_allocation": True # Focus computational resources
}

reasoning_tool = CognitiveReasoningTool(optimized_reasoning_config)
```

#### Query Optimization Techniques
```python
# Optimize reasoning queries for better performance
async def optimized_reasoning_example():
    reasoning_tool = CognitiveReasoningTool()
    
    # Use specific, focused queries
    focused_query = "What are the top 3 benefits of renewable energy for the environment?"
    
    # Instead of broad, open-ended queries
    # broad_query = "Tell me everything about energy and the environment"
    
    result = await reasoning_tool.execute({
        "query": focused_query,
        "reasoning_mode": "logical",
        "max_steps": 20,  # Limit steps for faster response
        "confidence_threshold": 0.75  # Stop when reasonably confident
    })
    
    return result
```

### Memory Performance Optimization

#### Efficient Memory Configuration
```python
# High-performance memory configuration
optimized_memory_config = {
    "atomspace_backend": "rocks",  # Persistent but fast
    "persistence_enabled": True,
    "indexing_enabled": True,      # Fast search
    "auto_cleanup": True,          # Prevent bloat  
    "compression_enabled": True,   # Efficient storage
    "cache_size": 10000,          # Large cache for frequent access
    "batch_operations": True       # Bulk operations
}

memory_tool = CognitiveMemoryTool(optimized_memory_config)
```

#### Batch Operations for Better Performance
```python
async def batch_memory_operations():
    """Demonstrate batch operations for better performance."""
    
    memory_tool = CognitiveMemoryTool()
    
    # Batch storage is more efficient than individual operations
    knowledge_batch = [
        {"content": "Python is great for AI", "context": "programming"},
        {"content": "Machine learning uses data", "context": "ai_theory"},
        {"content": "Neural networks learn patterns", "context": "deep_learning"}
    ]
    
    # More efficient than storing individually
    start_time = time.time()
    batch_result = await memory_tool.store_knowledge_batch(knowledge_batch)
    batch_time = time.time() - start_time
    
    print(f"Batch storage: {batch_time:.3f}s for {len(knowledge_batch)} items")
```

## Performance Testing Integration with Development

### Continuous Performance Testing

```bash
#!/bin/bash
# continuous_performance_testing.sh

echo "🔄 Continuous Performance Testing for PyCog-Zero"

# Run cpp2py pipeline performance tests
echo "1. Running cpp2py pipeline benchmarks..."
python3 scripts/cpp2py_conversion_pipeline.py test --benchmark-only --benchmark-rounds 3

# Run cognitive performance benchmarks  
echo "2. Running cognitive performance benchmarks..."
python3 -c "
import asyncio
from docs.cognitive_benchmark_suite import CognitiveBenchmarkSuite
suite = CognitiveBenchmarkSuite()
asyncio.run(suite.run_cognitive_benchmarks())
"

# Generate combined performance report
echo "3. Generating combined performance report..."
timestamp=$(date +%Y%m%d_%H%M%S)
echo "Performance test completed at: $timestamp" > performance_summary_$timestamp.txt

echo "✅ Continuous performance testing completed"
```

### Performance Regression Detection

```python
#!/usr/bin/env python3
"""Performance regression detection for PyCog-Zero"""

import json
import statistics
from pathlib import Path
from datetime import datetime

class PerformanceRegressionDetector:
    """Detect performance regressions in cognitive operations."""
    
    def __init__(self, baseline_file="performance_baseline.json"):
        self.baseline_file = baseline_file
        
    def load_baseline(self):
        """Load performance baseline from file."""
        
        if Path(self.baseline_file).exists():
            with open(self.baseline_file, 'r') as f:
                return json.load(f)
        return None
    
    def save_baseline(self, performance_data):
        """Save current performance as new baseline."""
        
        with open(self.baseline_file, 'w') as f:
            json.dump(performance_data, f, indent=2)
    
    def detect_regressions(self, current_results):
        """Detect performance regressions compared to baseline."""
        
        baseline = self.load_baseline()
        if not baseline:
            print("No baseline found - saving current results as baseline")
            self.save_baseline(current_results)
            return []
        
        regressions = []
        
        # Compare reasoning performance
        if "Reasoning Performance" in current_results and "Reasoning Performance" in baseline:
            current_reasoning = current_results["Reasoning Performance"]
            baseline_reasoning = baseline["Reasoning Performance"]
            
            for scenario in current_reasoning:
                if scenario in baseline_reasoning:
                    current_time = current_reasoning[scenario]["average_time"]
                    baseline_time = baseline_reasoning[scenario]["average_time"]
                    
                    # Detect significant regression (>20% slower)
                    if current_time > baseline_time * 1.2:
                        regressions.append({
                            "type": "reasoning_regression",
                            "scenario": scenario,
                            "current_time": current_time,
                            "baseline_time": baseline_time,
                            "regression_percent": ((current_time - baseline_time) / baseline_time) * 100
                        })
        
        return regressions
    
    def generate_regression_report(self, regressions):
        """Generate performance regression report."""
        
        if not regressions:
            print("✅ No performance regressions detected")
            return
        
        print(f"⚠️ Performance Regressions Detected ({len(regressions)}):")
        
        for regression in regressions:
            print(f"  • {regression['type']}: {regression['scenario']}")
            print(f"    Current: {regression['current_time']:.3f}s")
            print(f"    Baseline: {regression['baseline_time']:.3f}s")
            print(f"    Regression: +{regression['regression_percent']:.1f}%")
            print()

# Usage
if __name__ == "__main__":
    detector = PerformanceRegressionDetector()
    
    # Load current results (would come from actual benchmark run)
    current_results = {}  # Placeholder
    
    regressions = detector.detect_regressions(current_results)
    detector.generate_regression_report(regressions)
```

## Performance Targets and SLAs

### Cognitive Performance SLAs

#### Response Time Targets
- **Simple Reasoning**: 95% of queries < 1 second
- **Complex Reasoning**: 95% of queries < 3 seconds
- **Memory Storage**: 99% of operations < 200ms
- **Memory Retrieval**: 99% of operations < 300ms
- **Attention Allocation**: 99% of operations < 100ms

#### Throughput Targets  
- **Reasoning Operations**: 100+ complex queries per minute per instance
- **Memory Operations**: 1000+ storage/retrieval operations per minute
- **Concurrent Users**: 50+ simultaneous cognitive sessions

#### Resource Utilization Targets
- **CPU Usage**: <80% under normal cognitive load
- **Memory Usage**: <4GB for standard cognitive operations
- **Response Time Percentiles**: 
  - 50th percentile < 1.0s
  - 90th percentile < 2.0s  
  - 99th percentile < 5.0s

### Performance Optimization Recommendations

#### For Development
1. **Use Memory Backend**: Faster than persistent storage
2. **Reduce Reasoning Steps**: Balance accuracy vs speed
3. **Enable Caching**: Avoid repeated computations
4. **Disable Verbose Logging**: Reduce I/O overhead

#### For Production
1. **Enable Persistent Storage**: Use AtomSpace-Rocks for data durability
2. **Configure Monitoring**: Real-time performance tracking
3. **Load Balancing**: Distribute cognitive load across instances
4. **Resource Scaling**: Auto-scale based on cognitive demand

This comprehensive performance benchmarking guide ensures optimal cognitive performance across all components of the PyCog-Zero system, from the cpp2py conversion pipeline to advanced cognitive reasoning operations.