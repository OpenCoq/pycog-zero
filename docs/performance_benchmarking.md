# Performance Benchmarking with cpp2py_conversion_pipeline.py

This document describes the performance benchmarking capabilities added to the PyCog-Zero cpp2py conversion pipeline.

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