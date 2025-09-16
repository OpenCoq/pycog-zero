"""
Performance benchmarks for the cpp2py conversion pipeline.
Tests pipeline operations and OpenCog component performance.
"""
import pytest
import time
import json
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, List

# Add project root to Python path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cpp2py_conversion_pipeline import CPP2PyConversionPipeline, Phase


class TestPipelinePerformance:
    """Performance tests for the cpp2py conversion pipeline."""
    
    def test_pipeline_initialization_performance(self, benchmark):
        """Benchmark pipeline initialization time."""
        
        def init_pipeline():
            with tempfile.TemporaryDirectory() as tmp_dir:
                return CPP2PyConversionPipeline(tmp_dir)
        
        result = benchmark(init_pipeline)
        assert result is not None
    
    def test_component_definitions_loading_performance(self, benchmark):
        """Benchmark component definitions loading."""
        
        def load_components():
            pipeline = CPP2PyConversionPipeline()
            return len(pipeline.components)
        
        result = benchmark(load_components)
        assert result > 0
    
    def test_phase_report_generation_performance(self, benchmark):
        """Benchmark phase report generation."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def generate_report():
            return pipeline.generate_phase_report(Phase.PHASE_1_CORE_EXTENSIONS)
        
        result = benchmark(generate_report)
        assert 'phase' in result
        assert 'components' in result
    
    def test_dependency_validation_performance(self, benchmark):
        """Benchmark dependency validation for all components."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_all_dependencies():
            results = {}
            for component_name in pipeline.components:
                results[component_name] = pipeline.validate_dependencies(component_name)
            return results
        
        result = benchmark(validate_all_dependencies)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestComponentPerformance:
    """Performance tests for individual components."""
    
    @pytest.mark.parametrize("component_name", [
        "cogutil", "atomspace", "ure", "pln", "opencog"
    ])
    def test_component_python_validation_performance(self, benchmark, component_name):
        """Benchmark Python bindings validation for components."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_component():
            # This will test validation logic even if component isn't cloned
            try:
                return pipeline.validate_python_bindings(component_name)
            except Exception:
                return False
        
        result = benchmark(validate_component)
        # Result can be True or False, we're measuring performance
        assert isinstance(result, bool)


class TestMemoryUsageTests:
    """Memory usage tests for pipeline operations."""
    
    def test_pipeline_memory_usage(self, performance_metrics):
        """Test memory usage during pipeline operations."""
        performance_metrics.start_measurement()
        
        # Perform multiple pipeline operations
        pipeline = CPP2PyConversionPipeline()
        
        # Generate reports for all phases
        for phase in Phase:
            report = pipeline.generate_phase_report(phase)
            assert report is not None
        
        # Validate multiple components
        test_components = ["cogutil", "atomspace", "ure"]
        for component in test_components:
            pipeline.validate_dependencies(component)
        
        performance_metrics.end_measurement()
        
        results = performance_metrics.get_results()
        
        # Memory usage should be reasonable (less than 100MB for these operations)
        assert results['memory_usage_mb'] < 100, f"Memory usage too high: {results['memory_usage_mb']}MB"
        
        # Duration should be reasonable (less than 10 seconds)
        assert results['duration_seconds'] < 10, f"Operations took too long: {results['duration_seconds']}s"


class TestCLIPerformance:
    """Performance tests for CLI commands."""
    
    def test_status_command_performance(self, benchmark):
        """Benchmark status command performance."""
        
        def run_status_command():
            script_path = project_root / "scripts" / "cpp2py_conversion_pipeline.py"
            result = subprocess.run([
                "python", str(script_path), "status"
            ], capture_output=True, text=True, cwd=project_root)
            return result.returncode == 0
        
        result = benchmark(run_status_command)
        assert result is True
    
    def test_help_command_performance(self, benchmark):
        """Benchmark help command performance."""
        
        def run_help_command():
            script_path = project_root / "scripts" / "cpp2py_conversion_pipeline.py"
            result = subprocess.run([
                "python", str(script_path), "--help"
            ], capture_output=True, text=True, cwd=project_root)
            return result.returncode == 0
        
        result = benchmark(run_help_command)
        assert result is True
    
    @pytest.mark.parametrize("component", ["cogutil", "atomspace"])
    def test_validation_command_performance(self, benchmark, component):
        """Benchmark validation command performance for components."""
        
        def run_validation_command():
            script_path = project_root / "scripts" / "cpp2py_conversion_pipeline.py"
            result = subprocess.run([
                "python", str(script_path), "validate", component, "--deps-only"
            ], capture_output=True, text=True, cwd=project_root)
            # Command may return 0 or 1 depending on component availability
            return result.returncode in [0, 1]
        
        result = benchmark(run_validation_command)
        assert result is True


class TestScalabilityTests:
    """Tests for pipeline scalability with multiple components."""
    
    def test_all_components_processing_performance(self, benchmark, performance_metrics):
        """Test processing all components for scalability."""
        
        def process_all_components():
            pipeline = CPP2PyConversionPipeline()
            results = {}
            
            # Process each component
            for component_name, component in pipeline.components.items():
                results[component_name] = {
                    'dependencies_valid': pipeline.validate_dependencies(component_name),
                    'phase': component.phase.value,
                    'priority': component.priority
                }
            
            return results
        
        performance_metrics.start_measurement()
        result = benchmark(process_all_components)
        performance_metrics.end_measurement()
        
        metrics = performance_metrics.get_results()
        
        # Should handle all components efficiently
        assert len(result) >= 8, "Should process all components"
        assert metrics['duration_seconds'] < 5, f"Processing took too long: {metrics['duration_seconds']}s"
    
    def test_concurrent_phase_report_generation(self, benchmark):
        """Test generating reports for all phases concurrently."""
        
        def generate_all_phase_reports():
            pipeline = CPP2PyConversionPipeline()
            reports = {}
            
            for phase in Phase:
                reports[phase.value] = pipeline.generate_phase_report(phase)
            
            return reports
        
        result = benchmark(generate_all_phase_reports)
        
        # Should generate reports for all phases
        assert len(result) == len(Phase), "Should generate reports for all phases"
        
        # Each report should have required structure
        for phase_name, report in result.items():
            assert 'phase' in report
            assert 'total_components' in report
            assert 'components' in report


@pytest.mark.slow
class TestExtendedPerformanceTests:
    """Extended performance tests that take longer to run."""
    
    def test_repeated_pipeline_operations(self, benchmark):
        """Test performance of repeated pipeline operations."""
        
        def repeated_operations():
            results = []
            for i in range(10):
                pipeline = CPP2PyConversionPipeline()
                report = pipeline.generate_phase_report(Phase.PHASE_0_FOUNDATION)
                results.append(report)
            return len(results)
        
        result = benchmark.pedantic(repeated_operations, iterations=3, rounds=2)
        assert result == 10
    
    def test_large_scale_validation(self, performance_metrics):
        """Test validation of all components multiple times."""
        performance_metrics.start_measurement()
        
        pipeline = CPP2PyConversionPipeline()
        
        # Run validation multiple times to test consistency
        for iteration in range(5):
            for component_name in pipeline.components:
                pipeline.validate_dependencies(component_name)
        
        performance_metrics.end_measurement()
        
        results = performance_metrics.get_results()
        
        # Should complete within reasonable time
        assert results['duration_seconds'] < 30, f"Large scale validation took too long: {results['duration_seconds']}s"