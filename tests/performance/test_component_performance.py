"""
Component-specific performance benchmarks for OpenCog components.
Tests the performance characteristics of individual components.
"""
import pytest
import time
import subprocess
import tempfile
import json
from pathlib import Path
from typing import Dict, Any

# Add project root to Python path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cpp2py_conversion_pipeline import CPP2PyConversionPipeline, Phase


class TestCogutilPerformance:
    """Performance tests specific to cogutil component."""
    
    def test_cogutil_validation_performance(self, benchmark):
        """Benchmark cogutil validation performance."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_cogutil():
            return pipeline.validate_dependencies("cogutil")
        
        result = benchmark(validate_cogutil)
        # cogutil has no dependencies, so should always be valid
        assert result is True
    
    def test_cogutil_python_bindings_validation_performance(self, benchmark):
        """Benchmark cogutil Python bindings validation."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_cogutil_bindings():
            # This tests the validation logic performance
            try:
                return pipeline.validate_python_bindings("cogutil")
            except Exception:
                return False
        
        result = benchmark(validate_cogutil_bindings)
        # Result depends on whether cogutil is cloned and configured
        assert isinstance(result, bool)
    
    def test_cogutil_status_reporting_performance(self, benchmark):
        """Benchmark cogutil status reporting performance."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def get_cogutil_status():
            report = pipeline.generate_phase_report(Phase.PHASE_0_FOUNDATION)
            return 'cogutil' in report['components']
        
        result = benchmark(get_cogutil_status)
        assert result is True


class TestAtomspacePerformance:
    """Performance tests specific to atomspace component."""
    
    def test_atomspace_validation_performance(self, benchmark):
        """Benchmark atomspace validation performance."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_atomspace():
            return pipeline.validate_dependencies("atomspace")
        
        result = benchmark(validate_atomspace)
        # atomspace depends on cogutil, so result depends on whether cogutil is available
        assert isinstance(result, bool)
    
    def test_atomspace_python_bindings_validation_performance(self, benchmark):
        """Benchmark atomspace Python bindings validation."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_atomspace_bindings():
            try:
                return pipeline.validate_python_bindings("atomspace")
            except Exception:
                return False
        
        result = benchmark(validate_atomspace_bindings)
        assert isinstance(result, bool)
    
    def test_atomspace_dependency_chain_performance(self, benchmark):
        """Benchmark atomspace dependency chain validation."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_dependency_chain():
            atomspace = pipeline.components["atomspace"]
            results = []
            
            # Validate each dependency
            for dep in atomspace.dependencies:
                results.append(pipeline.validate_dependencies(dep))
            
            return all(results) if results else True
        
        result = benchmark(validate_dependency_chain)
        assert isinstance(result, bool)


class TestUREPerformance:
    """Performance tests specific to URE (Unified Rule Engine) component."""
    
    def test_ure_validation_performance(self, benchmark):
        """Benchmark URE validation performance."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_ure():
            return pipeline.validate_dependencies("ure")
        
        result = benchmark(validate_ure)
        assert isinstance(result, bool)
    
    def test_ure_complex_dependency_validation_performance(self, benchmark):
        """Benchmark URE complex dependency validation."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_ure_complex_deps():
            ure = pipeline.components["ure"]
            # URE depends on both atomspace and unify
            dependency_results = {}
            
            for dep in ure.dependencies:
                dependency_results[dep] = pipeline.validate_dependencies(dep)
            
            return dependency_results
        
        result = benchmark(validate_ure_complex_deps)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestPLNPerformance:
    """Performance tests specific to PLN (Probabilistic Logic Networks) component."""
    
    def test_pln_validation_performance(self, benchmark):
        """Benchmark PLN validation performance."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_pln():
            return pipeline.validate_dependencies("pln")
        
        result = benchmark(validate_pln)
        assert isinstance(result, bool)
    
    def test_pln_advanced_dependency_performance(self, benchmark):
        """Benchmark PLN advanced dependency validation."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_pln_advanced_deps():
            pln = pipeline.components["pln"]
            results = {}
            
            # PLN is in advanced learning phase
            for dep in pln.dependencies:
                start_time = time.perf_counter()
                is_valid = pipeline.validate_dependencies(dep)
                end_time = time.perf_counter()
                
                results[dep] = {
                    'valid': is_valid,
                    'validation_time': end_time - start_time
                }
            
            return results
        
        result = benchmark(validate_pln_advanced_deps)
        assert isinstance(result, dict)


class TestOpenCogPerformance:
    """Performance tests specific to main OpenCog integration component."""
    
    def test_opencog_validation_performance(self, benchmark):
        """Benchmark OpenCog main component validation performance."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_opencog():
            return pipeline.validate_dependencies("opencog")
        
        result = benchmark(validate_opencog)
        assert isinstance(result, bool)
    
    def test_opencog_complete_dependency_chain_performance(self, benchmark):
        """Benchmark complete OpenCog dependency chain validation."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_complete_chain():
            opencog = pipeline.components["opencog"]
            dependency_chain = {}
            
            # OpenCog has multiple dependencies - this tests the complete chain
            for dep in opencog.dependencies:
                dependency_chain[dep] = {
                    'direct_validation': pipeline.validate_dependencies(dep),
                    'component_exists': dep in pipeline.components
                }
                
                # Validate sub-dependencies
                if dep in pipeline.components:
                    sub_component = pipeline.components[dep]
                    dependency_chain[dep]['sub_dependencies'] = []
                    
                    for sub_dep in sub_component.dependencies:
                        dependency_chain[dep]['sub_dependencies'].append({
                            'name': sub_dep,
                            'valid': pipeline.validate_dependencies(sub_dep)
                        })
            
            return dependency_chain
        
        result = benchmark(validate_complete_chain)
        assert isinstance(result, dict)
        assert len(result) > 0


class TestCrossComponentPerformance:
    """Performance tests across multiple components."""
    
    def test_phase_based_validation_performance(self, benchmark):
        """Benchmark validation performance for each phase."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def validate_by_phases():
            phase_results = {}
            
            for phase in Phase:
                phase_components = [
                    comp for comp in pipeline.components.values()
                    if comp.phase == phase
                ]
                
                phase_results[phase.value] = {
                    'component_count': len(phase_components),
                    'validation_results': []
                }
                
                for component in phase_components:
                    start_time = time.perf_counter()
                    is_valid = pipeline.validate_dependencies(component.name)
                    end_time = time.perf_counter()
                    
                    phase_results[phase.value]['validation_results'].append({
                        'component': component.name,
                        'valid': is_valid,
                        'time': end_time - start_time
                    })
            
            return phase_results
        
        result = benchmark(validate_by_phases)
        assert isinstance(result, dict)
        assert len(result) == len(Phase)
    
    def test_dependency_graph_traversal_performance(self, benchmark):
        """Benchmark dependency graph traversal performance."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def traverse_dependency_graph():
            traversal_results = {}
            
            for component_name, component in pipeline.components.items():
                traversal_results[component_name] = {
                    'dependencies': component.dependencies,
                    'dependency_validation': [],
                    'traversal_depth': 0
                }
                
                # Traverse dependency chain
                current_deps = component.dependencies[:]
                depth = 0
                
                while current_deps and depth < 10:  # Prevent infinite loops
                    depth += 1
                    next_level_deps = []
                    
                    for dep in current_deps:
                        if dep in pipeline.components:
                            dep_component = pipeline.components[dep]
                            next_level_deps.extend(dep_component.dependencies)
                            
                            traversal_results[component_name]['dependency_validation'].append({
                                'level': depth,
                                'dependency': dep,
                                'valid': pipeline.validate_dependencies(dep)
                            })
                    
                    current_deps = list(set(next_level_deps))  # Remove duplicates
                
                traversal_results[component_name]['traversal_depth'] = depth
            
            return traversal_results
        
        result = benchmark(traverse_dependency_graph)
        assert isinstance(result, dict)
        assert len(result) > 0
    
    def test_bulk_component_operations_performance(self, benchmark):
        """Benchmark bulk operations on all components."""
        
        pipeline = CPP2PyConversionPipeline()
        
        def bulk_operations():
            results = {
                'total_components': len(pipeline.components),
                'operations': {}
            }
            
            # Perform multiple operations on all components
            operations = ['validate_dependencies', 'get_phase', 'get_priority']
            
            for operation in operations:
                start_time = time.perf_counter()
                
                if operation == 'validate_dependencies':
                    op_results = {
                        name: pipeline.validate_dependencies(name)
                        for name in pipeline.components
                    }
                elif operation == 'get_phase':
                    op_results = {
                        name: comp.phase.value
                        for name, comp in pipeline.components.items()
                    }
                elif operation == 'get_priority':
                    op_results = {
                        name: comp.priority
                        for name, comp in pipeline.components.items()
                    }
                
                end_time = time.perf_counter()
                
                results['operations'][operation] = {
                    'results': op_results,
                    'execution_time': end_time - start_time,
                    'components_processed': len(op_results)
                }
            
            return results
        
        result = benchmark(bulk_operations)
        assert isinstance(result, dict)
        assert 'total_components' in result
        assert 'operations' in result
        assert len(result['operations']) == 3