"""
Performance tests for PLN optimization in real-time agent operations.
Tests the performance improvements from Issue #45 implementation.
"""
import pytest
import time
import asyncio
import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


class TestPLNOptimizationPerformance:
    """Performance tests for optimized PLN integration."""
    
    def test_pln_lazy_initialization_performance(self, benchmark):
        """Test performance of lazy PLN initialization."""
        
        def lazy_initialization():
            # Import only when needed for testing
            from python.tools.cognitive_reasoning import PLNReasoningTool
            
            # Multiple initializations should be fast after first
            start_time = time.perf_counter()
            tool = PLNReasoningTool(None)
            init_time = time.perf_counter() - start_time
            
            # Second initialization should be very fast (already initialized)
            start_time = time.perf_counter()
            tool._initialize_pln_lazy()
            second_init_time = time.perf_counter() - start_time
            
            return {
                "first_init": init_time,
                "lazy_init": second_init_time,
                "initialized": tool._initialized
            }
        
        result = benchmark(lazy_initialization)
        assert isinstance(result, dict)
        assert result["lazy_init"] < result["first_init"]  # Lazy should be faster
    
    def test_pln_caching_performance(self, benchmark):
        """Test performance improvement from result caching."""
        
        def test_caching():
            from python.tools.cognitive_reasoning import PLNReasoningTool
            
            tool = PLNReasoningTool(None)
            test_atoms = ["test_atom_1", "test_atom_2"]
            
            # First call - cache miss
            start_time = time.perf_counter()
            result1 = tool.forward_chain(test_atoms, max_steps=3)
            first_call_time = time.perf_counter() - start_time
            
            # Second call - cache hit
            start_time = time.perf_counter()
            result2 = tool.forward_chain(test_atoms, max_steps=3)
            cached_call_time = time.perf_counter() - start_time
            
            metrics = tool.get_performance_metrics()
            
            return {
                "first_call_time": first_call_time,
                "cached_call_time": cached_call_time,
                "cache_hit_rate": metrics["cache_hit_rate"],
                "total_requests": metrics["total_requests"]
            }
        
        result = benchmark(test_caching)
        assert result["cached_call_time"] < result["first_call_time"]
        assert result["cache_hit_rate"] > 0  # Should have cache hits
        assert result["total_requests"] >= 2
    
    @pytest.mark.asyncio
    async def test_concurrent_pln_processing_performance(self):
        """Test concurrent processing performance improvement."""
        
        try:
            from python.tools.cognitive_reasoning import CognitiveReasoningTool
            
            tool = CognitiveReasoningTool()
            tool._initialize_if_needed()
            
            if not tool.initialized:
                pytest.skip("OpenCog not available for concurrent processing test")
            
            # Test sequential vs concurrent processing
            test_atoms = ["test_concept_1", "test_concept_2", "test_concept_3"]
            context = {
                "reasoning_config": {
                    "forward_chaining": True,
                    "backward_chaining": True,
                    "reasoning_timeout": 2.0
                }
            }
            
            # Measure concurrent processing time
            start_time = time.perf_counter()
            results = await tool._enhanced_pln_reasoning_with_tool(test_atoms, context)
            concurrent_time = time.perf_counter() - start_time
            
            # Get performance metrics
            performance_report = tool.get_performance_report()
            
            assert concurrent_time < 5.0  # Should complete within 5 seconds
            assert "pln_metrics" in performance_report
            assert performance_report["tool_performance"]["total_operations"] > 0
            
        except ImportError:
            pytest.skip("Required cognitive reasoning tools not available")
    
    def test_pln_memory_optimization_performance(self, benchmark):
        """Test memory usage optimization in PLN operations."""
        
        def memory_usage_test():
            from python.tools.cognitive_reasoning import PLNReasoningTool
            import gc
            
            tool = PLNReasoningTool(None)
            
            # Test with larger atom sets
            large_atom_set = [f"concept_{i}" for i in range(100)]
            
            # Perform multiple operations to test memory management
            results = []
            for i in range(10):
                subset = large_atom_set[i*10:(i+1)*10]
                result = tool.forward_chain(subset, max_steps=2)
                results.append(len(result) if result else 0)
            
            # Force garbage collection
            gc.collect()
            
            metrics = tool.get_performance_metrics()
            
            return {
                "operations_completed": len(results),
                "cache_size": metrics["cache_size"],
                "avg_result_length": sum(results) / len(results) if results else 0,
                "cache_hit_rate": metrics["cache_hit_rate"]
            }
        
        result = benchmark(memory_usage_test)
        assert result["operations_completed"] == 10
        assert result["cache_size"] <= 100  # Should not exceed max cache size
        
    @pytest.mark.asyncio
    async def test_real_time_response_performance(self):
        """Test that PLN operations meet real-time response requirements."""
        
        try:
            from python.tools.cognitive_reasoning import CognitiveReasoningTool
            
            tool = CognitiveReasoningTool()
            tool._initialize_if_needed()
            
            if not tool.initialized:
                pytest.skip("OpenCog not available for real-time test")
            
            # Test multiple quick queries
            queries = [
                "What is artificial intelligence?",
                "How does machine learning work?",
                "What are cognitive architectures?",
                "Explain reasoning systems",
                "Define knowledge representation"
            ]
            
            response_times = []
            
            for query in queries:
                start_time = time.perf_counter()
                response = await tool.execute(query, operation="reason")
                end_time = time.perf_counter()
                
                response_time = end_time - start_time
                response_times.append(response_time)
                
                # Each response should be under 3 seconds for real-time usage
                assert response_time < 3.0, f"Query '{query}' took {response_time:.2f}s (too slow for real-time)"
            
            # Average response time should be under 2 seconds
            avg_response_time = sum(response_times) / len(response_times)
            assert avg_response_time < 2.0, f"Average response time {avg_response_time:.2f}s too slow"
            
            # Get final performance report
            performance_report = tool.get_performance_report()
            
            assert performance_report["tool_performance"]["total_operations"] >= len(queries)
            assert performance_report["initialization_status"]["initialized"] is True
            
        except ImportError:
            pytest.skip("Required cognitive reasoning tools not available")
    
    def test_pln_performance_monitoring(self, benchmark):
        """Test performance monitoring and metrics collection."""
        
        def performance_monitoring():
            from python.tools.cognitive_reasoning import PLNReasoningTool
            
            tool = PLNReasoningTool(None)
            
            # Perform several operations to generate metrics
            test_operations = [
                (["concept_a", "concept_b"], 3),
                (["idea_1", "idea_2", "idea_3"], 2),
                (["entity_x"], 5),
                (["test_concept"], 1),
            ]
            
            for atoms, steps in test_operations:
                tool.forward_chain(atoms, max_steps=steps)
                tool.backward_chain(atoms, max_steps=steps)
            
            metrics = tool.get_performance_metrics()
            
            return metrics
        
        result = benchmark(performance_monitoring)
        
        # Verify metrics structure
        expected_keys = ["cache_hit_rate", "cache_size", "total_requests", "avg_reasoning_time_ms", "initialized"]
        for key in expected_keys:
            assert key in result, f"Missing metric: {key}"
        
        assert result["total_requests"] > 0
        assert result["cache_hit_rate"] >= 0.0
        assert result["avg_reasoning_time_ms"] >= 0.0


class TestPLNOptimizationBenchmarks:
    """Benchmark tests comparing optimized vs unoptimized PLN operations."""
    
    def test_comparison_initialization_time(self, benchmark):
        """Compare initialization time between optimized and unoptimized versions."""
        
        def optimized_initialization():
            from python.tools.cognitive_reasoning import PLNReasoningTool
            
            # Optimized version with lazy loading
            start = time.perf_counter()
            tool = PLNReasoningTool(None)
            init_time = time.perf_counter() - start
            
            # Actual initialization only happens on first use
            start = time.perf_counter()  
            tool._initialize_pln_lazy()
            lazy_init_time = time.perf_counter() - start
            
            return {
                "constructor_time": init_time,
                "lazy_init_time": lazy_init_time,
                "total_time": init_time + lazy_init_time
            }
        
        result = benchmark(optimized_initialization)
        
        # Constructor should be very fast (no expensive operations)
        assert result["constructor_time"] < 0.1, "Constructor too slow"
        
        # Total time should be reasonable
        assert result["total_time"] < 1.0, "Total initialization too slow"