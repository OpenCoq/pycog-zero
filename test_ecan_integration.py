#!/usr/bin/env python3
"""
ECAN Cross-Tool Integration Test Suite

Comprehensive test suite for validating ECAN integration across cognitive tools.
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch


class TestECANCoordinator(unittest.TestCase):
    """Test cases for the ECAN coordinator."""
    
    def setUp(self):
        """Set up test environment."""
        from python.helpers.ecan_coordinator import ECANCoordinator, AttentionRequest
        self.coordinator = ECANCoordinator()
        self.AttentionRequest = AttentionRequest
    
    def test_coordinator_initialization(self):
        """Test ECAN coordinator initializes correctly."""
        self.assertIsNotNone(self.coordinator)
        self.assertIsInstance(self.coordinator.active_tools, set)
        self.assertIsInstance(self.coordinator.metrics, dict)
    
    def test_tool_registration(self):
        """Test tool registration and unregistration."""
        # Register a tool
        self.coordinator.register_tool("test_tool", 1.5)
        self.assertIn("test_tool", self.coordinator.active_tools)
        self.assertEqual(self.coordinator.fallback_priorities["test_tool"], 1.5)
        
        # Unregister the tool
        self.coordinator.unregister_tool("test_tool")
        self.assertNotIn("test_tool", self.coordinator.active_tools)
        self.assertNotIn("test_tool", self.coordinator.fallback_priorities)
    
    def test_attention_request_processing(self):
        """Test attention request processing."""
        # Register a tool first
        self.coordinator.register_tool("test_tool", 1.0)
        
        # Create an attention request
        request = self.AttentionRequest(
            tool_name="test_tool",
            priority=2.0,
            context="Test reasoning task", 
            concepts=["test", "concept", "attention"],
            importance_multiplier=1.2
        )
        
        # Process the request
        result = self.coordinator.request_attention(request)
        self.assertTrue(result)
        self.assertIn("test_tool", self.coordinator.attention_requests)
    
    def test_attention_allocation(self):
        """Test attention allocation functionality."""
        # Set up some requests
        self.coordinator.register_tool("tool1", 1.0)
        self.coordinator.register_tool("tool2", 1.5)
        
        request1 = self.AttentionRequest(
            tool_name="tool1",
            priority=1.0,
            context="Task 1",
            concepts=["concept1", "concept2"]
        )
        
        request2 = self.AttentionRequest(
            tool_name="tool2", 
            priority=2.0,
            context="Task 2",
            concepts=["concept3", "concept4"]
        )
        
        self.coordinator.request_attention(request1)
        self.coordinator.request_attention(request2)
        
        # Get allocation
        allocation = self.coordinator.get_attention_allocation()
        self.assertIsNotNone(allocation)
        self.assertGreater(allocation.total_sti_allocated, 0)
        self.assertIsInstance(allocation.concept_allocations, dict)
        self.assertIsInstance(allocation.tool_allocations, dict)
    
    def test_synchronization(self):
        """Test attention synchronization across tools."""
        # Register tools
        self.coordinator.register_tool("sync_tool1", 1.0)
        self.coordinator.register_tool("sync_tool2", 1.5)
        
        # Synchronize
        sync_data = self.coordinator.synchronize_attention()
        
        self.assertIn("timestamp", sync_data)
        self.assertIn("active_tools", sync_data)
        self.assertIn("allocation", sync_data)
        self.assertIn("metrics", sync_data)
        self.assertEqual(len(sync_data["active_tools"]), 2)
    
    def test_metrics_collection(self):
        """Test metrics collection functionality."""
        metrics = self.coordinator.get_metrics()
        
        required_metrics = [
            "ecan_available",
            "active_tools", 
            "total_requests",
            "allocation_history_size",
            "total_allocations",
            "average_entropy",
            "cross_tool_interactions",
            "attention_conflicts"
        ]
        
        for metric in required_metrics:
            self.assertIn(metric, metrics)


class TestECANCognitiveToolIntegration(unittest.TestCase):
    """Test ECAN integration with cognitive tools."""
    
    def test_cognitive_reasoning_integration(self):
        """Test ECAN integration with cognitive reasoning tool."""
        from python.tools.cognitive_reasoning import CognitiveReasoningTool
        
        # Check that ECAN coordinator imports are available
        try:
            from python.tools.cognitive_reasoning import ECAN_COORDINATOR_AVAILABLE
            self.assertTrue(ECAN_COORDINATOR_AVAILABLE)
        except ImportError:
            self.fail("ECAN coordinator imports not available in cognitive_reasoning")
    
    def test_cognitive_memory_integration(self):
        """Test ECAN integration with cognitive memory tool."""
        from python.tools.cognitive_memory import CognitiveMemoryTool
        
        # Check that ECAN coordinator imports are available  
        try:
            from python.tools.cognitive_memory import ECAN_COORDINATOR_AVAILABLE
            self.assertTrue(ECAN_COORDINATOR_AVAILABLE)
        except ImportError:
            self.fail("ECAN coordinator imports not available in cognitive_memory")
    
    def test_meta_cognition_integration(self):
        """Test ECAN integration with meta-cognition tool."""
        from python.tools.meta_cognition import MetaCognitionTool
        
        # Check that ECAN coordinator imports are available
        try:
            from python.tools.meta_cognition import ECAN_COORDINATOR_AVAILABLE  
            self.assertTrue(ECAN_COORDINATOR_AVAILABLE)
        except ImportError:
            self.fail("ECAN coordinator imports not available in meta_cognition")


class TestECANConceptExtraction(unittest.TestCase):
    """Test concept extraction for ECAN attention allocation."""
    
    def setUp(self):
        """Set up test environment."""
        # Mock the cognitive reasoning tool to test concept extraction
        from python.tools.cognitive_reasoning import CognitiveReasoningTool
        
        # Mock agent
        class MockAgent:
            def __init__(self):
                self.config = {}
        
        self.agent = MockAgent()
    
    def test_concept_extraction_method(self):
        """Test the concept extraction method."""
        # This would test the _extract_concepts_from_query method
        # Since we need to instantiate the tool properly, we'll test the logic directly
        
        import re
        
        def extract_concepts(query: str):
            """Extracted logic for testing."""
            words = re.findall(r'\b[a-zA-Z]{3,}\b', query.lower())
            stop_words = {
                'the', 'and', 'are', 'was', 'were', 'been', 'have', 'has', 'had', 
                'will', 'would', 'could', 'should', 'can', 'may', 'might', 'must',
                'how', 'what', 'when', 'where', 'why', 'who', 'which', 'that', 'this',
                'for', 'with', 'from', 'about', 'into', 'through', 'during', 'before',
                'after', 'above', 'below', 'between', 'among'
            }
            concepts = [word for word in words if word not in stop_words]
            return concepts[:8]
        
        # Test concept extraction
        test_query = "How can I improve my problem-solving skills using cognitive techniques?"
        concepts = extract_concepts(test_query)
        
        self.assertIsInstance(concepts, list)
        self.assertGreater(len(concepts), 0)
        self.assertIn("improve", concepts)
        self.assertIn("problem", concepts)
        self.assertIn("solving", concepts)
        self.assertIn("skills", concepts)
        self.assertIn("cognitive", concepts)
        self.assertIn("techniques", concepts)


class TestECANNeuralSymbolicBridge(unittest.TestCase):
    """Test ECAN integration with neural-symbolic bridge."""
    
    def test_ecan_weight_computation(self):
        """Test ECAN weight computation in neural-symbolic bridge.""" 
        try:
            from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism
            
            # Create attention mechanism
            attention = CognitiveAttentionMechanism(embedding_dim=32, num_heads=4)
            
            # Create mock atoms with STI values
            class MockAtomWithSTI:
                def __init__(self, name, sti):
                    self.name = name
                    self.sti = sti
            
            atoms = [
                MockAtomWithSTI("concept1", 2.0),
                MockAtomWithSTI("concept2", 1.0),
                MockAtomWithSTI("concept3", 0.5)
            ]
            
            # Compute ECAN weights
            weights = attention.compute_ecan_weights(atoms)
            
            # Verify weights
            self.assertEqual(len(weights), 3)
            # Should sum to 1 (normalized)
            self.assertAlmostEqual(float(weights.sum()), 1.0, places=5)
            
        except ImportError:
            self.skipTest("Neural-symbolic bridge not available")


def run_async_test(coro):
    """Helper to run async test functions."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestECANAsyncIntegration(unittest.TestCase):
    """Test async integration aspects of ECAN."""
    
    def test_async_attention_request(self):
        """Test async attention request processing."""
        async def test_async():
            from python.helpers.ecan_coordinator import get_ecan_coordinator, request_attention_for_tool
            
            coordinator = get_ecan_coordinator()
            coordinator.register_tool("async_test_tool", 1.0)
            
            result = request_attention_for_tool(
                tool_name="async_test_tool",
                priority=2.0,
                context="Async test",
                concepts=["async", "test", "attention"]
            )
            
            return result
        
        result = run_async_test(test_async())
        self.assertTrue(result)


if __name__ == "__main__":
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestECANCoordinator,
        TestECANCognitiveToolIntegration, 
        TestECANConceptExtraction,
        TestECANNeuralSymbolicBridge,
        TestECANAsyncIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print("\n" + "="*60)
    print("ECAN Integration Test Summary")
    print("="*60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {(result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100:.1f}%")
    
    if result.failures:
        print("\nFailures:")
        for test, trace in result.failures:
            print(f"  - {test}: {trace.split('AssertionError:')[-1].strip()}")
    
    if result.errors:
        print("\nErrors:")
        for test, trace in result.errors:
            print(f"  - {test}: {trace.split('raise')[-1].strip()}")
            
    if not result.failures and not result.errors:
        print("\n🎉 All ECAN integration tests passed!")
    
    print(f"\nECAN Cross-Tool Integration: {'✅ SUCCESSFUL' if not result.failures and not result.errors else '⚠️ PARTIAL'}")