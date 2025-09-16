#!/usr/bin/env python3
"""
Comprehensive tests for enhanced cognitive reasoning tool with new atomspace bindings.
Tests both OpenCog-enabled and fallback modes.
"""

import unittest
import asyncio
import json
import sys
import os

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from python.tools.cognitive_reasoning import CognitiveReasoningTool


class MockAgent:
    """Mock Agent-Zero instance for testing."""
    def __init__(self):
        self.agent_name = "test_agent"
        self.capabilities = ["cognitive_reasoning", "memory", "metacognition"]
        self.tools = []
    
    def get_capabilities(self):
        return self.capabilities
    
    def get_tools(self):
        return self.tools


class TestEnhancedCognitiveReasoning(unittest.TestCase):
    """Test suite for enhanced cognitive reasoning tool."""
    
    def setUp(self):
        """Setup test environment."""
        self.mock_agent = MockAgent()
        self.tool = CognitiveReasoningTool(
            agent=self.mock_agent,
            name='cognitive_reasoning',
            method=None,
            args={},
            message='test_message',
            loop_data=None
        )
    
    def test_tool_initialization(self):
        """Test tool initialization and configuration loading."""
        # Test basic initialization
        self.assertIsNotNone(self.tool)
        
        # Initialize the tool
        self.tool._initialize_if_needed()
        
        # Check configuration is loaded
        self.assertIsNotNone(self.tool.config)
        self.assertIsInstance(self.tool.config, dict)
        
        # Verify fallback mode is active (since OpenCog not installed)
        self.assertFalse(self.tool.initialized)
        self.assertTrue(hasattr(self.tool, '_cognitive_initialized'))
    
    def test_configuration_loading(self):
        """Test configuration loading from multiple sources."""
        config = self.tool._load_cognitive_config()
        
        # Verify default configuration structure
        self.assertIn("cognitive_mode", config)
        self.assertIn("opencog_enabled", config)
        self.assertIn("reasoning_config", config)
        self.assertIn("atomspace_config", config)
        
        # Check reasoning configuration
        reasoning_config = config.get("reasoning_config", {})
        self.assertIn("pln_enabled", reasoning_config)
        self.assertIn("pattern_matching", reasoning_config)
    
    async def test_basic_reasoning_operation(self):
        """Test basic reasoning operation in fallback mode."""
        query = "What is the relationship between learning and memory?"
        
        response = await self.tool.execute(query)
        
        # Verify response structure
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'message'))
        self.assertTrue(hasattr(response, 'break_loop'))
        
        # Check fallback reasoning was performed
        self.assertIn("Fallback cognitive reasoning", response.message)
        self.assertIn("Data:", response.message)
        
        # Parse and verify data structure
        data_part = response.message.split("Data: ")[1]
        data = json.loads(data_part)
        
        self.assertEqual(data["query"], query)
        self.assertEqual(data["operation"], "fallback_reason")
        self.assertIn("patterns_identified", data)
        self.assertIn("reasoning_steps", data)
        self.assertEqual(data["status"], "fallback_success")
    
    async def test_pattern_analysis_operation(self):
        """Test pattern analysis operation."""
        query = "How do neural networks learn patterns?"
        
        response = await self.tool.execute(query, operation="analyze_patterns")
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertIn("pattern analysis", response.message.lower())
        
        # Parse data
        data_part = response.message.split("Data: ")[1]
        data = json.loads(data_part)
        
        self.assertEqual(data["operation"], "fallback_analyze_patterns")
        self.assertIn("patterns", data)
        
        # Check pattern detection
        patterns = data["patterns"]
        self.assertIn("word_count", patterns)
        self.assertIn("question_words", patterns)
        self.assertGreater(patterns["word_count"], 0)
    
    async def test_status_operation(self):
        """Test status retrieval operation."""
        response = await self.tool.execute("", operation="status")
        
        # Verify response
        self.assertIsNotNone(response)
        self.assertIn("status retrieved", response.message.lower())
        
        # Parse status data
        data_part = response.message.split("Data: ")[1]
        data = json.loads(data_part)
        
        status = data["status"]
        self.assertIn("opencog_available", status)
        self.assertIn("cognitive_reasoning_initialized", status)
        self.assertIn("fallback_mode", status)
        self.assertIn("configuration", status)
        
        # Verify expected status values
        self.assertFalse(status["opencog_available"])  # OpenCog not installed
        self.assertTrue(status["fallback_mode"])
    
    async def test_cross_reference_operation(self):
        """Test cross-reference operation."""
        query = "machine learning concepts"
        
        response = await self.tool.execute(query, operation="cross_reference")
        
        # Verify response structure
        self.assertIsNotNone(response)
        
        # Should handle missing atomspace tools gracefully
        if "tools not loaded" in response.message:
            # Expected when atomspace tools are not available
            data_part = response.message.split("Data: ")[1]
            data = json.loads(data_part)
            self.assertEqual(data["status"], "tools_unavailable")
        else:
            # If tools are available, should have cross-reference data
            self.assertIn("cross-reference", response.message.lower())
    
    async def test_share_knowledge_operation(self):
        """Test knowledge sharing operation."""
        query = "cognitive reasoning patterns"
        
        response = await self.tool.execute(query, operation="share_knowledge")
        
        # Verify response
        self.assertIsNotNone(response)
        
        # Should handle missing tool hub gracefully
        if "hub not available" in response.message:
            data_part = response.message.split("Data: ")[1]
            data = json.loads(data_part)
            self.assertEqual(data["status"], "hub_unavailable")
        else:
            self.assertIn("knowledge shared", response.message.lower())
    
    def test_query_to_atoms_parsing(self):
        """Test query parsing to atoms."""
        # Initialize tool first
        self.tool._initialize_if_needed()
        
        query = "machine learning neural networks"
        context = {
            "related_concepts": ["algorithm", "training"],
            "memory_associations": ["supervised_learning"]
        }
        
        # In fallback mode, this should return empty list
        atoms = self.tool.parse_query_to_atoms(query, context)
        self.assertIsInstance(atoms, list)
        
        # Should be empty in fallback mode (no OpenCog)
        self.assertEqual(len(atoms), 0)
    
    def test_enhanced_pattern_matching(self):
        """Test enhanced pattern matching reasoning."""
        # Initialize tool
        self.tool._initialize_if_needed()
        
        atoms = []  # Empty in fallback mode
        context = {
            "related_concepts": ["pattern", "recognition"],
            "memory_associations": ["neural_net"]
        }
        
        # Should handle empty atoms gracefully
        results = self.tool.enhanced_pattern_matching_reasoning(atoms, context)
        self.assertIsInstance(results, list)
    
    def test_enhanced_pln_reasoning(self):
        """Test enhanced PLN reasoning."""
        # Initialize tool
        self.tool._initialize_if_needed()
        
        atoms = []  # Empty in fallback mode
        context = {
            "reasoning_hints": ["probabilistic"],
            "tool_data": {"memory": "available"}
        }
        
        # Should handle empty atoms gracefully
        results = self.tool.enhanced_pln_reasoning(atoms, context)
        self.assertIsInstance(results, list)
    
    def test_backward_chaining_reasoning(self):
        """Test backward chaining reasoning."""
        # Initialize tool
        self.tool._initialize_if_needed()
        
        atoms = []  # Empty in fallback mode
        context = {"goal": "understanding"}
        
        # Should handle empty atoms gracefully
        results = self.tool.backward_chaining_reasoning(atoms, context)
        self.assertIsInstance(results, list)
    
    async def test_fallback_reasoning_patterns(self):
        """Test fallback reasoning pattern detection."""
        test_cases = [
            ("What is machine learning?", ["question_pattern"]),
            ("How does learning work?", ["question_pattern"]),
            ("I want to learn programming", ["learning_pattern"]),
            ("This works because of algorithms", ["causal_pattern"]),
            ("Therefore, neural networks are useful", ["causal_pattern"]),
        ]
        
        for query, expected_patterns in test_cases:
            response = await self.tool.execute(query)
            
            # Parse response data
            data_part = response.message.split("Data: ")[1]
            data = json.loads(data_part)
            
            patterns = data["patterns_identified"]
            
            # Check if expected patterns are detected
            for expected_pattern in expected_patterns:
                self.assertIn(expected_pattern, patterns, 
                            f"Pattern '{expected_pattern}' not found for query: '{query}'")
    
    def test_format_reasoning_for_agent(self):
        """Test formatting reasoning results for Agent-Zero."""
        # Test with empty results
        formatted = self.tool.format_reasoning_for_agent([])
        self.assertIsInstance(formatted, list)
        self.assertTrue(len(formatted) > 0)
        self.assertIn("No reasoning results", formatted[0])
        
        # Test with mock results
        class MockResult:
            def __str__(self):
                return "MockLink(concept1, concept2)"
            
            def __init__(self, type_name):
                self.type_name = type_name
        
        mock_results = [MockResult("InheritanceLink"), MockResult("EvaluationLink")]
        formatted = self.tool.format_reasoning_for_agent(mock_results)
        
        self.assertIsInstance(formatted, list)
        self.assertTrue(len(formatted) > 0)
        # Should contain summary as first element
        self.assertIn("summary", formatted[0].lower())
    
    async def test_disabled_cognitive_mode(self):
        """Test behavior when cognitive mode is disabled."""
        # Temporarily disable cognitive mode
        original_config = self.tool.config
        self.tool.config = {"cognitive_mode": False}
        
        response = await self.tool.execute("test query")
        
        # Should return disabled message
        self.assertIn("disabled", response.message.lower())
        
        # Parse data
        data_part = response.message.split("Data: ")[1]
        data = json.loads(data_part)
        self.assertEqual(data["error"], "Cognitive mode disabled in configuration")
        
        # Restore configuration
        self.tool.config = original_config
    
    async def test_unknown_operation(self):
        """Test handling of unknown operations."""
        response = await self.tool.execute("test query", operation="unknown_operation")
        
        # Should default to reasoning operation
        self.assertIsNotNone(response)
        # Should perform fallback reasoning
        self.assertIn("Fallback cognitive reasoning", response.message)


class TestAsyncOperations(unittest.IsolatedAsyncioTestCase):
    """Test class for async operations."""
    
    async def test_async_reasoning_operations(self):
        """Test all async reasoning operations."""
        mock_agent = MockAgent()
        tool = CognitiveReasoningTool(
            agent=mock_agent,
            name='cognitive_reasoning',
            method=None,
            args={},
            message='test_message',
            loop_data=None
        )
        
        operations = [
            ("reason", "What is AI?"),
            ("analyze_patterns", "How do patterns work?"),
            ("status", ""),
            ("cross_reference", "machine learning"),
            ("share_knowledge", "neural networks")
        ]
        
        for operation, query in operations:
            with self.subTest(operation=operation):
                response = await tool.execute(query, operation=operation)
                self.assertIsNotNone(response)
                self.assertTrue(hasattr(response, 'message'))
                self.assertTrue(hasattr(response, 'break_loop'))


def run_tests():
    """Run all cognitive reasoning tests."""
    print("Running Enhanced Cognitive Reasoning Tool Tests...")
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestEnhancedCognitiveReasoning))
    suite.addTests(loader.loadTestsFromTestCase(TestAsyncOperations))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    if result.wasSuccessful():
        print(f"\n✓ All {result.testsRun} tests passed!")
        return True
    else:
        print(f"\n✗ {len(result.failures)} failures, {len(result.errors)} errors out of {result.testsRun} tests")
        return False


if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)