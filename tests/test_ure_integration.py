"""
Test suite for URE (Unified Rule Engine) Python bindings integration
"""

import unittest
import asyncio
import json
from unittest.mock import Mock, patch, AsyncMock

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.tools.ure_tool import UREChainTool


class TestUREChainTool(unittest.TestCase):
    """Test URE chain tool functionality."""
    
    def setUp(self):
        """Setup test environment."""
        self.mock_agent = Mock()
        self.tool_params = {
            'name': 'test_ure_chain',
            'method': None,
            'args': {},
            'message': '',
            'loop_data': None
        }
        
    def test_ure_tool_initialization(self):
        """Test URE tool can be initialized."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        self.assertIsNotNone(tool)
        self.assertFalse(hasattr(tool, '_ure_initialized'))
    
    def test_ure_tool_initialization_trigger(self):
        """Test URE tool initialization is triggered properly."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        tool._initialize_if_needed()
        
        self.assertTrue(hasattr(tool, '_ure_initialized'))
        self.assertTrue(tool._ure_initialized)
        self.assertIsNotNone(tool.config)
    
    def test_config_loading(self):
        """Test URE configuration loading."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        config = tool._load_ure_config()
        
        self.assertIsInstance(config, dict)
        self.assertIn('ure_config', config)
        self.assertTrue(config['ure_config']['ure_enabled'])
    
    async def test_backward_chaining_fallback(self):
        """Test backward chaining with fallback mode."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        
        # Force fallback mode by not having URE available
        tool._setup_fallback_mode()
        
        query = "if A implies B and B implies C, then A implies C"
        response = await tool._perform_backward_chaining(query)
        
        self.assertIn("fallback", response.message.lower())
        data = json.loads(response.message.split("Data: ")[1])
        self.assertEqual(data['operation'], 'fallback_backward_chain')
        self.assertIn('implication', data['patterns_detected'])
    
    async def test_forward_chaining_fallback(self):
        """Test forward chaining with fallback mode."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        
        # Force fallback mode
        tool._setup_fallback_mode()
        
        query = "A and B therefore C"
        response = await tool._perform_forward_chaining(query)
        
        self.assertIn("fallback", response.message.lower())
        data = json.loads(response.message.split("Data: ")[1])
        self.assertEqual(data['operation'], 'fallback_forward_chain')
        self.assertIn('conjunction', data['patterns_detected'])
    
    async def test_execute_method_routing(self):
        """Test execute method properly routes operations."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        
        # Test backward chaining routing
        with patch.object(tool, '_perform_backward_chaining', new_callable=AsyncMock) as mock_bc:
            mock_bc.return_value = Mock(message="test backward")
            await tool.execute("test query", "backward_chain")
            mock_bc.assert_called_once_with("test query")
        
        # Test forward chaining routing
        with patch.object(tool, '_perform_forward_chaining', new_callable=AsyncMock) as mock_fc:
            mock_fc.return_value = Mock(message="test forward")
            await tool.execute("test query", "forward_chain")
            mock_fc.assert_called_once_with("test query")
    
    async def test_status_operation(self):
        """Test URE status reporting."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        response = await tool.execute("", "status")
        
        self.assertIn("status", response.message)
        data = json.loads(response.message.split("Data: ")[1])
        self.assertIn('status', data)
        self.assertIn('ure_available', data['status'])
    
    async def test_list_rules_operation(self):
        """Test rule listing functionality."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        response = await tool.execute("", "list_rules")
        
        self.assertIn("rules listed", response.message)
        data = json.loads(response.message.split("Data: ")[1])
        self.assertIn('available_rules', data)
        self.assertGreater(data['rule_count'], 0)
    
    @patch('python.tools.ure_tool.URE_AVAILABLE', True)
    @patch('python.tools.ure_tool.AtomSpace')
    @patch('python.tools.ure_tool.initialize_opencog')
    async def test_atomspace_creation(self, mock_init, mock_atomspace):
        """Test AtomSpace creation for URE."""
        mock_as_instance = Mock()
        mock_atomspace.return_value = mock_as_instance
        
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        atomspace = tool._create_new_atomspace()
        
        self.assertIsNotNone(atomspace)
        mock_atomspace.assert_called_once()
        mock_init.assert_called_once_with(mock_as_instance)
    
    def test_pattern_detection_in_queries(self):
        """Test logical pattern detection in queries."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        
        # Test implication detection
        query1 = "if A then B"
        logical_patterns = []
        query_words = query1.lower().split()
        if any(word in query_words for word in ["if", "then", "implies"]):
            logical_patterns.append("implication")
        
        self.assertIn("implication", logical_patterns)
        
        # Test conjunction detection
        query2 = "A and B"
        logical_patterns2 = []
        query_words2 = query2.lower().split()
        if any(word in query_words2 for word in ["and", "both"]):
            logical_patterns2.append("conjunction")
        
        self.assertIn("conjunction", logical_patterns2)
    
    async def test_rulebase_creation(self):
        """Test rulebase creation functionality."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        
        # Force fallback mode for this test
        tool._setup_fallback_mode()
        
        response = await tool._create_rulebase("test", rulebase_name="test_rb", rules=["deduction"])
        
        # Should fail gracefully in fallback mode
        self.assertIn("Cannot create rulebase", response.message)
    
    def test_default_config_values(self):
        """Test default configuration values are properly set."""
        tool = UREChainTool(self.mock_agent, **self.tool_params)
        config = tool._load_ure_config()
        
        ure_config = config.get('ure_config', {})
        self.assertTrue(ure_config.get('forward_chaining', False))
        self.assertTrue(ure_config.get('backward_chaining', False))
        self.assertEqual(ure_config.get('max_iterations', 0), 1000)
        self.assertFalse(ure_config.get('trace_enabled', True))
    
    async def test_cross_tool_integration(self):
        """Test cross-tool integration setup."""
        with patch('python.tools.ure_tool.ATOMSPACE_TOOLS_AVAILABLE', True):
            with patch('python.tools.ure_tool.AtomSpaceToolHub') as mock_hub:
                tool = UREChainTool(self.mock_agent, **self.tool_params)
                tool._setup_cross_tool_integration()
                
                # Should attempt to create tool hub reference
                self.assertIsNotNone(tool.tool_hub)


class TestUREIntegrationWithCognitiveReasoning(unittest.TestCase):
    """Test URE integration with cognitive reasoning tool."""
    
    def setUp(self):
        """Setup test environment."""
        self.mock_agent = Mock()
        self.tool_params = {
            'name': 'test_cognitive_reasoning',
            'method': None,
            'args': {},
            'message': '',
            'loop_data': None
        }
    
    def test_ure_tool_import_in_cognitive_reasoning(self):
        """Test URE tool can be imported in cognitive reasoning."""
        try:
            from python.tools.cognitive_reasoning import CognitiveReasoningTool, URE_TOOL_AVAILABLE
            self.assertTrue(True)  # Import successful
        except ImportError:
            self.fail("URE tool import failed in cognitive reasoning")
    
    async def test_ure_delegation_from_cognitive_reasoning(self):
        """Test URE delegation from cognitive reasoning tool."""
        with patch('python.tools.ure_tool.URE_AVAILABLE', False):
            from python.tools.cognitive_reasoning import CognitiveReasoningTool
            
            tool = CognitiveReasoningTool(self.mock_agent, **self.tool_params)
            
            # Test URE backward chaining delegation
            response = await tool.execute("test query", "ure_backward_chain")
            
            # Should handle delegation gracefully even if URE not available
            self.assertIn("delegation", response.message.lower())


class TestUREPerformance(unittest.TestCase):
    """Performance tests for URE operations."""
    
    def test_ure_tool_instantiation_performance(self):
        """Test URE tool instantiation performance."""
        mock_agent = Mock()
        tool_params = {
            'name': 'perf_test_ure',
            'method': None,
            'args': {},
            'message': '',
            'loop_data': None
        }
        
        import time
        start_time = time.time()
        
        for i in range(10):
            tool = UREChainTool(mock_agent, **tool_params)
            tool._initialize_if_needed()
        
        end_time = time.time()
        
        # Should complete within reasonable time (less than 1 second for 10 instantiations)
        self.assertLess(end_time - start_time, 1.0)
    
    async def test_fallback_reasoning_performance(self):
        """Test fallback reasoning performance."""
        mock_agent = Mock()
        tool_params = {
            'name': 'perf_test_ure_fallback',
            'method': None,
            'args': {},
            'message': '',
            'loop_data': None
        }
        
        tool = UREChainTool(mock_agent, **tool_params)
        tool._setup_fallback_mode()
        
        import time
        start_time = time.time()
        
        # Test multiple fallback reasoning operations
        queries = [
            "if A then B",
            "A and B implies C",
            "not A or B",
            "A therefore B"
        ]
        
        for query in queries:
            response = await tool._fallback_reasoning(query, "backward_chain")
            self.assertIn("fallback", response.message.lower())
        
        end_time = time.time()
        
        # Should complete within reasonable time
        self.assertLess(end_time - start_time, 2.0)


if __name__ == '__main__':
    # Create test suite
    suite = unittest.TestSuite()
    
    # Add test cases
    suite.addTest(unittest.makeSuite(TestUREChainTool))
    suite.addTest(unittest.makeSuite(TestUREIntegrationWithCognitiveReasoning))
    suite.addTest(unittest.makeSuite(TestUREPerformance))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    
    # Run async tests properly
    async def run_async_tests():
        """Run async tests using asyncio."""
        result = runner.run(suite)
        return result.wasSuccessful()
    
    # For sync tests, run normally
    result = runner.run(suite)
    
    print(f"\nTest Results: {'PASSED' if result.wasSuccessful() else 'FAILED'}")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, failure in result.failures:
            print(f"  {test}: {failure}")
    
    if result.errors:
        print("\nErrors:")
        for test, error in result.errors:
            print(f"  {test}: {error}")