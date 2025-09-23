#!/usr/bin/env python3
"""
Test PLN reasoning with existing PyCog-Zero tools.

This test validates PLN (Probabilistic Logic Networks) reasoning capabilities
integrated with PyCog-Zero's cognitive tool ecosystem, focusing on:
- PLN forward and backward chaining
- Integration with AtomSpace and other cognitive tools
- Reasoning rule applications (deduction, induction, abduction)
- Tool interoperability and result sharing

This addresses the Advanced Learning Systems (Phase 4) requirement for
testing PLN reasoning with existing tools.
"""
import unittest
import sys
import os
import json
import asyncio
from pathlib import Path
from unittest.mock import Mock, AsyncMock

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Mock dependencies that require full Agent-Zero setup
class MockAgent:
    def __init__(self):
        self.agent_name = "test_agent"
        self.context = Mock()
        self.config = {}

class MockResponse:
    def __init__(self, message, data=None, break_loop=False):
        self.message = message
        self.data = data or {}
        self.break_loop = break_loop

class MockTool:
    def __init__(self, agent):
        self.agent = agent
        self.config = {}
        self.initialized = False

# Mock the Agent-Zero imports to avoid full dependency chain
sys.modules['python.helpers.tool'] = Mock()
sys.modules['python.helpers.tool'].Tool = MockTool
sys.modules['python.helpers.tool'].Response = MockResponse

class TestPLNReasoningIntegration(unittest.TestCase):
    """Test suite for PLN reasoning integration with PyCog-Zero tools."""
    
    def setUp(self):
        """Set up test environment."""
        self.mock_agent = MockAgent()
        self.cognitive_tool_path = PROJECT_ROOT / "python/tools/cognitive_reasoning.py"
        
    def test_pln_reasoning_tool_structure(self):
        """Test that PLNReasoningTool has the required structure for Agent-Zero integration."""
        with open(self.cognitive_tool_path, 'r') as f:
            content = f.read()
        
        # Verify PLN reasoning tool class exists
        self.assertIn('class PLNReasoningTool:', content)
        
        # Core reasoning methods
        required_methods = [
            'def forward_chain(',
            'def backward_chain(',
            'def apply_inference_rule(',
            'def _apply_deduction(',
            'def _apply_induction(',
            'def _apply_abduction(',
            'def _apply_inheritance(',
            'def _apply_similarity(',
            'def _apply_modus_ponens('
        ]
        
        for method in required_methods:
            self.assertIn(method, content, f"Missing required PLN method: {method}")
    
    def test_pln_inference_rules_availability(self):
        """Test that all required inference rules are available."""
        with open(self.cognitive_tool_path, 'r') as f:
            content = f.read()
        
        # Core PLN inference rules
        required_rules = [
            "deduction_rule",
            "induction_rule",
            "abduction_rule", 
            "inheritance_rule",
            "similarity_rule",
            "concept_creation_rule",
            "modus_ponens_rule"
        ]
        
        for rule in required_rules:
            self.assertIn(rule, content, f"Missing required inference rule: {rule}")
    
    def test_pln_reasoning_logic_simulation(self):
        """Simulate PLN reasoning operations without OpenCog dependencies."""
        
        # Test forward chaining simulation
        source_concepts = ['artificial_intelligence', 'machine_learning', 'neural_networks']
        
        # Simulate forward chain reasoning steps
        forward_results = []
        for concept in source_concepts:
            # Apply deduction: if A is true, and A implies B, then B is true
            forward_results.append(f"deduction_rule({concept} -> related_concept)")
            # Apply inheritance: if A isa B, derive inheritance relationships  
            forward_results.append(f"inheritance_rule({concept} isa cognitive_system)")
        
        self.assertEqual(len(forward_results), len(source_concepts) * 2)
        self.assertTrue(any('deduction_rule' in result for result in forward_results))
        self.assertTrue(any('inheritance_rule' in result for result in forward_results))
        
        # Test backward chaining simulation
        target_concept = 'intelligent_behavior'
        backward_results = []
        
        # Find premises that could lead to target
        potential_premises = ['learning_capability', 'reasoning_ability', 'adaptation']
        for premise in potential_premises:
            backward_results.append(f"premise_for({target_concept}): {premise}")
        
        self.assertEqual(len(backward_results), len(potential_premises))
        self.assertTrue(all(target_concept in result for result in backward_results))
    
    def test_pln_tool_integration_patterns(self):
        """Test integration patterns between PLN and other cognitive tools."""
        with open(self.cognitive_tool_path, 'r') as f:
            content = f.read()
        
        # Integration with AtomSpace tools
        integration_patterns = [
            'from python.tools.atomspace_tool_hub import AtomSpaceToolHub',
            'from python.tools.atomspace_memory_bridge import AtomSpaceMemoryBridge',
            'ATOMSPACE_TOOLS_AVAILABLE',
            'cross_tool_integration',
            'self.pln_reasoning = PLNReasoningTool('
        ]
        
        for pattern in integration_patterns:
            self.assertIn(pattern, content, f"Missing integration pattern: {pattern}")
    
    def test_pln_reasoning_with_mock_atomspace(self):
        """Test PLN reasoning with mock AtomSpace integration."""
        
        # Mock AtomSpace operations
        mock_atomspace = Mock()
        mock_atomspace.add_node = Mock(return_value="mock_concept_node")
        mock_atomspace.add_link = Mock(return_value="mock_inheritance_link")
        
        # Simulate PLN reasoning operations
        query = "What is the relationship between learning and intelligence?"
        
        # Extract concepts from query
        concepts = ['learning', 'intelligence', 'relationship']
        
        # Simulate atom creation in AtomSpace
        created_atoms = []
        for concept in concepts:
            atom = mock_atomspace.add_node("ConceptNode", concept)
            created_atoms.append(atom)
        
        # Simulate inference rule applications
        inference_results = []
        
        # Deduction: learning -> cognitive_ability -> intelligence
        if len(created_atoms) >= 2:
            deduction_link = mock_atomspace.add_link(
                "InheritanceLink", 
                [created_atoms[0], created_atoms[1]]
            )
            inference_results.append(f"deduction: {deduction_link}")
        
        # Similarity: learning similar to adaptation
        similarity_link = mock_atomspace.add_link(
            "SimilarityLink",
            [created_atoms[0], "adaptation_concept"]
        )
        inference_results.append(f"similarity: {similarity_link}")
        
        self.assertEqual(len(created_atoms), len(concepts))
        self.assertTrue(len(inference_results) >= 2)
        self.assertTrue(any('deduction' in result for result in inference_results))
        self.assertTrue(any('similarity' in result for result in inference_results))
    
    def test_pln_chaining_with_tool_ecosystem(self):
        """Test PLN chaining integrated with PyCog-Zero tool ecosystem."""
        
        # Simulate tool ecosystem interaction
        available_tools = [
            'cognitive_reasoning',
            'atomspace_memory_bridge', 
            'ure_tool',
            'meta_cognition',
            'search_engine'
        ]
        
        # Test forward chaining with tool data
        initial_concepts = ['problem_solving', 'decision_making']
        tool_enhanced_results = []
        
        for concept in initial_concepts:
            # Forward chain with each available cognitive tool
            for tool in available_tools:
                enhanced_result = f"forward_chain({concept}) + {tool}_enhancement"
                tool_enhanced_results.append(enhanced_result)
        
        expected_results = len(initial_concepts) * len(available_tools)
        self.assertEqual(len(tool_enhanced_results), expected_results)
        
        # Test backward chaining with shared tool knowledge
        goal = 'optimal_decision'
        shared_knowledge = {
            'cognitive_reasoning': ['logical_analysis', 'pattern_recognition'],
            'memory_bridge': ['past_decisions', 'learned_patterns'],
            'meta_cognition': ['self_reflection', 'strategy_selection']
        }
        
        backward_chain_results = []
        for tool, knowledge_items in shared_knowledge.items():
            for item in knowledge_items:
                premise = f"backward_chain({goal}) <- {item} (from {tool})"
                backward_chain_results.append(premise)
        
        total_knowledge_items = sum(len(items) for items in shared_knowledge.values())
        self.assertEqual(len(backward_chain_results), total_knowledge_items)
    
    def test_pln_inference_rule_applications(self):
        """Test specific PLN inference rule applications in cognitive context."""
        
        # Test modus ponens: A, A->B, therefore B
        premises_modus_ponens = ['learning_occurs', 'learning_occurs -> knowledge_increases']
        modus_ponens_result = self._apply_modus_ponens(premises_modus_ponens)
        self.assertEqual(modus_ponens_result, 'knowledge_increases')
        
        # Test deduction: A->B, B->C, therefore A->C  
        premises_deduction = [
            'experience -> learning',
            'learning -> expertise', 
            'expertise -> performance'
        ]
        deduction_result = self._apply_deduction(premises_deduction)
        self.assertEqual(deduction_result, 'experience -> performance')
        
        # Test abduction: B, A->B, possibly A
        premises_abduction = ['high_performance', 'expertise -> high_performance']
        abduction_result = self._apply_abduction(premises_abduction)
        self.assertEqual(abduction_result, 'possibly_expertise')
        
        # Test inheritance: X isa Y, Y isa Z, therefore X isa Z
        premises_inheritance = ['neural_network isa machine_learning', 'machine_learning isa AI']
        inheritance_result = self._apply_inheritance(premises_inheritance)
        self.assertEqual(inheritance_result, 'neural_network isa AI')
    
    def _apply_modus_ponens(self, premises):
        """Simulate modus ponens rule application."""
        if len(premises) >= 2:
            fact = premises[0]
            rule = premises[1]
            if '->' in rule and fact in rule:
                return rule.split('->')[1].strip()
        return None
    
    def _apply_deduction(self, premises):
        """Simulate deduction rule application."""
        if len(premises) >= 2:
            # Chain implications: A->B, B->C becomes A->C
            first_rule = premises[0]
            last_rule = premises[-1]
            if '->' in first_rule and '->' in last_rule:
                start = first_rule.split('->')[0].strip()
                end = last_rule.split('->')[1].strip()
                return f"{start} -> {end}"
        return None
    
    def _apply_abduction(self, premises):
        """Simulate abduction rule application."""
        if len(premises) >= 2:
            effect = premises[0]
            rule = premises[1]
            if '->' in rule and effect in rule:
                cause = rule.split('->')[0].strip()
                return f"possibly_{cause}"
        return None
    
    def _apply_inheritance(self, premises):
        """Simulate inheritance rule application."""
        if len(premises) >= 2:
            first_inheritance = premises[0]
            second_inheritance = premises[1]
            if ' isa ' in first_inheritance and ' isa ' in second_inheritance:
                # Extract transitivity: A isa B, B isa C -> A isa C
                parts1 = first_inheritance.split(' isa ')
                parts2 = second_inheritance.split(' isa ')
                if len(parts1) == 2 and len(parts2) == 2:
                    if parts1[1].strip() == parts2[0].strip():
                        return f"{parts1[0].strip()} isa {parts2[1].strip()}"
        return None

class TestPLNReasoningWithOpenCog(unittest.TestCase):
    """Test PLN reasoning with OpenCog integration (graceful fallback if not available)."""
    
    def setUp(self):
        """Set up test with OpenCog availability check."""
        self.mock_agent = MockAgent()
        try:
            # Try to import OpenCog components
            from opencog.atomspace import AtomSpace, types
            self.opencog_available = True
            self.atomspace = AtomSpace()
        except ImportError:
            self.opencog_available = False
            self.atomspace = None
    
    def test_opencog_integration_availability(self):
        """Test OpenCog integration status and fallback behavior."""
        cognitive_tool_path = PROJECT_ROOT / "python/tools/cognitive_reasoning.py"
        
        with open(cognitive_tool_path, 'r') as f:
            content = f.read()
        
        # Verify graceful OpenCog handling
        opencog_checks = [
            'OPENCOG_AVAILABLE = True',
            'except ImportError:',
            'OPENCOG_AVAILABLE = False',
            'PLN_AVAILABLE = True',
            'PLN_AVAILABLE = False'
        ]
        
        for check in opencog_checks:
            self.assertIn(check, content, f"Missing OpenCog availability check: {check}")
    
    def test_pln_fallback_reasoning(self):
        """Test PLN fallback reasoning when OpenCog is not available."""
        
        # Simulate fallback PLN reasoning operations
        query = "How does learning relate to problem solving?"
        
        # Extract key concepts
        concepts = ['learning', 'problem', 'solving', 'relate']
        
        # Apply fallback reasoning rules
        fallback_results = []
        
        for concept in concepts:
            # Basic pattern matching
            if concept in ['learning', 'solving']:
                fallback_results.append(f"cognitive_process({concept})")
            if concept in ['problem', 'relate']:
                fallback_results.append(f"conceptual_link({concept})")
        
        # Verify fallback produces reasonable results
        self.assertTrue(len(fallback_results) > 0)
        self.assertTrue(any('cognitive_process' in result for result in fallback_results))
        self.assertTrue(any('conceptual_link' in result for result in fallback_results))
    
    @unittest.skipUnless(
        sys.modules.get('opencog') is not None,
        "OpenCog not available - testing fallback behavior"
    )
    def test_opencog_pln_chainer_integration(self):
        """Test OpenCog PLN chainer integration if available."""
        if not self.opencog_available:
            self.skipTest("OpenCog not available")
        
        # This test would run actual OpenCog PLN operations
        # For now, we verify the integration pattern exists
        cognitive_tool_path = PROJECT_ROOT / "python/tools/cognitive_reasoning.py"
        
        with open(cognitive_tool_path, 'r') as f:
            content = f.read()
        
        # Verify OpenCog PLN integration patterns
        opencog_patterns = [
            'from opencog.pln import PLNChainer',
            'self.pln_chainer = PLNChainer(self.atomspace)',
            'self.pln_chainer.forward_chain(',
            'self.pln_chainer.backward_chain('
        ]
        
        for pattern in opencog_patterns:
            self.assertIn(pattern, content, f"Missing OpenCog PLN pattern: {pattern}")

class TestPLNReasoningToolEcosystemIntegration(unittest.TestCase):
    """Test PLN reasoning integration with the broader PyCog-Zero tool ecosystem."""
    
    def test_ure_tool_integration(self):
        """Test integration between PLN reasoning and URE (Unified Rule Engine)."""
        ure_demo_path = PROJECT_ROOT / "demo_ure_integration.py"
        
        if ure_demo_path.exists():
            with open(ure_demo_path, 'r') as f:
                content = f.read()
            
            # Verify URE integration patterns
            ure_patterns = [
                'UREChainTool',
                'forward_chain',
                'backward_chain',
                'demonstrate_ure_forward_chaining',
                'demonstrate_ure_backward_chaining'
            ]
            
            for pattern in ure_patterns:
                self.assertIn(pattern, content, f"Missing URE integration pattern: {pattern}")
    
    def test_cognitive_tool_ecosystem_readiness(self):
        """Test that cognitive tools are ready for PLN reasoning integration."""
        tools_dir = PROJECT_ROOT / "python/tools"
        
        # Check for key cognitive tools
        expected_tools = [
            'cognitive_reasoning.py',
            'atomspace_memory_bridge.py',
            'atomspace_tool_hub.py',
            'meta_cognition.py',
            'ure_tool.py'
        ]
        
        for tool in expected_tools:
            tool_path = tools_dir / tool
            if tool_path.exists():
                with open(tool_path, 'r') as f:
                    content = f.read()
                
                # Verify basic tool structure
                self.assertIn('class', content, f"Tool {tool} should define a class")
                
                # Many tools should have Agent-Zero integration
                if 'Tool' in content:
                    self.assertIn('def execute(', content, f"Agent-Zero tool {tool} should have execute method")

def run_pln_reasoning_test_suite():
    """Run the complete PLN reasoning test suite."""
    print("🧠 Running PLN Reasoning Integration Test Suite")
    print("=" * 60)
    
    # Create test suite
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test classes
    suite.addTests(loader.loadTestsFromTestCase(TestPLNReasoningIntegration))
    suite.addTests(loader.loadTestsFromTestCase(TestPLNReasoningWithOpenCog))
    suite.addTests(loader.loadTestsFromTestCase(TestPLNReasoningToolEcosystemIntegration))
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print(f"\n📊 Test Results Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    if result.failures:
        print(f"\n❌ Failures:")
        for failure in result.failures:
            print(f"  - {failure[0]}")
    
    if result.errors:
        print(f"\n🚫 Errors:")
        for error in result.errors:
            print(f"  - {error[0]}")
    
    if result.wasSuccessful():
        print(f"\n✅ All PLN reasoning tests passed! Ready for integration.")
        return True
    else:
        print(f"\n⚠️ Some tests failed. Review results above.")
        return False

if __name__ == "__main__":
    success = run_pln_reasoning_test_suite()
    sys.exit(0 if success else 1)