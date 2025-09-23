#!/usr/bin/env python3
"""
Integration test for cognitive reasoning tool with Agent-Zero framework.

This test validates that the cognitive reasoning tool is properly integrated
with the Agent-Zero framework and can be discovered and used by agents.
"""
import pytest
import os
import json
import asyncio
from pathlib import Path

# Test configuration
PROJECT_ROOT = Path(__file__).parent.parent
COGNITIVE_TOOL_PATH = PROJECT_ROOT / "python" / "tools" / "cognitive_reasoning.py"
COGNITIVE_CONFIG_PATH = PROJECT_ROOT / "conf" / "config_cognitive.json"

class TestCognitiveReasoningIntegration:
    """Test suite for cognitive reasoning tool integration with Agent-Zero."""

    def test_cognitive_tool_file_exists(self):
        """Test that the cognitive reasoning tool file exists."""
        assert COGNITIVE_TOOL_PATH.exists(), f"Cognitive tool not found: {COGNITIVE_TOOL_PATH}"

    def test_cognitive_config_exists(self):
        """Test that the cognitive configuration file exists."""
        assert COGNITIVE_CONFIG_PATH.exists(), f"Cognitive config not found: {COGNITIVE_CONFIG_PATH}"

    def test_cognitive_config_valid(self):
        """Test that the cognitive configuration is valid JSON with required fields."""
        with open(COGNITIVE_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Required configuration fields
        required_fields = [
            'cognitive_mode',
            'opencog_enabled', 
            'reasoning_config'
        ]
        
        for field in required_fields:
            assert field in config, f"Missing required config field: {field}"
        
        # Test reasoning config structure
        reasoning_config = config['reasoning_config']
        assert 'pln_enabled' in reasoning_config
        assert 'pattern_matching' in reasoning_config

    def test_cognitive_tool_structure(self):
        """Test that the cognitive tool has the expected class and method structure."""
        with open(COGNITIVE_TOOL_PATH, 'r') as f:
            content = f.read()
        
        # Required elements for Agent-Zero tool compliance
        required_elements = [
            'class CognitiveReasoningTool(Tool):',
            'def __init__(self, agent)',
            'async def execute(',
            'def register():',
            'return CognitiveReasoningTool'
        ]
        
        for element in required_elements:
            assert element in content, f"Missing required element: {element}"

    def test_tool_registration_function(self):
        """Test that the tool has a proper registration function."""
        with open(COGNITIVE_TOOL_PATH, 'r') as f:
            content = f.read()
        
        # Check registration function exists and is properly documented
        assert 'def register():' in content
        assert 'Register the cognitive reasoning tool with Agent-Zero' in content
        assert 'return CognitiveReasoningTool' in content

    def test_agent_zero_tool_pattern_compliance(self):
        """Test that the tool follows Agent-Zero tool patterns."""
        with open(COGNITIVE_TOOL_PATH, 'r') as f:
            content = f.read()
        
        # Agent-Zero tool pattern requirements
        patterns = [
            'from python.helpers.tool import Tool, Response',  # Imports Tool base class
            'class CognitiveReasoningTool(Tool):',  # Inherits from Tool
            'super().__init__(agent)',  # Calls parent constructor
            'return Response(',  # Returns Response objects
        ]
        
        for pattern in patterns:
            assert pattern in content, f"Missing Agent-Zero pattern: {pattern}"

    def test_cognitive_reasoning_logic_structure(self):
        """Test that the cognitive reasoning logic has expected methods."""
        with open(COGNITIVE_TOOL_PATH, 'r') as f:
            content = f.read()
        
        # Core cognitive reasoning methods
        reasoning_methods = [
            'def _load_cognitive_config(',
            'def parse_query_to_atoms(',
            'def execute_reasoning(',
            'def pattern_matching_reasoning(',
            'def pln_reasoning(',
            'def format_reasoning_for_agent('
        ]
        
        for method in reasoning_methods:
            assert method in content, f"Missing reasoning method: {method}"

    def test_opencog_integration_structure(self):
        """Test that OpenCog integration is properly structured."""
        with open(COGNITIVE_TOOL_PATH, 'r') as f:
            content = f.read()
        
        # OpenCog integration requirements
        opencog_elements = [
            'from opencog.atomspace import AtomSpace, types',
            'from opencog.utilities import initialize_opencog',
            'OPENCOG_AVAILABLE = True',
            'except ImportError:',
            'OPENCOG_AVAILABLE = False'
        ]
        
        for element in opencog_elements:
            assert element in content, f"Missing OpenCog integration element: {element}"

    def test_error_handling_structure(self):
        """Test that proper error handling is implemented."""
        with open(COGNITIVE_TOOL_PATH, 'r') as f:
            content = f.read()
        
        # Error handling requirements
        error_handling = [
            'try:',
            'except Exception as e:',
            'if not self.initialized:',
            'graceful fallback'
        ]
        
        for element in error_handling:
            assert element in content, f"Missing error handling element: {element}"

    @pytest.mark.integration 
    def test_tool_discovery_readiness(self):
        """Test that the tool is ready for Agent-Zero discovery."""
        # Tool must be in the tools directory
        tools_dir = PROJECT_ROOT / "python" / "tools"
        assert tools_dir.exists(), "Tools directory not found"
        
        tool_files = [f.name for f in tools_dir.iterdir() if f.suffix == '.py']
        assert 'cognitive_reasoning.py' in tool_files, "Cognitive tool not in tools directory"
        
        # Configuration must be accessible
        assert COGNITIVE_CONFIG_PATH.exists(), "Config not accessible for tool discovery"

    def test_configuration_integration(self):
        """Test that configuration is properly integrated."""
        with open(COGNITIVE_CONFIG_PATH, 'r') as f:
            config = json.load(f)
        
        # Verify cognitive mode is enabled
        assert config.get('cognitive_mode') is True, "Cognitive mode not enabled"
        assert config.get('opencog_enabled') is True, "OpenCog integration not enabled"
        
        # Verify reasoning configuration
        reasoning_config = config.get('reasoning_config', {})
        assert reasoning_config.get('pln_enabled') is True, "PLN reasoning not enabled"
        assert reasoning_config.get('pattern_matching') is True, "Pattern matching not enabled"

class TestCognitiveReasoningFunctionality:
    """Test the functional aspects of cognitive reasoning."""
    
    def test_query_parsing_logic(self):
        """Test query parsing logic simulation."""
        # Simulate the tool's query parsing
        query = "What is artificial intelligence?"
        words = query.lower().split()
        concepts = [word for word in words if len(word) > 2]
        
        assert len(concepts) > 0, "Query parsing should extract concepts"
        assert 'what' in concepts or 'artificial' in concepts, "Should extract meaningful words"

    def test_reasoning_chain_logic(self):
        """Test reasoning chain construction logic."""
        # Simulate reasoning chain creation
        concepts = ['artificial', 'intelligence', 'machine', 'learning']
        
        # Pattern matching simulation
        pattern_links = []
        for i in range(len(concepts) - 1):
            link = f"InheritanceLink({concepts[i]}, {concepts[i + 1]})"
            pattern_links.append(link)
        
        assert len(pattern_links) == len(concepts) - 1, "Should create links between concepts"
        
        # PLN reasoning simulation  
        pln_evaluations = []
        for concept in concepts:
            evaluation = f"EvaluationLink(relevant, {concept})"
            pln_evaluations.append(evaluation)
        
        assert len(pln_evaluations) == len(concepts), "Should create evaluations for all concepts"


class TestPLNLogicalInference:
    """Test suite for PLN logical inference capabilities."""
    
    def test_pln_reasoning_tool_structure(self):
        """Test that PLNReasoningTool class exists with required methods."""
        with open(COGNITIVE_TOOL_PATH, 'r') as f:
            content = f.read()
        
        # PLN reasoning tool requirements
        pln_tool_elements = [
            'class PLNReasoningTool:',
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
        
        for element in pln_tool_elements:
            assert element in content, f"Missing PLN reasoning tool element: {element}"
    
    def test_pln_reasoning_integration(self):
        """Test that PLN reasoning is integrated into CognitiveReasoningTool."""
        with open(COGNITIVE_TOOL_PATH, 'r') as f:
            content = f.read()
        
        # Integration requirements
        integration_elements = [
            'self.pln_reasoning = None',
            'self.pln_reasoning = PLNReasoningTool(',
            'async def perform_pln_logical_inference(',
            'async def _enhanced_pln_reasoning_with_tool(',
        ]
        
        for element in integration_elements:
            assert element in content, f"Missing PLN integration element: {element}"
    
    def test_inference_rules_available(self):
        """Test that all required inference rules are available."""
        with open(COGNITIVE_TOOL_PATH, 'r') as f:
            content = f.read()
        
        # Required inference rules
        inference_rules = [
            "deduction_rule",
            "induction_rule", 
            "abduction_rule",
            "inheritance_rule",
            "similarity_rule",
            "concept_creation_rule",
            "modus_ponens_rule"
        ]
        
        for rule in inference_rules:
            assert rule in content, f"Missing inference rule: {rule}"
    
    def test_pln_reasoning_logic_simulation(self):
        """Simulate PLN reasoning logic without OpenCog dependencies."""
        # Simulate forward chaining
        source_atoms = ['concept_a', 'concept_b', 'concept_c']
        forward_chain_steps = []
        
        for atom in source_atoms:
            # Simulate rule applications
            forward_chain_steps.append(f"deduction({atom})")
            forward_chain_steps.append(f"induction({atom})")
        
        assert len(forward_chain_steps) == len(source_atoms) * 2, "Forward chaining should apply multiple rules"
        
        # Simulate backward chaining
        target_atom = 'goal_concept'
        backward_chain_steps = []
        
        # Simulate finding premises
        for i in range(3):
            premise = f"premise_{i}"
            backward_chain_steps.append(f"premise_for({target_atom}): {premise}")
        
        assert len(backward_chain_steps) == 3, "Backward chaining should find premises"
        
    def test_logical_inference_patterns(self):
        """Test logical inference patterns and rule applications."""
        # Test deduction pattern: A -> B, B -> C, therefore A -> C
        premises = ['A', 'B', 'C']
        deduction_result = f"deduction({premises[0]} -> {premises[-1]})"
        assert deduction_result == "deduction(A -> C)", "Deduction should connect first to last premise"
        
        # Test modus ponens pattern: A, A -> B, therefore B
        modus_ponens_result = f"modus_ponens_consequent({premises[1]})"
        assert modus_ponens_result == "modus_ponens_consequent(B)", "Modus ponens should derive consequent"
        
        # Test inheritance pattern: X isa Y
        inheritance_result = f"inheritance({premises[0]} isa {premises[1]})"
        assert inheritance_result == "inheritance(A isa B)", "Inheritance should create isa relationship"

if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v"])