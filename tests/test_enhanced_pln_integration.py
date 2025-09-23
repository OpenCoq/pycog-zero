#!/usr/bin/env python3
"""
Test suite for enhanced PLN (Probabilistic Logic Networks) integration.
Tests the new probabilistic reasoning capabilities and TorchPLN integration.
"""

import os
import sys
import pytest
import json
from pathlib import Path

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, str(Path(__file__).parent.parent))

from python.tools.cognitive_reasoning import PLNReasoningTool, CognitiveReasoningTool

class TestEnhancedPLNIntegration:
    """Test suite for enhanced PLN probabilistic reasoning."""
    
    def test_pln_reasoning_tool_enhanced_structure(self):
        """Test that PLNReasoningTool has enhanced probabilistic methods."""
        pln_tool = PLNReasoningTool()
        
        # Test enhanced reasoning rules
        enhanced_rules = [
            "fuzzy_conjunction_rule",
            "fuzzy_disjunction_rule", 
            "consequent_disjunction_elimination_rule",
            "contraposition_rule"
        ]
        
        for rule in enhanced_rules:
            assert rule in pln_tool.reasoning_rules, f"Missing enhanced rule: {rule}"
    
    def test_probabilistic_atom_creation(self):
        """Test creation of atoms with probabilistic truth values."""
        pln_tool = PLNReasoningTool()
        
        # Test without atomspace (should handle gracefully)
        atom = pln_tool.create_probabilistic_atom("test_concept", 0.8, 0.9)
        assert atom is None  # Should be None without atomspace
    
    def test_probabilistic_reasoning_fallbacks(self):
        """Test that probabilistic reasoning methods have proper fallbacks."""
        pln_tool = PLNReasoningTool()
        
        # Test modus ponens fallback
        result = pln_tool._fallback_modus_ponens(None, None)
        assert result is None  # Should handle None inputs gracefully
        
        # Test fuzzy conjunction fallback
        result = pln_tool._fallback_fuzzy_conjunction([])
        assert result is None  # Should handle empty input gracefully
    
    def test_enhanced_inference_rules(self):
        """Test that enhanced inference rules are properly integrated."""
        pln_tool = PLNReasoningTool()
        
        # Test fuzzy disjunction
        result = pln_tool._apply_fuzzy_disjunction([])
        assert result == []  # Should handle empty premises
        
        # Test consequent disjunction elimination
        result = pln_tool._apply_consequent_disjunction_elimination([])
        assert result == []  # Should handle insufficient premises
        
        # Test contraposition
        result = pln_tool._apply_contraposition([])
        assert result == []  # Should handle empty premises
    
    def test_enhanced_forward_chaining(self):
        """Test enhanced forward chaining with probabilistic reasoning."""
        pln_tool = PLNReasoningTool()
        
        # Test with empty source atoms
        result = pln_tool.forward_chain([], max_steps=3)
        assert isinstance(result, list)
        
        # Test fallback forward chain
        result = pln_tool._enhanced_fallback_forward_chain([], max_steps=2)
        assert isinstance(result, list)
        assert len(result) == 0  # No atoms to process
    
    def test_enhanced_backward_chaining(self):
        """Test enhanced backward chaining with probabilistic reasoning."""
        pln_tool = PLNReasoningTool()
        
        # Test with empty target atoms
        result = pln_tool.backward_chain([], max_steps=3)
        assert isinstance(result, list)
        
        # Test fallback backward chain
        result = pln_tool._enhanced_fallback_backward_chain([], max_steps=2)
        assert isinstance(result, list)
        assert len(result) == 0  # No targets to process
    
    def test_premise_finding_methods(self):
        """Test premise finding methods for different reasoning types."""
        pln_tool = PLNReasoningTool()
        
        # Test modus ponens premise finding
        result = pln_tool._find_premises_for_modus_ponens(None)
        assert isinstance(result, list)
        assert len(result) == 0  # No atomspace, should return empty list
        
        # Test deduction premise finding
        result = pln_tool._find_premises_for_deduction(None)
        assert isinstance(result, list)
        assert len(result) == 0  # No atomspace, should return empty list
    
    def test_enhanced_apply_inference_rule(self):
        """Test enhanced apply_inference_rule method with new rules."""
        pln_tool = PLNReasoningTool()
        
        # Test enhanced rules
        enhanced_rules = [
            "fuzzy_conjunction_rule",
            "fuzzy_disjunction_rule",
            "consequent_disjunction_elimination_rule", 
            "contraposition_rule"
        ]
        
        for rule in enhanced_rules:
            result = pln_tool.apply_inference_rule(rule, [])
            assert isinstance(result, list), f"Rule {rule} should return list"
            assert len(result) == 0, f"Rule {rule} should return empty list for empty premises"
    
    def test_probabilistic_reasoning_integration(self):
        """Test integration of probabilistic reasoning with Agent-Zero framework."""
        # Test that PLNReasoningTool can be initialized independently
        pln_tool = PLNReasoningTool()
        
        # Check that enhanced PLN reasoning capabilities are available
        assert hasattr(pln_tool, 'probabilistic_modus_ponens')
        assert hasattr(pln_tool, 'probabilistic_deduction')
        assert hasattr(pln_tool, 'fuzzy_conjunction')
        assert hasattr(pln_tool, 'create_probabilistic_atom')
        
        # Test that the integration structure exists in the cognitive reasoning module
        from python.tools.cognitive_reasoning import CognitiveReasoningTool
        
        # Check that CognitiveReasoningTool class has PLN integration methods
        assert hasattr(CognitiveReasoningTool, 'perform_pln_logical_inference')
        assert hasattr(CognitiveReasoningTool, '_enhanced_pln_reasoning_with_tool')
    
    def test_error_handling_and_warnings(self):
        """Test proper error handling and warning messages."""
        pln_tool = PLNReasoningTool()
        
        # Test that methods don't crash with invalid inputs
        result = pln_tool.probabilistic_deduction([None, None])
        assert result is None or isinstance(result, list)
        
        result = pln_tool._apply_fuzzy_disjunction([None])
        assert isinstance(result, list)
        
        result = pln_tool._apply_contraposition([None])
        assert isinstance(result, list)


class TestPLNComponentIntegration:
    """Test PLN component integration with cpp2py pipeline."""
    
    def test_pln_component_cloned(self):
        """Test that PLN component was successfully cloned."""
        pln_component_path = Path("components/pln")
        assert pln_component_path.exists(), "PLN component should be cloned"
        
        # Check for key PLN files
        torchpln_path = pln_component_path / "opencog" / "torchpln" / "pln"
        assert torchpln_path.exists(), "TorchPLN should be available"
        
        common_py = torchpln_path / "common.py"
        assert common_py.exists(), "PLN common.py should be available"
    
    def test_pln_conversion_status(self):
        """Test that PLN conversion status is properly tracked."""
        status_file = Path("components/pln/conversion_status.json")
        
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
                
            assert status.get("component") == "pln"
            assert "cloned_at" in status
            assert status.get("status") == "cloned"


class TestTorchPLNIntegration:
    """Test TorchPLN specific integration features."""
    
    def test_torchpln_import_handling(self):
        """Test that TorchPLN imports are handled gracefully."""
        pln_tool = PLNReasoningTool()
        
        # The tool should initialize without crashing even if TorchPLN is not available
        assert pln_tool is not None
        assert isinstance(pln_tool.reasoning_rules, list)
        assert len(pln_tool.reasoning_rules) > 7  # Should have enhanced rules
    
    def test_tensor_truth_value_handling(self):
        """Test handling of TensorTruthValue functionality."""
        pln_tool = PLNReasoningTool()
        
        # Should handle TensorTruthValue operations gracefully
        # Even if TorchPLN is not available, fallbacks should work
        atom = pln_tool.create_probabilistic_atom("test", 0.7, 0.8)
        assert atom is None  # Expected without atomspace
    
    def test_pln_rule_base_initialization(self):
        """Test PLN rule base initialization process."""
        pln_tool = PLNReasoningTool()
        
        # Rule base should be None if TorchPLN is not available, but shouldn't crash
        # This tests the graceful degradation
        assert hasattr(pln_tool, 'rule_base')


if __name__ == "__main__":
    pytest.main([__file__, "-v"])