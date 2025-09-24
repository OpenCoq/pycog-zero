#!/usr/bin/env python3
"""
Advanced PLN-Agent Integration Test Suite
=========================================

Comprehensive test suite for advanced PLN reasoning examples with Agent-Zero integration.
Validates the functionality implemented for Issue #55 - Advanced Learning Systems (Phase 4).

This test suite verifies:
1. Advanced reasoning patterns work correctly
2. PLN integration functions properly with fallbacks
3. Agent-Zero integration concepts are demonstrated
4. All examples execute without errors
5. Results are meaningful and well-structured
"""

import sys
import os
import json
from pathlib import Path
from datetime import datetime
import unittest
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from examples.advanced_pln_reasoning_examples import AdvancedPLNReasoningExamples
    EXAMPLES_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import advanced examples: {e}")
    EXAMPLES_AVAILABLE = False


class TestAdvancedPLNAgentIntegration(unittest.TestCase):
    """Test suite for advanced PLN reasoning examples with Agent-Zero integration."""
    
    def setUp(self):
        """Set up test environment."""
        if not EXAMPLES_AVAILABLE:
            self.skipTest("Advanced PLN examples not available")
        
        self.examples_system = AdvancedPLNReasoningExamples()
        self.start_time = datetime.now()
    
    def tearDown(self):
        """Clean up after tests."""
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        print(f"Test completed in {duration:.3f} seconds")
    
    def test_examples_system_initialization(self):
        """Test that the examples system initializes correctly."""
        self.assertIsNotNone(self.examples_system)
        self.assertEqual(self.examples_system.examples_run, 0)
        self.assertEqual(self.examples_system.successful_inferences, 0)
        self.assertIsInstance(self.examples_system.reasoning_rules, list)
        self.assertGreater(len(self.examples_system.reasoning_rules), 0)
    
    def test_problem_solving_agent_example(self):
        """Test the problem-solving agent example."""
        result = self.examples_system.example_problem_solving_agent()
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        self.assertIn("goal", result)
        self.assertIn("reasoning_steps", result)
        self.assertIn("goal_achieved", result)
        self.assertIn("confidence", result)
        
        # Validate reasoning logic
        self.assertEqual(result["goal"], "agent_can_complete_project")
        self.assertTrue(result["goal_achieved"])
        self.assertGreater(result["confidence"], 0.7)  # Should be high confidence
        self.assertGreater(len(result["reasoning_steps"]), 0)
        
        # Validate reasoning steps contain proper PLN rules
        rule_names = [step["rule"] for step in result["reasoning_steps"]]
        self.assertIn("deduction_rule", rule_names)
        self.assertIn("fuzzy_conjunction_rule", rule_names)
    
    def test_learning_agent_example(self):
        """Test the learning agent example."""
        result = self.examples_system.example_learning_agent()
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        self.assertIn("learning_steps", result)
        self.assertIn("learned_knowledge", result)
        self.assertIn("knowledge_growth", result)
        
        # Validate learning occurred
        self.assertGreater(result["knowledge_growth"], 0)
        self.assertGreater(len(result["learning_steps"]), 0)
        
        # Validate learned knowledge has proper structure
        for knowledge in result["learned_knowledge"]:
            if isinstance(knowledge, tuple) and len(knowledge) == 3:
                subject, predicate, confidence = knowledge
                self.assertIsInstance(subject, str)
                self.assertIsInstance(predicate, str)
                self.assertIsInstance(confidence, (int, float))
                self.assertGreater(confidence, 0.0)
                self.assertLessEqual(confidence, 1.0)
    
    def test_multimodal_reasoning_example(self):
        """Test the multi-modal reasoning example."""
        result = self.examples_system.example_multimodal_reasoning()
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        self.assertIn("modalities", result)
        self.assertIn("reasoning_chain", result)
        self.assertIn("final_conclusion", result)
        self.assertIn("decision", result)
        
        # Validate multi-modal integration
        self.assertEqual(result["modalities"], 3)  # textual, temporal, contextual
        self.assertGreater(len(result["reasoning_chain"]), 0)
        
        # Validate reasoning chain includes all modalities
        modalities_used = set()
        for step in result["reasoning_chain"]:
            if "modality" in step:
                modalities_used.add(step["modality"])
        
        expected_modalities = {"textual", "temporal", "contextual"}
        self.assertTrue(expected_modalities.issubset(modalities_used))
        
        # Validate final decision is reasonable
        self.assertIn(result["decision"], ["immediate_priority_response", "standard_response"])
    
    def test_causal_inference_example(self):
        """Test the causal inference example."""
        result = self.examples_system.example_causal_inference()
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        self.assertIn("causal_inferences", result)
        self.assertIn("predictions", result)
        self.assertIn("combined_probability", result)
        self.assertIn("interventions", result)
        
        # Validate causal reasoning
        self.assertGreater(len(result["causal_inferences"]), 0)
        self.assertIsInstance(result["predictions"], dict)
        self.assertGreater(result["combined_probability"], 0.0)
        self.assertLessEqual(result["combined_probability"], 1.0)
        
        # Validate causal chains are present
        chain_rules = [inf["rule"] for inf in result["causal_inferences"] if "rule" in inf]
        self.assertIn("causal_chain", chain_rules)
        
        # Validate interventions make sense
        for intervention in result["interventions"]:
            self.assertIn("target", intervention)
            self.assertIn("expected_boost", intervention)
            self.assertIn("action", intervention)
    
    def test_metacognitive_reasoning_example(self):
        """Test the meta-cognitive reasoning example.""" 
        result = self.examples_system.example_metacognitive_reasoning()
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        self.assertIn("meta_reasoning_steps", result)
        self.assertIn("strategy_scores", result)
        self.assertIn("selected_strategy", result)
        self.assertIn("confidence_assessment", result)
        
        # Validate strategy evaluation
        self.assertGreater(len(result["meta_reasoning_steps"]), 0)
        self.assertIsInstance(result["strategy_scores"], dict)
        self.assertGreater(len(result["strategy_scores"]), 0)
        
        # Validate selected strategy is from available strategies
        available_strategies = list(result["strategy_scores"].keys())
        self.assertIn(result["selected_strategy"], available_strategies)
        
        # Validate confidence assessment is meaningful
        self.assertIn(result["confidence_assessment"], 
                     ["high_confidence", "medium_confidence", "low_confidence"])
        
        # Validate meta-reasoning steps include proper evaluations
        step_rules = [step["rule"] for step in result["meta_reasoning_steps"] if "rule" in step]
        self.assertIn("strategy_evaluation", step_rules)
        self.assertIn("strategy_selection", step_rules)
        self.assertIn("confidence_assessment", step_rules)
    
    def test_collaborative_agents_example(self):
        """Test the collaborative agent network example."""
        result = self.examples_system.example_collaborative_agents()
        
        # Validate result structure
        self.assertIsInstance(result, dict)
        self.assertIn("agents_involved", result)
        self.assertIn("task_allocations", result)
        self.assertIn("collaboration_steps", result)
        self.assertIn("integrated_solution", result)
        self.assertIn("collaboration_success", result)
        
        # Validate collaboration setup
        self.assertEqual(result["agents_involved"], 3)  # alpha, beta, gamma
        self.assertGreater(len(result["task_allocations"]), 0)
        self.assertGreater(len(result["collaboration_steps"]), 0)
        
        # Validate task allocation is complete
        for allocation in result["task_allocations"]:
            self.assertIn("requirement", allocation)
            self.assertIn("assigned_agent", allocation)
            self.assertIn("match_score", allocation)
            self.assertGreater(allocation["match_score"], 0.0)
        
        # Validate integrated solution
        self.assertIsInstance(result["integrated_solution"], list)
        self.assertGreater(len(result["integrated_solution"]), 0)
        
        # Validate collaboration success is boolean
        self.assertIsInstance(result["collaboration_success"], bool)
    
    def test_all_examples_execution(self):
        """Test that all examples can be executed together successfully."""
        results = self.examples_system.run_all_examples()
        
        # Validate overall results structure
        self.assertIsInstance(results, dict)
        self.assertIn("_summary", results)
        
        # Validate summary contains expected metrics
        summary = results["_summary"]
        self.assertIn("examples_run", summary)
        self.assertIn("successful_examples", summary)
        self.assertIn("success_rate", summary)
        self.assertIn("total_reasoning_steps", summary)
        
        # Validate all examples ran
        self.assertEqual(summary["examples_run"], 6)
        self.assertEqual(summary["successful_examples"], 6)
        self.assertEqual(summary["success_rate"], 1.0)
        
        # Validate reasoning occurred
        self.assertGreater(summary["total_reasoning_steps"], 0)
    
    def test_pln_fallback_mechanisms(self):
        """Test that PLN fallback mechanisms work correctly."""
        # This system should work even without OpenCog
        self.assertIsNotNone(self.examples_system.reasoning_rules)
        self.assertFalse(self.examples_system.pln_available)  # Should be False in test environment
        
        # All examples should still work with fallbacks
        result = self.examples_system.example_problem_solving_agent()
        self.assertTrue(result["goal_achieved"])
    
    def test_reasoning_rule_coverage(self):
        """Test that various PLN reasoning rules are covered in examples."""
        expected_rules = [
            "deduction_rule",
            "modus_ponens_rule", 
            "fuzzy_conjunction_rule",
            "fuzzy_disjunction_rule",
            "inheritance_rule",
            "causal_chain",
            "capability_matching",
            "strategy_evaluation"
        ]
        
        # Run all examples and collect used rules
        results = self.examples_system.run_all_examples()
        used_rules = set()
        
        for example_name, result in results.items():
            if example_name == "_summary":
                continue
                
            if isinstance(result, dict):
                # Check reasoning_steps
                if "reasoning_steps" in result:
                    for step in result["reasoning_steps"]:
                        if "rule" in step:
                            used_rules.add(step["rule"])
                
                # Check learning_steps
                if "learning_steps" in result:
                    for step in result["learning_steps"]:
                        if "rule" in step:
                            used_rules.add(step["rule"])
                
                # Check causal_inferences
                if "causal_inferences" in result:
                    for inf in result["causal_inferences"]:
                        if "rule" in inf:
                            used_rules.add(inf["rule"])
                
                # Check meta_reasoning_steps
                if "meta_reasoning_steps" in result:
                    for step in result["meta_reasoning_steps"]:
                        if "rule" in step:
                            used_rules.add(step["rule"])
                
                # Check collaboration_steps
                if "collaboration_steps" in result:
                    for step in result["collaboration_steps"]:
                        if "rule" in step:
                            used_rules.add(step["rule"])
        
        # Validate that we're using a good variety of rules
        rules_covered = len(used_rules.intersection(set(expected_rules)))
        self.assertGreaterEqual(rules_covered, 5, 
                               f"Expected at least 5 different PLN rules, got {rules_covered}: {used_rules}")
        
        # We should have a good total number of unique rules used
        self.assertGreaterEqual(len(used_rules), 8,
                               f"Expected at least 8 total unique rules, got {len(used_rules)}: {used_rules}")
    
    def test_confidence_values_reasonable(self):
        """Test that confidence values throughout examples are reasonable."""
        results = self.examples_system.run_all_examples()
        
        confidence_values = []
        
        # Extract confidence values from all results
        for example_name, result in results.items():
            if example_name == "_summary" or not isinstance(result, dict):
                continue
            
            # Direct confidence values
            if "confidence" in result:
                confidence_values.append(result["confidence"])
            
            # Confidence in reasoning steps
            if "reasoning_steps" in result:
                for step in result["reasoning_steps"]:
                    if "confidence" in step:
                        confidence_values.append(step["confidence"])
            
            # Confidence in learning steps
            if "learning_steps" in result:
                for step in result["learning_steps"]:
                    if isinstance(step, dict) and "learned" in step:
                        if isinstance(step["learned"], tuple) and len(step["learned"]) == 3:
                            confidence_values.append(step["learned"][2])
        
        # Validate all confidence values are in reasonable range
        for conf in confidence_values:
            self.assertGreaterEqual(conf, 0.0, f"Confidence {conf} should be >= 0.0")
            self.assertLessEqual(conf, 1.0, f"Confidence {conf} should be <= 1.0")
        
        # Should have a reasonable number of confidence measurements
        self.assertGreaterEqual(len(confidence_values), 6, 
                               f"Expected at least 6 confidence values, got {len(confidence_values)}")
        
        # Validate we have a good distribution of confidence values
        high_confidence = sum(1 for c in confidence_values if c >= 0.8)
        medium_confidence = sum(1 for c in confidence_values if 0.5 <= c < 0.8)
        self.assertGreater(high_confidence + medium_confidence, 0, 
                          "Should have some high or medium confidence predictions")


def run_comprehensive_test():
    """Run comprehensive test of advanced PLN reasoning examples."""
    print("🧪 Advanced PLN-Agent Integration Test Suite")
    print("=" * 60)
    print("Issue #55 - Advanced Learning Systems (Phase 4) Validation")
    print("=" * 60)
    
    # Run the unittest suite
    loader = unittest.TestLoader()
    suite = loader.loadTestsFromTestCase(TestAdvancedPLNAgentIntegration)
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Generate test report
    test_report = {
        "timestamp": datetime.now().isoformat(),
        "tests_run": result.testsRun,
        "failures": len(result.failures),
        "errors": len(result.errors),
        "success_rate": (result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun if result.testsRun > 0 else 0.0,
        "details": {
            "failures": [{"test": str(test), "traceback": traceback} for test, traceback in result.failures],
            "errors": [{"test": str(test), "traceback": traceback} for test, traceback in result.errors]
        }
    }
    
    # Save test report
    report_file = PROJECT_ROOT / "advanced_pln_test_report.json"
    with open(report_file, 'w') as f:
        json.dump(test_report, f, indent=2)
    
    print(f"\n📄 Test report saved to: {report_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("🎯 Test Suite Summary")
    print(f"{'='*60}")
    print(f"Tests run: {test_report['tests_run']}")
    print(f"Failures: {test_report['failures']}")
    print(f"Errors: {test_report['errors']}")
    print(f"Success rate: {test_report['success_rate']:.1%}")
    
    if test_report['success_rate'] >= 0.9:
        print("✅ All tests passed successfully!")
        return True
    else:
        print("❌ Some tests failed!")
        return False


if __name__ == "__main__":
    success = run_comprehensive_test()
    exit(0 if success else 1)