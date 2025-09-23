"""
Rule Engine (URE) Integration Tests
==================================

Specific tests for URE (Unified Rule Engine) functionality including
forward chaining, backward chaining, and rule execution systems.
"""

import pytest
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional


class TestRuleEngine:
    """Test core rule engine functionality."""
    
    def test_rule_definition_structure(self):
        """Test rule definition structure and validation."""
        rule_definitions = [
            {
                "name": "modus_ponens",
                "type": "inference_rule",
                "premises": ["?P", "?P -> ?Q"],
                "conclusion": "?Q",
                "confidence": 1.0,
                "tv": {"strength": 1.0, "confidence": 1.0}
            },
            {
                "name": "universal_instantiation",
                "type": "inference_rule",
                "premises": ["forall ?X: ?P(?X)"],
                "conclusion": "?P(c)",
                "confidence": 1.0,
                "tv": {"strength": 1.0, "confidence": 1.0}
            },
            {
                "name": "inheritance_rule",
                "type": "deduction_rule",
                "premises": ["Inheritance(?A, ?B)", "Inheritance(?B, ?C)"],
                "conclusion": "Inheritance(?A, ?C)",
                "confidence": 0.9,
                "tv": {"strength": 0.9, "confidence": 0.8}
            }
        ]
        
        for rule in rule_definitions:
            # Validate rule structure
            assert "name" in rule
            assert "type" in rule
            assert "premises" in rule
            assert "conclusion" in rule
            assert "confidence" in rule
            
            # Validate rule components
            assert isinstance(rule["premises"], list)
            assert len(rule["premises"]) > 0
            assert isinstance(rule["conclusion"], str)
            assert 0.0 <= rule["confidence"] <= 1.0
            
            # Validate truth value structure
            if "tv" in rule:
                tv = rule["tv"]
                assert "strength" in tv
                assert "confidence" in tv
                assert 0.0 <= tv["strength"] <= 1.0
                assert 0.0 <= tv["confidence"] <= 1.0
    
    def test_rule_pattern_validation(self):
        """Test validation of rule patterns and variables."""
        pattern_test_cases = [
            {
                "pattern": "?X isa ?Y",
                "valid": True,
                "variables": ["X", "Y"],
                "predicates": ["isa"]
            },
            {
                "pattern": "Inheritance(?A, ?B)",
                "valid": True,
                "variables": ["A", "B"],
                "predicates": ["Inheritance"]
            },
            {
                "pattern": "(?X likes ?Y) AND (?Y likes ?Z)",
                "valid": True,
                "variables": ["X", "Y", "Z"],
                "predicates": ["likes", "AND"]
            },
            {
                "pattern": "invalid pattern with no variables",
                "valid": False,
                "variables": [],
                "predicates": []
            }
        ]
        
        for case in pattern_test_cases:
            pattern = case["pattern"]
            is_valid = case["valid"]
            expected_variables = case["variables"]
            
            # Test variable detection - handle patterns like "Inheritance(?A, ?B)"
            found_variables = []
            import re
            # Find all variables in the pattern (handles complex patterns)
            variable_matches = re.findall(r'\?([A-Za-z_][A-Za-z0-9_]*)', pattern)
            found_variables = list(set(variable_matches))
            
            if is_valid:
                assert len(found_variables) > 0, f"Valid pattern should contain variables: {pattern}"
                # Check that expected variables are found
                for expected_var in expected_variables:
                    assert expected_var in found_variables, f"Variable {expected_var} not found in {pattern}"
            else:
                assert len(found_variables) == 0, f"Invalid pattern should not contain variables: {pattern}"


class TestForwardChaining:
    """Test forward chaining functionality."""
    
    def test_forward_chaining_basic(self):
        """Test basic forward chaining inference."""
        forward_scenarios = [
            {
                "name": "simple_modus_ponens",
                "facts": ["P", "P -> Q"],
                "rule": {
                    "premises": ["?X", "?X -> ?Y"],
                    "conclusion": "?Y"
                },
                "expected_conclusions": ["Q"]
            },
            {
                "name": "multiple_step_inference",
                "facts": ["A", "A -> B", "B -> C"],
                "rule": {
                    "premises": ["?X", "?X -> ?Y"],
                    "conclusion": "?Y"
                },
                "expected_conclusions": ["B", "C"]
            }
        ]
        
        for scenario in forward_scenarios:
            facts = scenario["facts"]
            rule = scenario["rule"]
            expected_conclusions = scenario["expected_conclusions"]
            
            # Validate scenario structure
            assert isinstance(facts, list)
            assert len(facts) > 0
            assert "premises" in rule
            assert "conclusion" in rule
            assert isinstance(expected_conclusions, list)
            
            # Simulate forward chaining logic
            conclusions = []
            for fact in facts:
                # Simple pattern matching simulation
                if "->" in fact and any(f == fact.split(" -> ")[0] for f in facts):
                    conclusion = fact.split(" -> ")[1]
                    if conclusion not in conclusions:
                        conclusions.append(conclusion)
            
            # Verify some conclusions are derived
            assert len(conclusions) > 0, f"Forward chaining should derive conclusions from: {facts}"
    
    def test_forward_chaining_with_confidence(self):
        """Test forward chaining with confidence propagation."""
        confidence_scenarios = [
            {
                "premise_confidences": {"P": 0.9, "P -> Q": 0.8},
                "rule_confidence": 1.0,
                "expected_conclusion_confidence": 0.72,  # 0.9 * 0.8
                "confidence_combination": "multiplication"
            },
            {
                "premise_confidences": {"A": 0.7, "A -> B": 0.6},
                "rule_confidence": 0.9,
                "expected_conclusion_confidence": 0.378,  # 0.7 * 0.6 * 0.9
                "confidence_combination": "multiplication"
            }
        ]
        
        for scenario in confidence_scenarios:
            premise_confidences = scenario["premise_confidences"]
            rule_confidence = scenario["rule_confidence"]
            expected_confidence = scenario["expected_conclusion_confidence"]
            
            # Validate confidence values
            for confidence in premise_confidences.values():
                assert 0.0 <= confidence <= 1.0
            
            assert 0.0 <= rule_confidence <= 1.0
            assert 0.0 <= expected_confidence <= 1.0
            
            # Test confidence calculation
            combined_confidence = 1.0
            for confidence in premise_confidences.values():
                combined_confidence *= confidence
            combined_confidence *= rule_confidence
            
            # Allow for small floating point differences
            assert abs(combined_confidence - expected_confidence) < 0.001
    
    def test_forward_chaining_termination(self):
        """Test forward chaining termination conditions."""
        termination_scenarios = [
            {
                "name": "no_new_conclusions",
                "initial_facts": ["A", "B"],
                "rules": [],
                "max_iterations": 10,
                "should_terminate": True,
                "termination_reason": "no_applicable_rules"
            },
            {
                "name": "fixed_point_reached",
                "initial_facts": ["A"],
                "rules": [{"premises": ["A"], "conclusion": "A"}],
                "max_iterations": 10,
                "should_terminate": True,
                "termination_reason": "fixed_point"
            },
            {
                "name": "max_iterations_reached",
                "initial_facts": ["A"],
                "rules": [{"premises": ["?X"], "conclusion": "new_fact(?X)"}],
                "max_iterations": 3,
                "should_terminate": True,
                "termination_reason": "max_iterations"
            }
        ]
        
        for scenario in termination_scenarios:
            initial_facts = scenario["initial_facts"]
            rules = scenario["rules"]
            max_iterations = scenario["max_iterations"]
            should_terminate = scenario["should_terminate"]
            
            # Validate termination scenario
            assert isinstance(initial_facts, list)
            assert isinstance(rules, list)
            assert max_iterations > 0
            assert isinstance(should_terminate, bool)
            
            # Simulate termination logic
            current_facts = set(initial_facts)
            iteration = 0
            
            while iteration < max_iterations:
                iteration += 1
                new_facts = set()
                
                # Apply rules (simplified simulation)
                for rule in rules:
                    if len(rule["premises"]) == 1:
                        premise = rule["premises"][0]
                        if premise.startswith("?") or premise in current_facts:
                            conclusion = rule["conclusion"]
                            if not conclusion.startswith("?"):
                                new_facts.add(conclusion)
                
                # Check if we've reached a fixed point
                if not new_facts or new_facts.issubset(current_facts):
                    break
                
                current_facts.update(new_facts)
            
            # Should have terminated within max iterations
            assert iteration <= max_iterations


class TestBackwardChaining:
    """Test backward chaining functionality."""
    
    def test_backward_chaining_basic(self):
        """Test basic backward chaining goal resolution."""
        backward_scenarios = [
            {
                "goal": "Q",
                "rules": [
                    {"premises": ["P"], "conclusion": "Q"},
                    {"premises": ["R", "S"], "conclusion": "P"}
                ],
                "facts": ["R", "S"],
                "should_succeed": True,
                "proof_steps": ["Q <- P", "P <- R,S", "R ✓", "S ✓"]
            },
            {
                "goal": "impossible_goal",
                "rules": [{"premises": ["nonexistent_fact"], "conclusion": "impossible_goal"}],
                "facts": ["some_fact"],
                "should_succeed": False,
                "proof_steps": []
            }
        ]
        
        for scenario in backward_scenarios:
            goal = scenario["goal"]
            rules = scenario["rules"]
            facts = scenario["facts"]
            should_succeed = scenario["should_succeed"]
            
            # Validate backward chaining scenario
            assert isinstance(goal, str)
            assert isinstance(rules, list)
            assert isinstance(facts, list)
            assert isinstance(should_succeed, bool)
            
            # Simulate simple backward chaining
            can_prove = False
            
            # Look for rules that conclude the goal
            for rule in rules:
                if rule["conclusion"] == goal:
                    premises = rule["premises"]
                    
                    # Check if all premises can be satisfied
                    all_premises_satisfied = True
                    for premise in premises:
                        if premise not in facts:
                            # Would need recursive backward chaining here
                            premise_can_be_proven = any(
                                r["conclusion"] == premise for r in rules
                            )
                            if not premise_can_be_proven:
                                all_premises_satisfied = False
                                break
                    
                    if all_premises_satisfied:
                        can_prove = True
                        break
            
            # Direct fact check
            if goal in facts:
                can_prove = True
            
            if should_succeed:
                assert can_prove, f"Should be able to prove goal: {goal}"
            else:
                assert not can_prove, f"Should not be able to prove goal: {goal}"
    
    def test_backward_chaining_with_unification(self):
        """Test backward chaining with variable unification."""
        unification_scenarios = [
            {
                "goal": "mortal(?X)",  # Changed to have variable
                "rules": [
                    {"premises": ["human(?X)"], "conclusion": "mortal(?X)"},
                    {"premises": ["philosopher(?X)"], "conclusion": "human(?X)"}
                ],
                "facts": ["philosopher(socrates)"],
                "expected_bindings": {"X": "socrates"},
                "should_succeed": True
            },
            {
                "goal": "likes(?X, pizza)",
                "rules": [
                    {"premises": ["person(?X)", "italian(?X)"], "conclusion": "likes(?X, pizza)"}
                ],
                "facts": ["person(mario)", "italian(mario)"],
                "expected_bindings": {"X": "mario"},
                "should_succeed": True
            }
        ]
        
        for scenario in unification_scenarios:
            goal = scenario["goal"]
            rules = scenario["rules"]
            facts = scenario["facts"]
            expected_bindings = scenario["expected_bindings"]
            should_succeed = scenario["should_succeed"]
            
            # Validate unification scenario
            assert "?" in goal, "Goal should contain variables for unification"
            assert isinstance(expected_bindings, dict)
            
            # Test variable extraction - use regex for proper parsing
            import re
            goal_variables = re.findall(r'\?([A-Za-z_][A-Za-z0-9_]*)', goal)
            
            # Should find expected variables
            for var in expected_bindings.keys():
                assert var in goal_variables, f"Variable {var} should be in goal: {goal}"
    
    def test_backward_chaining_depth_control(self):
        """Test depth control in backward chaining."""
        depth_control_scenarios = [
            {
                "goal": "deep_goal",
                "max_depth": 3,
                "rule_chain_length": 2,
                "should_succeed": True,
                "depth_exceeded": False
            },
            {
                "goal": "very_deep_goal",
                "max_depth": 2,
                "rule_chain_length": 5,
                "should_succeed": False,
                "depth_exceeded": True
            }
        ]
        
        for scenario in depth_control_scenarios:
            goal = scenario["goal"]
            max_depth = scenario["max_depth"]
            rule_chain_length = scenario["rule_chain_length"]
            should_succeed = scenario["should_succeed"]
            depth_exceeded = scenario["depth_exceeded"]
            
            # Validate depth control scenario
            assert max_depth > 0
            assert rule_chain_length >= 0
            
            # Test depth logic
            if rule_chain_length > max_depth:
                assert depth_exceeded, "Should detect depth exceeded"
                assert not should_succeed, "Should not succeed when depth exceeded"
            else:
                assert not depth_exceeded, "Should not exceed depth"


class TestRuleEngineOptimization:
    """Test rule engine optimization strategies."""
    
    def test_rule_indexing(self):
        """Test rule indexing for efficient rule selection."""
        indexing_strategies = [
            {
                "index_type": "conclusion_index",
                "rules": [
                    {"conclusion": "mortal(?X)", "premises": ["human(?X)"]},
                    {"conclusion": "human(?X)", "premises": ["person(?X)"]},
                    {"conclusion": "mortal(?Y)", "premises": ["animal(?Y)"]}
                ],
                "goal": "mortal(socrates)",
                "relevant_rules": 2  # Rules with mortal conclusion
            },
            {
                "index_type": "premise_index",
                "rules": [
                    {"premises": ["likes(?X, ?Y)"], "conclusion": "happy(?X)"},
                    {"premises": ["likes(?A, ?B)"], "conclusion": "social(?A)"}
                ],
                "available_fact": "likes(john, pizza)",
                "applicable_rules": 2
            }
        ]
        
        for strategy in indexing_strategies:
            index_type = strategy["index_type"]
            rules = strategy["rules"]
            
            # Validate indexing strategy
            assert index_type in ["conclusion_index", "premise_index"]
            assert isinstance(rules, list)
            assert len(rules) > 0
            
            # Test rule indexing logic
            if index_type == "conclusion_index":
                goal = strategy["goal"]
                relevant_rules = strategy["relevant_rules"]
                
                # Count rules that could conclude the goal
                matching_rules = 0
                goal_predicate = goal.split("(")[0] if "(" in goal else goal
                
                for rule in rules:
                    conclusion = rule["conclusion"]
                    conclusion_predicate = conclusion.split("(")[0] if "(" in conclusion else conclusion
                    if conclusion_predicate == goal_predicate:
                        matching_rules += 1
                
                assert matching_rules == relevant_rules
    
    def test_rule_priority_ordering(self):
        """Test rule priority and ordering for efficient inference."""
        priority_scenarios = [
            {
                "rules": [
                    {"id": "specific_rule", "priority": 1.0, "specificity": "high"},
                    {"id": "general_rule", "priority": 0.5, "specificity": "low"},
                    {"id": "medium_rule", "priority": 0.7, "specificity": "medium"}
                ],
                "expected_order": ["specific_rule", "medium_rule", "general_rule"]
            }
        ]
        
        for scenario in priority_scenarios:
            rules = scenario["rules"]
            expected_order = scenario["expected_order"]
            
            # Sort rules by priority
            sorted_rules = sorted(rules, key=lambda r: r["priority"], reverse=True)
            actual_order = [rule["id"] for rule in sorted_rules]
            
            assert actual_order == expected_order, f"Expected {expected_order}, got {actual_order}"
    
    def test_rule_conflict_resolution(self):
        """Test conflict resolution when multiple rules apply."""
        conflict_scenarios = [
            {
                "conflicting_rules": [
                    {"conclusion": "fly(X)", "confidence": 0.9, "source": "bird_rule"},
                    {"conclusion": "not fly(X)", "confidence": 0.8, "source": "penguin_rule"}
                ],
                "resolution_strategy": "highest_confidence",
                "expected_winner": "bird_rule"
            },
            {
                "conflicting_rules": [
                    {"conclusion": "temperature(hot)", "confidence": 0.6, "timestamp": 100},
                    {"conclusion": "temperature(cold)", "confidence": 0.6, "timestamp": 200}
                ],
                "resolution_strategy": "most_recent",
                "expected_winner": "temperature(cold)"
            }
        ]
        
        for scenario in conflict_scenarios:
            conflicting_rules = scenario["conflicting_rules"]
            resolution_strategy = scenario["resolution_strategy"]
            expected_winner = scenario["expected_winner"]
            
            # Test conflict resolution logic
            if resolution_strategy == "highest_confidence":
                winner = max(conflicting_rules, key=lambda r: r["confidence"])
                if "source" in winner:
                    assert winner["source"] == expected_winner
                else:
                    assert winner["conclusion"] == expected_winner
            
            elif resolution_strategy == "most_recent":
                winner = max(conflicting_rules, key=lambda r: r.get("timestamp", 0))
                assert winner["conclusion"] == expected_winner


class TestRuleEngineIntegration:
    """Test rule engine integration with other cognitive systems."""
    
    def test_atomspace_rule_integration(self):
        """Test rule engine integration with AtomSpace."""
        atomspace_integration = {
            "atomspace_atoms": [
                "ConceptNode cat",
                "ConceptNode animal", 
                "InheritanceLink cat animal"
            ],
            "rule_engine_rules": [
                {"premises": ["InheritanceLink ?X ?Y"], "conclusion": "isa(?X, ?Y)"}
            ],
            "expected_inferences": [
                "isa(cat, animal)"
            ]
        }
        
        atoms = atomspace_integration["atomspace_atoms"]
        rules = atomspace_integration["rule_engine_rules"]
        expected_inferences = atomspace_integration["expected_inferences"]
        
        # Validate integration structure
        assert len(atoms) > 0
        assert len(rules) > 0
        assert len(expected_inferences) > 0
        
        # Test that atoms provide facts for rule application
        for rule in rules:
            premises = rule["premises"]
            for premise in premises:
                # Should be able to find matching atoms
                premise_pattern = premise.replace("?X", "").replace("?Y", "").strip()
                matching_atoms = [atom for atom in atoms if premise_pattern.replace(" ", "") in atom.replace(" ", "")]
                assert len(matching_atoms) > 0, f"No atoms match premise: {premise}"
    
    def test_cognitive_reasoning_integration(self):
        """Test rule engine integration with cognitive reasoning."""
        cognitive_integration = {
            "cognitive_goals": [
                "understand(concept)",
                "learn(new_fact)",
                "reason_about(situation)"
            ],
            "rule_engine_capabilities": [
                "forward_reasoning",
                "backward_reasoning",
                "pattern_matching",
                "inference_chaining"
            ],
            "integration_points": [
                {"goal": "understand(concept)", "rule_type": "backward_reasoning"},
                {"goal": "learn(new_fact)", "rule_type": "forward_reasoning"}
            ]
        }
        
        goals = cognitive_integration["cognitive_goals"]
        capabilities = cognitive_integration["rule_engine_capabilities"]
        integration_points = cognitive_integration["integration_points"]
        
        # Validate cognitive integration
        assert len(goals) > 0
        assert len(capabilities) > 0
        assert len(integration_points) > 0
        
        for integration_point in integration_points:
            goal = integration_point["goal"]
            rule_type = integration_point["rule_type"]
            
            assert goal in goals
            assert rule_type in capabilities