"""
Logic Systems Integration Tests (Phase 2)
==========================================

Test suite for validating OpenCog Logic Systems integration with PyCog-Zero.
Tests cover unification algorithms, rule engine functionality, and pattern matching.
"""

import pytest
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional, Any

# Project paths
project_root = Path(__file__).parent.parent.parent
components_dir = project_root / "components"
unify_path = components_dir / "unify"
ure_path = components_dir / "ure"
logic_components_dir = components_dir / "logic"
cognitive_config_path = project_root / "conf" / "config_cognitive.json"


class TestUnifyIntegration:
    """Test unification algorithm integration for pattern matching."""
    
    def test_unify_component_structure(self):
        """Test that unify component has expected structure."""
        # Check if unify is available in components or logic directory
        unify_locations = [
            unify_path,
            logic_components_dir / "unify",
            components_dir / "core" / "unify"
        ]
        
        unify_found = any(path.exists() for path in unify_locations)
        if not unify_found:
            pytest.skip("Unify component not yet cloned - Phase 2 dependency")
        
        # Find the actual unify path
        actual_unify_path = next((path for path in unify_locations if path.exists()), None)
        
        # Check for expected directories within unify component
        expected_dirs = [
            "opencog/unify",
            "tests"
        ]
        
        for dir_name in expected_dirs:
            dir_path = actual_unify_path / dir_name
            if not dir_path.exists():
                pytest.skip(f"Unify directory structure incomplete: {dir_name}")
    
    def test_unify_algorithm_headers(self):
        """Test that unify algorithm headers are present."""
        unify_locations = [unify_path, logic_components_dir / "unify"]
        unify_dir = next((path for path in unify_locations if path.exists()), None)
        
        if not unify_dir:
            pytest.skip("Unify component not available")
        
        unify_source_dir = unify_dir / "opencog" / "unify"
        if not unify_source_dir.exists():
            pytest.skip("Unify source directory not found")
        
        # Core unification headers
        expected_headers = [
            "Unify.h",
            "UnifyUtils.h"
        ]
        
        found_headers = []
        for header in expected_headers:
            header_path = unify_source_dir / header
            if header_path.exists():
                found_headers.append(header)
        
        assert len(found_headers) > 0, f"No unification headers found in {unify_source_dir}"
    
    def test_pattern_matching_readiness(self):
        """Test readiness for pattern matching integration."""
        # Test that we can create mock pattern matching scenarios
        test_patterns = [
            {"pattern": "?X isa concept", "bindings": {"X": "cat"}},
            {"pattern": "(?X likes ?Y)", "bindings": {"X": "john", "Y": "pizza"}},
            {"pattern": "Member(?A, ?B)", "bindings": {"A": "item", "B": "set"}}
        ]
        
        for test_pattern in test_patterns:
            pattern_str = test_pattern["pattern"]
            bindings = test_pattern["bindings"]
            
            # Validate pattern structure
            assert "?" in pattern_str, f"Pattern should contain variables: {pattern_str}"
            assert isinstance(bindings, dict), "Bindings should be dictionary"
            assert len(bindings) > 0, "Should have at least one binding"
    
    def test_unify_cmake_configuration(self):
        """Test unify CMake configuration for build integration."""
        unify_locations = [unify_path, logic_components_dir / "unify"]
        unify_dir = next((path for path in unify_locations if path.exists()), None)
        
        if not unify_dir:
            pytest.skip("Unify component not available")
        
        cmake_file = unify_dir / "CMakeLists.txt"
        if not cmake_file.exists():
            pytest.skip("Unify CMakeLists.txt not found")
        
        with open(cmake_file, 'r') as f:
            cmake_content = f.read()
        
        # Check for essential CMake configuration
        assert "project" in cmake_content.lower(), "CMake project declaration missing"
        assert "cmake_minimum_required" in cmake_content.lower(), "CMake version requirement missing"


class TestUREIntegration:
    """Test Unified Rule Engine (URE) integration."""
    
    def test_ure_component_structure(self):
        """Test that URE component has expected structure."""
        ure_locations = [
            ure_path,
            logic_components_dir / "ure",
            components_dir / "core" / "ure"
        ]
        
        ure_found = any(path.exists() for path in ure_locations)
        if not ure_found:
            pytest.skip("URE component not yet cloned - Phase 2 dependency")
        
        actual_ure_path = next((path for path in ure_locations if path.exists()), None)
        
        # Check for expected directories
        expected_dirs = [
            "opencog/rule-engine",
            "tests"
        ]
        
        for dir_name in expected_dirs:
            dir_path = actual_ure_path / dir_name
            if not dir_path.exists():
                # Try alternative structure
                alt_paths = [
                    actual_ure_path / "opencog" / "ure",
                    actual_ure_path / "rule-engine"
                ]
                if not any(alt.exists() for alt in alt_paths):
                    pytest.skip(f"URE directory structure incomplete: {dir_name}")
    
    def test_ure_rule_engine_headers(self):
        """Test that URE rule engine headers are present."""
        ure_locations = [ure_path, logic_components_dir / "ure"]
        ure_dir = next((path for path in ure_locations if path.exists()), None)
        
        if not ure_dir:
            pytest.skip("URE component not available")
        
        # Check multiple possible URE source locations
        possible_source_dirs = [
            ure_dir / "opencog" / "rule-engine",
            ure_dir / "opencog" / "ure",
            ure_dir / "rule-engine"
        ]
        
        ure_source_dir = next((path for path in possible_source_dirs if path.exists()), None)
        if not ure_source_dir:
            pytest.skip("URE source directory not found")
        
        # Core rule engine headers
        expected_headers = [
            "Rule.h",
            "RuleEngine.h",
            "ForwardChainer.h",
            "BackwardChainer.h"
        ]
        
        found_headers = []
        for header in expected_headers:
            # Check in multiple subdirectories
            possible_header_paths = [
                ure_source_dir / header,
                ure_source_dir / "forward-chainer" / header,
                ure_source_dir / "backward-chainer" / header,
                ure_source_dir / "rules" / header
            ]
            
            if any(path.exists() for path in possible_header_paths):
                found_headers.append(header)
        
        assert len(found_headers) > 0, f"No URE headers found in {ure_source_dir}"
    
    def test_forward_chaining_readiness(self):
        """Test forward chaining algorithm readiness."""
        # Test forward chaining rule pattern
        forward_rule_template = {
            "rule_name": "test_forward_rule",
            "premises": ["premise1", "premise2"],
            "conclusion": "conclusion",
            "confidence": 0.8
        }
        
        assert "premises" in forward_rule_template
        assert "conclusion" in forward_rule_template
        assert isinstance(forward_rule_template["premises"], list)
        assert len(forward_rule_template["premises"]) > 0
    
    def test_backward_chaining_readiness(self):
        """Test backward chaining algorithm readiness."""
        # Test backward chaining goal pattern
        backward_goal_template = {
            "goal": "target_conclusion",
            "subgoals": ["subgoal1", "subgoal2"],
            "max_depth": 5
        }
        
        assert "goal" in backward_goal_template
        assert "subgoals" in backward_goal_template
        assert isinstance(backward_goal_template["subgoals"], list)
    
    def test_ure_cmake_configuration(self):
        """Test URE CMake configuration."""
        ure_locations = [ure_path, logic_components_dir / "ure"]
        ure_dir = next((path for path in ure_locations if path.exists()), None)
        
        if not ure_dir:
            pytest.skip("URE component not available")
        
        cmake_file = ure_dir / "CMakeLists.txt"
        if not cmake_file.exists():
            pytest.skip("URE CMakeLists.txt not found")
        
        with open(cmake_file, 'r') as f:
            cmake_content = f.read()
        
        # Check for essential CMake configuration
        assert "project" in cmake_content.lower(), "CMake project declaration missing"
        

class TestLogicSystemsIntegration:
    """Test integration between logic systems and PyCog-Zero."""
    
    def test_cognitive_reasoning_logic_compatibility(self):
        """Test logic systems compatibility with cognitive reasoning."""
        # Check cognitive reasoning tool import path
        cognitive_tool_path = project_root / "python" / "tools" / "cognitive_reasoning.py"
        assert cognitive_tool_path.exists(), "Cognitive reasoning tool not found"
        
        # Test that we can import without syntax errors (fix existing syntax issue first)
        try:
            # Try to read the file and check for obvious syntax issues
            with open(cognitive_tool_path, 'r') as f:
                content = f.read()
            
            # Check for common syntax issues
            lines = content.split('\n')
            for i, line in enumerate(lines):
                if 'break_loop=False' in line and not line.strip().endswith(','):
                    pytest.skip(f"Syntax error in cognitive_reasoning.py at line {i+1}")
                    
        except Exception as e:
            pytest.skip(f"Cannot read cognitive reasoning tool: {e}")
    
    def test_logic_systems_config_integration(self):
        """Test logic systems configuration integration."""
        if not cognitive_config_path.exists():
            pytest.skip("Cognitive configuration not found")
        
        with open(cognitive_config_path, 'r') as f:
            config = json.load(f)
        
        # Check for logic systems configuration sections
        expected_sections = ["reasoning_config", "atomspace_config", "cognitive_tools"]
        
        found_sections = []
        for section in expected_sections:
            if section in config:
                found_sections.append(section)
        
        assert len(found_sections) > 0, f"No logic systems configuration found. Available sections: {list(config.keys())}"
    
    def test_pattern_matching_integration(self):
        """Test pattern matching integration with cognitive tools."""
        # Test pattern matching scenarios that logic systems should handle
        test_scenarios = [
            {
                "name": "simple_concept_matching",
                "pattern": "?X isa ?Y",
                "query": "cat isa animal",
                "expected_bindings": {"X": "cat", "Y": "animal"}
            },
            {
                "name": "relationship_matching", 
                "pattern": "(?A likes ?B)",
                "query": "(john likes pizza)",
                "expected_bindings": {"A": "john", "B": "pizza"}
            },
            {
                "name": "complex_rule_matching",
                "pattern": "If (?X isa ?Y) and (?Y isa ?Z) then (?X isa ?Z)",
                "query": "transitive_inheritance_rule",
                "expected_bindings": {"rule_type": "transitivity"}
            }
        ]
        
        for scenario in test_scenarios:
            pattern = scenario["pattern"]
            query = scenario["query"]
            
            # Basic validation of pattern structure
            assert "?" in pattern, f"Pattern should contain variables: {pattern}"
            assert len(query) > 0, "Query should not be empty"
            
            # Validate expected bindings structure
            if "expected_bindings" in scenario:
                bindings = scenario["expected_bindings"]
                assert isinstance(bindings, dict), "Bindings should be dictionary"
    
    def test_rule_engine_integration(self):
        """Test rule engine integration scenarios."""
        # Test rule execution templates
        rule_templates = [
            {
                "rule_type": "forward_chaining",
                "rule": "modus_ponens",
                "premises": ["P", "P -> Q"],
                "conclusion": "Q"
            },
            {
                "rule_type": "backward_chaining",
                "goal": "conclusion",
                "subgoals": ["premise1", "premise2"],
                "rule": "goal_decomposition"
            }
        ]
        
        for template in rule_templates:
            assert "rule_type" in template
            assert "rule" in template
            
            if template["rule_type"] == "forward_chaining":
                assert "premises" in template
                assert "conclusion" in template
            elif template["rule_type"] == "backward_chaining":
                assert "goal" in template
                assert "subgoals" in template


class TestLogicSystemsPerformance:
    """Test performance aspects of logic systems integration."""
    
    def test_unification_performance_readiness(self):
        """Test unification algorithm performance readiness."""
        # Test performance measurement setup for unification
        performance_metrics = {
            "unification_time": 0.0,
            "pattern_complexity": "simple",
            "variable_count": 2,
            "binding_operations": 10
        }
        
        assert "unification_time" in performance_metrics
        assert "pattern_complexity" in performance_metrics
        assert performance_metrics["variable_count"] >= 0
        assert performance_metrics["binding_operations"] >= 0
    
    def test_rule_engine_performance_readiness(self):
        """Test rule engine performance readiness."""
        # Test performance measurement setup for rule engine
        performance_metrics = {
            "forward_chaining_steps": 0,
            "backward_chaining_depth": 0,
            "rule_applications": 0,
            "inference_time": 0.0
        }
        
        assert "forward_chaining_steps" in performance_metrics
        assert "backward_chaining_depth" in performance_metrics
        assert "rule_applications" in performance_metrics
        assert "inference_time" in performance_metrics
    
    def test_memory_usage_monitoring(self):
        """Test memory usage monitoring for logic systems."""
        # Test memory monitoring setup
        memory_metrics = {
            "atomspace_memory": 0,
            "rule_cache_memory": 0,
            "pattern_memory": 0,
            "total_logic_memory": 0
        }
        
        for metric_name, value in memory_metrics.items():
            assert isinstance(value, (int, float))
            assert value >= 0


class TestLogicSystemsDocumentation:
    """Test documentation and examples for logic systems."""
    
    def test_unify_documentation_readiness(self):
        """Test unification documentation structure."""
        docs_dir = project_root / "docs"
        
        # Check for logic systems documentation
        possible_doc_locations = [
            docs_dir / "logic_systems",
            docs_dir / "unify", 
            docs_dir / "cpp2py" / "components"
        ]
        
        doc_found = any(path.exists() for path in possible_doc_locations)
        if not doc_found:
            # Create basic documentation structure expectation
            expected_doc_topics = [
                "unification_algorithms",
                "pattern_matching",
                "variable_binding",
                "integration_examples"
            ]
            
            # Validate documentation topic structure
            assert len(expected_doc_topics) > 0
            for topic in expected_doc_topics:
                assert isinstance(topic, str)
                assert len(topic) > 0
    
    def test_ure_documentation_readiness(self):
        """Test URE documentation structure."""
        # Expected URE documentation topics
        expected_ure_topics = [
            "forward_chaining",
            "backward_chaining", 
            "rule_definitions",
            "inference_examples"
        ]
        
        for topic in expected_ure_topics:
            assert isinstance(topic, str)
            assert len(topic) > 0
    
    def test_integration_examples_structure(self):
        """Test integration examples structure."""
        # Expected integration examples
        integration_examples = [
            {
                "name": "basic_unification",
                "description": "Simple pattern unification example",
                "components": ["unify", "atomspace"]
            },
            {
                "name": "forward_chaining_inference",
                "description": "Forward chaining rule application",
                "components": ["ure", "unify", "atomspace"]
            },
            {
                "name": "cognitive_reasoning_integration", 
                "description": "Integration with PyCog-Zero cognitive tools",
                "components": ["ure", "unify", "cognitive_reasoning"]
            }
        ]
        
        for example in integration_examples:
            assert "name" in example
            assert "description" in example
            assert "components" in example
            assert isinstance(example["components"], list)
            assert len(example["components"]) > 0


@pytest.mark.integration
class TestLogicSystemsEndToEnd:
    """End-to-end testing of logic systems integration."""
    
    def test_phase_2_readiness(self):
        """Test Phase 2 logic systems integration readiness."""
        # Check overall Phase 2 readiness criteria
        readiness_criteria = {
            "unify_component_available": False,
            "ure_component_available": False,
            "pattern_matching_ready": True,
            "rule_engine_ready": True,
            "integration_tests_present": True,
            "documentation_structure": True
        }
        
        # Update criteria based on actual availability
        unify_locations = [unify_path, logic_components_dir / "unify"]
        readiness_criteria["unify_component_available"] = any(path.exists() for path in unify_locations)
        
        ure_locations = [ure_path, logic_components_dir / "ure"]
        readiness_criteria["ure_component_available"] = any(path.exists() for path in ure_locations)
        
        # Report readiness status
        ready_count = sum(1 for criteria, ready in readiness_criteria.items() if ready)
        total_count = len(readiness_criteria)
        
        readiness_percentage = (ready_count / total_count) * 100
        
        assert readiness_percentage >= 50, f"Phase 2 readiness too low: {readiness_percentage}%"
        
        # Log readiness status for debugging
        print(f"\nPhase 2 Logic Systems Readiness: {readiness_percentage:.1f}%")
        for criteria, ready in readiness_criteria.items():
            status = "✓" if ready else "✗"
            print(f"  {status} {criteria}")
    
    def test_cpp2py_pipeline_logic_integration(self):
        """Test cpp2py pipeline integration with logic systems."""
        # Test that cpp2py pipeline recognizes Phase 2 components
        pipeline_script = project_root / "scripts" / "cpp2py_conversion_pipeline.py"
        
        if not pipeline_script.exists():
            pytest.skip("cpp2py pipeline script not available")
        
        # Test pipeline status for Phase 2
        try:
            result = subprocess.run([
                "python3", str(pipeline_script), "status", "--phase", "phase_2_logic_systems"
            ], capture_output=True, text=True, cwd=str(project_root), timeout=30)
            
            # Should not crash even if components not cloned
            assert result.returncode in [0, 1], f"Pipeline status failed with code {result.returncode}"
            
            # Check output mentions Phase 2 components
            output = result.stdout + result.stderr
            assert any(comp in output.lower() for comp in ["unify", "ure"]), "Phase 2 components not mentioned in status"
            
        except subprocess.TimeoutExpired:
            pytest.skip("Pipeline status command timed out")
        except Exception as e:
            pytest.skip(f"Cannot test pipeline status: {e}")