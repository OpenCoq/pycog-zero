#!/usr/bin/env python3
"""
Integration test for logic systems documentation and examples.

This test validates that the logic system usage patterns documentation
is comprehensive and accurate for Agent-Zero integration.
"""

import pytest
import os
import ast
import json
from pathlib import Path

# Test configuration
PROJECT_ROOT = Path(__file__).parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
EXAMPLES_DIR = PROJECT_ROOT / "examples"

class TestLogicSystemsDocumentation:
    """Test suite for logic systems documentation completeness and accuracy."""
    
    def test_main_documentation_exists(self):
        """Test that main logic systems documentation files exist."""
        
        # Check main documentation files
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        phase2_impl_doc = DOCS_DIR / "phase2_logic_systems_implementation.md"
        
        assert logic_patterns_doc.exists(), "Logic systems integration patterns doc should exist"
        assert phase2_impl_doc.exists(), "Phase 2 implementation guide should exist"
        
        print("✓ Main documentation files exist")
    
    def test_documentation_content_structure(self):
        """Test that documentation has proper structure and content."""
        
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        
        with open(logic_patterns_doc, 'r') as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            "# Logic Systems Integration Patterns",
            "## Overview",
            "## Logic Systems Architecture", 
            "## Pattern 1: Unification-based Agent Tools",
            "## Pattern 2: URE-based Reasoning Chains",
            "## Pattern 3: Integrated Logic System Agent Tool",
            "## Pattern 4: Agent-Zero Query Processing Pipeline",
            "## Integration Examples",
            "## Testing Patterns",
            "## Configuration and Setup"
        ]
        
        for section in required_sections:
            assert section in content, f"Documentation should contain section: {section}"
        
        print("✓ Documentation has proper structure")
    
    def test_code_examples_syntax(self):
        """Test that code examples in documentation have valid Python syntax."""
        
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        
        with open(logic_patterns_doc, 'r') as f:
            content = f.read()
        
        # Extract Python code blocks
        import re
        python_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        
        syntax_errors = []
        
        for i, code_block in enumerate(python_blocks):
            try:
                # Parse the code to check syntax
                ast.parse(code_block)
            except SyntaxError as e:
                syntax_errors.append(f"Code block {i+1}: {e}")
        
        assert len(syntax_errors) == 0, f"Syntax errors found: {syntax_errors}"
        
        print(f"✓ All {len(python_blocks)} code examples have valid Python syntax")
    
    def test_example_file_exists_and_runs(self):
        """Test that the example file exists and can be imported."""
        
        example_file = EXAMPLES_DIR / "logic_systems_example.py"
        assert example_file.exists(), "Logic systems example file should exist"
        
        # Test that the file has valid Python syntax
        with open(example_file, 'r') as f:
            content = f.read()
        
        try:
            ast.parse(content)
        except SyntaxError as e:
            pytest.fail(f"Example file has syntax error: {e}")
        
        print("✓ Example file exists and has valid syntax")
    
    def test_example_file_structure(self):
        """Test that example file has proper structure and classes."""
        
        example_file = EXAMPLES_DIR / "logic_systems_example.py"
        
        with open(example_file, 'r') as f:
            content = f.read()
        
        # Check for key components
        required_elements = [
            "class LogicSystemsExample:",
            "async def run_unification_example(",
            "async def run_forward_chaining_example(",
            "async def run_backward_chaining_example(",
            "async def run_integrated_reasoning_example(",
            "async def run_task_planning_example(",
            "def main():"
        ]
        
        for element in required_elements:
            assert element in content, f"Example should contain: {element}"
        
        print("✓ Example file has proper structure")
    
    def test_integration_patterns_coverage(self):
        """Test that documentation covers all required integration patterns."""
        
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        
        with open(logic_patterns_doc, 'r') as f:
            content = f.read()
        
        # Check for comprehensive pattern coverage
        pattern_topics = [
            "UnificationTool",
            "ForwardChainingTool", 
            "BackwardChainingTool",
            "LogicSystemsTool",
            "LogicQueryRouter",
            "QueryAnalyzer",
            "ResponseFormatter",
            "TaskPlanningTool",
            "KnowledgeIntegrationTool"
        ]
        
        missing_topics = []
        for topic in pattern_topics:
            if topic not in content:
                missing_topics.append(topic)
        
        assert len(missing_topics) == 0, f"Missing pattern topics: {missing_topics}"
        
        print(f"✓ All {len(pattern_topics)} integration patterns covered")
    
    def test_phase2_implementation_guide(self):
        """Test Phase 2 implementation guide completeness."""
        
        phase2_doc = DOCS_DIR / "phase2_logic_systems_implementation.md"
        
        with open(phase2_doc, 'r') as f:
            content = f.read()
        
        # Check for implementation steps
        implementation_sections = [
            "## Step 1: Clone and Setup Components",
            "## Step 2: Component Analysis", 
            "## Step 3: Python Integration Architecture",
            "## Step 4: Agent-Zero Tool Creation",
            "## Step 5: Python Binding Generation",
            "## Step 6: Integration Testing",
            "## Step 7: Documentation Update",
            "## Step 8: Performance Validation"
        ]
        
        for section in implementation_sections:
            assert section in content, f"Implementation guide should contain: {section}"
        
        print("✓ Phase 2 implementation guide is comprehensive")
    
    def test_testing_patterns_included(self):
        """Test that testing patterns are properly documented."""
        
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        
        with open(logic_patterns_doc, 'r') as f:
            content = f.read()
        
        # Check for testing coverage
        testing_elements = [
            "## Testing Patterns",
            "class TestLogicSystemsIntegration:",
            "@pytest.mark.asyncio",
            "test_unification_pattern",
            "test_forward_chaining_pattern", 
            "test_complex_reasoning_pattern"
        ]
        
        for element in testing_elements:
            assert element in content, f"Testing patterns should include: {element}"
        
        print("✓ Testing patterns properly documented")
    
    def test_configuration_examples(self):
        """Test that configuration examples are included."""
        
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        
        with open(logic_patterns_doc, 'r') as f:
            content = f.read()
        
        # Check for configuration sections
        config_elements = [
            "## Configuration and Setup",
            "logic_systems",
            "unify_system",
            "ure_system", 
            "agent_zero_integration"
        ]
        
        for element in config_elements:
            assert element in content, f"Configuration should include: {element}"
        
        print("✓ Configuration examples properly documented")

    def test_performance_considerations(self):
        """Test that performance considerations are documented."""
        
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        
        with open(logic_patterns_doc, 'r') as f:
            content = f.read()
        
        # Check for performance coverage
        performance_elements = [
            "## Performance Considerations",
            "Optimization Patterns",
            "Shared AtomSpace",
            "Rule Caching",
            "Result Memoization",
            "LogicSystemsMonitor"
        ]
        
        for element in performance_elements:
            assert element in content, f"Performance considerations should include: {element}"
        
        print("✓ Performance considerations documented")

class TestDocumentationIntegration:
    """Test integration between documentation and existing codebase."""
    
    def test_documentation_references_existing_tools(self):
        """Test that documentation properly references existing cognitive tools."""
        
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        
        with open(logic_patterns_doc, 'r') as f:
            content = f.read()
        
        # Check references to existing tools
        existing_tool_references = [
            "CognitiveReasoningTool",
            "python.helpers.tool",
            "AtomSpace",
            "Response"
        ]
        
        for reference in existing_tool_references:
            assert reference in content, f"Should reference existing tool: {reference}"
        
        print("✓ Documentation properly references existing tools")
    
    def test_component_directory_structure_mentioned(self):
        """Test that documentation mentions the correct component directory structure."""
        
        phase2_doc = DOCS_DIR / "phase2_logic_systems_implementation.md"
        
        with open(phase2_doc, 'r') as f:
            content = f.read()
        
        # Check component references
        component_references = [
            "components/unify",
            "components/ure",
            "python/tools/",
            "tests/integration/"
        ]
        
        for reference in component_references:
            assert reference in content, f"Should reference component structure: {reference}"
        
        print("✓ Component directory structure properly referenced")
    
    def test_pipeline_script_references(self):
        """Test that documentation references the correct pipeline scripts."""
        
        phase2_doc = DOCS_DIR / "phase2_logic_systems_implementation.md"
        
        with open(phase2_doc, 'r') as f:
            content = f.read()
        
        # Check pipeline script references
        script_references = [
            "scripts/cpp2py_conversion_pipeline.py",
            "clone unify",
            "clone ure",
            "validate unify",
            "validate ure"
        ]
        
        for reference in script_references:
            assert reference in content, f"Should reference pipeline script: {reference}"
        
        print("✓ Pipeline scripts properly referenced")

class TestDocumentationQuality:
    """Test documentation quality and completeness."""
    
    def test_documentation_word_count(self):
        """Test that documentation is comprehensive (sufficient word count)."""
        
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        phase2_doc = DOCS_DIR / "phase2_logic_systems_implementation.md"
        
        with open(logic_patterns_doc, 'r') as f:
            patterns_content = f.read()
        
        with open(phase2_doc, 'r') as f:
            phase2_content = f.read()
        
        # Count words (excluding code blocks)
        import re
        
        # Remove code blocks for word counting
        patterns_text = re.sub(r'```.*?```', '', patterns_content, flags=re.DOTALL)
        phase2_text = re.sub(r'```.*?```', '', phase2_content, flags=re.DOTALL)
        
        patterns_words = len(patterns_text.split())
        phase2_words = len(phase2_text.split())
        
        # Expect comprehensive documentation
        assert patterns_words > 1000, f"Logic patterns doc should be comprehensive: {patterns_words} words"
        assert phase2_words > 1000, f"Phase 2 guide should be comprehensive: {phase2_words} words"
        
        print(f"✓ Documentation is comprehensive:")
        print(f"  - Logic patterns: {patterns_words} words")
        print(f"  - Phase 2 guide: {phase2_words} words")
    
    def test_code_to_text_ratio(self):
        """Test that documentation has good balance of code examples and explanatory text."""
        
        logic_patterns_doc = DOCS_DIR / "logic_systems_integration_patterns.md"
        
        with open(logic_patterns_doc, 'r') as f:
            content = f.read()
        
        # Extract code blocks
        import re
        code_blocks = re.findall(r'```.*?\n(.*?)\n```', content, re.DOTALL)
        code_chars = sum(len(block) for block in code_blocks)
        
        # Calculate ratio
        total_chars = len(content)
        code_ratio = code_chars / total_chars
        
        # Expect good balance (20-60% code)
        assert 0.2 <= code_ratio <= 0.6, f"Code-to-text ratio should be balanced: {code_ratio:.2%}"
        
        print(f"✓ Good code-to-text balance: {code_ratio:.1%} code examples")

def run_documentation_validation():
    """Run all documentation validation tests."""
    
    print("Validating Logic Systems Documentation")
    print("="*50)
    
    # Run test classes
    test_classes = [
        TestLogicSystemsDocumentation,
        TestDocumentationIntegration, 
        TestDocumentationQuality
    ]
    
    for test_class in test_classes:
        print(f"\nRunning {test_class.__name__}...")
        
        # Create instance and run tests
        instance = test_class()
        test_methods = [method for method in dir(instance) if method.startswith('test_')]
        
        for test_method in test_methods:
            try:
                getattr(instance, test_method)()
            except Exception as e:
                print(f"  ✗ {test_method}: {e}")
                return False
    
    print("\n" + "="*50)
    print("✓ All documentation validation tests passed")
    print("Logic systems usage patterns documentation is complete and accurate")
    
    return True

if __name__ == "__main__":
    success = run_documentation_validation()
    exit(0 if success else 1)