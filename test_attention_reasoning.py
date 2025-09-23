#!/usr/bin/env python3
"""
Simple test script for attention-based reasoning examples.

This script demonstrates basic attention mechanisms without requiring
full Agent-Zero dependencies, suitable for quick testing and validation.
"""

import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def test_attention_configuration():
    """Test that attention configuration loads correctly."""
    import json
    
    print("Testing Attention Configuration...")
    
    try:
        with open('conf/config_cognitive.json', 'r') as f:
            config = json.load(f)
        
        # Check for attention config
        attention_config = config.get('attention_config', {})
        if not attention_config:
            print("❌ No attention_config found in cognitive configuration")
            return False
        
        # Validate key attention parameters
        required_sections = ['ecan_enabled', 'attention_mechanisms', 'ecan_config']
        missing_sections = [section for section in required_sections if section not in attention_config]
        
        if missing_sections:
            print(f"❌ Missing attention config sections: {missing_sections}")
            return False
        
        # Check specific attention mechanisms
        mechanisms = attention_config['attention_mechanisms']
        expected_mechanisms = ['multi_head_attention', 'self_attention', 'cross_attention']
        
        for mechanism in expected_mechanisms:
            if mechanism not in mechanisms:
                print(f"❌ Missing attention mechanism: {mechanism}")
                return False
            
            if not mechanisms[mechanism].get('enabled', False):
                print(f"⚠️  Attention mechanism {mechanism} is disabled")
        
        # Check ECAN configuration
        ecan_config = attention_config['ecan_config']
        ecan_params = ['sti_decay_factor', 'lti_decay_factor', 'sti_threshold', 'lti_threshold']
        
        for param in ecan_params:
            if param not in ecan_config:
                print(f"❌ Missing ECAN parameter: {param}")
                return False
        
        print("✅ Attention configuration validation passed")
        print(f"   - Multi-head attention heads: {mechanisms['multi_head_attention'].get('num_heads', 'N/A')}")
        print(f"   - ECAN STI decay factor: {ecan_config['sti_decay_factor']}")
        print(f"   - ECAN importance diffusion: {ecan_config.get('importance_diffusion', {}).get('enabled', False)}")
        
        return True
        
    except FileNotFoundError:
        print("❌ Configuration file conf/config_cognitive.json not found")
        return False
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in configuration file: {e}")
        return False
    except Exception as e:
        print(f"❌ Error loading attention configuration: {e}")
        return False

def test_documentation_structure():
    """Test that attention-based reasoning documentation exists and is well-formed."""
    
    print("\\nTesting Documentation Structure...")
    
    doc_file = 'docs/attention_based_reasoning.md'
    
    try:
        with open(doc_file, 'r') as f:
            content = f.read()
        
        # Check for key sections
        required_sections = [
            '# Attention-Based Reasoning in PyCog-Zero',
            '## Overview',
            '## Key Concepts',
            '### 1. Economic Cognitive Attention Networks (ECAN)',
            '### 2. Neural Attention Mechanisms',
            '## Configuration Reference',
            '## Basic Attention-Based Reasoning Examples',
            '### Example 1: Single-Concept Attention Focus',
            '### Example 2: Multi-Head Attention Analysis',
            '### Example 3: ECAN-Weighted Attention Reasoning',
            '## Advanced Attention Patterns',
            '## Attention Visualization Examples',
            '## Integration with Agent-Zero Framework',
            '## Performance Optimization and Best Practices',
            '## Testing Attention-Based Reasoning',
            '## Summary and Next Steps'
        ]
        
        missing_sections = []
        for section in required_sections:
            if section not in content:
                missing_sections.append(section)
        
        if missing_sections:
            print(f"❌ Missing documentation sections:")
            for section in missing_sections:
                print(f"   - {section}")
            return False
        
        # Check for code examples
        python_code_blocks = content.count('```python')
        json_code_blocks = content.count('```json')
        
        if python_code_blocks < 5:
            print(f"⚠️  Only {python_code_blocks} Python code examples found (expected at least 5)")
        
        if json_code_blocks < 1:
            print(f"⚠️  Only {json_code_blocks} JSON config examples found (expected at least 1)")
        
        # Calculate documentation metrics
        lines = content.count('\\n')
        examples = content.count('### Example')
        
        print("✅ Documentation structure validation passed")
        print(f"   - Document length: {lines} lines")
        print(f"   - Python code examples: {python_code_blocks}")
        print(f"   - JSON config examples: {json_code_blocks}")
        print(f"   - Numbered examples: {examples}")
        
        return True
        
    except FileNotFoundError:
        print(f"❌ Documentation file {doc_file} not found")
        return False
    except Exception as e:
        print(f"❌ Error reading documentation: {e}")
        return False

def test_neural_symbolic_bridge_availability():
    """Test if neural symbolic bridge components are available."""
    
    print("\\nTesting Neural Symbolic Bridge Availability...")
    
    try:
        # Test import without full initialization
        import importlib.util
        
        bridge_file = 'python/helpers/neural_symbolic_bridge.py'
        
        if not os.path.exists(bridge_file):
            print(f"❌ Neural symbolic bridge file not found: {bridge_file}")
            return False
        
        # Check for key classes in the file
        with open(bridge_file, 'r') as f:
            bridge_content = f.read()
        
        required_classes = [
            'class NeuralSymbolicBridge',
            'class CognitiveAttentionMechanism',
            'def forward('
        ]
        
        missing_classes = []
        for cls in required_classes:
            if cls not in bridge_content:
                missing_classes.append(cls)
        
        if missing_classes:
            print(f"❌ Missing neural symbolic bridge components:")
            for cls in missing_classes:
                print(f"   - {cls}")
            return False
        
        # Check for attention mechanism methods
        attention_methods = [
            'def forward(',
            'def compute_ecan_weights(',
            'MultiheadAttention'
        ]
        
        available_methods = []
        for method in attention_methods:
            if method in bridge_content:
                available_methods.append(method)
        
        print("✅ Neural symbolic bridge components available")
        print(f"   - Available attention methods: {len(available_methods)}/{len(attention_methods)}")
        
        # Check for graceful fallbacks
        fallback_checks = [
            'PYTORCH_AVAILABLE',
            'MockTensor',
            'MockModule'
        ]
        
        fallbacks_found = sum(1 for check in fallback_checks if check in bridge_content)
        print(f"   - Fallback mechanisms: {fallbacks_found}/{len(fallback_checks)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking neural symbolic bridge: {e}")
        return False

def test_cognitive_reasoning_tool_integration():
    """Test that cognitive reasoning tool has attention integration points."""
    
    print("\\nTesting Cognitive Reasoning Tool Integration...")
    
    try:
        tool_file = 'python/tools/cognitive_reasoning.py'
        
        if not os.path.exists(tool_file):
            print(f"❌ Cognitive reasoning tool not found: {tool_file}")
            return False
        
        with open(tool_file, 'r') as f:
            tool_content = f.read()
        
        # Check for attention-related integration points
        attention_features = [
            'attention_allocation',
            'cross_tool_sharing',
            '_load_cognitive_config',
            'analyze_patterns',
            'cross_reference'
        ]
        
        available_features = []
        for feature in attention_features:
            if feature in tool_content:
                available_features.append(feature)
        
        if len(available_features) < 3:  # Require at least basic integration
            print(f"❌ Insufficient attention integration in cognitive reasoning tool")
            print(f"   Found: {available_features}")
            return False
        
        # Check for configuration loading
        config_loading = [
            'config_cognitive.json',
            '_load_cognitive_config',
            'fallback'
        ]
        
        config_features = sum(1 for feature in config_loading if feature in tool_content)
        
        print("✅ Cognitive reasoning tool integration validated")
        print(f"   - Attention features: {len(available_features)}/{len(attention_features)}")
        print(f"   - Configuration loading: {config_features}/{len(config_loading)}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error checking cognitive reasoning tool: {e}")
        return False

def main():
    """Run all attention-based reasoning tests."""
    
    print("🧠 PyCog-Zero Attention-Based Reasoning Validation")
    print("=" * 55)
    
    tests = [
        test_attention_configuration,
        test_documentation_structure,
        test_neural_symbolic_bridge_availability,
        test_cognitive_reasoning_tool_integration
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            results.append(False)
    
    # Summary
    print("\\n" + "=" * 55)
    print("📊 Test Results Summary")
    print("=" * 55)
    
    passed = sum(results)
    total = len(results)
    
    test_names = [
        "Attention Configuration",
        "Documentation Structure", 
        "Neural Symbolic Bridge",
        "Cognitive Tool Integration"
    ]
    
    for i, (name, result) in enumerate(zip(test_names, results)):
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{name:<25} {status}")
    
    print("-" * 55)
    print(f"Overall Result: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 All attention-based reasoning components validated successfully!")
        print("   Ready for implementation and Agent-Zero integration.")
        return 0
    else:
        print("\\n⚠️  Some validation tests failed. Please review the issues above.")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)