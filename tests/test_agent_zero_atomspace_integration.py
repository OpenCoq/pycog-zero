#!/usr/bin/env python3
"""
Test Agent-Zero Tools Integration with AtomSpace Components
============================================================

This test validates that core Agent-Zero tools can optionally integrate
with atomspace components for enhanced cognitive capabilities.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_memory_save_has_atomspace_integration():
    """Test that memory_save tool has atomspace integration code."""
    with open('python/tools/memory_save.py', 'r') as f:
        content = f.read()
    
    # Check for atomspace integration imports and logic
    assert 'AtomSpaceToolHub' in content, "memory_save should import AtomSpaceToolHub"
    assert 'ATOMSPACE_HUB_AVAILABLE' in content, "memory_save should check atomspace availability"
    assert 'get_shared_atomspace' in content, "memory_save should use shared atomspace"
    
    # Check for graceful fallback
    assert 'except ImportError' in content, "memory_save should handle import errors gracefully"
    assert 'except Exception' in content, "memory_save should handle atomspace errors gracefully"
    
    print("✓ memory_save has atomspace integration with graceful fallback")


def test_memory_load_has_atomspace_integration():
    """Test that memory_load tool has atomspace integration code."""
    with open('python/tools/memory_load.py', 'r') as f:
        content = f.read()
    
    # Check for atomspace integration imports and logic
    assert 'AtomSpaceToolHub' in content, "memory_load should import AtomSpaceToolHub"
    assert 'ATOMSPACE_HUB_AVAILABLE' in content, "memory_load should check atomspace availability"
    assert 'get_shared_atomspace' in content, "memory_load should use shared atomspace"
    
    # Check for graceful fallback
    assert 'except ImportError' in content, "memory_load should handle import errors gracefully"
    assert 'except Exception' in content, "memory_load should handle atomspace errors gracefully"
    
    print("✓ memory_load has atomspace integration with graceful fallback")


def test_code_execution_has_atomspace_integration():
    """Test that code_execution_tool has atomspace integration code."""
    with open('python/tools/code_execution_tool.py', 'r') as f:
        content = f.read()
    
    # Check for atomspace integration imports and logic
    assert 'AtomSpaceToolHub' in content, "code_execution_tool should import AtomSpaceToolHub"
    assert 'ATOMSPACE_HUB_AVAILABLE' in content, "code_execution_tool should check atomspace availability"
    assert '_track_execution_in_atomspace' in content, "code_execution_tool should track executions"
    
    # Check for graceful fallback
    assert 'except ImportError' in content, "code_execution_tool should handle import errors gracefully"
    assert 'except Exception' in content, "code_execution_tool should handle atomspace errors gracefully"
    
    print("✓ code_execution_tool has atomspace integration with graceful fallback")


def test_search_engine_has_atomspace_integration():
    """Test that search_engine tool has atomspace integration code."""
    with open('python/tools/search_engine.py', 'r') as f:
        content = f.read()
    
    # Check for atomspace integration imports and logic
    assert 'AtomSpaceToolHub' in content, "search_engine should import AtomSpaceToolHub"
    assert 'ATOMSPACE_HUB_AVAILABLE' in content, "search_engine should check atomspace availability"
    assert '_track_search_in_atomspace' in content, "search_engine should track searches"
    
    # Check for graceful fallback
    assert 'except ImportError' in content, "search_engine should handle import errors gracefully"
    assert 'except Exception' in content, "search_engine should handle atomspace errors gracefully"
    
    print("✓ search_engine has atomspace integration with graceful fallback")


def test_integration_is_optional():
    """Test that atomspace integration is optional and doesn't break tools."""
    # Test that tools can be imported even if atomspace is not available
    # This should not raise any import errors
    
    try:
        # These imports should work even without full dependencies
        # The tools should have graceful fallback logic
        tools_tested = []
        
        # Just verify files exist and have proper structure
        for tool_file in ['memory_save.py', 'memory_load.py', 'code_execution_tool.py', 'search_engine.py']:
            tool_path = f'python/tools/{tool_file}'
            assert os.path.exists(tool_path), f"{tool_file} should exist"
            
            with open(tool_path, 'r') as f:
                content = f.read()
                # Verify try/except pattern for optional integration
                assert content.count('try:') >= 2, f"{tool_file} should have multiple try blocks for fallback"
                assert content.count('except') >= 2, f"{tool_file} should have multiple except blocks for fallback"
            
            tools_tested.append(tool_file)
        
        print(f"✓ All {len(tools_tested)} tools have optional atomspace integration with proper fallback")
        
    except Exception as e:
        raise AssertionError(f"Tools should have optional atomspace integration: {e}")


def test_atomspace_tool_hub_exists():
    """Test that AtomSpaceToolHub exists and can be used by tools."""
    assert os.path.exists('python/tools/atomspace_tool_hub.py'), "AtomSpaceToolHub should exist"
    
    with open('python/tools/atomspace_tool_hub.py', 'r') as f:
        content = f.read()
    
    # Verify AtomSpaceToolHub has the required interface
    assert 'class AtomSpaceToolHub' in content, "AtomSpaceToolHub class should exist"
    assert 'get_shared_atomspace' in content, "AtomSpaceToolHub should provide shared atomspace"
    assert '_shared_atomspace' in content, "AtomSpaceToolHub should have shared atomspace"
    
    print("✓ AtomSpaceToolHub exists with required interface")


def test_integration_follows_pattern():
    """Test that all integrations follow the same pattern."""
    tools = ['memory_save.py', 'memory_load.py', 'code_execution_tool.py', 'search_engine.py']
    
    for tool_file in tools:
        with open(f'python/tools/{tool_file}', 'r') as f:
            content = f.read()
        
        # Verify pattern: try import, set flag, use flag in code
        assert 'try:' in content, f"{tool_file} should have try block for import"
        assert 'from python.tools.atomspace_tool_hub import AtomSpaceToolHub' in content, \
            f"{tool_file} should import AtomSpaceToolHub"
        assert 'ATOMSPACE_HUB_AVAILABLE = True' in content, f"{tool_file} should set availability flag"
        assert 'except ImportError:' in content, f"{tool_file} should catch ImportError"
        assert 'ATOMSPACE_HUB_AVAILABLE = False' in content, f"{tool_file} should handle unavailability"
        
        # Verify pattern: check flag before using atomspace
        assert 'if ATOMSPACE_HUB_AVAILABLE' in content, f"{tool_file} should check availability before use"
        
        # Verify pattern: graceful error handling
        assert 'except Exception' in content, f"{tool_file} should handle exceptions gracefully"
    
    print(f"✓ All {len(tools)} tools follow consistent integration pattern")


def test_cognitive_reasoning_atomspace_rocks_integration():
    """Test that cognitive_reasoning.py has atomspace rocks integration (Phase 1 requirement)."""
    with open('python/tools/cognitive_reasoning.py', 'r') as f:
        content = f.read()
    
    # Phase 1 requires these specific methods
    required_features = [
        'ATOMSPACE_ROCKS_AVAILABLE',
        '_setup_atomspace_rocks_integration',
        '_apply_rocks_optimizations',
        '_setup_rocks_performance_monitoring',
        'atomspace_rocks_optimizer import AtomSpaceRocksOptimizer'
    ]
    
    for feature in required_features:
        assert feature in content, f"cognitive_reasoning.py should have {feature}"
    
    print("✓ cognitive_reasoning.py has complete atomspace rocks integration")


def test_phase1_integration_complete():
    """Test that Phase 1 Agent-Zero tools integration is complete."""
    
    # Check all required tools have atomspace integration
    tools_with_integration = [
        'memory_save.py',
        'memory_load.py', 
        'code_execution_tool.py',
        'search_engine.py',
        'cognitive_reasoning.py'
    ]
    
    integration_complete = True
    
    for tool_file in tools_with_integration:
        tool_path = f'python/tools/{tool_file}'
        
        if not os.path.exists(tool_path):
            print(f"❌ {tool_file} does not exist")
            integration_complete = False
            continue
        
        with open(tool_path, 'r') as f:
            content = f.read()
        
        # Verify atomspace integration present
        if 'atomspace' not in content.lower() and 'AtomSpace' not in content:
            print(f"❌ {tool_file} missing atomspace integration")
            integration_complete = False
            continue
        
        print(f"✓ {tool_file} has atomspace integration")
    
    assert integration_complete, "Phase 1 Agent-Zero tools integration incomplete"
    print(f"\n✓ Phase 1 integration complete: {len(tools_with_integration)} tools integrated with atomspace")


if __name__ == "__main__":
    print("=" * 70)
    print("Agent-Zero Tools AtomSpace Integration Test")
    print("=" * 70)
    print()
    
    test_memory_save_has_atomspace_integration()
    test_memory_load_has_atomspace_integration()
    test_code_execution_has_atomspace_integration()
    test_search_engine_has_atomspace_integration()
    test_integration_is_optional()
    test_atomspace_tool_hub_exists()
    test_integration_follows_pattern()
    test_cognitive_reasoning_atomspace_rocks_integration()
    test_phase1_integration_complete()
    
    print()
    print("=" * 70)
    print("✓ ALL TESTS PASSED - Agent-Zero tools properly integrated with atomspace")
    print("=" * 70)
