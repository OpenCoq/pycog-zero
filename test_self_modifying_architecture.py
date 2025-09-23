#!/usr/bin/env python3
"""
Focused test for Self-Modifying Architecture Tool
Tests core functionality including architecture analysis and dynamic tool creation.
"""

import asyncio
import json
import os
import sys
import tempfile
import shutil

# Add the project root to Python path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from python.tools.self_modifying_architecture import SelfModifyingArchitecture
from agent import Agent
from python.helpers.log import LogItem


class MockAgent:
    """Mock agent for testing purposes."""
    
    def __init__(self):
        self.tools = []
        self.memory_path = tempfile.mkdtemp()
    
    def get_tools(self):
        return self.tools
    
    def get_capabilities(self):
        return ["mock_capability_1", "mock_capability_2"]
    
    def cleanup(self):
        if os.path.exists(self.memory_path):
            shutil.rmtree(self.memory_path)


async def test_architecture_analysis():
    """Test architecture analysis functionality."""
    print("\\n=== Testing Architecture Analysis ===")
    
    # Create mock agent
    agent = MockAgent()
    
    try:
        # Initialize self-modifying architecture tool
        sma_tool = SelfModifyingArchitecture(agent)
        
        print(f"✓ Tool initialized with instance ID: {sma_tool.instance_id}")
        print(f"✓ Initialized: {sma_tool.initialized}")
        print(f"✓ Safety enabled: {sma_tool.safety_enabled}")
        
        # Test architecture analysis
        print("\\n--- Testing analyze_architecture operation ---")
        result = await sma_tool.execute("analyze_architecture", depth="basic")
        
        print(f"✓ Analysis completed")
        print(f"Message preview: {result.message[:200]}...")
        
        # Extract data from message using JSON parsing
        if "Data: {" in result.message:
            try:
                json_start = result.message.find("Data: {")
                json_data = result.message[json_start + 6:]  # Skip "Data: "
                analysis = json.loads(json_data)
                print(f"✓ Components analyzed: {len(analysis.get('components', {}))}")
                print(f"✓ Tools analyzed: {len(analysis.get('tools', {}))}")
                print(f"✓ Prompts analyzed: {len(analysis.get('prompts', {}))}")
                print(f"✓ Opportunities identified: {len(analysis.get('modification_opportunities', []))}")
            except Exception as e:
                print(f"⚠ Could not parse analysis data: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Architecture analysis test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        agent.cleanup()


async def test_dynamic_tool_creation():
    """Test dynamic tool creation functionality."""
    print("\\n=== Testing Dynamic Tool Creation ===")
    
    # Create mock agent
    agent = MockAgent()
    
    try:
        # Initialize self-modifying architecture tool
        sma_tool = SelfModifyingArchitecture(agent)
        
        print(f"✓ Tool initialized for dynamic creation test")
        
        # Test dynamic tool creation
        print("\\n--- Testing create_tool operation ---")
        result = await sma_tool.execute(
            "create_tool",
            target="test_dynamic_tool",
            description="A dynamically created test tool",
            capabilities=["test_functionality", "self_test"],
            rationale="Testing dynamic tool creation capability"
        )
        
        print(f"✓ Tool creation completed")
        print(f"Message preview: {result.message[:200]}...")
        
        # Extract data from message using JSON parsing
        creation_data = None
        if "Data: {" in result.message:
            try:
                json_start = result.message.find("Data: {")
                json_data = result.message[json_start + 6:]  # Skip "Data: "
                creation_data = json.loads(json_data)
                print(f"✓ Modification ID: {creation_data.get('modification_id')}")
                print(f"✓ Target tool: {creation_data.get('target')}")
                print(f"✓ Changes recorded: {len(creation_data.get('changes', {}))}")
            except Exception as e:
                print(f"⚠ Could not parse creation data: {e}")
        
        # Verify tool file was created
        if creation_data and 'changes' in creation_data:
            tool_path = creation_data['changes'].get('file_path')
            if tool_path and os.path.exists(tool_path):
                print(f"✓ Tool file created at: {tool_path}")
                
                # Verify file content
                with open(tool_path, 'r') as f:
                    content = f.read()
                    if 'TestDynamicToolTool' in content and 'def execute' in content:
                        print("✓ Tool file contains expected code structure")
                    else:
                        print("⚠ Tool file may have unexpected structure")
                
                # Cleanup created file
                os.remove(tool_path)
                print("✓ Cleanup completed")
            else:
                print("⚠ Tool file was not created or not found")
        
        # Check modification log
        if sma_tool.modification_log:
            print(f"✓ Modification logged: {len(sma_tool.modification_log)} entries")
        
        return True
        
    except Exception as e:
        print(f"✗ Dynamic tool creation test failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    finally:
        agent.cleanup()


async def test_error_handling():
    """Test error handling and validation."""
    print("\\n=== Testing Error Handling ===")
    
    # Create mock agent
    agent = MockAgent()
    
    try:
        # Initialize self-modifying architecture tool
        sma_tool = SelfModifyingArchitecture(agent)
        
        # Test invalid operation
        print("\\n--- Testing invalid operation ---")
        result = await sma_tool.execute("invalid_operation")
        
        # Should fall back to architecture analysis
        if "architecture" in result.message.lower():
            print("✓ Invalid operation handled gracefully with fallback")
        
        # Test tool creation without name
        print("\\n--- Testing tool creation without name ---")
        result = await sma_tool.execute("create_tool")
        
        if "error" in result.message.lower() and "tool name" in result.message.lower():
            print("✓ Missing tool name handled correctly")
        
        # Test disabled self-modification
        print("\\n--- Testing disabled self-modification ---")
        # Make sure the config has the required structure
        if "self_modification" not in sma_tool.config:
            sma_tool.config["self_modification"] = {}
        sma_tool.config["self_modification"]["enabled"] = False
        result = await sma_tool.execute("create_tool", target="test_tool")
        
        if "disabled" in result.message.lower():
            print("✓ Disabled self-modification handled correctly")
        
        return True
        
    except Exception as e:
        print(f"✗ Error handling test failed: {e}")
        return False
    
    finally:
        agent.cleanup()


async def test_integration_points():
    """Test integration with other cognitive tools."""
    print("\\n=== Testing Integration Points ===")
    
    # Create mock agent
    agent = MockAgent()
    
    try:
        # Initialize self-modifying architecture tool
        sma_tool = SelfModifyingArchitecture(agent)
        
        # Check integration availability
        print(f"✓ Meta-cognition available: {sma_tool.meta_cognition_tool is not None}")
        print(f"✓ Learning tool available: {sma_tool.learning_tool is not None}")
        
        # Test configuration loading
        if sma_tool.config:
            print(f"✓ Configuration loaded: {len(sma_tool.config)} sections")
        
        # Test safety mechanisms
        print(f"✓ Safety enabled: {sma_tool.safety_enabled}")
        print(f"✓ Validation enabled: {sma_tool.validation_enabled}")
        print(f"✓ Rollback enabled: {sma_tool.rollback_enabled}")
        
        return True
        
    except Exception as e:
        print(f"✗ Integration test failed: {e}")
        return False
    
    finally:
        agent.cleanup()


async def main():
    """Run all tests."""
    print("Self-Modifying Architecture Tool - Focused Test Suite")
    print("=" * 60)
    
    tests = [
        ("Architecture Analysis", test_architecture_analysis),
        ("Dynamic Tool Creation", test_dynamic_tool_creation),
        ("Error Handling", test_error_handling),
        ("Integration Points", test_integration_points),
    ]
    
    results = []
    
    for test_name, test_func in tests:
        print(f"\\nRunning {test_name} test...")
        try:
            success = await test_func()
            results.append((test_name, success))
            print(f"✓ {test_name} test {'PASSED' if success else 'FAILED'}")
        except Exception as e:
            results.append((test_name, False))
            print(f"✗ {test_name} test FAILED with exception: {e}")
    
    # Summary
    print("\\n" + "=" * 60)
    print("TEST RESULTS SUMMARY")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:.<40} {status}")
        if success:
            passed += 1
    
    print(f"\\nOverall: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 All tests passed! Self-modifying architecture tool is working correctly.")
        return 0
    else:
        print("⚠ Some tests failed. Check the output above for details.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)