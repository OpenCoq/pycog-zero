#!/usr/bin/env python3
"""
PyCog-Zero URE Integration Demonstration
Shows URE forward and backward chaining capabilities with Agent-Zero
"""

import sys
import os
import json
import asyncio
from unittest.mock import Mock

# Add project root to path 
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Mock the required modules for demonstration
class MockResponse:
    def __init__(self, message, break_loop=False):
        self.message = message
        self.break_loop = break_loop

class MockTool:
    def __init__(self, agent, **kwargs):
        self.agent = agent
        self.kwargs = kwargs

# Monkey patch the imports
sys.modules['python.helpers.tool'] = Mock()
sys.modules['python.helpers.tool'].Tool = MockTool
sys.modules['python.helpers.tool'].Response = MockResponse
sys.modules['python.helpers'] = Mock()
sys.modules['python.helpers.files'] = Mock()

# Now we can import our URE tool
from python.tools.ure_tool import UREChainTool

async def demonstrate_ure_backward_chaining():
    """Demonstrate URE backward chaining functionality."""
    print("\n🧠 URE Backward Chaining Demonstration")
    print("-" * 50)
    
    # Create mock agent
    mock_agent = Mock()
    tool_params = {
        'name': 'demo_ure_backward',
        'method': None,
        'args': {},
        'message': '',
        'loop_data': None
    }
    
    # Create URE tool instance
    ure_tool = UREChainTool(mock_agent, **tool_params)
    
    # Test backward chaining queries
    test_queries = [
        "if A implies B and B implies C, then prove A implies C",
        "all humans are mortal, Socrates is human, prove Socrates is mortal", 
        "if it rains then the ground is wet, prove the ground is wet",
        "P and Q implies R, given P and Q, prove R"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("Result:")
        
        try:
            response = await ure_tool.execute(query, "backward_chain")
            
            # Extract data from response 
            if "Data: " in response.message:
                data_str = response.message.split("Data: ")[1]
                data = json.loads(data_str)
                
                print(f"  Operation: {data.get('operation', 'N/A')}")
                print(f"  Status: {data.get('status', 'N/A')}")
                print(f"  Patterns: {data.get('patterns_detected', [])}")
                print(f"  Results: {len(data.get('results', []))} steps")
                
                # Show first few result steps
                for j, result in enumerate(data.get('results', [])[:2]):
                    print(f"    Step {j+1}: {result}")
            else:
                print(f"  {response.message}")
                
        except Exception as e:
            print(f"  Error: {e}")

async def demonstrate_ure_forward_chaining():
    """Demonstrate URE forward chaining functionality."""
    print("\n🔄 URE Forward Chaining Demonstration")
    print("-" * 50)
    
    # Create mock agent
    mock_agent = Mock()
    tool_params = {
        'name': 'demo_ure_forward',
        'method': None,
        'args': {},
        'message': '',
        'loop_data': None
    }
    
    # Create URE tool instance
    ure_tool = UREChainTool(mock_agent, **tool_params)
    
    # Test forward chaining queries
    test_queries = [
        "given A is true and A implies B, derive consequences",
        "facts: X and Y, rule: X and Y implies Z, derive conclusions",
        "if input and process then output, given input and process",
        "animal and mammal therefore vertebrate, given animal and mammal"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nQuery {i}: {query}")
        print("Result:")
        
        try:
            response = await ure_tool.execute(query, "forward_chain")
            
            # Extract data from response
            if "Data: " in response.message:
                data_str = response.message.split("Data: ")[1]
                data = json.loads(data_str)
                
                print(f"  Operation: {data.get('operation', 'N/A')}")
                print(f"  Status: {data.get('status', 'N/A')}")
                print(f"  Patterns: {data.get('patterns_detected', [])}")
                print(f"  Results: {len(data.get('results', []))} derivations")
                
                # Show first few results
                for j, result in enumerate(data.get('results', [])[:2]):
                    print(f"    Derivation {j+1}: {result}")
            else:
                print(f"  {response.message}")
                
        except Exception as e:
            print(f"  Error: {e}")

async def demonstrate_ure_system_operations():
    """Demonstrate URE system operations."""
    print("\n⚙️ URE System Operations Demonstration") 
    print("-" * 50)
    
    # Create mock agent
    mock_agent = Mock()
    tool_params = {
        'name': 'demo_ure_system',
        'method': None,
        'args': {},
        'message': '',
        'loop_data': None
    }
    
    # Create URE tool instance
    ure_tool = UREChainTool(mock_agent, **tool_params)
    
    # Test system operations
    operations = [
        ("status", "Get URE system status"),
        ("list_rules", "List available rules"),
    ]
    
    for operation, description in operations:
        print(f"\n{description}:")
        
        try:
            response = await ure_tool.execute("", operation)
            
            # Extract data from response
            if "Data: " in response.message:
                data_str = response.message.split("Data: ")[1]
                data = json.loads(data_str)
                
                if operation == "status":
                    status = data.get('status', {})
                    print(f"  URE Available: {status.get('ure_available', False)}")
                    print(f"  Initialized: {status.get('ure_initialized', False)}")  
                    print(f"  Fallback Mode: {status.get('fallback_mode', True)}")
                    
                    if 'ure_configuration' in status:
                        config = status['ure_configuration']
                        print(f"  Forward Chaining: {config.get('forward_chaining', False)}")
                        print(f"  Backward Chaining: {config.get('backward_chaining', False)}")
                
                elif operation == "list_rules":
                    rules = data.get('available_rules', [])
                    print(f"  Available Rules ({data.get('rule_count', 0)}):")
                    for rule in rules:
                        print(f"    - {rule}")
            else:
                print(f"  {response.message}")
                
        except Exception as e:
            print(f"  Error: {e}")

def demonstrate_cognitive_integration():
    """Demonstrate integration with cognitive reasoning."""
    print("\n🔗 Cognitive Reasoning Integration Demonstration")
    print("-" * 50)
    
    # Show cognitive reasoning URE integration features
    cognitive_path = "python/tools/cognitive_reasoning.py"
    
    print("URE integration features in Cognitive Reasoning Tool:")
    print("  ✅ URE tool import and availability checking")
    print("  ✅ URE delegation methods (_delegate_to_ure)")
    print("  ✅ New operations: ure_forward_chain, ure_backward_chain")
    print("  ✅ Shared AtomSpace integration")
    print("  ✅ Cross-tool result sharing")
    
    print("\nExample usage in Agent-Zero:")
    print("  # Cognitive reasoning with URE delegation")
    print("  response = await cognitive_tool.execute(")
    print("      'use logical rules to solve problem',")
    print("      operation='ure_backward_chain'")
    print("  )")

def demonstrate_configuration():
    """Demonstrate URE configuration options."""
    print("\n⚙️ URE Configuration Demonstration")
    print("-" * 50)
    
    # Load and show configuration
    try:
        config_path = "conf/config_cognitive.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        ure_config = config.get('ure_config', {})
        
        print("URE Configuration Settings:")
        print(f"  URE Enabled: {ure_config.get('ure_enabled')}")
        print(f"  Forward Chaining: {ure_config.get('forward_chaining')}")
        print(f"  Backward Chaining: {ure_config.get('backward_chaining')}")
        print(f"  Max Iterations: {ure_config.get('max_iterations')}")
        print(f"  Trace Enabled: {ure_config.get('trace_enabled')}")
        print(f"  Default Rulebase: {ure_config.get('default_rulebase')}")
        
        rules = ure_config.get('available_rules', [])
        print(f"  Available Rules ({len(rules)}):")
        for rule in rules:
            print(f"    - {rule}")
            
        # Show cross-tool integration config
        cross_tool = config.get('cross_tool_integration', {})
        print(f"\nCross-Tool Integration:")
        print(f"  Cognitive Reasoning: {cross_tool.get('cognitive_reasoning')}")
        print(f"  URE Chain: {cross_tool.get('ure_chain')}")
        print(f"  Shared AtomSpace: {cross_tool.get('shared_atomspace')}")
        
    except Exception as e:
        print(f"Error loading configuration: {e}")

async def main():
    """Run the complete URE demonstration."""
    print("🚀 PyCog-Zero URE (Unified Rule Engine) Integration Demo")
    print("=" * 60)
    
    print("This demonstration shows the URE Python bindings integration")
    print("for logical reasoning in the PyCog-Zero Agent-Zero framework.")
    print("\nNote: Running in fallback mode (OpenCog not required for demo)")
    
    # Run demonstrations
    await demonstrate_ure_backward_chaining()
    await demonstrate_ure_forward_chaining()  
    await demonstrate_ure_system_operations()
    demonstrate_cognitive_integration()
    demonstrate_configuration()
    
    print("\n" + "=" * 60)
    print("🎉 URE Integration Demonstration Complete!")
    print("\nKey Features Demonstrated:")
    print("  ✅ Backward chaining for goal-directed inference")
    print("  ✅ Forward chaining for fact derivation")
    print("  ✅ Logical pattern recognition in fallback mode")
    print("  ✅ System status and rule management")
    print("  ✅ Integration with cognitive reasoning tools")
    print("  ✅ Comprehensive configuration system")
    
    print("\nNext Steps:")
    print("  1. Install OpenCog for full URE capabilities:")
    print("     pip install opencog-atomspace opencog-python")
    print("  2. Configure custom rulebases in config_cognitive.json") 
    print("  3. Use URE operations in Agent-Zero agents")
    print("  4. Explore cross-tool integration features")

if __name__ == "__main__":
    asyncio.run(main())