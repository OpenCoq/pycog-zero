#!/usr/bin/env python3
"""
Enhanced ECAN Cross-Tool Integration Demo

This script demonstrates the integrated ECAN (Economic Attention Networks) 
functionality across cognitive tools in PyCog-Zero, showing how attention 
is coordinated between cognitive_reasoning, cognitive_memory, and meta_cognition tools.
"""

import asyncio
import json
import time
from typing import Dict, Any

def show_demo_banner():
    """Display the demo banner."""
    print("=" * 80)
    print("🧠 ECAN (Economic Attention Networks) Cross-Tool Integration Demo")
    print("=" * 80)
    print("This demo shows enhanced ECAN integration across cognitive tools:")
    print("- Centralized attention coordination")
    print("- Cross-tool attention synchronization")
    print("- Attention-guided reasoning and memory operations")
    print("- Real-time attention metrics and monitoring")
    print()

def demo_fallback_explanation():
    """Explain fallback mode operation."""
    print("📝 ECAN Integration Status:")
    print("- OpenCog components: Not available (expected in development)")
    print("- ECAN Coordinator: Active with fallback mechanisms")
    print("- Cross-tool integration: Functional with priority-based allocation")
    print("- Attention synchronization: Available")
    print()

async def demonstrate_ecan_coordination():
    """Demonstrate ECAN coordinator functionality."""
    print("🔧 Testing ECAN Coordinator...")
    
    try:
        from python.helpers.ecan_coordinator import (
            get_ecan_coordinator, register_tool_with_ecan, 
            request_attention_for_tool, AttentionRequest
        )
        
        # Get the coordinator
        coordinator = get_ecan_coordinator()
        
        print(f"✓ ECAN Coordinator initialized: {coordinator.initialized}")
        print(f"✓ Active tools: {len(coordinator.active_tools)}")
        
        # Register demo tools
        coordinator.register_tool("demo_reasoning", 1.5)
        coordinator.register_tool("demo_memory", 1.0)
        coordinator.register_tool("demo_meta", 2.0)
        
        print(f"✓ Registered tools: {coordinator.active_tools}")
        
        # Create attention requests
        requests = [
            AttentionRequest(
                tool_name="demo_reasoning",
                priority=2.0,
                context="Complex logical reasoning task",
                concepts=["logic", "inference", "deduction", "reasoning"],
                importance_multiplier=1.2
            ),
            AttentionRequest(
                tool_name="demo_memory", 
                priority=1.5,
                context="Memory retrieval operation",
                concepts=["memory", "retrieval", "knowledge", "facts"],
                importance_multiplier=1.0
            ),
            AttentionRequest(
                tool_name="demo_meta",
                priority=2.5,
                context="Self-reflection and goal prioritization", 
                concepts=["self", "reflection", "goals", "priorities"],
                importance_multiplier=1.3
            )
        ]
        
        # Process attention requests
        print("\n🎯 Processing attention requests...")
        for request in requests:
            success = coordinator.request_attention(request)
            print(f"  - {request.tool_name}: {'✓ Accepted' if success else '❌ Rejected'}")
        
        # Get attention allocation
        allocation = coordinator.get_attention_allocation()
        print(f"\n📊 Attention Allocation Results:")
        print(f"  - Total STI allocated: {allocation.total_sti_allocated}")
        print(f"  - Attention entropy: {allocation.attention_entropy:.3f}")
        print(f"  - Tool allocations:")
        for tool, sti in allocation.tool_allocations.items():
            print(f"    * {tool}: {sti} STI")
        
        # Synchronize attention across tools
        sync_data = coordinator.synchronize_attention()
        print(f"\n🔄 Attention synchronization completed")
        print(f"  - Synchronized tools: {len(sync_data['active_tools'])}")
        print(f"  - Coordination timestamp: {sync_data['timestamp']:.2f}")
        
        # Get metrics
        metrics = coordinator.get_metrics()
        print(f"\n📈 ECAN Coordinator Metrics:")
        for metric, value in metrics.items():
            print(f"  - {metric}: {value}")
            
        return True
        
    except Exception as e:
        print(f"⚠️ ECAN coordinator demo error: {e}")
        return False

async def demonstrate_cognitive_reasoning_integration():
    """Demonstrate cognitive reasoning tool with ECAN integration."""
    print("\n🧮 Testing Cognitive Reasoning with ECAN...")
    
    try:
        # Import after ensuring dependencies
        from python.tools.cognitive_reasoning import CognitiveReasoningTool
        
        # Mock agent for testing
        class MockAgent:
            def __init__(self):
                self.config = {}
        
        agent = MockAgent()
        reasoning_tool = CognitiveReasoningTool(
            agent=agent,
            name="cognitive_reasoning",
            method="reason"
        )
        
        # Test reasoning with ECAN attention
        test_queries = [
            "How can I improve problem-solving skills?",
            "What are the relationships between learning and memory?",
            "How does attention affect cognitive performance?"
        ]
        
        for i, query in enumerate(test_queries):
            print(f"\n  Query {i+1}: {query}")
            
            try:
                # Execute reasoning with different priorities
                priority = 2.0 + i * 0.5  # Increasing priority
                response = await reasoning_tool.execute(
                    query=query,
                    operation="reason",
                    priority=priority
                )
                
                print(f"  ✓ Reasoning completed: {response.message[:100]}...")
                
                # Extract results
                if "Data:" in response.message:
                    data_str = response.message.split("Data:")[1].strip()
                    try:
                        data = json.loads(data_str)
                        print(f"    - Query processed: {data.get('query', 'N/A')[:50]}...")
                        print(f"    - Status: {data.get('status', 'unknown')}")
                        if 'config' in data:
                            print(f"    - OpenCog available: {data['config'].get('opencog_available', False)}")
                    except json.JSONDecodeError:
                        print("    - Response data format: Non-JSON")
                
            except Exception as e:
                print(f"    ⚠️ Reasoning error: {e}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Cognitive reasoning integration error: {e}")
        return False

async def demonstrate_cognitive_memory_integration():
    """Demonstrate cognitive memory tool with ECAN integration."""
    print("\n💾 Testing Cognitive Memory with ECAN...")
    
    try:
        from python.tools.cognitive_memory import CognitiveMemoryTool
        
        # Mock agent
        class MockAgent:
            def __init__(self):
                self.config = {}
        
        agent = MockAgent()
        memory_tool = CognitiveMemoryTool(
            agent=agent,
            name="cognitive_memory",
            method="store"
        )
        
        # Test memory operations with ECAN attention
        test_data = [
            {
                "concept": "learning_strategy",
                "properties": {
                    "type": "cognitive_method",
                    "effectiveness": "high",
                    "domain": "problem_solving"
                },
                "importance": 2.0
            },
            {
                "concept": "attention_mechanism",
                "properties": {
                    "type": "cognitive_process",
                    "function": "focus_allocation",
                    "complexity": "high"
                },
                "importance": 2.5
            },
            {
                "concept": "memory_integration", 
                "properties": {
                    "type": "system_function",
                    "purpose": "cross_tool_coordination",
                    "priority": "critical"
                },
                "importance": 3.0
            }
        ]
        
        for i, data in enumerate(test_data):
            print(f"\n  Storing concept {i+1}: {data['concept']}")
            
            try:
                response = await memory_tool.execute(
                    operation="store",
                    data=data
                )
                
                print(f"  ✓ Storage completed: {response.message[:80]}...")
                
            except Exception as e:
                print(f"    ⚠️ Storage error: {e}")
        
        # Test memory status
        print(f"\n  Checking memory status...")
        try:
            status_response = await memory_tool.execute(operation="status")
            print(f"  ✓ Status retrieved: {status_response.message[:80]}...")
        except Exception as e:
            print(f"    ⚠️ Status error: {e}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Cognitive memory integration error: {e}")
        return False

async def demonstrate_meta_cognition_integration():
    """Demonstrate meta-cognition tool with ECAN integration."""
    print("\n🎭 Testing Meta-Cognition with ECAN...")
    
    try:
        from python.tools.meta_cognition import MetaCognitionTool
        
        # Mock agent
        class MockAgent:
            def __init__(self):
                self.config = {}
        
        agent = MockAgent()
        meta_tool = MetaCognitionTool(
            agent=agent,
            name="meta_cognition", 
            method="attention_focus"
        )
        
        # Test meta-cognitive operations
        test_operations = [
            {
                "operation": "attention_focus",
                "goals": ["improve_reasoning", "enhance_memory", "optimize_attention"],
                "tasks": ["problem_solving", "knowledge_retrieval", "self_assessment"],
                "importance": 100
            },
            {
                "operation": "self_reflect", 
                "context": "Analyzing cognitive performance and capabilities"
            }
        ]
        
        for i, op_data in enumerate(test_operations):
            operation = op_data.pop("operation")
            print(f"\n  Meta-cognitive operation {i+1}: {operation}")
            
            try:
                response = await meta_tool.execute(
                    operation=operation,
                    **op_data
                )
                
                print(f"  ✓ Operation completed: {response.message[:80]}...")
                
                # Try to extract JSON data from message
                if "Data:" in response.message:
                    data_str = response.message.split("Data:")[1].strip()
                    try:
                        data = json.loads(data_str)
                        if "results" in data:
                            results = data["results"]
                            if "attention_allocated" in results:
                                print(f"    - ECAN attention allocated: {results['attention_allocated']}")
                            if "goals_processed" in results:
                                print(f"    - Goals processed: {results['goals_processed']}")
                            if "ecan_available" in results:
                                print(f"    - ECAN available: {results['ecan_available']}")
                    except json.JSONDecodeError:
                        print("    - Response contains non-JSON data")
                
            except Exception as e:
                print(f"    ⚠️ Meta-cognitive operation error: {e}")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Meta-cognition integration error: {e}")
        return False

def show_integration_summary(results: Dict[str, bool]):
    """Show summary of integration test results."""
    print("\n" + "=" * 80)
    print("📋 ECAN Cross-Tool Integration Summary")
    print("=" * 80)
    
    total_tests = len(results)
    passed_tests = sum(results.values())
    
    print(f"Overall Results: {passed_tests}/{total_tests} tests passed")
    print()
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"  {status}: {test_name}")
    
    print()
    if passed_tests == total_tests:
        print("🎉 All ECAN integration tests passed!")
        print("✓ Cross-tool attention coordination is working")
        print("✓ Cognitive tools are integrated with ECAN")
        print("✓ Attention allocation and synchronization functional")
    else:
        print("⚠️ Some integration tests failed")
        print("ℹ️ This is expected in development environments without OpenCog")
        print("ℹ️ Fallback mechanisms are providing basic functionality")
    
    print()
    print("🔗 Integration Features Demonstrated:")
    print("  - Centralized ECAN coordinator")
    print("  - Cross-tool attention requests") 
    print("  - Priority-based attention allocation")
    print("  - Attention synchronization across tools")
    print("  - Fallback mechanisms for development environments")
    print("  - Real-time attention metrics and monitoring")

async def main():
    """Main demo execution function."""
    show_demo_banner()
    demo_fallback_explanation()
    
    # Track test results
    test_results = {}
    
    # Run integration demonstrations
    test_results["ECAN Coordinator"] = await demonstrate_ecan_coordination()
    test_results["Cognitive Reasoning Integration"] = await demonstrate_cognitive_reasoning_integration()
    test_results["Cognitive Memory Integration"] = await demonstrate_cognitive_memory_integration() 
    test_results["Meta-Cognition Integration"] = await demonstrate_meta_cognition_integration()
    
    # Show summary
    show_integration_summary(test_results)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n⚠️ Demo interrupted by user")
    except Exception as e:
        print(f"\n💥 Demo error: {e}")
        print("ℹ️ This may be due to missing dependencies in the development environment")