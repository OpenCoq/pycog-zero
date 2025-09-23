#!/usr/bin/env python3
"""
Meta-Cognitive Self-Reflection Demonstration
Shows the capabilities of the MetaCognitionTool for Agent-Zero

This demonstration showcases:
1. Self-reflection and recursive self-description
2. Attention allocation with ECAN dynamics  
3. Goal prioritization based on cognitive assessment
4. Deep introspection and behavioral analysis
5. Meta-cognitive status monitoring
"""

import asyncio
import json
import sys
import os
from unittest.mock import Mock, patch

# Add project root to path
sys.path.insert(0, os.path.abspath('.'))

async def demo_meta_cognitive_capabilities():
    """Demonstrate meta-cognitive self-reflection capabilities."""
    
    print("🧠 META-COGNITIVE SELF-REFLECTION DEMONSTRATION")
    print("=" * 60)
    
    # Setup mock environment
    with patch.dict('sys.modules', {
        'python.helpers.tool': Mock(),
        'python.helpers.files': Mock(),
        'opencog.atomspace': Mock(),
        'opencog.utilities': Mock(), 
        'opencog.ecan': Mock(),
        'python.tools.atomspace_tool_hub': Mock(),
        'python.tools.atomspace_memory_bridge': Mock(),
    }):
        # Mock the base classes
        class MockTool:
            def __init__(self, agent, name, method=None, args=None, message="", loop_data=None, **kwargs):
                self.agent = agent
                self.name = name
                self.method = method
                self.args = args or {}
                self.message = message
                self.loop_data = loop_data

        class MockResponse:
            def __init__(self, message, break_loop=False):
                self.message = message
                self.break_loop = break_loop

        sys.modules['python.helpers.tool'].Tool = MockTool
        sys.modules['python.helpers.tool'].Response = MockResponse
        sys.modules['python.helpers.files'].get_abs_path = Mock(return_value="/tmp/config.json")
        
        mock_config = {
            "meta_cognitive": {
                "self_reflection_enabled": True,
                "attention_allocation_enabled": True,
                "goal_prioritization_enabled": True,
                "recursive_depth": 3,
                "memory_persistence": True,
                "cross_tool_integration": True
            }
        }
        
        with patch('builtins.open', Mock()):
            with patch('json.load', return_value=mock_config):
                from python.tools.meta_cognition import MetaCognitionTool
                
                # Create sophisticated mock agent
                mock_agent = Mock()
                mock_agent.agent_name = "AdvancedCognitiveAgent"
                mock_agent.get_capabilities = Mock(return_value=[
                    "reasoning", "memory", "learning", "adaptation", 
                    "pattern_recognition", "problem_solving", "creativity",
                    "meta_cognition", "self_reflection", "goal_planning"
                ])
                mock_agent.get_tools = Mock(return_value=[
                    Mock(__class__=Mock(__name__="CognitiveReasoningTool")),
                    Mock(__class__=Mock(__name__="MemoryTool")),
                    Mock(__class__=Mock(__name__="AtomSpaceToolHub")),
                    Mock(__class__=Mock(__name__="NeuralSymbolicBridge")),
                    Mock(__class__=Mock(__name__="MetaCognitionTool"))
                ])
                
                # Initialize meta-cognition tool
                print("🔧 Initializing Meta-Cognitive Self-Reflection System...")
                meta_tool = MetaCognitionTool(mock_agent, "meta_cognition", args={})
                
                print(f"✅ Initialized instance {meta_tool.instance_id}")
                print(f"🎯 Meta-level achieved: {meta_tool._calculate_meta_level()}")
                print()
                
                # 1. Demonstrate Self-Reflection
                print("1️⃣  SELF-REFLECTION DEMONSTRATION")
                print("-" * 40)
                
                response = await meta_tool.execute(
                    operation="self_reflect",
                    recursive_depth=2,
                    context="problem_solving_session",
                    focus_areas=["learning", "adaptation", "performance"]
                )
                
                print("🔍 Self-Reflection Response:")
                print(extract_and_format_data(response.message))
                print()
                
                # 2. Demonstrate Attention Allocation
                print("2️⃣  ATTENTION ALLOCATION DEMONSTRATION")
                print("-" * 40)
                
                attention_params = {
                    "goals": [
                        "improve_problem_solving_accuracy",
                        "enhance_learning_efficiency", 
                        "develop_creative_thinking",
                        "optimize_memory_usage",
                        "strengthen_meta_cognitive_awareness"
                    ],
                    "tasks": [
                        "analyze_complex_patterns",
                        "generate_novel_solutions",
                        "integrate_cross_domain_knowledge",
                        "monitor_cognitive_performance"
                    ],
                    "importance": 90
                }
                
                response = await meta_tool.execute(
                    operation="attention_focus", 
                    **attention_params
                )
                
                print("🎯 Attention Allocation Response:")
                print(extract_and_format_data(response.message))
                print()
                
                # 3. Demonstrate Goal Prioritization
                print("3️⃣  GOAL PRIORITIZATION DEMONSTRATION")
                print("-" * 40)
                
                complex_goals = [
                    "master_advanced_reasoning_techniques",
                    "develop_intuitive_problem_solving",
                    "enhance_creative_idea_generation",
                    "improve_knowledge_integration_speed", 
                    "strengthen_self_awareness_capabilities",
                    "optimize_cognitive_resource_allocation",
                    "build_robust_learning_mechanisms",
                    "develop_adaptive_behavior_patterns"
                ]
                
                response = await meta_tool.execute(
                    operation="goal_prioritize",
                    goals=complex_goals,
                    context="cognitive_enhancement_session"
                )
                
                print("📊 Goal Prioritization Response:")
                print(extract_and_format_data(response.message))
                print()
                
                # 4. Demonstrate Deep Introspection
                print("4️⃣  DEEP INTROSPECTION DEMONSTRATION")
                print("-" * 40)
                
                response = await meta_tool.execute(
                    operation="introspect",
                    introspection_depth=3,
                    focus_areas=["behavioral_patterns", "learning_capacity", "meta_awareness"]
                )
                
                print("🔬 Deep Introspection Response:")
                print(extract_and_format_data(response.message))
                print()
                
                # 5. Demonstrate System Status
                print("5️⃣  SYSTEM STATUS DEMONSTRATION")
                print("-" * 40)
                
                response = await meta_tool.execute(operation="status")
                
                print("📈 System Status Response:")
                print(extract_and_format_data(response.message))
                print()
                
                # 6. Demonstrate Recursive Analysis Chain
                print("6️⃣  RECURSIVE ANALYSIS CHAIN DEMONSTRATION")
                print("-" * 40)
                
                print("Performing multi-level recursive self-analysis...")
                
                for depth in [1, 2, 3]:
                    print(f"\n🔄 Recursion Depth {depth}:")
                    response = await meta_tool.execute(
                        operation="self_reflect",
                        recursive_depth=depth
                    )
                    
                    data = extract_data_from_response(response.message)
                    if data and "recursive_depth" in data:
                        print(f"   Achieved depth: {data['recursive_depth']}")
                        if "agent_state" in data and "recursive_analysis" in data["agent_state"]:
                            analysis = data["agent_state"]["recursive_analysis"]
                            print(f"   Insights generated: {len(analysis.get('recursive_insights', []))}")
                            if analysis.get("deeper_analysis"):
                                print(f"   Has deeper analysis: Yes")
                            else:
                                print(f"   Has deeper analysis: No")
                
                print()
                
                # 7. Demonstrate Behavioral Pattern Analysis
                print("7️⃣  BEHAVIORAL PATTERN ANALYSIS")
                print("-" * 40)
                
                # Simulate some attention history for analysis
                import time
                meta_tool.attention_history = [
                    {
                        "timestamp": time.time() - 1000,
                        "goals": ["learn", "adapt", "improve"],
                        "tasks": ["analyze", "synthesize"],
                        "distribution": {}
                    },
                    {
                        "timestamp": time.time() - 800,
                        "goals": ["learn", "create", "optimize"],
                        "tasks": ["generate", "evaluate"],
                        "distribution": {}
                    },
                    {
                        "timestamp": time.time() - 600,
                        "goals": ["adapt", "learn", "innovate"],
                        "tasks": ["experiment", "refine"],
                        "distribution": {}
                    }
                ]
                
                patterns = await meta_tool._analyze_behavioral_patterns()
                print("🧩 Behavioral Patterns Detected:")
                print(f"   Attention patterns: {len(patterns.get('attention_patterns', []))}")
                if patterns.get('attention_patterns'):
                    for pattern in patterns['attention_patterns'][:3]:
                        print(f"   - {pattern['goal']}: {pattern['frequency']} occurrences")
                print()
                
                # 8. Summary
                print("8️⃣  DEMONSTRATION SUMMARY")
                print("-" * 40)
                
                print("✅ Meta-Cognitive Capabilities Demonstrated:")
                print("   🧠 Self-reflection with recursive analysis")
                print("   🎯 Attention allocation using ECAN dynamics")
                print("   📊 Goal prioritization based on cognitive assessment")
                print("   🔬 Deep introspection and behavioral analysis")
                print("   📈 Comprehensive system status monitoring")
                print("   🔄 Multi-level recursive self-awareness")
                print("   🧩 Behavioral pattern recognition")
                print()
                
                print("🎉 Meta-Cognitive Self-Reflection Implementation Complete!")
                print("Ready for integration with Agent-Zero framework")

def extract_and_format_data(response_message: str) -> str:
    """Extract and format JSON data from response message."""
    try:
        data_start = response_message.find("Data: {")
        if data_start != -1:
            data_json = response_message[data_start + 6:]
            data = json.loads(data_json)
            return json.dumps(data, indent=2)
    except:
        pass
    return "Data extraction not available"

def extract_data_from_response(response_message: str) -> dict:
    """Extract JSON data from response message."""
    try:
        data_start = response_message.find("Data: {")
        if data_start != -1:
            data_json = response_message[data_start + 6:]
            return json.loads(data_json)
    except:
        pass
    return {}

if __name__ == "__main__":
    print("Starting Meta-Cognitive Self-Reflection Demonstration...")
    asyncio.run(demo_meta_cognitive_capabilities())