#!/usr/bin/env python3
"""
Test script for Multi-Agent Cognitive Collaboration Framework
Validates the implementation of multi-agent coordination, collaboration, and integration.
"""

import asyncio
import json
import time
import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Test the multi-agent framework components
try:
    from python.helpers.multi_agent import (
        MultiAgentSystem, 
        CognitiveMultiAgentSystem, 
        create_cognitive_agent_network,
        AgentProfile,
        CollaborationTask
    )
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Multi-agent framework not available: {e}")
    MULTI_AGENT_AVAILABLE = False

try:
    from python.tools.multi_agent_collaboration import MultiAgentCollaborationTool
    MULTI_AGENT_TOOL_AVAILABLE = True
except ImportError as e:
    print(f"Multi-agent tool not available: {e}")
    MULTI_AGENT_TOOL_AVAILABLE = False


class MultiAgentFrameworkTester:
    """Comprehensive tester for multi-agent cognitive collaboration framework."""
    
    def __init__(self):
        self.test_results = {
            "basic_multi_agent_system": False,
            "cognitive_multi_agent_system": False,
            "agent_registration": False,
            "task_coordination": False,
            "collaborative_reasoning": False,
            "memory_sharing": False,
            "end_to_end_workflow": False,
            "tool_integration": False
        }
        
    async def run_all_tests(self):
        """Run all multi-agent framework tests."""
        print("=" * 80)
        print("MULTI-AGENT COGNITIVE COLLABORATION FRAMEWORK TESTS")
        print("=" * 80)
        
        if not MULTI_AGENT_AVAILABLE:
            print("❌ Multi-agent framework not available")
            return False
        
        # Test basic multi-agent system
        await self.test_basic_multi_agent_system()
        
        # Test cognitive multi-agent system
        await self.test_cognitive_multi_agent_system()
        
        # Test agent registration
        await self.test_agent_registration()
        
        # Test task coordination
        await self.test_task_coordination()
        
        # Test collaborative reasoning
        await self.test_collaborative_reasoning()
        
        # Test memory sharing
        await self.test_memory_sharing()
        
        # Test end-to-end workflow
        await self.test_end_to_end_workflow()
        
        # Test tool integration if available
        if MULTI_AGENT_TOOL_AVAILABLE:
            await self.test_tool_integration()
        
        # Print results
        self.print_test_results()
        
        return all(self.test_results.values())
    
    async def test_basic_multi_agent_system(self):
        """Test basic MultiAgentSystem functionality."""
        print("\n1. Testing Basic Multi-Agent System")
        print("-" * 40)
        
        try:
            # Create basic system
            system = MultiAgentSystem("TestSystem")
            
            # Register agents
            agent1 = system.register_agent(
                agent_id="test_agent_1",
                name="TestAgent1",
                role="tester",
                capabilities=["testing", "validation"]
            )
            
            # Check registration
            assert len(system.agents) == 1
            assert agent1.agent_id == "test_agent_1"
            assert agent1.status == "active"
            
            # Get status
            status = system.get_coordination_status()
            assert status["total_agents"] == 1
            assert status["active_agents"] == 1
            
            print("  ✓ System creation and agent registration working")
            print(f"  ✓ Status: {status['total_agents']} agents, {status['system_status']}")
            
            self.test_results["basic_multi_agent_system"] = True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results["basic_multi_agent_system"] = False
    
    async def test_cognitive_multi_agent_system(self):
        """Test CognitiveMultiAgentSystem functionality."""
        print("\n2. Testing Cognitive Multi-Agent System")
        print("-" * 40)
        
        try:
            # Create cognitive system
            system = create_cognitive_agent_network(num_agents=3)
            
            # Check initialization
            assert len(system.cognitive_agents) >= 3
            assert len(system.agents) >= 3
            
            # Check agent profiles
            for agent_id in system.cognitive_agents:
                profile = system.agents[agent_id]
                assert len(profile.capabilities) > 0
                assert profile.status == "active"
            
            # Test system status
            status = system.get_coordination_status()
            assert status["total_agents"] >= 3
            assert status["system_status"] == "operational"
            
            print(f"  ✓ Cognitive system initialized with {len(system.cognitive_agents)} agents")
            print(f"  ✓ System status: {status['system_status']}")
            
            self.test_results["cognitive_multi_agent_system"] = True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results["cognitive_multi_agent_system"] = False
    
    async def test_agent_registration(self):
        """Test agent registration and management."""
        print("\n3. Testing Agent Registration")
        print("-" * 40)
        
        try:
            system = MultiAgentSystem("RegistrationTest")
            
            # Register multiple agents with different profiles
            agents = [
                ("reasoner_1", "CognitiveReasoner", "reasoning_specialist", ["logical_reasoning", "inference"]),
                ("analyzer_1", "CognitiveAnalyzer", "analysis_specialist", ["data_analysis", "pattern_recognition"]),
                ("coordinator_1", "CognitiveCoordinator", "coordination_specialist", ["task_coordination", "communication"])
            ]
            
            for agent_id, name, role, capabilities in agents:
                profile = system.register_agent(
                    agent_id=agent_id,
                    name=name,
                    role=role,
                    capabilities=capabilities,
                    specializations=[f"{role}_expert"]
                )
                
                assert profile.agent_id == agent_id
                assert profile.name == name
                assert profile.role == role
                assert set(capabilities).issubset(set(profile.capabilities))
            
            # Check total registration
            assert len(system.agents) == 3
            assert len(system.coordination_history) == 3  # One registration event per agent
            
            print(f"  ✓ Registered {len(system.agents)} agents successfully")
            print(f"  ✓ Coordination history: {len(system.coordination_history)} events")
            
            self.test_results["agent_registration"] = True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results["agent_registration"] = False
    
    async def test_task_coordination(self):
        """Test task creation and coordination."""
        print("\n4. Testing Task Coordination")
        print("-" * 40)
        
        try:
            system = create_cognitive_agent_network(num_agents=3)
            
            # Create collaboration task
            task = system.create_collaboration_task(
                description="Test collaborative reasoning task",
                required_capabilities=["logical_reasoning", "analysis"],
                priority=2
            )
            
            # Check task creation
            assert task.status in ["active", "pending"]
            assert len(task.required_capabilities) == 2
            assert task.priority == 2
            
            # Coordinate the task
            coordination_plan = await system.coordinate_task(task.task_id)
            
            # Check coordination
            assert coordination_plan["task_id"] == task.task_id
            assert "coordination_strategy" in coordination_plan
            assert len(coordination_plan["assigned_agents"]) > 0
            
            print(f"  ✓ Task created with ID: {task.task_id}")
            print(f"  ✓ Assigned to {len(task.assigned_agents)} agents")
            print(f"  ✓ Coordination strategy: {task.coordination_strategy}")
            
            self.test_results["task_coordination"] = True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results["task_coordination"] = False
    
    async def test_collaborative_reasoning(self):
        """Test collaborative reasoning across agents."""
        print("\n5. Testing Collaborative Reasoning")
        print("-" * 40)
        
        try:
            system = create_cognitive_agent_network(num_agents=3)
            
            # Perform collaborative reasoning
            problem = "How to optimize multi-agent communication protocols?"
            result = await system.collaborative_reasoning(problem)
            
            # Check reasoning result
            assert "problem" in result
            assert "collaborative_solution" in result
            assert "agent_contributions" in result
            assert "overall_confidence" in result
            assert "cognitive_insights" in result
            
            assert result["problem"] == problem
            assert result["agent_contributions"] > 0
            assert 0 <= result["overall_confidence"] <= 1
            assert len(result["cognitive_insights"]) > 0
            
            print(f"  ✓ Problem: {problem[:50]}...")
            print(f"  ✓ Agent contributions: {result['agent_contributions']}")
            print(f"  ✓ Overall confidence: {result['overall_confidence']:.2f}")
            print(f"  ✓ Cognitive insights: {len(result['cognitive_insights'])}")
            
            self.test_results["collaborative_reasoning"] = True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results["collaborative_reasoning"] = False
    
    async def test_memory_sharing(self):
        """Test shared memory and knowledge sharing."""
        print("\n6. Testing Memory Sharing")
        print("-" * 40)
        
        try:
            system = create_cognitive_agent_network(num_agents=3)
            
            # Simulate memory sharing
            knowledge_items = [
                ("cognitive_reasoner_1", "logical_reasoning_patterns"),
                ("cognitive_analyzer_2", "data_analysis_techniques"), 
                ("cognitive_coordinator_3", "coordination_protocols")
            ]
            
            memory_stats = await system.simulate_memory_sharing(knowledge_items)
            
            # Check memory sharing results
            assert "atoms" in memory_stats
            assert "events" in memory_stats
            assert memory_stats["atoms"] >= len(knowledge_items)
            assert memory_stats["events"] >= len(knowledge_items)
            
            print(f"  ✓ Knowledge items shared: {len(knowledge_items)}")
            print(f"  ✓ Memory atoms created: {memory_stats['atoms']}")
            print(f"  ✓ Memory events: {memory_stats['events']}")
            print(f"  ✓ Shared memory size: {system._get_memory_size()}")
            
            self.test_results["memory_sharing"] = True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results["memory_sharing"] = False
    
    async def test_end_to_end_workflow(self):
        """Test complete end-to-end multi-agent workflow."""
        print("\n7. Testing End-to-End Workflow")
        print("-" * 40)
        
        try:
            system = create_cognitive_agent_network(num_agents=3)
            
            # Run end-to-end workflow
            workflow_result = await system.simulate_end_to_end_workflow()
            
            # Check workflow results
            assert "workflow_id" in workflow_result
            assert "steps_completed" in workflow_result
            assert "total_steps" in workflow_result
            assert "overall_success" in workflow_result
            assert "workflow_steps" in workflow_result
            
            assert workflow_result["steps_completed"] == workflow_result["total_steps"]
            assert workflow_result["overall_success"] == True
            assert len(workflow_result["workflow_steps"]) >= 6
            
            # Check all steps completed
            for step in workflow_result["workflow_steps"]:
                assert step["status"] == "completed"
            
            print(f"  ✓ Workflow ID: {workflow_result['workflow_id']}")
            print(f"  ✓ Steps completed: {workflow_result['steps_completed']}/{workflow_result['total_steps']}")
            print(f"  ✓ Overall success: {workflow_result['overall_success']}")
            
            self.test_results["end_to_end_workflow"] = True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results["end_to_end_workflow"] = False
    
    async def test_tool_integration(self):
        """Test multi-agent tool integration with Agent-Zero."""
        print("\n8. Testing Tool Integration")
        print("-" * 40)
        
        try:
            # Create mock agent for tool testing
            class MockAgent:
                def __init__(self):
                    self.number = 0
                    self.agent_name = "test_agent"
            
            mock_agent = MockAgent()
            
            # Create tool instance
            tool = MultiAgentCollaborationTool(
                agent=mock_agent,
                num_agents=3
            )
            
            # Test status operation
            status_response = await tool.execute(operation="status")
            assert status_response.message is not None
            assert "data" in status_response.__dict__
            
            # Test agents listing
            agents_response = await tool.execute(operation="agents")
            assert agents_response.message is not None
            
            # Test collaborative reasoning
            collab_response = await tool.execute(
                operation="collaborate",
                problem="Test collaborative reasoning integration"
            )
            assert collab_response.message is not None
            
            print("  ✓ Tool initialization successful")
            print("  ✓ Status operation working")
            print("  ✓ Agent listing working")
            print("  ✓ Collaborative reasoning working")
            
            self.test_results["tool_integration"] = True
            
        except Exception as e:
            print(f"  ❌ Error: {e}")
            self.test_results["tool_integration"] = False
    
    def print_test_results(self):
        """Print comprehensive test results."""
        print("\n" + "=" * 80)
        print("MULTI-AGENT FRAMEWORK TEST RESULTS")
        print("=" * 80)
        
        total_tests = len(self.test_results)
        passed_tests = sum(self.test_results.values())
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "❌ FAIL"
            test_display = test_name.replace("_", " ").title()
            print(f"{test_display:.<50} {status}")
        
        print("-" * 80)
        print(f"SUMMARY: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests == total_tests:
            print("🎉 ALL TESTS PASSED - Multi-Agent Framework is working correctly!")
            return True
        else:
            print(f"⚠️  {total_tests - passed_tests} test(s) failed - Framework needs attention")
            return False


async def main():
    """Main test function."""
    print("Starting Multi-Agent Cognitive Collaboration Framework Tests...")
    
    tester = MultiAgentFrameworkTester()
    success = await tester.run_all_tests()
    
    if success:
        print("\n✅ Multi-Agent Cognitive Collaboration Framework is READY!")
        return 0
    else:
        print("\n❌ Multi-Agent Framework tests failed - check implementation")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())