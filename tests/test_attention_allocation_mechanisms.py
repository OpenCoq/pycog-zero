"""
Comprehensive test suite for attention allocation mechanisms with Agent-Zero framework.

This test suite validates the attention allocation functionality as requested in issue #49:
"Test attention allocation mechanisms with Agent-Zero framework"

Tests cover both ECAN and fallback attention mechanisms, performance benchmarks,
and integration with Agent-Zero framework components.
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch
from typing import Dict, Any, List

# Test imports (graceful handling)
try:
    from python.tools.meta_cognition import MetaCognitionTool
    from python.helpers.tool import Tool, Response
    META_COGNITION_AVAILABLE = True
except ImportError as e:
    print(f"Meta-cognition tool not available for testing: {e}")
    META_COGNITION_AVAILABLE = False


@pytest.mark.skipif(not META_COGNITION_AVAILABLE, reason="Meta-cognition tool not available")
class TestAttentionAllocationMechanisms:
    """Test suite focused on attention allocation mechanisms."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock Agent-Zero agent for testing."""
        agent = Mock()
        agent.agent_name = "attention_test_agent"
        agent.get_capabilities = Mock(return_value=[
            "reasoning", "memory", "learning", "attention", "goal_prioritization"
        ])
        agent.get_tools = Mock(return_value=[
            Mock(__class__=Mock(__name__="CognitiveReasoningTool")),
            Mock(__class__=Mock(__name__="MetaCognitionTool"))
        ])
        return agent
    
    @pytest.fixture
    def meta_tool(self, mock_agent):
        """Create MetaCognitionTool instance for testing."""
        return MetaCognitionTool(
            agent=mock_agent,
            name="meta_cognition",
            args={}
        )
    
    @pytest.mark.asyncio
    async def test_basic_attention_allocation(self, meta_tool):
        """Test basic attention allocation functionality."""
        test_params = {
            "goals": ["primary_objective", "secondary_goal", "tertiary_task"],
            "tasks": ["immediate_action", "scheduled_work", "background_process"],
            "importance": 100
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        assert isinstance(response, Response)
        assert "attention allocated" in response.message.lower()
        assert not response.break_loop
        
        # Verify response structure
        data_start = response.message.find("Data: {")
        assert data_start != -1, "Response should contain JSON data"
        
        data_json = response.message[data_start + 6:]
        data = json.loads(data_json)
        
        # Validate data structure
        assert data["operation"] == "attention_focus"
        assert data["goals_count"] == 3
        assert data["tasks_count"] == 3
        assert data["status"] == "success"
        assert "results" in data
        
        results = data["results"]
        assert results["attention_allocated"] is True
        assert "ecan_available" in results
        assert results["goals_processed"] == 3
        assert results["tasks_processed"] == 3
    
    @pytest.mark.asyncio
    async def test_empty_goals_and_tasks(self, meta_tool):
        """Test attention allocation with empty goals and tasks."""
        test_params = {
            "goals": [],
            "tasks": [],
            "importance": 50
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        assert isinstance(response, Response)
        data_start = response.message.find("Data: {")
        data = json.loads(response.message[data_start + 6:])
        
        assert data["goals_count"] == 0
        assert data["tasks_count"] == 0
        assert data["status"] == "success"
        
        results = data["results"]
        assert results["goals_processed"] == 0
        assert results["tasks_processed"] == 0
    
    @pytest.mark.asyncio
    async def test_large_goal_set_attention(self, meta_tool):
        """Test attention allocation with large sets of goals and tasks."""
        large_goals = [f"goal_{i}" for i in range(20)]
        large_tasks = [f"task_{i}" for i in range(15)]
        
        test_params = {
            "goals": large_goals,
            "tasks": large_tasks,
            "importance": 80
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        data_start = response.message.find("Data: {")
        data = json.loads(response.message[data_start + 6:])
        
        assert data["goals_count"] == 20
        assert data["tasks_count"] == 15
        
        results = data["results"]
        assert results["goals_processed"] == 20
        assert results["tasks_processed"] == 15
        
        # Verify fallback mode properly handles large sets
        if results.get("fallback_mode"):
            assert "prioritized_goals" in results
            assert "prioritized_tasks" in results
            assert len(results["prioritized_goals"]) == 20
            assert len(results["prioritized_tasks"]) == 15
    
    @pytest.mark.asyncio
    async def test_priority_ordering(self, meta_tool):
        """Test that attention allocation correctly orders priorities."""
        test_params = {
            "goals": ["critical_mission", "important_goal", "nice_to_have"],
            "tasks": ["urgent_task", "normal_task", "low_priority"],
            "importance": 100
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        data_start = response.message.find("Data: {")
        data = json.loads(response.message[data_start + 6:])
        
        results = data["results"]
        
        # Check priority ordering in fallback mode
        if results.get("fallback_mode"):
            goals = results["prioritized_goals"]
            tasks = results["prioritized_tasks"]
            
            # Verify goals are prioritized in descending order
            for i in range(len(goals) - 1):
                assert goals[i]["priority"] >= goals[i + 1]["priority"]
                assert goals[i]["rank"] == i + 1
            
            # Verify tasks are prioritized in descending order
            for i in range(len(tasks) - 1):
                assert tasks[i]["priority"] >= tasks[i + 1]["priority"]
                assert tasks[i]["rank"] == i + 1
            
            # First goal should have highest priority
            assert goals[0]["goal"] == "critical_mission"
            assert tasks[0]["task"] == "urgent_task"
    
    @pytest.mark.asyncio
    async def test_importance_scaling(self, meta_tool):
        """Test that importance parameter correctly scales priorities."""
        base_goals = ["goal_1", "goal_2"]
        base_tasks = ["task_1", "task_2"]
        
        # Test with different importance levels
        for importance in [25, 50, 75, 100]:
            test_params = {
                "goals": base_goals,
                "tasks": base_tasks,
                "importance": importance
            }
            
            response = await meta_tool.execute(operation="attention_focus", **test_params)
            
            data_start = response.message.find("Data: {")
            data = json.loads(response.message[data_start + 6:])
            
            results = data["results"]
            
            if results.get("fallback_mode"):
                goals = results["prioritized_goals"]
                tasks = results["prioritized_tasks"]
                
                # Verify priority scaling
                assert goals[0]["priority"] == importance * 1.0  # First goal gets full importance
                assert goals[1]["priority"] == importance * 0.9  # Second goal gets 90%
                
                assert tasks[0]["priority"] == importance * 0.7  # Tasks get 70% base
                assert tasks[1]["priority"] == importance * 0.63  # Second task: 70% * 90%
    
    @pytest.mark.asyncio
    async def test_attention_distribution(self, meta_tool):
        """Test attention distribution functionality."""
        test_params = {
            "goals": ["analyze_data", "generate_insights", "make_decisions"],
            "tasks": ["collect_info", "process_data", "output_results"],
            "importance": 90
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        data_start = response.message.find("Data: {")
        data = json.loads(response.message[data_start + 6:])
        
        results = data["results"]
        
        # Check distribution data
        assert "distribution" in results
        distribution = results["distribution"]
        
        if results.get("fallback_mode"):
            assert "top_goals" in distribution
            assert "top_tasks" in distribution
            
            # Top goals should be present and ordered
            top_goals = distribution["top_goals"]
            assert len(top_goals) <= 3  # Should limit to top 3
            if top_goals:
                assert top_goals[0] == "analyze_data"  # First goal should be highest priority
            
            # Top tasks should be present and ordered
            top_tasks = distribution["top_tasks"]
            assert len(top_tasks) <= 3
            if top_tasks:
                assert top_tasks[0] == "collect_info"  # First task should be highest priority
    
    @pytest.mark.asyncio
    async def test_ecan_fallback_behavior(self, meta_tool):
        """Test proper fallback behavior when ECAN is not available."""
        test_params = {
            "goals": ["test_goal"],
            "tasks": ["test_task"],
            "importance": 75
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        data_start = response.message.find("Data: {")
        data = json.loads(response.message[data_start + 6:])
        
        results = data["results"]
        
        # Should indicate ECAN availability status
        assert "ecan_available" in results
        
        # If ECAN is not available, should use fallback mode
        if not results["ecan_available"]:
            assert results.get("fallback_mode") is True
            assert "prioritized_goals" in results
            assert "prioritized_tasks" in results
    
    @pytest.mark.asyncio
    async def test_attention_allocation_performance(self, meta_tool):
        """Test attention allocation performance with timing."""
        large_goals = [f"performance_goal_{i}" for i in range(100)]
        large_tasks = [f"performance_task_{i}" for i in range(100)]
        
        test_params = {
            "goals": large_goals,
            "tasks": large_tasks,
            "importance": 85
        }
        
        start_time = time.time()
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        end_time = time.time()
        
        execution_time = end_time - start_time
        
        # Performance assertion - should complete in reasonable time
        assert execution_time < 5.0, f"Attention allocation took too long: {execution_time}s"
        
        # Verify correctness despite large input
        data_start = response.message.find("Data: {")
        data = json.loads(response.message[data_start + 6:])
        
        assert data["goals_count"] == 100
        assert data["tasks_count"] == 100
        assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_invalid_importance_values(self, meta_tool):
        """Test attention allocation with edge case importance values."""
        test_cases = [
            {"importance": 0, "expected_min": 0},
            {"importance": -10, "expected_min": 0},  # Should handle negative values
            {"importance": 1000, "expected_max": 1000}  # Should handle large values
        ]
        
        for case in test_cases:
            test_params = {
                "goals": ["test_goal"],
                "tasks": ["test_task"],
                "importance": case["importance"]
            }
            
            response = await meta_tool.execute(operation="attention_focus", **test_params)
            
            # Should not fail with invalid importance values
            assert isinstance(response, Response)
            assert "error" not in response.message.lower() or "failed" not in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_concurrent_attention_allocation(self, meta_tool):
        """Test concurrent attention allocation requests."""
        async def allocate_attention_task(task_id):
            test_params = {
                "goals": [f"concurrent_goal_{task_id}"],
                "tasks": [f"concurrent_task_{task_id}"],
                "importance": 60 + task_id
            }
            return await meta_tool.execute(operation="attention_focus", **test_params)
        
        # Run multiple attention allocation tasks concurrently
        tasks = [allocate_attention_task(i) for i in range(5)]
        results = await asyncio.gather(*tasks)
        
        # All tasks should complete successfully
        for i, response in enumerate(results):
            assert isinstance(response, Response)
            data_start = response.message.find("Data: {")
            data = json.loads(response.message[data_start + 6:])
            assert data["status"] == "success"
            assert data["goals_count"] == 1
            assert data["tasks_count"] == 1
    
    @pytest.mark.asyncio
    async def test_attention_allocation_with_agent_zero_integration(self, meta_tool):
        """Test attention allocation integrated with Agent-Zero framework features."""
        # Simulate Agent-Zero specific goals and tasks
        agent_zero_goals = [
            "enhance_reasoning_capability",
            "improve_memory_efficiency", 
            "optimize_learning_process",
            "increase_adaptation_speed"
        ]
        
        agent_zero_tasks = [
            "analyze_current_performance",
            "identify_improvement_areas",
            "implement_optimizations",
            "validate_enhancements",
            "update_knowledge_base"
        ]
        
        test_params = {
            "goals": agent_zero_goals,
            "tasks": agent_zero_tasks,
            "importance": 95
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        data_start = response.message.find("Data: {")
        data = json.loads(response.message[data_start + 6:])
        
        # Verify all Agent-Zero specific items are processed
        assert data["goals_count"] == len(agent_zero_goals)
        assert data["tasks_count"] == len(agent_zero_tasks)
        
        results = data["results"]
        assert results["goals_processed"] == len(agent_zero_goals)
        assert results["tasks_processed"] == len(agent_zero_tasks)
        
        # Verify reasoning capability gets highest priority (first in list)
        if results.get("fallback_mode") and results.get("prioritized_goals"):
            goals = results["prioritized_goals"]
            assert goals[0]["goal"] == "enhance_reasoning_capability"
            
            # Verify priority decreases appropriately
            for i in range(len(goals) - 1):
                assert goals[i]["priority"] >= goals[i + 1]["priority"]
    

@pytest.mark.skipif(not META_COGNITION_AVAILABLE, reason="Meta-cognition tool not available")
class TestAttentionAllocationPerformance:
    """Performance-focused tests for attention allocation."""
    
    @pytest.fixture
    def mock_agent(self):
        agent = Mock()
        agent.agent_name = "performance_test_agent"
        agent.get_capabilities = Mock(return_value=["attention", "reasoning"])
        agent.get_tools = Mock(return_value=[])
        return agent
    
    @pytest.fixture
    def meta_tool(self, mock_agent):
        return MetaCognitionTool(agent=mock_agent, name="meta_cognition", args={})
    
    @pytest.mark.asyncio
    async def test_scalability_benchmark(self, meta_tool):
        """Benchmark attention allocation scalability with different input sizes."""
        sizes = [10, 50, 100, 200]
        results = {}
        
        for size in sizes:
            goals = [f"scalability_goal_{i}" for i in range(size)]
            tasks = [f"scalability_task_{i}" for i in range(size)]
            
            test_params = {
                "goals": goals,
                "tasks": tasks,
                "importance": 70
            }
            
            start_time = time.time()
            response = await meta_tool.execute(operation="attention_focus", **test_params)
            end_time = time.time()
            
            execution_time = end_time - start_time
            results[size] = execution_time
            
            # Verify correctness
            data_start = response.message.find("Data: {")
            data = json.loads(response.message[data_start + 6:])
            assert data["goals_count"] == size
            assert data["tasks_count"] == size
        
        # Performance should not degrade exponentially
        # Expect roughly linear or sub-linear scaling
        assert results[200] < results[10] * 50, "Performance degrades too severely with scale"
    
    @pytest.mark.asyncio
    async def test_memory_efficiency(self, meta_tool):
        """Test that attention allocation doesn't consume excessive memory."""
        try:
            import psutil
            import os
        except ImportError:
            pytest.skip("psutil not available for memory testing")
        
        process = psutil.Process(os.getpid())
        initial_memory = process.memory_info().rss
        
        # Run multiple attention allocation operations
        for _ in range(10):
            test_params = {
                "goals": [f"memory_goal_{i}" for i in range(50)],
                "tasks": [f"memory_task_{i}" for i in range(50)],
                "importance": 80
            }
            
            await meta_tool.execute(operation="attention_focus", **test_params)
        
        final_memory = process.memory_info().rss
        memory_increase = final_memory - initial_memory
        
        # Memory increase should be reasonable (less than 100MB)
        assert memory_increase < 100 * 1024 * 1024, f"Memory usage increased by {memory_increase / 1024 / 1024:.2f}MB"


@pytest.mark.skipif(not META_COGNITION_AVAILABLE, reason="Meta-cognition tool not available")  
class TestAttentionAllocationIntegration:
    """Integration tests for attention allocation with other components."""
    
    @pytest.fixture
    def mock_agent_with_tools(self):
        agent = Mock()
        agent.agent_name = "integration_test_agent"
        
        # Mock multiple tools to test integration
        mock_tools = [
            Mock(__class__=Mock(__name__="CognitiveReasoningTool")),
            Mock(__class__=Mock(__name__="MemoryTool")),
            Mock(__class__=Mock(__name__="MetaCognitionTool"))
        ]
        agent.get_tools = Mock(return_value=mock_tools)
        agent.get_capabilities = Mock(return_value=[
            "reasoning", "memory", "meta_cognition", "learning", "adaptation"
        ])
        
        return agent
    
    @pytest.fixture
    def meta_tool(self, mock_agent_with_tools):
        return MetaCognitionTool(
            agent=mock_agent_with_tools, 
            name="meta_cognition", 
            args={}
        )
    
    @pytest.mark.asyncio
    async def test_attention_allocation_with_capability_assessment(self, meta_tool):
        """Test attention allocation considering agent capabilities."""
        # Goals aligned with agent capabilities
        capability_aligned_goals = [
            "improve_reasoning_accuracy",  # Aligns with reasoning capability
            "enhance_memory_retention",    # Aligns with memory capability
            "develop_meta_understanding"   # Aligns with meta_cognition capability
        ]
        
        test_params = {
            "goals": capability_aligned_goals,
            "tasks": ["assess_capabilities", "optimize_performance"],
            "importance": 85
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        data_start = response.message.find("Data: {")
        data = json.loads(response.message[data_start + 6:])
        
        assert data["status"] == "success"
        assert data["goals_count"] == 3
        
        results = data["results"]
        
        # Should successfully process capability-aligned goals
        assert results["goals_processed"] == 3
        assert results["attention_allocated"] is True
    
    @pytest.mark.asyncio
    async def test_multi_tool_attention_coordination(self, meta_tool):
        """Test attention allocation coordinating across multiple tools."""
        tool_specific_tasks = [
            "reasoning_tool_task",
            "memory_tool_task", 
            "meta_cognition_tool_task"
        ]
        
        test_params = {
            "goals": ["coordinate_tools_efficiently"],
            "tasks": tool_specific_tasks,
            "importance": 90
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        data_start = response.message.find("Data: {")
        data = json.loads(response.message[data_start + 6:])
        
        results = data["results"]
        
        # Should handle tool coordination tasks
        assert results["tasks_processed"] == 3
        
        if results.get("fallback_mode"):
            tasks = results["prioritized_tasks"]
            assert len(tasks) == 3
            
            # All tasks should have priorities assigned
            for task in tasks:
                assert "priority" in task
                assert "rank" in task
                assert task["priority"] > 0