"""
Test suite for MetaCognitionTool - Meta-cognitive self-reflection capabilities
Tests self-reflection, attention allocation, goal prioritization, and deep introspection
"""

import pytest
import asyncio
import json
import time
from unittest.mock import Mock, patch, AsyncMock
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
class TestMetaCognitionTool:
    """Test suite for meta-cognitive self-reflection capabilities."""
    
    @pytest.fixture
    def mock_agent(self):
        """Create a mock Agent-Zero agent for testing."""
        agent = Mock()
        agent.agent_name = "test_agent"
        agent.get_capabilities = Mock(return_value=["reasoning", "memory", "learning"])
        agent.get_tools = Mock(return_value=[Mock(__class__=Mock(__name__="TestTool"))])
        return agent
    
    @pytest.fixture
    def meta_tool(self, mock_agent):
        """Create MetaCognitionTool instance for testing."""
        return MetaCognitionTool(
            agent=mock_agent,
            name="meta_cognition",
            args={}
        )
    
    def test_meta_cognition_tool_initialization(self, mock_agent):
        """Test that MetaCognitionTool initializes correctly."""
        tool = MetaCognitionTool(
            agent=mock_agent,
            name="meta_cognition",
            args={}
        )
        
        assert tool.agent == mock_agent
        assert tool.name == "meta_cognition"
        assert tool.instance_id > 0
        assert tool.config is not None
        assert "meta_cognitive" in tool.config
        assert tool.last_self_description is None
        assert tool.attention_history == []
        assert tool.goal_priorities == {}
    
    def test_meta_level_calculation(self, meta_tool):
        """Test meta-cognitive level calculation."""
        meta_level = meta_tool._calculate_meta_level()
        assert isinstance(meta_level, (int, float))
        assert 1.0 <= meta_level <= 5.0
    
    @pytest.mark.asyncio
    async def test_self_reflection_basic(self, meta_tool):
        """Test basic self-reflection functionality."""
        response = await meta_tool.execute(operation="self_reflect")
        
        assert isinstance(response, Response)
        assert "self-description" in response.message.lower() or "Generated meta-cognitive" in response.message
        assert "Data:" in response.message
        assert not response.break_loop
        
        # Verify data structure
        data_start = response.message.find("Data: {")
        if data_start != -1:
            data_json = response.message[data_start + 6:]
            data = json.loads(data_json)
            assert "operation" in data
            assert "agent_state" in data
            assert "status" in data
    
    @pytest.mark.asyncio
    async def test_attention_allocation(self, meta_tool):
        """Test attention allocation functionality."""
        test_params = {
            "goals": ["learn_new_skills", "improve_performance", "solve_complex_problems"],
            "tasks": ["analyze_data", "generate_report", "optimize_workflow"],
            "importance": 80
        }
        
        response = await meta_tool.execute(operation="attention_focus", **test_params)
        
        assert isinstance(response, Response)
        assert "attention allocated" in response.message.lower()
        assert not response.break_loop
        
        # Verify response data
        data_start = response.message.find("Data: {")
        if data_start != -1:
            data_json = response.message[data_start + 6:]
            data = json.loads(data_json)
            assert data["operation"] == "attention_focus"
            assert data["goals_count"] == 3
            assert data["tasks_count"] == 3
            assert data["status"] == "success"
    
    @pytest.mark.asyncio
    async def test_goal_prioritization(self, meta_tool):
        """Test goal prioritization functionality."""
        test_goals = [
            "improve cognitive capabilities",
            "enhance learning speed",
            "develop better memory",
            "increase problem-solving accuracy",
            "expand knowledge base"
        ]
        
        response = await meta_tool.execute(operation="goal_prioritize", goals=test_goals)
        
        assert isinstance(response, Response)
        assert "prioritized" in response.message.lower()
        assert not response.break_loop
        
        # Verify prioritization results
        data_start = response.message.find("Data: {")
        if data_start != -1:
            data_json = response.message[data_start + 6:]
            data = json.loads(data_json)
            assert data["operation"] == "goal_prioritize"
            assert len(data["prioritized_goals"]) == len(test_goals)
            assert "top_3_goals" in data
            
            # Check that goals are properly ranked
            prioritized = data["prioritized_goals"]
            for i, goal in enumerate(prioritized[:-1]):
                # Each goal should have required fields
                assert "goal" in goal
                assert "priority_score" in goal
                assert "final_rank" in goal
                # Priority should generally decrease (allowing for ties)
                assert goal["priority_score"] >= prioritized[i + 1]["priority_score"] - 0.01
    
    @pytest.mark.asyncio
    async def test_deep_introspection(self, meta_tool):
        """Test deep introspection functionality."""
        response = await meta_tool.execute(operation="introspect", introspection_depth=2)
        
        assert isinstance(response, Response)
        assert "deep introspection" in response.message.lower()
        assert not response.break_loop
        
        # Verify introspection data structure
        data_start = response.message.find("Data: {")
        if data_start != -1:
            data_json = response.message[data_start + 6:]
            data = json.loads(data_json)
            assert data["operation"] == "introspect"
            assert "introspection_summary" in data
            assert "full_analysis" in data
            
            # Check analysis components
            analysis = data["full_analysis"]
            assert "timestamp" in analysis
            assert "current_state" in analysis
            assert "behavioral_patterns" in analysis
            assert "learning_assessment" in analysis
            assert "recommendations" in analysis
    
    @pytest.mark.asyncio
    async def test_status_retrieval(self, meta_tool):
        """Test meta-cognitive status retrieval."""
        response = await meta_tool.execute(operation="status")
        
        assert isinstance(response, Response)
        assert "status retrieved" in response.message.lower()
        assert not response.break_loop
        
        # Verify status data
        data_start = response.message.find("Data: {")
        if data_start != -1:
            data_json = response.message[data_start + 6:]
            data = json.loads(data_json)
            assert data["operation"] == "status"
            assert "status" in data
            
            status = data["status"]
            assert "system_status" in status
            assert "capabilities_status" in status
            assert "state_summary" in status
            assert "configuration" in status
    
    @pytest.mark.asyncio
    async def test_recursive_self_analysis(self, meta_tool):
        """Test recursive self-analysis with different depths."""
        # Test with depth 1
        response1 = await meta_tool.execute(operation="self_reflect", recursive_depth=1)
        data1_start = response1.message.find("Data: {")
        if data1_start != -1:
            data1 = json.loads(response1.message[data1_start + 6:])
            assert data1["recursive_depth"] == 1
        
        # Test with depth 3
        response3 = await meta_tool.execute(operation="self_reflect", recursive_depth=3)
        data3_start = response3.message.find("Data: {")
        if data3_start != -1:
            data3 = json.loads(response3.message[data3_start + 6:])
            assert data3["recursive_depth"] == 3
            # Deeper analysis should have more complex structure
            if "agent_state" in data3 and "recursive_analysis" in data3["agent_state"]:
                assert "deeper_analysis" in data3["agent_state"]["recursive_analysis"]
    
    @pytest.mark.asyncio
    async def test_agent_state_collection(self, meta_tool):
        """Test comprehensive agent state collection."""
        agent_state = await meta_tool._collect_agent_state()
        
        # Verify required state components
        assert "agent_id" in agent_state
        assert "timestamp" in agent_state
        assert "meta_level" in agent_state
        assert "capabilities" in agent_state
        assert "active_tools" in agent_state
        assert "memory_usage" in agent_state
        assert "attention_allocation" in agent_state
        assert "cognitive_load" in agent_state
        assert "performance_metrics" in agent_state
        assert "tool_integration_status" in agent_state
        
        # Verify data types
        assert isinstance(agent_state["capabilities"], list)
        assert isinstance(agent_state["active_tools"], list)
        assert isinstance(agent_state["memory_usage"], dict)
        assert isinstance(agent_state["cognitive_load"], dict)
        assert isinstance(agent_state["performance_metrics"], dict)
    
    def test_cognitive_load_assessment(self, meta_tool):
        """Test cognitive load assessment."""
        # Mock large atomspace for testing
        with patch.object(meta_tool, 'initialized', True), \
             patch.object(meta_tool, 'atomspace', Mock()):
            meta_tool.atomspace.__len__ = Mock(return_value=5000)  # Medium size
            
            cognitive_load = asyncio.run(meta_tool._assess_cognitive_load())
            
            assert "level" in cognitive_load
            assert cognitive_load["level"] in ["low", "medium", "high"]
            assert "load_score" in cognitive_load
            assert 0.0 <= cognitive_load["load_score"] <= 1.0
            assert "factors" in cognitive_load
            assert isinstance(cognitive_load["factors"], list)
    
    def test_attention_distribution(self, meta_tool):
        """Test attention distribution calculation."""
        attention_dist = asyncio.run(meta_tool._get_attention_distribution())
        
        assert "total_sti_available" in attention_dist
        assert "focused_concepts" in attention_dist
        assert "attention_entropy" in attention_dist
        assert "top_attended_items" in attention_dist
        assert isinstance(attention_dist["focused_concepts"], list)
        assert isinstance(attention_dist["top_attended_items"], list)
    
    @pytest.mark.asyncio
    async def test_behavioral_pattern_analysis(self, meta_tool):
        """Test behavioral pattern analysis."""
        # Add some mock attention history
        meta_tool.attention_history = [
            {
                "timestamp": time.time() - 300,
                "goals": ["learn", "adapt"],
                "tasks": ["analyze", "optimize"],
                "distribution": {}
            },
            {
                "timestamp": time.time() - 200,
                "goals": ["learn", "improve"],
                "tasks": ["analyze", "report"],
                "distribution": {}
            }
        ]
        
        patterns = await meta_tool._analyze_behavioral_patterns()
        
        assert "attention_patterns" in patterns
        assert "goal_prioritization_patterns" in patterns
        assert "tool_usage_patterns" in patterns
        assert "performance_trends" in patterns
        
        # Check that attention patterns were analyzed
        if patterns["attention_patterns"]:
            assert "goal" in patterns["attention_patterns"][0]
            assert "frequency" in patterns["attention_patterns"][0]
    
    @pytest.mark.asyncio
    async def test_learning_assessment(self, meta_tool):
        """Test learning capacity assessment."""
        learning_assessment = await meta_tool._assess_learning_capacity()
        
        assert "learning_score" in learning_assessment
        assert "adaptation_indicators" in learning_assessment
        assert "knowledge_growth" in learning_assessment
        assert "meta_learning_capability" in learning_assessment
        
        assert 0.0 <= learning_assessment["learning_score"] <= 1.0
        assert isinstance(learning_assessment["adaptation_indicators"], list)
        assert isinstance(learning_assessment["meta_learning_capability"], bool)
    
    @pytest.mark.asyncio
    async def test_improvement_recommendations(self, meta_tool):
        """Test self-improvement recommendation generation."""
        # Mock agent state with various conditions
        mock_agent_state = {
            "cognitive_load": {"level": "high", "load_score": 0.8},
            "performance_metrics": {"error_rate": 0.15},
            "meta_level": 2.5,
            "tool_integration_status": {"integration_quality": "low"}
        }
        
        mock_patterns = {"attention_patterns": []}
        mock_learning = {"learning_score": 0.3}
        
        recommendations = await meta_tool._generate_improvement_recommendations(
            mock_agent_state, mock_patterns, mock_learning
        )
        
        assert isinstance(recommendations, list)
        
        if recommendations:  # If recommendations were generated
            for rec in recommendations:
                assert "category" in rec
                assert "priority" in rec
                assert "recommendation" in rec
                assert "rationale" in rec
                assert rec["priority"] in ["low", "medium", "high"]
    
    def test_fallback_attention_allocation(self, meta_tool):
        """Test fallback attention allocation without ECAN."""
        goals = ["goal1", "goal2", "goal3"]
        tasks = ["task1", "task2"]
        importance = 100
        
        result = asyncio.run(meta_tool._fallback_attention_allocation(goals, tasks, importance))
        
        assert "attention_allocated" in result
        assert "fallback_mode" in result
        assert "prioritized_goals" in result
        assert "prioritized_tasks" in result
        assert result["attention_allocated"] is True
        assert result["fallback_mode"] is True
        
        # Check goal prioritization
        assert len(result["prioritized_goals"]) == len(goals)
        for goal in result["prioritized_goals"]:
            assert "goal" in goal
            assert "priority" in goal
            assert "rank" in goal
    
    @pytest.mark.asyncio
    async def test_error_handling(self, meta_tool):
        """Test error handling in meta-cognitive operations."""
        # Test with invalid operation
        response = await meta_tool.execute(operation="invalid_operation")
        
        # Should default to self-reflection
        assert isinstance(response, Response)
        assert not response.break_loop
        
        # Test with None goals for prioritization
        response = await meta_tool.execute(operation="goal_prioritize", goals=None)
        assert "No goals provided" in response.message
    
    @pytest.mark.asyncio
    async def test_atomspace_integration(self, meta_tool):
        """Test AtomSpace integration when available."""
        if meta_tool.initialized and meta_tool.atomspace:
            # Test storing self-description
            mock_agent_state = {
                "meta_level": 3.0,
                "cognitive_load": "medium",
                "timestamp": time.time()
            }
            
            await meta_tool._store_self_description_in_atomspace(mock_agent_state)
            
            # Verify atoms were created (basic check)
            assert len(meta_tool.atomspace) > 0
    
    def test_configuration_loading(self, meta_tool):
        """Test cognitive configuration loading."""
        config = meta_tool.config
        
        assert isinstance(config, dict)
        assert "meta_cognitive" in config
        
        meta_config = config["meta_cognitive"]
        assert "self_reflection_enabled" in meta_config
        assert "attention_allocation_enabled" in meta_config
        assert "goal_prioritization_enabled" in meta_config
        assert "recursive_depth" in meta_config
        assert isinstance(meta_config["recursive_depth"], int)
        assert meta_config["recursive_depth"] >= 1


@pytest.mark.skipif(not META_COGNITION_AVAILABLE, reason="Meta-cognition tool not available")
class TestMetaCognitionIntegration:
    """Integration tests for meta-cognition with other systems."""
    
    @pytest.fixture
    def mock_agent_with_tools(self):
        """Create mock agent with multiple tools."""
        agent = Mock()
        agent.agent_name = "integration_test_agent"
        
        # Mock multiple tools
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
    
    def test_multiple_instances_shared_atomspace(self, mock_agent_with_tools):
        """Test that multiple MetaCognitionTool instances share atomspace."""
        tool1 = MetaCognitionTool(mock_agent_with_tools, "meta1", args={})
        tool2 = MetaCognitionTool(mock_agent_with_tools, "meta2", args={})
        
        # Both tools should have different instance IDs
        assert tool1.instance_id != tool2.instance_id
        
        # If both are initialized, they should share atomspace
        if tool1.initialized and tool2.initialized:
            assert tool1.atomspace is tool2.atomspace
    
    @pytest.mark.asyncio
    async def test_cross_tool_capability_assessment(self, mock_agent_with_tools):
        """Test capability assessment with multiple tools."""
        tool = MetaCognitionTool(mock_agent_with_tools, "meta", args={})
        
        capabilities = await tool._get_agent_capabilities()
        
        # Should include capabilities from mock agent
        assert "reasoning" in capabilities
        assert "memory" in capabilities
        assert "meta_cognition" in capabilities
        
        # Should include meta-cognitive capabilities
        assert "meta_cognitive_reflection" in capabilities
        assert "self_description_generation" in capabilities


@pytest.mark.skipif(not META_COGNITION_AVAILABLE, reason="Meta-cognition tool not available")
class TestMetaCognitionPerformance:
    """Performance and stress tests for meta-cognition."""
    
    @pytest.mark.asyncio
    async def test_large_goal_prioritization(self):
        """Test prioritization with large number of goals."""
        mock_agent = Mock()
        mock_agent.agent_name = "perf_test_agent"
        mock_agent.get_capabilities = Mock(return_value=["test"])
        mock_agent.get_tools = Mock(return_value=[])
        
        tool = MetaCognitionTool(mock_agent, "meta", args={})
        
        # Create 50 goals
        large_goal_list = [f"goal_{i}" for i in range(50)]
        
        start_time = time.time()
        response = await tool.execute(operation="goal_prioritize", goals=large_goal_list)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time (< 5 seconds)
        assert execution_time < 5.0
        assert isinstance(response, Response)
        assert "prioritized" in response.message.lower()
    
    @pytest.mark.asyncio
    async def test_deep_recursive_analysis(self):
        """Test deep recursive analysis performance."""
        mock_agent = Mock()
        mock_agent.agent_name = "recursive_test_agent"
        mock_agent.get_capabilities = Mock(return_value=["test"])
        mock_agent.get_tools = Mock(return_value=[])
        
        tool = MetaCognitionTool(mock_agent, "meta", args={})
        
        # Test with maximum recursive depth
        start_time = time.time()
        response = await tool.execute(operation="self_reflect", recursive_depth=5)
        execution_time = time.time() - start_time
        
        # Should complete within reasonable time
        assert execution_time < 10.0
        assert isinstance(response, Response)
    
    def test_memory_usage_monitoring(self):
        """Test that memory usage monitoring works correctly."""
        mock_agent = Mock()
        mock_agent.agent_name = "memory_test_agent"
        mock_agent.get_capabilities = Mock(return_value=["test"])
        mock_agent.get_tools = Mock(return_value=[])
        
        tool = MetaCognitionTool(mock_agent, "meta", args={})
        
        memory_stats = asyncio.run(tool._get_memory_statistics())
        
        # Should have basic memory information
        assert "atomspace_size" in memory_stats
        assert "total_memory_mb" in memory_stats
        assert isinstance(memory_stats["atomspace_size"], int)


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])