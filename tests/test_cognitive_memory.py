"""
Test suite for CognitiveMemoryTool
"""

import pytest
import asyncio
import os
import tempfile
from unittest.mock import MagicMock

# Import the cognitive memory tool
from python.tools.cognitive_memory import CognitiveMemoryTool


class MockAgent:
    """Mock Agent-Zero instance for testing."""
    def __init__(self):
        self.capabilities = ["cognitive_memory", "reasoning", "persistence"]
        self.tools = []
    
    def get_capabilities(self):
        return self.capabilities
    
    def get_tools(self):
        return self.tools


@pytest.fixture
def mock_agent():
    return MockAgent()


@pytest.fixture
def temp_memory_dir():
    """Create a temporary directory for memory files during testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


@pytest.mark.asyncio
async def test_cognitive_memory_tool_initialization(mock_agent):
    """Test that the cognitive memory tool initializes correctly."""
    tool = CognitiveMemoryTool(mock_agent)
    
    # Tool should be created regardless of OpenCog availability
    assert tool is not None
    assert tool.agent == mock_agent
    

@pytest.mark.asyncio
async def test_store_knowledge_invalid_data(mock_agent):
    """Test storing knowledge with invalid data."""
    tool = CognitiveMemoryTool(mock_agent)
    
    # Test with no data
    response = await tool.execute("store", data=None)
    assert "Invalid data" in response.message
    
    # Test with missing concept
    response = await tool.execute("store", data={"properties": {"type": "test"}})
    assert "Invalid data" in response.message


@pytest.mark.asyncio
async def test_store_knowledge_fallback_mode(mock_agent):
    """Test storing knowledge when OpenCog is not available (fallback mode)."""
    tool = CognitiveMemoryTool(mock_agent)
    
    knowledge_data = {
        "concept": "machine_learning",
        "properties": {
            "type": "AI_technique",
            "applications": "pattern_recognition"
        }
    }
    
    response = await tool.execute("store", data=knowledge_data)
    
    # Should work even without OpenCog (fallback mode)
    assert "machine_learning" in response.message.lower() or "fallback" in response.message.lower()


@pytest.mark.asyncio
async def test_retrieve_knowledge_invalid_data(mock_agent):
    """Test retrieving knowledge with invalid data."""
    tool = CognitiveMemoryTool(mock_agent)
    
    # Test with no data
    response = await tool.execute("retrieve", data=None)
    assert "Invalid data" in response.message
    
    # Test with missing concept
    response = await tool.execute("retrieve", data={"properties": {"type": "test"}})
    assert "Invalid data" in response.message


@pytest.mark.asyncio
async def test_retrieve_knowledge_fallback_mode(mock_agent):
    """Test retrieving knowledge when OpenCog is not available."""
    tool = CognitiveMemoryTool(mock_agent)
    
    response = await tool.execute("retrieve", data={"concept": "machine_learning"})
    
    # Should handle gracefully even without OpenCog
    assert "fallback" in response.message.lower() or "not found" in response.message.lower()


@pytest.mark.asyncio
async def test_create_associations_invalid_data(mock_agent):
    """Test creating associations with invalid data."""
    tool = CognitiveMemoryTool(mock_agent)
    
    # Test with no data
    response = await tool.execute("associate", data=None)
    assert "Invalid data" in response.message
    
    # Test with missing source
    response = await tool.execute("associate", data={"target": "AI"})
    assert "Invalid data" in response.message
    
    # Test with missing target
    response = await tool.execute("associate", data={"source": "ML"})
    assert "Invalid data" in response.message


@pytest.mark.asyncio
async def test_create_associations_fallback_mode(mock_agent):
    """Test creating associations when OpenCog is not available."""
    tool = CognitiveMemoryTool(mock_agent)
    
    association_data = {
        "source": "machine_learning",
        "target": "artificial_intelligence",
        "type": "subset_of",
        "strength": 0.8
    }
    
    response = await tool.execute("associate", data=association_data)
    
    # Should handle gracefully even without OpenCog
    assert "not available" in response.message.lower() or "association" in response.message.lower()


@pytest.mark.asyncio
async def test_cognitive_reasoning_fallback_mode(mock_agent):
    """Test cognitive reasoning when OpenCog is not available."""
    tool = CognitiveMemoryTool(mock_agent)
    
    response = await tool.execute("reason", data={"query": "What is machine learning?"})
    
    # Should handle gracefully even without OpenCog
    assert "not available" in response.message.lower() or "reasoning" in response.message.lower()


@pytest.mark.asyncio
async def test_unknown_operation(mock_agent):
    """Test handling of unknown operations."""
    tool = CognitiveMemoryTool(mock_agent)
    
    response = await tool.execute("invalid_operation", data={})
    
    assert "Unknown cognitive memory operation" in response.message
    assert "store, retrieve, associate, reason" in response.message


def test_memory_file_path(mock_agent):
    """Test that memory file path is correctly set."""
    tool = CognitiveMemoryTool(mock_agent)
    
    assert tool.memory_file.endswith("memory/cognitive_atomspace.pkl")


def test_serialize_atomspace_without_opencog(mock_agent):
    """Test AtomSpace serialization when OpenCog is not available."""
    tool = CognitiveMemoryTool(mock_agent)
    
    # Should return empty dict when not initialized
    result = tool.serialize_atomspace()
    assert result == {}


def test_restore_atomspace_without_opencog(mock_agent):
    """Test AtomSpace restoration when OpenCog is not available."""
    tool = CognitiveMemoryTool(mock_agent)
    
    # Should handle gracefully when not initialized
    tool.restore_atomspace({"atoms": []})
    # No exception should be raised


def test_persistent_memory_operations_without_opencog(mock_agent):
    """Test persistent memory operations when OpenCog is not available."""
    tool = CognitiveMemoryTool(mock_agent)
    
    # These should not raise exceptions
    tool.load_persistent_memory()
    tool.save_persistent_memory()


@pytest.mark.asyncio
async def test_comprehensive_workflow(mock_agent):
    """Test a complete workflow of operations."""
    tool = CognitiveMemoryTool(mock_agent)
    
    # Store some knowledge
    store_response = await tool.execute("store", data={
        "concept": "neural_networks",
        "properties": {
            "type": "machine_learning_model",
            "inspired_by": "biological_neurons"
        },
        "relationships": {
            "used_in": ["deep_learning", "computer_vision"]
        }
    })
    
    # Retrieve the knowledge
    retrieve_response = await tool.execute("retrieve", data={
        "concept": "neural_networks"
    })
    
    # Create an association
    associate_response = await tool.execute("associate", data={
        "source": "neural_networks",
        "target": "deep_learning",
        "type": "enables",
        "strength": 0.9
    })
    
    # Perform reasoning
    reason_response = await tool.execute("reason", data={
        "query": "neural networks"
    })
    
    # All operations should complete without errors
    assert store_response is not None
    assert retrieve_response is not None
    assert associate_response is not None
    assert reason_response is not None


if __name__ == "__main__":
    # Run a simple test if executed directly
    async def main():
        mock_agent = MockAgent()
        tool = CognitiveMemoryTool(mock_agent)
        
        print("Testing cognitive memory tool...")
        
        # Test basic operations
        response = await tool.execute("store", data={
            "concept": "test_concept",
            "properties": {"type": "test"}
        })
        print(f"Store test: {response.message}")
        
        response = await tool.execute("retrieve", data={"concept": "test_concept"})
        print(f"Retrieve test: {response.message}")
        
        print("✓ Basic tests completed")
    
    asyncio.run(main())