#!/usr/bin/env python3
"""
Enhanced test suite for CognitiveMemoryTool with improved persistence
"""

import pytest
import asyncio
import os
import tempfile
import pickle
import json
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


def create_tool_instance(mock_agent):
    """Helper to create a properly initialized tool instance."""
    tool = CognitiveMemoryTool.__new__(CognitiveMemoryTool)
    tool.agent = mock_agent
    tool._initialize_if_needed()
    return tool


@pytest.mark.asyncio
async def test_cognitive_memory_tool_initialization(mock_agent):
    """Test that the cognitive memory tool initializes correctly."""
    tool = create_tool_instance(mock_agent)
    
    # Tool should be created regardless of OpenCog availability
    assert tool is not None
    assert tool.agent == mock_agent
    assert hasattr(tool, '_fallback_memory')
    assert hasattr(tool, 'memory_file')


@pytest.mark.asyncio
async def test_store_knowledge_basic(mock_agent):
    """Test basic knowledge storage."""
    tool = create_tool_instance(mock_agent)
    
    knowledge_data = {
        "concept": "test_concept",
        "properties": {"type": "test", "category": "example"}
    }
    
    response = await tool.execute("store", data=knowledge_data)
    
    assert "stored knowledge" in response.message.lower()
    assert "test_concept" in response.message


@pytest.mark.asyncio
async def test_retrieve_knowledge_basic(mock_agent):
    """Test basic knowledge retrieval."""
    tool = create_tool_instance(mock_agent)
    
    # Store knowledge first
    knowledge_data = {
        "concept": "retrieval_test",
        "properties": {"type": "test", "value": "42"}
    }
    
    await tool.execute("store", data=knowledge_data)
    
    # Now retrieve it
    response = await tool.execute("retrieve", data={"concept": "retrieval_test"})
    
    assert "retrieved knowledge" in response.message.lower()
    assert "retrieval_test" in response.message


@pytest.mark.asyncio
async def test_create_associations(mock_agent):
    """Test association creation."""
    tool = create_tool_instance(mock_agent)
    
    assoc_data = {
        "source": "concept_a",
        "target": "concept_b",
        "type": "related_to",
        "strength": 0.8
    }
    
    response = await tool.execute("associate", data=assoc_data)
    
    assert "created association" in response.message.lower()
    assert "concept_a" in response.message
    assert "concept_b" in response.message


@pytest.mark.asyncio
async def test_cognitive_reasoning(mock_agent):
    """Test cognitive reasoning functionality."""
    tool = create_tool_instance(mock_agent)
    
    # Store some data to reason about
    await tool.execute("store", data={
        "concept": "reasoning_test",
        "properties": {"type": "example"}
    })
    
    response = await tool.execute("reason", data={"query": "reasoning"})
    
    assert "reasoning completed" in response.message.lower()
    assert "reasoning_test" in response.message


@pytest.mark.asyncio
async def test_memory_status(mock_agent):
    """Test memory status reporting."""
    tool = create_tool_instance(mock_agent)
    
    response = await tool.execute("status")
    
    assert "memory status" in response.message.lower()
    # Should contain JSON data with status information
    assert "{" in response.message and "}" in response.message


@pytest.mark.asyncio
async def test_persistence_roundtrip(mock_agent):
    """Test complete persistence roundtrip: save and load."""
    tool1 = create_tool_instance(mock_agent)
    
    # Store complex knowledge
    knowledge_data = {
        "concept": "persistence_test",
        "properties": {"type": "test", "importance": "high"},
        "relationships": {"related_to": ["memory", "storage"]}
    }
    
    await tool1.execute("store", data=knowledge_data)
    
    # Create associations
    await tool1.execute("associate", data={
        "source": "persistence_test",
        "target": "data_integrity",
        "type": "ensures",
        "strength": 0.9
    })
    
    # Get the memory file path
    memory_file = tool1.memory_file
    assert os.path.exists(memory_file)
    
    # Create a fresh tool instance (simulating restart)
    tool2 = create_tool_instance(mock_agent)
    
    # Verify data was loaded
    response = await tool2.execute("retrieve", data={"concept": "persistence_test"})
    assert "persistence_test" in response.message
    assert "related_to" in response.message
    
    # Verify reasoning works on loaded data
    reason_response = await tool2.execute("reason", data={"query": "persistence"})
    assert "persistence_test" in reason_response.message


@pytest.mark.asyncio
async def test_invalid_operations(mock_agent):
    """Test handling of invalid operations."""
    tool = create_tool_instance(mock_agent)
    
    # Test unknown operation
    response = await tool.execute("invalid_operation")
    assert "unknown" in response.message.lower()
    
    # Test store without concept
    response = await tool.execute("store", data={"properties": {"test": "value"}})
    assert "invalid data" in response.message.lower()
    
    # Test retrieve without concept
    response = await tool.execute("retrieve", data={"other_field": "value"})
    assert "invalid data" in response.message.lower()
    
    # Test associate without required fields
    response = await tool.execute("associate", data={"source": "only_source"})
    assert "invalid data" in response.message.lower()


@pytest.mark.asyncio
async def test_memory_validation(mock_agent):
    """Test memory data validation."""
    tool = create_tool_instance(mock_agent)
    
    # Test valid data
    valid_data = {
        "atoms": [{"type": "ConceptNode", "name": "test"}],
        "metadata": {"total_atoms": 1}
    }
    assert tool._validate_memory_data(valid_data) == True
    
    # Test invalid data structures
    assert tool._validate_memory_data(None) == False
    assert tool._validate_memory_data([]) == False
    assert tool._validate_memory_data({"atoms": "not_a_list"}) == False
    assert tool._validate_memory_data({"atoms": []}) == False  # Missing metadata
    assert tool._validate_memory_data({
        "atoms": [{"no_type": "invalid"}], 
        "metadata": {}
    }) == False


def test_memory_file_corruption_handling(mock_agent, temp_memory_dir):
    """Test handling of corrupted memory files."""
    tool = create_tool_instance(mock_agent)
    
    # Create a corrupted file
    corrupted_file = os.path.join(temp_memory_dir, "corrupted.pkl")
    with open(corrupted_file, 'w') as f:
        f.write("not valid pickle data")
    
    # Should return None for corrupted file
    result = tool._load_memory_file(corrupted_file)
    assert result is None
    
    # Test with non-existent file
    result = tool._load_memory_file("non_existent.pkl")
    assert result is None


@pytest.mark.asyncio
async def test_complex_knowledge_graph(mock_agent):
    """Test building and querying a complex knowledge graph."""
    tool = create_tool_instance(mock_agent)
    
    # Build a small knowledge graph
    concepts = ["machine_learning", "neural_networks", "deep_learning", "artificial_intelligence"]
    
    for concept in concepts:
        await tool.execute("store", data={
            "concept": concept,
            "properties": {"domain": "ai", "complexity": "high"}
        })
    
    # Create associations
    associations = [
        ("machine_learning", "neural_networks", "includes"),
        ("neural_networks", "deep_learning", "enables"),
        ("deep_learning", "artificial_intelligence", "contributes_to"),
        ("artificial_intelligence", "machine_learning", "encompasses")
    ]
    
    for source, target, rel_type in associations:
        await tool.execute("associate", data={
            "source": source,
            "target": target,
            "type": rel_type,
            "strength": 0.8
        })
    
    # Test reasoning on the graph
    response = await tool.execute("reason", data={"query": "learning"})
    assert "machine_learning" in response.message
    assert "deep_learning" in response.message
    
    # Test retrieving specific concepts
    response = await tool.execute("retrieve", data={"concept": "machine_learning"})
    assert "includes" in response.message
    assert "neural_networks" in response.message


if __name__ == "__main__":
    # Run a quick test
    async def quick_test():
        mock_agent = MockAgent()
        tool = create_tool_instance(mock_agent)
        
        print("Running quick test of enhanced cognitive memory...")
        
        # Test basic operations
        await tool.execute("store", data={
            "concept": "quick_test",
            "properties": {"type": "test"}
        })
        
        result = await tool.execute("retrieve", data={"concept": "quick_test"})
        print(f"✓ Store/Retrieve: {result.message}")
        
        result = await tool.execute("status")
        print(f"✓ Status: {result.message}")
        
        print("✓ Quick test completed successfully")
    
    asyncio.run(quick_test())