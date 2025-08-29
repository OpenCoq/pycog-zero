"""
Tests for AtomSpace-Agent-Zero Tool Integration
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import the tools to test
from python.tools.atomspace_memory_bridge import AtomSpaceMemoryBridge
from python.tools.atomspace_search_engine import AtomSpaceSearchEngine
from python.tools.atomspace_document_query import AtomSpaceDocumentQuery
from python.tools.atomspace_code_context import AtomSpaceCodeContext
from python.tools.atomspace_tool_hub import AtomSpaceToolHub
from python.helpers.atomspace_memory_extension import AtomSpaceMemoryExtension, add_atomspace_extension_to_memory


class TestAtomSpaceToolIntegration:
    """Test suite for AtomSpace-Agent-Zero tool integration."""
    
    def setup_method(self):
        """Set up test fixtures."""
        # Mock agent
        self.mock_agent = Mock()
        self.mock_agent.agent_name = "test_agent"
        self.mock_agent.context = Mock()
        self.mock_agent.context.log = Mock()
        self.mock_agent.context.log.log = Mock(return_value=Mock())
        
        # Common tool arguments
        self.tool_args = {
            "name": "test_tool",
            "method": "test_method",
            "args": {},
            "message": "test message",
            "loop_data": None
        }
    
    def test_atomspace_memory_bridge_initialization(self):
        """Test AtomSpace memory bridge tool initialization."""
        tool = AtomSpaceMemoryBridge(self.mock_agent, **self.tool_args)
        assert tool is not None
        assert hasattr(tool, '_bridge_initialized')
        assert hasattr(tool, 'atomspace')
        assert hasattr(tool, 'initialized')
    
    @pytest.mark.asyncio
    async def test_atomspace_memory_bridge_execution(self):
        """Test AtomSpace memory bridge execution."""
        tool = AtomSpaceMemoryBridge(self.mock_agent, **self.tool_args)
        
        # Test without AtomSpace (fallback mode)
        response = await tool.execute(operation="bridge_memory")
        assert response is not None
        assert "bridge" in response.message.lower()
        assert response.break_loop is False
    
    def test_atomspace_search_engine_initialization(self):
        """Test AtomSpace search engine tool initialization."""
        tool = AtomSpaceSearchEngine(self.mock_agent, **self.tool_args)
        assert tool is not None
        assert hasattr(tool, '_search_initialized')
        assert hasattr(tool, 'atomspace')
        assert hasattr(tool, 'initialized')
    
    @pytest.mark.asyncio
    async def test_atomspace_search_engine_execution(self):
        """Test AtomSpace search engine execution."""
        tool = AtomSpaceSearchEngine(self.mock_agent, **self.tool_args)
        
        # Mock searxng search to avoid network calls
        with patch('python.helpers.searxng.search') as mock_search:
            mock_search.return_value = {
                "results": [
                    {"title": "Test Result", "url": "http://test.com", "content": "Test content"}
                ]
            }
            
            response = await tool.execute(query="test query")
            assert response is not None
            assert response.break_loop is False
    
    def test_atomspace_document_query_initialization(self):
        """Test AtomSpace document query tool initialization."""
        tool = AtomSpaceDocumentQuery(self.mock_agent, **self.tool_args)
        assert tool is not None
        assert hasattr(tool, '_doc_atomspace_initialized')
        assert hasattr(tool, 'atomspace')
        assert hasattr(tool, 'initialized')
    
    @pytest.mark.asyncio
    async def test_atomspace_document_query_execution(self):
        """Test AtomSpace document query execution."""
        tool = AtomSpaceDocumentQuery(self.mock_agent, **self.tool_args)
        
        # Mock DocumentQueryHelper
        with patch('python.tools.atomspace_document_query.DocumentQueryHelper') as MockHelper:
            mock_helper = MockHelper.return_value
            mock_helper.document_get_content = AsyncMock(return_value="Test document content")
            
            response = await tool.execute(document="test.txt")
            assert response is not None
            assert "content" in response.message.lower()
            assert response.break_loop is False
    
    def test_atomspace_code_context_initialization(self):
        """Test AtomSpace code context tool initialization."""
        tool = AtomSpaceCodeContext(self.mock_agent, **self.tool_args)
        assert tool is not None
        assert hasattr(tool, '_code_atomspace_initialized')
        assert hasattr(tool, 'atomspace')
        assert hasattr(tool, 'initialized')
        assert hasattr(tool, 'execution_counter')
    
    @pytest.mark.asyncio
    async def test_atomspace_code_context_execution(self):
        """Test AtomSpace code context execution."""
        tool = AtomSpaceCodeContext(self.mock_agent, **self.tool_args)
        
        # Test storing code context
        response = await tool.execute(
            operation="store_context",
            code="print('hello world')",
            result="hello world",
            language="python"
        )
        assert response is not None
        assert response.break_loop is False
    
    def test_atomspace_tool_hub_initialization(self):
        """Test AtomSpace tool hub initialization."""
        tool = AtomSpaceToolHub(self.mock_agent, **self.tool_args)
        assert tool is not None
        assert hasattr(tool, 'atomspace')
        assert hasattr(tool, 'initialized')
    
    @pytest.mark.asyncio
    async def test_atomspace_tool_hub_execution(self):
        """Test AtomSpace tool hub execution."""
        tool = AtomSpaceToolHub(self.mock_agent, **self.tool_args)
        
        # Test status operation
        response = await tool.execute(operation="status")
        assert response is not None
        assert "status" in response.message.lower()
        assert response.break_loop is False
    
    def test_atomspace_memory_extension_initialization(self):
        """Test AtomSpace memory extension initialization."""
        # Mock memory instance
        mock_memory = Mock()
        mock_memory.memory_subdir = "test"
        
        extension = AtomSpaceMemoryExtension(mock_memory)
        assert extension is not None
        assert extension.memory == mock_memory
        assert hasattr(extension, 'atomspace')
        assert hasattr(extension, 'initialized')
    
    def test_add_atomspace_extension_to_memory(self):
        """Test adding AtomSpace extension to memory instance."""
        # Mock memory instance
        mock_memory = Mock()
        mock_memory.memory_subdir = "test"
        
        enhanced_memory = add_atomspace_extension_to_memory(mock_memory)
        assert enhanced_memory == mock_memory
        assert hasattr(enhanced_memory, 'atomspace_extension')
        assert hasattr(enhanced_memory, 'create_knowledge_graph')
        assert hasattr(enhanced_memory, 'find_related_concepts')
        assert hasattr(enhanced_memory, 'get_concept_neighbors')
        assert hasattr(enhanced_memory, 'analyze_knowledge_graph')
        assert hasattr(enhanced_memory, 'save_knowledge_graph')
    
    @pytest.mark.asyncio
    async def test_tool_operations_fallback_mode(self):
        """Test that all tools work in fallback mode (without OpenCog)."""
        tools = [
            AtomSpaceMemoryBridge(self.mock_agent, **self.tool_args),
            AtomSpaceSearchEngine(self.mock_agent, **self.tool_args),
            AtomSpaceDocumentQuery(self.mock_agent, **self.tool_args),
            AtomSpaceCodeContext(self.mock_agent, **self.tool_args),
            AtomSpaceToolHub(self.mock_agent, **self.tool_args)
        ]
        
        for tool in tools:
            # All tools should initialize without errors
            assert tool is not None
            
            # All tools should handle execution gracefully
            response = await tool.execute()
            assert response is not None
            assert hasattr(response, 'message')
            assert hasattr(response, 'break_loop')
    
    def test_tool_registration_functions(self):
        """Test that all tools have proper registration functions."""
        from python.tools.atomspace_memory_bridge import register as register_bridge
        from python.tools.atomspace_search_engine import register as register_search
        from python.tools.atomspace_document_query import register as register_doc
        from python.tools.atomspace_code_context import register as register_code
        from python.tools.atomspace_tool_hub import register as register_hub
        
        # All registration functions should return tool classes
        assert register_bridge() == AtomSpaceMemoryBridge
        assert register_search() == AtomSpaceSearchEngine
        assert register_doc() == AtomSpaceDocumentQuery
        assert register_code() == AtomSpaceCodeContext
        assert register_hub() == AtomSpaceToolHub
    
    @pytest.mark.asyncio
    async def test_tool_error_handling(self):
        """Test error handling in tools."""
        tool = AtomSpaceMemoryBridge(self.mock_agent, **self.tool_args)
        
        # Test with invalid operation
        response = await tool.execute(operation="invalid_operation")
        assert response is not None
        assert "unknown" in response.message.lower()
        assert response.break_loop is False
    
    @pytest.mark.asyncio
    async def test_cross_tool_integration(self):
        """Test integration between different AtomSpace tools."""
        # Create multiple tools
        memory_bridge = AtomSpaceMemoryBridge(self.mock_agent, **self.tool_args)
        tool_hub = AtomSpaceToolHub(self.mock_agent, **self.tool_args)
        
        # Test hub status with other tools present
        hub_response = await tool_hub.execute(operation="status")
        assert hub_response is not None
        
        # Test memory bridge with hub coordination
        memory_response = await memory_bridge.execute(operation="bridge_memory")
        assert memory_response is not None
    
    def test_atomspace_availability_handling(self):
        """Test handling of AtomSpace availability across tools."""
        tools = [
            AtomSpaceMemoryBridge(self.mock_agent, **self.tool_args),
            AtomSpaceSearchEngine(self.mock_agent, **self.tool_args),
            AtomSpaceDocumentQuery(self.mock_agent, **self.tool_args),
            AtomSpaceCodeContext(self.mock_agent, **self.tool_args),
            AtomSpaceToolHub(self.mock_agent, **self.tool_args)
        ]
        
        # All tools should handle AtomSpace unavailability gracefully
        for tool in tools:
            # Tools should not crash when AtomSpace is unavailable
            assert tool is not None
            # initialized flag should reflect AtomSpace availability
            # (Will be False in test environment without OpenCog)
            assert hasattr(tool, 'initialized')


class TestAtomSpaceMemoryExtension:
    """Test suite for AtomSpace memory extension."""
    
    def setup_method(self):
        """Set up test fixtures."""
        self.mock_memory = Mock()
        self.mock_memory.memory_subdir = "test"
        self.mock_memory.db = Mock()
        self.mock_memory.db.get_all_docs = Mock(return_value={})
    
    def test_extension_initialization(self):
        """Test memory extension initialization."""
        extension = AtomSpaceMemoryExtension(self.mock_memory)
        assert extension is not None
        assert extension.memory == self.mock_memory
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_creation(self):
        """Test knowledge graph creation from memory."""
        extension = AtomSpaceMemoryExtension(self.mock_memory)
        
        # Mock memory documents
        mock_doc = Mock()
        mock_doc.page_content = "This is a test document with important content"
        mock_doc.metadata = {"source": "test", "type": "document"}
        
        self.mock_memory.db.get_all_docs.return_value = {"doc1": mock_doc}
        
        # Test knowledge graph creation
        result = await extension.create_knowledge_graph_from_memory()
        
        # Should complete without error (even if AtomSpace unavailable)
        assert isinstance(result, bool)
    
    def test_concept_finding(self):
        """Test finding related concepts."""
        extension = AtomSpaceMemoryExtension(self.mock_memory)
        
        # Test finding concepts (should work even without AtomSpace)
        related = extension.find_related_concepts("test")
        assert isinstance(related, list)
    
    def test_neighbor_analysis(self):
        """Test concept neighbor analysis."""
        extension = AtomSpaceMemoryExtension(self.mock_memory)
        
        # Test neighbor analysis (should work even without AtomSpace)
        neighbors = extension.get_concept_neighbors("test")
        assert isinstance(neighbors, dict)
    
    def test_graph_structure_analysis(self):
        """Test knowledge graph structure analysis."""
        extension = AtomSpaceMemoryExtension(self.mock_memory)
        
        # Test structure analysis (should work even without AtomSpace)
        analysis = extension.analyze_knowledge_graph_structure()
        assert isinstance(analysis, dict)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])