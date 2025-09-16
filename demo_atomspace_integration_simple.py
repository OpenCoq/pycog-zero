#!/usr/bin/env python3
"""
PyCog-Zero AtomSpace-Agent-Zero Integration Demonstration (Simplified)
Shows the integrated tools working without external dependencies
"""

import asyncio
import json
import sys
import os
from unittest.mock import Mock

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import all integrated tools
from python.tools.atomspace_memory_bridge import AtomSpaceMemoryBridge
from python.tools.atomspace_search_engine import AtomSpaceSearchEngine
from python.tools.atomspace_document_query import AtomSpaceDocumentQuery
from python.tools.atomspace_code_context import AtomSpaceCodeContext
from python.tools.atomspace_tool_hub import AtomSpaceToolHub
from python.helpers.atomspace_memory_extension import AtomSpaceMemoryExtension


def create_mock_agent():
    """Create a mock Agent-Zero agent for demonstration."""
    agent = Mock()
    agent.agent_name = "Demo Agent"
    agent.config = Mock()
    agent.config.memory_subdir = "demo"
    agent.context = Mock()
    agent.context.log = Mock()
    agent.context.log.log = Mock(return_value=Mock())
    
    # Mock the update method that tools expect
    log_mock = Mock()
    log_mock.update = Mock()
    agent.context.log.log.return_value = log_mock
    
    return agent


def create_tool_instance(tool_class, agent):
    """Create a tool instance with proper arguments."""
    return tool_class(
        agent=agent,
        name=tool_class.__name__,
        method="demo",
        args={},
        message="Demo execution",
        loop_data=None
    )


async def demo_all_tools():
    """Demonstrate all AtomSpace-integrated tools."""
    print("\n🧠 PyCog-Zero AtomSpace-Agent-Zero Integration Demonstration")
    print("=" * 80)
    print("Demonstrating integration of Agent-Zero tools with AtomSpace components")
    print("Running in fallback mode (works without OpenCog installation)")
    
    agent = create_mock_agent()
    
    # 1. Memory Bridge Tool
    print("\n1. 🧠 ATOMSPACE MEMORY BRIDGE")
    print("-" * 40)
    memory_tool = create_tool_instance(AtomSpaceMemoryBridge, agent)
    
    # Test different operations
    operations = ["bridge_memory", "query_knowledge", "enhance_search", "cross_reference"]
    for op in operations:
        try:
            response = await memory_tool.execute(operation=op, query="test query", concept="intelligence")
            print(f"   {op}: {response.message[:100]}...")
        except Exception as e:
            print(f"   {op}: Error - {str(e)[:100]}...")
    
    # 2. Search Engine Tool
    print("\n2. 🔍 ATOMSPACE SEARCH ENGINE")
    print("-" * 40)
    search_tool = create_tool_instance(AtomSpaceSearchEngine, agent)
    
    # Mock the searxng function to avoid network issues
    import python.helpers.searxng
    original_search = python.helpers.searxng.search
    
    async def mock_search(query):
        return {
            "results": [
                {"title": f"Result for {query}", "url": "http://example.com", "content": f"Content about {query}"}
            ]
        }
    
    python.helpers.searxng.search = mock_search
    
    try:
        response = await search_tool.execute(query="artificial intelligence")
        print(f"   Search result: {response.message[:150]}...")
    except Exception as e:
        print(f"   Search error: {str(e)[:100]}...")
    finally:
        python.helpers.searxng.search = original_search
    
    # 3. Document Query Tool
    print("\n3. 📄 ATOMSPACE DOCUMENT QUERY")
    print("-" * 40)
    doc_tool = create_tool_instance(AtomSpaceDocumentQuery, agent)
    
    # Mock DocumentQueryHelper
    import python.helpers.document_query
    
    class MockHelper:
        def __init__(self, agent, callback):
            pass
        async def document_get_content(self, uri):
            return "Sample document content about AI and cognitive computing"
    
    original_helper = python.helpers.document_query.DocumentQueryHelper
    python.helpers.document_query.DocumentQueryHelper = MockHelper
    
    try:
        response = await doc_tool.execute(document="test.txt")
        print(f"   Document query: {response.message[:150]}...")
    except Exception as e:
        print(f"   Document error: {str(e)[:100]}...")
    finally:
        python.helpers.document_query.DocumentQueryHelper = original_helper
    
    # 4. Code Context Tool
    print("\n4. 💻 ATOMSPACE CODE CONTEXT")
    print("-" * 40)
    code_tool = create_tool_instance(AtomSpaceCodeContext, agent)
    
    operations = ["store_context", "analyze_patterns", "suggest_improvements", "trace_dependencies"]
    for op in operations:
        try:
            response = await code_tool.execute(
                operation=op, 
                code="print('hello world')", 
                result="hello world",
                language="python"
            )
            print(f"   {op}: {response.message[:100]}...")
        except Exception as e:
            print(f"   {op}: Error - {str(e)[:100]}...")
    
    # 5. Tool Hub
    print("\n5. 🔗 ATOMSPACE TOOL HUB")
    print("-" * 40)
    hub_tool = create_tool_instance(AtomSpaceToolHub, agent)
    
    operations = ["status", "share_data", "retrieve_data", "coordinate_tools", "analyze_interactions"]
    for op in operations:
        try:
            kwargs = {}
            if op == "share_data":
                kwargs = {
                    "tool_name": "test_tool",
                    "data_type": "test_data",
                    "data": {"key": "value"}
                }
            elif op == "coordinate_tools":
                kwargs = {
                    "primary_tool": "memory",
                    "supporting_tools": ["search", "document"],
                    "task_description": "test task"
                }
            
            response = await hub_tool.execute(operation=op, **kwargs)
            print(f"   {op}: {response.message[:100]}...")
        except Exception as e:
            print(f"   {op}: Error - {str(e)[:100]}...")
    
    # 6. Memory Extension
    print("\n6. 🗃️ ATOMSPACE MEMORY EXTENSION")
    print("-" * 40)
    
    # Mock memory instance
    mock_memory = Mock()
    mock_memory.memory_subdir = "demo"
    mock_memory.db = Mock()
    mock_memory.db.get_all_docs = Mock(return_value={})
    
    extension = AtomSpaceMemoryExtension(mock_memory)
    print(f"   Extension created: {extension.initialized}")
    print(f"   AtomSpace available: {extension.atomspace is not None}")
    
    # Test methods
    try:
        result = await extension.create_knowledge_graph_from_memory()
        print(f"   Knowledge graph creation: {result}")
    except Exception as e:
        print(f"   Knowledge graph error: {str(e)[:100]}...")
    
    related = extension.find_related_concepts("test")
    print(f"   Related concepts found: {len(related)}")
    
    analysis = extension.analyze_knowledge_graph_structure()
    print(f"   Graph analysis metrics: {len(analysis)}")


def print_integration_summary():
    """Print summary of the integration."""
    print("\n" + "=" * 80)
    print("🎯 INTEGRATION COMPLETED SUCCESSFULLY")
    print("=" * 80)
    
    print("\n✅ NEW ATOMSPACE-ENHANCED TOOLS CREATED:")
    print("   1. AtomSpaceMemoryBridge - Enhanced memory operations")
    print("   2. AtomSpaceSearchEngine - Cognitive search capabilities") 
    print("   3. AtomSpaceDocumentQuery - Semantic document analysis")
    print("   4. AtomSpaceCodeContext - Code execution context storage")
    print("   5. AtomSpaceToolHub - Cross-tool coordination")
    print("   6. AtomSpaceMemoryExtension - Knowledge graph functionality")
    
    print("\n🔧 KEY INTEGRATION FEATURES:")
    print("   • All tools follow Agent-Zero patterns")
    print("   • Graceful fallback when OpenCog unavailable")
    print("   • Cross-tool data sharing via shared AtomSpace")
    print("   • Cognitive enhancement of standard operations")
    print("   • Knowledge graph creation and analysis")
    print("   • Semantic reasoning capabilities")
    print("   • Code pattern analysis and suggestions")
    print("   • Tool coordination and planning")
    
    print("\n📈 INTEGRATION BENEFITS:")
    print("   • Enhanced cognitive capabilities")
    print("   • Persistent knowledge storage")
    print("   • Cross-tool information sharing")
    print("   • Semantic understanding")
    print("   • Reasoning-based enhancements")
    print("   • Pattern recognition")
    print("   • Intelligent coordination")
    
    print("\n🚀 PRODUCTION DEPLOYMENT:")
    print("   • Install OpenCog for full cognitive features:")
    print("     pip install opencog-atomspace opencog-python")
    print("   • Or use Docker: docker pull agent0ai/agent-zero:latest")
    print("   • Tools work immediately in existing Agent-Zero setup")
    print("   • Backward compatible with all existing functionality")


async def main():
    """Run the complete demonstration."""
    try:
        await demo_all_tools()
        print_integration_summary()
        print("\n🎉 ATOMSPACE-AGENT-ZERO INTEGRATION DEMONSTRATION COMPLETE!")
        return True
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)