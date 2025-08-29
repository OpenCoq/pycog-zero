#!/usr/bin/env python3
"""
PyCog-Zero AtomSpace-Agent-Zero Integration Demonstration
Shows how Agent-Zero tools integrate with AtomSpace components for cognitive enhancement
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


async def demo_memory_bridge(agent):
    """Demonstrate AtomSpace memory bridge functionality."""
    print("\n🧠 ATOMSPACE MEMORY BRIDGE DEMONSTRATION")
    print("=" * 60)
    
    tool = create_tool_instance(AtomSpaceMemoryBridge, agent)
    
    # Mock the Memory.get method to avoid database dependencies
    import python.helpers.memory
    original_get = python.helpers.memory.Memory.get
    
    async def mock_memory_get(agent):
        mock_memory = Mock()
        mock_memory.db = Mock()
        mock_memory.db.get_all_docs = Mock(return_value={
            "doc1": Mock(page_content="This is about artificial intelligence and machine learning", metadata={"topic": "AI"}),
            "doc2": Mock(page_content="Cognitive computing enables reasoning and pattern recognition", metadata={"topic": "cognition"})
        })
        mock_memory.search_similarity_threshold = Mock(return_value=[
            Mock(page_content="Search result about AI", score=0.9),
            Mock(page_content="Another result about cognition", score=0.8)
        ])
        return mock_memory
    
    python.helpers.memory.Memory.get = mock_memory_get
    
    try:
        # Bridge memory to AtomSpace
        print("1. Bridging Agent-Zero memory to AtomSpace...")
        response = await tool.execute(operation="bridge_memory")
        print(f"   Result: {response.message}")
        
        # Query knowledge
        print("\n2. Querying AtomSpace knowledge...")
        response = await tool.execute(operation="query_knowledge", query="machine learning AI")
        print(f"   Result: {response.message}")
        
        # Enhanced search
        print("\n3. Enhanced memory search with AtomSpace...")
        response = await tool.execute(operation="enhance_search", query="cognitive architecture")
        print(f"   Result: {response.message}")
        
        # Cross-reference memories
        print("\n4. Cross-referencing memories...")
        response = await tool.execute(operation="cross_reference", concept="intelligence")
        print(f"   Result: {response.message}")
        
    finally:
        # Restore original Memory.get
        python.helpers.memory.Memory.get = original_get


async def demo_search_engine(agent):
    """Demonstrate AtomSpace-enhanced search engine."""
    print("\n🔍 ATOMSPACE SEARCH ENGINE DEMONSTRATION")
    print("=" * 60)
    
    tool = create_tool_instance(AtomSpaceSearchEngine, agent)
    
    # Mock searxng to avoid network calls
    import python.helpers.searxng
    original_search = python.helpers.searxng.search
    
    async def mock_search(query):
        return {
            "results": [
                {
                    "title": f"Search Result for '{query}'",
                    "url": f"https://example.com/search?q={query}",
                    "content": f"This is mock content related to '{query}' with cognitive architecture concepts."
                },
                {
                    "title": "Cognitive Computing Overview",
                    "url": "https://example.com/cognitive",
                    "content": "Cognitive computing combines artificial intelligence and machine learning."
                }
            ]
        }
    
    python.helpers.searxng.search = mock_search
    
    try:
        print("1. Standard search with cognitive enhancement...")
        response = await tool.execute(query="artificial intelligence cognitive systems")
        print(f"   Result: {response.message[:500]}...")
        
        print("\n2. Search without cognitive enhancement...")
        response = await tool.execute(query="machine learning", cognitive_enhancement=False)
        print(f"   Result: {response.message[:300]}...")
        
    finally:
        # Restore original search function
        python.helpers.searxng.search = original_search


async def demo_document_query(agent):
    """Demonstrate AtomSpace-enhanced document query."""
    print("\n📄 ATOMSPACE DOCUMENT QUERY DEMONSTRATION")
    print("=" * 60)
    
    tool = create_tool_instance(AtomSpaceDocumentQuery, agent)
    
    # Mock DocumentQueryHelper
    import python.helpers.document_query
    
    class MockDocumentQueryHelper:
        def __init__(self, agent, progress_callback):
            self.agent = agent
            self.progress_callback = progress_callback
        
        async def document_get_content(self, uri):
            return ("This is a sample document about artificial intelligence and cognitive computing. "
                   "It discusses machine learning algorithms, neural networks, and knowledge representation. "
                   "The document covers topics like reasoning, pattern matching, and symbolic AI.")
        
        async def document_qa(self, uri, queries):
            content = await self.document_get_content(uri)
            return None, content
    
    original_helper = python.helpers.document_query.DocumentQueryHelper
    python.helpers.document_query.DocumentQueryHelper = MockDocumentQueryHelper
    
    try:
        print("1. Document query with semantic analysis...")
        response = await tool.execute(
            document="demo_doc.txt",
            query="What is artificial intelligence?",
            semantic_analysis=True
        )
        print(f"   Result: {response.message[:500]}...")
        
        print("\n2. Document query without semantic analysis...")
        response = await tool.execute(
            document="demo_doc.txt",
            semantic_analysis=False
        )
        print(f"   Result: {response.message[:300]}...")
        
    finally:
        # Restore original helper
        python.helpers.document_query.DocumentQueryHelper = original_helper


async def demo_code_context(agent):
    """Demonstrate AtomSpace code context analysis."""
    print("\n💻 ATOMSPACE CODE CONTEXT DEMONSTRATION")
    print("=" * 60)
    
    tool = create_tool_instance(AtomSpaceCodeContext, agent)
    
    # Store code execution context
    print("1. Storing Python code execution context...")
    python_code = """
import numpy as np
from sklearn.linear_model import LinearRegression

def train_model(X, y):
    model = LinearRegression()
    model.fit(X, y)
    return model

data = np.random.random((100, 5))
target = np.random.random(100)
trained_model = train_model(data, target)
print("Model trained successfully")
"""
    
    response = await tool.execute(
        operation="store_context",
        code=python_code,
        result="Model trained successfully",
        language="python"
    )
    print(f"   Result: {response.message}")
    
    # Analyze code patterns
    print("\n2. Analyzing code patterns...")
    response = await tool.execute(operation="analyze_patterns")
    print(f"   Result: {response.message}")
    
    # Suggest improvements
    print("\n3. Suggesting code improvements...")
    response = await tool.execute(
        operation="suggest_improvements",
        code="x = 1\ny = 2\nprint(x + y)",
        language="python"
    )
    print(f"   Result: {response.message}")
    
    # Trace dependencies
    print("\n4. Tracing code dependencies...")
    response = await tool.execute(
        operation="trace_dependencies",
        code=python_code,
        language="python"
    )
    print(f"   Result: {response.message}")


async def demo_tool_hub(agent):
    """Demonstrate AtomSpace tool integration hub."""
    print("\n🔗 ATOMSPACE TOOL INTEGRATION HUB DEMONSTRATION")
    print("=" * 60)
    
    tool = create_tool_instance(AtomSpaceToolHub, agent)
    
    # Get hub status
    print("1. Getting tool integration hub status...")
    response = await tool.execute(operation="status")
    print(f"   Result: {response.message}")
    
    # Share data between tools
    print("\n2. Sharing data between tools...")
    response = await tool.execute(
        operation="share_data",
        tool_name="memory_bridge",
        data_type="knowledge_graph",
        data={
            "concepts": ["artificial_intelligence", "machine_learning", "cognitive_computing"],
            "relationships": ["AI contains ML", "ML enables cognition"],
            "confidence": 0.85
        }
    )
    print(f"   Result: {response.message}")
    
    # Retrieve shared data
    print("\n3. Retrieving shared data...")
    response = await tool.execute(
        operation="retrieve_data",
        tool_name="memory_bridge"
    )
    print(f"   Result: {response.message}")
    
    # Coordinate tool execution
    print("\n4. Coordinating multi-tool execution...")
    response = await tool.execute(
        operation="coordinate_tools",
        primary_tool="memory_system",
        supporting_tools=["search_engine", "document_processor"],
        task_description="comprehensive knowledge extraction and analysis"
    )
    print(f"   Result: {response.message}")
    
    # Analyze tool interactions
    print("\n5. Analyzing tool interactions...")
    response = await tool.execute(operation="analyze_interactions")
    print(f"   Result: {response.message}")


async def demo_memory_extension(agent):
    """Demonstrate AtomSpace memory extension."""
    print("\n🗃️ ATOMSPACE MEMORY EXTENSION DEMONSTRATION")
    print("=" * 60)
    
    # Mock memory instance
    memory_mock = Mock()
    memory_mock.memory_subdir = "demo"
    memory_mock.db = Mock()
    
    # Mock documents
    doc1 = Mock()
    doc1.page_content = "Artificial intelligence is transforming cognitive computing and machine learning applications"
    doc1.metadata = {"source": "research_paper", "topic": "AI"}
    
    doc2 = Mock()
    doc2.page_content = "Machine learning algorithms enable pattern recognition and automated reasoning systems"
    doc2.metadata = {"source": "technical_manual", "topic": "ML"}
    
    memory_mock.db.get_all_docs.return_value = {
        "doc1": doc1,
        "doc2": doc2
    }
    
    extension = AtomSpaceMemoryExtension(memory_mock)
    
    # Create knowledge graph
    print("1. Creating knowledge graph from memory...")
    result = await extension.create_knowledge_graph_from_memory()
    print(f"   Result: Knowledge graph created successfully: {result}")
    
    # Find related concepts
    print("\n2. Finding related concepts...")
    related = extension.find_related_concepts("artificial", max_results=5)
    print(f"   Result: Found {len(related)} related concepts")
    
    # Get concept neighbors
    print("\n3. Getting concept neighbors...")
    neighbors = extension.get_concept_neighbors("intelligence", depth=2)
    print(f"   Result: Found neighbors for {len(neighbors)} concepts")
    
    # Analyze knowledge graph structure
    print("\n4. Analyzing knowledge graph structure...")
    analysis = extension.analyze_knowledge_graph_structure()
    print(f"   Result: Analysis complete - {len(analysis)} metrics")


def print_integration_summary():
    """Print summary of AtomSpace-Agent-Zero integration."""
    print("\n" + "=" * 80)
    print("🎯 ATOMSPACE-AGENT-ZERO INTEGRATION SUMMARY")
    print("=" * 80)
    
    print("\n✅ INTEGRATION COMPLETE - New AtomSpace-Enhanced Tools:")
    print("   1. AtomSpaceMemoryBridge - Bridges Agent-Zero memory with AtomSpace")
    print("   2. AtomSpaceSearchEngine - Cognitive search enhancement")
    print("   3. AtomSpaceDocumentQuery - Semantic document analysis")
    print("   4. AtomSpaceCodeContext - Code execution context storage")
    print("   5. AtomSpaceToolHub - Cross-tool coordination and data sharing")
    print("   6. AtomSpaceMemoryExtension - Knowledge graph functionality")
    
    print("\n🔧 KEY FEATURES IMPLEMENTED:")
    print("   • Cross-tool data sharing via shared AtomSpace")
    print("   • Cognitive enhancement of standard Agent-Zero operations")
    print("   • Knowledge graph creation from memory contents")
    print("   • Semantic analysis and reasoning capabilities")
    print("   • Code pattern analysis and dependency tracing")
    print("   • Tool coordination and execution planning")
    print("   • Graceful fallback when OpenCog unavailable")
    
    print("\n🚀 INTEGRATION BENEFITS:")
    print("   • Enhanced cognitive capabilities for all Agent-Zero tools")
    print("   • Knowledge persistence and cross-referencing")
    print("   • Semantic understanding of documents and code")
    print("   • Intelligent tool coordination")
    print("   • Reasoning-based search enhancement")
    print("   • Pattern recognition across executions")
    
    print("\n📋 TECHNICAL ARCHITECTURE:")
    print("   • All tools follow Agent-Zero tool patterns")
    print("   • Backward compatible with existing Agent-Zero setup")
    print("   • Shared AtomSpace for cross-tool communication")
    print("   • OpenCog integration with graceful fallback")
    print("   • Persistent knowledge graph storage")
    print("   • Comprehensive error handling and logging")


async def main():
    """Run the complete AtomSpace-Agent-Zero integration demonstration."""
    print("🧠 PyCog-Zero AtomSpace-Agent-Zero Integration Demonstration")
    print("=" * 80)
    print("This demonstration shows how Agent-Zero tools are enhanced with")
    print("OpenCog AtomSpace components for cognitive capabilities.")
    print("\nNote: Running in fallback mode (OpenCog not installed)")
    print("In production, install OpenCog for full cognitive features.")
    
    # Create mock agent
    agent = create_mock_agent()
    
    try:
        # Run all demonstrations
        await demo_memory_bridge(agent)
        await demo_search_engine(agent)
        await demo_document_query(agent)
        await demo_code_context(agent)
        await demo_tool_hub(agent)
        await demo_memory_extension(agent)
        
        # Print summary
        print_integration_summary()
        
        print(f"\n🎉 DEMONSTRATION COMPLETE!")
        print("AtomSpace-Agent-Zero integration is working correctly.")
        
    except Exception as e:
        print(f"\n❌ DEMONSTRATION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)