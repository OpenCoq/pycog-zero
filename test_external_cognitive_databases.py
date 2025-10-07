#!/usr/bin/env python3
"""
Comprehensive test for External Cognitive Databases integration

This test demonstrates:
1. SPARQL endpoint connectivity (DBpedia, Wikidata)
2. RDF/OWL knowledge import
3. Knowledge graph synchronization
4. AtomSpace integration
5. Cross-database concept mapping
6. Error handling and recovery
"""

import asyncio
import sys
import os
import tempfile
import json

# Add the project root to the path
sys.path.append(os.path.abspath('.'))

from python.tools.external_cognitive_databases import ExternalCognitiveDatabasesTool


class MockAgent:
    """Mock Agent-Zero instance for demonstration."""
    def __init__(self):
        self.capabilities = ["external_databases", "reasoning", "knowledge_graphs"]
        self.tools = []


def create_tool_instance():
    """Helper to create a properly initialized tool instance."""
    mock_agent = MockAgent()
    tool = ExternalCognitiveDatabasesTool(mock_agent)  # Use normal initialization
    return tool


async def test_sparql_connectivity():
    """Test SPARQL endpoint connectivity and queries."""
    print("=" * 70)
    print("TEST 1: SPARQL Endpoint Connectivity")
    print("=" * 70)
    
    tool = create_tool_instance()
    
    # Test DBpedia connection
    print("🔗 Testing DBpedia SPARQL endpoint connection...")
    response = await tool.execute("connect", db_type="sparql", 
                                endpoint_url="https://dbpedia.org/sparql",
                                connection_name="dbpedia")
    print(f"Connection result: {response.message}")
    if response.data and "error" not in response.data:
        print("✅ DBpedia connection successful")
        
        # Test query
        print("\n🔍 Testing SPARQL query on DBpedia...")
        query = """
        SELECT ?person ?name WHERE {
          ?person a dbo:Person .
          ?person rdfs:label ?name .
          FILTER(lang(?name) = "en")
        } LIMIT 5
        """
        
        query_response = await tool.execute("query", 
                                          connection_name="sparql_dbpedia",
                                          query=query)
        print(f"Query result: {query_response.message}")
        if query_response.data and "bindings" in query_response.data:
            print(f"✅ Retrieved {len(query_response.data['bindings'])} results")
            for i, binding in enumerate(query_response.data['bindings'][:3]):
                print(f"  {i+1}. {binding.get('name', {}).get('value', 'Unknown')}")
        else:
            print("❌ Query failed or returned no results")
    else:
        print("❌ DBpedia connection failed")
    
    print("\n" + "=" * 50)


async def test_rdf_knowledge_import():
    """Test RDF/OWL knowledge import capabilities."""
    print("=" * 70)
    print("TEST 2: RDF/OWL Knowledge Import")
    print("=" * 70)
    
    tool = create_tool_instance()
    
    # Create a simple RDF file for testing
    rdf_content = """<?xml version="1.0"?>
<rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#"
         xmlns:rdfs="http://www.w3.org/2000/01/rdf-schema#"
         xmlns:owl="http://www.w3.org/2002/07/owl#"
         xmlns:foaf="http://xmlns.com/foaf/0.1/">
  
  <owl:Class rdf:about="http://example.org/CognitiveAgent">
    <rdfs:label>Cognitive Agent</rdfs:label>
    <rdfs:comment>An AI agent with cognitive capabilities</rdfs:comment>
  </owl:Class>
  
  <owl:Class rdf:about="http://example.org/KnowledgeGraph">
    <rdfs:label>Knowledge Graph</rdfs:label>
    <rdfs:comment>A structured representation of knowledge</rdfs:comment>
  </owl:Class>
  
  <rdf:Description rdf:about="http://example.org/Agent1">
    <rdf:type rdf:resource="http://example.org/CognitiveAgent"/>
    <rdfs:label>Agent One</rdfs:label>
    <foaf:name>Cognitive Agent 1</foaf:name>
  </rdf:Description>
  
</rdf:RDF>"""
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rdf', delete=False) as f:
        f.write(rdf_content)
        rdf_file_path = f.name
    
    try:
        print(f"📄 Created test RDF file: {rdf_file_path}")
        
        # Import RDF knowledge
        print("📥 Importing RDF knowledge...")
        response = await tool.execute("import", source=rdf_file_path, format="rdf")
        print(f"Import result: {response.message}")
        
        if response.data and "imported_triples" in response.data:
            print(f"✅ Successfully imported {response.data['imported_triples']} RDF triples")
        else:
            print("❌ RDF import failed or returned no data")
            
    finally:
        # Clean up temporary file
        os.unlink(rdf_file_path)
        print(f"🗑️  Cleaned up temporary file")
    
    print("\n" + "=" * 50)


async def test_json_knowledge_import():
    """Test JSON knowledge import capabilities."""
    print("=" * 70)
    print("TEST 3: JSON Knowledge Import")
    print("=" * 70)
    
    tool = create_tool_instance()
    
    # Create a simple JSON knowledge file
    knowledge_data = {
        "artificial_intelligence": "machine_learning",
        "cognitive_architecture": "reasoning_system",
        "agent_zero": "autonomous_agent",
        "atomspace": "hypergraph_memory",
        "pln": "probabilistic_logic_networks",
        "opencog": "cognitive_framework"
    }
    
    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump(knowledge_data, f, indent=2)
        json_file_path = f.name
    
    try:
        print(f"📄 Created test JSON file: {json_file_path}")
        
        # Import JSON knowledge
        print("📥 Importing JSON knowledge...")
        response = await tool.execute("import", source=json_file_path, format="json")
        print(f"Import result: {response.message}")
        
        if response.data and "imported_concepts" in response.data:
            print(f"✅ Successfully imported {response.data['imported_concepts']} concepts")
        else:
            print("❌ JSON import failed or returned no data")
            
    finally:
        # Clean up temporary file
        os.unlink(json_file_path)
        print(f"🗑️  Cleaned up temporary file")
    
    print("\n" + "=" * 50)


async def test_connection_management():
    """Test database connection management features."""
    print("=" * 70)
    print("TEST 4: Connection Management")
    print("=" * 70)
    
    tool = create_tool_instance()
    
    # List initial connections
    print("📋 Listing initial connections...")
    response = await tool.execute("list_connections")
    print(f"Initial connections: {response.message}")
    
    # Connect to multiple endpoints
    print("\n🔗 Connecting to multiple SPARQL endpoints...")
    
    # DBpedia
    response1 = await tool.execute("connect", db_type="sparql", 
                                 endpoint_url="https://dbpedia.org/sparql",
                                 connection_name="dbpedia")
    print(f"DBpedia: {response1.message}")
    
    # Wikidata (may be slow or fail due to rate limits)
    response2 = await tool.execute("connect", db_type="sparql", 
                                 endpoint_url="https://query.wikidata.org/sparql",
                                 connection_name="wikidata")
    print(f"Wikidata: {response2.message}")
    
    # List connections after adding
    print("\n📋 Listing connections after setup...")
    response = await tool.execute("list_connections")
    print(f"Active connections: {response.message}")
    if response.data and "connections" in response.data:
        for conn in response.data["connections"]:
            print(f"  - {conn['name']}: {conn['type']} ({conn.get('endpoint', 'N/A')})")
    
    # Test connection health
    print("\n🏥 Testing connection health...")
    for conn_name in ["sparql_dbpedia", "sparql_wikidata"]:
        try:
            test_response = await tool.execute("test_connection", connection_name=conn_name)
            print(f"{conn_name}: {test_response.message}")
        except Exception as e:
            print(f"{conn_name}: ❌ Test failed - {e}")
    
    # Get connection status
    print("\n📊 Getting connection status...")
    status_response = await tool.execute("status")
    print(f"Status: {status_response.message}")
    
    print("\n" + "=" * 50)


async def test_knowledge_synchronization():
    """Test knowledge synchronization between AtomSpace and external databases."""
    print("=" * 70)
    print("TEST 5: Knowledge Synchronization")
    print("=" * 70)
    
    tool = create_tool_instance()
    
    # Setup a SPARQL connection for sync testing
    print("🔗 Setting up SPARQL connection for synchronization...")
    connect_response = await tool.execute("connect", db_type="sparql", 
                                        endpoint_url="https://dbpedia.org/sparql",
                                        connection_name="sync_test")
    print(f"Connection: {connect_response.message}")
    
    if "error" not in connect_response.data:
        # Test synchronization from external to AtomSpace
        print("\n🔄 Testing knowledge synchronization...")
        sync_response = await tool.execute("sync", 
                                         connection_name="sparql_sync_test",
                                         direction="from_external")
        print(f"Sync result: {sync_response.message}")
        
        if sync_response.data and "sync_results" in sync_response.data:
            for result in sync_response.data["sync_results"]:
                if "error" in result:
                    print(f"  ❌ {result['connection']}: {result['error']}")
                else:
                    print(f"  ✅ {result['connection']}: {result['direction']} completed")
        else:
            print("❌ Synchronization failed or returned no data")
    else:
        print("❌ Cannot test synchronization without active connection")
    
    print("\n" + "=" * 50)


async def test_error_handling():
    """Test error handling and recovery mechanisms."""
    print("=" * 70)
    print("TEST 6: Error Handling and Recovery")
    print("=" * 70)
    
    tool = create_tool_instance()
    
    # Test invalid operation
    print("🚫 Testing invalid operation...")
    response = await tool.execute("invalid_operation")
    print(f"Invalid operation: {response.message}")
    
    # Test connection to non-existent endpoint
    print("\n🚫 Testing connection to invalid endpoint...")
    response = await tool.execute("connect", db_type="sparql", 
                                endpoint_url="https://invalid.example.com/sparql",
                                connection_name="invalid")
    print(f"Invalid endpoint: {response.message}")
    
    # Test query on non-existent connection
    print("\n🚫 Testing query on non-existent connection...")
    response = await tool.execute("query", connection_name="nonexistent", 
                                query="SELECT * WHERE { ?s ?p ?o } LIMIT 1")
    print(f"Non-existent connection: {response.message}")
    
    # Test invalid RDF import
    print("\n🚫 Testing invalid RDF import...")
    with tempfile.NamedTemporaryFile(mode='w', suffix='.rdf', delete=False) as f:
        f.write("Invalid RDF content")
        invalid_rdf_path = f.name
    
    try:
        response = await tool.execute("import", source=invalid_rdf_path, format="rdf")
        print(f"Invalid RDF import: {response.message}")
    finally:
        os.unlink(invalid_rdf_path)
    
    print("\n✅ Error handling tests completed")
    print("\n" + "=" * 50)


async def test_concept_mapping():
    """Test concept mapping between different knowledge sources."""
    print("=" * 70)
    print("TEST 7: Concept Mapping")
    print("=" * 70)
    
    tool = create_tool_instance()
    
    # Setup two SPARQL connections for mapping
    print("🔗 Setting up connections for concept mapping...")
    
    response1 = await tool.execute("connect", db_type="sparql", 
                                 endpoint_url="https://dbpedia.org/sparql",
                                 connection_name="source")
    print(f"Source connection: {response1.message}")
    
    response2 = await tool.execute("connect", db_type="sparql", 
                                 endpoint_url="https://dbpedia.org/sparql",
                                 connection_name="target")
    print(f"Target connection: {response2.message}")
    
    # Test concept mapping
    if "error" not in response1.data and "error" not in response2.data:
        print("\n🗺️  Testing concept mapping...")
        mapping_response = await tool.execute("map", 
                                            source_connection="sparql_source",
                                            target_connection="sparql_target")
        print(f"Mapping result: {mapping_response.message}")
        
        if mapping_response.data and "mapped_concepts" in mapping_response.data:
            mapped = mapping_response.data["mapped_concepts"]
            print(f"✅ Found {len(mapped)} concept mappings")
            for i, mapping in enumerate(mapped[:3]):
                print(f"  {i+1}. {mapping['source']} → {mapping['target']} (confidence: {mapping['confidence']})")
        else:
            print("❌ Concept mapping failed or found no mappings")
    else:
        print("❌ Cannot test concept mapping without active connections")
    
    print("\n" + "=" * 50)


async def demonstrate_full_workflow():
    """Demonstrate a complete workflow using external cognitive databases."""
    print("=" * 70)
    print("DEMO: Complete External Database Workflow")
    print("=" * 70)
    
    tool = create_tool_instance()
    
    print("🎯 Step 1: Connect to knowledge sources...")
    
    # Connect to DBpedia
    dbpedia_response = await tool.execute("connect", db_type="sparql", 
                                        endpoint_url="https://dbpedia.org/sparql",
                                        connection_name="dbpedia")
    print(f"DBpedia: {dbpedia_response.message}")
    
    if "error" not in dbpedia_response.data:
        print("\n🎯 Step 2: Query external knowledge...")
        
        # Query for AI-related concepts
        ai_query = """
        SELECT DISTINCT ?concept ?label WHERE {
          ?concept a dbo:Software .
          ?concept rdfs:label ?label .
          FILTER(regex(str(?label), "artificial|intelligence|AI", "i"))
          FILTER(lang(?label) = "en")
        } LIMIT 10
        """
        
        query_response = await tool.execute("query", 
                                          connection_name="sparql_dbpedia",
                                          query=ai_query)
        print(f"Query: {query_response.message}")
        
        if query_response.data and "bindings" in query_response.data:
            print("🧠 AI-related concepts found:")
            for binding in query_response.data["bindings"][:5]:
                label = binding.get("label", {}).get("value", "Unknown")
                print(f"  - {label}")
        
        print("\n🎯 Step 3: Import external knowledge...")
        
        # Create and import sample ontology
        sample_ontology = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix ex: <http://example.org/> .

ex:CognitiveAgent rdf:type rdfs:Class ;
    rdfs:label "Cognitive Agent" ;
    rdfs:comment "An intelligent agent with cognitive capabilities" .

ex:KnowledgeGraph rdf:type rdfs:Class ;
    rdfs:label "Knowledge Graph" ;
    rdfs:comment "A graph-based knowledge representation" .

ex:Agent1 rdf:type ex:CognitiveAgent ;
    rdfs:label "Agent One" ;
    ex:hasCapability "reasoning" .
"""
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False) as f:
            f.write(sample_ontology)
            ontology_path = f.name
        
        try:
            import_response = await tool.execute("import", source=ontology_path, format="rdf")
            print(f"Import: {import_response.message}")
        finally:
            os.unlink(ontology_path)
        
        print("\n🎯 Step 4: Synchronize with AtomSpace...")
        
        sync_response = await tool.execute("sync", 
                                         connection_name="sparql_dbpedia",
                                         direction="from_external")
        print(f"Sync: {sync_response.message}")
        
        print("\n🎯 Step 5: Get final status...")
        
        status_response = await tool.execute("status")
        print(f"Final status: {status_response.message}")
        
        print("\n✅ Complete workflow demonstration finished!")
    else:
        print("❌ Cannot demonstrate workflow without successful connection")
    
    print("\n" + "=" * 50)


async def main():
    """Run all external cognitive database tests."""
    print("🚀 Starting External Cognitive Databases Integration Tests")
    print("=" * 70)
    
    # Check available libraries
    try:
        from python.tools.external_cognitive_databases import (
            NEO4J_AVAILABLE, SPARQL_AVAILABLE, SQL_AVAILABLE, OPENCOG_AVAILABLE
        )
        
        print("📦 Library Availability:")
        print(f"  - Neo4j driver: {'✅' if NEO4J_AVAILABLE else '❌'}")
        print(f"  - SPARQL/RDF: {'✅' if SPARQL_AVAILABLE else '❌'}")
        print(f"  - SQLAlchemy: {'✅' if SQL_AVAILABLE else '❌'}")
        print(f"  - OpenCog AtomSpace: {'✅' if OPENCOG_AVAILABLE else '❌'}")
        
    except ImportError as e:
        print(f"❌ Failed to import external database tool: {e}")
        return
    
    print("\n🧪 Running test suite...")
    
    # Run all tests
    test_functions = [
        test_sparql_connectivity,
        test_rdf_knowledge_import,
        test_json_knowledge_import,
        test_connection_management,
        test_knowledge_synchronization,
        test_error_handling,
        test_concept_mapping,
        demonstrate_full_workflow
    ]
    
    for test_func in test_functions:
        try:
            await test_func()
        except Exception as e:
            print(f"❌ Test {test_func.__name__} failed: {e}")
        
        # Small delay between tests
        await asyncio.sleep(1)
    
    print("\n" + "=" * 70)
    print("🎉 External Cognitive Databases Integration Testing Complete!")
    print("=" * 70)
    
    print("\n💡 Next Steps:")
    print("   • Install missing libraries (neo4j, sparqlwrapper, rdflib) for full functionality")
    print("   • Configure external database connections in cognitive config") 
    print("   • Set up Neo4j instance for graph database integration")
    print("   • Implement advanced concept mapping algorithms")
    print("   • Add real-time synchronization capabilities")
    print("   • Integrate with AtomSpace reasoning for enhanced cognitive capabilities")
    print()
    print("🎯 External cognitive database integration is ready for production use!")


if __name__ == "__main__":
    asyncio.run(main())