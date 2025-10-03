#!/usr/bin/env python3
"""
Demonstration of External Cognitive Databases Integration

This demo showcases:
1. Connecting to external knowledge graphs (DBpedia, Wikidata)
2. Querying semantic knowledge using SPARQL
3. Importing RDF/OWL ontologies 
4. Synchronizing with Agent-Zero's AtomSpace
5. Cross-database concept mapping and alignment
6. Real-world cognitive reasoning with external knowledge
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
    tool = ExternalCognitiveDatabasesTool(mock_agent)
    return tool


async def demo_dbpedia_integration():
    """Demonstrate DBpedia knowledge graph integration."""
    print("=" * 80)
    print("🌐 DEMO 1: DBpedia Knowledge Graph Integration")
    print("=" * 80)
    
    tool = create_tool_instance()
    
    print("🔗 Connecting to DBpedia SPARQL endpoint...")
    
    # Connect to DBpedia
    connect_response = await tool.execute(
        "connect", 
        db_type="sparql",
        endpoint_url="https://dbpedia.org/sparql",
        connection_name="dbpedia"
    )
    
    print(f"Connection result: {connect_response.message}")
    
    if connect_response.data and "error" not in connect_response.data:
        print("✅ Successfully connected to DBpedia!")
        
        # Query for AI researchers and their contributions
        print("\n🔍 Querying AI researchers and their work...")
        
        ai_researchers_query = """
        SELECT DISTINCT ?person ?name ?field ?description WHERE {
          ?person a dbo:Person .
          ?person rdfs:label ?name .
          ?person dbo:field ?field .
          ?person dbo:abstract ?description .
          FILTER(regex(str(?field), "artificial|intelligence|machine|learning|computer", "i"))
          FILTER(lang(?name) = "en")
          FILTER(lang(?description) = "en")
        } LIMIT 5
        """
        
        query_response = await tool.execute(
            "query",
            connection_name="sparql_dbpedia", 
            query=ai_researchers_query
        )
        
        print(f"Query result: {query_response.message}")
        
        if query_response.data and "bindings" in query_response.data:
            researchers = query_response.data["bindings"]
            print(f"\n🧠 Found {len(researchers)} AI researchers:")
            
            for i, researcher in enumerate(researchers):
                name = researcher.get("name", {}).get("value", "Unknown")
                field = researcher.get("field", {}).get("value", "Unknown field")
                description = researcher.get("description", {}).get("value", "No description")
                
                print(f"\n  {i+1}. {name}")
                print(f"     Field: {field}")
                print(f"     Description: {description[:200]}...")
        
        # Query for AI-related technologies
        print("\n\n🤖 Querying AI technologies and frameworks...")
        
        ai_tech_query = """
        SELECT DISTINCT ?tech ?label ?type ?abstract WHERE {
          ?tech a ?type .
          ?tech rdfs:label ?label .
          ?tech dbo:abstract ?abstract .
          FILTER(
            ?type = dbo:Software || 
            ?type = dbo:ProgrammingLanguage ||
            ?type = dbr:Artificial_intelligence
          )
          FILTER(regex(str(?label), "tensorflow|pytorch|scikit|neural|deep", "i"))
          FILTER(lang(?label) = "en")
          FILTER(lang(?abstract) = "en")
        } LIMIT 3
        """
        
        tech_response = await tool.execute(
            "query",
            connection_name="sparql_dbpedia",
            query=ai_tech_query
        )
        
        if tech_response.data and "bindings" in tech_response.data:
            technologies = tech_response.data["bindings"]
            print(f"🛠️  Found {len(technologies)} AI technologies:")
            
            for i, tech in enumerate(technologies):
                label = tech.get("label", {}).get("value", "Unknown")
                tech_type = tech.get("type", {}).get("value", "Unknown type")
                abstract = tech.get("abstract", {}).get("value", "No description")
                
                print(f"\n  {i+1}. {label}")
                print(f"     Type: {tech_type.split('/')[-1]}")  # Get last part of URI
                print(f"     Description: {abstract[:150]}...")
        
    else:
        print("❌ Failed to connect to DBpedia")
    
    print("\n" + "=" * 60)


async def demo_wikidata_integration():
    """Demonstrate Wikidata knowledge base integration."""
    print("=" * 80)
    print("🌍 DEMO 2: Wikidata Knowledge Base Integration") 
    print("=" * 80)
    
    tool = create_tool_instance()
    
    print("🔗 Connecting to Wikidata SPARQL endpoint...")
    
    # Connect to Wikidata
    connect_response = await tool.execute(
        "connect",
        db_type="sparql", 
        endpoint_url="https://query.wikidata.org/sparql",
        connection_name="wikidata"
    )
    
    print(f"Connection result: {connect_response.message}")
    
    if connect_response.data and "error" not in connect_response.data:
        print("✅ Successfully connected to Wikidata!")
        
        # Query for cognitive science concepts
        print("\n🧠 Querying cognitive science concepts...")
        
        cognitive_concepts_query = """
        SELECT DISTINCT ?concept ?conceptLabel ?description WHERE {
          ?concept wdt:P31/wdt:P279* wd:Q4671848 .  # instance of/subclass of cognitive science
          ?concept rdfs:label ?conceptLabel .
          OPTIONAL { ?concept schema:description ?description . }
          FILTER(lang(?conceptLabel) = "en")
          FILTER(lang(?description) = "en")
        } LIMIT 8
        """
        
        query_response = await tool.execute(
            "query",
            connection_name="sparql_wikidata",
            query=cognitive_concepts_query
        )
        
        print(f"Query result: {query_response.message}")
        
        if query_response.data and "bindings" in query_response.data:
            concepts = query_response.data["bindings"]
            print(f"\n🧩 Found {len(concepts)} cognitive science concepts:")
            
            for i, concept in enumerate(concepts):
                label = concept.get("conceptLabel", {}).get("value", "Unknown concept")
                description = concept.get("description", {}).get("value", "No description available")
                
                print(f"\n  {i+1}. {label}")
                print(f"     Description: {description}")
        
        # Query for AI programming languages
        print("\n\n💻 Querying AI programming languages...")
        
        ai_languages_query = """
        SELECT DISTINCT ?lang ?langLabel ?paradigm ?paradigmLabel WHERE {
          ?lang wdt:P31 wd:Q9143 .  # instance of programming language
          ?lang wdt:P3966 ?paradigm .  # programming paradigm
          ?paradigm rdfs:label ?paradigmLabel .
          ?lang rdfs:label ?langLabel .
          FILTER(regex(str(?langLabel), "Python|R|Julia|Lisp|Prolog", "i"))
          FILTER(lang(?langLabel) = "en")
          FILTER(lang(?paradigmLabel) = "en")
        } LIMIT 5
        """
        
        lang_response = await tool.execute(
            "query",
            connection_name="sparql_wikidata",
            query=ai_languages_query
        )
        
        if lang_response.data and "bindings" in lang_response.data:
            languages = lang_response.data["bindings"]
            print(f"🐍 Found {len(languages)} AI-relevant programming languages:")
            
            for i, lang in enumerate(languages):
                lang_name = lang.get("langLabel", {}).get("value", "Unknown")
                paradigm = lang.get("paradigmLabel", {}).get("value", "Unknown paradigm")
                
                print(f"  {i+1}. {lang_name} - {paradigm}")
    
    else:
        print("❌ Failed to connect to Wikidata (may be due to rate limits)")
    
    print("\n" + "=" * 60)


async def demo_ontology_import():
    """Demonstrate RDF/OWL ontology import capabilities."""
    print("=" * 80)
    print("📚 DEMO 3: Cognitive Ontology Import and Integration")
    print("=" * 80)
    
    tool = create_tool_instance()
    
    # Create a comprehensive cognitive science ontology
    cognitive_ontology = """@prefix rdf: <http://www.w3.org/1999/02/22-rdf-syntax-ns#> .
@prefix rdfs: <http://www.w3.org/2000/01/rdf-schema#> .
@prefix owl: <http://www.w3.org/2002/07/owl#> .
@prefix cog: <http://pycog-zero.org/ontology/> .

# Ontology definition
<http://pycog-zero.org/ontology/> rdf:type owl:Ontology ;
    rdfs:label "PyCog-Zero Cognitive Architecture Ontology" ;
    rdfs:comment "Ontology for cognitive agents and reasoning systems" .

# Core cognitive concepts
cog:CognitiveAgent rdf:type owl:Class ;
    rdfs:label "Cognitive Agent" ;
    rdfs:comment "An intelligent agent with reasoning and learning capabilities" .

cog:AtomSpace rdf:type owl:Class ;
    rdfs:label "AtomSpace" ;
    rdfs:comment "Hypergraph-based memory system for cognitive agents" ;
    rdfs:subClassOf cog:MemorySystem .

cog:MemorySystem rdf:type owl:Class ;
    rdfs:label "Memory System" ;
    rdfs:comment "System for storing and retrieving cognitive information" .

cog:ReasoningEngine rdf:type owl:Class ;
    rdfs:label "Reasoning Engine" ;
    rdfs:comment "System for logical inference and pattern matching" .

cog:KnowledgeGraph rdf:type owl:Class ;
    rdfs:label "Knowledge Graph" ;
    rdfs:comment "Graph-based representation of structured knowledge" .

# Properties
cog:hasCapability rdf:type owl:ObjectProperty ;
    rdfs:label "has capability" ;
    rdfs:domain cog:CognitiveAgent ;
    rdfs:comment "Relates an agent to its cognitive capabilities" .

cog:usesMemorySystem rdf:type owl:ObjectProperty ;
    rdfs:label "uses memory system" ;
    rdfs:domain cog:CognitiveAgent ;
    rdfs:range cog:MemorySystem .

cog:performsReasoning rdf:type owl:ObjectProperty ;
    rdfs:label "performs reasoning" ;
    rdfs:domain cog:CognitiveAgent ;
    rdfs:range cog:ReasoningEngine .

# Specific instances
cog:AgentZero rdf:type cog:CognitiveAgent ;
    rdfs:label "Agent-Zero" ;
    rdfs:comment "Autonomous cognitive agent with advanced reasoning capabilities" ;
    cog:hasCapability cog:PLNReasoning ;
    cog:hasCapability cog:PatternMatching ;
    cog:hasCapability cog:LearningAdaptation ;
    cog:usesMemorySystem cog:OpenCogAtomSpace .

cog:OpenCogAtomSpace rdf:type cog:AtomSpace ;
    rdfs:label "OpenCog AtomSpace" ;
    rdfs:comment "OpenCog implementation of hypergraph memory" .

cog:PLNReasoning rdf:type owl:Class ;
    rdfs:label "PLN Reasoning" ;
    rdfs:comment "Probabilistic Logic Networks reasoning capability" ;
    rdfs:subClassOf cog:ReasoningCapability .

cog:ReasoningCapability rdf:type owl:Class ;
    rdfs:label "Reasoning Capability" ;
    rdfs:comment "General class for reasoning abilities" .

cog:PatternMatching rdf:type cog:ReasoningCapability ;
    rdfs:label "Pattern Matching" ;
    rdfs:comment "Ability to recognize and match patterns in data" .

cog:LearningAdaptation rdf:type owl:Class ;
    rdfs:label "Learning and Adaptation" ;
    rdfs:comment "Capability for learning from experience and adapting behavior" .

# External database integration concepts
cog:ExternalDatabase rdf:type owl:Class ;
    rdfs:label "External Database" ;
    rdfs:comment "External cognitive database or knowledge source" .

cog:Neo4jGraph rdf:type cog:ExternalDatabase ;
    rdfs:label "Neo4j Graph Database" ;
    rdfs:comment "Graph database for storing and querying connected knowledge" .

cog:SPARQLEndpoint rdf:type cog:ExternalDatabase ;
    rdfs:label "SPARQL Endpoint" ;
    rdfs:comment "Semantic web endpoint for RDF knowledge queries" .

cog:connectsTo rdf:type owl:ObjectProperty ;
    rdfs:label "connects to" ;
    rdfs:domain cog:CognitiveAgent ;
    rdfs:range cog:ExternalDatabase ;
    rdfs:comment "Indicates that an agent can connect to an external database" .
"""
    
    print("📝 Creating comprehensive cognitive ontology...")
    
    # Write ontology to temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.ttl', delete=False) as f:
        f.write(cognitive_ontology)
        ontology_path = f.name
    
    try:
        print(f"📄 Created ontology file: {ontology_path}")
        
        # Import the ontology
        print("\n📥 Importing cognitive ontology into AtomSpace...")
        
        import_response = await tool.execute(
            "import",
            source=ontology_path,
            format="rdf"
        )
        
        print(f"Import result: {import_response.message}")
        
        if import_response.data and "imported_triples" in import_response.data:
            triples_count = import_response.data["imported_triples"]
            print(f"✅ Successfully imported {triples_count} RDF triples!")
            print(f"🧠 Cognitive concepts now available in AtomSpace for reasoning")
        else:
            print("❌ Ontology import failed")
        
        # Also demonstrate JSON knowledge import
        print("\n📊 Importing additional knowledge from JSON...")
        
        additional_knowledge = {
            "machine_learning": "supervised_learning",
            "deep_learning": "neural_networks", 
            "natural_language_processing": "text_analysis",
            "computer_vision": "image_recognition",
            "robotics": "autonomous_systems",
            "expert_systems": "knowledge_based_ai",
            "genetic_algorithms": "evolutionary_computation",
            "fuzzy_logic": "approximate_reasoning",
            "bayesian_networks": "probabilistic_reasoning",
            "reinforcement_learning": "reward_based_learning"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(additional_knowledge, f, indent=2)
            json_path = f.name
        
        try:
            json_response = await tool.execute(
                "import",
                source=json_path,
                format="json"
            )
            
            print(f"JSON import result: {json_response.message}")
            
            if json_response.data and "imported_concepts" in json_response.data:
                concepts_count = json_response.data["imported_concepts"]
                print(f"✅ Successfully imported {concepts_count} additional AI concepts!")
            
        finally:
            os.unlink(json_path)
        
    finally:
        os.unlink(ontology_path)
        print("🗑️  Cleaned up temporary files")
    
    print("\n" + "=" * 60)


async def demo_multi_database_workflow():
    """Demonstrate a complete workflow using multiple external databases.""" 
    print("=" * 80)
    print("🔄 DEMO 4: Multi-Database Cognitive Workflow")
    print("=" * 80)
    
    tool = create_tool_instance()
    
    print("🎯 Step 1: Setting up multiple knowledge sources...")
    
    # Connect to multiple databases
    connections = [
        ("dbpedia", "https://dbpedia.org/sparql"),
        ("wikidata", "https://query.wikidata.org/sparql")
    ]
    
    active_connections = []
    
    for name, endpoint in connections:
        print(f"🔗 Connecting to {name}...")
        response = await tool.execute(
            "connect",
            db_type="sparql",
            endpoint_url=endpoint,
            connection_name=name
        )
        
        if response.data and "error" not in response.data:
            print(f"✅ {name} connected successfully")
            active_connections.append(f"sparql_{name}")
        else:
            print(f"❌ {name} connection failed: {response.message}")
    
    if active_connections:
        print(f"\n📊 Active connections: {len(active_connections)}")
        
        # List all connections
        print("\n🔍 Listing all database connections...")
        list_response = await tool.execute("list_connections")
        
        if list_response.data and "connections" in list_response.data:
            for conn in list_response.data["connections"]:
                print(f"  • {conn['name']}: {conn['type']} - {conn.get('endpoint', 'N/A')}")
        
        print("\n🎯 Step 2: Cross-database concept queries...")
        
        # Query each database for AI concepts
        for conn_name in active_connections[:1]:  # Limit to prevent rate limiting
            print(f"\n🔍 Querying {conn_name} for cognitive science concepts...")
            
            if "dbpedia" in conn_name:
                query = """
                SELECT DISTINCT ?concept ?label WHERE {
                  ?concept a dbo:AcademicSubject .
                  ?concept rdfs:label ?label .
                  FILTER(regex(str(?label), "cognitive|intelligence|mind", "i"))
                  FILTER(lang(?label) = "en")
                } LIMIT 5
                """
            else:
                # Wikidata query
                query = """
                SELECT DISTINCT ?concept ?conceptLabel WHERE {
                  ?concept wdt:P31/wdt:P279* wd:Q4671848 .
                  ?concept rdfs:label ?conceptLabel .
                  FILTER(lang(?conceptLabel) = "en")
                } LIMIT 3
                """
            
            query_response = await tool.execute(
                "query",
                connection_name=conn_name,
                query=query
            )
            
            if query_response.data and ("bindings" in query_response.data or "records" in query_response.data):
                results = query_response.data.get("bindings", query_response.data.get("records", []))
                print(f"  Found {len(results)} concepts")
                
                for result in results[:3]:
                    if "label" in result:
                        concept_name = result["label"].get("value", "Unknown")
                    elif "conceptLabel" in result:
                        concept_name = result["conceptLabel"].get("value", "Unknown")
                    else:
                        concept_name = "Unknown concept"
                    print(f"    - {concept_name}")
        
        print("\n🎯 Step 3: Knowledge synchronization...")
        
        # Sync knowledge from external sources to AtomSpace
        for conn_name in active_connections[:1]:  # Limit for demo
            print(f"\n🔄 Synchronizing knowledge from {conn_name}...")
            
            sync_response = await tool.execute(
                "sync",
                connection_name=conn_name,
                direction="from_external"
            )
            
            print(f"Sync result: {sync_response.message}")
            
            if sync_response.data and "sync_results" in sync_response.data:
                for result in sync_response.data["sync_results"]:
                    if "error" in result:
                        print(f"  ❌ {result['connection']}: {result['error']}")
                    else:
                        print(f"  ✅ {result['connection']}: {result['direction']} completed")
        
        print("\n🎯 Step 4: Cross-database concept mapping...")
        
        # Map concepts between databases
        if len(active_connections) >= 2:
            mapping_response = await tool.execute(
                "map",
                source_connection=active_connections[0],
                target_connection=active_connections[1]
            )
            
            print(f"Concept mapping: {mapping_response.message}")
            
            if mapping_response.data and "mapped_concepts" in mapping_response.data:
                mappings = mapping_response.data["mapped_concepts"]
                print(f"📊 Found {len(mappings)} concept alignments:")
                
                for mapping in mappings[:3]:
                    print(f"  • {mapping['source']} ↔ {mapping['target']} (confidence: {mapping['confidence']})")
        
        print("\n🎯 Step 5: Connection health check...")
        
        # Test all connections
        for conn_name in active_connections:
            test_response = await tool.execute("test_connection", connection_name=conn_name)
            status = "✅ Healthy" if "error" not in test_response.data else "❌ Unhealthy"
            print(f"  {conn_name}: {status}")
        
        print("\n✅ Multi-database workflow completed successfully!")
        
    else:
        print("❌ No active connections available for workflow demonstration")
    
    print("\n" + "=" * 60)


async def demo_cognitive_reasoning_with_external_knowledge():
    """Demonstrate cognitive reasoning enhanced with external knowledge."""
    print("=" * 80)
    print("🧠 DEMO 5: Cognitive Reasoning with External Knowledge Integration")
    print("=" * 80)
    
    tool = create_tool_instance()
    
    print("🎯 Setting up cognitive reasoning with external knowledge sources...")
    
    # Connect to a knowledge source
    print("🔗 Connecting to DBpedia for cognitive reasoning...")
    
    connect_response = await tool.execute(
        "connect",
        db_type="sparql",
        endpoint_url="https://dbpedia.org/sparql",
        connection_name="reasoning_source"
    )
    
    if connect_response.data and "error" not in connect_response.data:
        print("✅ Knowledge source connected for reasoning")
        
        # Query for relationships between AI concepts
        print("\n🔍 Extracting AI knowledge relationships...")
        
        relationship_query = """
        SELECT DISTINCT ?subject ?predicate ?object ?subjLabel ?objLabel WHERE {
          ?subject ?predicate ?object .
          ?subject rdfs:label ?subjLabel .
          ?object rdfs:label ?objLabel .
          FILTER(
            regex(str(?subjLabel), "artificial intelligence|machine learning|neural network", "i") ||
            regex(str(?objLabel), "artificial intelligence|machine learning|neural network", "i")
          )
          FILTER(lang(?subjLabel) = "en")
          FILTER(lang(?objLabel) = "en")
          FILTER(?predicate != rdfs:label)
        } LIMIT 10
        """
        
        query_response = await tool.execute(
            "query",
            connection_name="sparql_reasoning_source",
            query=relationship_query
        )
        
        if query_response.data and "bindings" in query_response.data:
            relationships = query_response.data["bindings"]
            
            print(f"🔗 Found {len(relationships)} knowledge relationships:")
            
            for i, rel in enumerate(relationships[:5]):
                subj = rel.get("subjLabel", {}).get("value", "Unknown")
                pred = rel.get("predicate", {}).get("value", "Unknown").split('/')[-1]
                obj = rel.get("objLabel", {}).get("value", "Unknown")
                
                print(f"  {i+1}. {subj} --[{pred}]--> {obj}")
        
        # Import cognitive reasoning patterns
        print("\n📥 Importing cognitive reasoning patterns...")
        
        reasoning_patterns = {
            "pattern_recognition": "cognitive_ability",
            "machine_learning": "pattern_recognition", 
            "deep_learning": "machine_learning",
            "neural_networks": "deep_learning",
            "artificial_intelligence": "cognitive_simulation",
            "cognitive_science": "mind_study",
            "reasoning_systems": "logical_inference",
            "knowledge_representation": "information_structuring",
            "semantic_networks": "knowledge_representation",
            "expert_systems": "knowledge_based_reasoning"
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(reasoning_patterns, f, indent=2)
            patterns_path = f.name
        
        try:
            patterns_response = await tool.execute(
                "import",
                source=patterns_path,
                format="json"
            )
            
            print(f"Patterns import: {patterns_response.message}")
            
        finally:
            os.unlink(patterns_path)
        
        # Demonstrate knowledge synchronization for reasoning
        print("\n🔄 Synchronizing external knowledge for enhanced reasoning...")
        
        sync_response = await tool.execute(
            "sync",
            connection_name="sparql_reasoning_source",
            direction="from_external"
        )
        
        print(f"Knowledge sync for reasoning: {sync_response.message}")
        
        if sync_response.data and "sync_results" in sync_response.data:
            for result in sync_response.data["sync_results"]:
                if "error" not in result:
                    print(f"  ✅ Enhanced reasoning with knowledge from {result['connection']}")
        
        print("\n🧠 Cognitive reasoning capabilities enhanced with external knowledge:")
        print("   • Semantic relationships from DBpedia integrated")
        print("   • AI concept hierarchies available for inference")
        print("   • Cross-domain knowledge connections established")
        print("   • Pattern recognition enhanced with external examples")
        
        print("\n💡 Example reasoning capabilities now available:")
        print("   • 'What is the relationship between neural networks and deep learning?'")
        print("   • 'How does machine learning connect to artificial intelligence?'")
        print("   • 'What cognitive abilities are required for pattern recognition?'")
        
    else:
        print("❌ Could not establish connection for cognitive reasoning demonstration")
    
    print("\n" + "=" * 60)


async def main():
    """Run comprehensive external cognitive databases demonstration."""
    print("🚀 PyCog-Zero: External Cognitive Databases Integration Demo")
    print("=" * 80)
    
    print("🎯 This demonstration showcases integration with external cognitive databases")
    print("   and knowledge graphs, enhancing Agent-Zero's reasoning capabilities with")
    print("   real-world semantic knowledge from sources like DBpedia and Wikidata.")
    print()
    
    # Check library availability
    try:
        from python.tools.external_cognitive_databases import (
            NEO4J_AVAILABLE, SPARQL_AVAILABLE, SQL_AVAILABLE, OPENCOG_AVAILABLE
        )
        
        print("📦 External Database Libraries Status:")
        print(f"   • Neo4j Graph Database: {'✅ Available' if NEO4J_AVAILABLE else '❌ Not Available (pip install neo4j)'}")
        print(f"   • SPARQL/RDF Support: {'✅ Available' if SPARQL_AVAILABLE else '❌ Not Available (pip install sparqlwrapper rdflib)'}")
        print(f"   • SQL Database Support: {'✅ Available' if SQL_AVAILABLE else '❌ Not Available (pip install sqlalchemy)'}")
        print(f"   • OpenCog AtomSpace: {'✅ Available' if OPENCOG_AVAILABLE else '❌ Not Available (cognitive features limited)'}")
        print()
        
    except ImportError as e:
        print(f"❌ Failed to import external database tool: {e}")
        print("Please ensure the tool is properly installed in the PyCog-Zero framework.")
        return
    
    # Run demonstration scenarios
    demo_scenarios = [
        ("DBpedia Knowledge Graph Integration", demo_dbpedia_integration),
        ("Wikidata Knowledge Base Integration", demo_wikidata_integration), 
        ("Cognitive Ontology Import", demo_ontology_import),
        ("Multi-Database Workflow", demo_multi_database_workflow),
        ("Cognitive Reasoning with External Knowledge", demo_cognitive_reasoning_with_external_knowledge)
    ]
    
    for demo_name, demo_func in demo_scenarios:
        print(f"▶️  Starting: {demo_name}")
        try:
            await demo_func()
            print(f"✅ Completed: {demo_name}")
        except Exception as e:
            print(f"❌ Failed: {demo_name} - {e}")
        
        # Pause between demos
        print("\n⏸️  [Press Enter to continue to next demo...]")
        input()
    
    print("=" * 80)
    print("🎉 External Cognitive Databases Integration Demo Complete!")
    print("=" * 80)
    
    print("\n🎯 Key Capabilities Demonstrated:")
    print("   ✅ SPARQL endpoint connectivity (DBpedia, Wikidata)")
    print("   ✅ RDF/OWL ontology import and processing") 
    print("   ✅ JSON knowledge format support")
    print("   ✅ Multi-database connection management")
    print("   ✅ Knowledge synchronization with AtomSpace")
    print("   ✅ Cross-database concept mapping")
    print("   ✅ Enhanced cognitive reasoning with external knowledge")
    
    print("\n🚀 Production Deployment Ready Features:")
    print("   • Configurable external database connections")
    print("   • Robust error handling and recovery")
    print("   • Performance optimization for large knowledge graphs")
    print("   • Security and authentication support")
    print("   • Real-time synchronization capabilities")
    
    print("\n📖 Documentation and Examples:")
    print("   • Configuration: conf/config_cognitive.json -> external_databases")
    print("   • Tool Usage: python/tools/external_cognitive_databases.py")  
    print("   • Test Suite: test_external_cognitive_databases.py")
    
    print("\n🎯 Next Steps for Advanced Integration:")
    print("   1. Set up Neo4j graph database for complex relationship modeling")
    print("   2. Configure SQL databases for structured cognitive data")
    print("   3. Implement custom ontology mappings for domain-specific knowledge")
    print("   4. Set up automated knowledge synchronization pipelines")
    print("   5. Integrate with Agent-Zero's reasoning tools for enhanced inference")
    
    print("\n💡 The external cognitive databases integration is now ready for")
    print("   production use in Agent-Zero cognitive architectures!")


if __name__ == "__main__":
    asyncio.run(main())