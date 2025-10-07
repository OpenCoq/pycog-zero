# External Cognitive Databases and Knowledge Graphs Integration

This document describes the external cognitive databases and knowledge graphs integration in PyCog-Zero, enabling Agent-Zero to connect with and reason over external knowledge sources.

## Overview

The External Cognitive Databases tool provides seamless integration between Agent-Zero's AtomSpace and external knowledge sources:

- **Neo4j Graph Databases**: Connect to Neo4j instances for complex relationship modeling
- **SPARQL Endpoints**: Query semantic web endpoints (DBpedia, Wikidata, custom endpoints)  
- **RDF/OWL Ontologies**: Import and process ontological knowledge
- **SQL Databases**: Connect to structured relational databases
- **Knowledge Synchronization**: Bidirectional sync between AtomSpace and external sources
- **Concept Mapping**: Automatic alignment of concepts across knowledge sources

## Configuration

### Basic Setup

External database integration is configured in `conf/config_cognitive.json`:

```json
{
  "external_databases": {
    "enabled": true,
    "neo4j": {
      "enabled": false,
      "uri": "bolt://localhost:7687",
      "username": "neo4j", 
      "password": "password",
      "database": "neo4j"
    },
    "sparql": {
      "enabled": true,
      "endpoints": [
        "https://dbpedia.org/sparql",
        "https://query.wikidata.org/sparql"
      ],
      "default_graphs": [],
      "timeout": 30
    },
    "sql_databases": {
      "enabled": false,
      "connections": {}
    },
    "synchronization": {
      "auto_sync": false,
      "sync_interval": 300,
      "bidirectional": true,
      "conflict_resolution": "latest_wins"
    },
    "knowledge_mapping": {
      "auto_map_concepts": true,
      "create_missing_relations": true,
      "confidence_threshold": 0.7
    }
  }
}
```

### Dependencies

Install required dependencies:

```bash
# Core external database libraries
pip install neo4j==5.27.0 rdflib==7.2.1 SPARQLWrapper==2.0.0

# Optional SQL database support  
pip install sqlalchemy psycopg2-binary  # PostgreSQL
pip install sqlalchemy pymysql  # MySQL
```

## Usage Examples

### Basic Connection Management

```python
from python.tools.external_cognitive_databases import ExternalCognitiveDatabasesTool

# Initialize tool
tool = ExternalCognitiveDatabasesTool(agent)

# Connect to SPARQL endpoint
response = await tool.execute(
    "connect", 
    db_type="sparql",
    endpoint_url="https://dbpedia.org/sparql", 
    connection_name="dbpedia"
)

# List active connections
response = await tool.execute("list_connections")
print(f"Active connections: {response.data['connections']}")

# Test connection health
response = await tool.execute("test_connection", connection_name="sparql_dbpedia")
```

### Neo4j Graph Database Integration

```python
# Connect to Neo4j
response = await tool.execute(
    "connect",
    db_type="neo4j",
    uri="bolt://localhost:7687",
    username="neo4j",
    password="password",
    connection_name="knowledge_graph"
)

# Query with Cypher
cypher_query = """
MATCH (agent:CognitiveAgent)-[r:HAS_CAPABILITY]->(capability:Capability)
RETURN agent.name, capability.name, r.confidence
"""

response = await tool.execute(
    "query",
    connection_name="neo4j_knowledge_graph",
    query=cypher_query
)

# Access results
records = response.data["records"]
for record in records:
    print(f"Agent: {record['agent.name']}, Capability: {record['capability.name']}")
```

### SPARQL Semantic Web Queries

```python
# Query DBpedia for AI researchers
sparql_query = """
SELECT DISTINCT ?person ?name ?field WHERE {
  ?person a dbo:Person .
  ?person rdfs:label ?name .
  ?person dbo:field ?field .
  FILTER(regex(str(?field), "artificial|intelligence|machine", "i"))
  FILTER(lang(?name) = "en")
} LIMIT 10
"""

response = await tool.execute(
    "query",
    connection_name="sparql_dbpedia",
    query=sparql_query
)

# Process results
bindings = response.data["bindings"]
for binding in bindings:
    name = binding["name"]["value"]
    field = binding["field"]["value"]
    print(f"Researcher: {name}, Field: {field}")
```

### RDF/OWL Ontology Import

```python
# Import RDF ontology
response = await tool.execute(
    "import",
    source="/path/to/cognitive_ontology.owl",
    format="rdf"
)

# Import JSON knowledge
response = await tool.execute(
    "import", 
    source="/path/to/knowledge.json",
    format="json"
)

print(f"Imported {response.data['imported_triples']} RDF triples")
```

### Knowledge Synchronization

```python
# Sync from external database to AtomSpace
response = await tool.execute(
    "sync",
    connection_name="sparql_dbpedia", 
    direction="from_external"
)

# Bidirectional sync
response = await tool.execute(
    "sync",
    connection_name="neo4j_knowledge_graph",
    direction="bidirectional"
)

# Check sync results
for result in response.data["sync_results"]:
    print(f"Sync {result['connection']}: {result['direction']} - {result['result']['imported_atoms']} atoms")
```

### Cross-Database Concept Mapping

```python
# Map concepts between databases
response = await tool.execute(
    "map",
    source_connection="sparql_dbpedia",
    target_connection="neo4j_knowledge_graph"
)

# View mappings
mappings = response.data["mapped_concepts"]
for mapping in mappings:
    print(f"Mapping: {mapping['source']} <-> {mapping['target']} (confidence: {mapping['confidence']})")
```

## Advanced Features

### Custom SPARQL Endpoints

```python
# Connect to custom endpoint
response = await tool.execute(
    "connect",
    db_type="sparql",
    endpoint_url="https://your-organization.com/sparql",
    connection_name="custom_endpoint"
)

# Query with parameters
response = await tool.execute(
    "query", 
    connection_name="sparql_custom_endpoint",
    query="SELECT ?s ?p ?o WHERE { ?s ?p ?o . FILTER(?s = $subject) }",
    parameters={"subject": "http://example.org/CognitiveAgent"}
)
```

### SQL Database Integration

```python
# Connect to PostgreSQL
response = await tool.execute(
    "connect",
    db_type="sql",
    connection_string="postgresql://user:pass@localhost/cognitive_db",
    connection_name="postgres"
)

# Query structured data
response = await tool.execute(
    "query",
    connection_name="sql_postgres", 
    query="SELECT * FROM knowledge_concepts WHERE domain = :domain",
    parameters={"domain": "artificial_intelligence"}
)
```

### Real-time Knowledge Updates

```python
# Enable auto-sync
tool.config["synchronization"]["auto_sync"] = True
tool.config["synchronization"]["sync_interval"] = 60  # 1 minute

# Manual sync trigger
response = await tool.execute("sync", connection_name="all")
```

## Integration with Agent-Zero Reasoning

### Enhanced Cognitive Queries

The external database integration enhances Agent-Zero's reasoning capabilities:

```python
# Use external knowledge in cognitive reasoning
from python.tools.cognitive_reasoning import CognitiveReasoningTool

cognitive_tool = CognitiveReasoningTool(agent)

# Reasoning query enhanced with external knowledge
response = await cognitive_tool.execute(
    "What are the key capabilities of modern AI systems?",
    hints=["use_external_knowledge", "dbpedia_integration"],
    external_sources=["sparql_dbpedia", "neo4j_knowledge_graph"]
)
```

### AtomSpace Integration

External knowledge is automatically converted to AtomSpace atoms:

```python
# Query AtomSpace for external knowledge
from opencog.atomspace import AtomSpace, types

atomspace = tool.atomspace

# Find concepts imported from external sources  
external_concepts = atomspace.get_atoms_by_type(types.ConceptNode)
dbpedia_concepts = [atom for atom in external_concepts if "dbpedia_" in atom.name]

print(f"Found {len(dbpedia_concepts)} concepts from DBpedia")
```

## Performance Optimization

### Connection Pooling

```python
# Configure connection pooling for high-volume usage
tool.config["performance"] = {
    "connection_pooling": True,
    "max_connections_per_endpoint": 10,
    "connection_timeout": 30,
    "query_cache_size": 1000
}
```

### Query Caching

```python
# Enable query result caching
tool.config["caching"] = {
    "enabled": True, 
    "cache_duration": 3600,  # 1 hour
    "max_cache_size": "100MB"
}
```

## Security Considerations

### Authentication

```python
# Neo4j with authentication
response = await tool.execute(
    "connect",
    db_type="neo4j",
    uri="bolt://secure-server:7687",
    username="cognitive_agent",
    password="secure_password",
    connection_name="secure_graph"
)

# SQL with SSL
response = await tool.execute(
    "connect", 
    db_type="sql",
    connection_string="postgresql://user:pass@server/db?sslmode=require",
    connection_name="secure_sql"
)
```

### Access Control

```python
# Configure access permissions
tool.config["security"] = {
    "allowed_endpoints": [
        "https://dbpedia.org/sparql",
        "https://query.wikidata.org/sparql"
    ],
    "blocked_queries": ["DROP", "DELETE", "INSERT"],
    "max_query_complexity": 1000
}
```

## Troubleshooting

### Common Issues

1. **Connection Timeouts**:
   ```python
   # Increase timeout for slow endpoints
   tool.config["sparql"]["timeout"] = 60
   ```

2. **Memory Usage**:
   ```python
   # Limit result set size
   response = await tool.execute(
       "query",
       connection_name="sparql_dbpedia",
       query="SELECT * WHERE { ?s ?p ?o } LIMIT 1000"  # Add LIMIT
   )
   ```

3. **Rate Limiting**:
   ```python
   # Add delays between queries
   import asyncio
   await asyncio.sleep(1)  # Wait 1 second between queries
   ```

### Error Handling

```python
try:
    response = await tool.execute("connect", db_type="sparql", endpoint_url="invalid-url")
except Exception as e:
    print(f"Connection failed: {e}")
    
# Check response for errors
if response.data and "error" in response.data:
    print(f"Error: {response.data['error']}")
```

## Testing

Run the comprehensive test suite:

```bash
# Run all external database tests
python test_external_cognitive_databases.py

# Run interactive demo
python demo_external_cognitive_databases.py
```

## API Reference

### Main Operations

- `connect(db_type, **params)`: Connect to external database
- `query(connection_name, query, parameters)`: Execute query
- `sync(connection_name, direction)`: Synchronize knowledge
- `import(source, format)`: Import knowledge from file/URL
- `map(source_connection, target_connection)`: Map concepts
- `list_connections()`: List active connections
- `test_connection(connection_name)`: Test connection health
- `disconnect(connection_name)`: Disconnect from database

### Supported Formats

- **SPARQL**: `db_type="sparql"`
- **Neo4j**: `db_type="neo4j"` 
- **SQL**: `db_type="sql"`
- **RDF/OWL**: `format="rdf"`
- **JSON**: `format="json"`

### Configuration Options

- `enabled`: Enable/disable external database integration
- `auto_sync`: Enable automatic synchronization
- `sync_interval`: Sync frequency in seconds
- `confidence_threshold`: Minimum confidence for concept mapping
- `timeout`: Query timeout in seconds

## Future Enhancements

- **Graph Neural Networks**: Integration with GNN models for enhanced reasoning
- **Federated Learning**: Distributed learning across multiple knowledge sources
- **Automated Ontology Alignment**: AI-powered concept mapping
- **Real-time Stream Processing**: Live knowledge updates from streaming sources
- **Blockchain Integration**: Decentralized knowledge verification and provenance

## Contributing

When adding new external database integrations:

1. Extend the `ExternalCognitiveDatabasesTool` class
2. Add connection methods for new database types
3. Implement query translation and result mapping
4. Add comprehensive tests and documentation
5. Update configuration schema

The external cognitive databases integration enables Agent-Zero to leverage the world's knowledge for enhanced reasoning and decision-making capabilities.