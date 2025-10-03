"""
External Cognitive Databases and Knowledge Graph Integration Tool for PyCog-Zero

This tool provides integration with external cognitive databases and knowledge graphs,
enabling seamless data exchange between Agent-Zero's AtomSpace and external systems:

- Neo4j graph database connectivity with Cypher queries
- SPARQL endpoints for RDF/OWL ontology access
- Generic SQL database integration for structured knowledge
- Knowledge graph synchronization and mapping
- External ontology import and reasoning capabilities
"""

from python.helpers.tool import Tool
from python.helpers.tool import Response as BaseResponse

# Extended Response class for this tool
class Response(BaseResponse):
    def __init__(self, message: str, break_loop: bool = False, data: dict = None):
        super().__init__(message, break_loop)
        self.data = data or {}
from python.helpers import files
import json
import asyncio
import time
from typing import Dict, Any, List, Optional, Union
import logging

# Database connectivity imports with graceful fallbacks
try:
    from neo4j import GraphDatabase
    from neo4j.exceptions import ServiceUnavailable, AuthError
    NEO4J_AVAILABLE = True
except ImportError:
    NEO4J_AVAILABLE = False
    print("Neo4j driver not available - install with: pip install neo4j")

try:
    from SPARQLWrapper import SPARQLWrapper, JSON, XML, RDF
    import rdflib
    from rdflib import Graph, Namespace, URIRef, Literal
    from rdflib.plugins.stores.sparqlstore import SPARQLStore
    SPARQL_AVAILABLE = True
except ImportError:
    SPARQL_AVAILABLE = False
    print("SPARQL/RDF libraries not available - install with: pip install sparqlwrapper rdflib")

try:
    import sqlalchemy
    from sqlalchemy import create_engine, text
    from sqlalchemy.orm import sessionmaker
    SQL_AVAILABLE = True
except ImportError:
    SQL_AVAILABLE = False
    print("SQLAlchemy not available - install with: pip install sqlalchemy")

# OpenCog AtomSpace integration
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    OPENCOG_AVAILABLE = False

# Internal tool integration
try:
    from python.tools.cognitive_reasoning import CognitiveReasoningTool
    from python.tools.atomspace_tool_hub import AtomSpaceToolHub
    INTERNAL_TOOLS_AVAILABLE = True
except ImportError:
    INTERNAL_TOOLS_AVAILABLE = False


class ExternalCognitiveDatabasesTool(Tool):
    """Agent-Zero tool for integrating external cognitive databases and knowledge graphs."""
    
    def __init__(self, agent, name="external_cognitive_databases", method=None, args=None, message="", loop_data=None, **kwargs):
        super().__init__(agent, name, method, args or {}, message, loop_data, **kwargs)
        self.description = "Integrate with external cognitive databases and knowledge graphs (Neo4j, SPARQL, RDF/OWL, SQL)"
        
        # Initialize connections storage
        self.connections = {}
        self.atomspace = None
        self.hub = None
        
        # Configuration from cognitive config
        self.config = self._load_config()
        self._initialize_if_needed()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load external database configuration from cognitive config."""
        try:
            config_file = files.get_abs_path("conf/config_cognitive.json")
            with open(config_file, 'r') as f:
                base_config = json.load(f)
            
            # Default external database configuration
            external_config = base_config.get("external_databases", {
                "enabled": True,
                "neo4j": {
                    "enabled": False,
                    "uri": "bolt://localhost:7687",
                    "username": "neo4j", 
                    "password": "password",
                    "database": "neo4j"
                },
                "sparql": {
                    "enabled": False,
                    "endpoints": [
                        "https://dbpedia.org/sparql",
                        "https://query.wikidata.org/sparql"
                    ],
                    "default_graphs": [],
                    "timeout": 30
                },
                "sql_databases": {
                    "enabled": False,
                    "connections": {}
                },
                "synchronization": {
                    "auto_sync": False,
                    "sync_interval": 300,  # 5 minutes
                    "bidirectional": True,
                    "conflict_resolution": "latest_wins"
                },
                "knowledge_mapping": {
                    "auto_map_concepts": True,
                    "create_missing_relations": True,
                    "confidence_threshold": 0.7
                }
            })
            
            return external_config
            
        except Exception as e:
            print(f"Warning: Could not load external database config: {e}")
            return {"enabled": False}
    
    def _initialize_if_needed(self):
        """Initialize AtomSpace and connections if not already done."""
        if not self.config.get("enabled", False):
            return
            
        # Initialize AtomSpace for cognitive integration
        if OPENCOG_AVAILABLE and not self.atomspace:
            self.atomspace = AtomSpace()
            initialize_opencog(self.atomspace)
        
        # Initialize tool hub integration
        if INTERNAL_TOOLS_AVAILABLE and not self.hub:
            try:
                self.hub = AtomSpaceToolHub.get_instance()
            except Exception as e:
                print(f"Warning: Could not initialize AtomSpace tool hub: {e}")
    
    async def execute(self, operation: str, **kwargs) -> Response:
        """Execute external database operations."""
        
        if not self.config.get("enabled", False):
            return Response(
                message="External cognitive databases integration is disabled",
                data={"error": "disabled", "available_operations": []}
            )
        
        operations = {
            "connect": self.connect_to_database,
            "query": self.query_external_database,
            "sync": self.synchronize_knowledge,
            "import": self.import_external_knowledge,
            "export": self.export_to_external,
            "map": self.map_knowledge_concepts,
            "status": self.get_connection_status,
            "list_connections": self.list_connections,
            "disconnect": self.disconnect_database,
            "test_connection": self.test_connection
        }
        
        if operation not in operations:
            return Response(
                message=f"Unknown operation: {operation}",
                data={
                    "error": "unknown_operation", 
                    "available_operations": list(operations.keys())
                }
            )
        
        try:
            return await operations[operation](**kwargs)
        except Exception as e:
            return Response(
                message=f"Error executing {operation}: {str(e)}",
                data={"error": str(e), "operation": operation}
            )
    
    async def connect_to_database(self, db_type: str, **connection_params) -> Response:
        """Connect to an external cognitive database."""
        
        if db_type == "neo4j":
            return await self._connect_neo4j(**connection_params)
        elif db_type == "sparql":
            return await self._connect_sparql(**connection_params)
        elif db_type == "sql":
            return await self._connect_sql(**connection_params)
        else:
            return Response(
                message=f"Unsupported database type: {db_type}",
                data={"error": "unsupported_type", "supported_types": ["neo4j", "sparql", "sql"]}
            )
    
    async def _connect_neo4j(self, uri: str = None, username: str = None, 
                           password: str = None, database: str = "neo4j",
                           connection_name: str = "default") -> Response:
        """Connect to Neo4j graph database."""
        
        if not NEO4J_AVAILABLE:
            return Response(
                message="Neo4j driver not available. Install with: pip install neo4j",
                data={"error": "driver_unavailable"}
            )
        
        # Use config defaults if not provided
        neo4j_config = self.config.get("neo4j", {})
        uri = uri or neo4j_config.get("uri", "bolt://localhost:7687")
        username = username or neo4j_config.get("username", "neo4j")
        password = password or neo4j_config.get("password", "password")
        
        try:
            driver = GraphDatabase.driver(uri, auth=(username, password))
            
            # Test connection
            with driver.session(database=database) as session:
                result = session.run("RETURN 1 as test")
                test_value = result.single()["test"]
                
            self.connections[f"neo4j_{connection_name}"] = {
                "type": "neo4j",
                "driver": driver,
                "database": database,
                "uri": uri,
                "connected_at": time.time()
            }
            
            return Response(
                message=f"Successfully connected to Neo4j database: {connection_name}",
                data={
                    "connection_name": connection_name,
                    "type": "neo4j",
                    "uri": uri,
                    "database": database,
                    "test_result": test_value
                }
            )
            
        except (ServiceUnavailable, AuthError) as e:
            return Response(
                message=f"Failed to connect to Neo4j: {str(e)}",
                data={"error": "connection_failed", "details": str(e)}
            )
    
    async def _connect_sparql(self, endpoint_url: str = None, 
                            connection_name: str = "default") -> Response:
        """Connect to SPARQL endpoint."""
        
        if not SPARQL_AVAILABLE:
            return Response(
                message="SPARQL libraries not available. Install with: pip install sparqlwrapper rdflib",
                data={"error": "libraries_unavailable"}
            )
        
        # Use config defaults if not provided
        sparql_config = self.config.get("sparql", {})
        endpoint_url = endpoint_url or sparql_config.get("endpoints", ["https://dbpedia.org/sparql"])[0]
        
        try:
            sparql = SPARQLWrapper(endpoint_url)
            sparql.setReturnFormat(JSON)
            
            # Test connection with a simple query
            sparql.setQuery("SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 1")
            results = sparql.query().convert()
            
            self.connections[f"sparql_{connection_name}"] = {
                "type": "sparql",
                "wrapper": sparql,
                "endpoint": endpoint_url,
                "connected_at": time.time()
            }
            
            return Response(
                message=f"Successfully connected to SPARQL endpoint: {connection_name}",
                data={
                    "connection_name": connection_name,
                    "type": "sparql", 
                    "endpoint": endpoint_url,
                    "test_results_count": len(results.get("results", {}).get("bindings", []))
                }
            )
            
        except Exception as e:
            return Response(
                message=f"Failed to connect to SPARQL endpoint: {str(e)}",
                data={"error": "connection_failed", "details": str(e)}
            )
    
    async def _connect_sql(self, connection_string: str, 
                         connection_name: str = "default") -> Response:
        """Connect to SQL database."""
        
        if not SQL_AVAILABLE:
            return Response(
                message="SQLAlchemy not available. Install with: pip install sqlalchemy",
                data={"error": "library_unavailable"}
            )
        
        try:
            engine = create_engine(connection_string)
            Session = sessionmaker(bind=engine)
            
            # Test connection
            with Session() as session:
                result = session.execute(text("SELECT 1 as test"))
                test_value = result.scalar()
            
            self.connections[f"sql_{connection_name}"] = {
                "type": "sql",
                "engine": engine,
                "session_factory": Session,
                "connection_string": connection_string,
                "connected_at": time.time()
            }
            
            return Response(
                message=f"Successfully connected to SQL database: {connection_name}",
                data={
                    "connection_name": connection_name,
                    "type": "sql",
                    "test_result": test_value
                }
            )
            
        except Exception as e:
            return Response(
                message=f"Failed to connect to SQL database: {str(e)}",
                data={"error": "connection_failed", "details": str(e)}
            )
    
    async def query_external_database(self, connection_name: str, query: str, 
                                    parameters: Dict = None, **kwargs) -> Response:
        """Execute query on external database."""
        
        if connection_name not in self.connections:
            return Response(
                message=f"Connection '{connection_name}' not found",
                data={"error": "connection_not_found", "available_connections": list(self.connections.keys())}
            )
        
        connection = self.connections[connection_name]
        parameters = parameters or {}
        
        try:
            if connection["type"] == "neo4j":
                return await self._query_neo4j(connection, query, parameters)
            elif connection["type"] == "sparql":
                return await self._query_sparql(connection, query, parameters)
            elif connection["type"] == "sql":
                return await self._query_sql(connection, query, parameters)
            else:
                return Response(
                    message=f"Unsupported connection type: {connection['type']}",
                    data={"error": "unsupported_type"}
                )
                
        except Exception as e:
            return Response(
                message=f"Query execution failed: {str(e)}",
                data={"error": "query_failed", "details": str(e)}
            )
    
    async def _query_neo4j(self, connection: Dict, query: str, parameters: Dict) -> Response:
        """Execute Cypher query on Neo4j."""
        
        driver = connection["driver"]
        database = connection["database"]
        
        with driver.session(database=database) as session:
            result = session.run(query, parameters)
            records = [record.data() for record in result]
        
        return Response(
            message=f"Neo4j query executed successfully. Retrieved {len(records)} records.",
            data={
                "records": records,
                "count": len(records),
                "query": query,
                "parameters": parameters
            }
        )
    
    async def _query_sparql(self, connection: Dict, query: str, parameters: Dict) -> Response:
        """Execute SPARQL query."""
        
        sparql = connection["wrapper"]
        
        # Apply parameters if provided (simple string replacement)
        formatted_query = query
        if parameters:
            for key, value in parameters.items():
                formatted_query = formatted_query.replace(f"${key}", str(value))
        
        sparql.setQuery(formatted_query)
        results = sparql.query().convert()
        
        bindings = results.get("results", {}).get("bindings", [])
        
        return Response(
            message=f"SPARQL query executed successfully. Retrieved {len(bindings)} bindings.",
            data={
                "bindings": bindings,
                "count": len(bindings),
                "query": formatted_query,
                "original_query": query,
                "parameters": parameters
            }
        )
    
    async def _query_sql(self, connection: Dict, query: str, parameters: Dict) -> Response:
        """Execute SQL query."""
        
        Session = connection["session_factory"]
        
        with Session() as session:
            result = session.execute(text(query), parameters)
            
            if result.returns_rows:
                rows = [dict(row._mapping) for row in result]
                return Response(
                    message=f"SQL query executed successfully. Retrieved {len(rows)} rows.",
                    data={
                        "rows": rows,
                        "count": len(rows),
                        "query": query,
                        "parameters": parameters
                    }
                )
            else:
                return Response(
                    message=f"SQL command executed successfully. Affected {result.rowcount} rows.",
                    data={
                        "rowcount": result.rowcount,
                        "query": query,
                        "parameters": parameters
                    }
                )
    
    async def synchronize_knowledge(self, connection_name: str = None, 
                                  direction: str = "bidirectional", **kwargs) -> Response:
        """Synchronize knowledge between AtomSpace and external database."""
        
        if not OPENCOG_AVAILABLE:
            return Response(
                message="OpenCog AtomSpace not available for synchronization",
                data={"error": "atomspace_unavailable"}
            )
        
        if not self.atomspace:
            self._initialize_if_needed()
        
        sync_results = []
        connections_to_sync = [connection_name] if connection_name else list(self.connections.keys())
        
        for conn_name in connections_to_sync:
            if conn_name not in self.connections:
                continue
                
            try:
                connection = self.connections[conn_name]
                
                if direction in ["bidirectional", "from_external"]:
                    # Import from external to AtomSpace
                    import_result = await self._import_to_atomspace(connection, conn_name)
                    sync_results.append({"connection": conn_name, "direction": "import", "result": import_result})
                
                if direction in ["bidirectional", "to_external"]:
                    # Export from AtomSpace to external
                    export_result = await self._export_from_atomspace(connection, conn_name)
                    sync_results.append({"connection": conn_name, "direction": "export", "result": export_result})
                    
            except Exception as e:
                sync_results.append({"connection": conn_name, "error": str(e)})
        
        return Response(
            message=f"Knowledge synchronization completed for {len(connections_to_sync)} connections.",
            data={
                "sync_results": sync_results,
                "connections_processed": len(connections_to_sync),
                "direction": direction
            }
        )
    
    async def _import_to_atomspace(self, connection: Dict, connection_name: str) -> Dict:
        """Import knowledge from external database to AtomSpace."""
        
        imported_atoms = 0
        
        if connection["type"] == "neo4j":
            # Import Neo4j nodes and relationships as ConceptNodes and Links
            query = "MATCH (n)-[r]->(m) RETURN n, r, m LIMIT 100"
            result = await self._query_neo4j(connection, query, {})
            
            for record in result.data.get("records", []):
                # Create atoms for nodes and relationships
                if "n" in record:
                    node_atom = self.atomspace.add_node(types.ConceptNode, f"neo4j_{record['n'].get('id', 'unknown')}")
                    imported_atoms += 1
                
                if "m" in record:
                    node_atom = self.atomspace.add_node(types.ConceptNode, f"neo4j_{record['m'].get('id', 'unknown')}")
                    imported_atoms += 1
        
        elif connection["type"] == "sparql":
            # Import SPARQL results as ConceptNodes and Links
            query = "SELECT ?s ?p ?o WHERE { ?s ?p ?o } LIMIT 100"
            result = await self._query_sparql(connection, query, {})
            
            for binding in result.data.get("bindings", []):
                if "s" in binding and "o" in binding:
                    subject_atom = self.atomspace.add_node(types.ConceptNode, binding["s"]["value"])
                    object_atom = self.atomspace.add_node(types.ConceptNode, binding["o"]["value"])
                    
                    if "p" in binding:
                        # Create relationship link
                        predicate_atom = self.atomspace.add_node(types.PredicateNode, binding["p"]["value"])
                        link_atom = self.atomspace.add_link(types.EvaluationLink, [predicate_atom, subject_atom, object_atom])
                        imported_atoms += 3
        
        return {"imported_atoms": imported_atoms, "connection": connection_name}
    
    async def _export_from_atomspace(self, connection: Dict, connection_name: str) -> Dict:
        """Export knowledge from AtomSpace to external database."""
        
        exported_items = 0
        
        # Get all ConceptNodes from AtomSpace
        concept_atoms = self.atomspace.get_atoms_by_type(types.ConceptNode)
        
        if connection["type"] == "neo4j" and len(concept_atoms) > 0:
            # Export concepts as Neo4j nodes
            driver = connection["driver"]
            database = connection["database"]
            
            with driver.session(database=database) as session:
                for atom in concept_atoms[:50]:  # Limit for demo
                    query = "MERGE (n:Concept {name: $name, atomspace_id: $id})"
                    session.run(query, {"name": atom.name, "id": str(atom)})
                    exported_items += 1
        
        return {"exported_items": exported_items, "connection": connection_name}
    
    async def import_external_knowledge(self, source: str, format: str = "auto", **kwargs) -> Response:
        """Import knowledge from external sources (files, URLs, etc.)."""
        
        if format == "rdf" or format == "owl":
            return await self._import_rdf_owl(source, **kwargs)
        elif format == "json":
            return await self._import_json_knowledge(source, **kwargs)
        else:
            # Auto-detect format based on source
            if source.endswith(('.rdf', '.owl', '.ttl')):
                return await self._import_rdf_owl(source, **kwargs)
            elif source.endswith('.json'):
                return await self._import_json_knowledge(source, **kwargs)
            else:
                return Response(
                    message=f"Unsupported knowledge format for source: {source}",
                    data={"error": "unsupported_format", "source": source}
                )
    
    async def _import_rdf_owl(self, source: str, **kwargs) -> Response:
        """Import RDF/OWL ontology knowledge."""
        
        if not SPARQL_AVAILABLE:
            return Response(
                message="RDF/OWL libraries not available",
                data={"error": "libraries_unavailable"}
            )
        
        try:
            g = rdflib.Graph()
            g.parse(source)
            
            imported_triples = 0
            
            if OPENCOG_AVAILABLE and self.atomspace:
                # Convert RDF triples to AtomSpace atoms
                for subj, pred, obj in g:
                    subject_atom = self.atomspace.add_node(types.ConceptNode, str(subj))
                    predicate_atom = self.atomspace.add_node(types.PredicateNode, str(pred))
                    object_atom = self.atomspace.add_node(types.ConceptNode, str(obj))
                    
                    # Create evaluation link
                    link_atom = self.atomspace.add_link(types.EvaluationLink, 
                                                       [predicate_atom, subject_atom, object_atom])
                    imported_triples += 1
            
            return Response(
                message=f"Successfully imported {imported_triples} RDF triples from {source}",
                data={
                    "imported_triples": imported_triples,
                    "source": source,
                    "total_graph_size": len(g)
                }
            )
            
        except Exception as e:
            return Response(
                message=f"Failed to import RDF/OWL knowledge: {str(e)}",
                data={"error": "import_failed", "details": str(e)}
            )
    
    async def _import_json_knowledge(self, source: str, **kwargs) -> Response:
        """Import knowledge from JSON format."""
        
        try:
            if source.startswith(('http://', 'https://')):
                # Load from URL (would need requests library)
                return Response(
                    message="URL import not implemented yet",
                    data={"error": "not_implemented"}
                )
            else:
                # Load from file
                with open(source, 'r') as f:
                    data = json.load(f)
            
            imported_concepts = 0
            
            if OPENCOG_AVAILABLE and self.atomspace and isinstance(data, dict):
                # Simple JSON to AtomSpace conversion
                for key, value in data.items():
                    concept_atom = self.atomspace.add_node(types.ConceptNode, key)
                    if isinstance(value, str):
                        value_atom = self.atomspace.add_node(types.ConceptNode, value)
                        # Create simple association
                        link_atom = self.atomspace.add_link(types.AssociativeLink, [concept_atom, value_atom])
                        imported_concepts += 1
            
            return Response(
                message=f"Successfully imported {imported_concepts} concepts from JSON",
                data={"imported_concepts": imported_concepts, "source": source}
            )
            
        except Exception as e:
            return Response(
                message=f"Failed to import JSON knowledge: {str(e)}",
                data={"error": "import_failed", "details": str(e)}
            )
    
    async def export_to_external(self, connection_name: str, format: str = "native", **kwargs) -> Response:
        """Export AtomSpace knowledge to external database."""
        
        if connection_name not in self.connections:
            return Response(
                message=f"Connection '{connection_name}' not found",
                data={"error": "connection_not_found"}
            )
        
        return await self._export_from_atomspace(self.connections[connection_name], connection_name)
    
    async def map_knowledge_concepts(self, source_connection: str, target_connection: str, **kwargs) -> Response:
        """Map and align concepts between different external databases."""
        
        mapping_config = self.config.get("knowledge_mapping", {})
        auto_map = mapping_config.get("auto_map_concepts", True)
        confidence_threshold = mapping_config.get("confidence_threshold", 0.7)
        
        if source_connection not in self.connections or target_connection not in self.connections:
            return Response(
                message="Source or target connection not found",
                data={"error": "connection_not_found"}
            )
        
        # Simple concept mapping (would need more sophisticated logic in production)
        mapped_concepts = []
        
        # Get concepts from source
        source_concepts = await self._get_concepts_from_connection(source_connection)
        target_concepts = await self._get_concepts_from_connection(target_connection)
        
        # Simple string matching for demonstration
        for source_concept in source_concepts[:10]:  # Limit for demo
            for target_concept in target_concepts[:10]:
                if source_concept.lower() in target_concept.lower() or target_concept.lower() in source_concept.lower():
                    mapped_concepts.append({
                        "source": source_concept,
                        "target": target_concept,
                        "confidence": 0.8  # Placeholder confidence
                    })
        
        return Response(
            message=f"Mapped {len(mapped_concepts)} concepts between {source_connection} and {target_connection}",
            data={
                "mapped_concepts": mapped_concepts,
                "source_connection": source_connection,
                "target_connection": target_connection,
                "confidence_threshold": confidence_threshold
            }
        )
    
    async def _get_concepts_from_connection(self, connection_name: str) -> List[str]:
        """Get concept names from a database connection."""
        
        connection = self.connections.get(connection_name)
        if not connection:
            return []
        
        concepts = []
        
        try:
            if connection["type"] == "neo4j":
                result = await self._query_neo4j(connection, "MATCH (n) RETURN DISTINCT labels(n) as labels LIMIT 50", {})
                for record in result.data.get("records", []):
                    if "labels" in record:
                        concepts.extend(record["labels"])
            
            elif connection["type"] == "sparql":
                result = await self._query_sparql(connection, "SELECT DISTINCT ?type WHERE { ?s a ?type } LIMIT 50", {})
                for binding in result.data.get("bindings", []):
                    if "type" in binding:
                        concepts.append(binding["type"]["value"])
        
        except Exception as e:
            print(f"Error getting concepts from {connection_name}: {e}")
        
        return list(set(concepts))  # Remove duplicates
    
    async def get_connection_status(self, connection_name: str = None) -> Response:
        """Get status of database connections."""
        
        if connection_name:
            if connection_name not in self.connections:
                return Response(
                    message=f"Connection '{connection_name}' not found",
                    data={"error": "connection_not_found"}
                )
            
            connection = self.connections[connection_name]
            status = {
                "name": connection_name,
                "type": connection["type"],
                "connected_at": connection["connected_at"],
                "active": True  # Simple check, could be more sophisticated
            }
        else:
            # Return status of all connections
            status = {}
            for name, conn in self.connections.items():
                status[name] = {
                    "type": conn["type"],
                    "connected_at": conn["connected_at"],
                    "active": True
                }
        
        return Response(
            message=f"Connection status retrieved",
            data={"status": status, "total_connections": len(self.connections)}
        )
    
    async def list_connections(self) -> Response:
        """List all available database connections."""
        
        connections_info = []
        for name, conn in self.connections.items():
            info = {
                "name": name,
                "type": conn["type"],
                "connected_at": conn["connected_at"]
            }
            
            # Add type-specific info
            if conn["type"] == "neo4j":
                info["uri"] = conn["uri"]
                info["database"] = conn["database"]
            elif conn["type"] == "sparql":
                info["endpoint"] = conn["endpoint"]
            elif conn["type"] == "sql":
                info["connection_string"] = conn["connection_string"][:50] + "..." if len(conn["connection_string"]) > 50 else conn["connection_string"]
            
            connections_info.append(info)
        
        return Response(
            message=f"Listed {len(connections_info)} active connections",
            data={
                "connections": connections_info,
                "count": len(connections_info)
            }
        )
    
    async def disconnect_database(self, connection_name: str) -> Response:
        """Disconnect from an external database."""
        
        if connection_name not in self.connections:
            return Response(
                message=f"Connection '{connection_name}' not found",
                data={"error": "connection_not_found"}
            )
        
        connection = self.connections[connection_name]
        
        try:
            # Close connection based on type
            if connection["type"] == "neo4j":
                connection["driver"].close()
            elif connection["type"] == "sql":
                connection["engine"].dispose()
            
            # Remove from connections
            del self.connections[connection_name]
            
            return Response(
                message=f"Successfully disconnected from {connection_name}",
                data={"disconnected_connection": connection_name}
            )
            
        except Exception as e:
            return Response(
                message=f"Error disconnecting from {connection_name}: {str(e)}",
                data={"error": "disconnect_failed", "details": str(e)}
            )
    
    async def test_connection(self, connection_name: str) -> Response:
        """Test an existing database connection."""
        
        if connection_name not in self.connections:
            return Response(
                message=f"Connection '{connection_name}' not found",
                data={"error": "connection_not_found"}
            )
        
        connection = self.connections[connection_name]
        
        try:
            if connection["type"] == "neo4j":
                result = await self._query_neo4j(connection, "RETURN 1 as test", {})
                test_result = result.data.get("records", [{}])[0].get("test", None)
            elif connection["type"] == "sparql":
                result = await self._query_sparql(connection, "SELECT (1 as ?test) WHERE {}", {})
                test_result = len(result.data.get("bindings", [])) >= 0
            elif connection["type"] == "sql":
                result = await self._query_sql(connection, "SELECT 1 as test", {})
                test_result = result.data.get("rows", [{}])[0].get("test", None)
            else:
                test_result = None
            
            return Response(
                message=f"Connection test successful for {connection_name}",
                data={
                    "connection_name": connection_name,
                    "type": connection["type"],
                    "test_result": test_result,
                    "status": "healthy"
                }
            )
            
        except Exception as e:
            return Response(
                message=f"Connection test failed for {connection_name}: {str(e)}",
                data={"error": "test_failed", "details": str(e)}
            )


def register():
    """Register the external cognitive databases tool with Agent-Zero."""
    return ExternalCognitiveDatabasesTool