"""
PyCog-Zero AtomSpace Tool Integration Hub
Provides cross-tool data sharing and cognitive coordination using AtomSpace
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
import asyncio
from typing import Dict, Any, List

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - tool integration hub will use fallback mode")
    OPENCOG_AVAILABLE = False


class AtomSpaceToolHub(Tool):
    """Central hub for cross-tool data sharing and cognitive coordination."""
    
    # Class-level shared AtomSpace for cross-tool communication
    _shared_atomspace = None
    _hub_initialized = False
    
    @classmethod
    def get_shared_atomspace(cls):
        """Get the shared AtomSpace instance for cross-tool communication."""
        if not cls._hub_initialized and OPENCOG_AVAILABLE:
            try:
                cls._shared_atomspace = AtomSpace()
                initialize_opencog(cls._shared_atomspace)
                cls._hub_initialized = True
                print("✓ AtomSpace Tool Integration Hub initialized")
            except Exception as e:
                print(f"⚠️ AtomSpace Tool Hub initialization failed: {e}")
        return cls._shared_atomspace
    
    def __init__(self, agent, name: str, method: str | None, args: dict, message: str, loop_data, **kwargs):
        super().__init__(agent, name, method, args, message, loop_data, **kwargs)
        self.atomspace = self.get_shared_atomspace()
        self.initialized = self._hub_initialized
        
    async def execute(self, operation: str = "status", **kwargs):
        """Hub operations: status, share_data, retrieve_data, coordinate_tools, analyze_interactions"""
        
        if operation == "status":
            return await self.get_hub_status()
        elif operation == "share_data":
            return await self.share_tool_data(**kwargs)
        elif operation == "retrieve_data":
            return await self.retrieve_shared_data(**kwargs)
        elif operation == "coordinate_tools":
            return await self.coordinate_tool_execution(**kwargs)
        elif operation == "analyze_interactions":
            return await self.analyze_tool_interactions()
        else:
            return Response(
                message="Unknown hub operation. Available: status, share_data, retrieve_data, coordinate_tools, analyze_interactions",
                break_loop=False
            )
    
    async def get_hub_status(self):
        """Get status of the tool integration hub."""
        try:
            status_info = {
                "atomspace_available": OPENCOG_AVAILABLE,
                "hub_initialized": self.initialized,
                "shared_atomspace_active": self.atomspace is not None
            }
            
            if self.initialized and self.atomspace:
                status_info.update({
                    "total_atoms": len(self.atomspace),
                    "concept_nodes": len(self.atomspace.get_atoms_by_type(types.ConceptNode)),
                    "evaluation_links": len(self.atomspace.get_atoms_by_type(types.EvaluationLink)),
                    "tool_registrations": await self.count_tool_registrations(),
                    "shared_data_items": await self.count_shared_data_items()
                })
            
            return Response(
                message=f"Tool Integration Hub Status: {json.dumps(status_info, indent=2)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error getting hub status: {e}",
                break_loop=False
            )
    
    async def share_tool_data(self, tool_name: str, data_type: str, data: Dict[str, Any], **kwargs):
        """Share data from one tool with other tools via AtomSpace."""
        if not self.initialized:
            return Response(
                message="AtomSpace hub not available - data sharing skipped",
                break_loop=False
            )
        
        try:
            # Create tool concept
            tool_concept = self.atomspace.add_node(types.ConceptNode, f"tool_{tool_name}")
            
            # Create data type concept
            data_type_concept = self.atomspace.add_node(types.ConceptNode, f"data_type_{data_type}")
            
            # Create data container concept
            data_id = f"shared_data_{tool_name}_{data_type}_{len(self.atomspace)}"
            data_concept = self.atomspace.add_node(types.ConceptNode, data_id)
            
            # Link tool to data
            self.atomspace.add_link(
                types.EvaluationLink,
                [
                    self.atomspace.add_node(types.PredicateNode, "shares_data"),
                    tool_concept,
                    data_concept
                ]
            )
            
            # Link data to type
            self.atomspace.add_link(
                types.EvaluationLink,
                [
                    self.atomspace.add_node(types.PredicateNode, "has_data_type"),
                    data_concept,
                    data_type_concept
                ]
            )
            
            # Store data content as concepts
            data_concepts_created = 0
            for key, value in data.items():
                key_concept = self.atomspace.add_node(types.ConceptNode, f"key_{key}")
                value_concept = self.atomspace.add_node(types.ConceptNode, f"value_{str(value)[:50]}")  # Limit length
                
                # Link data to key-value pairs
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "contains_data"),
                        data_concept,
                        key_concept,
                        value_concept
                    ]
                )
                data_concepts_created += 1
            
            # Add timestamp metadata
            import datetime
            timestamp_concept = self.atomspace.add_node(
                types.ConceptNode, 
                f"timestamp_{datetime.datetime.now().isoformat()}"
            )
            self.atomspace.add_link(
                types.EvaluationLink,
                [
                    self.atomspace.add_node(types.PredicateNode, "created_at"),
                    data_concept,
                    timestamp_concept
                ]
            )
            
            return Response(
                message=f"Data shared successfully: {tool_name} -> {data_type}. "
                       f"Data ID: {data_id}, Concepts created: {data_concepts_created}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error sharing tool data: {e}",
                break_loop=False
            )
    
    async def retrieve_shared_data(self, tool_name: str = None, data_type: str = None, **kwargs):
        """Retrieve shared data from AtomSpace based on tool name or data type."""
        if not self.initialized:
            return Response(
                message="AtomSpace hub not available - data retrieval failed",
                break_loop=False
            )
        
        try:
            retrieved_data = {}
            
            # Find data concepts based on filters
            data_concepts = []
            
            if tool_name:
                # Find data shared by specific tool
                tool_concept = None
                for atom in self.atomspace.get_atoms_by_type(types.ConceptNode):
                    if hasattr(atom, 'name') and atom.name == f"tool_{tool_name}":
                        tool_concept = atom
                        break
                
                if tool_concept:
                    for link in tool_concept.incoming:
                        if (link.type == types.EvaluationLink and 
                            len(link.out) >= 3 and
                            hasattr(link.out[0], 'name') and 
                            link.out[0].name == "shares_data"):
                            data_concepts.append(link.out[2])
            
            elif data_type:
                # Find data of specific type
                for link in self.atomspace.get_atoms_by_type(types.EvaluationLink):
                    if (len(link.out) >= 3 and 
                        hasattr(link.out[0], 'name') and 
                        link.out[0].name == "has_data_type" and
                        hasattr(link.out[2], 'name') and 
                        link.out[2].name == f"data_type_{data_type}"):
                        data_concepts.append(link.out[1])
            
            else:
                # Find all shared data
                for link in self.atomspace.get_atoms_by_type(types.EvaluationLink):
                    if (len(link.out) >= 3 and 
                        hasattr(link.out[0], 'name') and 
                        link.out[0].name == "shares_data"):
                        data_concepts.append(link.out[2])
            
            # Extract data from concepts
            for data_concept in data_concepts[:10]:  # Limit to 10 results
                if hasattr(data_concept, 'name'):
                    data_id = data_concept.name
                    retrieved_data[data_id] = {}
                    
                    # Find data content
                    for link in data_concept.incoming:
                        if (link.type == types.EvaluationLink and 
                            len(link.out) >= 4 and
                            hasattr(link.out[0], 'name') and 
                            link.out[0].name == "contains_data"):
                            key = link.out[2].name if hasattr(link.out[2], 'name') else "unknown_key"
                            value = link.out[3].name if hasattr(link.out[3], 'name') else "unknown_value"
                            
                            # Clean up the key/value prefixes
                            if key.startswith("key_"):
                                key = key[4:]
                            if value.startswith("value_"):
                                value = value[6:]
                            
                            retrieved_data[data_id][key] = value
            
            return Response(
                message=f"Retrieved shared data. Items found: {len(retrieved_data)}. "
                       f"Data: {json.dumps(retrieved_data, indent=2) if retrieved_data else 'No data found'}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error retrieving shared data: {e}",
                break_loop=False
            )
    
    async def coordinate_tool_execution(self, primary_tool: str, supporting_tools: List[str], task_description: str, **kwargs):
        """Coordinate execution between multiple tools using AtomSpace."""
        if not self.initialized:
            return Response(
                message="AtomSpace hub not available - tool coordination limited",
                break_loop=False
            )
        
        try:
            # Create task concept
            task_concept = self.atomspace.add_node(types.ConceptNode, f"task_{task_description.replace(' ', '_')}")
            
            # Create primary tool concept
            primary_tool_concept = self.atomspace.add_node(types.ConceptNode, f"tool_{primary_tool}")
            
            # Link primary tool to task
            self.atomspace.add_link(
                types.EvaluationLink,
                [
                    self.atomspace.add_node(types.PredicateNode, "primary_tool_for"),
                    primary_tool_concept,
                    task_concept
                ]
            )
            
            # Create supporting tool concepts and links
            coordination_plan = []
            for supporting_tool in supporting_tools:
                support_tool_concept = self.atomspace.add_node(types.ConceptNode, f"tool_{supporting_tool}")
                
                # Link supporting tool to task
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "supports_task"),
                        support_tool_concept,
                        task_concept
                    ]
                )
                
                # Create coordination relationship
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "coordinates_with"),
                        primary_tool_concept,
                        support_tool_concept
                    ]
                )
                
                coordination_plan.append(f"{supporting_tool} supports {primary_tool}")
            
            # Create execution strategy based on available data
            execution_strategy = await self.create_execution_strategy(primary_tool, supporting_tools, task_description)
            
            coordination_result = {
                "task": task_description,
                "primary_tool": primary_tool,
                "supporting_tools": supporting_tools,
                "coordination_plan": coordination_plan,
                "execution_strategy": execution_strategy,
                "coordination_id": task_concept.name if hasattr(task_concept, 'name') else "unknown"
            }
            
            return Response(
                message=f"Tool coordination plan created: {json.dumps(coordination_result, indent=2)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error coordinating tools: {e}",
                break_loop=False
            )
    
    async def create_execution_strategy(self, primary_tool: str, supporting_tools: List[str], task_description: str):
        """Create an execution strategy based on tool capabilities and past interactions."""
        try:
            strategy = []
            
            # Analyze past tool interactions
            tool_interactions = await self.analyze_tool_interaction_patterns()
            
            # Basic strategy based on tool types
            if "memory" in primary_tool.lower():
                strategy.append("1. Initialize memory system")
                if "search" in [t.lower() for t in supporting_tools]:
                    strategy.append("2. Use search to gather relevant information")
                    strategy.append("3. Store search results in memory")
                if "document" in [t.lower() for t in supporting_tools]:
                    strategy.append("4. Process documents and extract knowledge")
                    strategy.append("5. Update memory with document insights")
            
            elif "search" in primary_tool.lower():
                strategy.append("1. Execute search query")
                if "memory" in [t.lower() for t in supporting_tools]:
                    strategy.append("2. Cross-reference with existing memory")
                    strategy.append("3. Store enhanced results")
                if "document" in [t.lower() for t in supporting_tools]:
                    strategy.append("4. Analyze found documents for deeper insights")
            
            elif "code" in primary_tool.lower():
                strategy.append("1. Analyze code structure and context")
                if "memory" in [t.lower() for t in supporting_tools]:
                    strategy.append("2. Reference past code patterns from memory")
                    strategy.append("3. Store execution context for future reference")
            
            else:
                strategy.append("1. Execute primary tool operation")
                strategy.append("2. Coordinate with supporting tools as needed")
                strategy.append("3. Share results through AtomSpace hub")
            
            # Add coordination steps
            strategy.append("4. Share intermediate results via AtomSpace")
            strategy.append("5. Coordinate final output synthesis")
            
            return strategy
            
        except Exception as e:
            return [f"Strategy generation failed: {e}"]
    
    async def analyze_tool_interactions(self):
        """Analyze interactions between tools stored in AtomSpace."""
        if not self.initialized:
            return Response(
                message="AtomSpace hub not available - interaction analysis skipped",
                break_loop=False
            )
        
        try:
            interaction_analysis = {}
            
            # Find all tool concepts
            tool_concepts = [
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if hasattr(atom, 'name') and atom.name.startswith("tool_")
            ]
            
            interaction_analysis["total_tools"] = len(tool_concepts)
            
            # Analyze coordination relationships
            coordination_links = []
            for link in self.atomspace.get_atoms_by_type(types.EvaluationLink):
                if (len(link.out) >= 3 and 
                    hasattr(link.out[0], 'name') and 
                    link.out[0].name == "coordinates_with"):
                    tool1 = link.out[1].name if hasattr(link.out[1], 'name') else "unknown"
                    tool2 = link.out[2].name if hasattr(link.out[2], 'name') else "unknown"
                    coordination_links.append(f"{tool1} <-> {tool2}")
            
            interaction_analysis["coordination_relationships"] = len(coordination_links)
            interaction_analysis["coordination_pairs"] = coordination_links[:10]  # Top 10
            
            # Analyze data sharing patterns
            sharing_patterns = {}
            for link in self.atomspace.get_atoms_by_type(types.EvaluationLink):
                if (len(link.out) >= 3 and 
                    hasattr(link.out[0], 'name') and 
                    link.out[0].name == "shares_data"):
                    tool = link.out[1].name if hasattr(link.out[1], 'name') else "unknown"
                    sharing_patterns[tool] = sharing_patterns.get(tool, 0) + 1
            
            interaction_analysis["data_sharing_patterns"] = sharing_patterns
            
            # Analyze task coordination
            task_concepts = [
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if hasattr(atom, 'name') and atom.name.startswith("task_")
            ]
            interaction_analysis["coordinated_tasks"] = len(task_concepts)
            
            return Response(
                message=f"Tool interaction analysis completed:\n{json.dumps(interaction_analysis, indent=2)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error analyzing tool interactions: {e}",
                break_loop=False
            )
    
    async def analyze_tool_interaction_patterns(self):
        """Internal method to analyze tool interaction patterns for strategy creation."""
        try:
            patterns = {}
            
            if not self.initialized:
                return patterns
            
            # Analyze coordination frequency
            for link in self.atomspace.get_atoms_by_type(types.EvaluationLink):
                if (len(link.out) >= 3 and 
                    hasattr(link.out[0], 'name') and 
                    link.out[0].name == "coordinates_with"):
                    tool1 = link.out[1].name if hasattr(link.out[1], 'name') else "unknown"
                    tool2 = link.out[2].name if hasattr(link.out[2], 'name') else "unknown"
                    
                    pair = tuple(sorted([tool1, tool2]))
                    patterns[pair] = patterns.get(pair, 0) + 1
            
            return patterns
            
        except Exception as e:
            print(f"Warning: Tool interaction pattern analysis failed: {e}")
            return {}
    
    async def count_tool_registrations(self):
        """Count the number of registered tools in AtomSpace."""
        try:
            if not self.initialized:
                return 0
            
            tool_count = len([
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if hasattr(atom, 'name') and atom.name.startswith("tool_")
            ])
            return tool_count
            
        except Exception as e:
            return 0
    
    async def count_shared_data_items(self):
        """Count the number of shared data items in AtomSpace."""
        try:
            if not self.initialized:
                return 0
            
            data_count = len([
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if hasattr(atom, 'name') and atom.name.startswith("shared_data_")
            ])
            return data_count
            
        except Exception as e:
            return 0


def register():
    """Register the AtomSpace Tool Integration Hub with Agent-Zero."""
    return AtomSpaceToolHub