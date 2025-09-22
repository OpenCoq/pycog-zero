"""
PyCog-Zero Cognitive Reasoning Tool
Integrates OpenCog cognitive architecture with Agent-Zero framework
Enhanced with new atomspace bindings and cross-tool integration
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
import asyncio
from typing import Dict, Any, List, Optional

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - install with: pip install opencog-atomspace opencog-python")
    OPENCOG_AVAILABLE = False

# Import other atomspace tools for enhanced integration
try:
    from python.tools.atomspace_tool_hub import AtomSpaceToolHub
    from python.tools.atomspace_memory_bridge import AtomSpaceMemoryBridge
    ATOMSPACE_TOOLS_AVAILABLE = True
except ImportError:
    ATOMSPACE_TOOLS_AVAILABLE = False


class CognitiveReasoningTool(Tool):
    """Agent-Zero tool for OpenCog cognitive reasoning with enhanced atomspace bindings."""
    
    # Class-level shared atomspace for cross-tool integration
    _shared_atomspace = None
    
    def _initialize_if_needed(self):
        """Initialize the cognitive reasoning system if not already done."""
        if hasattr(self, '_cognitive_initialized'):
            return
        
        self._cognitive_initialized = True
        self.atomspace = None
        self.initialized = False
        self.config = self._load_cognitive_config()
        self.tool_hub = None
        self.memory_bridge = None
        
        # Initialize with shared or new atomspace
        if OPENCOG_AVAILABLE and self.config.get("opencog_enabled", True):
            try:
                # Try to use shared atomspace from tool hub
                if ATOMSPACE_TOOLS_AVAILABLE:
                    shared_atomspace = AtomSpaceToolHub.get_shared_atomspace()
                    if shared_atomspace:
                        self.atomspace = shared_atomspace
                        print("✓ Using shared AtomSpace from tool hub")
                    else:
                        self.atomspace = self._create_new_atomspace()
                else:
                    self.atomspace = self._create_new_atomspace()
                
                if self.atomspace:
                    self.initialized = True
                    print("✓ OpenCog cognitive reasoning initialized with enhanced atomspace bindings")
                    
                    # Initialize cross-tool integration
                    self._setup_cross_tool_integration()
                    
            except Exception as e:
                print(f"⚠️ OpenCog initialization failed: {e}")
                self._setup_fallback_mode()
        else:
            self._setup_fallback_mode()
    
    def _create_new_atomspace(self):
        """Create a new AtomSpace instance."""
        try:
            atomspace = AtomSpace()
            initialize_opencog(atomspace)
            
            # Set as shared atomspace if none exists
            if not CognitiveReasoningTool._shared_atomspace:
                CognitiveReasoningTool._shared_atomspace = atomspace
            
            return atomspace
        except Exception as e:
            print(f"⚠️ Failed to create new AtomSpace: {e}")
            return None
    
    def _setup_cross_tool_integration(self):
        """Setup integration with other atomspace tools."""
        if not ATOMSPACE_TOOLS_AVAILABLE:
            return
        
        try:
            # Initialize tool hub reference for data sharing
            if hasattr(self, 'agent'):
                # Create dummy parameters for tool initialization
                dummy_params = {
                    'name': 'cognitive_reasoning_hub_integration',
                    'method': None,
                    'args': {},
                    'message': '',
                    'loop_data': None
                }
                self.tool_hub = AtomSpaceToolHub(self.agent, **dummy_params)
                
                # Register this tool with the hub
                if self.tool_hub.initialized:
                    asyncio.create_task(self._register_with_hub())
            
        except Exception as e:
            print(f"⚠️ Cross-tool integration setup failed: {e}")
    
    async def _register_with_hub(self):
        """Register this cognitive reasoning tool with the AtomSpace hub."""
        try:
            registration_data = {
                "tool_type": "cognitive_reasoning",
                "capabilities": ["reasoning", "pattern_matching", "inference"],
                "atomspace_operations": ["create_atoms", "query_patterns", "inference_chains"],
                "status": "active"
            }
            
            await self.tool_hub.share_tool_data(
                tool_name="cognitive_reasoning",
                data_type="registration",
                data=registration_data
            )
        except Exception as e:
            print(f"⚠️ Tool registration failed: {e}")
    
    def _setup_fallback_mode(self):
        """Setup fallback mode when OpenCog is not available."""
        self.atomspace = None
        self.initialized = False
        print("⚠️ Running in fallback mode - limited cognitive reasoning capabilities")
    
    def _load_cognitive_config(self):
        """Load cognitive configuration from Agent-Zero settings."""
        try:
            # Try to import settings and get cognitive config
            from python.helpers import settings
            return settings.get_cognitive_config()
        except Exception:
            # Fallback to direct config file loading
            try:
                config_file = files.get_abs_path("conf/config_cognitive.json")
                with open(config_file, 'r') as f:
                    return json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cognitive config: {e}")
                return {
                    "cognitive_mode": True,
                    "opencog_enabled": True,
                    "neural_symbolic_bridge": True,
                    "reasoning_config": {
                        "pln_enabled": True,
                        "pattern_matching": True,
                        "forward_chaining": False,
                        "backward_chaining": True
                    },
                    "atomspace_config": {
                        "persistence_backend": "memory",
                        "cross_tool_sharing": True,
                        "attention_allocation": "basic"
                    }
                }
    
    async def execute(self, query: str, operation: str = "reason", **kwargs):
        """Execute cognitive reasoning on Agent-Zero queries with enhanced atomspace operations."""
        
        # Initialize if needed
        self._initialize_if_needed()
        
        if not self.config.get("cognitive_mode", True):
            return Response(
                message="Cognitive mode is disabled\\n"
                       f"Data: {json.dumps({'error': 'Cognitive mode disabled in configuration'})}",
                break_loop=False
            )
        
        # Handle different reasoning operations
        if operation == "reason":
            return await self._perform_reasoning(query, **kwargs)
        elif operation == "analyze_patterns":
            return await self._analyze_patterns(query, **kwargs)
        elif operation == "cross_reference":
            return await self._cross_reference_with_tools(query, **kwargs)
        elif operation == "status":
            return await self._get_reasoning_status()
        elif operation == "share_knowledge":
            return await self._share_knowledge_with_hub(query, **kwargs)
        else:
            # Default to reasoning
            return await self._perform_reasoning(query, **kwargs)
    
    async def _perform_reasoning(self, query: str, **kwargs):
        """Perform core cognitive reasoning operations."""
        if not self.initialized:
            return await self._fallback_reasoning(query, **kwargs)
        
        try:
            # Enhanced reasoning with cross-tool integration
            reasoning_context = await self._build_reasoning_context(query, **kwargs)
            
            # Convert natural language query to AtomSpace representation
            query_atoms = self.parse_query_to_atoms(query, reasoning_context)
            
#<<<<<<< copilot/fix-41
            # Perform multi-modal reasoning based on configuration
            reasoning_results = await self._execute_enhanced_reasoning(query_atoms, reasoning_context)
#=======
            # Check if atomspace-rocks optimization is available
            storage_optimization = self._get_storage_optimization_info()
            
            # Perform reasoning based on configuration
            reasoning_results = self.execute_reasoning(query_atoms)
#>>>>>>> main
            
            # Format results for Agent-Zero consumption
            reasoning_steps = self.format_reasoning_for_agent(reasoning_results)
            
            # Share results with other tools if enabled
            if self.config.get("atomspace_config", {}).get("cross_tool_sharing", True):
                await self._share_reasoning_results(query, reasoning_results)
            
            return Response(
                message=f"Enhanced cognitive reasoning completed for: {query}\\n"
                       f"Data: {json.dumps({
                           'query': query,
                           'operation': 'reason',
                           'atoms_created': len(query_atoms),
                           'reasoning_steps': reasoning_steps,
                           'context_size': len(reasoning_context.get('related_concepts', [])),
                           'cross_tool_integration': ATOMSPACE_TOOLS_AVAILABLE,
                           'status': 'success',
                           'config': {
                               'pln_enabled': self.config.get('reasoning_config', {}).get('pln_enabled', True),
                               'pattern_matching': self.config.get('reasoning_config', {}).get('pattern_matching', True),
                               'cross_tool_sharing': self.config.get('atomspace_config', {}).get('cross_tool_sharing', True)
                           }
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Enhanced cognitive reasoning error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error', 'fallback_available': True})}",
                break_loop=False
            )
    
    async def _fallback_reasoning(self, query: str, **kwargs):
        """Fallback reasoning when OpenCog is not available."""
        try:
            # Simple pattern-based reasoning as fallback
            query_words = query.lower().split()
            patterns = []
            
            # Basic pattern recognition
            if any(word in query_words for word in ["what", "how", "why", "when", "where"]):
                patterns.append("question_pattern")
            
            if any(word in query_words for word in ["learn", "understand", "know"]):
                patterns.append("learning_pattern")
            
            if any(word in query_words for word in ["because", "since", "therefore"]):
                patterns.append("causal_pattern")
            
            reasoning_steps = [
                f"Identified patterns: {', '.join(patterns) if patterns else 'general_inquiry'}",
                f"Query analysis: {len(query_words)} words, {len(patterns)} patterns",
                "Fallback reasoning: basic linguistic analysis performed"
            ]
            
            return Response(
                message=f"Fallback cognitive reasoning completed for: {query}\\n"
                       f"Data: {json.dumps({
                           'query': query,
                           'operation': 'fallback_reason',
                           'patterns_identified': patterns,
                           'reasoning_steps': reasoning_steps,
                           'status': 'fallback_success',
                           'note': 'Limited reasoning without OpenCog - install opencog-atomspace for full capabilities'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Fallback reasoning error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'fallback_error'})}",
                break_loop=False
            )
    
    async def _build_reasoning_context(self, query: str, **kwargs) -> Dict[str, Any]:
        """Build enhanced reasoning context using cross-tool integration."""
        context = {
            "query": query,
            "related_concepts": [],
            "memory_associations": [],
            "tool_data": {},
            "reasoning_hints": kwargs.get("hints", [])
        }
        
        try:
            # Get related concepts from memory bridge if available
            if self.tool_hub and self.tool_hub.initialized:
                # Retrieve relevant shared data
                shared_data_response = await self.tool_hub.retrieve_shared_data(
                    data_type="knowledge"
                )
                
                if "Data:" in shared_data_response.message:
                    context["tool_data"]["shared_knowledge"] = "Available"
                
                # Get memory associations if memory bridge is integrated
                try:
                    memory_data_response = await self.tool_hub.retrieve_shared_data(
                        tool_name="memory_bridge"
                    )
                    if "Data:" in memory_data_response.message:
                        context["memory_associations"] = ["memory_bridge_integrated"]
                except Exception:
                    pass
            
            # Extract concepts from query for reasoning
            query_words = query.lower().split()
            for word in query_words:
                if len(word) > 3:  # Filter meaningful words
                    context["related_concepts"].append(word)
            
        except Exception as e:
            print(f"⚠️ Context building warning: {e}")
        
        return context
    
    async def _execute_enhanced_reasoning(self, atoms: List, context: Dict[str, Any]) -> List:
        """Execute enhanced reasoning with multiple strategies."""
        results = []
        reasoning_config = self.config.get("reasoning_config", {})
        
        try:
            # Pattern matching reasoning (enhanced)
            if reasoning_config.get("pattern_matching", True):
                pattern_results = self.enhanced_pattern_matching_reasoning(atoms, context)
                results.extend(pattern_results)
            
            # PLN reasoning (enhanced)
            if reasoning_config.get("pln_enabled", True):
                pln_results = self.enhanced_pln_reasoning(atoms, context)
                results.extend(pln_results)
            
            # Backward chaining if enabled
            if reasoning_config.get("backward_chaining", True):
                backward_results = self.backward_chaining_reasoning(atoms, context)
                results.extend(backward_results)
            
            # Cross-tool reasoning integration
            if ATOMSPACE_TOOLS_AVAILABLE and context.get("tool_data"):
                cross_tool_results = await self.cross_tool_reasoning(atoms, context)
                results.extend(cross_tool_results)
        
        except Exception as e:
            print(f"⚠️ Enhanced reasoning warning: {e}")
            # Fallback to basic reasoning
            results.extend(self.pattern_matching_reasoning(atoms))
        
        return results
    
    def parse_query_to_atoms(self, query: str, context: Dict[str, Any] = None):
        """Convert Agent-Zero query to OpenCog atoms with enhanced context integration."""
        if not self.initialized:
            return []
        
        context = context or {}
        atoms = []
        
        try:
            # Enhanced query parsing with context
            words = query.lower().split()
            related_concepts = context.get("related_concepts", [])
            
            # Create concept nodes from query words and context
            for word in words:
                if len(word) > 2:  # Skip short words
                    concept_node = self.atomspace.add_node(types.ConceptNode, word)
                    atoms.append(concept_node)
            
            # Add concepts from reasoning context
            for concept in related_concepts[:5]:  # Limit to 5 context concepts
                if concept not in [atom.name for atom in atoms if hasattr(atom, 'name')]:
                    context_node = self.atomspace.add_node(types.ConceptNode, f"context_{concept}")
                    atoms.append(context_node)
            
            # Create query concept for the entire query
            query_concept = self.atomspace.add_node(types.ConceptNode, f"query_{hash(query) % 10000}")
            atoms.append(query_concept)
            
            # Link query concept to component words
            for word_atom in atoms[:-1]:  # Exclude the query concept itself
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "component_of"),
                        word_atom,
                        query_concept
                    ]
                )
        
        except Exception as e:
            print(f"⚠️ Query parsing warning: {e}")
        
        return atoms
    
#<<<<<<< copilot/fix-41
    def enhanced_pattern_matching_reasoning(self, atoms: List, context: Dict[str, Any]) -> List:
        """Enhanced pattern matching reasoning with context awareness."""
#=======
    def _get_storage_optimization_info(self):
        """Get information about atomspace-rocks storage optimization availability."""
        try:
            # Try to import atomspace-rocks optimizer
            from python.tools.atomspace_rocks_optimizer import AtomSpaceRocksOptimizer
            from python.helpers.enhanced_atomspace_rocks import get_rocks_storage_info
            
            rocks_info = get_rocks_storage_info()
            return {
                "atomspace_rocks_available": True,
                "optimizer_available": True,
                "rocks_info": rocks_info,
                "integration_ready": True
            }
        except ImportError:
            return {
                "atomspace_rocks_available": False,
                "optimizer_available": False,
                "integration_ready": False,
                "message": "AtomSpace-Rocks optimization not available"
            }
    
    def execute_reasoning(self, atoms):
        """Perform reasoning operations based on configuration."""
        if not atoms:
            return []
        
#>>>>>>> main
        results = []
        
        try:
            # Create enhanced inheritance relationships
            for i in range(len(atoms) - 1):
                # Basic inheritance
                inheritance_link = self.atomspace.add_link(
                    types.InheritanceLink, 
                    [atoms[i], atoms[i + 1]]
                )
                results.append(inheritance_link)
                
                # Add similarity relationships for context concepts
                if i < len(atoms) - 2:
                    similarity_link = self.atomspace.add_link(
                        types.SimilarityLink,
                        [atoms[i], atoms[i + 2]]
                    )
                    results.append(similarity_link)
            
            # Context-based pattern matching
            memory_associations = context.get("memory_associations", [])
            for association in memory_associations[:3]:  # Limit to 3
                association_node = self.atomspace.add_node(types.ConceptNode, f"memory_{association}")
                for atom in atoms[:2]:  # Link to first 2 atoms
                    association_link = self.atomspace.add_link(
                        types.EvaluationLink,
                        [
                            self.atomspace.add_node(types.PredicateNode, "associated_with"),
                            atom,
                            association_node
                        ]
                    )
                    results.append(association_link)
        
        except Exception as e:
            print(f"⚠️ Enhanced pattern matching warning: {e}")
            # Fallback to basic pattern matching
            results.extend(self.pattern_matching_reasoning(atoms))
        
        return results
    
    def enhanced_pln_reasoning(self, atoms: List, context: Dict[str, Any]) -> List:
        """Enhanced PLN reasoning with probabilistic truth values."""
        results = []
        
        try:
            # Create evaluation links with enhanced truth values
            for atom in atoms:
                # Basic relevance evaluation
                relevance_link = self.atomspace.add_link(
                    types.EvaluationLink,
                    [self.atomspace.add_node(types.PredicateNode, "relevant"), atom]
                )
                results.append(relevance_link)
                
                # Context-based confidence evaluation
                if context.get("reasoning_hints"):
                    confidence_link = self.atomspace.add_link(
                        types.EvaluationLink,
                        [
                            self.atomspace.add_node(types.PredicateNode, "confidence_high"),
                            atom
                        ]
                    )
                    results.append(confidence_link)
                
                # Cross-tool integration markers
                if context.get("tool_data"):
                    integration_link = self.atomspace.add_link(
                        types.EvaluationLink,
                        [
                            self.atomspace.add_node(types.PredicateNode, "cross_tool_relevant"),
                            atom
                        ]
                    )
                    results.append(integration_link)
        
        except Exception as e:
            print(f"⚠️ Enhanced PLN reasoning warning: {e}")
            # Fallback to basic PLN
            results.extend(self.pln_reasoning(atoms))
        
        return results
    
    def backward_chaining_reasoning(self, atoms: List, context: Dict[str, Any]) -> List:
        """Backward chaining reasoning implementation."""
        results = []
        
        try:
            # Implement backward chaining for goal-directed reasoning
            if atoms:
                goal_atom = atoms[-1]  # Use last atom as goal
                
                # Create reasoning chain backwards
                for i, atom in enumerate(reversed(atoms[:-1])):
                    step_predicate = self.atomspace.add_node(
                        types.PredicateNode, 
                        f"reasoning_step_{len(atoms) - i}"
                    )
                    
                    chain_link = self.atomspace.add_link(
                        types.EvaluationLink,
                        [step_predicate, atom, goal_atom]
                    )
                    results.append(chain_link)
                
                # Add goal achievement link
                achievement_link = self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "achieves_goal"),
                        atoms[0] if atoms else goal_atom,
                        goal_atom
                    ]
                )
                results.append(achievement_link)
        
        except Exception as e:
            print(f"⚠️ Backward chaining warning: {e}")
        
        return results
    
    async def cross_tool_reasoning(self, atoms: List, context: Dict[str, Any]) -> List:
        """Reasoning that integrates with other atomspace tools."""
        results = []
        
        try:
            if self.tool_hub and self.tool_hub.initialized:
                # Share reasoning atoms with tool hub
                reasoning_data = {
                    "atoms_count": len(atoms),
                    "reasoning_type": "cognitive_reasoning",
                    "context_concepts": context.get("related_concepts", [])[:3],
                    "timestamp": str(asyncio.get_event_loop().time())
                }
                
                await self.tool_hub.share_tool_data(
                    tool_name="cognitive_reasoning",
                    data_type="reasoning_session",
                    data=reasoning_data
                )
                
                # Create cross-tool integration atoms
                for i, atom in enumerate(atoms[:3]):  # Limit to first 3
                    cross_tool_node = self.atomspace.add_node(
                        types.ConceptNode, 
                        f"cross_tool_concept_{i}"
                    )
                    
                    integration_link = self.atomspace.add_link(
                        types.EvaluationLink,
                        [
                            self.atomspace.add_node(types.PredicateNode, "shared_with_hub"),
                            atom,
                            cross_tool_node
                        ]
                    )
                    results.append(integration_link)
        
        except Exception as e:
            print(f"⚠️ Cross-tool reasoning warning: {e}")
        
        return results
    
    def pattern_matching_reasoning(self, atoms):
        """Basic pattern matching reasoning (preserved for backward compatibility)."""
        results = []
        
        try:
            # Create simple inheritance relationships between concepts
            for i in range(len(atoms) - 1):
                inheritance_link = self.atomspace.add_link(
                    types.InheritanceLink, 
                    [atoms[i], atoms[i + 1]]
                )
                results.append(inheritance_link)
        except Exception as e:
            print(f"⚠️ Basic pattern matching warning: {e}")
        
        return results
    
    def pln_reasoning(self, atoms):
        """Basic PLN reasoning (preserved for backward compatibility)."""
        results = []
        
        try:
            # Create evaluation links with truth values
            for atom in atoms:
                evaluation_link = self.atomspace.add_link(
                    types.EvaluationLink,
                    [self.atomspace.add_node(types.PredicateNode, "relevant"), atom]
                )
                results.append(evaluation_link)
        except Exception as e:
            print(f"⚠️ Basic PLN reasoning warning: {e}")
        
        return results
    
    async def _analyze_patterns(self, query: str, **kwargs):
        """Analyze patterns in the query and atomspace."""
        if not self.initialized:
            return await self._fallback_pattern_analysis(query)
        
        try:
            # Build context for pattern analysis
            context = await self._build_reasoning_context(query, **kwargs)
            atoms = self.parse_query_to_atoms(query, context)
            
            # Analyze patterns in atomspace
            pattern_analysis = {
                "query_patterns": [],
                "atomspace_patterns": [],
                "cross_references": [],
                "pattern_strength": 0.0
            }
            
            # Query pattern analysis
            query_words = query.lower().split()
            if any(word in query_words for word in ["what", "how", "why"]):
                pattern_analysis["query_patterns"].append("interrogative")
            
            if any(word in query_words for word in ["because", "therefore", "since"]):
                pattern_analysis["query_patterns"].append("causal")
            
            # AtomSpace pattern analysis
            if atoms:
                concept_nodes = [atom for atom in atoms if atom.type == types.ConceptNode]
                pattern_analysis["atomspace_patterns"].append(f"concept_nodes: {len(concept_nodes)}")
                
                # Find existing patterns in atomspace
                inheritance_links = self.atomspace.get_atoms_by_type(types.InheritanceLink)
                pattern_analysis["atomspace_patterns"].append(f"inheritance_patterns: {len(inheritance_links)}")
                
                evaluation_links = self.atomspace.get_atoms_by_type(types.EvaluationLink)
                pattern_analysis["atomspace_patterns"].append(f"evaluation_patterns: {len(evaluation_links)}")
                
                # Calculate pattern strength
                total_atoms = len(self.atomspace)
                pattern_analysis["pattern_strength"] = len(atoms) / max(total_atoms, 1)
            
            return Response(
                message=f"Pattern analysis completed for: {query}\\n"
                       f"Data: {json.dumps({
                           'query': query,
                           'operation': 'analyze_patterns',
                           'analysis': pattern_analysis,
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Pattern analysis error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error'})}",
                break_loop=False
            )
    
    async def _fallback_pattern_analysis(self, query: str):
        """Fallback pattern analysis without OpenCog."""
        try:
            query_words = query.lower().split()
            patterns = {
                "word_count": len(query_words),
                "question_words": [w for w in query_words if w in ["what", "how", "why", "when", "where"]],
                "causal_words": [w for w in query_words if w in ["because", "since", "therefore"]],
                "action_words": [w for w in query_words if w in ["do", "make", "create", "build"]]
            }
            
            return Response(
                message=f"Fallback pattern analysis for: {query}\\n"
                       f"Data: {json.dumps({
                           'query': query,
                           'operation': 'fallback_analyze_patterns',
                           'patterns': patterns,
                           'status': 'fallback_success'
                       })}",
                break_loop=False
            )
        except Exception as e:
            return Response(
                message=f"Fallback pattern analysis error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'fallback_error'})}",
                break_loop=False
            )
    
    async def _cross_reference_with_tools(self, query: str, **kwargs):
        """Cross-reference query with other atomspace tools."""
        try:
            if not ATOMSPACE_TOOLS_AVAILABLE:
                return Response(
                    message="Cross-referencing not available - atomspace tools not loaded\\n"
                           f"Data: {json.dumps({'status': 'tools_unavailable'})}",
                    break_loop=False
                )
            
            if not self.tool_hub or not self.tool_hub.initialized:
                return Response(
                    message="Tool hub not available for cross-referencing\\n"
                           f"Data: {json.dumps({'status': 'hub_unavailable'})}",
                    break_loop=False
                )
            
            # Coordinate with other tools
            supporting_tools = kwargs.get("supporting_tools", ["memory_bridge", "search_engine"])
            
            coordination_response = await self.tool_hub.coordinate_tool_execution(
                primary_tool="cognitive_reasoning",
                supporting_tools=supporting_tools,
                task_description=f"Cross-reference analysis for: {query}"
            )
            
            # Retrieve shared data from other tools
            shared_data_response = await self.tool_hub.retrieve_shared_data(
                data_type="knowledge"
            )
            
            cross_reference_results = {
                "coordination_plan": coordination_response.data if hasattr(coordination_response, 'data') else {},
                "shared_data_available": "Data:" in shared_data_response.message,
                "cross_reference_status": "active"
            }
            
            return Response(
                message=f"Cross-reference analysis completed for: {query}\\n"
                       f"Data: {json.dumps({
                           'query': query,
                           'operation': 'cross_reference',
                           'results': cross_reference_results,
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Cross-reference error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error'})}",
                break_loop=False
            )
    
    async def _get_reasoning_status(self):
        """Get comprehensive status of the cognitive reasoning system."""
        try:
            status = {
                "opencog_available": OPENCOG_AVAILABLE,
                "atomspace_tools_available": ATOMSPACE_TOOLS_AVAILABLE,
                "cognitive_reasoning_initialized": self.initialized,
                "config_loaded": bool(self.config),
                "fallback_mode": not self.initialized,
                "cross_tool_integration": bool(self.tool_hub and self.tool_hub.initialized)
            }
            
            if self.initialized and self.atomspace:
                status.update({
                    "atomspace_size": len(self.atomspace),
                    "concept_nodes": len(self.atomspace.get_atoms_by_type(types.ConceptNode)),
                    "evaluation_links": len(self.atomspace.get_atoms_by_type(types.EvaluationLink)),
                    "inheritance_links": len(self.atomspace.get_atoms_by_type(types.InheritanceLink))
                })
            
            # Configuration status
            if self.config:
                status["configuration"] = {
                    "cognitive_mode": self.config.get("cognitive_mode", False),
                    "opencog_enabled": self.config.get("opencog_enabled", False),
                    "pln_enabled": self.config.get("reasoning_config", {}).get("pln_enabled", False),
                    "pattern_matching": self.config.get("reasoning_config", {}).get("pattern_matching", False),
                    "cross_tool_sharing": self.config.get("atomspace_config", {}).get("cross_tool_sharing", False)
                }
            
            return Response(
                message=f"Cognitive reasoning status retrieved\\n"
                       f"Data: {json.dumps({
                           'operation': 'status',
                           'status': status,
                           'timestamp': str(asyncio.get_event_loop().time())
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Status retrieval error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error'})}",
                break_loop=False
            )
    
    async def _share_knowledge_with_hub(self, query: str, **kwargs):
        """Share reasoning knowledge with the atomspace hub."""
        try:
            if not self.tool_hub or not self.tool_hub.initialized:
                return Response(
                    message="Tool hub not available for knowledge sharing\\n"
                           f"Data: {json.dumps({'status': 'hub_unavailable'})}",
                    break_loop=False
                )
            
            # Prepare knowledge data for sharing
            knowledge_data = {
                "query": query,
                "reasoning_type": "cognitive_reasoning",
                "concepts_extracted": [],
                "knowledge_quality": "high" if self.initialized else "fallback"
            }
            
            # Extract concepts if initialized
            if self.initialized:
                context = await self._build_reasoning_context(query, **kwargs)
                atoms = self.parse_query_to_atoms(query, context)
                knowledge_data["concepts_extracted"] = [
                    atom.name for atom in atoms if hasattr(atom, 'name')
                ][:5]  # Limit to 5 concepts
            else:
                # Fallback concept extraction
                query_words = query.lower().split()
                knowledge_data["concepts_extracted"] = [
                    word for word in query_words if len(word) > 3
                ][:5]
            
            # Share with hub
            share_response = await self.tool_hub.share_tool_data(
                tool_name="cognitive_reasoning",
                data_type="knowledge",
                data=knowledge_data
            )
            
            return Response(
                message=f"Knowledge shared with hub for: {query}\\n"
                       f"Data: {json.dumps({
                           'query': query,
                           'operation': 'share_knowledge',
                           'shared_concepts': len(knowledge_data['concepts_extracted']),
                           'hub_response': share_response.message if hasattr(share_response, 'message') else 'Shared',
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Knowledge sharing error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error'})}",
                break_loop=False
            )
    
    async def _share_reasoning_results(self, query: str, results: List):
        """Share reasoning results with other tools via hub."""
        try:
            if self.tool_hub and self.tool_hub.initialized:
                results_data = {
                    "query": query,
                    "results_count": len(results),
                    "reasoning_completed": True,
                    "result_types": [str(type(result).__name__) for result in results[:3]]
                }
                
                await self.tool_hub.share_tool_data(
                    tool_name="cognitive_reasoning",
                    data_type="reasoning_results",
                    data=results_data
                )
        except Exception as e:
            print(f"⚠️ Failed to share reasoning results: {e}")
    
    def format_reasoning_for_agent(self, results):
        """Format OpenCog results for Agent-Zero consumption with enhanced details."""
        if not results:
            return ["No reasoning results generated - check configuration or query complexity"]
        
        formatted = []
        result_types = {}
        
        for result in results:
            try:
                result_str = str(result)
                result_type = str(type(result).__name__)
                
                # Count result types
                result_types[result_type] = result_types.get(result_type, 0) + 1
                
                # Format based on result type
                if hasattr(result, 'type') and hasattr(result, 'out'):
                    if result.type == types.InheritanceLink:
                        formatted.append(f"Inheritance: {result_str}")
                    elif result.type == types.EvaluationLink:
                        formatted.append(f"Evaluation: {result_str}")
                    elif result.type == types.SimilarityLink:
                        formatted.append(f"Similarity: {result_str}")
                    else:
                        formatted.append(f"Relationship: {result_str}")
                else:
                    formatted.append(f"Created: {result_str}")
                    
            except Exception as e:
                formatted.append(f"Result processing error: {e}")
        
        # Add summary
        if result_types:
            type_summary = ", ".join([f"{count} {rtype}" for rtype, count in result_types.items()])
            formatted.insert(0, f"Reasoning summary: {type_summary}")
        
        return formatted


def register():
    """Register the cognitive reasoning tool with Agent-Zero."""
    return CognitiveReasoningTool