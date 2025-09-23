"""
PyCog-Zero URE (Unified Rule Engine) Tool
Provides Agent-Zero integration for OpenCog's URE forward and backward chaining
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
import asyncio
from typing import Dict, Any, List, Optional

# Try to import OpenCog URE components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    from opencog.ure import ForwardChainer, BackwardChainer
    from opencog.type_constructors import *
    URE_AVAILABLE = True
except ImportError:
    print("OpenCog URE not available - install with: pip install opencog-atomspace opencog-python opencog-ure")
    URE_AVAILABLE = False

# Import other atomspace tools for enhanced integration
try:
    from python.tools.atomspace_tool_hub import AtomSpaceToolHub
    from python.tools.cognitive_reasoning import CognitiveReasoningTool
    ATOMSPACE_TOOLS_AVAILABLE = True
except ImportError:
    ATOMSPACE_TOOLS_AVAILABLE = False


class UREChainTool(Tool):
    """Agent-Zero tool for URE forward and backward chaining operations."""
    
    # Class-level shared atomspace for cross-tool integration
    _shared_atomspace = None
    
    def _initialize_if_needed(self):
        """Initialize the URE system if not already done."""
        if hasattr(self, '_ure_initialized'):
            return
        
        self._ure_initialized = True
        self.atomspace = None
        self.initialized = False
        self.config = self._load_ure_config()
        self.tool_hub = None
        
        # Initialize with shared or new atomspace
        if URE_AVAILABLE and self.config.get("ure_enabled", True):
            try:
                # Try to use shared atomspace from tool hub or cognitive reasoning
                if ATOMSPACE_TOOLS_AVAILABLE:
                    shared_atomspace = AtomSpaceToolHub.get_shared_atomspace()
                    if shared_atomspace:
                        self.atomspace = shared_atomspace
                        print("✓ URE using shared AtomSpace from tool hub")
                    elif hasattr(CognitiveReasoningTool, '_shared_atomspace') and CognitiveReasoningTool._shared_atomspace:
                        self.atomspace = CognitiveReasoningTool._shared_atomspace
                        print("✓ URE using shared AtomSpace from cognitive reasoning")
                    else:
                        self.atomspace = self._create_new_atomspace()
                else:
                    self.atomspace = self._create_new_atomspace()
                
                if self.atomspace:
                    self.initialized = True
                    print("✓ OpenCog URE initialized - forward and backward chaining available")
                    
                    # Initialize cross-tool integration
                    self._setup_cross_tool_integration()
                    
            except Exception as e:
                print(f"⚠️ OpenCog URE initialization failed: {e}")
                self._setup_fallback_mode()
        else:
            self._setup_fallback_mode()
    
    def _create_new_atomspace(self):
        """Create a new AtomSpace instance for URE."""
        try:
            atomspace = AtomSpace()
            initialize_opencog(atomspace)
            
            # Set as shared atomspace if none exists
            if not UREChainTool._shared_atomspace:
                UREChainTool._shared_atomspace = atomspace
            
            return atomspace
        except Exception as e:
            print(f"⚠️ Failed to create new AtomSpace for URE: {e}")
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
                    'name': 'ure_chain_hub_integration',
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
            print(f"⚠️ URE cross-tool integration setup failed: {e}")
    
    async def _register_with_hub(self):
        """Register this URE tool with the AtomSpace hub."""
        try:
            registration_data = {
                "tool_type": "ure_chaining",
                "capabilities": ["forward_chaining", "backward_chaining", "rule_execution"],
                "atomspace_operations": ["create_rulebase", "execute_chains", "goal_inference"],
                "status": "active"
            }
            
            await self.tool_hub.share_tool_data(
                tool_name="ure_chain",
                data_type="registration",
                data=registration_data
            )
        except Exception as e:
            print(f"⚠️ URE tool registration failed: {e}")
    
    def _setup_fallback_mode(self):
        """Setup fallback mode when URE is not available."""
        self.atomspace = None
        self.initialized = False
        print("⚠️ Running in fallback mode - URE chaining not available")
    
    def _load_ure_config(self):
        """Load URE configuration from Agent-Zero settings."""
        try:
            # Try to import settings and get cognitive config
            from python.helpers import settings
            config = settings.get_cognitive_config()
            # Add URE-specific defaults
            config.setdefault("ure_config", {
                "ure_enabled": True,
                "forward_chaining": True,
                "backward_chaining": True,
                "max_iterations": 1000,
                "complexity_penalty": 0.01,
                "trace_enabled": False
            })
            return config
        except Exception:
            # Fallback to direct config file loading
            try:
                config_file = files.get_abs_path("conf/config_cognitive.json")
                with open(config_file, 'r') as f:
                    config = json.load(f)
                    config.setdefault("ure_config", {
                        "ure_enabled": True,
                        "forward_chaining": True,
                        "backward_chaining": True,
                        "max_iterations": 1000,
                        "complexity_penalty": 0.01,
                        "trace_enabled": False
                    })
                    return config
            except Exception as e:
                print(f"Warning: Could not load URE config: {e}")
                return {
                    "cognitive_mode": True,
                    "opencog_enabled": True,
                    "ure_enabled": True,
                    "ure_config": {
                        "forward_chaining": True,
                        "backward_chaining": True,
                        "max_iterations": 1000,
                        "complexity_penalty": 0.01,
                        "trace_enabled": False
                    }
                }
    
    async def execute(self, query: str, operation: str = "backward_chain", **kwargs):
        """Execute URE chaining operations on Agent-Zero queries."""
        
        # Initialize if needed
        self._initialize_if_needed()
        
        if not self.config.get("ure_enabled", True):
            return Response(
                message="URE chaining is disabled\\n"
                       f"Data: {json.dumps({'error': 'URE disabled in configuration'})}",
                break_loop=False
            )
        
        # Handle different URE operations
        if operation == "forward_chain":
            return await self._perform_forward_chaining(query, **kwargs)
        elif operation == "backward_chain":
            return await self._perform_backward_chaining(query, **kwargs)
        elif operation == "create_rulebase":
            return await self._create_rulebase(query, **kwargs)
        elif operation == "status":
            return await self._get_ure_status()
        elif operation == "list_rules":
            return await self._list_available_rules(**kwargs)
        else:
            # Default to backward chaining
            return await self._perform_backward_chaining(query, **kwargs)
    
    async def _perform_backward_chaining(self, query: str, **kwargs):
        """Perform backward chaining inference to prove a goal."""
        if not self.initialized:
            return await self._fallback_reasoning(query, "backward_chain", **kwargs)
        
        try:
            # Parse the query into a target atom for backward chaining
            target_atom = await self._parse_query_to_target(query, **kwargs)
            
            # Get or create rulebase
            rulebase = kwargs.get("rulebase", "default_rulebase")
            rbs_atom = self._get_or_create_rulebase(rulebase)
            
            # Setup variable declaration if provided
            vardecl = kwargs.get("vardecl")
            vardecl_atom = None
            if vardecl:
                vardecl_atom = self._parse_variable_declaration(vardecl)
            
            # Configure tracing if enabled
            trace_as = None
            if self.config.get("ure_config", {}).get("trace_enabled", False):
                trace_as = AtomSpace()  # Create separate atomspace for tracing
            
            # Create backward chainer
            chainer = BackwardChainer(
                self.atomspace,
                rbs_atom,
                target_atom,
                vardecl_atom,
                trace_as
            )
            
            # Execute chaining
            chainer.do_chain()
            results = chainer.get_results()
            
            # Format results for Agent-Zero
            formatted_results = self._format_ure_results(results, "backward_chain")
            
            # Share results with other tools if enabled
            if self.config.get("atomspace_config", {}).get("cross_tool_sharing", True):
                await self._share_ure_results(query, "backward_chain", formatted_results)
            
            return Response(
                message=f"Backward chaining completed for: {query}\\n"
                       f"Data: {json.dumps({
                           'query': query,
                           'operation': 'backward_chain',
                           'target_atom': str(target_atom),
                           'rulebase': rulebase,
                           'results_count': len(formatted_results),
                           'results': formatted_results,
                           'trace_enabled': trace_as is not None,
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Backward chaining error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error', 'operation': 'backward_chain'})}",
                break_loop=False
            )
    
    async def _perform_forward_chaining(self, query: str, **kwargs):
        """Perform forward chaining inference from a source."""
        if not self.initialized:
            return await self._fallback_reasoning(query, "forward_chain", **kwargs)
        
        try:
            # Parse the query into a source atom for forward chaining
            source_atom = await self._parse_query_to_source(query, **kwargs)
            
            # Get or create rulebase
            rulebase = kwargs.get("rulebase", "default_rulebase")
            rbs_atom = self._get_or_create_rulebase(rulebase)
            
            # Setup variable declaration if provided
            vardecl = kwargs.get("vardecl")
            vardecl_atom = None
            if vardecl:
                vardecl_atom = self._parse_variable_declaration(vardecl)
            
            # Configure focus set
            focus_set = kwargs.get("focus_set", [])
            
            # Configure tracing if enabled
            trace_as = None
            if self.config.get("ure_config", {}).get("trace_enabled", False):
                trace_as = AtomSpace()  # Create separate atomspace for tracing
            
            # Create forward chainer
            chainer = ForwardChainer(
                self.atomspace,
                rbs_atom,
                source_atom,
                vardecl_atom,
                trace_as,
                focus_set
            )
            
            # Execute chaining
            chainer.do_chain()
            results = chainer.get_results()
            
            # Format results for Agent-Zero
            formatted_results = self._format_ure_results(results, "forward_chain")
            
            # Share results with other tools if enabled
            if self.config.get("atomspace_config", {}).get("cross_tool_sharing", True):
                await self._share_ure_results(query, "forward_chain", formatted_results)
            
            return Response(
                message=f"Forward chaining completed for: {query}\\n"
                       f"Data: {json.dumps({
                           'query': query,
                           'operation': 'forward_chain',
                           'source_atom': str(source_atom),
                           'rulebase': rulebase,
                           'results_count': len(formatted_results),
                           'results': formatted_results,
                           'focus_set_size': len(focus_set),
                           'trace_enabled': trace_as is not None,
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Forward chaining error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error', 'operation': 'forward_chain'})}",
                break_loop=False
            )
    
    async def _fallback_reasoning(self, query: str, operation: str, **kwargs):
        """Fallback reasoning when URE is not available."""
        try:
            # Simple logical analysis as fallback
            query_words = query.lower().split()
            logical_patterns = []
            
            # Detect logical patterns
            if any(word in query_words for word in ["if", "then", "implies"]):
                logical_patterns.append("implication")
            
            if any(word in query_words for word in ["and", "both"]):
                logical_patterns.append("conjunction")
            
            if any(word in query_words for word in ["or", "either"]):
                logical_patterns.append("disjunction")
            
            if any(word in query_words for word in ["not", "isn't", "doesn't"]):
                logical_patterns.append("negation")
            
            fallback_results = [
                f"Pattern analysis: {', '.join(logical_patterns) if logical_patterns else 'basic_logic'}",
                f"Query structure: {len(query_words)} terms analyzed",
                f"Operation requested: {operation}",
                "Fallback mode: basic logical pattern recognition performed"
            ]
            
            return Response(
                message=f"Fallback URE reasoning completed for: {query}\\n"
                       f"Data: {json.dumps({
                           'query': query,
                           'operation': f'fallback_{operation}',
                           'patterns_detected': logical_patterns,
                           'results': fallback_results,
                           'status': 'fallback_success',
                           'note': 'Limited reasoning without OpenCog URE - install opencog-ure for full capabilities'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Fallback URE reasoning error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'fallback_error'})}",
                break_loop=False
            )
    
    async def _parse_query_to_target(self, query: str, **kwargs) -> Any:
        """Parse query into a target atom for backward chaining."""
        if not self.initialized:
            return None
        
        # Simple parsing - create a concept node from query
        # In a more sophisticated implementation, this would parse complex logical expressions
        concept_name = f"goal_{hash(query) % 10000}"
        target_atom = self.atomspace.add_node(types.ConceptNode, concept_name)
        
        # Add some basic structure based on query analysis
        words = query.lower().split()
        for word in words[:3]:  # Use first 3 significant words
            if len(word) > 2:
                word_concept = self.atomspace.add_node(types.ConceptNode, word)
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "relates_to"),
                        target_atom,
                        word_concept
                    ]
                )
        
        return target_atom
    
    async def _parse_query_to_source(self, query: str, **kwargs) -> Any:
        """Parse query into a source atom for forward chaining."""
        if not self.initialized:
            return None
        
        # Simple parsing - create a concept node from query
        concept_name = f"source_{hash(query) % 10000}"
        source_atom = self.atomspace.add_node(types.ConceptNode, concept_name)
        
        # Add some basic structure based on query analysis
        words = query.lower().split()
        for word in words[:3]:  # Use first 3 significant words
            if len(word) > 2:
                word_concept = self.atomspace.add_node(types.ConceptNode, word)
                self.atomspace.add_link(
                    types.InheritanceLink,
                    [source_atom, word_concept]
                )
        
        return source_atom
    
    def _parse_variable_declaration(self, vardecl_spec: str) -> Any:
        """Parse variable declaration specification."""
        if not self.initialized:
            return None
        
        # Simple variable declaration - create a typed variable
        var_node = self.atomspace.add_node(types.VariableNode, "$X")
        type_node = self.atomspace.add_node(types.TypeNode, "ConceptNode")
        
        return self.atomspace.add_link(
            types.TypedVariableLink,
            [var_node, type_node]
        )
    
    def _get_or_create_rulebase(self, rulebase_name: str) -> Any:
        """Get or create a rulebase for URE operations."""
        if not self.initialized:
            return None
        
        # Create or get rulebase concept node
        rbs_atom = self.atomspace.add_node(types.ConceptNode, rulebase_name)
        
        # Add basic URE configuration (simplified for this implementation)
        # In a full implementation, this would load actual rules
        if rulebase_name == "default_rulebase":
            self._setup_default_rules(rbs_atom)
        
        return rbs_atom
    
    def _setup_default_rules(self, rbs_atom: Any):
        """Setup default rules for the rulebase."""
        if not self.initialized:
            return
        
        # Create some basic deduction rules (simplified)
        # In a full implementation, this would load actual URE rules
        deduction_rule = self.atomspace.add_node(types.ConceptNode, "deduction_rule")
        modus_ponens_rule = self.atomspace.add_node(types.ConceptNode, "modus_ponens_rule")
        
        # Associate rules with rulebase
        for rule in [deduction_rule, modus_ponens_rule]:
            self.atomspace.add_link(
                types.MemberLink,
                [rule, rbs_atom]
            )
    
    def _format_ure_results(self, results: Any, operation: str) -> List[str]:
        """Format URE results for Agent-Zero consumption."""
        if not results:
            return [f"No {operation} results generated"]
        
        formatted = []
        try:
            if hasattr(results, 'get_out'):
                # Results is a SetLink with multiple results
                for result in results.get_out():
                    formatted.append(f"{operation}: {str(result)}")
            else:
                # Single result
                formatted.append(f"{operation}: {str(results)}")
        except Exception as e:
            formatted.append(f"Result formatting error: {e}")
        
        return formatted
    
    async def _create_rulebase(self, query: str, **kwargs):
        """Create a new rulebase with specified rules."""
        if not self.initialized:
            return Response(
                message="Cannot create rulebase - URE not initialized\\n"
                       f"Data: {json.dumps({'error': 'URE not available'})}",
                break_loop=False
            )
        
        try:
            rulebase_name = kwargs.get("rulebase_name", f"rulebase_{hash(query) % 1000}")
            rules = kwargs.get("rules", ["deduction", "modus_ponens"])
            
            rbs_atom = self._get_or_create_rulebase(rulebase_name)
            
            # Add specified rules
            for rule_name in rules:
                rule_atom = self.atomspace.add_node(types.ConceptNode, f"{rule_name}_rule")
                self.atomspace.add_link(
                    types.MemberLink,
                    [rule_atom, rbs_atom]
                )
            
            return Response(
                message=f"Rulebase created: {rulebase_name}\\n"
                       f"Data: {json.dumps({
                           'rulebase_name': rulebase_name,
                           'rules_added': rules,
                           'rulebase_atom': str(rbs_atom),
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Rulebase creation error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error'})}",
                break_loop=False
            )
    
    async def _get_ure_status(self):
        """Get comprehensive status of the URE system."""
        try:
            status = {
                "ure_available": URE_AVAILABLE,
                "atomspace_tools_available": ATOMSPACE_TOOLS_AVAILABLE,
                "ure_initialized": self.initialized,
                "config_loaded": bool(self.config),
                "fallback_mode": not self.initialized,
                "cross_tool_integration": bool(self.tool_hub and self.tool_hub.initialized)
            }
            
            if self.initialized and self.atomspace:
                status.update({
                    "atomspace_size": len(self.atomspace),
                    "concept_nodes": len(self.atomspace.get_atoms_by_type(types.ConceptNode)),
                    "member_links": len(self.atomspace.get_atoms_by_type(types.MemberLink)),
                    "evaluation_links": len(self.atomspace.get_atoms_by_type(types.EvaluationLink))
                })
            
            # URE-specific configuration status
            if self.config:
                ure_config = self.config.get("ure_config", {})
                status["ure_configuration"] = {
                    "forward_chaining": ure_config.get("forward_chaining", False),
                    "backward_chaining": ure_config.get("backward_chaining", False),
                    "max_iterations": ure_config.get("max_iterations", 1000),
                    "trace_enabled": ure_config.get("trace_enabled", False)
                }
            
            return Response(
                message=f"URE system status retrieved\\n"
                       f"Data: {json.dumps({
                           'operation': 'status',
                           'status': status,
                           'timestamp': str(asyncio.get_event_loop().time())
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"URE status retrieval error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error'})}",
                break_loop=False
            )
    
    async def _list_available_rules(self, **kwargs):
        """List available URE rules."""
        try:
            if not self.initialized:
                available_rules = ["deduction", "modus_ponens", "syllogism"]  # Fallback list
                rule_status = "fallback"
            else:
                # In a full implementation, this would query the actual rulebase
                available_rules = ["deduction", "modus_ponens", "syllogism", "abduction", "induction"]
                rule_status = "active"
            
            return Response(
                message=f"Available URE rules listed\\n"
                       f"Data: {json.dumps({
                           'operation': 'list_rules',
                           'available_rules': available_rules,
                           'rule_count': len(available_rules),
                           'status': rule_status
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Rule listing error: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'status': 'error'})}",
                break_loop=False
            )
    
    async def _share_ure_results(self, query: str, operation: str, results: List[str]):
        """Share URE results with other tools via hub."""
        try:
            if self.tool_hub and self.tool_hub.initialized:
                results_data = {
                    "query": query,
                    "operation": operation,
                    "results_count": len(results),
                    "ure_completed": True,
                    "result_preview": results[:3] if results else []
                }
                
                await self.tool_hub.share_tool_data(
                    tool_name="ure_chain",
                    data_type="ure_results",
                    data=results_data
                )
        except Exception as e:
            print(f"⚠️ Failed to share URE results: {e}")


def register():
    """Register the URE chain tool with Agent-Zero."""
    return UREChainTool