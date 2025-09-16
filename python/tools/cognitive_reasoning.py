"""
PyCog-Zero Cognitive Reasoning Tool
Integrates OpenCog cognitive architecture with Agent-Zero framework
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - install with: pip install opencog-atomspace opencog-python")
    OPENCOG_AVAILABLE = False


class CognitiveReasoningTool(Tool):
    """Agent-Zero tool for OpenCog cognitive reasoning."""
    
    def _initialize_if_needed(self):
        """Initialize the cognitive reasoning system if not already done."""
        if hasattr(self, '_cognitive_initialized'):
            return
        
        self._cognitive_initialized = True
        self.atomspace = None
        self.initialized = False
        self.config = self._load_cognitive_config()
        
        if OPENCOG_AVAILABLE and self.config.get("opencog_enabled", True):
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self.initialized = True
                print("✓ OpenCog cognitive reasoning initialized")
            except Exception as e:
                print(f"⚠️ OpenCog initialization failed: {e}")
    
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
                    "reasoning_config": {
                        "pln_enabled": True,
                        "pattern_matching": True
                    }
                }
    
    async def execute(self, query: str, **kwargs):
        """Execute cognitive reasoning on Agent-Zero queries."""
        
        # Initialize if needed
        self._initialize_if_needed()
        
        if not self.config.get("cognitive_mode", True):
            return Response(
                message="Cognitive mode is disabled",
                data={"error": "Cognitive mode disabled in configuration"}
            )
        
        if not self.initialized:
            return Response(
                message="Cognitive reasoning not available - OpenCog not properly initialized",
                data={"error": "OpenCog dependencies missing or failed to initialize"}
            )
        
        try:
            # Convert natural language query to AtomSpace representation
            query_atoms = self.parse_query_to_atoms(query)
            
            # Check if atomspace-rocks optimization is available
            storage_optimization = self._get_storage_optimization_info()
            
            # Perform reasoning based on configuration
            reasoning_results = self.execute_reasoning(query_atoms)
            
            # Format results for Agent-Zero consumption
            reasoning_steps = self.format_reasoning_for_agent(reasoning_results)
            
            return Response(
                message=f"Cognitive reasoning completed for: {query}",
                data={
                    "query": query,
                    "atoms_created": len(query_atoms),
                    "reasoning_steps": reasoning_steps,
                    "storage_optimization": storage_optimization,
                    "status": "success",
                    "config": {
                        "pln_enabled": self.config.get("reasoning_config", {}).get("pln_enabled", True),
                        "pattern_matching": self.config.get("reasoning_config", {}).get("pattern_matching", True)
                    }
                }
            )
            
        except Exception as e:
            return Response(
                message=f"Cognitive reasoning error: {str(e)}",
                data={"error": str(e), "status": "error"}
            )
    
    def parse_query_to_atoms(self, query: str):
        """Convert Agent-Zero query to OpenCog atoms."""
        if not self.initialized:
            return []
            
        # Simple demonstration - create concept nodes from query words
        words = query.lower().split()
        atoms = []
        
        for word in words:
            if len(word) > 2:  # Skip short words
                concept_node = self.atomspace.add_node(types.ConceptNode, word)
                atoms.append(concept_node)
        
        return atoms
    
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
        
        results = []
        reasoning_config = self.config.get("reasoning_config", {})
        
        # Pattern matching (if enabled)
        if reasoning_config.get("pattern_matching", True):
            results.extend(self.pattern_matching_reasoning(atoms))
        
        # PLN reasoning (if enabled)
        if reasoning_config.get("pln_enabled", True):
            results.extend(self.pln_reasoning(atoms))
        
        return results
    
    def pattern_matching_reasoning(self, atoms):
        """Perform pattern matching reasoning."""
        results = []
        
        # Create simple inheritance relationships between concepts
        for i in range(len(atoms) - 1):
            inheritance_link = self.atomspace.add_link(
                types.InheritanceLink, 
                [atoms[i], atoms[i + 1]]
            )
            results.append(inheritance_link)
        
        return results
    
    def pln_reasoning(self, atoms):
        """Perform PLN (Probabilistic Logic Networks) reasoning."""
        # Placeholder for PLN reasoning - would integrate with actual PLN when available
        results = []
        
        # Create evaluation links with truth values
        for atom in atoms:
            evaluation_link = self.atomspace.add_link(
                types.EvaluationLink,
                [self.atomspace.add_node(types.PredicateNode, "relevant"), atom]
            )
            results.append(evaluation_link)
        
        return results
    
    def format_reasoning_for_agent(self, results):
        """Format OpenCog results for Agent-Zero consumption."""
        if not results:
            return ["No reasoning results generated"]
        
        formatted = []
        for result in results:
            formatted.append(f"Created relationship: {str(result)}")
        
        return formatted


def register():
    """Register the cognitive reasoning tool with Agent-Zero."""
    return CognitiveReasoningTool