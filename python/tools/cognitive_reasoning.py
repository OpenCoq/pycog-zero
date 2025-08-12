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
    
    def __init__(self, agent):
        super().__init__(agent)
        self.atomspace = None
        self.initialized = False
        
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self.initialized = True
                print("✓ OpenCog cognitive reasoning initialized")
            except Exception as e:
                print(f"⚠️ OpenCog initialization failed: {e}")
    
    async def execute(self, query: str, **kwargs):
        """Execute cognitive reasoning on Agent-Zero queries."""
        
        if not self.initialized:
            return Response(
                message="Cognitive reasoning not available - OpenCog not properly initialized",
                data={"error": "OpenCog dependencies missing or failed to initialize"}
            )
        
        try:
            # Convert natural language query to AtomSpace representation
            query_atoms = self.parse_query_to_atoms(query)
            
            # For demonstration, perform basic reasoning
            reasoning_results = self.basic_reasoning(query_atoms)
            
            # Format results for Agent-Zero consumption
            reasoning_steps = self.format_reasoning_for_agent(reasoning_results)
            
            return Response(
                message=f"Cognitive reasoning completed for: {query}",
                data={
                    "query": query,
                    "atoms_created": len(query_atoms),
                    "reasoning_steps": reasoning_steps,
                    "status": "success"
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
    
    def basic_reasoning(self, atoms):
        """Perform basic reasoning operations on atoms."""
        if not atoms:
            return []
        
        results = []
        
        # Create simple inheritance relationships between concepts
        for i in range(len(atoms) - 1):
            inheritance_link = self.atomspace.add_link(
                types.InheritanceLink, 
                [atoms[i], atoms[i + 1]]
            )
            results.append(inheritance_link)
        
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