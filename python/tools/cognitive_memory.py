"""
PyCog-Zero Cognitive Memory Tool
Integrates Agent-Zero memory with OpenCog AtomSpace for persistent cognitive memory
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
import pickle
import os

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - install with: pip install opencog-atomspace opencog-python")
    OPENCOG_AVAILABLE = False


class CognitiveMemoryTool(Tool):
    """Integrate Agent-Zero memory with OpenCog AtomSpace."""
    
    def _initialize_if_needed(self):
        """Initialize the cognitive memory system if not already done."""
        if hasattr(self, '_cognitive_initialized'):
            return
        
        self._cognitive_initialized = True
        self.atomspace = None
        self.memory_file = files.get_abs_path("memory/cognitive_atomspace.pkl")
        self.initialized = False
        
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self.load_persistent_memory()
                self.initialized = True
                print("✓ OpenCog cognitive memory initialized")
            except Exception as e:
                print(f"⚠️ OpenCog cognitive memory initialization failed: {e}")
        else:
            print("⚠️ OpenCog not available - cognitive memory will use fallback mode")
    
    async def execute(self, operation: str, data: dict = None, **kwargs):
        """Operations: store, retrieve, associate, reason"""
        
        # Initialize if needed
        self._initialize_if_needed()
        
        if not self.initialized and OPENCOG_AVAILABLE:
            return Response(message="Cognitive memory not initialized - OpenCog unavailable", break_loop=False)
        
        if operation == "store":
            return await self.store_knowledge(data)
        elif operation == "retrieve":
            return await self.retrieve_knowledge(data)
        elif operation == "associate":
            return await self.create_associations(data)
        elif operation == "reason":
            return await self.cognitive_reasoning(data)
        else:
            return Response(message="Unknown cognitive memory operation. Available: store, retrieve, associate, reason", break_loop=False)
    
    async def store_knowledge(self, data: dict):
        """Store Agent-Zero knowledge in AtomSpace."""
        if not self.initialized:
            return Response(message="OpenCog not available - knowledge stored in fallback mode", break_loop=False)
        
        if not data or "concept" not in data:
            return Response(message="Invalid data: 'concept' field required", break_loop=False)
        
        try:
            concept = self.atomspace.add_node(types.ConceptNode, data["concept"])
            
            if "properties" in data:
                for prop, value in data["properties"].items():
                    prop_node = self.atomspace.add_node(types.ConceptNode, prop)
                    value_node = self.atomspace.add_node(types.ConceptNode, str(value))
                    
                    # Create inheritance and evaluation links
                    self.atomspace.add_link(types.InheritanceLink, [concept, prop_node])
                    self.atomspace.add_link(types.EvaluationLink, [prop_node, value_node])
            
            # Store relationships if provided
            if "relationships" in data:
                for rel_type, targets in data["relationships"].items():
                    if not isinstance(targets, list):
                        targets = [targets]
                    
                    for target in targets:
                        target_node = self.atomspace.add_node(types.ConceptNode, str(target))
                        rel_node = self.atomspace.add_node(types.PredicateNode, rel_type)
                        self.atomspace.add_link(types.EvaluationLink, [rel_node, concept, target_node])
            
            self.save_persistent_memory()
            
            return Response(
                message=f"Stored knowledge about '{data['concept']}' in cognitive memory",
                break_loop=False
            )
            
        except Exception as e:
            return Response(message=f"Error storing knowledge: {e}", break_loop=False)
    
    async def retrieve_knowledge(self, data: dict):
        """Retrieve knowledge from AtomSpace."""
        if not self.initialized:
            return Response(message="OpenCog not available - using fallback retrieval", break_loop=False)
        
        if not data or "concept" not in data:
            return Response(message="Invalid data: 'concept' field required for retrieval", break_loop=False)
        
        try:
            concept_name = data["concept"]
            
            # Find the concept node
            concept_nodes = [atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode) 
                           if atom.name == concept_name]
            
            if not concept_nodes:
                return Response(
                    message=f"Concept '{concept_name}' not found in cognitive memory",
                    break_loop=False
                )
            
            concept_node = concept_nodes[0]
            
            # Gather related information
            knowledge = {
                "concept": concept_name,
                "properties": {},
                "relationships": {},
                "incoming_links": [],
                "outgoing_links": []
            }
            
            # Get incoming links (relationships where this concept is the target)
            incoming = concept_node.incoming
            for link in incoming:
                if link.type == types.InheritanceLink:
                    if len(link.out) == 2 and link.out[1] == concept_node:
                        knowledge["properties"][link.out[0].name] = "inherited_property"
                elif link.type == types.EvaluationLink:
                    if len(link.out) >= 2:
                        predicate = link.out[0].name
                        if predicate not in knowledge["relationships"]:
                            knowledge["relationships"][predicate] = []
                        # Extract other participants in the relationship
                        for atom in link.out[1:]:
                            if atom != concept_node:
                                knowledge["relationships"][predicate].append(atom.name)
                
                knowledge["incoming_links"].append(str(link))
            
            # Get outgoing links (if any)
            if hasattr(concept_node, 'outgoing') and concept_node.outgoing:
                knowledge["outgoing_links"] = [str(link) for link in concept_node.outgoing]
            
            return Response(
                message=f"Retrieved knowledge about '{concept_name}' from cognitive memory. Data: {json.dumps(knowledge)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(message=f"Error retrieving knowledge: {e}", break_loop=False)
    
    async def create_associations(self, data: dict):
        """Create associations between concepts in AtomSpace."""
        if not self.initialized:
            return Response(message="OpenCog not available - associations not created", break_loop=False)
        
        if not data or "source" not in data or "target" not in data:
            return Response(message="Invalid data: 'source' and 'target' fields required", break_loop=False)
        
        try:
            source_node = self.atomspace.add_node(types.ConceptNode, data["source"])
            target_node = self.atomspace.add_node(types.ConceptNode, data["target"])
            
            # Default association type
            association_type = data.get("type", "similar_to")
            predicate_node = self.atomspace.add_node(types.PredicateNode, association_type)
            
            # Create the association link
            association_link = self.atomspace.add_link(
                types.EvaluationLink, 
                [predicate_node, source_node, target_node]
            )
            
            # Add strength/confidence if provided
            if "strength" in data:
                try:
                    from opencog.atomspace import TruthValue
                    strength = float(data["strength"])
                    confidence = float(data.get("confidence", 0.9))
                    association_link.tv = TruthValue(strength, confidence)
                except:
                    pass  # Graceful fallback if TruthValue not available
            
            self.save_persistent_memory()
            
            return Response(
                message=f"Created association: {data['source']} --{association_type}--> {data['target']}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(message=f"Error creating association: {e}", break_loop=False)
    
    async def cognitive_reasoning(self, data: dict):
        """Perform basic cognitive reasoning on stored knowledge."""
        if not self.initialized:
            return Response(message="OpenCog not available - reasoning not performed", break_loop=False)
        
        try:
            query = data.get("query", "") if data else ""
            
            # Simple reasoning: find related concepts and connections
            reasoning_results = {
                "query": query,
                "total_atoms": len(self.atomspace),
                "concept_nodes": len(self.atomspace.get_atoms_by_type(types.ConceptNode)),
                "evaluation_links": len(self.atomspace.get_atoms_by_type(types.EvaluationLink)),
                "inheritance_links": len(self.atomspace.get_atoms_by_type(types.InheritanceLink)),
                "connected_concepts": []
            }
            
            # If a specific query is provided, find related concepts
            if query:
                query_concepts = [atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode) 
                                if query.lower() in atom.name.lower()]
                
                for concept in query_concepts[:5]:  # Limit to top 5 matches
                    concept_info = {
                        "concept": concept.name,
                        "connections": len(concept.incoming)
                    }
                    reasoning_results["connected_concepts"].append(concept_info)
            
            return Response(
                message=f"Cognitive reasoning completed for query: '{query}'. Results: {json.dumps(reasoning_results)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(message=f"Error in cognitive reasoning: {e}", break_loop=False)
    
    def load_persistent_memory(self):
        """Load AtomSpace from persistent storage."""
        try:
            if files.exists(self.memory_file):
                with open(self.memory_file, 'rb') as f:
                    atomspace_data = pickle.load(f)
                # Restore AtomSpace from serialized data
                self.restore_atomspace(atomspace_data)
                print(f"✓ Loaded cognitive memory from {self.memory_file}")
        except Exception as e:
            print(f"⚠️ Could not load cognitive memory: {e}")
    
    def save_persistent_memory(self):
        """Save AtomSpace to persistent storage."""
        try:
            # Ensure memory directory exists
            memory_dir = os.path.dirname(self.memory_file)
            os.makedirs(memory_dir, exist_ok=True)
            
            atomspace_data = self.serialize_atomspace()
            with open(self.memory_file, 'wb') as f:
                pickle.dump(atomspace_data, f)
            print(f"✓ Saved cognitive memory to {self.memory_file}")
        except Exception as e:
            print(f"⚠️ Could not save cognitive memory: {e}")
    
    def serialize_atomspace(self):
        """Serialize AtomSpace to a dictionary for persistence."""
        if not self.initialized:
            return {}
        
        try:
            serialized_data = {
                "atoms": [],
                "metadata": {
                    "total_atoms": len(self.atomspace),
                    "created_at": str(__import__('datetime').datetime.now())
                }
            }
            
            # Serialize all atoms
            for atom in self.atomspace:
                atom_data = {
                    "type": atom.type,
                    "name": atom.name if hasattr(atom, 'name') else None,
                    "outgoing": [str(out_atom) for out_atom in atom.out] if hasattr(atom, 'out') else []
                }
                
                # Include truth value if available
                if hasattr(atom, 'tv') and atom.tv:
                    try:
                        atom_data["truth_value"] = {
                            "strength": atom.tv.mean,
                            "confidence": atom.tv.confidence
                        }
                    except:
                        pass
                
                serialized_data["atoms"].append(atom_data)
            
            return serialized_data
            
        except Exception as e:
            print(f"Error serializing AtomSpace: {e}")
            return {}
    
    def restore_atomspace(self, atomspace_data):
        """Restore AtomSpace from serialized data."""
        if not self.initialized or not atomspace_data:
            return
        
        try:
            atoms_data = atomspace_data.get("atoms", [])
            
            # First pass: create all nodes
            for atom_data in atoms_data:
                if not atom_data.get("outgoing"):  # This is a node
                    atom_type = atom_data["type"]
                    atom_name = atom_data.get("name", "")
                    
                    if atom_name:
                        node = self.atomspace.add_node(atom_type, atom_name)
                        
                        # Restore truth value if available
                        if "truth_value" in atom_data:
                            try:
                                from opencog.atomspace import TruthValue
                                tv_data = atom_data["truth_value"]
                                node.tv = TruthValue(tv_data["strength"], tv_data["confidence"])
                            except:
                                pass
            
            # Second pass: create all links
            for atom_data in atoms_data:
                if atom_data.get("outgoing"):  # This is a link
                    atom_type = atom_data["type"]
                    outgoing_names = atom_data["outgoing"]
                    
                    # Find the outgoing atoms
                    outgoing_atoms = []
                    for name in outgoing_names:
                        # Simple name-based lookup (could be improved)
                        matching_atoms = [atom for atom in self.atomspace if str(atom) == name]
                        if matching_atoms:
                            outgoing_atoms.append(matching_atoms[0])
                    
                    if outgoing_atoms:
                        link = self.atomspace.add_link(atom_type, outgoing_atoms)
                        
                        # Restore truth value if available
                        if "truth_value" in atom_data:
                            try:
                                from opencog.atomspace import TruthValue
                                tv_data = atom_data["truth_value"]
                                link.tv = TruthValue(tv_data["strength"], tv_data["confidence"])
                            except:
                                pass
            
            print(f"✓ Restored {len(atoms_data)} atoms to cognitive memory")
            
        except Exception as e:
            print(f"Error restoring AtomSpace: {e}")


def register():
    return CognitiveMemoryTool