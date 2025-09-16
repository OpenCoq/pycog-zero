"""
PyCog-Zero AtomSpace-Enhanced Memory Extension
Extends Agent-Zero Memory class with AtomSpace knowledge graph functionality
"""

from python.helpers.memory import Memory as BaseMemory
from python.helpers import files
import json
import os

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - AtomSpace memory extensions will use fallback mode")
    OPENCOG_AVAILABLE = False


class AtomSpaceMemoryExtension:
    """Extension class that adds AtomSpace knowledge graph functionality to Agent-Zero Memory."""
    
    def __init__(self, memory_instance: BaseMemory):
        """Initialize AtomSpace extension for a Memory instance."""
        self.memory = memory_instance
        self.atomspace = None
        self.initialized = False
        self.atomspace_file = None
        
        if hasattr(memory_instance, 'memory_subdir'):
            memory_dir = BaseMemory._abs_db_dir(memory_instance.memory_subdir)
            self.atomspace_file = files.get_abs_path(memory_dir, "knowledge_graph.atomspace")
        
        self._initialize_atomspace()
    
    def _initialize_atomspace(self):
        """Initialize AtomSpace for knowledge graph functionality."""
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self.load_atomspace_from_file()
                self.initialized = True
                print("✓ AtomSpace memory extension initialized")
            except Exception as e:
                print(f"⚠️ AtomSpace memory extension initialization failed: {e}")
    
    def save_atomspace_to_file(self):
        """Save AtomSpace knowledge graph to file."""
        if not self.initialized or not self.atomspace_file:
            return
        
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.atomspace_file), exist_ok=True)
            
            # Serialize AtomSpace data
            atomspace_data = {
                "atoms": [],
                "metadata": {
                    "total_atoms": len(self.atomspace),
                    "saved_at": str(__import__('datetime').datetime.now())
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
                
                atomspace_data["atoms"].append(atom_data)
            
            with open(self.atomspace_file, 'w') as f:
                json.dump(atomspace_data, f, indent=2)
            
            print(f"✓ Saved AtomSpace knowledge graph: {len(atomspace_data['atoms'])} atoms")
            
        except Exception as e:
            print(f"⚠️ Failed to save AtomSpace knowledge graph: {e}")
    
    def load_atomspace_from_file(self):
        """Load AtomSpace knowledge graph from file."""
        if not self.initialized or not self.atomspace_file or not os.path.exists(self.atomspace_file):
            return
        
        try:
            with open(self.atomspace_file, 'r') as f:
                atomspace_data = json.load(f)
            
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
                        # Simple name-based lookup
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
            
            print(f"✓ Loaded AtomSpace knowledge graph: {len(atoms_data)} atoms")
            
        except Exception as e:
            print(f"⚠️ Failed to load AtomSpace knowledge graph: {e}")
    
    async def create_knowledge_graph_from_memory(self):
        """Create knowledge graph from existing memory documents."""
        if not self.initialized:
            return False
        
        try:
            # Get all documents from memory
            all_docs = self.memory.db.get_all_docs()
            
            concepts_created = 0
            for doc_id, document in all_docs.items():
                # Create document concept
                doc_concept = self.atomspace.add_node(types.ConceptNode, f"memory_doc_{doc_id}")
                
                # Extract content concepts
                content_words = document.page_content.lower().split()[:20]  # First 20 words
                for word in content_words:
                    if len(word) > 3 and word.isalpha():  # Filter meaningful words
                        word_concept = self.atomspace.add_node(types.ConceptNode, word)
                        
                        # Create contains relationship
                        self.atomspace.add_link(
                            types.EvaluationLink,
                            [
                                self.atomspace.add_node(types.PredicateNode, "document_contains"),
                                doc_concept,
                                word_concept
                            ]
                        )
                        concepts_created += 1
                
                # Process metadata
                if hasattr(document, 'metadata') and document.metadata:
                    for meta_key, meta_value in document.metadata.items():
                        meta_concept = self.atomspace.add_node(types.ConceptNode, f"meta_{meta_key}_{meta_value}")
                        self.atomspace.add_link(
                            types.EvaluationLink,
                            [
                                self.atomspace.add_node(types.PredicateNode, "has_metadata"),
                                doc_concept,
                                meta_concept
                            ]
                        )
                        concepts_created += 1
            
            # Save the knowledge graph
            self.save_atomspace_to_file()
            
            print(f"✓ Created knowledge graph from memory: {concepts_created} concepts")
            return True
            
        except Exception as e:
            print(f"⚠️ Failed to create knowledge graph from memory: {e}")
            return False
    
    def find_related_concepts(self, concept_name: str, max_results: int = 10):
        """Find concepts related to the given concept through AtomSpace relationships."""
        if not self.initialized:
            return []
        
        try:
            related = []
            
            # Find concept nodes that match or contain the concept name
            matching_concepts = [
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if hasattr(atom, 'name') and concept_name.lower() in atom.name.lower()
            ]
            
            for concept in matching_concepts[:5]:  # Limit to avoid too many results
                # Find what this concept is connected to
                for link in concept.incoming:
                    if link.type == types.EvaluationLink:
                        for atom in link.out:
                            if (atom != concept and 
                                atom.type == types.ConceptNode and 
                                hasattr(atom, 'name')):
                                related.append({
                                    "concept": atom.name,
                                    "relationship": link.out[0].name if hasattr(link.out[0], 'name') else "unknown",
                                    "source_concept": concept.name
                                })
            
            return related[:max_results]
            
        except Exception as e:
            print(f"Warning: Failed to find related concepts: {e}")
            return []
    
    def get_concept_neighbors(self, concept_name: str, depth: int = 2):
        """Get neighboring concepts in the knowledge graph up to specified depth."""
        if not self.initialized:
            return {}
        
        try:
            neighbors = {}
            visited = set()
            to_visit = [(concept_name, 0)]
            
            while to_visit and len(neighbors) < 50:  # Limit total results
                current_concept, current_depth = to_visit.pop(0)
                
                if current_concept in visited or current_depth >= depth:
                    continue
                
                visited.add(current_concept)
                neighbors[current_concept] = {
                    "depth": current_depth,
                    "connections": []
                }
                
                # Find concept node
                concept_nodes = [
                    atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                    if hasattr(atom, 'name') and atom.name == current_concept
                ]
                
                for concept_node in concept_nodes:
                    for link in concept_node.incoming:
                        if link.type == types.EvaluationLink and len(link.out) >= 3:
                            predicate = link.out[0].name if hasattr(link.out[0], 'name') else "unknown"
                            connected_concepts = [
                                atom.name for atom in link.out[1:]
                                if (atom != concept_node and 
                                    atom.type == types.ConceptNode and 
                                    hasattr(atom, 'name'))
                            ]
                            
                            for connected in connected_concepts:
                                neighbors[current_concept]["connections"].append({
                                    "target": connected,
                                    "relationship": predicate
                                })
                                
                                if connected not in visited and current_depth < depth - 1:
                                    to_visit.append((connected, current_depth + 1))
            
            return neighbors
            
        except Exception as e:
            print(f"Warning: Failed to get concept neighbors: {e}")
            return {}
    
    def analyze_knowledge_graph_structure(self):
        """Analyze the structure of the knowledge graph."""
        if not self.initialized:
            return {}
        
        try:
            analysis = {
                "total_atoms": len(self.atomspace),
                "concept_nodes": len(self.atomspace.get_atoms_by_type(types.ConceptNode)),
                "evaluation_links": len(self.atomspace.get_atoms_by_type(types.EvaluationLink)),
                "inheritance_links": len(self.atomspace.get_atoms_by_type(types.InheritanceLink)),
            }
            
            # Analyze predicate usage
            predicate_usage = {}
            for link in self.atomspace.get_atoms_by_type(types.EvaluationLink):
                if len(link.out) > 0 and link.out[0].type == types.PredicateNode:
                    predicate = link.out[0].name if hasattr(link.out[0], 'name') else "unknown"
                    predicate_usage[predicate] = predicate_usage.get(predicate, 0) + 1
            
            analysis["top_predicates"] = sorted(predicate_usage.items(), key=lambda x: x[1], reverse=True)[:10]
            
            # Analyze concept connectivity
            concept_connectivity = {}
            for concept in self.atomspace.get_atoms_by_type(types.ConceptNode):
                if hasattr(concept, 'name'):
                    concept_connectivity[concept.name] = len(concept.incoming)
            
            # Top connected concepts
            top_connected = sorted(concept_connectivity.items(), key=lambda x: x[1], reverse=True)[:10]
            analysis["most_connected_concepts"] = top_connected
            
            # Calculate graph density
            max_possible_edges = analysis["concept_nodes"] * (analysis["concept_nodes"] - 1) / 2
            actual_edges = analysis["evaluation_links"]
            analysis["graph_density"] = actual_edges / max_possible_edges if max_possible_edges > 0 else 0
            
            return analysis
            
        except Exception as e:
            print(f"Warning: Knowledge graph analysis failed: {e}")
            return {}


# Extend Memory class with AtomSpace functionality
def add_atomspace_extension_to_memory(memory_instance: BaseMemory):
    """Add AtomSpace extension methods to a Memory instance."""
    extension = AtomSpaceMemoryExtension(memory_instance)
    
    # Add extension methods to the memory instance
    memory_instance.atomspace_extension = extension
    memory_instance.create_knowledge_graph = extension.create_knowledge_graph_from_memory
    memory_instance.find_related_concepts = extension.find_related_concepts
    memory_instance.get_concept_neighbors = extension.get_concept_neighbors
    memory_instance.analyze_knowledge_graph = extension.analyze_knowledge_graph_structure
    memory_instance.save_knowledge_graph = extension.save_atomspace_to_file
    
    return memory_instance


# Enhanced Memory class factory
class AtomSpaceEnhancedMemory(BaseMemory):
    """Enhanced Memory class with built-in AtomSpace functionality."""
    
    def __init__(self, agent, db, memory_subdir: str):
        super().__init__(agent, db, memory_subdir)
        self.atomspace_extension = AtomSpaceMemoryExtension(self)
        
        # Add AtomSpace methods
        self.create_knowledge_graph = self.atomspace_extension.create_knowledge_graph_from_memory
        self.find_related_concepts = self.atomspace_extension.find_related_concepts
        self.get_concept_neighbors = self.atomspace_extension.get_concept_neighbors
        self.analyze_knowledge_graph = self.atomspace_extension.analyze_knowledge_graph_structure
        self.save_knowledge_graph = self.atomspace_extension.save_atomspace_to_file
    
    async def insert_text_with_knowledge_graph(self, text: str, metadata: dict = None):
        """Insert text and automatically update knowledge graph."""
        # Insert using standard method
        doc_id = await self.insert_text(text, metadata)
        
        # Update knowledge graph if AtomSpace is available
        if self.atomspace_extension.initialized:
            try:
                # Create concept for this specific document
                doc_concept = self.atomspace_extension.atomspace.add_node(
                    types.ConceptNode, 
                    f"memory_doc_{doc_id}"
                )
                
                # Extract and link concepts from text
                words = text.lower().split()[:20]  # First 20 words
                for word in words:
                    if len(word) > 3 and word.isalpha():
                        word_concept = self.atomspace_extension.atomspace.add_node(types.ConceptNode, word)
                        self.atomspace_extension.atomspace.add_link(
                            types.EvaluationLink,
                            [
                                self.atomspace_extension.atomspace.add_node(types.PredicateNode, "document_contains"),
                                doc_concept,
                                word_concept
                            ]
                        )
                
                # Save updated knowledge graph
                self.atomspace_extension.save_atomspace_to_file()
                
            except Exception as e:
                print(f"Warning: Failed to update knowledge graph: {e}")
        
        return doc_id
    
    async def search_with_knowledge_graph(self, query: str, limit: int = 10, threshold: float = 0.7):
        """Enhanced search that combines vector similarity with knowledge graph insights."""
        # Perform standard search
        standard_results = await self.search_similarity_threshold(
            query=query, 
            limit=limit, 
            threshold=threshold
        )
        
        # Enhance with knowledge graph if available
        if self.atomspace_extension.initialized:
            try:
                # Find related concepts for query terms
                query_words = [word.strip().lower() for word in query.split() if len(word.strip()) > 2]
                related_concepts = []
                
                for word in query_words:
                    concepts = self.find_related_concepts(word, max_results=5)
                    related_concepts.extend(concepts)
                
                # Add knowledge graph insights to results metadata
                for doc in standard_results:
                    if hasattr(doc, 'metadata'):
                        doc.metadata = doc.metadata or {}
                        doc.metadata['knowledge_graph_insights'] = {
                            'related_concepts_count': len(related_concepts),
                            'related_concepts': [c['concept'] for c in related_concepts[:5]]
                        }
                
            except Exception as e:
                print(f"Warning: Knowledge graph enhancement failed: {e}")
        
        return standard_results