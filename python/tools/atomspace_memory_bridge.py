"""
PyCog-Zero AtomSpace Memory Bridge Tool
Enhances Agent-Zero memory tools with OpenCog AtomSpace integration
"""

from python.helpers.tool import Tool, Response
from python.helpers.memory import Memory
from python.helpers import files
import json
import asyncio

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - AtomSpace memory bridge will use fallback mode")
    OPENCOG_AVAILABLE = False


class AtomSpaceMemoryBridge(Tool):
    """Bridge between Agent-Zero memory system and OpenCog AtomSpace."""
    
    def __init__(self, agent, name: str, method: str | None, args: dict, message: str, loop_data, **kwargs):
        super().__init__(agent, name, method, args, message, loop_data, **kwargs)
        self._bridge_initialized = False
        self.atomspace = None
        self.initialized = False
        
    def _initialize_bridge(self):
        """Initialize the AtomSpace memory bridge if not already done."""
        if self._bridge_initialized:
            return
        
        self._bridge_initialized = True
        
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self.initialized = True
                print("✓ AtomSpace memory bridge initialized")
            except Exception as e:
                print(f"⚠️ AtomSpace memory bridge initialization failed: {e}")
    
    async def execute(self, operation: str = "bridge_memory", query: str = "", **kwargs):
        """Bridge operations: bridge_memory, query_knowledge, enhance_search, cross_reference"""
        
        self._initialize_bridge()
        
        if operation == "bridge_memory":
            return await self.bridge_agent_memory_to_atomspace()
        elif operation == "query_knowledge":
            return await self.query_atomspace_knowledge(query)
        elif operation == "enhance_search":
            return await self.enhance_memory_search_with_atomspace(query, **kwargs)
        elif operation == "cross_reference":
            return await self.cross_reference_memories(query, **kwargs)
        else:
            return Response(
                message="Unknown bridge operation. Available: bridge_memory, query_knowledge, enhance_search, cross_reference",
                break_loop=False
            )
    
    async def bridge_agent_memory_to_atomspace(self):
        """Bridge Agent-Zero memory contents to AtomSpace for cognitive processing."""
        if not self.initialized:
            return Response(
                message="AtomSpace bridge not available - memory bridging skipped",
                break_loop=False
            )
        
        try:
            # Get Agent-Zero memory database
            memory_db = await Memory.get(self.agent)
            
            # Get all documents from memory
            all_docs = memory_db.db.get_all_docs()
            
            bridged_count = 0
            for doc_id, document in all_docs.items():
                # Create concept node for each memory document
                doc_concept = self.atomspace.add_node(types.ConceptNode, f"memory_doc_{doc_id}")
                
                # Extract content as concept
                content_words = document.page_content.lower().split()[:10]  # First 10 words
                for word in content_words:
                    if len(word) > 2:  # Skip short words
                        word_concept = self.atomspace.add_node(types.ConceptNode, word)
                        
                        # Create relationship between document and content concepts
                        self.atomspace.add_link(
                            types.EvaluationLink,
                            [
                                self.atomspace.add_node(types.PredicateNode, "contains"),
                                doc_concept,
                                word_concept
                            ]
                        )
                
                # Bridge metadata if available
                if hasattr(document, 'metadata') and document.metadata:
                    for meta_key, meta_value in document.metadata.items():
                        meta_node = self.atomspace.add_node(types.ConceptNode, f"{meta_key}_{meta_value}")
                        self.atomspace.add_link(
                            types.EvaluationLink,
                            [
                                self.atomspace.add_node(types.PredicateNode, "has_metadata"),
                                doc_concept,
                                meta_node
                            ]
                        )
                
                bridged_count += 1
            
            return Response(
                message=f"Successfully bridged {bridged_count} memory documents to AtomSpace. "
                       f"Total atoms in AtomSpace: {len(self.atomspace)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error bridging memory to AtomSpace: {e}",
                break_loop=False
            )
    
    async def query_atomspace_knowledge(self, query: str):
        """Query AtomSpace for enhanced knowledge retrieval."""
        if not self.initialized:
            return Response(
                message=f"AtomSpace not available - falling back to standard memory search for: '{query}'",
                break_loop=False
            )
        
        try:
            query_words = query.lower().split()
            related_concepts = []
            reasoning_chains = []
            
            # Find concepts related to query terms
            for word in query_words:
                if len(word) > 2:
                    # Find concept nodes matching query words
                    matching_concepts = [
                        atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                        if word in atom.name.lower()
                    ]
                    
                    for concept in matching_concepts[:3]:  # Limit to top 3 per word
                        related_concepts.append(concept.name)
                        
                        # Find reasoning chains through this concept
                        for link in concept.incoming:
                            if link.type == types.EvaluationLink and len(link.out) >= 3:
                                predicate = link.out[0].name
                                other_concepts = [atom.name for atom in link.out[2:] if atom != concept]
                                reasoning_chains.append(f"{concept.name} --{predicate}--> {', '.join(other_concepts)}")
            
            knowledge_result = {
                "query": query,
                "related_concepts": list(set(related_concepts)),
                "reasoning_chains": reasoning_chains[:5],  # Top 5 chains
                "atomspace_size": len(self.atomspace)
            }
            
            return Response(
                message=f"AtomSpace knowledge query for '{query}' completed. "
                       f"Found {len(related_concepts)} related concepts and {len(reasoning_chains)} reasoning chains. "
                       f"Results: {json.dumps(knowledge_result)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error querying AtomSpace knowledge: {e}",
                break_loop=False
            )
    
    async def enhance_memory_search_with_atomspace(self, query: str, threshold: float = 0.7, limit: int = 10):
        """Enhance standard memory search with AtomSpace reasoning."""
        try:
            # First, perform standard Agent-Zero memory search
            memory_db = await Memory.get(self.agent)
            standard_docs = await memory_db.search_similarity_threshold(
                query=query, 
                limit=limit, 
                threshold=threshold
            )
            
            enhanced_results = []
            
            # Add standard results
            for doc in standard_docs:
                enhanced_results.append({
                    "type": "memory_search",
                    "content": doc.page_content[:200],  # First 200 chars
                    "score": getattr(doc, 'score', 'N/A'),
                    "source": "Agent-Zero Memory"
                })
            
            # If AtomSpace is available, add cognitive enhancements
            if self.initialized:
                atomspace_knowledge = await self.query_atomspace_knowledge(query)
                atomspace_data = json.loads(atomspace_knowledge.message.split("Results: ")[-1])
                
                enhanced_results.append({
                    "type": "cognitive_enhancement",
                    "related_concepts": atomspace_data["related_concepts"],
                    "reasoning_chains": atomspace_data["reasoning_chains"],
                    "source": "AtomSpace Reasoning"
                })
            
            result_summary = f"Enhanced memory search for '{query}' completed.\n"
            result_summary += f"Standard memory results: {len(standard_docs)}\n"
            if self.initialized:
                result_summary += f"Cognitive enhancements: {len(atomspace_data.get('related_concepts', []))} concepts, "
                result_summary += f"{len(atomspace_data.get('reasoning_chains', []))} reasoning chains\n"
            result_summary += f"Total enhanced results: {len(enhanced_results)}"
            
            return Response(
                message=result_summary,
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in enhanced memory search: {e}",
                break_loop=False
            )
    
    async def cross_reference_memories(self, query: str = "", concept: str = None, depth: int = 2):
        """Cross-reference memories using AtomSpace graph traversal."""
        # Use concept from kwargs if provided, otherwise use query
        target_concept = concept or query or "unknown"
        
        if not self.initialized:
            return Response(
                message=f"AtomSpace not available - cannot cross-reference memories for concept: '{target_concept}'",
                break_loop=False
            )
        
        try:
            # Find the concept in AtomSpace
            matching_concepts = [
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if target_concept.lower() in atom.name.lower()
            ]
            
            if not matching_concepts:
                return Response(
                    message=f"No matching concepts found in AtomSpace for: '{target_concept}'",
                    break_loop=False
                )
            
            cross_references = {}
            
            for base_concept in matching_concepts[:3]:  # Limit to top 3 matches
                connections = []
                
                # Traverse connections up to specified depth
                visited = set()
                to_visit = [(base_concept, 0)]
                
                while to_visit and len(connections) < 20:  # Limit total connections
                    current_concept, current_depth = to_visit.pop(0)
                    
                    if current_concept in visited or current_depth >= depth:
                        continue
                    
                    visited.add(current_concept)
                    
                    # Explore incoming links
                    for link in current_concept.incoming:
                        if link.type == types.EvaluationLink and len(link.out) >= 3:
                            predicate = link.out[0].name
                            connected_concepts = [atom for atom in link.out[1:] if atom != current_concept]
                            
                            for connected in connected_concepts:
                                if connected not in visited:
                                    connections.append({
                                        "from": current_concept.name,
                                        "relation": predicate,
                                        "to": connected.name,
                                        "depth": current_depth
                                    })
                                    
                                    if current_depth < depth - 1:
                                        to_visit.append((connected, current_depth + 1))
                
                cross_references[base_concept.name] = connections
            
            return Response(
                message=f"Cross-reference analysis for '{target_concept}' completed. "
                       f"Found {sum(len(refs) for refs in cross_references.values())} cross-references. "
                       f"Results: {json.dumps(cross_references)}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in cross-referencing memories: {e}",
                break_loop=False
            )


def register():
    """Register the AtomSpace memory bridge tool with Agent-Zero."""
    return AtomSpaceMemoryBridge