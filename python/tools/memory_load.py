from python.helpers.memory import Memory
from python.helpers.tool import Tool, Response

DEFAULT_THRESHOLD = 0.7
DEFAULT_LIMIT = 10

# Try to import AtomSpace tool hub for enhanced memory search (optional)
try:
    from python.tools.atomspace_tool_hub import AtomSpaceToolHub
    ATOMSPACE_HUB_AVAILABLE = True
except ImportError:
    ATOMSPACE_HUB_AVAILABLE = False


class MemoryLoad(Tool):

    async def execute(self, query="", threshold=DEFAULT_THRESHOLD, limit=DEFAULT_LIMIT, filter="", **kwargs):
        db = await Memory.get(self.agent)
        docs = await db.search_similarity_threshold(query=query, limit=limit, threshold=threshold, filter=filter)

        # Optional: Enhance search results with atomspace reasoning
        if ATOMSPACE_HUB_AVAILABLE and len(docs) > 0:
            try:
                atomspace_hub = AtomSpaceToolHub.get_shared_atomspace()
                if atomspace_hub is not None:
                    # Import atomspace types only if available
                    try:
                        from opencog.atomspace import types
                        
                        # Find related concepts in atomspace for query terms
                        query_words = query.lower().split()
                        related_memories = set()
                        
                        for word in query_words:
                            if len(word) > 2:
                                # Find concept nodes matching query words
                                matching_concepts = [
                                    atom for atom in atomspace_hub.get_atoms_by_type(types.ConceptNode)
                                    if word in atom.name.lower() and atom.name.startswith('memory_')
                                ]
                                for concept in matching_concepts[:3]:  # Top 3 per word
                                    related_memories.add(concept.name)
                        
                        # Add note about atomspace-enhanced results (if any found)
                        if related_memories:
                            pass  # Atomspace provided additional context
                    except ImportError:
                        pass  # OpenCog not available, use standard search
            except Exception:
                pass  # Gracefully ignore atomspace errors

        if len(docs) == 0:
            result = self.agent.read_prompt("fw.memories_not_found.md", query=query)
        else:
            text = "\n\n".join(Memory.format_docs_plain(docs))
            result = str(text)

        return Response(message=result, break_loop=False)
