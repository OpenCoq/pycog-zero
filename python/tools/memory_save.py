from python.helpers.memory import Memory
from python.helpers.tool import Tool, Response

# Try to import AtomSpace tool hub for cross-tool integration (optional)
try:
    from python.tools.atomspace_tool_hub import AtomSpaceToolHub
    ATOMSPACE_HUB_AVAILABLE = True
except ImportError:
    ATOMSPACE_HUB_AVAILABLE = False


class MemorySave(Tool):

    async def execute(self, text="", area="", **kwargs):

        if not area:
            area = Memory.Area.MAIN.value

        metadata = {"area": area, **kwargs}

        db = await Memory.get(self.agent)
        id = await db.insert_text(text, metadata)

        # Optional: Share saved memory with atomspace for cross-tool access
        if ATOMSPACE_HUB_AVAILABLE:
            try:
                atomspace_hub = AtomSpaceToolHub.get_shared_atomspace()
                if atomspace_hub is not None:
                    # Import atomspace types only if available
                    try:
                        from opencog.atomspace import types
                        # Create concept node for the saved memory
                        memory_node = atomspace_hub.add_node(types.ConceptNode, f"memory_{id}")
                        area_node = atomspace_hub.add_node(types.ConceptNode, f"area_{area}")
                        
                        # Link memory to its area
                        atomspace_hub.add_link(types.InheritanceLink, [memory_node, area_node])
                        
                        # Store memory content concepts (first few words)
                        content_words = text.lower().split()[:5]
                        for word in content_words:
                            if len(word) > 2:
                                word_node = atomspace_hub.add_node(types.ConceptNode, word)
                                atomspace_hub.add_link(
                                    types.EvaluationLink,
                                    [
                                        atomspace_hub.add_node(types.PredicateNode, "contains"),
                                        memory_node,
                                        word_node
                                    ]
                                )
                    except ImportError:
                        pass  # OpenCog not available, skip atomspace integration
            except Exception:
                pass  # Gracefully ignore atomspace errors

        result = self.agent.read_prompt("fw.memory_saved.md", memory_id=id)
        return Response(message=result, break_loop=False)
