import os
import asyncio
from python.helpers import dotenv, memory, perplexity_search, duckduckgo_search
from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from python.helpers.errors import handle_error
from python.helpers.searxng import search as searxng

SEARCH_ENGINE_RESULTS = 10

# Try to import AtomSpace tool hub for enhanced search tracking (optional)
try:
    from python.tools.atomspace_tool_hub import AtomSpaceToolHub
    ATOMSPACE_HUB_AVAILABLE = True
except ImportError:
    ATOMSPACE_HUB_AVAILABLE = False


class SearchEngine(Tool):
    async def execute(self, query="", **kwargs):


        searxng_result = await self.searxng_search(query)

        await self.agent.handle_intervention(
            searxng_result
        )  # wait for intervention and handle it, if paused

        # Optional: Track search queries in atomspace for learning
        if ATOMSPACE_HUB_AVAILABLE and searxng_result:
            await self._track_search_in_atomspace(query, searxng_result)

        return Response(message=searxng_result, break_loop=False)


    async def searxng_search(self, question):
        results = await searxng(question)
        return self.format_result_searxng(results, "Search Engine")

    def format_result_searxng(self, result, source):
        if isinstance(result, Exception):
            handle_error(result)
            return f"{source} search failed: {str(result)}"

        outputs = []
        for item in result["results"]:
            outputs.append(f"{item['title']}\n{item['url']}\n{item['content']}")

        return "\n\n".join(outputs[:SEARCH_ENGINE_RESULTS]).strip()
    
    async def _track_search_in_atomspace(self, query: str, result_str: str):
        """Track search queries and patterns in atomspace (optional)."""
        try:
            atomspace_hub = AtomSpaceToolHub.get_shared_atomspace()
            if atomspace_hub is not None:
                # Import atomspace types only if available
                try:
                    from opencog.atomspace import types
                    import time
                    
                    # Create search query node
                    search_id = f"search_{int(time.time())}"
                    search_node = atomspace_hub.add_node(types.ConceptNode, search_id)
                    
                    # Store query terms as concepts
                    query_words = query.lower().split()
                    for word in query_words[:5]:  # Limit to first 5 words
                        if len(word) > 2:
                            word_node = atomspace_hub.add_node(types.ConceptNode, f"query_{word}")
                            atomspace_hub.add_link(
                                types.EvaluationLink,
                                [
                                    atomspace_hub.add_node(types.PredicateNode, "queries"),
                                    search_node,
                                    word_node
                                ]
                            )
                    
                    # Track successful search (found results)
                    success = result_str and "failed" not in result_str.lower()
                    status_node = atomspace_hub.add_node(
                        types.ConceptNode, 
                        f"status_{'success' if success else 'failure'}"
                    )
                    atomspace_hub.add_link(
                        types.EvaluationLink,
                        [
                            atomspace_hub.add_node(types.PredicateNode, "has_status"),
                            search_node,
                            status_node
                        ]
                    )
                except ImportError:
                    pass  # OpenCog not available, skip tracking
        except Exception:
            pass  # Gracefully ignore atomspace errors
