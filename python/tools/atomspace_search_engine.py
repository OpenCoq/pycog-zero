"""
PyCog-Zero AtomSpace-Enhanced Search Engine Tool
Integrates Agent-Zero search capabilities with OpenCog AtomSpace cognitive search
"""

import os
import asyncio
from python.helpers import dotenv, memory, perplexity_search, duckduckgo_search
from python.helpers.tool import Tool, Response
from python.helpers.print_style import PrintStyle
from python.helpers.errors import handle_error
from python.helpers.searxng import search as searxng
import json

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - cognitive search will use standard search only")
    OPENCOG_AVAILABLE = False

SEARCH_ENGINE_RESULTS = 10


class AtomSpaceSearchEngine(Tool):
    """Enhanced search engine with AtomSpace cognitive capabilities."""
    
    def __init__(self, agent, name: str, method: str | None, args: dict, message: str, loop_data, **kwargs):
        super().__init__(agent, name, method, args, message, loop_data, **kwargs)
        self._search_initialized = False
        self.atomspace = None
        self.initialized = False
        
    def _initialize_search_atomspace(self):
        """Initialize AtomSpace for cognitive search enhancement."""
        if self._search_initialized:
            return
        
        self._search_initialized = True
        
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self.initialized = True
                print("✓ AtomSpace search enhancement initialized")
            except Exception as e:
                print(f"⚠️ AtomSpace search initialization failed: {e}")
    
    async def execute(self, query="", cognitive_enhancement=True, **kwargs):
        """Execute enhanced search with optional cognitive processing."""
        
        self._initialize_search_atomspace()
        
        # Perform standard search
        standard_result = await self.searxng_search(query)
        
        # Add cognitive enhancement if available and requested
        if cognitive_enhancement and self.initialized:
            enhanced_result = await self.enhance_search_with_atomspace(query, standard_result)
            return Response(message=enhanced_result, break_loop=False)
        else:
            return Response(message=standard_result, break_loop=False)

    async def searxng_search(self, question):
        """Perform standard searxng search."""
        results = await searxng(question)
        return self.format_result_searxng(results, "Search Engine")

    def format_result_searxng(self, result, source):
        """Format searxng results."""
        if isinstance(result, Exception):
            handle_error(result)
            return f"{source} search failed: {str(result)}"

        outputs = []
        for item in result["results"]:
            outputs.append(f"{item['title']}\n{item['url']}\n{item['content']}")

        return "\n\n".join(outputs[:SEARCH_ENGINE_RESULTS]).strip()
    
    async def enhance_search_with_atomspace(self, query: str, standard_result: str):
        """Enhance search results with AtomSpace cognitive processing."""
        try:
            # Store search results in AtomSpace for cognitive processing
            await self.store_search_results_in_atomspace(query, standard_result)
            
            # Generate cognitive insights
            cognitive_insights = await self.generate_cognitive_search_insights(query)
            
            # Combine standard results with cognitive enhancement
            enhanced_output = f"=== STANDARD SEARCH RESULTS ===\n{standard_result}\n\n"
            enhanced_output += f"=== COGNITIVE SEARCH ENHANCEMENT ===\n{cognitive_insights}"
            
            return enhanced_output
            
        except Exception as e:
            print(f"Warning: Cognitive search enhancement failed: {e}")
            return f"{standard_result}\n\n[Note: Cognitive enhancement unavailable: {e}]"
    
    async def store_search_results_in_atomspace(self, query: str, results: str):
        """Store search results and query in AtomSpace for cognitive processing."""
        if not self.initialized:
            return
        
        try:
            # Create query concept
            query_concept = self.atomspace.add_node(types.ConceptNode, f"search_query_{query}")
            
            # Extract key terms from query
            query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
            
            for term in query_terms:
                term_concept = self.atomspace.add_node(types.ConceptNode, term)
                
                # Create relationship: query contains term
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "contains_term"),
                        query_concept,
                        term_concept
                    ]
                )
            
            # Process search results - extract key concepts
            result_lines = results.split('\n')
            result_concepts = []
            
            for line in result_lines:
                words = line.lower().split()
                for word in words:
                    # Simple heuristic: words longer than 3 chars, no numbers
                    if len(word) > 3 and word.isalpha():
                        word_concept = self.atomspace.add_node(types.ConceptNode, word)
                        result_concepts.append(word_concept)
                        
                        # Create relationship: query result contains concept
                        self.atomspace.add_link(
                            types.EvaluationLink,
                            [
                                self.atomspace.add_node(types.PredicateNode, "search_result_contains"),
                                query_concept,
                                word_concept
                            ]
                        )
                
                # Limit processing to avoid too many concepts
                if len(result_concepts) > 50:
                    break
            
            print(f"✓ Stored search results in AtomSpace: {len(result_concepts)} concepts extracted")
            
        except Exception as e:
            print(f"Warning: Failed to store search results in AtomSpace: {e}")
    
    async def generate_cognitive_search_insights(self, query: str):
        """Generate cognitive insights about the search query and results."""
        if not self.initialized:
            return "Cognitive insights unavailable (AtomSpace not initialized)"
        
        try:
            insights = []
            
            # Analyze query terms and their relationships
            query_terms = [term.strip().lower() for term in query.split() if len(term.strip()) > 2]
            
            insights.append(f"🧠 Cognitive Analysis for Query: '{query}'")
            insights.append(f"📊 AtomSpace State: {len(self.atomspace)} total atoms")
            
            # Find related concepts for each query term
            for term in query_terms:
                related_concepts = self.find_related_concepts(term)
                if related_concepts:
                    insights.append(f"🔗 '{term}' connects to: {', '.join(related_concepts[:5])}")
            
            # Find cross-connections between query terms
            cross_connections = self.find_cross_connections(query_terms)
            if cross_connections:
                insights.append(f"⚡ Cross-connections found: {cross_connections}")
            
            # Reasoning suggestions
            reasoning_suggestions = self.generate_reasoning_suggestions(query_terms)
            if reasoning_suggestions:
                insights.append(f"💡 Reasoning suggestions: {reasoning_suggestions}")
            
            # Search pattern analysis
            search_patterns = self.analyze_search_patterns()
            if search_patterns:
                insights.append(f"📈 Search patterns: {search_patterns}")
            
            return "\n".join(insights)
            
        except Exception as e:
            return f"Cognitive insights generation failed: {e}"
    
    def find_related_concepts(self, term: str):
        """Find concepts related to a given term through AtomSpace links."""
        try:
            related = []
            
            # Find concept nodes that contain the term
            matching_concepts = [
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if term in atom.name.lower()
            ]
            
            for concept in matching_concepts[:3]:  # Limit to avoid too many results
                # Find what this concept is connected to
                for link in concept.incoming:
                    if link.type == types.EvaluationLink:
                        for atom in link.out:
                            if atom != concept and atom.type == types.ConceptNode:
                                related.append(atom.name)
            
            return list(set(related))[:10]  # Return unique concepts, max 10
            
        except Exception as e:
            print(f"Warning: Failed to find related concepts for '{term}': {e}")
            return []
    
    def find_cross_connections(self, terms: list):
        """Find connections between different query terms."""
        try:
            connections = []
            
            for i, term1 in enumerate(terms):
                for term2 in terms[i+1:]:
                    # Look for concepts that connect both terms
                    term1_concepts = [
                        atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                        if term1 in atom.name.lower()
                    ]
                    term2_concepts = [
                        atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                        if term2 in atom.name.lower()
                    ]
                    
                    # Find shared connections
                    for concept1 in term1_concepts:
                        for concept2 in term2_concepts:
                            # Check if they're connected through any links
                            for link in concept1.incoming:
                                if concept2 in link.out:
                                    connections.append(f"{term1}↔{term2}")
                                    break
            
            return list(set(connections))[:5]  # Max 5 connections
            
        except Exception as e:
            print(f"Warning: Failed to find cross-connections: {e}")
            return []
    
    def generate_reasoning_suggestions(self, terms: list):
        """Generate reasoning suggestions based on AtomSpace patterns."""
        try:
            suggestions = []
            
            # Analyze patterns in the AtomSpace
            evaluation_links = self.atomspace.get_atoms_by_type(types.EvaluationLink)
            predicates = {}
            
            # Count predicate usage
            for link in evaluation_links:
                if len(link.out) > 0 and link.out[0].type == types.PredicateNode:
                    predicate = link.out[0].name
                    predicates[predicate] = predicates.get(predicate, 0) + 1
            
            # Suggest based on common patterns
            top_predicates = sorted(predicates.items(), key=lambda x: x[1], reverse=True)[:3]
            
            for predicate, count in top_predicates:
                suggestions.append(f"Consider exploring '{predicate}' relationships ({count} instances)")
            
            return "; ".join(suggestions)
            
        except Exception as e:
            print(f"Warning: Failed to generate reasoning suggestions: {e}")
            return ""
    
    def analyze_search_patterns(self):
        """Analyze patterns in search queries and results."""
        try:
            patterns = []
            
            # Count search-related atoms
            search_queries = [
                atom for atom in self.atomspace.get_atoms_by_type(types.ConceptNode)
                if "search_query_" in atom.name
            ]
            
            search_result_links = [
                atom for atom in self.atomspace.get_atoms_by_type(types.EvaluationLink)
                if len(atom.out) > 0 and atom.out[0].type == types.PredicateNode 
                and "search_result_contains" in atom.out[0].name
            ]
            
            patterns.append(f"{len(search_queries)} previous queries")
            patterns.append(f"{len(search_result_links)} result relationships")
            
            # Simple pattern detection
            if len(search_queries) > 5:
                patterns.append("High search activity detected")
            
            return "; ".join(patterns)
            
        except Exception as e:
            print(f"Warning: Failed to analyze search patterns: {e}")
            return ""


def register():
    """Register the AtomSpace-enhanced search engine tool with Agent-Zero."""
    return AtomSpaceSearchEngine