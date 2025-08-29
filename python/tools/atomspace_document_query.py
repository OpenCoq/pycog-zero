"""
PyCog-Zero AtomSpace-Enhanced Document Query Tool
Integrates Agent-Zero document query capabilities with OpenCog AtomSpace semantic understanding
"""

from python.helpers.tool import Tool, Response
from python.helpers.document_query import DocumentQueryHelper
import json

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - document semantic analysis will be limited")
    OPENCOG_AVAILABLE = False


class AtomSpaceDocumentQuery(Tool):
    """Enhanced document query tool with AtomSpace semantic understanding."""
    
    def __init__(self, agent, name: str, method: str | None, args: dict, message: str, loop_data, **kwargs):
        super().__init__(agent, name, method, args, message, loop_data, **kwargs)
        self._doc_atomspace_initialized = False
        self.atomspace = None
        self.initialized = False
    
    def _initialize_document_atomspace(self):
        """Initialize AtomSpace for document semantic processing."""
        if self._doc_atomspace_initialized:
            return
        
        self._doc_atomspace_initialized = True
        
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self.initialized = True
                print("✓ AtomSpace document semantic analysis initialized")
            except Exception as e:
                print(f"⚠️ AtomSpace document analysis initialization failed: {e}")

    async def execute(self, document=None, query=None, queries=None, semantic_analysis=True, **kwargs):
        """Execute document query with optional semantic analysis."""
        
        self._initialize_document_atomspace()
        
        # Validate inputs
        document_uri = document or None
        query_list = queries if queries else ([query] if query else [])
        
        if not isinstance(document_uri, str) or not document_uri:
            return Response(message="Error: no document provided", break_loop=False)
        
        try:
            progress = []

            # Logging callback
            def progress_callback(msg):
                progress.append(msg)
                self.log.update(progress="\n".join(progress))
            
            # Perform standard document query
            helper = DocumentQueryHelper(self.agent, progress_callback)
            
            if not query_list:
                content = await helper.document_get_content(document_uri)
                semantic_result = ""
            else:
                _, content = await helper.document_qa(document_uri, query_list)
                semantic_result = ""
            
            # Add semantic analysis if available and requested
            if semantic_analysis and self.initialized:
                semantic_result = await self.perform_semantic_analysis(document_uri, content, query_list)
                
                # Combine standard result with semantic analysis
                enhanced_content = f"{content}\n\n=== SEMANTIC ANALYSIS ===\n{semantic_result}"
                return Response(message=enhanced_content, break_loop=False)
            else:
                return Response(message=content, break_loop=False)
                
        except Exception as e:  # pylint: disable=broad-exception-caught
            return Response(message=f"Error processing document: {e}", break_loop=False)
    
    async def perform_semantic_analysis(self, document_uri: str, content: str, queries: list):
        """Perform semantic analysis of document content using AtomSpace."""
        try:
            analysis_results = []
            
            # Store document content in AtomSpace
            doc_concepts = await self.store_document_in_atomspace(document_uri, content)
            analysis_results.append(f"📄 Document stored in AtomSpace: {len(doc_concepts)} concepts extracted")
            
            # Analyze document structure and semantics
            semantic_structure = await self.analyze_document_semantics(content)
            analysis_results.append(f"🔍 Semantic structure: {semantic_structure}")
            
            # Process queries against document knowledge
            if queries:
                query_analysis = await self.analyze_queries_against_document(queries, doc_concepts)
                analysis_results.append(f"❓ Query analysis: {query_analysis}")
            
            # Generate document insights
            insights = await self.generate_document_insights(doc_concepts)
            analysis_results.append(f"💡 Document insights: {insights}")
            
            # Cross-reference with existing knowledge
            cross_refs = await self.cross_reference_with_existing_knowledge(doc_concepts)
            analysis_results.append(f"🔗 Cross-references: {cross_refs}")
            
            return "\n".join(analysis_results)
            
        except Exception as e:
            return f"Semantic analysis failed: {e}"
    
    async def store_document_in_atomspace(self, document_uri: str, content: str):
        """Store document content as concepts and relationships in AtomSpace."""
        if not self.initialized:
            return []
        
        try:
            # Create document concept
            doc_concept = self.atomspace.add_node(types.ConceptNode, f"document_{document_uri.split('/')[-1]}")
            
            # Extract sentences and key terms
            sentences = content.split('.')
            concepts_created = []
            
            for i, sentence in enumerate(sentences[:20]):  # Limit to first 20 sentences
                sentence = sentence.strip()
                if len(sentence) < 10:  # Skip very short sentences
                    continue
                
                # Create sentence concept
                sentence_concept = self.atomspace.add_node(types.ConceptNode, f"sentence_{i}")
                concepts_created.append(sentence_concept)
                
                # Link sentence to document
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "contains_sentence"),
                        doc_concept,
                        sentence_concept
                    ]
                )
                
                # Extract key terms from sentence
                words = [word.strip().lower() for word in sentence.split() 
                        if len(word.strip()) > 3 and word.strip().isalpha()]
                
                for word in words[:10]:  # Limit words per sentence
                    word_concept = self.atomspace.add_node(types.ConceptNode, word)
                    concepts_created.append(word_concept)
                    
                    # Link word to sentence
                    self.atomspace.add_link(
                        types.EvaluationLink,
                        [
                            self.atomspace.add_node(types.PredicateNode, "sentence_contains_word"),
                            sentence_concept,
                            word_concept
                        ]
                    )
            
            # Create document metadata relationships
            doc_metadata = {
                "length": len(content),
                "sentences": len(sentences),
                "estimated_concepts": len(concepts_created)
            }
            
            for key, value in doc_metadata.items():
                meta_concept = self.atomspace.add_node(types.ConceptNode, f"{key}_{value}")
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "document_metadata"),
                        doc_concept,
                        meta_concept
                    ]
                )
            
            return concepts_created
            
        except Exception as e:
            print(f"Warning: Failed to store document in AtomSpace: {e}")
            return []
    
    async def analyze_document_semantics(self, content: str):
        """Analyze semantic structure of the document."""
        try:
            # Simple semantic analysis
            word_count = len(content.split())
            sentence_count = len(content.split('.'))
            paragraph_count = len(content.split('\n\n'))
            
            # Analyze concept density in AtomSpace
            if self.initialized:
                concept_nodes = len(self.atomspace.get_atoms_by_type(types.ConceptNode))
                evaluation_links = len(self.atomspace.get_atoms_by_type(types.EvaluationLink))
                concept_density = evaluation_links / max(concept_nodes, 1)
            else:
                concept_density = 0
            
            semantic_metrics = {
                "word_count": word_count,
                "sentence_count": sentence_count,
                "paragraph_count": paragraph_count,
                "avg_sentence_length": word_count / max(sentence_count, 1),
                "concept_density": round(concept_density, 2)
            }
            
            return json.dumps(semantic_metrics)
            
        except Exception as e:
            return f"Semantic analysis error: {e}"
    
    async def analyze_queries_against_document(self, queries: list, doc_concepts: list):
        """Analyze how queries relate to document concepts."""
        if not self.initialized:
            return "Query analysis unavailable (AtomSpace not initialized)"
        
        try:
            query_analysis = {}
            
            for query in queries:
                query_words = [word.strip().lower() for word in query.split() 
                             if len(word.strip()) > 2]
                
                matches = []
                semantic_connections = []
                
                for query_word in query_words:
                    # Direct concept matches
                    direct_matches = [
                        concept.name for concept in doc_concepts
                        if hasattr(concept, 'name') and query_word in concept.name.lower()
                    ]
                    matches.extend(direct_matches)
                    
                    # Semantic connections through AtomSpace
                    for concept in doc_concepts[:10]:  # Limit processing
                        if hasattr(concept, 'incoming'):
                            for link in concept.incoming:
                                if (link.type == types.EvaluationLink and 
                                    len(link.out) > 2 and 
                                    query_word in str(link).lower()):
                                    semantic_connections.append(concept.name if hasattr(concept, 'name') else str(concept))
                
                query_analysis[query] = {
                    "direct_matches": len(set(matches)),
                    "semantic_connections": len(set(semantic_connections)),
                    "coverage_score": min(1.0, (len(set(matches)) + len(set(semantic_connections))) / len(query_words))
                }
            
            return json.dumps(query_analysis)
            
        except Exception as e:
            return f"Query analysis error: {e}"
    
    async def generate_document_insights(self, doc_concepts: list):
        """Generate insights about the document using AtomSpace reasoning."""
        if not self.initialized:
            return "Document insights unavailable (AtomSpace not initialized)"
        
        try:
            insights = []
            
            # Analyze concept frequency and relationships
            concept_frequency = {}
            relationship_patterns = {}
            
            for concept in doc_concepts:
                if hasattr(concept, 'name'):
                    concept_frequency[concept.name] = concept_frequency.get(concept.name, 0) + 1
                
                if hasattr(concept, 'incoming'):
                    for link in concept.incoming:
                        if link.type == types.EvaluationLink and len(link.out) > 0:
                            predicate = link.out[0].name if hasattr(link.out[0], 'name') else str(link.out[0])
                            relationship_patterns[predicate] = relationship_patterns.get(predicate, 0) + 1
            
            # Top concepts
            top_concepts = sorted(concept_frequency.items(), key=lambda x: x[1], reverse=True)[:5]
            if top_concepts:
                insights.append(f"Key concepts: {', '.join([concept for concept, _ in top_concepts])}")
            
            # Top relationship patterns
            top_relationships = sorted(relationship_patterns.items(), key=lambda x: x[1], reverse=True)[:3]
            if top_relationships:
                insights.append(f"Main relationships: {', '.join([rel for rel, _ in top_relationships])}")
            
            # Document complexity metrics
            if len(doc_concepts) > 100:
                insights.append("High conceptual complexity detected")
            elif len(doc_concepts) > 50:
                insights.append("Moderate conceptual complexity")
            else:
                insights.append("Low conceptual complexity")
            
            return "; ".join(insights) if insights else "No specific insights generated"
            
        except Exception as e:
            return f"Insight generation error: {e}"
    
    async def cross_reference_with_existing_knowledge(self, doc_concepts: list):
        """Cross-reference document concepts with existing AtomSpace knowledge."""
        if not self.initialized:
            return "Cross-referencing unavailable (AtomSpace not initialized)"
        
        try:
            cross_references = []
            
            # Get all existing concepts in AtomSpace
            all_concepts = self.atomspace.get_atoms_by_type(types.ConceptNode)
            doc_concept_names = {concept.name for concept in doc_concepts if hasattr(concept, 'name')}
            
            # Find overlapping concepts
            existing_concepts = {concept.name for concept in all_concepts if hasattr(concept, 'name')}
            overlap = doc_concept_names.intersection(existing_concepts)
            
            if overlap:
                cross_references.append(f"{len(overlap)} concepts overlap with existing knowledge")
                
                # Analyze overlapping concept connections
                connected_domains = set()
                for overlapping_concept_name in list(overlap)[:5]:  # Limit processing
                    matching_concepts = [c for c in all_concepts if hasattr(c, 'name') and c.name == overlapping_concept_name]
                    
                    for concept in matching_concepts:
                        for link in concept.incoming:
                            if link.type == types.EvaluationLink:
                                for atom in link.out:
                                    if hasattr(atom, 'name') and atom.name not in doc_concept_names:
                                        # This connects to knowledge outside the document
                                        connected_domains.add(atom.name)
                
                if connected_domains:
                    cross_references.append(f"Connected domains: {', '.join(list(connected_domains)[:5])}")
            else:
                cross_references.append("No direct concept overlap with existing knowledge")
            
            # Novel concepts introduced by this document
            novel_concepts = doc_concept_names - existing_concepts
            if novel_concepts:
                cross_references.append(f"{len(novel_concepts)} new concepts introduced")
            
            return "; ".join(cross_references) if cross_references else "No cross-references found"
            
        except Exception as e:
            return f"Cross-referencing error: {e}"


def register():
    """Register the AtomSpace-enhanced document query tool with Agent-Zero."""
    return AtomSpaceDocumentQuery