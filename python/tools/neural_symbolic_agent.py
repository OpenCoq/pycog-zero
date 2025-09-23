"""
Neural-Symbolic Agent Tool for PyCog-Zero

This tool provides neural-symbolic cognitive processing capabilities for Agent-Zero,
integrating PyTorch neural networks with OpenCog symbolic reasoning through the
neural-symbolic bridge.
"""

# Try to import dependencies with graceful fallback
try:
    from python.helpers.tool import Tool, Response
    AGENT_ZERO_AVAILABLE = True
except ImportError:
    print("⚠️ Agent-Zero framework not available")
    AGENT_ZERO_AVAILABLE = False
    # Mock Tool and Response for fallback
    class Tool:
        def __init__(self, agent, name: str, method: str | None, args: dict, message: str, loop_data, **kwargs):
            pass
    
    class Response:
        def __init__(self, message: str, break_loop: bool):
            self.message = message
            self.break_loop = break_loop

from python.helpers.neural_symbolic_bridge import NeuralSymbolicBridge, CognitiveAttentionMechanism

# Try to import PyTorch (will use mock from bridge if not available)
try:
    import torch
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False
    # Use mock torch from neural_symbolic_bridge
    from python.helpers.neural_symbolic_bridge import torch

import asyncio
from typing import Dict, List, Any, Optional
import json
import math

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    OPENCOG_AVAILABLE = True
except ImportError:
    OPENCOG_AVAILABLE = False
    # Mock for graceful fallback
    class MockAtomSpace:
        def add_node(self, atom_type, name):
            return MockAtom(name)
    
    class MockAtom:
        def __init__(self, name):
            self.name = name
    
    class MockTypes:
        ConceptNode = "ConceptNode"
        PredicateNode = "PredicateNode"
    
    types = MockTypes()


class NeuralSymbolicTool(Tool):
    """Agent-Zero tool for neural-symbolic cognitive processing."""
    
    def __init__(self, agent, name: str, method: str | None, args: dict, message: str, loop_data, **kwargs):
        """Initialize the neural-symbolic tool."""
        super().__init__(agent, name, method, args, message, loop_data, **kwargs)
        self._initialized = False
        self.atomspace = None
        self.bridge = None
        self.attention_mechanism = None
        self.embedding_dim = 128
    
    def _initialize_components(self):
        """Initialize the neural-symbolic components if not already done."""
        if self._initialized:
            return
        
        try:
            # Initialize AtomSpace
            if OPENCOG_AVAILABLE:
                self.atomspace = AtomSpace()
                print("✓ OpenCog AtomSpace initialized for neural-symbolic processing")
            else:
                self.atomspace = MockAtomSpace()
                print("⚠️ Using mock AtomSpace - install OpenCog for full functionality")
            
            # Initialize neural-symbolic bridge
            self.bridge = NeuralSymbolicBridge(self.atomspace, self.embedding_dim)
            
            # Initialize cognitive attention mechanism
            self.attention_mechanism = CognitiveAttentionMechanism(self.embedding_dim, num_heads=8)
            
            self._initialized = True
            print("✓ Neural-symbolic bridge components initialized")
            
        except Exception as e:
            print(f"⚠️ Neural-symbolic tool initialization warning: {e}")
            # Create minimal fallback components
            self.atomspace = MockAtomSpace() if not OPENCOG_AVAILABLE else AtomSpace()
            self.bridge = NeuralSymbolicBridge(self.atomspace, self.embedding_dim)
            self.attention_mechanism = CognitiveAttentionMechanism(self.embedding_dim)
            self._initialized = True
    
    async def execute(self, operation: str = "embed_concepts", **kwargs) -> Response:
        """
        Execute neural-symbolic operations.
        
        Available operations:
        - embed_concepts: Convert concepts to neural embeddings
        - neural_reasoning: Apply neural attention to symbolic concepts
        - symbolic_grounding: Ground neural outputs in symbolic representation
        - bridge_tensors: Convert between tensors and atoms
        - train_embeddings: Train the embedding network
        """
        self._initialize_components()
        
        try:
            if operation == "embed_concepts":
                return await self.embed_conceptual_knowledge(kwargs)
            elif operation == "neural_reasoning":
                return await self.perform_neural_reasoning(kwargs)
            elif operation == "symbolic_grounding":
                return await self.ground_symbols_in_experience(kwargs)
            elif operation == "bridge_tensors":
                return await self.bridge_tensor_operations(kwargs)
            elif operation == "train_embeddings":
                return await self.train_embedding_network(kwargs)
            elif operation == "analyze_attention":
                return await self.analyze_attention_patterns(kwargs)
            elif operation == "cross_modal_fusion":
                return await self.perform_cross_modal_fusion(kwargs)
            elif operation == "temporal_reasoning":
                return await self.perform_temporal_reasoning(kwargs)
            elif operation == "meta_cognitive_analysis":
                return await self.perform_meta_cognitive_analysis(kwargs)
            else:
                return Response(
                    message=f"Unknown operation '{operation}'. Available: embed_concepts, neural_reasoning, symbolic_grounding, "
                           f"bridge_tensors, train_embeddings, analyze_attention, cross_modal_fusion, temporal_reasoning, meta_cognitive_analysis",
                    break_loop=False
                )
        
        except Exception as e:
            return Response(
                message=f"Neural-symbolic processing error: {str(e)}",
                break_loop=False
            )
    
    async def embed_conceptual_knowledge(self, data: dict) -> Response:
        """
        Convert Agent-Zero conceptual knowledge to neural embeddings.
        
        Args:
            data: Dictionary with 'concepts' key containing list of concept strings
            
        Returns:
            Response with embedding information
        """
        concepts = data.get("concepts", [])
        if not concepts:
            return Response(
                message="No concepts provided for embedding",
                break_loop=False
            )
        
        try:
            # Create embeddings for concepts
            embeddings = self.bridge.embed_concepts(concepts)
            
            # Store embeddings info
            embedding_stats = {
                "num_concepts": len(concepts),
                "embedding_shape": list(embeddings.shape),
                "embedding_dim": self.bridge.embedding_dim,
                "mean_magnitude": float(torch.norm(embeddings, dim=1).mean()),
                "opencog_available": OPENCOG_AVAILABLE
            }
            
            return Response(
                message=f"Successfully embedded {len(concepts)} concepts into {self.embedding_dim}-dimensional neural space. "
                       f"Mean embedding magnitude: {embedding_stats['mean_magnitude']:.3f}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error embedding concepts: {str(e)}",
                break_loop=False
            )
    
    async def perform_neural_reasoning(self, data: dict) -> Response:
        """
        Combine neural processing with symbolic reasoning using attention mechanisms.
        
        Args:
            data: Dictionary with query, concepts, and optional context
            
        Returns:
            Response with reasoning results
        """
        query = data.get("query", "")
        concepts = data.get("concepts", [])
        context = data.get("context", [])
        
        if not concepts:
            return Response(
                message="No concepts provided for neural reasoning",
                break_loop=False
            )
        
        try:
            # Convert concepts to embeddings
            concept_embeddings = self.bridge.embed_concepts(concepts)
            
            # Apply attention mechanism
            attended_output, attention_weights = self.attention_mechanism(concept_embeddings)
            
            # Analyze attention patterns
            max_attention_idx = torch.argmax(attention_weights.mean(dim=0))
            focused_concept = concepts[max_attention_idx] if max_attention_idx < len(concepts) else "unknown"
            
            # Compute attention distribution
            attention_dist = attention_weights.mean(dim=0).tolist()
            
            reasoning_result = {
                "query": query,
                "num_concepts_processed": len(concepts),
                "focused_concept": focused_concept,
                "attention_entropy": float(-torch.sum(attention_weights.mean(dim=0) * torch.log(attention_weights.mean(dim=0) + 1e-8))),
                "output_magnitude": float(torch.norm(attended_output).mean())
            }
            
            return Response(
                message=f"Neural reasoning completed on {len(concepts)} concepts. "
                       f"Primary focus: '{focused_concept}' with attention entropy: {reasoning_result['attention_entropy']:.3f}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in neural reasoning: {str(e)}",
                break_loop=False
            )
    
    async def ground_symbols_in_experience(self, data: dict) -> Response:
        """
        Ground neural outputs back into symbolic representation.
        
        Args:
            data: Dictionary with tensor data or embeddings to ground
            
        Returns:
            Response with grounding results
        """
        concepts = data.get("concepts", [])
        num_outputs = data.get("num_outputs", 5)
        
        if not concepts:
            return Response(
                message="No concepts provided for symbolic grounding",
                break_loop=False
            )
        
        try:
            # Create embeddings and process through attention
            embeddings = self.bridge.embed_concepts(concepts)
            attended_output, _ = self.attention_mechanism(embeddings)
            
            # Convert back to symbolic atoms
            grounded_atoms = self.bridge.tensor_to_atomspace(
                attended_output.squeeze(0)[:num_outputs],
                atom_names=[f"grounded_concept_{i}" for i in range(num_outputs)]
            )
            
            grounding_result = {
                "input_concepts": len(concepts),
                "grounded_atoms": len(grounded_atoms),
                "atom_names": [atom.name if hasattr(atom, 'name') else str(atom) for atom in grounded_atoms[:3]]
            }
            
            return Response(
                message=f"Successfully grounded {len(concepts)} input concepts into {len(grounded_atoms)} symbolic atoms. "
                       f"Sample atoms: {', '.join(grounding_result['atom_names'])}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in symbolic grounding: {str(e)}",
                break_loop=False
            )
    
    async def bridge_tensor_operations(self, data: dict) -> Response:
        """
        Perform bidirectional conversion between tensors and AtomSpace.
        
        Args:
            data: Dictionary with operation details
            
        Returns:
            Response with conversion results
        """
        operation_type = data.get("type", "concepts_to_tensors")
        concepts = data.get("concepts", [])
        
        try:
            if operation_type == "concepts_to_tensors":
                if not concepts:
                    return Response(
                        message="No concepts provided for tensor conversion",
                        break_loop=False
                    )
                
                # Convert concepts to atoms to tensors
                atoms = []
                for concept in concepts:
                    if OPENCOG_AVAILABLE:
                        atom = self.atomspace.add_node(types.ConceptNode, concept)
                    else:
                        atom = MockAtom(concept)
                    atoms.append(atom)
                
                tensor = self.bridge.atomspace_to_tensor(atoms)
                
                return Response(
                    message=f"Converted {len(concepts)} concepts to tensor shape {list(tensor.shape)}. "
                           f"Tensor norm: {float(torch.norm(tensor)):.3f}",
                    break_loop=False
                )
            
            elif operation_type == "tensors_to_atoms":
                num_atoms = data.get("num_atoms", 3)
                # Create random tensor and convert to atoms
                random_tensor = torch.randn(num_atoms, self.embedding_dim)
                atoms = self.bridge.tensor_to_atomspace(random_tensor)
                
                atom_names = [atom.name if hasattr(atom, 'name') else str(atom) for atom in atoms]
                
                return Response(
                    message=f"Generated {len(atoms)} atoms from tensor: {', '.join(atom_names)}",
                    break_loop=False
                )
            
            else:
                return Response(
                    message=f"Unknown bridge operation type: {operation_type}",
                    break_loop=False
                )
        
        except Exception as e:
            return Response(
                message=f"Error in tensor bridge operations: {str(e)}",
                break_loop=False
            )
    
    async def train_embedding_network(self, data: dict) -> Response:
        """
        Train the neural embedding network on provided concepts.
        
        Args:
            data: Dictionary with training parameters
            
        Returns:
            Response with training results
        """
        concepts = data.get("concepts", [])
        epochs = data.get("epochs", 50)
        learning_rate = data.get("learning_rate", 0.001)
        
        if not concepts:
            return Response(
                message="No concepts provided for training",
                break_loop=False
            )
        
        try:
            # Create target embeddings (could be from pre-trained model)
            embeddings = self.bridge.embed_concepts(concepts)
            target_embeddings = embeddings + torch.randn_like(embeddings) * 0.1  # Add some noise as target
            
            # Create atoms for training
            atoms = []
            for concept in concepts:
                if OPENCOG_AVAILABLE:
                    atom = self.atomspace.add_node(types.ConceptNode, concept)
                else:
                    atom = MockAtom(concept)
                atoms.append(atom)
            
            # Train the embedding network
            self.bridge.train_embeddings(atoms, target_embeddings, learning_rate, epochs)
            
            return Response(
                message=f"Training completed on {len(concepts)} concepts over {epochs} epochs. "
                       f"Learning rate: {learning_rate}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in training embedding network: {str(e)}",
                break_loop=False
            )
    
    async def analyze_attention_patterns(self, data: dict) -> Response:
        """
        Analyze attention patterns in neural-symbolic processing with advanced mechanisms.
        
        Args:
            data: Dictionary with concepts and analysis options
            
        Returns:
            Response with comprehensive attention analysis
        """
        concepts = data.get("concepts", [])
        analysis_mode = data.get("mode", "basic")  # basic, advanced, multi_scale, hierarchical, temporal, meta
        use_advanced = data.get("use_advanced", True)
        
        if not concepts:
            return Response(
                message="No concepts provided for attention analysis",
                break_loop=False
            )
        
        try:
            # Create embeddings
            embeddings = self.bridge.embed_concepts(concepts)
            
            # Apply attention with specified mode
            attended_output, attention_weights = self.attention_mechanism(
                embeddings, 
                use_advanced=use_advanced, 
                attention_mode=analysis_mode
            )
            
            # Perform comprehensive attention analysis
            analysis = self.attention_mechanism.analyze_attention_patterns(embeddings, attention_weights)
            
            # Add concept-specific analysis
            max_attention_idx = analysis.get('max_attention_index', 0)
            max_attention_concept = concepts[max_attention_idx] if max_attention_idx < len(concepts) else concepts[0]
            
            # Enhanced analysis for different modes
            if use_advanced and analysis_mode != "basic":
                analysis = self._enhance_attention_analysis(analysis, analysis_mode, embeddings, attention_weights)
            
            # Create comprehensive report
            analysis_report = {
                "mode": analysis_mode,
                "concepts_analyzed": len(concepts),
                "max_attention_concept": max_attention_concept,
                "attention_entropy": analysis.get('attention_entropy', 0.1),
                "attention_variance": analysis.get('attention_variance', 0.1),
                "embedding_dim": self.embedding_dim,
                "advanced_features_used": use_advanced and self.attention_mechanism.advanced_available
            }
            
            # Add mode-specific insights
            if analysis_mode == "multi_scale":
                analysis_report["multi_scale_insights"] = analysis.get('multi_scale_patterns', {})
            elif analysis_mode == "hierarchical":
                analysis_report["hierarchical_insights"] = analysis.get('hierarchical_structure', {})
            elif analysis_mode == "temporal":
                analysis_report["temporal_insights"] = analysis.get('temporal_consistency', 0.5)
                if hasattr(self.attention_mechanism, 'temporal'):
                    analysis_report["memory_state"] = self.attention_mechanism.temporal.get_memory_summary()
            
            message_parts = [
                f"Advanced attention analysis completed using {analysis_mode} mode on {len(concepts)} concepts.",
                f"Primary focus: '{max_attention_concept}' with entropy: {analysis_report['attention_entropy']:.3f}.",
            ]
            
            if use_advanced and self.attention_mechanism.advanced_available:
                message_parts.append("Advanced attention mechanisms were successfully utilized.")
            else:
                message_parts.append("Using basic attention (advanced features not available).")
            
            return Response(
                message=" ".join(message_parts),
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in advanced attention analysis: {str(e)}",
                break_loop=False
            )
    
    def _enhance_attention_analysis(self, base_analysis: dict, mode: str, embeddings: torch.Tensor, attention_weights: torch.Tensor) -> dict:
        """Enhance attention analysis with mode-specific insights."""
        enhanced = base_analysis.copy()
        
        try:
            if mode == "multi_scale" and hasattr(self.attention_mechanism, 'multi_scale'):
                # Multi-scale specific analysis
                enhanced['scale_distribution'] = self._analyze_scale_distribution(attention_weights)
                enhanced['cross_scale_coherence'] = self._measure_cross_scale_coherence(attention_weights)
                
            elif mode == "hierarchical" and hasattr(self.attention_mechanism, 'hierarchical'):
                # Hierarchical specific analysis
                enhanced['hierarchy_depth'] = self._estimate_hierarchy_depth(attention_weights)
                enhanced['level_coherence'] = self._measure_level_coherence(attention_weights)
                
            elif mode == "temporal" and hasattr(self.attention_mechanism, 'temporal'):
                # Temporal specific analysis
                enhanced['temporal_drift'] = self._measure_temporal_drift()
                enhanced['memory_utilization'] = self._calculate_memory_utilization()
                
            elif mode == "meta" and hasattr(self.attention_mechanism, 'meta_attention'):
                # Meta-attention specific analysis
                enhanced['attention_diversity'] = self._measure_attention_diversity(attention_weights)
                enhanced['meta_coherence'] = self._measure_meta_coherence(attention_weights)
                
        except Exception as e:
            enhanced['enhancement_error'] = str(e)
            
        return enhanced
    
    def _analyze_scale_distribution(self, attention_weights: torch.Tensor) -> dict:
        """Analyze distribution across different attention scales."""
        try:
            if isinstance(attention_weights, list):
                # Multiple scale weights
                scale_energies = []
                for i, weights in enumerate(attention_weights):
                    energy = float(torch.sum(weights * weights)) if hasattr(weights, 'sum') else 0.1
                    scale_energies.append(energy)
                
                total_energy = sum(scale_energies)
                normalized_energies = [e / (total_energy + 1e-8) for e in scale_energies]
                
                return {
                    'scale_energies': scale_energies,
                    'normalized_distribution': normalized_energies,
                    'dominant_scale': int(torch.argmax(torch.tensor(scale_energies))) if scale_energies else 0,
                    'scale_entropy': -sum(p * math.log(p + 1e-8) for p in normalized_energies if p > 0)
                }
            else:
                return {'single_scale': float(torch.sum(attention_weights * attention_weights)) if hasattr(attention_weights, 'sum') else 0.1}
        except:
            return {'analysis_failed': True}
    
    def _measure_cross_scale_coherence(self, attention_weights: torch.Tensor) -> float:
        """Measure coherence across different attention scales."""
        try:
            if isinstance(attention_weights, list) and len(attention_weights) > 1:
                coherence_sum = 0.0
                pairs = 0
                
                for i in range(len(attention_weights)):
                    for j in range(i + 1, len(attention_weights)):
                        w1, w2 = attention_weights[i], attention_weights[j]
                        if hasattr(w1, 'flatten') and hasattr(w2, 'flatten'):
                            # Reshape to compatible sizes if needed
                            if w1.shape != w2.shape:
                                min_size = min(w1.numel(), w2.numel())
                                w1_flat = w1.flatten()[:min_size]
                                w2_flat = w2.flatten()[:min_size]
                            else:
                                w1_flat = w1.flatten()
                                w2_flat = w2.flatten()
                            
                            coherence = float(torch.cosine_similarity(w1_flat, w2_flat, dim=0))
                            coherence_sum += coherence
                            pairs += 1
                
                return coherence_sum / pairs if pairs > 0 else 0.5
            else:
                return 1.0  # Perfect coherence for single scale
        except:
            return 0.5
    
    def _estimate_hierarchy_depth(self, attention_weights: torch.Tensor) -> int:
        """Estimate the effective depth of hierarchical attention."""
        try:
            if isinstance(attention_weights, list):
                return len(attention_weights)
            else:
                # Estimate based on attention pattern complexity
                if hasattr(attention_weights, 'std'):
                    complexity = float(attention_weights.std())
                    return min(5, max(1, int(complexity * 10)))  # Rough estimation
                else:
                    return 1
        except:
            return 1
    
    def _measure_level_coherence(self, attention_weights: torch.Tensor) -> float:
        """Measure coherence between hierarchical levels."""
        try:
            if isinstance(attention_weights, list) and len(attention_weights) > 1:
                # Similar to cross-scale coherence
                return self._measure_cross_scale_coherence(attention_weights)
            else:
                return 1.0
        except:
            return 0.5
    
    def _measure_temporal_drift(self) -> float:
        """Measure how much attention patterns have drifted over time."""
        try:
            if hasattr(self.attention_mechanism, 'temporal') and self.attention_mechanism.temporal.attention_memory:
                memory = self.attention_mechanism.temporal.attention_memory
                if len(memory) > 1:
                    recent = memory[-1]
                    earlier = memory[max(0, len(memory) - 5)]  # Compare with 5 steps ago
                    
                    if hasattr(recent, 'shape') and hasattr(earlier, 'shape') and recent.shape == earlier.shape:
                        drift = float(torch.norm(recent - earlier))
                        return min(1.0, drift)  # Normalize to [0, 1]
                    
            return 0.0  # No drift if no temporal data
        except:
            return 0.0
    
    def _calculate_memory_utilization(self) -> float:
        """Calculate how much of the attention memory is being utilized."""
        try:
            if hasattr(self.attention_mechanism, 'temporal'):
                summary = self.attention_mechanism.temporal.get_memory_summary()
                return summary.get('memory_utilization', 0.0)
            else:
                return 0.0
        except:
            return 0.0
    
    def _measure_attention_diversity(self, attention_weights: torch.Tensor) -> float:
        """Measure diversity in attention patterns for meta-attention."""
        try:
            if isinstance(attention_weights, list):
                # Calculate diversity across different attention mechanisms
                patterns = []
                for weights in attention_weights:
                    if hasattr(weights, 'flatten'):
                        pattern = weights.flatten()
                        patterns.append(pattern)
                
                if len(patterns) > 1:
                    diversity = 0.0
                    for i in range(len(patterns)):
                        for j in range(i + 1, len(patterns)):
                            # Measure dissimilarity
                            if patterns[i].shape == patterns[j].shape:
                                similarity = float(torch.cosine_similarity(patterns[i], patterns[j], dim=0))
                                diversity += (1.0 - similarity)
                    
                    return diversity / (len(patterns) * (len(patterns) - 1) / 2)
                else:
                    return 0.0
            else:
                # Single pattern - measure internal diversity
                if hasattr(attention_weights, 'std'):
                    return float(attention_weights.std())
                else:
                    return 0.1
        except:
            return 0.1
    
    def _measure_meta_coherence(self, attention_weights: torch.Tensor) -> float:
        """Measure coherence at the meta-attention level."""
        try:
            # Meta-coherence measures how well different attention mechanisms agree
            if isinstance(attention_weights, list) and len(attention_weights) > 1:
                return 1.0 - self._measure_attention_diversity(attention_weights)
            else:
                return 1.0
        except:
            return 0.5

    async def perform_cross_modal_fusion(self, data: dict) -> Response:
        """
        Perform cross-modal fusion between neural and symbolic representations.
        
        Args:
            data: Dictionary with neural_concepts and symbolic_concepts
            
        Returns:
            Response with fusion results
        """
        neural_concepts = data.get("neural_concepts", [])
        symbolic_concepts = data.get("symbolic_concepts", [])
        
        if not neural_concepts or not symbolic_concepts:
            return Response(
                message="Both neural_concepts and symbolic_concepts are required for cross-modal fusion",
                break_loop=False
            )
        
        try:
            # Create embeddings for both modalities
            neural_embeddings = self.bridge.embed_concepts(neural_concepts)
            symbolic_embeddings = self.bridge.embed_concepts(symbolic_concepts)
            
            # Apply cross-modal attention
            fused_embeddings, attention_weights = self.attention_mechanism.apply_cross_modal_attention(
                neural_embeddings, symbolic_embeddings
            )
            
            # Analyze fusion quality
            fusion_analysis = {
                "neural_concepts": len(neural_concepts),
                "symbolic_concepts": len(symbolic_concepts),
                "fusion_dimension": list(fused_embeddings.shape),
                "cross_modal_coherence": self._measure_cross_modal_coherence(attention_weights),
                "fusion_strength": float(torch.norm(fused_embeddings).mean()) if hasattr(fused_embeddings, 'norm') else 1.0
            }
            
            return Response(
                message=f"Cross-modal fusion completed between {len(neural_concepts)} neural and {len(symbolic_concepts)} symbolic concepts. "
                       f"Coherence: {fusion_analysis['cross_modal_coherence']:.3f}, "
                       f"Fusion strength: {fusion_analysis['fusion_strength']:.3f}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in cross-modal fusion: {str(e)}",
                break_loop=False
            )
    
    def _measure_cross_modal_coherence(self, attention_weights: dict) -> float:
        """Measure coherence between cross-modal attention patterns."""
        try:
            if 'neural_to_symbolic' in attention_weights and 'symbolic_to_neural' in attention_weights:
                neural_attn = attention_weights['neural_to_symbolic']
                symbolic_attn = attention_weights['symbolic_to_neural']
                
                # Measure symmetry in cross-modal attention
                if hasattr(neural_attn, 'flatten') and hasattr(symbolic_attn, 'flatten'):
                    neural_flat = neural_attn.flatten()
                    symbolic_flat = symbolic_attn.flatten()
                    
                    # Pad or truncate to same size
                    min_size = min(len(neural_flat), len(symbolic_flat))
                    neural_flat = neural_flat[:min_size]
                    symbolic_flat = symbolic_flat[:min_size]
                    
                    coherence = float(torch.cosine_similarity(neural_flat, symbolic_flat, dim=0))
                    return (coherence + 1.0) / 2.0  # Normalize to [0, 1]
                    
            return 0.5  # Neutral coherence
        except:
            return 0.5
    
    async def perform_temporal_reasoning(self, data: dict) -> Response:
        """
        Perform temporal reasoning with memory-based attention.
        
        Args:
            data: Dictionary with concepts and temporal context
            
        Returns:
            Response with temporal reasoning results
        """
        concepts = data.get("concepts", [])
        sequence_length = data.get("sequence_length", 5)
        use_memory = data.get("use_memory", True)
        
        if not concepts:
            return Response(
                message="No concepts provided for temporal reasoning",
                break_loop=False
            )
        
        try:
            reasoning_results = []
            
            # Perform reasoning over temporal sequence
            for step in range(sequence_length):
                # Create embeddings for current step
                step_embeddings = self.bridge.embed_concepts(concepts)
                
                # Apply temporal attention
                if use_memory and hasattr(self.attention_mechanism, 'temporal'):
                    temporal_output, memory_weights = self.attention_mechanism.temporal(
                        step_embeddings, timestamp=step
                    )
                else:
                    temporal_output, memory_weights = self.attention_mechanism(
                        step_embeddings, attention_mode="temporal"
                    )
                
                # Analyze temporal step
                step_analysis = {
                    "step": step,
                    "output_magnitude": float(torch.norm(temporal_output).mean()) if hasattr(temporal_output, 'norm') else 1.0,
                    "memory_influence": float(memory_weights.mean()) if hasattr(memory_weights, 'mean') else 0.1
                }
                reasoning_results.append(step_analysis)
            
            # Analyze temporal consistency
            temporal_consistency = self._analyze_temporal_consistency(reasoning_results)
            memory_summary = self.attention_mechanism.temporal.get_memory_summary() if hasattr(self.attention_mechanism, 'temporal') else {}
            
            return Response(
                message=f"Temporal reasoning completed over {sequence_length} steps with {len(concepts)} concepts. "
                       f"Temporal consistency: {temporal_consistency:.3f}, "
                       f"Memory utilization: {memory_summary.get('memory_utilization', 0):.3f}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in temporal reasoning: {str(e)}",
                break_loop=False
            )
    
    def _analyze_temporal_consistency(self, reasoning_results: List[dict]) -> float:
        """Analyze consistency across temporal reasoning steps."""
        try:
            if len(reasoning_results) < 2:
                return 1.0
            
            # Measure consistency in output magnitudes
            magnitudes = [result['output_magnitude'] for result in reasoning_results]
            magnitude_std = torch.tensor(magnitudes).std() if len(magnitudes) > 1 else torch.tensor(0.0)
            
            # Consistency is inverse of variation (normalized)
            consistency = 1.0 / (1.0 + float(magnitude_std))
            
            return consistency
        except:
            return 0.5
    
    async def perform_meta_cognitive_analysis(self, data: dict) -> Response:
        """
        Perform meta-cognitive analysis of attention patterns and reasoning processes.
        
        Args:
            data: Dictionary with concepts and analysis parameters
            
        Returns:
            Response with meta-cognitive insights
        """
        concepts = data.get("concepts", [])
        analysis_depth = data.get("depth", "comprehensive")  # basic, detailed, comprehensive
        include_patterns = data.get("include_patterns", True)
        
        if not concepts:
            return Response(
                message="No concepts provided for meta-cognitive analysis",
                break_loop=False
            )
        
        try:
            # Perform multi-modal attention analysis
            embeddings = self.bridge.embed_concepts(concepts)
            
            # Collect attention patterns from different modes
            attention_patterns = {}
            
            # Basic attention
            basic_output, basic_weights = self.attention_mechanism(
                embeddings, attention_mode="basic"
            )
            attention_patterns['basic'] = basic_weights
            
            # Advanced attention modes if available
            if self.attention_mechanism.advanced_available:
                for mode in ["multi_scale", "hierarchical", "temporal"]:
                    try:
                        adv_output, adv_weights = self.attention_mechanism(
                            embeddings, use_advanced=True, attention_mode=mode
                        )
                        attention_patterns[mode] = adv_weights
                    except:
                        pass  # Skip unavailable modes
            
            # Meta-attention analysis
            if hasattr(self.attention_mechanism, 'meta_attention'):
                weights_list = list(attention_patterns.values())
                meta_output, meta_weights = self.attention_mechanism.meta_attention(
                    embeddings, weights_list
                )
                attention_patterns['meta'] = meta_weights
            
            # Comprehensive analysis
            meta_analysis = {
                "concepts_analyzed": len(concepts),
                "attention_modes_used": list(attention_patterns.keys()),
                "pattern_diversity": self._calculate_pattern_diversity(attention_patterns),
                "cognitive_complexity": self._estimate_cognitive_complexity(attention_patterns),
                "attention_coherence": self._measure_overall_coherence(attention_patterns),
                "meta_insights": self._generate_meta_insights(attention_patterns, concepts)
            }
            
            # Generate summary insights
            insights = []
            if meta_analysis['pattern_diversity'] > 0.7:
                insights.append("High pattern diversity indicates rich cognitive processing")
            if meta_analysis['cognitive_complexity'] > 0.6:
                insights.append("Complex attention patterns suggest sophisticated reasoning")
            if meta_analysis['attention_coherence'] > 0.8:
                insights.append("Strong coherence across attention modes")
            
            return Response(
                message=f"Meta-cognitive analysis completed on {len(concepts)} concepts using {len(attention_patterns)} attention modes. "
                       f"Diversity: {meta_analysis['pattern_diversity']:.3f}, "
                       f"Complexity: {meta_analysis['cognitive_complexity']:.3f}, "
                       f"Coherence: {meta_analysis['attention_coherence']:.3f}. "
                       f"Key insights: {'; '.join(insights) if insights else 'Analysis complete'}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Error in meta-cognitive analysis: {str(e)}",
                break_loop=False
            )
    
    def _calculate_pattern_diversity(self, attention_patterns: dict) -> float:
        """Calculate diversity across different attention patterns."""
        try:
            if len(attention_patterns) < 2:
                return 0.0
            
            patterns = list(attention_patterns.values())
            diversity_sum = 0.0
            comparisons = 0
            
            for i in range(len(patterns)):
                for j in range(i + 1, len(patterns)):
                    # Flatten patterns for comparison
                    p1 = patterns[i].flatten() if hasattr(patterns[i], 'flatten') else torch.tensor([0.1])
                    p2 = patterns[j].flatten() if hasattr(patterns[j], 'flatten') else torch.tensor([0.1])
                    
                    # Ensure same size
                    min_size = min(len(p1), len(p2))
                    p1_trunc = p1[:min_size]
                    p2_trunc = p2[:min_size]
                    
                    # Calculate dissimilarity
                    if len(p1_trunc) > 0 and len(p2_trunc) > 0:
                        similarity = float(torch.cosine_similarity(p1_trunc, p2_trunc, dim=0))
                        diversity_sum += (1.0 - similarity)
                        comparisons += 1
            
            return diversity_sum / comparisons if comparisons > 0 else 0.0
        except:
            return 0.5
    
    def _estimate_cognitive_complexity(self, attention_patterns: dict) -> float:
        """Estimate cognitive complexity based on attention patterns."""
        try:
            complexity_factors = []
            
            for mode, pattern in attention_patterns.items():
                if hasattr(pattern, 'std'):
                    # Higher standard deviation suggests more complex patterns
                    std_factor = float(pattern.std())
                    complexity_factors.append(std_factor)
                else:
                    complexity_factors.append(0.1)
            
            # Number of attention modes also contributes to complexity
            mode_complexity = len(attention_patterns) / 5.0  # Normalize assuming max 5 modes
            pattern_complexity = sum(complexity_factors) / len(complexity_factors) if complexity_factors else 0.1
            
            return min(1.0, (mode_complexity + pattern_complexity) / 2.0)
        except:
            return 0.5
    
    def _measure_overall_coherence(self, attention_patterns: dict) -> float:
        """Measure overall coherence across all attention patterns."""
        try:
            # High coherence means patterns are similar/consistent
            # This is the inverse of diversity
            diversity = self._calculate_pattern_diversity(attention_patterns)
            return 1.0 - diversity
        except:
            return 0.5
    
    def _generate_meta_insights(self, attention_patterns: dict, concepts: List[str]) -> List[str]:
        """Generate meta-cognitive insights from attention analysis."""
        insights = []
        
        try:
            # Analyze which modes are most active
            mode_activities = {}
            for mode, pattern in attention_patterns.items():
                if hasattr(pattern, 'sum'):
                    activity = float(pattern.sum())
                    mode_activities[mode] = activity
                else:
                    mode_activities[mode] = 1.0
            
            if mode_activities:
                most_active = max(mode_activities, key=mode_activities.get)
                insights.append(f"Most active attention mode: {most_active}")
            
            # Analyze concept focus patterns
            if len(concepts) > 0:
                insights.append(f"Processing {len(concepts)} concepts with {len(attention_patterns)} attention mechanisms")
            
            # Pattern-specific insights
            if 'hierarchical' in attention_patterns:
                insights.append("Hierarchical processing detected")
            if 'temporal' in attention_patterns:
                insights.append("Temporal dynamics incorporated")
            if 'meta' in attention_patterns:
                insights.append("Meta-attention analysis applied")
                
        except Exception as e:
            insights.append(f"Insight generation encountered: {str(e)}")
        
        return insights


def register():
    """Register the neural-symbolic tool with Agent-Zero."""
    return NeuralSymbolicTool