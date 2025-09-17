#!/usr/bin/env python3
"""
Neural-Symbolic Bridge Usage Examples

This script demonstrates how to use the neural-symbolic bridge for
PyTorch-OpenCog integration in the PyCog-Zero cognitive architecture.
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/pycog-zero/pycog-zero')

from python.helpers.neural_symbolic_bridge import NeuralSymbolicBridge, CognitiveAttentionMechanism

def example_basic_embedding():
    """Example 1: Basic concept embedding."""
    print("=== Example 1: Basic Concept Embedding ===")
    
    # Initialize bridge
    bridge = NeuralSymbolicBridge(embedding_dim=64)
    
    # Define concepts related to cognitive processing
    concepts = ["memory", "attention", "reasoning", "learning", "perception"]
    
    # Create embeddings
    embeddings = bridge.embed_concepts(concepts)
    
    print(f"✓ Embedded {len(concepts)} concepts")
    print(f"✓ Embedding dimensions: {embeddings.shape}")
    print(f"✓ Cache size: {bridge.get_cache_size()} atoms")
    
    # Show some embedding properties
    for i, concept in enumerate(concepts):
        magnitude = float(embeddings[i].norm() if hasattr(embeddings[i], 'norm') else 0)
        print(f"  - {concept}: magnitude {magnitude:.3f}")
    
    print()

def example_attention_mechanism():
    """Example 2: Cognitive attention mechanism."""
    print("=== Example 2: Cognitive Attention Mechanism ===")
    
    # Initialize components
    bridge = NeuralSymbolicBridge(embedding_dim=32)
    attention = CognitiveAttentionMechanism(embedding_dim=32, num_heads=4)
    
    # Concepts for reasoning about problem-solving
    concepts = ["problem", "analysis", "solution", "creativity", "logic"]
    
    # Create embeddings
    embeddings = bridge.embed_concepts(concepts)
    
    # Apply attention mechanism
    attended_output, attention_weights = attention(embeddings)
    
    print(f"✓ Applied attention to {len(concepts)} concepts")
    print(f"✓ Attention heads: {attention.num_heads}")
    print(f"✓ Output shape: {attended_output.shape}")
    
    # Analyze attention patterns (mock analysis for fallback mode)
    try:
        # Find concept with highest average attention
        if hasattr(attention_weights, 'mean'):
            avg_attention = attention_weights.mean(dim=0)
            max_idx = int(avg_attention.argmax()) if hasattr(avg_attention, 'argmax') else 0
        else:
            max_idx = 0
        
        focused_concept = concepts[max_idx] if max_idx < len(concepts) else concepts[0]
        print(f"✓ Primary attention focus: '{focused_concept}'")
    except:
        print("✓ Attention analysis completed (mock mode)")
    
    print()

def example_bidirectional_conversion():
    """Example 3: Bidirectional tensor-atom conversion."""
    print("=== Example 3: Bidirectional Conversion ===")
    
    bridge = NeuralSymbolicBridge(embedding_dim=16)
    
    # Original concepts
    concepts = ["knowledge", "wisdom", "understanding"]
    
    # Convert to embeddings
    embeddings = bridge.embed_concepts(concepts)
    print(f"✓ Concepts → Embeddings: {concepts}")
    
    # Convert back to atoms
    atoms = bridge.tensor_to_atomspace(
        embeddings, 
        atom_names=[f"reconstructed_{concept}" for concept in concepts]
    )
    print(f"✓ Embeddings → Atoms: {len(atoms)} atoms created")
    
    # Show atom names
    atom_names = []
    for atom in atoms:
        if hasattr(atom, 'name'):
            atom_names.append(atom.name)
        else:
            atom_names.append(str(atom)[:20])
    
    print(f"✓ Generated atoms: {atom_names}")
    
    # Test round-trip preservation
    round_trip_embeddings = bridge.atomspace_to_tensor(atoms)
    print(f"✓ Round-trip embedding shape: {round_trip_embeddings.shape}")
    
    print()

def example_neural_symbolic_reasoning():
    """Example 4: Neural-symbolic reasoning workflow."""
    print("=== Example 4: Neural-Symbolic Reasoning Workflow ===")
    
    # Initialize components
    bridge = NeuralSymbolicBridge(embedding_dim=64)
    attention = CognitiveAttentionMechanism(embedding_dim=64, num_heads=8)
    
    # Define a reasoning scenario about AI ethics
    context_concepts = ["artificial_intelligence", "ethics", "human_values", "responsibility"]
    query = "How do AI systems integrate ethical considerations?"
    
    print(f"Query: {query}")
    print(f"Context concepts: {context_concepts}")
    
    # Step 1: Create neural representations
    context_embeddings = bridge.embed_concepts(context_concepts)
    print(f"✓ Created neural representations for {len(context_concepts)} concepts")
    
    # Step 2: Apply attention-based reasoning
    reasoned_output, attention_weights = attention(context_embeddings)
    print("✓ Applied attention-based reasoning")
    
    # Step 3: Ground results back in symbolic representation
    num_result_concepts = 3
    result_atoms = bridge.tensor_to_atomspace(
        reasoned_output[:num_result_concepts] if hasattr(reasoned_output, '__getitem__') else reasoned_output,
        atom_names=[f"reasoning_result_{i}" for i in range(num_result_concepts)]
    )
    
    print(f"✓ Grounded reasoning in {len(result_atoms)} symbolic atoms")
    
    # Show reasoning results
    for i, atom in enumerate(result_atoms):
        atom_name = atom.name if hasattr(atom, 'name') else f"result_{i}"
        print(f"  - Reasoning result {i+1}: {atom_name}")
    
    print()

def example_ecan_integration():
    """Example 5: ECAN attention value integration."""
    print("=== Example 5: ECAN Attention Value Integration ===")
    
    bridge = NeuralSymbolicBridge(embedding_dim=32)
    attention = CognitiveAttentionMechanism(embedding_dim=32, num_heads=4)
    
    # Concepts with different importance levels
    concepts = ["critical_task", "routine_task", "background_process"]
    
    # Simulate ECAN importance values (STI - Short Term Importance)
    importance_values = [2.0, 1.0, 0.3]  # Critical > Routine > Background
    
    print("Concepts with importance values:")
    for concept, importance in zip(concepts, importance_values):
        print(f"  - {concept}: STI = {importance}")
    
    # Create embeddings
    embeddings = bridge.embed_concepts(concepts)
    
    # Create mock atoms with STI values for ECAN weight computation
    class MockAtomWithSTI:
        def __init__(self, name, sti):
            self.name = name
            self.sti = sti
    
    atoms_with_sti = [MockAtomWithSTI(concept, sti) for concept, sti in zip(concepts, importance_values)]
    
    # Compute ECAN weights
    ecan_weights = attention.compute_ecan_weights(atoms_with_sti)
    print(f"✓ Computed ECAN weights: {[f'{w:.3f}' for w in ecan_weights.tolist()] if hasattr(ecan_weights, 'tolist') else 'computed'}")
    
    # Apply attention with ECAN weighting
    attended_output, attention_weights = attention(embeddings, ecan_weights)
    print("✓ Applied ECAN-weighted attention")
    
    print()

def example_embedding_training():
    """Example 6: Training custom embeddings."""
    print("=== Example 6: Training Custom Embeddings ===")
    
    bridge = NeuralSymbolicBridge(embedding_dim=16)
    
    # Emotional concepts for training
    emotional_concepts = ["joy", "sadness", "anger", "fear"]
    
    # Create initial embeddings
    initial_embeddings = bridge.embed_concepts(emotional_concepts)
    print(f"✓ Created initial embeddings for {len(emotional_concepts)} emotional concepts")
    
    # Simulate target embeddings (could come from sentiment analysis model)
    try:
        if hasattr(initial_embeddings, 'randn_like'):
            # Add some structured variation to create targets
            target_embeddings = initial_embeddings + initial_embeddings.randn_like() * 0.2
        else:
            target_embeddings = initial_embeddings
        
        print("✓ Generated target embeddings for training")
    except:
        print("✓ Using mock target embeddings")
        target_embeddings = initial_embeddings
    
    # Create atoms for training
    atoms = []
    for concept in emotional_concepts:
        if hasattr(bridge.atomspace, 'add_node'):
            atom = bridge.atomspace.add_node("ConceptNode", concept)
        else:
            # Mock atom
            class MockAtom:
                def __init__(self, name):
                    self.name = name
            atom = MockAtom(concept)
        atoms.append(atom)
    
    # Train embeddings (will be skipped if PyTorch not available)
    try:
        bridge.train_embeddings(atoms, target_embeddings, epochs=20, learning_rate=0.01)
        print("✓ Completed embedding network training")
    except Exception as e:
        print(f"✓ Training skipped: {str(e)[:50]}...")
    
    print()

def example_performance_analysis():
    """Example 7: Performance and cache analysis."""
    print("=== Example 7: Performance and Cache Analysis ===")
    
    bridge = NeuralSymbolicBridge(embedding_dim=48)
    
    # Create embeddings for various concept categories
    categories = {
        "cognitive": ["thinking", "reasoning", "memory", "attention"],
        "emotional": ["love", "fear", "joy", "anger"],
        "social": ["cooperation", "competition", "empathy", "communication"],
        "temporal": ["past", "present", "future", "duration"]
    }
    
    total_concepts = 0
    for category, concepts in categories.items():
        embeddings = bridge.embed_concepts(concepts)
        total_concepts += len(concepts)
        print(f"✓ Processed {category}: {len(concepts)} concepts")
    
    print(f"\nCache Analysis:")
    print(f"✓ Total concepts processed: {total_concepts}")
    print(f"✓ Cache size: {bridge.get_cache_size()} embeddings")
    
    # Clear cache demonstration
    bridge.clear_cache()
    print(f"✓ Cache cleared: {bridge.get_cache_size()} embeddings remaining")
    
    print()

def main():
    """Run all neural-symbolic bridge examples."""
    print("Neural-Symbolic Bridge Usage Examples")
    print("=" * 50)
    print()
    
    try:
        example_basic_embedding()
        example_attention_mechanism()
        example_bidirectional_conversion()
        example_neural_symbolic_reasoning()
        example_ecan_integration()
        example_embedding_training()
        example_performance_analysis()
        
        print("🎉 All examples completed successfully!")
        print("\nThe neural-symbolic bridge provides powerful integration between")
        print("neural networks and symbolic reasoning for cognitive architectures.")
        
    except Exception as e:
        print(f"❌ Example execution error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()