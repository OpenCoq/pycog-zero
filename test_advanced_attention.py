#!/usr/bin/env python3
"""
Test advanced neural-symbolic attention mechanisms.
"""

import sys
import os
sys.path.insert(0, '/home/runner/work/pycog-zero/pycog-zero')

def test_basic_attention():
    """Test basic attention mechanism."""
    print("=== Testing Basic Attention Mechanism ===")
    
    try:
        from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism
        
        # Initialize attention mechanism
        attention = CognitiveAttentionMechanism(embedding_dim=64, num_heads=4)
        
        print("✓ Basic attention mechanism initialized")
        print(f"✓ Embedding dimension: {attention.embedding_dim}")
        print(f"✓ Number of heads: {attention.num_heads}")
        print(f"✓ Advanced attention available: {attention.advanced_available}")
        
        # Test attention summary
        summary = attention.get_attention_summary()
        print(f"✓ Attention summary: {summary}")
        
    except Exception as e:
        print(f"❌ Error in basic attention test: {e}")
        import traceback
        traceback.print_exc()

def test_advanced_attention():
    """Test advanced attention mechanisms."""
    print("\n=== Testing Advanced Attention Mechanisms ===")
    
    try:
        from python.helpers.advanced_attention import (
            MultiScaleAttention, HierarchicalAttention, 
            CrossModalAttention, TemporalAttention, MetaAttention
        )
        
        # Test MultiScaleAttention
        multi_scale = MultiScaleAttention(embedding_dim=32)
        print("✓ MultiScaleAttention initialized")
        
        # Test HierarchicalAttention
        hierarchical = HierarchicalAttention(embedding_dim=32)
        print("✓ HierarchicalAttention initialized")
        
        # Test CrossModalAttention
        cross_modal = CrossModalAttention(embedding_dim=32)
        print("✓ CrossModalAttention initialized")
        
        # Test TemporalAttention
        temporal = TemporalAttention(embedding_dim=32)
        print("✓ TemporalAttention initialized")
        print(f"✓ Temporal memory summary: {temporal.get_memory_summary()}")
        
        # Test MetaAttention
        meta = MetaAttention(embedding_dim=32)
        print("✓ MetaAttention initialized")
        
    except Exception as e:
        print(f"❌ Error in advanced attention test: {e}")
        import traceback
        traceback.print_exc()

def test_neural_symbolic_integration():
    """Test neural-symbolic integration with advanced attention."""
    print("\n=== Testing Neural-Symbolic Integration ===")
    
    try:
        from python.helpers.neural_symbolic_bridge import NeuralSymbolicBridge, CognitiveAttentionMechanism
        
        # Initialize components
        bridge = NeuralSymbolicBridge(embedding_dim=32)
        attention = CognitiveAttentionMechanism(embedding_dim=32, num_heads=4)
        
        print("✓ Neural-symbolic bridge and attention initialized")
        
        # Test concept embedding
        concepts = ["reasoning", "memory", "attention"]
        embeddings = bridge.embed_concepts(concepts)
        print(f"✓ Created embeddings for {len(concepts)} concepts")
        print(f"✓ Embedding shape: {embeddings.shape}")
        
        # Test basic attention
        attended_output, attention_weights = attention(embeddings)
        print(f"✓ Applied attention, output shape: {attended_output.shape}")
        
        # Test advanced attention if available
        if attention.advanced_available:
            # Test different attention modes
            for mode in ["multi_scale", "hierarchical", "temporal"]:
                try:
                    adv_output, adv_weights = attention(
                        embeddings, use_advanced=True, attention_mode=mode
                    )
                    print(f"✓ Advanced attention mode '{mode}' working")
                except Exception as e:
                    print(f"⚠️ Advanced attention mode '{mode}' failed: {e}")
        
        # Test attention pattern analysis
        analysis = attention.analyze_attention_patterns(embeddings, attention_weights)
        print(f"✓ Attention analysis completed: {list(analysis.keys())}")
        
    except Exception as e:
        print(f"❌ Error in neural-symbolic integration test: {e}")
        import traceback
        traceback.print_exc()

def test_neural_symbolic_tool():
    """Test the neural-symbolic agent tool."""
    print("\n=== Testing Neural-Symbolic Agent Tool ===")
    
    try:
        # Mock agent class for testing
        class MockAgent:
            def __init__(self):
                self.name = "test_agent"
        
        # Import the tool
        from python.tools.neural_symbolic_agent import NeuralSymbolicTool
        
        # Initialize tool
        agent = MockAgent()
        tool = NeuralSymbolicTool(
            agent=agent,
            name="neural_symbolic_agent",
            method=None,
            args={},
            message="test",
            loop_data={}
        )
        
        print("✓ Neural-symbolic tool initialized")
        print(f"✓ Tool embedding dimension: {tool.embedding_dim}")
        print(f"✓ Tool components initialized: {tool._initialized}")
        
        # The tool will auto-initialize on first use
        tool._initialize_components()
        print("✓ Tool components initialized successfully")
        
    except Exception as e:
        print(f"❌ Error in neural-symbolic tool test: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("Advanced Neural-Symbolic Integration Tests")
    print("=" * 50)
    
    test_basic_attention()
    test_advanced_attention() 
    test_neural_symbolic_integration()
    test_neural_symbolic_tool()
    
    print("\n" + "=" * 50)
    print("✓ All tests completed!")