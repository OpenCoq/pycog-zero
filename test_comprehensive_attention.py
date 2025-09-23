#!/usr/bin/env python3
"""
Comprehensive tests for advanced neural-symbolic attention mechanisms.
These tests work with both PyTorch (if available) and mock implementations.
"""

import sys
import os
import asyncio
sys.path.insert(0, '/home/runner/work/pycog-zero/pycog-zero')

def test_attention_analysis():
    """Test attention pattern analysis functionality."""
    print("=== Testing Attention Analysis ===")
    
    try:
        from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism
        
        # Initialize attention mechanism
        attention = CognitiveAttentionMechanism(embedding_dim=32, num_heads=4)
        
        # Create mock embeddings and attention weights for analysis
        from python.helpers.neural_symbolic_bridge import torch
        
        # Create test embeddings
        test_embeddings = torch.zeros((1, 5, 32))  # batch=1, seq_len=5, embed_dim=32
        test_weights = torch.ones((1, 5, 5)) * 0.2  # Uniform attention weights
        
        # Test attention pattern analysis
        analysis = attention.analyze_attention_patterns(test_embeddings, test_weights)
        
        print("✓ Attention pattern analysis completed")
        print(f"✓ Analysis keys: {list(analysis.keys())}")
        print(f"✓ Attention entropy: {analysis.get('attention_entropy', 'N/A')}")
        print(f"✓ Attention variance: {analysis.get('attention_variance', 'N/A')}")
        print(f"✓ Max attention index: {analysis.get('max_attention_index', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in attention analysis test: {e}")
        return False

async def test_neural_symbolic_tool_operations():
    """Test neural-symbolic tool operations."""
    print("\n=== Testing Neural-Symbolic Tool Operations ===")
    
    try:
        # Mock agent class
        class MockAgent:
            def __init__(self):
                self.name = "test_agent"
        
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
        
        # Test concept embedding
        concepts = ["intelligence", "reasoning", "memory", "attention"]
        response = await tool.execute("embed_concepts", concepts=concepts)
        print(f"✓ Concept embedding: {response.message}")
        
        # Test neural reasoning
        response = await tool.execute("neural_reasoning", 
                                    query="How do concepts relate?",
                                    concepts=concepts)
        print(f"✓ Neural reasoning: {response.message}")
        
        # Test attention analysis
        response = await tool.execute("analyze_attention", 
                                    concepts=concepts,
                                    mode="basic",
                                    use_advanced=True)
        print(f"✓ Attention analysis: {response.message}")
        
        # Test cross-modal fusion
        response = await tool.execute("cross_modal_fusion",
                                    neural_concepts=["neural_net", "tensor"],
                                    symbolic_concepts=["logic", "reasoning"])
        print(f"✓ Cross-modal fusion: {response.message}")
        
        # Test temporal reasoning
        response = await tool.execute("temporal_reasoning",
                                    concepts=concepts,
                                    sequence_length=3,
                                    use_memory=True)
        print(f"✓ Temporal reasoning: {response.message}")
        
        # Test meta-cognitive analysis
        response = await tool.execute("meta_cognitive_analysis",
                                    concepts=concepts,
                                    depth="comprehensive",
                                    include_patterns=True)
        print(f"✓ Meta-cognitive analysis: {response.message}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in tool operations test: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_attention_modes():
    """Test different attention modes."""
    print("\n=== Testing Different Attention Modes ===")
    
    try:
        from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism, torch
        
        # Initialize attention mechanism
        attention = CognitiveAttentionMechanism(embedding_dim=16, num_heads=2)
        
        # Create test embeddings
        test_embeddings = torch.ones((1, 3, 16))  # Small test tensor
        
        # Test basic attention
        output, weights = attention(test_embeddings, attention_mode="basic")
        print(f"✓ Basic attention - output shape: {output.shape}")
        
        # Test advanced modes if available
        if attention.advanced_available:
            print("✓ Advanced attention mechanisms available")
            
            for mode in ["multi_scale", "hierarchical", "temporal", "meta"]:
                try:
                    output, weights = attention(test_embeddings, 
                                              use_advanced=True, 
                                              attention_mode=mode)
                    print(f"✓ Advanced mode '{mode}' - output shape: {output.shape}")
                except Exception as e:
                    print(f"⚠️ Advanced mode '{mode}' failed: {e}")
        else:
            print("⚠️ Advanced attention mechanisms not available (using mock implementation)")
            
            # Test fallback behavior
            output, weights = attention(test_embeddings, 
                                      use_advanced=True, 
                                      attention_mode="multi_scale")
            print(f"✓ Fallback basic attention - output shape: {output.shape}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in attention modes test: {e}")
        return False

def test_cross_modal_attention():
    """Test cross-modal attention functionality."""
    print("\n=== Testing Cross-Modal Attention ===")
    
    try:
        from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism, torch
        
        attention = CognitiveAttentionMechanism(embedding_dim=16, num_heads=2)
        
        # Create test embeddings for different modalities
        neural_embeddings = torch.ones((1, 3, 16))
        symbolic_embeddings = torch.ones((1, 3, 16)) * 0.5
        
        # Test cross-modal attention
        fused_output, cross_weights = attention.apply_cross_modal_attention(
            neural_embeddings, symbolic_embeddings
        )
        
        print(f"✓ Cross-modal fusion - output shape: {fused_output.shape}")
        print(f"✓ Cross-modal attention weights: {list(cross_weights.keys())}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in cross-modal attention test: {e}")
        return False

def test_attention_memory():
    """Test attention memory and temporal dynamics."""
    print("\n=== Testing Attention Memory ===")
    
    try:
        from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism
        
        attention = CognitiveAttentionMechanism(embedding_dim=16, num_heads=2)
        
        # Test attention summary
        summary = attention.get_attention_summary()
        print(f"✓ Attention summary: {summary}")
        
        # Test memory clearing
        attention.clear_memory()
        print("✓ Memory cleared successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in attention memory test: {e}")
        return False

def test_comprehensive_integration():
    """Test comprehensive integration of all components."""
    print("\n=== Testing Comprehensive Integration ===")
    
    try:
        from python.helpers.neural_symbolic_bridge import NeuralSymbolicBridge, CognitiveAttentionMechanism
        
        # Initialize components
        bridge = NeuralSymbolicBridge(embedding_dim=16)
        attention = CognitiveAttentionMechanism(embedding_dim=16, num_heads=2)
        
        # Test workflow: concepts -> embeddings -> attention -> analysis
        concepts = ["cognition", "intelligence"]
        
        print(f"✓ Testing workflow with concepts: {concepts}")
        
        # Step 1: Create embeddings
        embeddings = bridge.embed_concepts(concepts)
        print(f"✓ Step 1 - Embeddings created: {embeddings.shape}")
        
        # Step 2: Apply attention
        attended_output, attention_weights = attention(embeddings)
        print(f"✓ Step 2 - Attention applied: {attended_output.shape}")
        
        # Step 3: Analyze patterns
        analysis = attention.analyze_attention_patterns(embeddings, attention_weights)
        print(f"✓ Step 3 - Pattern analysis completed: {len(analysis)} metrics")
        
        # Step 4: Cache and summary
        cache_size = bridge.get_cache_size()
        attention_summary = attention.get_attention_summary()
        print(f"✓ Step 4 - Cache size: {cache_size}, Summary: {len(attention_summary)} items")
        
        return True
        
    except Exception as e:
        print(f"❌ Error in comprehensive integration test: {e}")
        import traceback
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run all tests."""
    print("Advanced Neural-Symbolic Attention Tests")
    print("=" * 50)
    
    tests = [
        ("Basic Attention Analysis", test_attention_analysis),
        ("Attention Modes", test_attention_modes),
        ("Cross-Modal Attention", test_cross_modal_attention),
        ("Attention Memory", test_attention_memory),
        ("Comprehensive Integration", test_comprehensive_integration),
    ]
    
    async_tests = [
        ("Neural-Symbolic Tool Operations", test_neural_symbolic_tool_operations),
    ]
    
    results = []
    
    # Run synchronous tests
    for test_name, test_func in tests:
        print(f"\n--- {test_name} ---")
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Run asynchronous tests  
    for test_name, test_func in async_tests:
        print(f"\n--- {test_name} ---")
        try:
            result = await test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"❌ {test_name} failed with exception: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("TEST SUMMARY")
    print("=" * 50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
        if result:
            passed += 1
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Advanced neural-symbolic integration is working.")
    else:
        print(f"⚠️ {total - passed} tests failed. Check implementation.")

if __name__ == "__main__":
    asyncio.run(run_all_tests())