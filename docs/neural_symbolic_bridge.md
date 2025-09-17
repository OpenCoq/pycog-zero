# Neural-Symbolic Bridge Documentation

## Overview

The Neural-Symbolic Bridge provides seamless integration between PyTorch neural networks and OpenCog symbolic reasoning for the PyCog-Zero cognitive architecture. This implementation enables bidirectional conversion between neural tensor representations and symbolic AtomSpace atoms.

## Components

### 1. NeuralSymbolicBridge Class

The core bridge between neural and symbolic representations.

```python
from python.helpers.neural_symbolic_bridge import NeuralSymbolicBridge

# Initialize bridge
bridge = NeuralSymbolicBridge(embedding_dim=128)

# Convert concepts to neural embeddings
concepts = ["love", "reasoning", "intelligence"]
embeddings = bridge.embed_concepts(concepts)  # Returns PyTorch tensor

# Convert embeddings back to symbolic atoms
atoms = bridge.tensor_to_atomspace(embeddings)
```

**Key Methods:**
- `embed_concepts(concepts)` - Convert concept strings to neural embeddings
- `atomspace_to_tensor(atoms)` - Convert OpenCog atoms to tensors
- `tensor_to_atomspace(tensor)` - Convert tensors back to atoms
- `create_atom_embedding(atom)` - Create embedding for individual atom
- `train_embeddings(atoms, targets)` - Train embedding network

### 2. CognitiveAttentionMechanism Class

Attention mechanism integrating OpenCog ECAN with neural networks.

```python
from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism

# Initialize attention mechanism
attention = CognitiveAttentionMechanism(embedding_dim=128, num_heads=8)

# Apply attention to embeddings
attended_output, attention_weights = attention(embeddings)

# Apply with ECAN importance weights
ecan_weights = torch.tensor([0.5, 1.0, 0.3])  # Importance values
attended_output, weights = attention(embeddings, ecan_weights)
```

**Key Features:**
- Multi-head attention mechanism
- ECAN weight integration
- Layer normalization and residual connections
- Feedforward processing

### 3. NeuralSymbolicTool Agent Tool

Agent-Zero tool providing neural-symbolic operations.

```python
# Available through Agent-Zero tool system
# Operation examples:

# 1. Embed concepts
{
    "operation": "embed_concepts",
    "concepts": ["reasoning", "memory", "attention"]
}

# 2. Neural reasoning with attention
{
    "operation": "neural_reasoning", 
    "query": "What connects memory and reasoning?",
    "concepts": ["memory", "reasoning", "attention", "knowledge"]
}

# 3. Symbolic grounding
{
    "operation": "symbolic_grounding",
    "concepts": ["abstract_thought", "concrete_action"],
    "num_outputs": 5
}
```

**Available Operations:**
- `embed_concepts` - Convert concepts to embeddings
- `neural_reasoning` - Apply attention-based reasoning
- `symbolic_grounding` - Ground neural outputs in symbols
- `bridge_tensors` - Bidirectional tensor/atom conversion
- `train_embeddings` - Train the embedding network
- `analyze_attention` - Analyze attention patterns

## Architecture Details

### Embedding Strategy

The bridge uses a multi-layer approach for creating atom embeddings:

1. **Type-based embedding** - Base vector from atom type
2. **Content-based embedding** - Modified by atom name/content using deterministic hashing
3. **Truth value embedding** - OpenCog truth values integrated when available

### Attention Integration

The attention mechanism combines neural attention with OpenCog's ECAN system:

- **STI/LTI Integration** - OpenCog attention values scale neural attention
- **Multi-head Processing** - Parallel attention computations
- **Residual Connections** - Preserve original information flow

### Graceful Fallbacks

The implementation gracefully handles missing dependencies:

- **PyTorch unavailable** - Mock tensor operations with basic functionality
- **OpenCog unavailable** - Mock AtomSpace with basic node creation
- **NumPy unavailable** - Python list-based fallbacks

## Usage Examples

### Basic Concept Embedding

```python
from python.helpers.neural_symbolic_bridge import NeuralSymbolicBridge

# Initialize bridge
bridge = NeuralSymbolicBridge(embedding_dim=64)

# Embed concepts
concepts = ["love", "fear", "joy", "anger"]
embeddings = bridge.embed_concepts(concepts)

print(f"Embedded {len(concepts)} concepts")
print(f"Embedding shape: {embeddings.shape}")
```

### Neural-Symbolic Reasoning

```python
# Initialize components
bridge = NeuralSymbolicBridge(embedding_dim=128)
attention = CognitiveAttentionMechanism(embedding_dim=128, num_heads=4)

# Create embeddings for reasoning
concepts = ["problem", "solution", "knowledge", "experience"]
embeddings = bridge.embed_concepts(concepts)

# Apply attention-based reasoning
reasoning_output, attention_weights = attention(embeddings)

# Analyze attention patterns
max_attention = torch.argmax(attention_weights.mean(dim=0))
focused_concept = concepts[max_attention]
print(f"Reasoning focused on: {focused_concept}")
```

### Training Custom Embeddings

```python
# Create training data
concepts = ["happy", "sad", "excited", "calm"]
embeddings = bridge.embed_concepts(concepts)

# Create target embeddings (e.g., from sentiment analysis)
target_embeddings = create_sentiment_embeddings(concepts)

# Train the embedding network
bridge.train_embeddings(
    atoms=bridge.atomspace.get_atoms_by_name(concepts),
    target_embeddings=target_embeddings,
    epochs=100,
    learning_rate=0.001
)
```

## Integration with Agent-Zero

The neural-symbolic bridge integrates seamlessly with Agent-Zero through the tool system:

### Tool Registration

The `neural_symbolic_agent.py` tool is automatically discovered by Agent-Zero when placed in the `python/tools/` directory.

### Usage in Agent Context

```python
# Agent can use neural-symbolic reasoning
response = await agent.use_tool(
    "neural_symbolic_agent",
    operation="neural_reasoning",
    concepts=["problem_solving", "creativity", "logic"],
    query="How do creativity and logic interact?"
)
```

## Performance Considerations

- **Embedding Caching** - Atom embeddings are cached for efficiency
- **Batch Processing** - Multiple concepts processed together
- **Memory Management** - Large embedding caches can be cleared
- **Training Efficiency** - Embedding network can be trained incrementally

## Dependencies

### Required (with fallbacks):
- **PyTorch** - Neural network operations (mock implementation available)
- **OpenCog** - Symbolic reasoning (mock AtomSpace available)  
- **NumPy** - Numerical operations (Python fallback available)

### Optional:
- **SciPy** - Advanced numerical operations
- **Matplotlib** - Visualization of attention patterns
- **NetworkX** - Graph-based reasoning visualization

## Error Handling

The implementation includes comprehensive error handling:

- **Graceful Dependency Failures** - Mock implementations when packages unavailable
- **Invalid Input Validation** - Proper error messages for malformed inputs
- **Memory Management** - Automatic cleanup of large embedding caches
- **Training Stability** - Gradient clipping and learning rate scheduling

## Future Extensions

The neural-symbolic bridge is designed for extensibility:

- **Pre-trained Embeddings** - Integration with Word2Vec, GloVe, BERT
- **Advanced Attention** - Transformer-style attention mechanisms
- **Symbolic Reasoning** - PLN (Probabilistic Logic Networks) integration
- **Multi-modal** - Support for visual and audio symbolic representations

## Testing

Basic functionality can be tested even without dependencies:

```bash
# Run basic integration tests
python3 /tmp/test_neural_symbolic_basic.py

# Run full tests (requires PyTorch)
python3 -m pytest tests/test_neural_symbolic_bridge.py -v
```

## Troubleshooting

### Common Issues

1. **Import Errors** - Missing dependencies will use mock implementations
2. **Memory Issues** - Large embedding caches can be cleared with `bridge.clear_cache()`
3. **Training Convergence** - Adjust learning rate and check target embeddings
4. **OpenCog Integration** - Verify OpenCog installation for full symbolic functionality

### Debug Mode

Enable debug logging in Agent-Zero to see neural-symbolic processing details:

```python
# Enable debug output
bridge = NeuralSymbolicBridge(embedding_dim=128, debug=True)
```

This documentation provides comprehensive guidance for using the neural-symbolic bridge in the PyCog-Zero cognitive architecture.