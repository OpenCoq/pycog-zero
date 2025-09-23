# Attention-Based Reasoning in PyCog-Zero

## Overview

This document provides comprehensive examples and patterns for using attention-based reasoning in the PyCog-Zero cognitive architecture. The attention system integrates OpenCog's Economic Cognitive Attention Networks (ECAN) with neural attention mechanisms, enabling sophisticated cognitive focus and reasoning capabilities.

## Key Concepts

### 1. Economic Cognitive Attention Networks (ECAN)

ECAN provides a biologically-inspired attention allocation system where cognitive resources are distributed based on importance and relevance.

**Core Components:**
- **STI (Short Term Importance)** - Immediate attention values
- **LTI (Long Term Importance)** - Persistent significance values  
- **VLTI (Very Long Term Importance)** - Historical importance patterns
- **Attention Bank** - Resource allocation management
- **Spreading Dynamics** - Attention propagation through concept networks

### 2. Neural Attention Mechanisms

Multi-head attention mechanisms that complement ECAN with neural processing capabilities.

**Architecture Features:**
- Multi-head self-attention with configurable heads
- Cross-attention between different concept domains
- Residual connections and layer normalization
- Feedforward processing with attention gates

## Configuration Reference

The attention system is configured through `conf/config_cognitive.json`:

```json
{
  "attention_config": {
    "ecan_enabled": true,
    "attention_mechanisms": {
      "multi_head_attention": {
        "enabled": true,
        "num_heads": 8,
        "dropout": 0.1,
        "bias": true
      },
      "self_attention": {
        "enabled": true,
        "temperature": 1.0,
        "normalization": "softmax"
      },
      "cross_attention": {
        "enabled": true,
        "query_key_same": false
      }
    },
    "ecan_config": {
      "attention_allocation_cycle": true,
      "forgetting_enabled": true,
      "spreading_enabled": true,
      "sti_decay_factor": 0.1,
      "lti_decay_factor": 0.01,
      "sti_threshold": 0.5,
      "lti_threshold": 0.3,
      "max_spread_percentage": 0.6,
      "hebbian_learning": true,
      "importance_diffusion": {
        "enabled": true,
        "diffusion_factor": 0.2,
        "max_neighbors": 10
      }
    }
  }
}
```

## Basic Attention-Based Reasoning Examples

### Example 1: Single-Concept Attention Focus

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool
from python.helpers.neural_symbolic_bridge import NeuralSymbolicBridge, CognitiveAttentionMechanism

# Initialize components
agent = None  # Your Agent-Zero instance
tool = CognitiveReasoningTool(agent, 'cognitive_reasoning', None, {}, '', None)
bridge = NeuralSymbolicBridge(embedding_dim=128)
attention = CognitiveAttentionMechanism(embedding_dim=128, num_heads=4)

# Basic attention-focused reasoning
response = await tool.execute(
    "What is the relationship between learning and memory?",
    operation="reason",
    attention_focus=["learning", "memory", "neural_networks", "cognition"]
)

print("Attention-focused reasoning result:")
print(response.message)
```

**Expected Output:**
```
Enhanced cognitive reasoning completed for: What is the relationship between learning and memory?
Data: {
  "query": "What is the relationship between learning and memory?",
  "operation": "reason",
  "atoms_created": 4,
  "reasoning_steps": ["Created concept atoms", "Applied attention mechanisms", "Established relationships"],
  "attention_analysis": {
    "primary_focus": "learning",
    "secondary_focus": "memory", 
    "attention_weights": [0.4, 0.35, 0.15, 0.1],
    "cross_references": 2
  }
}
```

### Example 2: Multi-Head Attention Analysis

```python
import torch
from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism

# Initialize attention mechanism with multiple heads
attention = CognitiveAttentionMechanism(embedding_dim=128, num_heads=8)

# Create concept embeddings for attention analysis
concepts = ["problem_solving", "creativity", "logic", "intuition", "experience"]
bridge = NeuralSymbolicBridge(embedding_dim=128)
embeddings = bridge.embed_concepts(concepts)

# Apply multi-head attention
attended_output, attention_weights = attention.forward(embeddings.unsqueeze(0))

# Analyze attention patterns across heads
print("Multi-Head Attention Analysis:")
for head in range(8):
    head_weights = attention_weights[0, head, :, :]  # [seq_len, seq_len]
    
    # Find the most attended concept for each query position
    max_attention_indices = torch.argmax(head_weights, dim=-1)
    
    print(f"\\nHead {head + 1} attention patterns:")
    for i, concept in enumerate(concepts):
        most_attended = concepts[max_attention_indices[i]]
        attention_score = head_weights[i, max_attention_indices[i]].item()
        print(f"  {concept} → {most_attended} (weight: {attention_score:.3f})")
```

### Example 3: ECAN-Weighted Attention Reasoning

```python
import torch
from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism, NeuralSymbolicBridge

# Setup components
bridge = NeuralSymbolicBridge(embedding_dim=128)
attention = CognitiveAttentionMechanism(embedding_dim=128, num_heads=4)

# Define concepts with varying importance levels
concepts = ["urgent_task", "routine_work", "long_term_goal", "distraction"]
concept_embeddings = bridge.embed_concepts(concepts)

# Simulate ECAN importance values (STI scores)
ecan_weights = torch.tensor([0.9, 0.3, 0.7, 0.1])  # High, Low, Medium, Very Low

# Apply ECAN-weighted attention
attended_output, attention_weights = attention.forward(
    concept_embeddings.unsqueeze(0), 
    ecan_weights=ecan_weights.unsqueeze(0)
)

# Analyze how ECAN weights influence attention
print("ECAN-Weighted Attention Analysis:")
print("Concept\\t\\tECAN Weight\\tMean Attention")
print("-" * 45)

mean_attention = attention_weights.mean(dim=1).squeeze()  # Average across heads
for i, concept in enumerate(concepts):
    ecan_score = ecan_weights[i].item()
    attn_score = mean_attention[i].item() if mean_attention.dim() > 0 else mean_attention.item()
    print(f"{concept:<15}\\t{ecan_score:.2f}\\t\\t{attn_score:.3f}")
```

## Advanced Attention Patterns

### Example 4: Cross-Modal Attention Integration

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool

# Initialize cognitive reasoning tool
tool = CognitiveReasoningTool(agent, 'cognitive_reasoning', None, {}, '', None)

# Cross-modal reasoning with attention
response = await tool.execute(
    "How do visual patterns relate to logical reasoning?",
    operation="analyze_patterns",
    cross_modal_concepts=["visual_processing", "pattern_recognition", "logical_inference"],
    attention_mode="cross_attention"
)

# Parse the attention analysis
import json
data = json.loads(response.message.split("Data: ")[1])
attention_analysis = data.get("attention_analysis", {})

print("Cross-Modal Attention Results:")
print(f"Primary modality focus: {attention_analysis.get('primary_modality')}")
print(f"Cross-modal connections: {attention_analysis.get('cross_connections')}")
print(f"Integration strength: {attention_analysis.get('integration_score')}")
```

### Example 5: Temporal Attention Dynamics

```python
from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism
import torch

# Setup for temporal reasoning
attention = CognitiveAttentionMechanism(embedding_dim=128, num_heads=6)
bridge = NeuralSymbolicBridge(embedding_dim=128)

# Temporal sequence of concepts
temporal_concepts = [
    ["past_experience", "historical_data", "learned_patterns"],
    ["current_situation", "immediate_context", "present_state"],
    ["future_goals", "predicted_outcomes", "planned_actions"]
]

print("Temporal Attention Dynamics Analysis:")
print("=" * 50)

for t, time_concepts in enumerate(temporal_concepts):
    time_labels = ["Past", "Present", "Future"]
    print(f"\\n{time_labels[t]} Context:")
    
    # Create embeddings for current time step
    embeddings = bridge.embed_concepts(time_concepts)
    
    # Apply attention
    attended_output, attention_weights = attention.forward(embeddings.unsqueeze(0))
    
    # Analyze attention within this temporal context
    mean_weights = attention_weights.mean(dim=1).squeeze()  # Average across heads
    
    for i, concept in enumerate(time_concepts):
        weight = mean_weights[i].item() if mean_weights.dim() > 0 else mean_weights.item()
        print(f"  {concept}: {weight:.3f}")
    
    # Calculate temporal focus strength
    focus_entropy = -torch.sum(mean_weights * torch.log(mean_weights + 1e-8)).item()
    print(f"  Focus Entropy: {focus_entropy:.3f} (lower = more focused)")
```

## Attention Visualization Examples

### Example 6: Attention Heatmap Generation

```python
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism, NeuralSymbolicBridge

def visualize_attention_heatmap(concepts, attention_weights, title="Attention Heatmap"):
    """Generate attention heatmap visualization."""
    
    # Convert to numpy for visualization
    attention_matrix = attention_weights.detach().cpu().numpy()
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        attention_matrix,
        xticklabels=concepts,
        yticklabels=concepts,
        annot=True,
        cmap='Blues',
        fmt='.3f',
        cbar_kws={'label': 'Attention Weight'}
    )
    
    plt.title(title)
    plt.xlabel('Attended Concepts')
    plt.ylabel('Query Concepts')
    plt.xticks(rotation=45)
    plt.yticks(rotation=45)
    plt.tight_layout()
    
    # Save the visualization
    plt.savefig(f'/tmp/attention_heatmap_{title.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.show()
    
    return f'/tmp/attention_heatmap_{title.lower().replace(" ", "_")}.png'

# Example usage
bridge = NeuralSymbolicBridge(embedding_dim=128)
attention = CognitiveAttentionMechanism(embedding_dim=128, num_heads=1)

# Define a reasoning scenario
reasoning_concepts = [
    "hypothesis", "evidence", "logic", "conclusion", 
    "uncertainty", "validation", "inference"
]

embeddings = bridge.embed_concepts(reasoning_concepts)
attended_output, attention_weights = attention.forward(embeddings.unsqueeze(0))

# Generate visualization
heatmap_path = visualize_attention_heatmap(
    reasoning_concepts, 
    attention_weights[0, 0],  # First head
    "Scientific Reasoning Attention Pattern"
)
print(f"Attention heatmap saved to: {heatmap_path}")
```

### Example 7: Dynamic Attention Evolution

```python
def track_attention_evolution(initial_concepts, reasoning_steps, max_steps=5):
    """Track how attention evolves through reasoning steps."""
    
    bridge = NeuralSymbolicBridge(embedding_dim=128)
    attention = CognitiveAttentionMechanism(embedding_dim=128, num_heads=4)
    
    attention_history = []
    current_concepts = initial_concepts.copy()
    
    print("Attention Evolution During Reasoning:")
    print("=" * 40)
    
    for step in range(max_steps):
        # Get embeddings for current concepts
        embeddings = bridge.embed_concepts(current_concepts)
        
        # Apply attention
        attended_output, attention_weights = attention.forward(embeddings.unsqueeze(0))
        
        # Calculate attention statistics
        mean_attention = attention_weights.mean(dim=1).squeeze()
        max_attention_idx = torch.argmax(mean_attention)
        focused_concept = current_concepts[max_attention_idx]
        
        attention_stats = {
            "step": step + 1,
            "focused_concept": focused_concept,
            "attention_entropy": -torch.sum(mean_attention * torch.log(mean_attention + 1e-8)).item(),
            "max_attention_weight": mean_attention[max_attention_idx].item(),
            "concepts": current_concepts.copy()
        }
        
        attention_history.append(attention_stats)
        
        print(f"\\nStep {step + 1}:")
        print(f"  Focus: {focused_concept}")
        print(f"  Attention Entropy: {attention_stats['attention_entropy']:.3f}")
        print(f"  Max Weight: {attention_stats['max_attention_weight']:.3f}")
        
        # Simulate reasoning step adding new concepts
        if step < len(reasoning_steps):
            current_concepts.append(reasoning_steps[step])
            print(f"  Added concept: {reasoning_steps[step]}")
    
    return attention_history

# Example: Problem-solving attention evolution
initial_concepts = ["problem", "constraints", "resources"]
reasoning_steps = ["analysis", "strategy", "implementation", "validation", "optimization"]

evolution = track_attention_evolution(initial_concepts, reasoning_steps)

# Analyze evolution patterns
print("\\nAttention Evolution Summary:")
print("-" * 30)
for stats in evolution:
    print(f"Step {stats['step']}: {stats['focused_concept']} "
          f"(entropy: {stats['attention_entropy']:.2f})")
```

## Integration with Agent-Zero Framework

### Example 8: Agent-Zero Attention-Guided Task Execution

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool
import json

async def attention_guided_reasoning(agent, query, context_concepts=None):
    """Perform attention-guided reasoning with Agent-Zero integration."""
    
    # Initialize cognitive reasoning tool
    tool = CognitiveReasoningTool(agent, 'cognitive_reasoning', None, {}, '', None)
    
    # Phase 1: Initial attention analysis
    initial_response = await tool.execute(
        query,
        operation="analyze_patterns",
        context_concepts=context_concepts or []
    )
    
    # Parse attention patterns
    try:
        data = json.loads(initial_response.message.split("Data: ")[1])
        attention_patterns = data.get("analysis", {})
    except:
        attention_patterns = {}
    
    print(f"Initial Attention Analysis for: '{query}'")
    print(f"Detected patterns: {attention_patterns}")
    
    # Phase 2: Focused reasoning based on attention
    focused_response = await tool.execute(
        query,
        operation="reason",
        attention_focus=attention_patterns.get("key_concepts", []),
        use_attention_weighting=True
    )
    
    # Phase 3: Cross-reference with related tools
    cross_ref_response = await tool.execute(
        query,
        operation="cross_reference",
        attention_context=attention_patterns
    )
    
    return {
        "initial_analysis": initial_response,
        "focused_reasoning": focused_response,
        "cross_references": cross_ref_response,
        "attention_summary": attention_patterns
    }
```

## Performance Optimization and Best Practices

### Example 9: Attention Computation Optimization

```python
import torch
from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism

def optimized_attention_reasoning(concepts, batch_size=32, use_caching=True):
    """Optimize attention computation for large-scale reasoning."""
    
    attention = CognitiveAttentionMechanism(embedding_dim=128, num_heads=8)
    bridge = NeuralSymbolicBridge(embedding_dim=128)
    
    # Enable caching for repeated concepts
    if use_caching:
        bridge.enable_embedding_cache()
    
    # Process concepts in batches
    results = []
    for i in range(0, len(concepts), batch_size):
        batch_concepts = concepts[i:i + batch_size]
        
        # Get embeddings (cached if available)
        embeddings = bridge.embed_concepts(batch_concepts)
        
        # Apply attention with gradient checkpointing for memory efficiency
        with torch.cuda.amp.autocast(enabled=torch.cuda.is_available()):
            attended_output, attention_weights = attention.forward(embeddings.unsqueeze(0))
        
        # Store batch results
        batch_results = {
            "concepts": batch_concepts,
            "attended_output": attended_output.detach(),
            "attention_weights": attention_weights.detach(),
            "batch_index": i // batch_size
        }
        results.append(batch_results)
    
    return results
```

## Testing Attention-Based Reasoning

### Unit Tests for Attention Mechanisms

```python
import torch
import pytest
from python.helpers.neural_symbolic_bridge import CognitiveAttentionMechanism, NeuralSymbolicBridge

def test_attention_mechanism_basic():
    """Test basic attention mechanism functionality."""
    attention = CognitiveAttentionMechanism(embedding_dim=64, num_heads=4)
    
    # Create test embeddings
    test_embeddings = torch.randn(1, 5, 64)  # batch_size=1, seq_len=5, embed_dim=64
    
    # Apply attention
    output, weights = attention.forward(test_embeddings)
    
    # Validate output shapes
    assert output.shape == test_embeddings.shape
    assert weights.shape == (1, 4, 5, 5)  # batch, heads, seq_len, seq_len
    
    # Validate attention weights sum to 1
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)))

def test_ecan_integration():
    """Test ECAN weight integration."""
    attention = CognitiveAttentionMechanism(embedding_dim=64, num_heads=2)
    
    test_embeddings = torch.randn(1, 3, 64)
    ecan_weights = torch.tensor([[0.8, 0.3, 0.1]])  # High, Medium, Low importance
    
    output, weights = attention.forward(test_embeddings, ecan_weights=ecan_weights)
    
    # Validate that high ECAN weights influence attention
    mean_attention = weights.mean(dim=1)  # Average across heads
    
    # The concept with highest ECAN weight should get more attention
    assert mean_attention[0, :, 0].sum() > mean_attention[0, :, 2].sum()

def test_concept_embedding_consistency():
    """Test that concept embeddings remain consistent."""
    bridge = NeuralSymbolicBridge(embedding_dim=64)
    
    concepts = ["reasoning", "memory", "attention"]
    
    # Get embeddings twice
    embeddings1 = bridge.embed_concepts(concepts)
    embeddings2 = bridge.embed_concepts(concepts)
    
    # Should be identical due to deterministic hashing
    assert torch.allclose(embeddings1, embeddings2)

# Run tests
if __name__ == "__main__":
    test_attention_mechanism_basic()
    test_ecan_integration()
    test_concept_embedding_consistency()
    print("All attention mechanism tests passed!")
```

## Summary and Next Steps

This documentation provides comprehensive examples for implementing attention-based reasoning in PyCog-Zero. The examples cover:

1. **Basic attention mechanisms** with single and multi-head attention
2. **ECAN integration** for biologically-inspired attention allocation
3. **Temporal and cross-modal attention** patterns
4. **Visualization techniques** for understanding attention dynamics
5. **Agent-Zero integration** patterns for practical applications
6. **Performance optimization** strategies
7. **Testing and validation** approaches

### Recommended Implementation Path

1. Start with **Example 1** to understand basic attention-focused reasoning
2. Experiment with **Examples 2-3** to explore multi-head and ECAN-weighted attention
3. Use **Examples 6-7** for visualization and debugging
4. Implement **Example 8** for Agent-Zero integration
5. Apply **Example 9** for performance optimization
6. Run the provided tests to validate implementation

### Integration with Existing PyCog-Zero Components

- **CognitiveReasoningTool**: Enhanced with attention-based operations
- **NeuralSymbolicBridge**: Provides embedding and attention infrastructure
- **Configuration**: Extended with comprehensive attention parameters
- **Cross-tool Integration**: Enables attention sharing across cognitive tools

This attention-based reasoning system forms a critical foundation for advanced cognitive capabilities in PyCog-Zero, enabling more sophisticated, biologically-inspired artificial intelligence behaviors.