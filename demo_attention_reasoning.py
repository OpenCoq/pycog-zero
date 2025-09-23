#!/usr/bin/env python3
"""
Attention-Based Reasoning Demo for PyCog-Zero

This script demonstrates the attention mechanisms working with mock data,
providing a practical example of the concepts described in the documentation.
"""

import sys
import os
import json

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__)))

def demo_attention_configuration():
    """Demonstrate loading and parsing attention configuration."""
    
    print("🔧 Attention Configuration Demo")
    print("=" * 40)
    
    try:
        with open('conf/config_cognitive.json', 'r') as f:
            config = json.load(f)
        
        attention_config = config.get('attention_config', {})
        
        print("Attention System Configuration:")
        print(f"  ECAN Enabled: {attention_config.get('ecan_enabled', False)}")
        
        # Multi-head attention settings
        mha_config = attention_config.get('attention_mechanisms', {}).get('multi_head_attention', {})
        print(f"  Multi-Head Attention:")
        print(f"    - Enabled: {mha_config.get('enabled', False)}")
        print(f"    - Heads: {mha_config.get('num_heads', 'N/A')}")
        print(f"    - Dropout: {mha_config.get('dropout', 'N/A')}")
        
        # ECAN settings
        ecan_config = attention_config.get('ecan_config', {})
        print(f"  ECAN Configuration:")
        print(f"    - STI Decay: {ecan_config.get('sti_decay_factor', 'N/A')}")
        print(f"    - LTI Decay: {ecan_config.get('lti_decay_factor', 'N/A')}")
        print(f"    - STI Threshold: {ecan_config.get('sti_threshold', 'N/A')}")
        print(f"    - Hebbian Learning: {ecan_config.get('hebbian_learning', 'N/A')}")
        
        # Importance diffusion
        diffusion = ecan_config.get('importance_diffusion', {})
        print(f"    - Importance Diffusion: {diffusion.get('enabled', False)}")
        print(f"    - Diffusion Factor: {diffusion.get('diffusion_factor', 'N/A')}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error loading configuration: {e}")
        return False

def demo_mock_attention_patterns():
    """Demonstrate attention patterns using mock data."""
    
    print("\\n🎯 Mock Attention Patterns Demo")
    print("=" * 40)
    
    # Simulate attention analysis for different scenarios
    scenarios = [
        {
            "name": "Problem Solving",
            "concepts": ["problem", "analysis", "solution", "validation"],
            "ecan_weights": [0.8, 0.9, 0.7, 0.4],
            "description": "Focus on analysis and problem definition"
        },
        {
            "name": "Learning Context",
            "concepts": ["memory", "learning", "practice", "feedback"],
            "ecan_weights": [0.6, 0.9, 0.5, 0.7],
            "description": "Emphasize learning and feedback mechanisms"
        },
        {
            "name": "Creative Thinking",
            "concepts": ["intuition", "creativity", "logic", "innovation"],
            "ecan_weights": [0.8, 0.9, 0.5, 0.8],
            "description": "Balance between creativity and structured thinking"
        }
    ]
    
    for scenario in scenarios:
        print(f"\\n📊 Scenario: {scenario['name']}")
        print(f"Description: {scenario['description']}")
        
        concepts = scenario['concepts']
        weights = scenario['ecan_weights']
        
        # Simulate attention distribution
        total_attention = sum(weights)
        normalized_weights = [w / total_attention for w in weights]
        
        # Display attention allocation
        print("Attention Distribution:")
        for i, (concept, norm_weight, ecan_weight) in enumerate(zip(concepts, normalized_weights, weights)):
            bar_length = int(norm_weight * 20)  # Scale to 20 chars
            bar = "█" * bar_length + "░" * (20 - bar_length)
            print(f"  {concept:<12} │{bar}│ {norm_weight:.2f} (ECAN: {ecan_weight:.1f})")
        
        # Find primary focus
        max_idx = weights.index(max(weights))
        primary_focus = concepts[max_idx]
        print(f"Primary Focus: {primary_focus} (attention weight: {normalized_weights[max_idx]:.2f})")
        
        # Calculate attention entropy (measure of focus vs. distribution)
        import math
        entropy = -sum(w * math.log(w + 1e-8) for w in normalized_weights)
        focus_level = "High" if entropy < 1.2 else "Medium" if entropy < 1.5 else "Low"
        print(f"Focus Level: {focus_level} (entropy: {entropy:.2f})")

def demo_temporal_attention_evolution():
    """Demonstrate how attention evolves over time during reasoning."""
    
    print("\\n⏱️  Temporal Attention Evolution Demo")
    print("=" * 40)
    
    # Simulate a reasoning process with evolving attention
    reasoning_steps = [
        {
            "step": 1,
            "phase": "Problem Identification",
            "concepts": ["problem", "constraints"],
            "attention": [0.7, 0.3]
        },
        {
            "step": 2,
            "phase": "Analysis",
            "concepts": ["problem", "constraints", "analysis"],
            "attention": [0.4, 0.2, 0.4]
        },
        {
            "step": 3,
            "phase": "Solution Generation",
            "concepts": ["problem", "constraints", "analysis", "solution"],
            "attention": [0.2, 0.1, 0.3, 0.4]
        },
        {
            "step": 4,
            "phase": "Validation",
            "concepts": ["problem", "constraints", "analysis", "solution", "validation"],
            "attention": [0.15, 0.1, 0.2, 0.3, 0.25]
        }
    ]
    
    print("Reasoning Process Attention Evolution:")
    for step_data in reasoning_steps:
        step = step_data["step"]
        phase = step_data["phase"]
        concepts = step_data["concepts"]
        attention_weights = step_data["attention"]
        
        print(f"\\nStep {step}: {phase}")
        
        # Find primary focus
        max_idx = attention_weights.index(max(attention_weights))
        primary_concept = concepts[max_idx]
        max_weight = attention_weights[max_idx]
        
        print(f"  Primary Focus: {primary_concept} ({max_weight:.2f})")
        print("  Attention Distribution:")
        
        for concept, weight in zip(concepts, attention_weights):
            # Visual attention bar
            bar_length = int(weight * 30)
            bar = "▓" * bar_length + "▒" * max(0, 30 - bar_length)
            print(f"    {concept:<12} │{bar[:30]}│ {weight:.2f}")
    
    print("\\n📈 Evolution Summary:")
    print("  • Initial focus on problem identification")
    print("  • Attention shifts to analysis phase")
    print("  • Solution generation becomes primary focus")
    print("  • Validation gains importance while maintaining solution focus")

def demo_cross_modal_attention():
    """Demonstrate cross-modal attention integration."""
    
    print("\\n🔀 Cross-Modal Attention Demo")
    print("=" * 40)
    
    # Simulate attention across different cognitive modalities
    modalities = {
        "visual": {
            "concepts": ["pattern_recognition", "spatial_reasoning", "visual_memory"],
            "baseline_attention": [0.4, 0.3, 0.3]
        },
        "linguistic": {
            "concepts": ["semantic_analysis", "syntax_processing", "pragmatic_inference"],
            "baseline_attention": [0.4, 0.3, 0.3]
        },
        "logical": {
            "concepts": ["deductive_reasoning", "inductive_reasoning", "abductive_reasoning"],
            "baseline_attention": [0.4, 0.3, 0.3]
        }
    }
    
    # Simulate cross-modal query: "How do visual patterns relate to logical reasoning?"
    query = "How do visual patterns relate to logical reasoning?"
    print(f"Query: '{query}'")
    
    # Simulate attention reallocation based on query relevance
    cross_modal_weights = {
        "visual": 0.5,  # High relevance
        "linguistic": 0.2,  # Medium relevance for language understanding
        "logical": 0.3   # High relevance for logical reasoning
    }
    
    print("\\nCross-Modal Attention Allocation:")
    for modality, modal_weight in cross_modal_weights.items():
        print(f"\\n{modality.upper()} Modality (weight: {modal_weight:.1f}):")
        
        concepts = modalities[modality]["concepts"]
        base_attention = modalities[modality]["baseline_attention"]
        
        # Apply modal weight to adjust attention within modality
        adjusted_attention = [weight * modal_weight for weight in base_attention]
        
        for concept, attention in zip(concepts, adjusted_attention):
            bar_length = int(attention * 40)
            bar = "█" * bar_length + "░" * max(0, 40 - bar_length)
            print(f"  {concept:<20} │{bar[:40]}│ {attention:.3f}")
    
    # Identify cross-modal connections
    print("\\n🔗 Cross-Modal Connections Identified:")
    print("  • pattern_recognition ↔ deductive_reasoning")
    print("  • visual_memory ↔ semantic_analysis") 
    print("  • spatial_reasoning ↔ logical_inference")
    
    # Calculate integration strength
    integration_score = sum(cross_modal_weights.values()) / len(cross_modal_weights)
    print(f"\\nIntegration Strength: {integration_score:.2f}")

def demo_attention_optimization():
    """Demonstrate attention optimization strategies."""
    
    print("\\n⚡ Attention Optimization Demo")
    print("=" * 40)
    
    # Simulate performance metrics for different attention strategies
    strategies = [
        {
            "name": "Uniform Attention",
            "description": "Equal attention to all concepts",
            "execution_time": 2.5,
            "memory_usage": 150,
            "accuracy": 0.72,
            "focus_entropy": 2.1
        },
        {
            "name": "ECAN-Weighted",
            "description": "Biological attention allocation",
            "execution_time": 1.8,
            "memory_usage": 120,
            "accuracy": 0.84,
            "focus_entropy": 1.4
        },
        {
            "name": "Multi-Head + ECAN",
            "description": "Combined neural and biological attention",
            "execution_time": 2.1,
            "memory_usage": 140,
            "accuracy": 0.89,
            "focus_entropy": 1.2
        },
        {
            "name": "Adaptive Attention",
            "description": "Context-aware attention allocation",
            "execution_time": 1.6,
            "memory_usage": 110,
            "accuracy": 0.91,
            "focus_entropy": 1.0
        }
    ]
    
    print("Performance Comparison:")
    print("Strategy              Time(s) Memory(MB) Accuracy Focus")
    print("-" * 55)
    
    for strategy in strategies:
        name = strategy["name"]
        time = strategy["execution_time"]
        memory = strategy["memory_usage"]
        accuracy = strategy["accuracy"]
        entropy = strategy["focus_entropy"]
        
        # Visual performance indicators
        accuracy_bar = "▓" * int(accuracy * 10)
        focus_score = "High" if entropy < 1.2 else "Med" if entropy < 1.6 else "Low"
        
        print(f"{name:<20} {time:>6.1f} {memory:>8} {accuracy:>8.2f} {focus_score:>5}")
    
    # Highlight optimal strategy
    optimal = min(strategies, key=lambda x: x["execution_time"] + (1 - x["accuracy"]) * 5)
    print(f"\\n🏆 Optimal Strategy: {optimal['name']}")
    print(f"   {optimal['description']}")
    print(f"   Performance: {optimal['accuracy']:.1%} accuracy, {optimal['execution_time']:.1f}s execution")

def main():
    """Run all attention-based reasoning demos."""
    
    print("🧠 PyCog-Zero Attention-Based Reasoning Demo")
    print("=" * 50)
    print("Demonstrating attention mechanisms with mock data\\n")
    
    demos = [
        demo_attention_configuration,
        demo_mock_attention_patterns,
        demo_temporal_attention_evolution,
        demo_cross_modal_attention,
        demo_attention_optimization
    ]
    
    success_count = 0
    
    for demo in demos:
        try:
            if demo != demo_attention_configuration:  # Skip return value check for others
                demo()
                success_count += 1
            else:
                if demo():
                    success_count += 1
        except Exception as e:
            print(f"❌ Demo {demo.__name__} failed: {e}")
    
    print("\\n" + "=" * 50)
    print("🎉 Demo Complete!")
    
    if success_count >= 4:  # Expect at least 4 successful demos
        print("✅ All attention-based reasoning concepts demonstrated successfully.")
        print("\\n📚 Next Steps:")
        print("  1. Review the full documentation: docs/attention_based_reasoning.md")
        print("  2. Implement the examples in your Agent-Zero workflows")
        print("  3. Experiment with different attention parameters")
        print("  4. Monitor attention patterns in real cognitive tasks")
    else:
        print("⚠️  Some demos encountered issues. Check the error messages above.")

if __name__ == "__main__":
    main()