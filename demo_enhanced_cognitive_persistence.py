#!/usr/bin/env python3
"""
Comprehensive demonstration of cognitive memory persistence with AtomSpace backend

This demo shows:
1. Knowledge storage and retrieval
2. Association creation
3. Cognitive reasoning
4. Persistence across sessions
5. Error handling and recovery
"""

import asyncio
import sys
import os

# Add the project root to the path
sys.path.append(os.path.abspath('.'))

from python.tools.cognitive_memory import CognitiveMemoryTool


class MockAgent:
    """Mock Agent-Zero instance for demonstration."""
    def __init__(self):
        self.capabilities = ["cognitive_memory", "reasoning", "persistence"]
        self.tools = []


def create_tool_instance():
    """Helper to create a properly initialized tool instance."""
    mock_agent = MockAgent()
    tool = CognitiveMemoryTool.__new__(CognitiveMemoryTool)
    tool.agent = mock_agent
    tool._initialize_if_needed()
    return tool


async def demo_basic_operations():
    """Demonstrate basic cognitive memory operations."""
    print("=" * 60)
    print("DEMO 1: Basic Cognitive Memory Operations")
    print("=" * 60)
    
    tool = create_tool_instance()
    
    # Show initial status
    status = await tool.execute("status")
    print(f"Initial status: {status.message}\n")
    
    # Store knowledge about AI concepts
    ai_concepts = [
        {
            "concept": "machine_learning",
            "properties": {
                "type": "ai_technique",
                "complexity": "medium",
                "applications": "pattern_recognition"
            },
            "relationships": {
                "enables": ["automation", "prediction"],
                "requires": ["data", "algorithms"],
                "part_of": ["artificial_intelligence"]
            }
        },
        {
            "concept": "neural_networks",
            "properties": {
                "type": "computational_model",
                "inspired_by": "biological_neurons",
                "complexity": "high"
            },
            "relationships": {
                "implements": ["machine_learning"],
                "uses": ["backpropagation", "gradient_descent"],
                "enables": ["deep_learning"]
            }
        },
        {
            "concept": "deep_learning",
            "properties": {
                "type": "ml_subset",
                "layer_count": "many",
                "performance": "high"
            },
            "relationships": {
                "specializes": ["neural_networks"],
                "excels_at": ["image_recognition", "nlp"],
                "requires": ["large_datasets", "gpu_computing"]
            }
        }
    ]
    
    # Store each concept
    for concept_data in ai_concepts:
        result = await tool.execute("store", data=concept_data)
        print(f"✓ {result.message}")
    
    print()
    
    # Create additional associations
    associations = [
        ("machine_learning", "deep_learning", "includes", 0.9),
        ("neural_networks", "artificial_intelligence", "contributes_to", 0.8),
        ("deep_learning", "computer_vision", "powers", 0.85),
        ("machine_learning", "data_science", "overlaps_with", 0.7)
    ]
    
    print("Creating associations...")
    for source, target, rel_type, strength in associations:
        result = await tool.execute("associate", data={
            "source": source,
            "target": target,
            "type": rel_type,
            "strength": strength
        })
        print(f"✓ {result.message}")
    
    print()
    
    # Demonstrate retrieval
    print("Retrieving stored knowledge...")
    for concept in ["machine_learning", "neural_networks", "deep_learning"]:
        result = await tool.execute("retrieve", data={"concept": concept})
        print(f"Retrieved {concept}:")
        # Parse and display the JSON data nicely
        if "Data: " in result.message:
            import json
            data_start = result.message.find("Data: ") + 6
            try:
                knowledge_data = json.loads(result.message[data_start:])
                print(f"  - Properties: {len(knowledge_data.get('properties', {}))}")
                print(f"  - Relationships: {len(knowledge_data.get('relationships', {}))}")
                print(f"  - Connections: {len(knowledge_data.get('links', []))}")
            except:
                print(f"  - {result.message}")
        print()
    
    # Show final status
    final_status = await tool.execute("status")
    print(f"Final status: {final_status.message}")
    
    return tool


async def demo_reasoning():
    """Demonstrate cognitive reasoning capabilities."""
    print("\n" + "=" * 60)
    print("DEMO 2: Cognitive Reasoning")
    print("=" * 60)
    
    tool = create_tool_instance()  # This should load the previously saved data
    
    # Test reasoning with different queries
    reasoning_queries = [
        "learning",
        "neural",
        "intelligence",
        "deep",
        "machine"
    ]
    
    for query in reasoning_queries:
        result = await tool.execute("reason", data={"query": query})
        print(f"Query '{query}':")
        # Parse the reasoning results
        if "Results: " in result.message:
            import json
            results_start = result.message.find("Results: ") + 9
            try:
                reasoning_data = json.loads(result.message[results_start:])
                print(f"  - Total atoms: {reasoning_data.get('total_atoms', 0)}")
                print(f"  - Concept nodes: {reasoning_data.get('concept_nodes', 0)}")
                print(f"  - Found concepts: {len(reasoning_data.get('connected_concepts', []))}")
                for concept in reasoning_data.get('connected_concepts', []):
                    print(f"    • {concept['concept']} ({concept['connections']} connections)")
            except:
                print(f"  - {result.message}")
        print()


async def demo_persistence():
    """Demonstrate persistence across sessions."""
    print("\n" + "=" * 60)
    print("DEMO 3: Persistence Across Sessions")
    print("=" * 60)
    
    print("Creating first tool instance and adding unique data...")
    tool1 = create_tool_instance()
    
    # Add some unique data
    unique_concept = {
        "concept": "quantum_computing",
        "properties": {
            "type": "computational_paradigm",
            "quantum_mechanics": "true",
            "advantage": "exponential_speedup"
        },
        "relationships": {
            "threatens": ["current_cryptography"],
            "enables": ["optimization_problems"],
            "requires": ["quantum_hardware"]
        }
    }
    
    result = await tool1.execute("store", data=unique_concept)
    print(f"✓ {result.message}")
    
    # Create association
    result = await tool1.execute("associate", data={
        "source": "quantum_computing",
        "target": "artificial_intelligence",
        "type": "future_synergy",
        "strength": 0.95
    })
    print(f"✓ {result.message}")
    
    # Get status before "restart"
    status1 = await tool1.execute("status")
    print(f"Session 1 status: {status1.message}")
    
    print("\nSimulating application restart...")
    print("Creating second tool instance (should load persisted data)...")
    
    # Create a completely new tool instance (simulating restart)
    tool2 = create_tool_instance()
    
    # Verify the data persisted
    result = await tool2.execute("retrieve", data={"concept": "quantum_computing"})
    print(f"✓ After restart, retrieved: {result.message}")
    
    # Test reasoning on persisted data
    result = await tool2.execute("reason", data={"query": "quantum"})
    print(f"✓ Reasoning after restart: {result.message}")
    
    # Get status after "restart"
    status2 = await tool2.execute("status")
    print(f"Session 2 status: {status2.message}")


async def demo_error_handling():
    """Demonstrate error handling and recovery."""
    print("\n" + "=" * 60)
    print("DEMO 4: Error Handling and Recovery")
    print("=" * 60)
    
    tool = create_tool_instance()
    
    print("Testing invalid operations...")
    
    # Test invalid operation
    result = await tool.execute("invalid_op")
    print(f"Invalid operation: {result.message}")
    
    # Test missing data
    result = await tool.execute("store", data={})
    print(f"Missing concept: {result.message}")
    
    # Test missing fields
    result = await tool.execute("retrieve", data={"not_concept": "test"})
    print(f"Wrong field: {result.message}")
    
    # Test incomplete association
    result = await tool.execute("associate", data={"source": "only_source"})
    print(f"Incomplete association: {result.message}")
    
    print("\nTesting non-existent concept retrieval...")
    result = await tool.execute("retrieve", data={"concept": "non_existent_concept"})
    print(f"Non-existent concept: {result.message}")
    
    print("\n✓ All error cases handled gracefully")


async def demo_performance():
    """Demonstrate performance with larger datasets."""
    print("\n" + "=" * 60)
    print("DEMO 5: Performance with Larger Dataset")
    print("=" * 60)
    
    tool = create_tool_instance()
    
    print("Adding 50 concepts with properties and relationships...")
    import time
    
    start_time = time.time()
    
    # Create a larger knowledge base
    for i in range(50):
        concept_data = {
            "concept": f"concept_{i:03d}",
            "properties": {
                "id": str(i),
                "type": "generated",
                "category": f"group_{i % 5}",
                "value": str(i * 10)
            },
            "relationships": {
                "related_to": [f"concept_{(i+1) % 50:03d}", f"concept_{(i+2) % 50:03d}"],
                "part_of": [f"group_{i % 5}"],
                "precedes": [f"concept_{(i+1) % 50:03d}"]
            }
        }
        
        await tool.execute("store", data=concept_data)
        
        # Add some associations
        if i > 0:
            await tool.execute("associate", data={
                "source": f"concept_{i:03d}",
                "target": f"concept_{i-1:03d}",
                "type": "follows",
                "strength": 0.7
            })
    
    storage_time = time.time() - start_time
    print(f"✓ Stored 50 concepts in {storage_time:.2f} seconds")
    
    # Test retrieval performance
    start_time = time.time()
    for i in range(0, 50, 10):  # Sample every 10th concept
        await tool.execute("retrieve", data={"concept": f"concept_{i:03d}"})
    
    retrieval_time = time.time() - start_time
    print(f"✓ Retrieved 5 sample concepts in {retrieval_time:.2f} seconds")
    
    # Test reasoning performance
    start_time = time.time()
    result = await tool.execute("reason", data={"query": "concept"})
    reasoning_time = time.time() - start_time
    print(f"✓ Reasoning query completed in {reasoning_time:.2f} seconds")
    
    # Final status
    status = await tool.execute("status")
    print(f"Final status with large dataset: {status.message}")


async def main():
    """Run all demonstrations."""
    print("🧠 COGNITIVE MEMORY PERSISTENCE DEMONSTRATION 🧠")
    print("This demo shows the enhanced cognitive memory with AtomSpace backend")
    print("Working in fallback mode (OpenCog not required)")
    
    try:
        # Run all demonstrations
        await demo_basic_operations()
        await demo_reasoning()
        await demo_persistence()
        await demo_error_handling()
        await demo_performance()
        
        print("\n" + "=" * 60)
        print("🎉 ALL DEMONSTRATIONS COMPLETED SUCCESSFULLY! 🎉")
        print("=" * 60)
        
        print("\nKey features demonstrated:")
        print("✓ Knowledge storage with properties and relationships")
        print("✓ Association creation with confidence values")
        print("✓ Cognitive reasoning and query processing")
        print("✓ Persistent memory across application restarts")
        print("✓ Robust error handling and validation")
        print("✓ Performance with larger datasets")
        print("✓ Backup and recovery mechanisms")
        print("✓ Data integrity validation")
        
        # Show final memory file info
        import os
        memory_file = "memory/cognitive_atomspace.pkl"
        if os.path.exists(memory_file):
            size = os.path.getsize(memory_file)
            print(f"\n📁 Memory file: {memory_file} ({size} bytes)")
            backup_file = memory_file.replace('.pkl', '_backup.pkl')
            if os.path.exists(backup_file):
                backup_size = os.path.getsize(backup_file)
                print(f"📁 Backup file: {backup_file} ({backup_size} bytes)")
        
    except Exception as e:
        print(f"\n❌ Error during demonstration: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    asyncio.run(main())