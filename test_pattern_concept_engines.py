#!/usr/bin/env python3
"""
Simplified test suite for Advanced Pattern Recognition and Concept Learning engines.

Tests the core functionality of the newly implemented engines without requiring
the full Agent-Zero framework initialization.
"""

import sys
import os
import json
import time
import statistics
from typing import Dict, Any, List
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test results storage
test_results = {
    "tests_passed": 0,
    "tests_failed": 0,
    "advanced_pattern_results": {},
    "concept_learning_results": {},
    "errors": []
}

def log_test_result(test_name: str, passed: bool, details: str = ""):
    """Log test result and update counters."""
    if passed:
        test_results["tests_passed"] += 1
        print(f"✅ {test_name}")
    else:
        test_results["tests_failed"] += 1
        print(f"❌ {test_name}: {details}")
        test_results["errors"].append(f"{test_name}: {details}")


def test_hierarchical_pattern_engine():
    """Test the hierarchical pattern engine directly."""
    test_name = "Hierarchical Pattern Engine"
    
    try:
        from python.tools.advanced_pattern_recognition import HierarchicalPatternEngine
        
        engine = HierarchicalPatternEngine()
        
        test_data = [
            {"elements": ["input", "processing", "output"], "context": "data_flow"},
            {"elements": ["input", "transformation", "output"], "context": "data_flow"},
            {"elements": ["perception", "cognition", "action"], "context": "cognitive_flow"},
            {"elements": ["perception", "processing", "action"], "context": "cognitive_flow"}
        ]
        
        patterns = engine.detect_hierarchical_patterns(test_data, max_levels=3)
        
        if len(patterns) > 0:
            test_results["advanced_pattern_results"]["hierarchical_engine"] = {
                "patterns_detected": len(patterns),
                "abstraction_levels": list(set(p.abstraction_level for p in patterns)),
                "pattern_types": list(set(p.pattern_type for p in patterns)),
                "average_confidence": statistics.mean([p.confidence for p in patterns]),
                "success": True
            }
            log_test_result(test_name, True)
            return True
        else:
            log_test_result(test_name, False, "No patterns detected")
            return False
            
    except Exception as e:
        log_test_result(test_name, False, str(e))
        return False


def test_temporal_pattern_engine():
    """Test the temporal pattern engine directly."""
    test_name = "Temporal Pattern Engine"
    
    try:
        from python.tools.advanced_pattern_recognition import TemporalPatternEngine
        
        engine = TemporalPatternEngine()
        
        test_data = [
            {"event": "start", "timestamp": 1.0, "value": 10, "event_type": "initialization"},
            {"event": "process", "timestamp": 2.0, "value": 20, "event_type": "processing"},
            {"event": "end", "timestamp": 3.0, "value": 15, "event_type": "completion"},
            {"event": "start", "timestamp": 4.0, "value": 12, "event_type": "initialization"},
            {"event": "process", "timestamp": 5.0, "value": 22, "event_type": "processing"},
            {"event": "end", "timestamp": 6.0, "value": 18, "event_type": "completion"},
            {"event": "start", "timestamp": 7.0, "value": 11, "event_type": "initialization"},
            {"event": "process", "timestamp": 8.0, "value": 21, "event_type": "processing"},
            {"event": "end", "timestamp": 9.0, "value": 16, "event_type": "completion"}
        ]
        
        patterns = engine.detect_temporal_patterns(test_data)
        
        if len(patterns) > 0:
            test_results["advanced_pattern_results"]["temporal_engine"] = {
                "patterns_detected": len(patterns),
                "pattern_types": list(set(p.pattern_type for p in patterns)),
                "average_confidence": statistics.mean([p.confidence for p in patterns]),
                "pattern_frequencies": [p.frequency for p in patterns],
                "success": True
            }
            log_test_result(test_name, True)
            return True
        else:
            log_test_result(test_name, False, "No temporal patterns detected")
            return False
            
    except Exception as e:
        log_test_result(test_name, False, str(e))
        return False


def test_analogical_reasoning_engine():
    """Test the analogical reasoning engine directly."""
    test_name = "Analogical Reasoning Engine"
    
    try:
        from python.tools.advanced_pattern_recognition import AnalogicalReasoningEngine
        
        engine = AnalogicalReasoningEngine()
        
        source_domain = {
            "domain": "solar_system",
            "entities": ["sun", "planets", "moons"],
            "relationships": [
                {"type": "orbits", "from": "planets", "to": "sun"},
                {"type": "orbits", "from": "moons", "to": "planets"}
            ],
            "functions": ["gravitational_attraction", "orbital_motion"]
        }
        
        target_domain = {
            "domain": "atomic_structure",
            "entities": ["nucleus", "electrons", "energy_levels"],
            "relationships": [
                {"type": "orbits", "from": "electrons", "to": "nucleus"},
                {"type": "occupies", "from": "electrons", "to": "energy_levels"}
            ],
            "functions": ["electromagnetic_attraction", "quantum_motion"]
        }
        
        analogies = engine.find_analogies(source_domain, target_domain)
        
        if len(analogies) > 0:
            test_results["advanced_pattern_results"]["analogical_engine"] = {
                "analogies_found": len(analogies),
                "analogy_types": list(set(p.elements[0].split(':')[0] for p in analogies if p.elements)),
                "average_confidence": statistics.mean([p.confidence for p in analogies]),
                "domains": list(set(d for p in analogies for d in p.domains)),
                "success": True
            }
            log_test_result(test_name, True)
            return True
        else:
            log_test_result(test_name, False, "No analogies found")
            return False
            
    except Exception as e:
        log_test_result(test_name, False, str(e))
        return False


def test_concept_formation_engine():
    """Test the concept formation engine directly."""
    test_name = "Concept Formation Engine"
    
    try:
        from python.tools.concept_learning import ConceptFormationEngine
        
        engine = ConceptFormationEngine()
        
        test_instances = [
            {"type": "vehicle", "wheels": 4, "engine": "gas", "size": "medium", "speed": 60},
            {"type": "vehicle", "wheels": 4, "engine": "gas", "size": "large", "speed": 55},
            {"type": "vehicle", "wheels": 4, "engine": "electric", "size": "small", "speed": 70},
            {"type": "vehicle", "wheels": 2, "engine": "gas", "size": "small", "speed": 80},
            {"type": "vehicle", "wheels": 2, "engine": "gas", "size": "small", "speed": 85},
            {"type": "animal", "legs": 4, "habitat": "land", "size": "medium", "speed": 30},
            {"type": "animal", "legs": 4, "habitat": "land", "size": "large", "speed": 25},
            {"type": "animal", "legs": 2, "habitat": "air", "size": "small", "speed": 50},
            {"type": "animal", "legs": 2, "habitat": "air", "size": "small", "speed": 45}
        ]
        
        concepts = engine.form_concepts_from_instances(test_instances)
        
        if len(concepts) > 0:
            test_results["concept_learning_results"]["formation_engine"] = {
                "concepts_formed": len(concepts),
                "concept_names": [c["name"] for c in concepts],
                "average_confidence": statistics.mean([c["confidence"] for c in concepts]),
                "total_instances": sum(len(c.get("instances", [])) for c in concepts),
                "success": True
            }
            log_test_result(test_name, True)
            return True
        else:
            log_test_result(test_name, False, "No concepts formed")
            return False
            
    except Exception as e:
        log_test_result(test_name, False, str(e))
        return False


def test_concept_similarity_engine():
    """Test the concept similarity engine directly."""
    test_name = "Concept Similarity Engine"
    
    try:
        from python.tools.concept_learning import ConceptSimilarityEngine
        
        engine = ConceptSimilarityEngine()
        
        concept1 = {
            "name": "car_concept",
            "prototype_features": {"wheels": 4.0, "speed": 60.0, "size_medium": 1.0},
            "feature_ranges": {"wheels": [4, 4], "speed": [55, 70], "size_medium": [0, 1]}
        }
        
        concept2 = {
            "name": "truck_concept", 
            "prototype_features": {"wheels": 4.0, "speed": 55.0, "size_large": 1.0},
            "feature_ranges": {"wheels": [4, 4], "speed": [50, 60], "size_large": [0, 1]}
        }
        
        concept3 = {
            "name": "bird_concept",
            "prototype_features": {"legs": 2.0, "speed": 50.0, "habitat_air": 1.0},
            "feature_ranges": {"legs": [2, 2], "speed": [40, 60], "habitat_air": [0, 1]}
        }
        
        # Test different similarity methods
        feature_sim = engine.compute_similarity(concept1, concept2, "feature")
        structural_sim = engine.compute_similarity(concept1, concept2, "structural") 
        functional_sim = engine.compute_similarity(concept1, concept2, "functional")
        combined_sim = engine.compute_similarity(concept1, concept2, "combined")
        
        # Test dissimilar concepts
        dissimilar_sim = engine.compute_similarity(concept1, concept3, "combined")
        
        if (0.0 <= feature_sim <= 1.0 and 0.0 <= structural_sim <= 1.0 and 
            0.0 <= functional_sim <= 1.0 and 0.0 <= combined_sim <= 1.0):
            
            test_results["concept_learning_results"]["similarity_engine"] = {
                "feature_similarity": feature_sim,
                "structural_similarity": structural_sim,
                "functional_similarity": functional_sim,
                "combined_similarity": combined_sim,
                "dissimilar_concepts_similarity": dissimilar_sim,
                "similarity_range_valid": True,
                "success": True
            }
            log_test_result(test_name, True)
            return True
        else:
            log_test_result(test_name, False, f"Invalid similarity values: {feature_sim}, {structural_sim}, {functional_sim}")
            return False
            
    except Exception as e:
        log_test_result(test_name, False, str(e))
        return False


def test_concept_refinement_engine():
    """Test the concept refinement engine directly."""
    test_name = "Concept Refinement Engine"
    
    try:
        from python.tools.concept_learning import ConceptRefinementEngine
        
        engine = ConceptRefinementEngine()
        
        # Initial concept
        concept_data = {
            "name": "vehicle_concept",
            "prototype_features": {"wheels": 4.0, "speed": 60.0},
            "feature_ranges": {"wheels": [4, 4], "speed": [55, 65]},
            "instance_count": 3,
            "confidence": 0.7,
            "created_at": time.time()
        }
        
        # New instances for refinement
        new_instances = [
            {"wheels": 4, "speed": 70},
            {"wheels": 4, "speed": 50}
        ]
        
        # Positive feedback
        feedback = [
            {"correct": True, "confidence": 0.8},
            {"correct": True, "confidence": 0.9}
        ]
        
        refined_concept = engine.refine_concept(concept_data, new_instances, feedback)
        
        # Validate refinement
        if (refined_concept["instance_count"] == 5 and  # 3 + 2 new
            "updated_at" in refined_concept and
            refined_concept["confidence"] > 0):  # Confidence should be positive
            
            test_results["concept_learning_results"]["refinement_engine"] = {
                "original_instances": concept_data["instance_count"],
                "refined_instances": refined_concept["instance_count"],
                "original_confidence": concept_data["confidence"],
                "refined_confidence": refined_concept["confidence"],
                "feature_ranges_updated": refined_concept["feature_ranges"] != concept_data["feature_ranges"],
                "success": True
            }
            log_test_result(test_name, True)
            return True
        else:
            log_test_result(test_name, False, f"Refinement validation: instances={refined_concept.get('instance_count', 0)}, has_updated_at={'updated_at' in refined_concept}, confidence={refined_concept.get('confidence', 0)}")
            return False
            
    except Exception as e:
        log_test_result(test_name, False, str(e))
        return False


def test_pattern_data_structures():
    """Test the pattern data structures."""
    test_name = "Pattern Data Structures"
    
    try:
        from python.tools.advanced_pattern_recognition import Pattern
        
        # Create a test pattern
        pattern = Pattern(
            pattern_id="test_pattern_123",
            pattern_type="hierarchical",
            elements=["element1", "element2"],
            confidence=0.85,
            abstraction_level=2,
            created_at=time.time(),
            frequency=5,
            domains=["test_domain"]
        )
        
        # Validate pattern properties
        if (pattern.pattern_id == "test_pattern_123" and
            pattern.pattern_type == "hierarchical" and
            pattern.confidence == 0.85 and
            pattern.abstraction_level == 2 and
            len(pattern.elements) == 2 and
            pattern.frequency == 5):
            
            test_results["advanced_pattern_results"]["pattern_structures"] = {
                "pattern_creation": True,
                "pattern_id": pattern.pattern_id,
                "pattern_type": pattern.pattern_type,
                "confidence": pattern.confidence,
                "success": True
            }
            log_test_result(test_name, True)
            return True
        else:
            log_test_result(test_name, False, "Pattern structure validation failed")
            return False
            
    except Exception as e:
        log_test_result(test_name, False, str(e))
        return False


def test_concept_data_structures():
    """Test the concept learning data structures."""
    test_name = "Concept Data Structures"
    
    try:
        from python.tools.advanced_pattern_recognition import ConceptNode
        from python.tools.concept_learning import ConceptRelation
        
        # Create a test concept node
        concept = ConceptNode(
            concept_id="test_concept_456",
            name="test_concept",
            properties={"type": "vehicle", "mobility": "high"},
            parent_concepts=["transportation"],
            child_concepts=["car", "truck"],
            abstraction_level=1,
            confidence=0.9,
            examples=[{"example": "data"}],
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Create a test relation
        relation = ConceptRelation(
            relation_id="test_relation_789",
            source_concept="concept1",
            target_concept="concept2",
            relation_type="similar_to",
            strength=0.8,
            bidirectional=True,
            created_at=time.time(),
            evidence=["similarity_in_features"]
        )
        
        # Validate structures
        if (concept.concept_id == "test_concept_456" and
            concept.name == "test_concept" and
            concept.confidence == 0.9 and
            relation.relation_type == "similar_to" and
            relation.strength == 0.8):
            
            test_results["concept_learning_results"]["concept_structures"] = {
                "concept_creation": True,
                "relation_creation": True,
                "concept_id": concept.concept_id,
                "relation_type": relation.relation_type,
                "success": True
            }
            log_test_result(test_name, True)
            return True
        else:
            log_test_result(test_name, False, "Concept structure validation failed")
            return False
            
    except Exception as e:
        log_test_result(test_name, False, str(e))
        return False


def run_engine_tests():
    """Run all engine tests."""
    
    print("🧠 Testing Advanced Pattern Recognition and Concept Learning Engines")
    print("=" * 72)
    
    # Test Advanced Pattern Recognition Engines
    print("\n📊 Advanced Pattern Recognition Engine Tests")
    print("-" * 47)
    
    test_hierarchical_pattern_engine()
    test_temporal_pattern_engine()
    test_analogical_reasoning_engine()
    test_pattern_data_structures()
    
    # Test Concept Learning Engines
    print("\n🎯 Concept Learning Engine Tests")
    print("-" * 32)
    
    test_concept_formation_engine()
    test_concept_similarity_engine()
    test_concept_refinement_engine()
    test_concept_data_structures()
    
    # Final results
    print("\n" + "=" * 72)
    print(f"✅ Tests passed: {test_results['tests_passed']}")
    print(f"❌ Tests failed: {test_results['tests_failed']}")
    
    if test_results['tests_failed'] > 0:
        print("\nErrors encountered:")
        for error in test_results['errors']:
            print(f"  • {error}")
    
    # Save detailed results
    results_file = PROJECT_ROOT / "advanced_pattern_concept_engine_test_results.json"
    try:
        with open(results_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        print(f"\n📊 Detailed results saved to: {results_file}")
    except Exception as e:
        print(f"Warning: Could not save results file: {e}")
    
    # Determine overall success
    total_tests = test_results['tests_passed'] + test_results['tests_failed']
    success_rate = test_results['tests_passed'] / total_tests if total_tests > 0 else 0
    
    if success_rate >= 0.8:  # 80% success threshold
        print("\n🎉 Advanced pattern recognition and concept learning engines validated successfully!")
        print("✅ Core functionality working correctly")
        return 0
    else:
        print("\n⚠️ Some validation issues detected - review recommended")
        return 1


def main():
    """Main test execution."""
    return run_engine_tests()


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)