#!/usr/bin/env python3
"""
Comprehensive test suite for Advanced Pattern Recognition and Concept Learning.

Tests the newly implemented capabilities:
1. Advanced Pattern Recognition Tool
   - Hierarchical pattern detection
   - Temporal pattern recognition
   - Analogical reasoning
   - Pattern transfer learning
   
2. Concept Learning Tool
   - Concept formation from instances
   - Concept refinement and adaptation
   - Concept similarity computation
   - Concept relationship creation

This validates the implementation of the long-term roadmap requirement:
"Advanced pattern recognition and concept learning"
"""

import sys
import os
import json
import asyncio
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
    "integration_results": {},
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


class TestAdvancedPatternRecognition:
    """Test class for Advanced Pattern Recognition Tool."""
    
    def __init__(self):
        self.tool = None
        self.test_data_hierarchical = [
            {"elements": ["input", "processing", "output"], "context": "data_flow"},
            {"elements": ["input", "transformation", "output"], "context": "data_flow"},
            {"elements": ["perception", "cognition", "action"], "context": "cognitive_flow"},
            {"elements": ["perception", "processing", "action"], "context": "cognitive_flow"},
            {"elements": ["sensor", "processor", "actuator"], "context": "system_flow"}
        ]
        
        self.test_data_temporal = [
            {"event": "start", "timestamp": 1.0, "value": 10},
            {"event": "process", "timestamp": 2.0, "value": 20},
            {"event": "end", "timestamp": 3.0, "value": 15},
            {"event": "start", "timestamp": 4.0, "value": 12},
            {"event": "process", "timestamp": 5.0, "value": 22},
            {"event": "end", "timestamp": 6.0, "value": 18},
            {"event": "start", "timestamp": 7.0, "value": 11},
            {"event": "process", "timestamp": 8.0, "value": 21},
            {"event": "end", "timestamp": 9.0, "value": 16}
        ]
        
        self.test_domains = {
            "source_domain": {
                "domain": "solar_system",
                "entities": ["sun", "planets", "moons"],
                "relationships": [
                    {"type": "orbits", "from": "planets", "to": "sun"},
                    {"type": "orbits", "from": "moons", "to": "planets"}
                ],
                "functions": ["gravitational_attraction", "orbital_motion"]
            },
            "target_domain": {
                "domain": "atomic_structure",
                "entities": ["nucleus", "electrons", "energy_levels"],
                "relationships": [
                    {"type": "orbits", "from": "electrons", "to": "nucleus"},
                    {"type": "occupies", "from": "electrons", "to": "energy_levels"}
                ],
                "functions": ["electromagnetic_attraction", "quantum_motion"]
            }
        }
    
    async def setup(self):
        """Initialize the advanced pattern recognition tool."""
        try:
            from python.tools.advanced_pattern_recognition import AdvancedPatternRecognitionTool
            self.tool = AdvancedPatternRecognitionTool()
            return True
        except Exception as e:
            print(f"Failed to initialize AdvancedPatternRecognitionTool: {e}")
            return False
    
    async def test_hierarchical_pattern_detection(self):
        """Test hierarchical pattern detection across multiple abstraction levels."""
        test_name = "Hierarchical Pattern Detection"
        
        try:
            response = await self.tool.execute("detect_hierarchical_patterns", 
                                             data=self.test_data_hierarchical, 
                                             max_levels=3)
            
            if response and hasattr(response, 'data') and response.data:
                patterns_detected = response.data.get("patterns_detected", 0)
                abstraction_levels = response.data.get("abstraction_levels", [])
                avg_confidence = response.data.get("average_confidence", 0)
                
                # Validation criteria
                if (patterns_detected > 0 and 
                    len(abstraction_levels) > 1 and 
                    avg_confidence > 0.1):
                    
                    test_results["advanced_pattern_results"]["hierarchical_detection"] = {
                        "patterns_detected": patterns_detected,
                        "abstraction_levels": abstraction_levels,
                        "average_confidence": avg_confidence,
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Insufficient patterns: {patterns_detected}, levels: {abstraction_levels}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_temporal_pattern_detection(self):
        """Test temporal pattern detection in sequence data."""
        test_name = "Temporal Pattern Detection"
        
        try:
            response = await self.tool.execute("detect_temporal_patterns", 
                                             sequence_data=self.test_data_temporal)
            
            if response and hasattr(response, 'data') and response.data:
                patterns_detected = response.data.get("patterns_detected", 0)
                sequence_length = response.data.get("sequence_length", 0)
                pattern_frequencies = response.data.get("pattern_frequencies", [])
                
                # Validation criteria
                if (patterns_detected > 0 and 
                    sequence_length == len(self.test_data_temporal) and 
                    len(pattern_frequencies) > 0):
                    
                    test_results["advanced_pattern_results"]["temporal_detection"] = {
                        "patterns_detected": patterns_detected,
                        "sequence_length": sequence_length,
                        "pattern_frequencies": pattern_frequencies,
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Patterns: {patterns_detected}, Length: {sequence_length}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_analogical_reasoning(self):
        """Test analogical reasoning between domains."""
        test_name = "Analogical Reasoning"
        
        try:
            response = await self.tool.execute("find_analogies", 
                                             source_domain=self.test_domains["source_domain"],
                                             target_domain=self.test_domains["target_domain"])
            
            if response and hasattr(response, 'data') and response.data:
                analogies_found = response.data.get("analogies_found", 0)
                source_domain = response.data.get("source_domain", "")
                target_domain = response.data.get("target_domain", "")
                analogy_types = response.data.get("analogy_types", [])
                
                # Validation criteria
                if (analogies_found > 0 and 
                    source_domain == "solar_system" and 
                    target_domain == "atomic_structure" and 
                    len(analogy_types) > 0):
                    
                    test_results["advanced_pattern_results"]["analogical_reasoning"] = {
                        "analogies_found": analogies_found,
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "analogy_types": analogy_types,
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Analogies: {analogies_found}, Types: {analogy_types}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_pattern_querying(self):
        """Test pattern querying functionality."""
        test_name = "Pattern Querying"
        
        try:
            response = await self.tool.execute("query_patterns", 
                                             pattern_type="hierarchical",
                                             min_confidence=0.1)
            
            if response and hasattr(response, 'data') and response.data:
                patterns_found = response.data.get("patterns_found", 0)
                total_patterns = response.data.get("total_patterns", 0)
                pattern_summaries = response.data.get("pattern_summaries", [])
                
                # Validation criteria
                if patterns_found >= 0 and total_patterns >= patterns_found:
                    test_results["advanced_pattern_results"]["pattern_querying"] = {
                        "patterns_found": patterns_found,
                        "total_patterns": total_patterns,
                        "pattern_summaries_count": len(pattern_summaries),
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Found: {patterns_found}, Total: {total_patterns}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_transfer_learning(self):
        """Test pattern transfer learning between domains."""
        test_name = "Pattern Transfer Learning"
        
        try:
            response = await self.tool.execute("transfer_learning",
                                             source_domain="cognitive_flow",
                                             target_domain="data_processing",
                                             transfer_type="pattern")
            
            if response and hasattr(response, 'data') and response.data:
                transferred_count = response.data.get("transferred_count", 0)
                source_domain = response.data.get("source_domain", "")
                target_domain = response.data.get("target_domain", "")
                
                # Validation criteria (transfer count can be 0 if no patterns exist)
                if (source_domain == "cognitive_flow" and 
                    target_domain == "data_processing"):
                    
                    test_results["advanced_pattern_results"]["transfer_learning"] = {
                        "transferred_count": transferred_count,
                        "source_domain": source_domain,
                        "target_domain": target_domain,
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Domain mismatch: {source_domain} -> {target_domain}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_statistics(self):
        """Test pattern recognition statistics."""
        test_name = "Pattern Statistics"
        
        try:
            response = await self.tool.execute("get_statistics")
            
            if response and hasattr(response, 'data') and response.data:
                pattern_stats = response.data.get("pattern_statistics", {})
                total_patterns = pattern_stats.get("total_patterns", 0)
                
                test_results["advanced_pattern_results"]["statistics"] = {
                    "total_patterns": total_patterns,
                    "pattern_statistics": pattern_stats,
                    "success": True
                }
                log_test_result(test_name, True)
                return True
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False


class TestConceptLearning:
    """Test class for Concept Learning Tool."""
    
    def __init__(self):
        self.tool = None
        self.test_instances = [
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
        
        self.concept_ids = []  # Store created concept IDs for testing
    
    async def setup(self):
        """Initialize the concept learning tool."""
        try:
            from python.tools.concept_learning import ConceptLearningTool
            self.tool = ConceptLearningTool()
            return True
        except Exception as e:
            print(f"Failed to initialize ConceptLearningTool: {e}")
            return False
    
    async def test_concept_formation(self):
        """Test concept formation from instances."""
        test_name = "Concept Formation"
        
        try:
            response = await self.tool.execute("form_concepts",
                                             instances=self.test_instances,
                                             domain="transportation_biology")
            
            if response and hasattr(response, 'data') and response.data:
                concepts_created = response.data.get("concepts_created", 0)
                instances_processed = response.data.get("instances_processed", 0)
                concept_names = response.data.get("concept_names", [])
                
                # Validation criteria
                if (concepts_created > 0 and 
                    instances_processed == len(self.test_instances) and 
                    len(concept_names) == concepts_created):
                    
                    test_results["concept_learning_results"]["concept_formation"] = {
                        "concepts_created": concepts_created,
                        "instances_processed": instances_processed,
                        "concept_names": concept_names,
                        "success": True
                    }
                    
                    # Store concept names for later tests
                    self.concept_names = concept_names
                    
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Created: {concepts_created}, Processed: {instances_processed}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_concept_querying(self):
        """Test concept querying functionality."""
        test_name = "Concept Querying"
        
        try:
            response = await self.tool.execute("query_concepts",
                                             domain="transportation_biology",
                                             min_confidence=0.0)
            
            if response and hasattr(response, 'data') and response.data:
                matching_concepts = response.data.get("matching_concepts", [])
                total_concepts = response.data.get("total_concepts", 0)
                
                # Store concept IDs for later tests
                self.concept_ids = [c["concept_id"] for c in matching_concepts]
                
                # Validation criteria
                if len(matching_concepts) > 0 and total_concepts >= len(matching_concepts):
                    test_results["concept_learning_results"]["concept_querying"] = {
                        "matching_concepts": len(matching_concepts),
                        "total_concepts": total_concepts,
                        "concept_ids": self.concept_ids[:3],  # First 3 for brevity
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Matching: {len(matching_concepts)}, Total: {total_concepts}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_concept_refinement(self):
        """Test concept refinement with new instances."""
        test_name = "Concept Refinement"
        
        try:
            if not self.concept_ids:
                log_test_result(test_name, False, "No concepts available for refinement")
                return False
            
            # Use first concept for refinement
            concept_id = self.concept_ids[0]
            
            new_instances = [
                {"type": "vehicle", "wheels": 4, "engine": "hybrid", "size": "medium", "speed": 65},
                {"type": "vehicle", "wheels": 4, "engine": "electric", "size": "large", "speed": 50}
            ]
            
            feedback = [
                {"correct": True, "confidence": 0.9},
                {"correct": True, "confidence": 0.8}
            ]
            
            response = await self.tool.execute("refine_concept",
                                             concept_id=concept_id,
                                             new_instances=new_instances,
                                             feedback=feedback)
            
            if response and hasattr(response, 'data') and response.data:
                refined_concept_id = response.data.get("concept_id", "")
                new_instances_count = response.data.get("new_instances", 0)
                feedback_items = response.data.get("feedback_items", 0)
                updated_confidence = response.data.get("updated_confidence", 0)
                
                # Validation criteria
                if (refined_concept_id == concept_id and 
                    new_instances_count == len(new_instances) and 
                    feedback_items == len(feedback) and 
                    updated_confidence > 0):
                    
                    test_results["concept_learning_results"]["concept_refinement"] = {
                        "concept_id": refined_concept_id,
                        "new_instances": new_instances_count,
                        "feedback_items": feedback_items,
                        "updated_confidence": updated_confidence,
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"ID: {refined_concept_id}, Instances: {new_instances_count}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_concept_similarity(self):
        """Test concept similarity computation."""
        test_name = "Concept Similarity"
        
        try:
            if len(self.concept_ids) < 2:
                log_test_result(test_name, False, "Need at least 2 concepts for similarity test")
                return False
            
            concept1_id = self.concept_ids[0]
            concept2_id = self.concept_ids[1]
            
            response = await self.tool.execute("compute_similarity",
                                             concept1_id=concept1_id,
                                             concept2_id=concept2_id,
                                             method="combined")
            
            if response and hasattr(response, 'data') and response.data:
                similarity = response.data.get("similarity", -1)
                method = response.data.get("method", "")
                concept1_name = response.data.get("concept1_name", "")
                concept2_name = response.data.get("concept2_name", "")
                
                # Validation criteria
                if (0.0 <= similarity <= 1.0 and 
                    method == "combined" and 
                    concept1_name and concept2_name):
                    
                    test_results["concept_learning_results"]["concept_similarity"] = {
                        "similarity": similarity,
                        "method": method,
                        "concept1_name": concept1_name,
                        "concept2_name": concept2_name,
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Similarity: {similarity}, Method: {method}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_find_similar_concepts(self):
        """Test finding similar concepts."""
        test_name = "Find Similar Concepts"
        
        try:
            if not self.concept_ids:
                log_test_result(test_name, False, "No concepts available for similarity search")
                return False
            
            concept_id = self.concept_ids[0]
            
            response = await self.tool.execute("find_similar_concepts",
                                             concept_id=concept_id,
                                             similarity_threshold=0.1,
                                             max_results=5)
            
            if response and hasattr(response, 'data') and response.data:
                target_concept = response.data.get("target_concept", "")
                similar_concepts = response.data.get("similar_concepts", [])
                similarity_threshold = response.data.get("similarity_threshold", 0)
                
                # Validation criteria (similar concepts can be empty if none meet threshold)
                if (target_concept == concept_id and 
                    isinstance(similar_concepts, list) and 
                    similarity_threshold == 0.1):
                    
                    test_results["concept_learning_results"]["find_similar_concepts"] = {
                        "target_concept": target_concept,
                        "similar_concepts_count": len(similar_concepts),
                        "similarity_threshold": similarity_threshold,
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Target: {target_concept}, Threshold: {similarity_threshold}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_concept_relations(self):
        """Test creating relationships between concepts."""
        test_name = "Concept Relations"
        
        try:
            if len(self.concept_ids) < 2:
                log_test_result(test_name, False, "Need at least 2 concepts for relations test")
                return False
            
            source_id = self.concept_ids[0]
            target_id = self.concept_ids[1]
            
            response = await self.tool.execute("create_relation",
                                             source_concept_id=source_id,
                                             target_concept_id=target_id,
                                             relation_type="similar_to",
                                             strength=0.7,
                                             bidirectional=True,
                                             evidence=["test_evidence"])
            
            if response and hasattr(response, 'data') and response.data:
                relation_id = response.data.get("relation_id", "")
                relation_type = response.data.get("relation_type", "")
                strength = response.data.get("strength", 0)
                bidirectional = response.data.get("bidirectional", False)
                
                # Validation criteria
                if (relation_id and 
                    relation_type == "similar_to" and 
                    strength == 0.7 and 
                    bidirectional == True):
                    
                    test_results["concept_learning_results"]["concept_relations"] = {
                        "relation_id": relation_id[:20],  # Truncate for brevity
                        "relation_type": relation_type,
                        "strength": strength,
                        "bidirectional": bidirectional,
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Type: {relation_type}, Strength: {strength}")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_concept_explanation(self):
        """Test concept explanation functionality."""
        test_name = "Concept Explanation"
        
        try:
            if not self.concept_ids:
                log_test_result(test_name, False, "No concepts available for explanation")
                return False
            
            concept_id = self.concept_ids[0]
            
            response = await self.tool.execute("explain_concept",
                                             concept_id=concept_id)
            
            if response and hasattr(response, 'data') and response.data:
                explanation = response.data.get("explanation", {})
                
                # Check required fields in explanation
                required_fields = ["concept_name", "concept_id", "confidence", "domain"]
                has_required = all(field in explanation for field in required_fields)
                
                if has_required and explanation["concept_id"] == concept_id:
                    test_results["concept_learning_results"]["concept_explanation"] = {
                        "concept_name": explanation.get("concept_name", ""),
                        "confidence": explanation.get("confidence", 0),
                        "domain": explanation.get("domain", ""),
                        "has_features": "key_features" in explanation,
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, f"Missing fields or ID mismatch")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False
    
    async def test_concept_statistics(self):
        """Test concept learning statistics."""
        test_name = "Concept Statistics"
        
        try:
            response = await self.tool.execute("get_concept_statistics")
            
            if response and hasattr(response, 'data') and response.data:
                totals = response.data.get("totals", {})
                averages = response.data.get("averages", {})
                distributions = response.data.get("distributions", {})
                
                # Validation criteria
                required_totals = ["concepts", "instances", "relations"]
                required_averages = ["confidence", "instances_per_concept"]
                required_distributions = ["domains", "relation_types", "instance_sources"]
                
                has_totals = all(field in totals for field in required_totals)
                has_averages = all(field in averages for field in required_averages)
                has_distributions = all(field in distributions for field in required_distributions)
                
                if has_totals and has_averages and has_distributions:
                    test_results["concept_learning_results"]["concept_statistics"] = {
                        "total_concepts": totals.get("concepts", 0),
                        "total_instances": totals.get("instances", 0),
                        "average_confidence": averages.get("confidence", 0),
                        "domain_count": len(distributions.get("domains", {})),
                        "success": True
                    }
                    log_test_result(test_name, True)
                    return True
                else:
                    log_test_result(test_name, False, "Missing required fields in statistics")
                    return False
            else:
                log_test_result(test_name, False, "No response data")
                return False
                
        except Exception as e:
            log_test_result(test_name, False, str(e))
            return False


async def run_comprehensive_tests():
    """Run all tests for advanced pattern recognition and concept learning."""
    
    print("🧠 Testing Advanced Pattern Recognition and Concept Learning")
    print("=" * 65)
    
    # Test Advanced Pattern Recognition
    print("\n📊 Advanced Pattern Recognition Tests")
    print("-" * 40)
    
    pattern_test = TestAdvancedPatternRecognition()
    pattern_setup_success = await pattern_test.setup()
    
    if pattern_setup_success:
        await pattern_test.test_hierarchical_pattern_detection()
        await pattern_test.test_temporal_pattern_detection()
        await pattern_test.test_analogical_reasoning()
        await pattern_test.test_pattern_querying()
        await pattern_test.test_transfer_learning()
        await pattern_test.test_statistics()
    else:
        log_test_result("Advanced Pattern Recognition Setup", False, "Tool initialization failed")
    
    # Test Concept Learning
    print("\n🎯 Concept Learning Tests")
    print("-" * 25)
    
    concept_test = TestConceptLearning()
    concept_setup_success = await concept_test.setup()
    
    if concept_setup_success:
        await concept_test.test_concept_formation()
        await concept_test.test_concept_querying()
        await concept_test.test_concept_refinement()
        await concept_test.test_concept_similarity()
        await concept_test.test_find_similar_concepts()
        await concept_test.test_concept_relations()
        await concept_test.test_concept_explanation()
        await concept_test.test_concept_statistics()
    else:
        log_test_result("Concept Learning Setup", False, "Tool initialization failed")
    
    # Integration test
    print("\n🔗 Integration Test")
    print("-" * 18)
    
    integration_success = test_integration_capabilities(pattern_setup_success, concept_setup_success)
    test_results["integration_results"]["tools_integration"] = {
        "pattern_tool_available": pattern_setup_success,
        "concept_tool_available": concept_setup_success,
        "integration_success": integration_success
    }
    
    if integration_success:
        log_test_result("Tools Integration", True)
    else:
        log_test_result("Tools Integration", False, "One or both tools failed to initialize")
    
    # Final results
    print("\n" + "=" * 65)
    print(f"✅ Tests passed: {test_results['tests_passed']}")
    print(f"❌ Tests failed: {test_results['tests_failed']}")
    
    if test_results['tests_failed'] > 0:
        print("\nErrors encountered:")
        for error in test_results['errors']:
            print(f"  • {error}")
    
    # Save detailed results
    results_file = PROJECT_ROOT / "advanced_pattern_concept_test_results.json"
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
        print("\n🎉 Advanced pattern recognition and concept learning validated successfully!")
        print("✅ Ready for integration with existing cognitive tools")
        return 0
    else:
        print("\n⚠️ Some validation issues detected - review recommended")
        return 1


def test_integration_capabilities(pattern_available: bool, concept_available: bool) -> bool:
    """Test integration capabilities between tools."""
    return pattern_available and concept_available


def main():
    """Main test execution."""
    return asyncio.run(run_comprehensive_tests())


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)