"""
Concept Learning Tool for PyCog-Zero

This tool implements advanced concept learning capabilities:
- Hierarchical concept formation and refinement
- Multi-modal concept representation (visual, textual, relational)
- Concept similarity and clustering
- Adaptive concept boundaries and prototypes
- Cross-domain concept mapping and transfer
- Concept explanation and reasoning

Works in conjunction with Advanced Pattern Recognition Tool for comprehensive
cognitive learning capabilities.
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
import time
import asyncio
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union, Set
from dataclasses import dataclass, asdict, field
from collections import defaultdict, deque
import statistics
import math
import re
from itertools import combinations
import hashlib

# Try to import OpenCog components
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    OPENCOG_AVAILABLE = False

# Import advanced pattern recognition for integration
try:
    from python.tools.advanced_pattern_recognition import AdvancedPatternRecognitionTool, Pattern
    ADVANCED_PATTERN_AVAILABLE = True
except ImportError:
    ADVANCED_PATTERN_AVAILABLE = False


@dataclass
class ConceptPrototype:
    """Represents a concept prototype with typical features."""
    prototype_id: str
    concept_id: str
    features: Dict[str, Any]
    weight: float
    confidence: float
    created_at: float
    usage_count: int = 0
    last_used: Optional[float] = None


@dataclass
class ConceptBoundary:
    """Defines the boundaries of a concept in feature space."""
    boundary_id: str
    concept_id: str
    feature_ranges: Dict[str, Tuple[Any, Any]]  # feature_name -> (min, max)
    fuzzy_boundaries: Dict[str, float]  # feature_name -> fuzziness_factor
    confidence: float
    created_at: float
    updated_at: float


@dataclass
class ConceptRelation:
    """Represents a relationship between concepts."""
    relation_id: str
    source_concept: str
    target_concept: str
    relation_type: str  # 'is_a', 'part_of', 'similar_to', 'opposite_of', etc.
    strength: float
    bidirectional: bool
    created_at: float
    evidence: List[str] = field(default_factory=list)


@dataclass
class ConceptInstance:
    """Represents a specific instance of a concept."""
    instance_id: str
    concept_id: str
    features: Dict[str, Any]
    confidence: float
    source: str  # Where this instance came from
    created_at: float
    validated: bool = False


class ConceptFormationEngine:
    """Engine for forming concepts from data and experiences."""
    
    def __init__(self):
        self.similarity_threshold = 0.7
        self.min_instances_for_concept = 3
        self.feature_importance_weights = {}
        
    def form_concepts_from_instances(self, instances: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Form concepts by clustering similar instances."""
        if len(instances) < self.min_instances_for_concept:
            return []
        
        # Extract features from instances
        feature_vectors = []
        instance_ids = []
        
        for i, instance in enumerate(instances):
            features = self._extract_features(instance)
            if features:
                feature_vectors.append(features)
                instance_ids.append(i)
        
        if len(feature_vectors) < self.min_instances_for_concept:
            return []
        
        # Cluster similar instances
        clusters = self._cluster_instances(feature_vectors, instance_ids)
        
        # Form concepts from clusters
        concepts = []
        for cluster in clusters:
            if len(cluster['instances']) >= self.min_instances_for_concept:
                concept = self._create_concept_from_cluster(cluster, instances)
                if concept:
                    concepts.append(concept)
        
        return concepts
    
    def _extract_features(self, instance: Dict[str, Any]) -> Optional[Dict[str, float]]:
        """Extract numerical features from an instance."""
        features = {}
        
        for key, value in instance.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
            elif isinstance(value, str):
                # Simple string features (length, word count, etc.)
                features[f"{key}_length"] = float(len(value))
                features[f"{key}_word_count"] = float(len(value.split()))
            elif isinstance(value, list):
                features[f"{key}_count"] = float(len(value))
                # If list contains numbers, add statistical features
                numeric_values = [v for v in value if isinstance(v, (int, float))]
                if numeric_values:
                    features[f"{key}_mean"] = float(statistics.mean(numeric_values))
                    features[f"{key}_std"] = float(statistics.stdev(numeric_values)) if len(numeric_values) > 1 else 0.0
        
        return features if features else None
    
    def _cluster_instances(self, feature_vectors: List[Dict[str, float]], 
                         instance_ids: List[int]) -> List[Dict[str, Any]]:
        """Cluster instances based on feature similarity."""
        clusters = []
        used_indices = set()
        
        for i, vector1 in enumerate(feature_vectors):
            if i in used_indices:
                continue
            
            # Start new cluster
            cluster = {
                'centroid': vector1.copy(),
                'instances': [instance_ids[i]],
                'feature_ranges': {k: [v, v] for k, v in vector1.items()}
            }
            used_indices.add(i)
            
            # Find similar instances
            for j, vector2 in enumerate(feature_vectors[i+1:], i+1):
                if j in used_indices:
                    continue
                
                similarity = self._calculate_feature_similarity(vector1, vector2)
                if similarity >= self.similarity_threshold:
                    # Add to cluster
                    cluster['instances'].append(instance_ids[j])
                    used_indices.add(j)
                    
                    # Update centroid and ranges
                    self._update_cluster(cluster, vector2)
            
            if len(cluster['instances']) >= self.min_instances_for_concept:
                clusters.append(cluster)
        
        return clusters
    
    def _calculate_feature_similarity(self, vector1: Dict[str, float], 
                                    vector2: Dict[str, float]) -> float:
        """Calculate similarity between two feature vectors."""
        # Get common features
        common_features = set(vector1.keys()) & set(vector2.keys())
        if not common_features:
            return 0.0
        
        # Calculate weighted cosine similarity
        dot_product = 0.0
        norm1 = 0.0
        norm2 = 0.0
        
        for feature in common_features:
            weight = self.feature_importance_weights.get(feature, 1.0)
            v1 = vector1[feature] * weight
            v2 = vector2[feature] * weight
            
            dot_product += v1 * v2
            norm1 += v1 * v1
            norm2 += v2 * v2
        
        if norm1 == 0.0 or norm2 == 0.0:
            return 0.0
        
        return dot_product / (math.sqrt(norm1) * math.sqrt(norm2))
    
    def _update_cluster(self, cluster: Dict[str, Any], new_vector: Dict[str, float]):
        """Update cluster centroid and feature ranges with new instance."""
        n = len(cluster['instances'])
        
        # Update centroid (running average)
        for feature, value in new_vector.items():
            if feature in cluster['centroid']:
                cluster['centroid'][feature] = (cluster['centroid'][feature] * (n-1) + value) / n
            else:
                cluster['centroid'][feature] = value
            
            # Update feature ranges
            if feature in cluster['feature_ranges']:
                current_min, current_max = cluster['feature_ranges'][feature]
                cluster['feature_ranges'][feature] = [min(current_min, value), max(current_max, value)]
            else:
                cluster['feature_ranges'][feature] = [value, value]
    
    def _create_concept_from_cluster(self, cluster: Dict[str, Any], 
                                   instances: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        """Create a concept from a cluster of instances."""
        if not cluster['instances']:
            return None
        
        # Generate concept name based on common features
        concept_name = self._generate_concept_name(cluster, instances)
        
        # Create concept prototype from centroid
        prototype_features = cluster['centroid']
        
        # Calculate concept confidence based on cluster cohesion
        confidence = self._calculate_cluster_confidence(cluster)
        
        return {
            'name': concept_name,
            'prototype_features': prototype_features,
            'feature_ranges': cluster['feature_ranges'],
            'instance_count': len(cluster['instances']),
            'confidence': confidence,
            'instances': cluster['instances']
        }
    
    def _generate_concept_name(self, cluster: Dict[str, Any], 
                             instances: List[Dict[str, Any]]) -> str:
        """Generate a name for the concept based on common characteristics."""
        # Simple heuristic: use most common string values or dominant features
        cluster_instances = [instances[i] for i in cluster['instances']]
        
        # Look for common string values
        string_features = defaultdict(list)
        for instance in cluster_instances:
            for key, value in instance.items():
                if isinstance(value, str) and len(value) < 50:  # Reasonable string length
                    string_features[key].append(value)
        
        # Find most common string values
        for feature_name, values in string_features.items():
            value_counts = defaultdict(int)
            for value in values:
                value_counts[value] += 1
            
            # If majority have same value, use it in name
            most_common_value, count = max(value_counts.items(), key=lambda x: x[1])
            if count >= len(cluster_instances) * 0.6:  # 60% threshold
                return f"{feature_name}_{most_common_value}".replace(" ", "_")
        
        # Fallback: use dominant numerical feature
        centroid = cluster['centroid']
        if centroid:
            dominant_feature = max(centroid.items(), key=lambda x: abs(x[1]))
            return f"concept_{dominant_feature[0]}_{int(dominant_feature[1])}"
        
        # Last resort: generic name
        return f"concept_{len(cluster['instances'])}_instances"
    
    def _calculate_cluster_confidence(self, cluster: Dict[str, Any]) -> float:
        """Calculate confidence based on cluster cohesion and size."""
        instance_count = len(cluster['instances'])
        
        # Base confidence from instance count
        count_confidence = min(instance_count / 10.0, 1.0)  # Max at 10 instances
        
        # Cohesion confidence from feature ranges
        cohesion_scores = []
        for feature, (min_val, max_val) in cluster['feature_ranges'].items():
            if max_val != min_val:
                # Normalized range (smaller range = higher cohesion)
                range_size = max_val - min_val
                centroid_val = cluster['centroid'].get(feature, 0)
                if centroid_val != 0:
                    normalized_range = range_size / abs(centroid_val)
                    cohesion_score = 1.0 / (1.0 + normalized_range)
                    cohesion_scores.append(cohesion_score)
        
        cohesion_confidence = statistics.mean(cohesion_scores) if cohesion_scores else 0.5
        
        # Combine confidences
        return (count_confidence + cohesion_confidence) / 2.0


class ConceptRefinementEngine:
    """Engine for refining and adapting concepts based on new evidence."""
    
    def __init__(self):
        self.adaptation_rate = 0.1
        self.boundary_expansion_threshold = 0.8
        self.boundary_contraction_threshold = 0.3
    
    def refine_concept(self, concept_data: Dict[str, Any], 
                      new_instances: List[Dict[str, Any]], 
                      feedback: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Refine a concept based on new instances and feedback."""
        
        # Process new instances
        if new_instances:
            concept_data = self._incorporate_new_instances(concept_data, new_instances)
        
        # Process feedback if available
        if feedback:
            concept_data = self._incorporate_feedback(concept_data, feedback)
        
        # Adjust boundaries based on new evidence
        concept_data = self._adjust_boundaries(concept_data)
        
        # Update confidence based on evidence
        concept_data = self._update_confidence(concept_data)
        
        return concept_data
    
    def _incorporate_new_instances(self, concept_data: Dict[str, Any], 
                                 new_instances: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Incorporate new instances into concept."""
        current_instances = concept_data.get('instance_count', 0)
        
        # Update prototype (running average)
        prototype_features = concept_data.get('prototype_features', {})
        
        for instance in new_instances:
            instance_features = self._extract_features(instance)
            if instance_features:
                for feature, value in instance_features.items():
                    if feature in prototype_features:
                        # Running average update
                        old_value = prototype_features[feature]
                        new_value = old_value + self.adaptation_rate * (value - old_value)
                        prototype_features[feature] = new_value
                    else:
                        prototype_features[feature] = value
        
        # Update feature ranges
        feature_ranges = concept_data.get('feature_ranges', {})
        for instance in new_instances:
            instance_features = self._extract_features(instance)
            if instance_features:
                for feature, value in instance_features.items():
                    if feature in feature_ranges:
                        current_min, current_max = feature_ranges[feature]
                        feature_ranges[feature] = [min(current_min, value), max(current_max, value)]
                    else:
                        feature_ranges[feature] = [value, value]
        
        # Update counts
        concept_data['prototype_features'] = prototype_features
        concept_data['feature_ranges'] = feature_ranges
        concept_data['instance_count'] = current_instances + len(new_instances)
        concept_data['updated_at'] = time.time()
        
        return concept_data
    
    def _extract_features(self, instance: Dict[str, Any]) -> Dict[str, float]:
        """Extract features (reuse from ConceptFormationEngine)."""
        features = {}
        
        for key, value in instance.items():
            if isinstance(value, (int, float)):
                features[key] = float(value)
            elif isinstance(value, str):
                features[f"{key}_length"] = float(len(value))
                features[f"{key}_word_count"] = float(len(value.split()))
            elif isinstance(value, list):
                features[f"{key}_count"] = float(len(value))
        
        return features
    
    def _incorporate_feedback(self, concept_data: Dict[str, Any], 
                            feedback: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Incorporate feedback about concept accuracy."""
        
        positive_feedback = sum(1 for f in feedback if f.get('correct', False))
        total_feedback = len(feedback)
        
        if total_feedback > 0:
            feedback_score = positive_feedback / total_feedback
            
            # Adjust confidence based on feedback
            current_confidence = concept_data.get('confidence', 0.5)
            new_confidence = current_confidence + self.adaptation_rate * (feedback_score - current_confidence)
            concept_data['confidence'] = max(0.0, min(1.0, new_confidence))
            
            # Store feedback for future reference
            concept_data.setdefault('feedback_history', []).extend(feedback)
        
        return concept_data
    
    def _adjust_boundaries(self, concept_data: Dict[str, Any]) -> Dict[str, Any]:
        """Adjust concept boundaries based on evidence."""
        
        confidence = concept_data.get('confidence', 0.5)
        feature_ranges = concept_data.get('feature_ranges', {})
        
        # Expand boundaries for high-confidence concepts (allow more instances)
        # Contract boundaries for low-confidence concepts (be more specific)
        
        if confidence > self.boundary_expansion_threshold:
            # Expand boundaries by small amount
            expansion_factor = 1.05
            for feature, (min_val, max_val) in feature_ranges.items():
                range_size = max_val - min_val
                expansion = range_size * (expansion_factor - 1) / 2
                feature_ranges[feature] = [min_val - expansion, max_val + expansion]
        
        elif confidence < self.boundary_contraction_threshold:
            # Contract boundaries
            contraction_factor = 0.95
            for feature, (min_val, max_val) in feature_ranges.items():
                range_size = max_val - min_val
                contraction = range_size * (1 - contraction_factor) / 2
                feature_ranges[feature] = [min_val + contraction, max_val - contraction]
        
        concept_data['feature_ranges'] = feature_ranges
        return concept_data
    
    def _update_confidence(self, concept_data: Dict[str, Any]) -> Dict[str, Any]:
        """Update concept confidence based on available evidence."""
        
        instance_count = concept_data.get('instance_count', 0)
        feedback_history = concept_data.get('feedback_history', [])
        
        # Base confidence from instance count
        count_confidence = min(instance_count / 20.0, 1.0)
        
        # Feedback confidence
        if feedback_history:
            positive_feedback = sum(1 for f in feedback_history if f.get('correct', False))
            feedback_confidence = positive_feedback / len(feedback_history)
        else:
            feedback_confidence = 0.5  # Neutral when no feedback
        
        # Combined confidence
        combined_confidence = (count_confidence + feedback_confidence) / 2.0
        concept_data['confidence'] = combined_confidence
        
        return concept_data


class ConceptSimilarityEngine:
    """Engine for computing concept similarities and relationships."""
    
    def __init__(self):
        self.similarity_methods = ['feature', 'structural', 'functional']
    
    def compute_similarity(self, concept1: Dict[str, Any], concept2: Dict[str, Any], 
                         method: str = 'feature') -> float:
        """Compute similarity between two concepts."""
        
        if method == 'feature':
            return self._feature_similarity(concept1, concept2)
        elif method == 'structural':
            return self._structural_similarity(concept1, concept2)
        elif method == 'functional':
            return self._functional_similarity(concept1, concept2)
        else:
            # Combined similarity
            feature_sim = self._feature_similarity(concept1, concept2)
            structural_sim = self._structural_similarity(concept1, concept2)
            functional_sim = self._functional_similarity(concept1, concept2)
            return (feature_sim + structural_sim + functional_sim) / 3.0
    
    def _feature_similarity(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> float:
        """Compute similarity based on prototype features."""
        
        proto1 = concept1.get('prototype_features', {})
        proto2 = concept2.get('prototype_features', {})
        
        if not proto1 or not proto2:
            return 0.0
        
        # Calculate cosine similarity
        common_features = set(proto1.keys()) & set(proto2.keys())
        if not common_features:
            return 0.0
        
        dot_product = sum(proto1[f] * proto2[f] for f in common_features)
        norm1 = math.sqrt(sum(proto1[f] ** 2 for f in common_features))
        norm2 = math.sqrt(sum(proto2[f] ** 2 for f in common_features))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return dot_product / (norm1 * norm2)
    
    def _structural_similarity(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> float:
        """Compute similarity based on structural properties."""
        
        # Compare feature ranges overlap
        ranges1 = concept1.get('feature_ranges', {})
        ranges2 = concept2.get('feature_ranges', {})
        
        if not ranges1 or not ranges2:
            return 0.0
        
        common_features = set(ranges1.keys()) & set(ranges2.keys())
        if not common_features:
            return 0.0
        
        overlap_scores = []
        for feature in common_features:
            min1, max1 = ranges1[feature]
            min2, max2 = ranges2[feature]
            
            # Calculate overlap
            overlap_start = max(min1, min2)
            overlap_end = min(max1, max2)
            
            if overlap_end > overlap_start:
                overlap_size = overlap_end - overlap_start
                range1_size = max1 - min1
                range2_size = max2 - min2
                
                if range1_size > 0 and range2_size > 0:
                    overlap_score = overlap_size / max(range1_size, range2_size)
                    overlap_scores.append(overlap_score)
        
        return statistics.mean(overlap_scores) if overlap_scores else 0.0
    
    def _functional_similarity(self, concept1: Dict[str, Any], concept2: Dict[str, Any]) -> float:
        """Compute similarity based on functional properties."""
        
        # Simple heuristic based on concept names and usage patterns
        name1 = concept1.get('name', '')
        name2 = concept2.get('name', '')
        
        # Word overlap in names
        words1 = set(name1.lower().split('_'))
        words2 = set(name2.lower().split('_'))
        
        if not words1 or not words2:
            return 0.0
        
        word_overlap = len(words1 & words2) / len(words1 | words2)
        
        # Usage pattern similarity (if available)
        usage_sim = 0.0
        if 'usage_patterns' in concept1 and 'usage_patterns' in concept2:
            # Simplified usage pattern comparison
            usage_sim = 0.5  # Placeholder
        
        return (word_overlap + usage_sim) / 2.0
    
    def find_concept_clusters(self, concepts: List[Dict[str, Any]], 
                            similarity_threshold: float = 0.7) -> List[List[str]]:
        """Find clusters of similar concepts."""
        
        concept_ids = [c.get('concept_id', str(i)) for i, c in enumerate(concepts)]
        clusters = []
        used_concepts = set()
        
        for i, concept1 in enumerate(concepts):
            if concept_ids[i] in used_concepts:
                continue
            
            # Start new cluster
            cluster = [concept_ids[i]]
            used_concepts.add(concept_ids[i])
            
            # Find similar concepts
            for j, concept2 in enumerate(concepts[i+1:], i+1):
                if concept_ids[j] in used_concepts:
                    continue
                
                similarity = self.compute_similarity(concept1, concept2)
                if similarity >= similarity_threshold:
                    cluster.append(concept_ids[j])
                    used_concepts.add(concept_ids[j])
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters


class ConceptLearningTool(Tool):
    """Advanced concept learning tool for PyCog-Zero."""
    
    def __init__(self, agent, name, method, args, message, loop_data, **kwargs):
        super().__init__(agent, name, method, args, message, loop_data, **kwargs)
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize concept learning engines."""
        # Initialize engines
        self.formation_engine = ConceptFormationEngine()
        self.refinement_engine = ConceptRefinementEngine()
        self.similarity_engine = ConceptSimilarityEngine()
        
        # Storage
        self.concepts = {}  # concept_id -> concept_data
        self.concept_relations = []  # List of ConceptRelation objects
        self.concept_instances = {}  # instance_id -> ConceptInstance
        
        # Initialize persistence
        self.concepts_file = files.get_abs_path("memory/concept_learning.json")
        self.relations_file = files.get_abs_path("memory/concept_relations.json")
        self.instances_file = files.get_abs_path("memory/concept_instances.json")
        self.load_persistent_data()
        
        # Initialize OpenCog integration if available
        self.atomspace = None
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self._create_concept_atoms()
            except Exception as e:
                print(f"OpenCog concept learning integration warning: {e}")
    
    def load_persistent_data(self):
        """Load persistent concept learning data."""
        try:
            if os.path.exists(self.concepts_file):
                with open(self.concepts_file, 'r') as f:
                    self.concepts = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load concepts data: {e}")
        
        try:
            if os.path.exists(self.relations_file):
                with open(self.relations_file, 'r') as f:
                    relation_data = json.load(f)
                    self.concept_relations = [ConceptRelation(**r) for r in relation_data]
        except Exception as e:
            print(f"Warning: Could not load relations data: {e}")
        
        try:
            if os.path.exists(self.instances_file):
                with open(self.instances_file, 'r') as f:
                    instance_data = json.load(f)
                    self.concept_instances = {iid: ConceptInstance(**i) 
                                            for iid, i in instance_data.items()}
        except Exception as e:
            print(f"Warning: Could not load instances data: {e}")
    
    def save_persistent_data(self):
        """Save concept learning data to persistent storage."""
        try:
            os.makedirs(os.path.dirname(self.concepts_file), exist_ok=True)
            with open(self.concepts_file, 'w') as f:
                json.dump(self.concepts, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save concepts data: {e}")
        
        try:
            os.makedirs(os.path.dirname(self.relations_file), exist_ok=True)
            with open(self.relations_file, 'w') as f:
                json.dump([asdict(r) for r in self.concept_relations], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save relations data: {e}")
        
        try:
            os.makedirs(os.path.dirname(self.instances_file), exist_ok=True)
            with open(self.instances_file, 'w') as f:
                json.dump({iid: asdict(i) for iid, i in self.concept_instances.items()}, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save instances data: {e}")
    
    def _create_concept_atoms(self):
        """Create initial concept-related atoms in AtomSpace."""
        if not self.atomspace:
            return
        
        # Core concept learning atoms
        concept_learning_node = self.atomspace.add_node(types.ConceptNode, "concept_learning")
        concept_formation_node = self.atomspace.add_node(types.ConceptNode, "concept_formation")
        concept_refinement_node = self.atomspace.add_node(types.ConceptNode, "concept_refinement")
        concept_similarity_node = self.atomspace.add_node(types.ConceptNode, "concept_similarity")
        
        # Relationships
        self.atomspace.add_link(types.InheritanceLink, [concept_formation_node, concept_learning_node])
        self.atomspace.add_link(types.InheritanceLink, [concept_refinement_node, concept_learning_node])
        self.atomspace.add_link(types.InheritanceLink, [concept_similarity_node, concept_learning_node])

    async def execute(self, operation: str, **kwargs) -> Response:
        """Execute concept learning operations."""
        
        if operation == "form_concepts":
            return await self.form_concepts(kwargs)
        elif operation == "refine_concept":
            return await self.refine_concept(kwargs)
        elif operation == "compute_similarity":
            return await self.compute_similarity(kwargs)
        elif operation == "find_similar_concepts":
            return await self.find_similar_concepts(kwargs)
        elif operation == "create_relation":
            return await self.create_relation(kwargs)
        elif operation == "query_concepts":
            return await self.query_concepts(kwargs)
        elif operation == "explain_concept":
            return await self.explain_concept(kwargs)
        elif operation == "get_concept_statistics":
            return await self.get_concept_statistics(kwargs)
        else:
            return Response(
                message=f"Unknown concept learning operation: {operation}",
                break_loop=False
            )

    async def form_concepts(self, data: Dict[str, Any]) -> Response:
        """Form new concepts from instances."""
        instances = data.get("instances", [])
        domain = data.get("domain", "general")
        
        if not instances:
            return Response(
                message="No instances provided for concept formation",
                break_loop=False
            )
        
        # Form concepts using formation engine
        new_concepts = self.formation_engine.form_concepts_from_instances(instances)
        
        concepts_created = 0
        for concept_data in new_concepts:
            # Generate unique concept ID
            concept_id = f"concept_{hashlib.md5(concept_data['name'].encode()).hexdigest()[:8]}_{int(time.time())}"
            
            # Add metadata
            concept_data.update({
                'concept_id': concept_id,
                'domain': domain,
                'created_at': time.time(),
                'updated_at': time.time()
            })
            
            # Store concept
            self.concepts[concept_id] = concept_data
            
            # Create concept instances
            for instance_idx in concept_data.get('instances', []):
                if instance_idx < len(instances):
                    instance_data = instances[instance_idx]
                    instance_id = f"instance_{concept_id}_{instance_idx}_{int(time.time())}"
                    
                    concept_instance = ConceptInstance(
                        instance_id=instance_id,
                        concept_id=concept_id,
                        features=instance_data,
                        confidence=concept_data['confidence'],
                        source="formation",
                        created_at=time.time()
                    )
                    self.concept_instances[instance_id] = concept_instance
            
            concepts_created += 1
        
        # Save to persistence
        self.save_persistent_data()
        
        # Create OpenCog atoms if available
        if self.atomspace:
            for concept_data in new_concepts:
                try:
                    concept_atom = self.atomspace.add_node(types.ConceptNode, concept_data['name'])
                    domain_atom = self.atomspace.add_node(types.ConceptNode, domain)
                    self.atomspace.add_link(types.EvaluationLink, [
                        self.atomspace.add_node(types.PredicateNode, "belongs_to_domain"),
                        concept_atom, domain_atom
                    ])
                except Exception as e:
                    print(f"Warning: Could not create OpenCog atoms: {e}")
        
        return Response(
            message=f"Formed {concepts_created} concepts from {len(instances)} instances",
            data={
                "concepts_created": concepts_created,
                "instances_processed": len(instances),
                "domain": domain,
                "concept_names": [c['name'] for c in new_concepts]
            }
        )

    async def refine_concept(self, data: Dict[str, Any]) -> Response:
        """Refine an existing concept with new instances and feedback."""
        concept_id = data.get("concept_id", "")
        new_instances = data.get("new_instances", [])
        feedback = data.get("feedback", [])
        
        if not concept_id or concept_id not in self.concepts:
            return Response(
                message=f"Concept {concept_id} not found",
                break_loop=False
            )
        
        # Get current concept data
        concept_data = self.concepts[concept_id]
        
        # Refine using refinement engine
        refined_concept = self.refinement_engine.refine_concept(concept_data, new_instances, feedback)
        
        # Update stored concept
        self.concepts[concept_id] = refined_concept
        
        # Add new concept instances
        new_instance_count = 0
        for instance_data in new_instances:
            instance_id = f"instance_{concept_id}_{int(time.time())}_{new_instance_count}"
            
            concept_instance = ConceptInstance(
                instance_id=instance_id,
                concept_id=concept_id,
                features=instance_data,
                confidence=refined_concept['confidence'],
                source="refinement",
                created_at=time.time()
            )
            self.concept_instances[instance_id] = concept_instance
            new_instance_count += 1
        
        # Save to persistence
        self.save_persistent_data()
        
        return Response(
            message=f"Refined concept {concept_data.get('name', concept_id)} with {len(new_instances)} new instances",
            data={
                "concept_id": concept_id,
                "concept_name": concept_data.get('name', 'unnamed'),
                "new_instances": len(new_instances),
                "feedback_items": len(feedback),
                "updated_confidence": refined_concept['confidence'],
                "total_instances": refined_concept.get('instance_count', 0)
            }
        )

    async def compute_similarity(self, data: Dict[str, Any]) -> Response:
        """Compute similarity between two concepts."""
        concept1_id = data.get("concept1_id", "")
        concept2_id = data.get("concept2_id", "")
        method = data.get("method", "combined")
        
        if concept1_id not in self.concepts or concept2_id not in self.concepts:
            return Response(
                message="One or both concepts not found",
                break_loop=False
            )
        
        concept1 = self.concepts[concept1_id]
        concept2 = self.concepts[concept2_id]
        
        # Compute similarity
        similarity = self.similarity_engine.compute_similarity(concept1, concept2, method)
        
        return Response(
            message=f"Similarity between concepts: {similarity:.3f}",
            data={
                "concept1_id": concept1_id,
                "concept1_name": concept1.get('name', 'unnamed'),
                "concept2_id": concept2_id,
                "concept2_name": concept2.get('name', 'unnamed'),
                "similarity": similarity,
                "method": method
            }
        )

    async def find_similar_concepts(self, data: Dict[str, Any]) -> Response:
        """Find concepts similar to a given concept."""
        concept_id = data.get("concept_id", "")
        similarity_threshold = data.get("similarity_threshold", 0.7)
        max_results = data.get("max_results", 10)
        
        if concept_id not in self.concepts:
            return Response(
                message=f"Concept {concept_id} not found",
                break_loop=False
            )
        
        target_concept = self.concepts[concept_id]
        
        # Compute similarities with all other concepts
        similarities = []
        for other_id, other_concept in self.concepts.items():
            if other_id != concept_id:
                similarity = self.similarity_engine.compute_similarity(target_concept, other_concept)
                if similarity >= similarity_threshold:
                    similarities.append({
                        'concept_id': other_id,
                        'concept_name': other_concept.get('name', 'unnamed'),
                        'similarity': similarity
                    })
        
        # Sort by similarity and limit results
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        similarities = similarities[:max_results]
        
        return Response(
            message=f"Found {len(similarities)} similar concepts",
            data={
                "target_concept": concept_id,
                "target_name": target_concept.get('name', 'unnamed'),
                "similar_concepts": similarities,
                "similarity_threshold": similarity_threshold
            }
        )

    async def create_relation(self, data: Dict[str, Any]) -> Response:
        """Create a relationship between two concepts."""
        source_concept_id = data.get("source_concept_id", "")
        target_concept_id = data.get("target_concept_id", "")
        relation_type = data.get("relation_type", "related_to")
        strength = data.get("strength", 0.8)
        bidirectional = data.get("bidirectional", False)
        evidence = data.get("evidence", [])
        
        if (source_concept_id not in self.concepts or 
            target_concept_id not in self.concepts):
            return Response(
                message="One or both concepts not found",
                break_loop=False
            )
        
        # Create relation
        relation_id = f"relation_{source_concept_id}_{target_concept_id}_{relation_type}_{int(time.time())}"
        
        relation = ConceptRelation(
            relation_id=relation_id,
            source_concept=source_concept_id,
            target_concept=target_concept_id,
            relation_type=relation_type,
            strength=strength,
            bidirectional=bidirectional,
            created_at=time.time(),
            evidence=evidence
        )
        
        self.concept_relations.append(relation)
        
        # Save to persistence
        self.save_persistent_data()
        
        # Create OpenCog atoms if available
        if self.atomspace:
            try:
                source_name = self.concepts[source_concept_id].get('name', source_concept_id)
                target_name = self.concepts[target_concept_id].get('name', target_concept_id)
                
                source_atom = self.atomspace.add_node(types.ConceptNode, source_name)
                target_atom = self.atomspace.add_node(types.ConceptNode, target_name)
                relation_atom = self.atomspace.add_node(types.PredicateNode, relation_type)
                
                self.atomspace.add_link(types.EvaluationLink, [relation_atom, source_atom, target_atom])
                
                if bidirectional:
                    self.atomspace.add_link(types.EvaluationLink, [relation_atom, target_atom, source_atom])
            except Exception as e:
                print(f"Warning: Could not create OpenCog relation atoms: {e}")
        
        return Response(
            message=f"Created {relation_type} relation between concepts",
            data={
                "relation_id": relation_id,
                "source_concept": source_concept_id,
                "target_concept": target_concept_id,
                "relation_type": relation_type,
                "strength": strength,
                "bidirectional": bidirectional
            }
        )

    async def query_concepts(self, data: Dict[str, Any]) -> Response:
        """Query concepts by various criteria."""
        domain = data.get("domain", "")
        min_confidence = data.get("min_confidence", 0.0)
        min_instances = data.get("min_instances", 0)
        name_pattern = data.get("name_pattern", "")
        
        # Filter concepts
        matching_concepts = []
        
        for concept_id, concept_data in self.concepts.items():
            # Apply filters
            if domain and concept_data.get('domain', '') != domain:
                continue
            if concept_data.get('confidence', 0) < min_confidence:
                continue
            if concept_data.get('instance_count', 0) < min_instances:
                continue
            if name_pattern and name_pattern.lower() not in concept_data.get('name', '').lower():
                continue
            
            # Include in results
            matching_concepts.append({
                'concept_id': concept_id,
                'name': concept_data.get('name', 'unnamed'),
                'confidence': concept_data.get('confidence', 0),
                'instance_count': concept_data.get('instance_count', 0),
                'domain': concept_data.get('domain', 'unknown'),
                'created_at': concept_data.get('created_at', 0)
            })
        
        # Sort by confidence
        matching_concepts.sort(key=lambda x: x['confidence'], reverse=True)
        
        return Response(
            message=f"Found {len(matching_concepts)} concepts matching criteria",
            data={
                "matching_concepts": matching_concepts,
                "total_concepts": len(self.concepts),
                "filters_applied": {
                    "domain": domain,
                    "min_confidence": min_confidence,
                    "min_instances": min_instances,
                    "name_pattern": name_pattern
                }
            }
        )

    async def explain_concept(self, data: Dict[str, Any]) -> Response:
        """Provide explanation for a concept."""
        concept_id = data.get("concept_id", "")
        
        if concept_id not in self.concepts:
            return Response(
                message=f"Concept {concept_id} not found",
                break_loop=False
            )
        
        concept_data = self.concepts[concept_id]
        
        # Generate explanation
        explanation = {
            'concept_name': concept_data.get('name', 'unnamed'),
            'concept_id': concept_id,
            'confidence': concept_data.get('confidence', 0),
            'instance_count': concept_data.get('instance_count', 0),
            'domain': concept_data.get('domain', 'unknown'),
            'created_at': concept_data.get('created_at', 0)
        }
        
        # Add prototype features
        prototype = concept_data.get('prototype_features', {})
        if prototype:
            explanation['key_features'] = dict(list(prototype.items())[:5])  # Top 5 features
        
        # Add feature ranges
        ranges = concept_data.get('feature_ranges', {})
        if ranges:
            explanation['feature_ranges'] = dict(list(ranges.items())[:3])  # Top 3 ranges
        
        # Find related concepts
        related_concepts = []
        for relation in self.concept_relations:
            if relation.source_concept == concept_id:
                related_concepts.append({
                    'target_concept': relation.target_concept,
                    'target_name': self.concepts.get(relation.target_concept, {}).get('name', 'unknown'),
                    'relation_type': relation.relation_type,
                    'strength': relation.strength
                })
            elif relation.bidirectional and relation.target_concept == concept_id:
                related_concepts.append({
                    'target_concept': relation.source_concept,
                    'target_name': self.concepts.get(relation.source_concept, {}).get('name', 'unknown'),
                    'relation_type': relation.relation_type,
                    'strength': relation.strength
                })
        
        explanation['related_concepts'] = related_concepts[:5]  # Limit to 5
        
        # Add instances count by source
        instance_sources = defaultdict(int)
        for instance in self.concept_instances.values():
            if instance.concept_id == concept_id:
                instance_sources[instance.source] += 1
        explanation['instances_by_source'] = dict(instance_sources)
        
        return Response(
            message=f"Explanation for concept '{explanation['concept_name']}'",
            data={"explanation": explanation}
        )

    async def get_concept_statistics(self, data: Dict[str, Any]) -> Response:
        """Get statistics about learned concepts."""
        
        # Overall statistics
        total_concepts = len(self.concepts)
        total_instances = len(self.concept_instances)
        total_relations = len(self.concept_relations)
        
        # Concept statistics
        if self.concepts:
            confidences = [c.get('confidence', 0) for c in self.concepts.values()]
            instance_counts = [c.get('instance_count', 0) for c in self.concepts.values()]
            
            avg_confidence = statistics.mean(confidences)
            avg_instances = statistics.mean(instance_counts)
        else:
            avg_confidence = 0
            avg_instances = 0
        
        # Domain distribution
        domain_counts = defaultdict(int)
        for concept in self.concepts.values():
            domain = concept.get('domain', 'unknown')
            domain_counts[domain] += 1
        
        # Relation type distribution
        relation_type_counts = defaultdict(int)
        for relation in self.concept_relations:
            relation_type_counts[relation.relation_type] += 1
        
        # Instance source distribution
        instance_source_counts = defaultdict(int)
        for instance in self.concept_instances.values():
            instance_source_counts[instance.source] += 1
        
        return Response(
            message=f"Concept learning statistics: {total_concepts} concepts, {total_instances} instances, {total_relations} relations",
            data={
                "totals": {
                    "concepts": total_concepts,
                    "instances": total_instances,
                    "relations": total_relations
                },
                "averages": {
                    "confidence": avg_confidence,
                    "instances_per_concept": avg_instances
                },
                "distributions": {
                    "domains": dict(domain_counts),
                    "relation_types": dict(relation_type_counts),
                    "instance_sources": dict(instance_source_counts)
                }
            }
        )


def register():
    """Register the concept learning tool with Agent-Zero."""
    return ConceptLearningTool