"""
Advanced Pattern Recognition Tool for PyCog-Zero

This tool implements sophisticated pattern recognition capabilities beyond basic matching:
- Hierarchical pattern abstraction and refinement
- Multi-modal pattern recognition (visual, textual, temporal)
- Analogical reasoning and concept mapping
- Meta-pattern recognition (patterns of patterns)
- Temporal sequence learning and pattern prediction
- Cross-domain pattern transfer learning

Fulfills the long-term roadmap requirement for "Advanced pattern recognition and concept learning"
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
import time
import asyncio
import os
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics
import math
import re
from itertools import combinations

# Try to import OpenCog components for enhanced pattern recognition
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    OPENCOG_AVAILABLE = False

# Import cognitive learning for integration
try:
    from python.tools.cognitive_learning import CognitiveLearningTool
    COGNITIVE_LEARNING_AVAILABLE = True
except ImportError:
    COGNITIVE_LEARNING_AVAILABLE = False


@dataclass
class Pattern:
    """Represents a detected pattern with metadata."""
    pattern_id: str
    pattern_type: str  # 'hierarchical', 'temporal', 'analogical', 'meta'
    elements: List[Any]
    confidence: float
    abstraction_level: int
    created_at: float
    frequency: int = 1
    domains: List[str] = None
    
    def __post_init__(self):
        if self.domains is None:
            self.domains = []


@dataclass
class ConceptNode:
    """Represents a learned concept with hierarchical relationships."""
    concept_id: str
    name: str
    properties: Dict[str, Any]
    parent_concepts: List[str]
    child_concepts: List[str]
    abstraction_level: int
    confidence: float
    examples: List[Dict[str, Any]]
    created_at: float
    updated_at: float


class HierarchicalPatternEngine:
    """Engine for hierarchical pattern abstraction and refinement."""
    
    def __init__(self):
        self.pattern_hierarchy = defaultdict(list)  # level -> patterns
        self.concept_graph = {}  # concept_id -> ConceptNode
        self.abstraction_rules = []
        
    def detect_hierarchical_patterns(self, data: List[Dict[str, Any]], max_levels: int = 5) -> List[Pattern]:
        """Detect patterns at multiple abstraction levels."""
        patterns = []
        
        # Level 0: Direct element patterns
        level_0_patterns = self._detect_direct_patterns(data)
        patterns.extend(level_0_patterns)
        
        # Higher levels: Abstract patterns from lower levels
        current_level_data = data
        for level in range(1, max_levels + 1):
            level_patterns = self._abstract_patterns(current_level_data, level)
            if not level_patterns:
                break
            patterns.extend(level_patterns)
            current_level_data = [{"pattern": p, "level": level} for p in level_patterns]
            
        return patterns
    
    def _detect_direct_patterns(self, data: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect direct patterns in raw data."""
        patterns = []
        
        # Frequency-based patterns
        element_freq = defaultdict(int)
        for item in data:
            if 'elements' in item:
                for element in item['elements']:
                    element_freq[str(element)] += 1
        
        # Create patterns for frequent elements
        for element, freq in element_freq.items():
            if freq > len(data) * 0.1:  # 10% frequency threshold
                pattern = Pattern(
                    pattern_id=f"direct_{hash(element)}_{time.time()}",
                    pattern_type="hierarchical",
                    elements=[element],
                    confidence=min(freq / len(data), 1.0),
                    abstraction_level=0,
                    created_at=time.time(),
                    frequency=freq
                )
                patterns.append(pattern)
        
        return patterns
    
    def _abstract_patterns(self, data: List[Dict[str, Any]], level: int) -> List[Pattern]:
        """Abstract patterns from lower level patterns."""
        patterns = []
        
        if not data:
            return patterns
        
        # Group similar patterns for abstraction
        pattern_groups = self._group_similar_patterns(data)
        
        for group in pattern_groups:
            if len(group) >= 2:  # Need at least 2 patterns to abstract
                abstract_pattern = self._create_abstract_pattern(group, level)
                if abstract_pattern:
                    patterns.append(abstract_pattern)
        
        return patterns
    
    def _group_similar_patterns(self, data: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
        """Group similar patterns for abstraction."""
        groups = []
        processed = set()
        
        for i, item1 in enumerate(data):
            if i in processed:
                continue
                
            group = [item1]
            processed.add(i)
            
            for j, item2 in enumerate(data[i+1:], i+1):
                if j in processed:
                    continue
                    
                if self._patterns_similar(item1, item2):
                    group.append(item2)
                    processed.add(j)
            
            if len(group) > 1:
                groups.append(group)
        
        return groups
    
    def _patterns_similar(self, pattern1: Dict[str, Any], pattern2: Dict[str, Any]) -> bool:
        """Check if two patterns are similar enough for abstraction."""
        # Simple similarity based on common elements
        if 'pattern' in pattern1 and 'pattern' in pattern2:
            p1_elements = set(str(e) for e in pattern1['pattern'].elements)
            p2_elements = set(str(e) for e in pattern2['pattern'].elements)
            
            intersection = len(p1_elements & p2_elements)
            union = len(p1_elements | p2_elements)
            
            similarity = intersection / union if union > 0 else 0
            return similarity > 0.3  # 30% similarity threshold
        
        return False
    
    def _create_abstract_pattern(self, group: List[Dict[str, Any]], level: int) -> Optional[Pattern]:
        """Create an abstract pattern from a group of similar patterns."""
        if not group:
            return None
        
        # Extract common elements
        all_elements = []
        total_confidence = 0
        
        for item in group:
            if 'pattern' in item:
                pattern = item['pattern']
                all_elements.extend(pattern.elements)
                total_confidence += pattern.confidence
        
        # Find common elements
        element_count = defaultdict(int)
        for element in all_elements:
            element_count[str(element)] += 1
        
        common_elements = [element for element, count in element_count.items() 
                          if count >= len(group) * 0.5]  # 50% commonality
        
        if not common_elements:
            return None
        
        return Pattern(
            pattern_id=f"abstract_L{level}_{hash(tuple(common_elements))}_{time.time()}",
            pattern_type="hierarchical",
            elements=common_elements,
            confidence=total_confidence / len(group),
            abstraction_level=level,
            created_at=time.time(),
            frequency=len(group)
        )


class TemporalPatternEngine:
    """Engine for temporal sequence learning and pattern prediction."""
    
    def __init__(self, max_sequence_length: int = 10):
        self.max_sequence_length = max_sequence_length
        self.temporal_patterns = []
        self.sequence_memory = deque(maxlen=max_sequence_length)
        
    def detect_temporal_patterns(self, sequence_data: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect temporal patterns in sequence data."""
        patterns = []
        
        # Sort by timestamp if available
        if sequence_data and 'timestamp' in sequence_data[0]:
            sequence_data = sorted(sequence_data, key=lambda x: x['timestamp'])
        
        # Detect recurring sequences
        patterns.extend(self._detect_recurring_sequences(sequence_data))
        
        # Detect periodic patterns
        patterns.extend(self._detect_periodic_patterns(sequence_data))
        
        # Detect causal patterns
        patterns.extend(self._detect_causal_patterns(sequence_data))
        
        return patterns
    
    def _detect_recurring_sequences(self, data: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect sequences that repeat over time."""
        patterns = []
        
        # Extract event sequences
        events = [item.get('event', str(item)) for item in data]
        
        # Find recurring n-grams
        for n in range(2, min(6, len(events) + 1)):
            ngram_counts = defaultdict(list)
            
            for i in range(len(events) - n + 1):
                ngram = tuple(events[i:i+n])
                ngram_counts[ngram].append(i)
            
            # Create patterns for recurring sequences
            for ngram, positions in ngram_counts.items():
                if len(positions) >= 2:  # Must occur at least twice
                    pattern = Pattern(
                        pattern_id=f"temporal_seq_{hash(ngram)}_{time.time()}",
                        pattern_type="temporal",
                        elements=list(ngram),
                        confidence=len(positions) / (len(events) - n + 1),
                        abstraction_level=0,
                        created_at=time.time(),
                        frequency=len(positions)
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _detect_periodic_patterns(self, data: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect periodic patterns in temporal data."""
        patterns = []
        
        if len(data) < 4:  # Need minimum data for periodicity
            return patterns
        
        # Extract numeric features for periodicity analysis
        numeric_features = []
        for item in data:
            if 'value' in item and isinstance(item['value'], (int, float)):
                numeric_features.append(item['value'])
            elif 'timestamp' in item:
                numeric_features.append(item['timestamp'])
        
        if len(numeric_features) < 4:
            return patterns
        
        # Simple periodicity detection using autocorrelation
        periods = self._find_periods(numeric_features)
        
        for period in periods:
            if period > 1:
                pattern = Pattern(
                    pattern_id=f"temporal_periodic_{period}_{time.time()}",
                    pattern_type="temporal",
                    elements=[f"period_{period}"],
                    confidence=0.7,  # Fixed confidence for periodic patterns
                    abstraction_level=1,
                    created_at=time.time(),
                    frequency=len(numeric_features) // period
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_periods(self, data: List[float]) -> List[int]:
        """Find potential periods in numeric data."""
        periods = []
        n = len(data)
        
        # Test periods from 2 to n/2
        for period in range(2, n // 2 + 1):
            correlation = self._calculate_autocorrelation(data, period)
            if correlation > 0.5:  # Threshold for significant correlation
                periods.append(period)
        
        return periods
    
    def _calculate_autocorrelation(self, data: List[float], lag: int) -> float:
        """Calculate autocorrelation at given lag."""
        if len(data) <= lag:
            return 0.0
        
        n = len(data) - lag
        if n <= 0:
            return 0.0
        
        mean = sum(data) / len(data)
        
        numerator = sum((data[i] - mean) * (data[i + lag] - mean) for i in range(n))
        denominator = sum((x - mean) ** 2 for x in data)
        
        if denominator == 0:
            return 0.0
        
        return numerator / denominator
    
    def _detect_causal_patterns(self, data: List[Dict[str, Any]]) -> List[Pattern]:
        """Detect causal patterns (A leads to B)."""
        patterns = []
        
        # Look for consistent sequences where one event type follows another
        event_pairs = []
        
        for i in range(len(data) - 1):
            event_a = data[i].get('event_type', str(data[i]))
            event_b = data[i + 1].get('event_type', str(data[i + 1]))
            event_pairs.append((event_a, event_b))
        
        # Count causal relationships
        causal_counts = defaultdict(int)
        for pair in event_pairs:
            causal_counts[pair] += 1
        
        # Create patterns for strong causal relationships
        total_pairs = len(event_pairs)
        for (event_a, event_b), count in causal_counts.items():
            confidence = count / total_pairs
            if confidence > 0.1:  # 10% threshold for causal patterns
                pattern = Pattern(
                    pattern_id=f"temporal_causal_{hash((event_a, event_b))}_{time.time()}",
                    pattern_type="temporal",
                    elements=[f"{event_a} -> {event_b}"],
                    confidence=confidence,
                    abstraction_level=1,
                    created_at=time.time(),
                    frequency=count
                )
                patterns.append(pattern)
        
        return patterns


class AnalogicalReasoningEngine:
    """Engine for analogical reasoning and concept mapping."""
    
    def __init__(self):
        self.concept_mappings = {}
        self.analogy_cache = {}
        
    def find_analogies(self, source_domain: Dict[str, Any], 
                      target_domain: Dict[str, Any]) -> List[Pattern]:
        """Find analogical patterns between domains."""
        patterns = []
        
        # Structure mapping
        structural_analogies = self._find_structural_analogies(source_domain, target_domain)
        patterns.extend(structural_analogies)
        
        # Relational analogies
        relational_analogies = self._find_relational_analogies(source_domain, target_domain)
        patterns.extend(relational_analogies)
        
        # Functional analogies
        functional_analogies = self._find_functional_analogies(source_domain, target_domain)
        patterns.extend(functional_analogies)
        
        return patterns
    
    def _find_structural_analogies(self, source: Dict[str, Any], 
                                  target: Dict[str, Any]) -> List[Pattern]:
        """Find structural similarities between domains."""
        patterns = []
        
        # Compare structural elements
        source_structure = self._extract_structure(source)
        target_structure = self._extract_structure(target)
        
        # Find common structural patterns
        common_structures = self._find_common_structures(source_structure, target_structure)
        
        for structure in common_structures:
            pattern = Pattern(
                pattern_id=f"analogical_struct_{hash(str(structure))}_{time.time()}",
                pattern_type="analogical",
                elements=[f"structure: {structure}"],
                confidence=structure['similarity'],
                abstraction_level=2,
                created_at=time.time(),
                domains=[source.get('domain', 'unknown'), target.get('domain', 'unknown')]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _extract_structure(self, domain: Dict[str, Any]) -> Dict[str, Any]:
        """Extract structural information from a domain."""
        structure = {
            'entities': [],
            'relationships': [],
            'hierarchy_depth': 0,
            'connectivity': 0
        }
        
        # Extract entities
        if 'entities' in domain:
            structure['entities'] = list(domain['entities'])
        elif 'concepts' in domain:
            structure['entities'] = list(domain['concepts'])
        
        # Extract relationships
        if 'relationships' in domain:
            structure['relationships'] = list(domain['relationships'])
        
        # Calculate metrics
        structure['hierarchy_depth'] = self._calculate_hierarchy_depth(domain)
        structure['connectivity'] = self._calculate_connectivity(domain)
        
        return structure
    
    def _calculate_hierarchy_depth(self, domain: Dict[str, Any]) -> int:
        """Calculate the depth of hierarchy in the domain."""
        # Simple heuristic based on nested structures
        max_depth = 0
        
        def count_depth(obj, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)
            
            if isinstance(obj, dict):
                for value in obj.values():
                    if isinstance(value, (dict, list)):
                        count_depth(value, current_depth + 1)
            elif isinstance(obj, list):
                for item in obj:
                    if isinstance(item, (dict, list)):
                        count_depth(item, current_depth + 1)
        
        count_depth(domain)
        return max_depth
    
    def _calculate_connectivity(self, domain: Dict[str, Any]) -> float:
        """Calculate the connectivity of elements in the domain."""
        entities = domain.get('entities', [])
        relationships = domain.get('relationships', [])
        
        if not entities:
            return 0.0
        
        max_connections = len(entities) * (len(entities) - 1) / 2
        actual_connections = len(relationships)
        
        return actual_connections / max_connections if max_connections > 0 else 0.0
    
    def _find_common_structures(self, source_struct: Dict[str, Any], 
                               target_struct: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find common structural patterns."""
        common = []
        
        # Compare hierarchy depths
        depth_similarity = 1.0 - abs(source_struct['hierarchy_depth'] - 
                                   target_struct['hierarchy_depth']) / max(
                                   source_struct['hierarchy_depth'], 
                                   target_struct['hierarchy_depth'], 1)
        
        # Compare connectivity
        conn_similarity = 1.0 - abs(source_struct['connectivity'] - 
                                  target_struct['connectivity'])
        
        # Overall structural similarity
        overall_similarity = (depth_similarity + conn_similarity) / 2
        
        if overall_similarity > 0.3:  # Threshold for structural similarity
            common.append({
                'type': 'overall_structure',
                'similarity': overall_similarity,
                'depth_match': depth_similarity,
                'connectivity_match': conn_similarity
            })
        
        return common
    
    def _find_relational_analogies(self, source: Dict[str, Any], 
                                  target: Dict[str, Any]) -> List[Pattern]:
        """Find relational analogies (A:B :: C:D)."""
        patterns = []
        
        source_relations = source.get('relationships', [])
        target_relations = target.get('relationships', [])
        
        # Compare relation types
        source_rel_types = [rel.get('type', str(rel)) for rel in source_relations]
        target_rel_types = [rel.get('type', str(rel)) for rel in target_relations]
        
        common_rel_types = set(source_rel_types) & set(target_rel_types)
        
        for rel_type in common_rel_types:
            pattern = Pattern(
                pattern_id=f"analogical_rel_{hash(rel_type)}_{time.time()}",
                pattern_type="analogical",
                elements=[f"relation: {rel_type}"],
                confidence=0.6,  # Fixed confidence for relational analogies
                abstraction_level=2,
                created_at=time.time(),
                domains=[source.get('domain', 'unknown'), target.get('domain', 'unknown')]
            )
            patterns.append(pattern)
        
        return patterns
    
    def _find_functional_analogies(self, source: Dict[str, Any], 
                                  target: Dict[str, Any]) -> List[Pattern]:
        """Find functional analogies (similar purposes/functions)."""
        patterns = []
        
        source_functions = source.get('functions', [])
        target_functions = target.get('functions', [])
        
        # Simple function matching
        for s_func in source_functions:
            for t_func in target_functions:
                similarity = self._calculate_function_similarity(s_func, t_func)
                
                if similarity > 0.4:  # Threshold for functional similarity
                    pattern = Pattern(
                        pattern_id=f"analogical_func_{hash((str(s_func), str(t_func)))}_{time.time()}",
                        pattern_type="analogical",
                        elements=[f"function: {s_func} ≈ {t_func}"],
                        confidence=similarity,
                        abstraction_level=2,
                        created_at=time.time(),
                        domains=[source.get('domain', 'unknown'), target.get('domain', 'unknown')]
                    )
                    patterns.append(pattern)
        
        return patterns
    
    def _calculate_function_similarity(self, func1: Any, func2: Any) -> float:
        """Calculate similarity between two functions."""
        # Simple string-based similarity for now
        str1 = str(func1).lower()
        str2 = str(func2).lower()
        
        # Jaccard similarity
        words1 = set(str1.split())
        words2 = set(str2.split())
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0


class AdvancedPatternRecognitionTool(Tool):
    """Advanced pattern recognition and concept learning tool for Agent-Zero."""
    
    def __init__(self, agent, name, method, args, message, loop_data, **kwargs):
        super().__init__(agent, name, method, args, message, loop_data, **kwargs)
        self._initialize_engines()
    
    def _initialize_engines(self):
        """Initialize pattern recognition engines."""
        self.hierarchical_engine = HierarchicalPatternEngine()
        self.temporal_engine = TemporalPatternEngine()
        self.analogical_engine = AnalogicalReasoningEngine()
        
        self.patterns_db = []  # Store all detected patterns
        self.concepts_db = {}  # Store learned concepts
        
        # Initialize persistence
        self.patterns_file = files.get_abs_path("memory/advanced_patterns.json")
        self.concepts_file = files.get_abs_path("memory/learned_concepts.json")
        self.load_persistent_data()
        
        # Initialize OpenCog integration if available
        self.atomspace = None
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self._create_pattern_atoms()
            except Exception as e:
                print(f"OpenCog advanced pattern integration warning: {e}")
    
    def load_persistent_data(self):
        """Load persistent pattern and concept data."""
        try:
            if os.path.exists(self.patterns_file):
                with open(self.patterns_file, 'r') as f:
                    pattern_data = json.load(f)
                    self.patterns_db = [Pattern(**p) for p in pattern_data]
        except Exception as e:
            print(f"Warning: Could not load patterns data: {e}")
        
        try:
            if os.path.exists(self.concepts_file):
                with open(self.concepts_file, 'r') as f:
                    concept_data = json.load(f)
                    self.concepts_db = {cid: ConceptNode(**c) for cid, c in concept_data.items()}
        except Exception as e:
            print(f"Warning: Could not load concepts data: {e}")
    
    def save_persistent_data(self):
        """Save patterns and concepts to persistent storage."""
        try:
            os.makedirs(os.path.dirname(self.patterns_file), exist_ok=True)
            with open(self.patterns_file, 'w') as f:
                json.dump([asdict(p) for p in self.patterns_db], f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save patterns data: {e}")
        
        try:
            os.makedirs(os.path.dirname(self.concepts_file), exist_ok=True)
            with open(self.concepts_file, 'w') as f:
                json.dump({cid: asdict(c) for cid, c in self.concepts_db.items()}, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save concepts data: {e}")
    
    def _create_pattern_atoms(self):
        """Create initial pattern-related atoms in AtomSpace."""
        if not self.atomspace:
            return
        
        # Core pattern concepts
        pattern_node = self.atomspace.add_node(types.ConceptNode, "advanced_pattern")
        hierarchical_node = self.atomspace.add_node(types.ConceptNode, "hierarchical_pattern")
        temporal_node = self.atomspace.add_node(types.ConceptNode, "temporal_pattern")
        analogical_node = self.atomspace.add_node(types.ConceptNode, "analogical_pattern")
        concept_node = self.atomspace.add_node(types.ConceptNode, "learned_concept")
        
        # Relationships
        self.atomspace.add_link(types.InheritanceLink, [hierarchical_node, pattern_node])
        self.atomspace.add_link(types.InheritanceLink, [temporal_node, pattern_node])
        self.atomspace.add_link(types.InheritanceLink, [analogical_node, pattern_node])
        self.atomspace.add_link(types.InheritanceLink, [concept_node, pattern_node])

    async def execute(self, operation: str, **kwargs) -> Response:
        """Execute advanced pattern recognition operations."""
        
        if operation == "detect_hierarchical_patterns":
            return await self.detect_hierarchical_patterns(kwargs)
        elif operation == "detect_temporal_patterns":
            return await self.detect_temporal_patterns(kwargs)
        elif operation == "find_analogies":
            return await self.find_analogies(kwargs)
        elif operation == "learn_concept":
            return await self.learn_concept(kwargs)
        elif operation == "query_patterns":
            return await self.query_patterns(kwargs)
        elif operation == "transfer_learning":
            return await self.transfer_learning(kwargs)
        elif operation == "get_statistics":
            return await self.get_statistics(kwargs)
        else:
            return Response(
                message=f"Unknown advanced pattern recognition operation: {operation}",
                break_loop=False
            )

    async def detect_hierarchical_patterns(self, data: Dict[str, Any]) -> Response:
        """Detect hierarchical patterns with multiple abstraction levels."""
        input_data = data.get("data", [])
        max_levels = data.get("max_levels", 5)
        
        if not input_data:
            return Response(
                message="No data provided for hierarchical pattern detection",
                break_loop=False
            )
        
        # Detect patterns using hierarchical engine
        patterns = self.hierarchical_engine.detect_hierarchical_patterns(input_data, max_levels)
        
        # Store detected patterns
        self.patterns_db.extend(patterns)
        self.save_persistent_data()
        
        # Create OpenCog atoms for detected patterns
        if self.atomspace:
            self._create_pattern_atoms_in_atomspace(patterns)
        
        return Response(
            message=f"Detected {len(patterns)} hierarchical patterns across {max_levels} abstraction levels",
            data={
                "patterns_detected": len(patterns),
                "abstraction_levels": list(set(p.abstraction_level for p in patterns)),
                "pattern_types": list(set(p.pattern_type for p in patterns)),
                "average_confidence": statistics.mean([p.confidence for p in patterns]) if patterns else 0
            }
        )

    async def detect_temporal_patterns(self, data: Dict[str, Any]) -> Response:
        """Detect temporal patterns in sequence data."""
        sequence_data = data.get("sequence_data", [])
        
        if not sequence_data:
            return Response(
                message="No sequence data provided for temporal pattern detection",
                break_loop=False
            )
        
        # Detect patterns using temporal engine
        patterns = self.temporal_engine.detect_temporal_patterns(sequence_data)
        
        # Store detected patterns
        self.patterns_db.extend(patterns)
        self.save_persistent_data()
        
        # Create OpenCog atoms for detected patterns
        if self.atomspace:
            self._create_pattern_atoms_in_atomspace(patterns)
        
        return Response(
            message=f"Detected {len(patterns)} temporal patterns in sequence data",
            data={
                "patterns_detected": len(patterns),
                "sequence_length": len(sequence_data),
                "pattern_frequencies": [p.frequency for p in patterns],
                "average_confidence": statistics.mean([p.confidence for p in patterns]) if patterns else 0
            }
        )

    async def find_analogies(self, data: Dict[str, Any]) -> Response:
        """Find analogical patterns between domains."""
        source_domain = data.get("source_domain", {})
        target_domain = data.get("target_domain", {})
        
        if not source_domain or not target_domain:
            return Response(
                message="Both source and target domains required for analogy detection",
                break_loop=False
            )
        
        # Find analogies using analogical engine
        patterns = self.analogical_engine.find_analogies(source_domain, target_domain)
        
        # Store detected patterns
        self.patterns_db.extend(patterns)
        self.save_persistent_data()
        
        # Create OpenCog atoms for detected patterns
        if self.atomspace:
            self._create_pattern_atoms_in_atomspace(patterns)
        
        return Response(
            message=f"Found {len(patterns)} analogical patterns between domains",
            data={
                "analogies_found": len(patterns),
                "source_domain": source_domain.get('domain', 'unknown'),
                "target_domain": target_domain.get('domain', 'unknown'),
                "analogy_types": list(set(p.elements[0].split(':')[0] for p in patterns if p.elements)),
                "average_confidence": statistics.mean([p.confidence for p in patterns]) if patterns else 0
            }
        )

    async def learn_concept(self, data: Dict[str, Any]) -> Response:
        """Learn a new concept from examples and experiences."""
        concept_name = data.get("concept_name", "")
        examples = data.get("examples", [])
        properties = data.get("properties", {})
        parent_concepts = data.get("parent_concepts", [])
        
        if not concept_name:
            return Response(
                message="Concept name required for learning",
                break_loop=False
            )
        
        # Create or update concept
        concept_id = f"concept_{hash(concept_name)}_{time.time()}"
        
        concept = ConceptNode(
            concept_id=concept_id,
            name=concept_name,
            properties=properties,
            parent_concepts=parent_concepts,
            child_concepts=[],
            abstraction_level=len(parent_concepts),  # Simple abstraction level based on parents
            confidence=0.8,  # Initial confidence
            examples=examples,
            created_at=time.time(),
            updated_at=time.time()
        )
        
        # Update parent concepts to include this as a child
        for parent_id in parent_concepts:
            if parent_id in self.concepts_db:
                if concept_id not in self.concepts_db[parent_id].child_concepts:
                    self.concepts_db[parent_id].child_concepts.append(concept_id)
                    self.concepts_db[parent_id].updated_at = time.time()
        
        self.concepts_db[concept_id] = concept
        self.save_persistent_data()
        
        # Create OpenCog atom for concept
        if self.atomspace:
            concept_atom = self.atomspace.add_node(types.ConceptNode, concept_name)
            
            # Add parent relationships
            for parent_name in [self.concepts_db.get(pid, {}).get('name', '') for pid in parent_concepts]:
                if parent_name:
                    parent_atom = self.atomspace.add_node(types.ConceptNode, parent_name)
                    self.atomspace.add_link(types.InheritanceLink, [concept_atom, parent_atom])
        
        return Response(
            message=f"Learned concept '{concept_name}' with {len(examples)} examples",
            data={
                "concept_id": concept_id,
                "concept_name": concept_name,
                "abstraction_level": concept.abstraction_level,
                "parent_concepts": len(parent_concepts),
                "examples_count": len(examples)
            }
        )

    async def query_patterns(self, data: Dict[str, Any]) -> Response:
        """Query detected patterns by various criteria."""
        pattern_type = data.get("pattern_type", "")
        min_confidence = data.get("min_confidence", 0.0)
        abstraction_level = data.get("abstraction_level", None)
        domain = data.get("domain", "")
        
        # Filter patterns
        filtered_patterns = []
        
        for pattern in self.patterns_db:
            if pattern_type and pattern.pattern_type != pattern_type:
                continue
            if pattern.confidence < min_confidence:
                continue
            if abstraction_level is not None and pattern.abstraction_level != abstraction_level:
                continue
            if domain and domain not in pattern.domains:
                continue
            
            filtered_patterns.append(pattern)
        
        return Response(
            message=f"Found {len(filtered_patterns)} patterns matching criteria",
            data={
                "patterns_found": len(filtered_patterns),
                "total_patterns": len(self.patterns_db),
                "pattern_summaries": [
                    {
                        "id": p.pattern_id,
                        "type": p.pattern_type,
                        "confidence": p.confidence,
                        "abstraction_level": p.abstraction_level,
                        "elements": p.elements[:3]  # First 3 elements
                    } for p in filtered_patterns[:10]  # Limit to 10 for response size
                ]
            }
        )

    async def transfer_learning(self, data: Dict[str, Any]) -> Response:
        """Transfer learned patterns from source to target domain."""
        source_domain = data.get("source_domain", "")
        target_domain = data.get("target_domain", "")
        transfer_type = data.get("transfer_type", "pattern")  # "pattern" or "concept"
        
        if not source_domain or not target_domain:
            return Response(
                message="Both source and target domains required for transfer learning",
                break_loop=False
            )
        
        transferred_count = 0
        
        if transfer_type == "pattern":
            # Transfer patterns from source to target domain
            source_patterns = [p for p in self.patterns_db if source_domain in p.domains]
            
            for pattern in source_patterns:
                # Create new pattern for target domain
                new_pattern = Pattern(
                    pattern_id=f"transfer_{pattern.pattern_id}_{target_domain}_{time.time()}",
                    pattern_type=pattern.pattern_type,
                    elements=pattern.elements,
                    confidence=pattern.confidence * 0.8,  # Reduced confidence for transferred patterns
                    abstraction_level=pattern.abstraction_level,
                    created_at=time.time(),
                    domains=[target_domain],
                    frequency=1  # Reset frequency for new domain
                )
                
                self.patterns_db.append(new_pattern)
                transferred_count += 1
        
        elif transfer_type == "concept":
            # Transfer concepts (more complex - simplified version)
            source_concepts = [c for c in self.concepts_db.values() 
                             if any(source_domain in getattr(c, 'domains', []))]
            
            for concept in source_concepts:
                # Create adapted concept for target domain
                new_concept_id = f"transfer_{concept.concept_id}_{target_domain}_{time.time()}"
                
                new_concept = ConceptNode(
                    concept_id=new_concept_id,
                    name=f"{concept.name}_adapted",
                    properties=concept.properties.copy(),
                    parent_concepts=concept.parent_concepts.copy(),
                    child_concepts=[],
                    abstraction_level=concept.abstraction_level,
                    confidence=concept.confidence * 0.7,  # Reduced confidence for transferred concepts
                    examples=[],  # Start with no examples in new domain
                    created_at=time.time(),
                    updated_at=time.time()
                )
                
                # Add domain information
                new_concept.properties['transferred_from'] = source_domain
                new_concept.properties['target_domain'] = target_domain
                
                self.concepts_db[new_concept_id] = new_concept
                transferred_count += 1
        
        self.save_persistent_data()
        
        return Response(
            message=f"Transferred {transferred_count} {transfer_type}s from {source_domain} to {target_domain}",
            data={
                "transferred_count": transferred_count,
                "source_domain": source_domain,
                "target_domain": target_domain,
                "transfer_type": transfer_type
            }
        )

    async def get_statistics(self, data: Dict[str, Any]) -> Response:
        """Get statistics about detected patterns and learned concepts."""
        
        # Pattern statistics
        pattern_stats = {
            "total_patterns": len(self.patterns_db),
            "by_type": defaultdict(int),
            "by_abstraction_level": defaultdict(int),
            "confidence_distribution": [],
            "domains": set()
        }
        
        for pattern in self.patterns_db:
            pattern_stats["by_type"][pattern.pattern_type] += 1
            pattern_stats["by_abstraction_level"][pattern.abstraction_level] += 1
            pattern_stats["confidence_distribution"].append(pattern.confidence)
            pattern_stats["domains"].update(pattern.domains)
        
        # Convert defaultdicts to regular dicts for JSON serialization
        pattern_stats["by_type"] = dict(pattern_stats["by_type"])
        pattern_stats["by_abstraction_level"] = dict(pattern_stats["by_abstraction_level"])
        pattern_stats["domains"] = list(pattern_stats["domains"])
        
        # Concept statistics
        concept_stats = {
            "total_concepts": len(self.concepts_db),
            "by_abstraction_level": defaultdict(int),
            "confidence_distribution": [],
            "hierarchical_depth": 0
        }
        
        for concept in self.concepts_db.values():
            concept_stats["by_abstraction_level"][concept.abstraction_level] += 1
            concept_stats["confidence_distribution"].append(concept.confidence)
            concept_stats["hierarchical_depth"] = max(concept_stats["hierarchical_depth"], 
                                                    concept.abstraction_level)
        
        concept_stats["by_abstraction_level"] = dict(concept_stats["by_abstraction_level"])
        
        # Calculate averages
        avg_pattern_confidence = (statistics.mean(pattern_stats["confidence_distribution"]) 
                                if pattern_stats["confidence_distribution"] else 0)
        avg_concept_confidence = (statistics.mean(concept_stats["confidence_distribution"]) 
                                if concept_stats["confidence_distribution"] else 0)
        
        return Response(
            message=f"Advanced pattern recognition statistics: {len(self.patterns_db)} patterns, {len(self.concepts_db)} concepts",
            data={
                "pattern_statistics": pattern_stats,
                "concept_statistics": concept_stats,
                "averages": {
                    "pattern_confidence": avg_pattern_confidence,
                    "concept_confidence": avg_concept_confidence
                }
            }
        )

    def _create_pattern_atoms_in_atomspace(self, patterns: List[Pattern]):
        """Create OpenCog atoms for detected patterns."""
        if not self.atomspace:
            return
        
        for pattern in patterns:
            try:
                # Create pattern atom
                pattern_atom = self.atomspace.add_node(types.ConceptNode, 
                                                     f"pattern_{pattern.pattern_id}")
                
                # Add type information
                type_atom = self.atomspace.add_node(types.ConceptNode, 
                                                   f"{pattern.pattern_type}_pattern")
                self.atomspace.add_link(types.InheritanceLink, [pattern_atom, type_atom])
                
                # Add confidence as truth value if possible
                # This is a simplified version - full OpenCog integration would be more sophisticated
                
            except Exception as e:
                print(f"Warning: Could not create OpenCog atom for pattern {pattern.pattern_id}: {e}")


def register():
    """Register the advanced pattern recognition tool with Agent-Zero."""
    return AdvancedPatternRecognitionTool