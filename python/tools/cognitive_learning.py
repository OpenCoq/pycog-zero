"""
PyCog-Zero Cognitive Learning Tool
Implements adaptive learning and behavior modification capabilities for Agent-Zero

This tool enables agents to:
- Learn from experience and adapt behavior
- Track performance patterns and adjust strategies
- Store and apply learned behaviors across sessions
- Improve decision-making through feedback loops
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
import time
import pickle
import asyncio
import os
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import defaultdict, deque
import statistics

# Try to import OpenCog components for enhanced cognitive learning
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    OPENCOG_AVAILABLE = False

# Import ECAN coordinator for attention-based learning
try:
    from python.helpers.ecan_coordinator import (
        get_ecan_coordinator, register_tool_with_ecan, 
        request_attention_for_tool, AttentionRequest
    )
    ECAN_AVAILABLE = True
except ImportError:
    ECAN_AVAILABLE = False


@dataclass
class LearningExperience:
    """Represents a single learning experience with outcomes."""
    timestamp: float
    context: Dict[str, Any]
    action_taken: str
    outcome: Dict[str, Any]
    success_score: float  # 0.0 to 1.0
    feedback: Optional[str] = None
    learning_type: str = "experiential"  # experiential, reinforcement, observational
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class BehavioralPattern:
    """Represents a learned behavioral pattern."""
    pattern_id: str
    context_features: List[str]
    preferred_actions: List[str]
    success_rate: float
    usage_count: int
    last_updated: float
    confidence_level: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ExperienceBuffer:
    """Buffer for storing and managing learning experiences."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.experiences: deque = deque(maxlen=max_size)
        self.experience_index = {}  # For fast lookups
        
    def add_experience(self, experience: LearningExperience):
        """Add a new learning experience."""
        self.experiences.append(experience)
        
        # Index by context type for quick retrieval
        context_type = experience.context.get("type", "general")
        if context_type not in self.experience_index:
            self.experience_index[context_type] = deque(maxlen=100)
        self.experience_index[context_type].append(experience)
    
    def get_relevant_experiences(self, context: Dict[str, Any], limit: int = 10) -> List[LearningExperience]:
        """Retrieve experiences relevant to current context."""
        context_type = context.get("type", "general")
        
        if context_type in self.experience_index:
            return list(self.experience_index[context_type])[-limit:]
        return list(self.experiences)[-limit:]
    
    def get_success_patterns(self, min_success_rate: float = 0.7) -> List[Dict[str, Any]]:
        """Extract successful behavioral patterns from experiences."""
        patterns = defaultdict(list)
        
        for exp in self.experiences:
            if exp.success_score >= min_success_rate:
                key = f"{exp.context.get('type', 'general')}_{exp.action_taken}"
                patterns[key].append(exp)
        
        successful_patterns = []
        for pattern_key, experiences in patterns.items():
            if len(experiences) >= 3:  # Minimum pattern strength
                avg_success = statistics.mean([exp.success_score for exp in experiences])
                successful_patterns.append({
                    "pattern": pattern_key,
                    "success_rate": avg_success,
                    "sample_count": len(experiences),
                    "latest_experience": experiences[-1].to_dict()
                })
        
        return successful_patterns


class AdaptiveBehaviorEngine:
    """Engine for adapting agent behavior based on learning."""
    
    def __init__(self):
        self.behavioral_patterns: Dict[str, BehavioralPattern] = {}
        self.adaptation_strategies = {}
        self.performance_metrics = {
            "success_rate_trend": deque(maxlen=50),
            "adaptation_effectiveness": 0.0,
            "learning_velocity": 0.0
        }
    
    def analyze_performance_trends(self, experiences: List[LearningExperience]) -> Dict[str, Any]:
        """Analyze performance trends from recent experiences."""
        if not experiences:
            return {"trend": "insufficient_data", "recommendation": "gather_more_experience"}
        
        recent_scores = [exp.success_score for exp in experiences[-20:]]  # Last 20 experiences
        
        if len(recent_scores) >= 5:
            trend_slope = self._calculate_trend_slope(recent_scores)
            avg_score = statistics.mean(recent_scores)
            
            trend_analysis = {
                "current_performance": avg_score,
                "trend_direction": "improving" if trend_slope > 0.01 else "declining" if trend_slope < -0.01 else "stable",
                "trend_strength": abs(trend_slope),
                "consistency": 1.0 - (statistics.stdev(recent_scores) if len(recent_scores) > 1 else 0),
                "recommendation": self._generate_adaptation_recommendation(avg_score, trend_slope)
            }
        else:
            trend_analysis = {
                "current_performance": statistics.mean(recent_scores),
                "trend_direction": "insufficient_data",
                "recommendation": "continue_learning"
            }
        
        return trend_analysis
    
    def _calculate_trend_slope(self, scores: List[float]) -> float:
        """Calculate the slope of performance trend."""
        if len(scores) < 2:
            return 0.0
        
        x_values = list(range(len(scores)))
        x_mean = statistics.mean(x_values)
        y_mean = statistics.mean(scores)
        
        numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, scores))
        denominator = sum((x - x_mean) ** 2 for x in x_values)
        
        return numerator / denominator if denominator != 0 else 0.0
    
    def _generate_adaptation_recommendation(self, avg_score: float, trend_slope: float) -> str:
        """Generate adaptation recommendations based on performance."""
        if avg_score < 0.3:
            return "major_strategy_revision_needed"
        elif avg_score < 0.6:
            if trend_slope < -0.01:
                return "adjust_approach_declining_performance"
            else:
                return "incremental_improvement_focus"
        elif avg_score < 0.8:
            return "optimize_current_strategy"
        else:
            return "maintain_current_approach"
    
    def update_behavioral_pattern(self, pattern_id: str, experience: LearningExperience):
        """Update or create behavioral patterns based on experience."""
        if pattern_id not in self.behavioral_patterns:
            self.behavioral_patterns[pattern_id] = BehavioralPattern(
                pattern_id=pattern_id,
                context_features=list(experience.context.keys()),
                preferred_actions=[experience.action_taken],
                success_rate=experience.success_score,
                usage_count=1,
                last_updated=time.time(),
                confidence_level=0.5
            )
        else:
            pattern = self.behavioral_patterns[pattern_id]
            # Update success rate using exponential moving average
            pattern.success_rate = 0.8 * pattern.success_rate + 0.2 * experience.success_score
            pattern.usage_count += 1
            pattern.last_updated = time.time()
            
            # Update confidence based on usage and consistency
            pattern.confidence_level = min(0.95, 0.3 + (pattern.usage_count * 0.1) + (pattern.success_rate * 0.4))
            
            # Add new action if successful and not already present
            if experience.success_score > 0.7 and experience.action_taken not in pattern.preferred_actions:
                pattern.preferred_actions.append(experience.action_taken)


class CognitiveLearningTool(Tool):
    """Cognitive learning and adaptation tool for Agent-Zero."""
    
    def _initialize_learning_system(self):
        """Initialize the cognitive learning system."""
        if hasattr(self, '_learning_initialized'):
            return
        
        self._learning_initialized = True
        self.experience_buffer = ExperienceBuffer()
        self.behavior_engine = AdaptiveBehaviorEngine()
        self.learning_config = {
            "learning_rate": 0.1,
            "adaptation_threshold": 0.3,
            "min_experiences_for_adaptation": 5,
            "pattern_confidence_threshold": 0.6
        }
        
        # Initialize persistence
        self.learning_data_file = files.get_abs_path("memory/cognitive_learning.json")
        self.load_learning_data()
        
        # Initialize OpenCog integration if available
        self.atomspace = None
        if OPENCOG_AVAILABLE:
            try:
                self.atomspace = AtomSpace()
                initialize_opencog(self.atomspace)
                self._create_learning_atoms()
            except Exception as e:
                print(f"OpenCog learning integration warning: {e}")
        
        # Register with ECAN for attention allocation
        if ECAN_AVAILABLE:
            try:
                register_tool_with_ecan("cognitive_learning", self)
            except Exception as e:
                print(f"ECAN registration warning: {e}")
    
    def _create_learning_atoms(self):
        """Create initial learning-related atoms in AtomSpace."""
        if not self.atomspace:
            return
        
        # Core learning concepts
        learning_node = self.atomspace.add_node(types.ConceptNode, "cognitive_learning")
        adaptation_node = self.atomspace.add_node(types.ConceptNode, "behavioral_adaptation")
        experience_node = self.atomspace.add_node(types.ConceptNode, "learning_experience")
        
        # Relationships
        self.atomspace.add_link(types.InheritanceLink, [adaptation_node, learning_node])
        self.atomspace.add_link(types.InheritanceLink, [experience_node, learning_node])
    
    def _create_response(self, message: str, data: Dict[str, Any] = None) -> Response:
        """Create a Response with embedded data in the message."""
        if data:
            message += f"\\nData: {json.dumps(data)}"
        return Response(message=message, break_loop=False)
    
    async def execute(self, operation: str, **kwargs) -> Response:
        """Execute cognitive learning operations."""
        
        # Initialize if needed
        self._initialize_learning_system()
        
        if operation == "record_experience":
            return await self.record_experience(kwargs)
        elif operation == "analyze_learning":
            return await self.analyze_learning_progress(kwargs)
        elif operation == "adapt_behavior":
            return await self.adapt_behavior(kwargs)
        elif operation == "get_recommendations":
            return await self.get_adaptation_recommendations(kwargs)
        elif operation == "export_learned_patterns":
            return await self.export_learned_patterns(kwargs)
        elif operation == "import_learned_patterns":
            return await self.import_learned_patterns(kwargs)
        else:
            return self._create_response(f"Unknown cognitive learning operation: {operation}")
    
    async def record_experience(self, data: Dict[str, Any]) -> Response:
        """Record a learning experience with outcome."""
        try:
            experience = LearningExperience(
                timestamp=time.time(),
                context=data.get("context", {}),
                action_taken=data.get("action", "unknown"),
                outcome=data.get("outcome", {}),
                success_score=float(data.get("success_score", 0.5)),
                feedback=data.get("feedback"),
                learning_type=data.get("learning_type", "experiential")
            )
            
            self.experience_buffer.add_experience(experience)
            
            # Update behavioral patterns
            context_signature = self._generate_context_signature(experience.context)
            self.behavior_engine.update_behavioral_pattern(context_signature, experience)
            
            # Store learning atom if OpenCog available
            if self.atomspace:
                self._store_experience_atom(experience)
            
            # Request attention for significant learning events
            if ECAN_AVAILABLE and experience.success_score > 0.8:
                request_attention_for_tool("cognitive_learning", AttentionRequest(
                    tool_name="cognitive_learning",
                    priority=experience.success_score,
                    context="high_success_experience", 
                    concepts=["learning", "success", "experience"],
                    importance_multiplier=1.5
                ))
            
            self.save_learning_data()
            
            return self._create_response(
                f"Recorded learning experience with {experience.success_score:.2f} success score",
                {
                    "experience_id": f"exp_{int(experience.timestamp)}",
                    "success_score": experience.success_score,
                    "learning_type": experience.learning_type,
                    "total_experiences": len(self.experience_buffer.experiences)
                }
            )
            
        except Exception as e:
            return self._create_response(f"Failed to record learning experience: {str(e)}")
    
    async def analyze_learning_progress(self, data: Dict[str, Any]) -> Response:
        """Analyze learning progress and trends."""
        try:
            analysis_period = data.get("period_hours", 24)
            cutoff_time = time.time() - (analysis_period * 3600)
            
            recent_experiences = [
                exp for exp in self.experience_buffer.experiences 
                if exp.timestamp >= cutoff_time
            ]
            
            if not recent_experiences:
                return self._create_response("No recent experiences to analyze", {"recommendation": "record_more_experiences"})
            
            # Performance trend analysis
            trend_analysis = self.behavior_engine.analyze_performance_trends(recent_experiences)
            
            # Learning velocity calculation
            learning_velocity = self._calculate_learning_velocity(recent_experiences)
            
            # Pattern extraction
            successful_patterns = self.experience_buffer.get_success_patterns()
            
            analysis_report = {
                "analysis_period_hours": analysis_period,
                "total_experiences": len(recent_experiences),
                "performance_trend": trend_analysis,
                "learning_velocity": learning_velocity,
                "successful_patterns": successful_patterns[:5],  # Top 5 patterns
                "behavioral_patterns_count": len(self.behavior_engine.behavioral_patterns),
                "adaptation_readiness": self._assess_adaptation_readiness(recent_experiences)
            }
            
            return self._create_response(
                f"Learning analysis complete - {trend_analysis['trend_direction']} performance trend",
                analysis_report
            )
            
        except Exception as e:
            return self._create_response(f"Learning analysis failed: {str(e)}")
    
    async def adapt_behavior(self, data: Dict[str, Any]) -> Response:
        """Adapt agent behavior based on learning."""
        try:
            context = data.get("context", {})
            force_adaptation = data.get("force", False)
            
            # Check if adaptation is warranted
            recent_experiences = list(self.experience_buffer.experiences)[-20:]
            adaptation_readiness = self._assess_adaptation_readiness(recent_experiences)
            
            if not force_adaptation and adaptation_readiness < self.learning_config["adaptation_threshold"]:
                return self._create_response(f"Adaptation not warranted (readiness: {adaptation_readiness:.2f})", {"recommendation": "continue_current_approach"})
            
            # Find relevant behavioral patterns
            relevant_patterns = self._find_relevant_patterns(context)
            
            if not relevant_patterns:
                return self._create_response("No relevant behavioral patterns found for adaptation", {"recommendation": "gather_more_experience_in_context"})
            
            # Generate adaptation strategy
            adaptation_strategy = self._generate_adaptation_strategy(relevant_patterns, context)
            
            return Response(
                message=f"Behavioral adaptation strategy generated",
                data={
                    "strategy": adaptation_strategy,
                    "relevant_patterns": len(relevant_patterns),
                    "confidence_level": max([p.confidence_level for p in relevant_patterns]),
                    "adaptation_readiness": adaptation_readiness
                },
                break_loop=False
            )
            
        except Exception as e:
            return self._create_response(f"Behavioral adaptation failed: {str(e)}")
    
    async def get_adaptation_recommendations(self, data: Dict[str, Any]) -> Response:
        """Get recommendations for improving learning and adaptation."""
        try:
            context = data.get("context", {})
            
            recommendations = []
            
            # Analyze experience diversity
            context_types = set()
            for exp in self.experience_buffer.experiences:
                context_types.add(exp.context.get("type", "general"))
            
            if len(context_types) < 3:
                recommendations.append({
                    "category": "experience_diversity",
                    "priority": "medium",
                    "recommendation": "Explore more diverse contexts and situations",
                    "rationale": f"Only {len(context_types)} context types experienced"
                })
            
            # Analyze learning velocity
            recent_experiences = list(self.experience_buffer.experiences)[-30:]
            if recent_experiences:
                learning_velocity = self._calculate_learning_velocity(recent_experiences)
                if learning_velocity < 0.1:
                    recommendations.append({
                        "category": "learning_velocity",
                        "priority": "high",
                        "recommendation": "Increase experimentation and reflection frequency",
                        "rationale": f"Low learning velocity detected: {learning_velocity:.3f}"
                    })
            
            # Analyze pattern confidence
            low_confidence_patterns = [
                p for p in self.behavior_engine.behavioral_patterns.values()
                if p.confidence_level < 0.5
            ]
            
            if len(low_confidence_patterns) > len(self.behavior_engine.behavioral_patterns) * 0.5:
                recommendations.append({
                    "category": "pattern_confidence",
                    "priority": "medium",
                    "recommendation": "Focus on reinforcing successful behavioral patterns",
                    "rationale": f"{len(low_confidence_patterns)} patterns have low confidence"
                })
            
            return Response(
                message=f"Generated {len(recommendations)} learning recommendations",
                data={
                    "recommendations": recommendations,
                    "total_patterns": len(self.behavior_engine.behavioral_patterns),
                    "total_experiences": len(self.experience_buffer.experiences)
                },
                break_loop=False
            )
            
        except Exception as e:
            return self._create_response(f"Failed to generate recommendations: {str(e)}")
    
    async def export_learned_patterns(self, data: Dict[str, Any]) -> Response:
        """Export learned behavioral patterns for sharing or backup."""
        try:
            export_data = {
                "timestamp": time.time(),
                "behavioral_patterns": {
                    pid: pattern.to_dict() 
                    for pid, pattern in self.behavior_engine.behavioral_patterns.items()
                },
                "learning_config": self.learning_config,
                "experience_summary": {
                    "total_experiences": len(self.experience_buffer.experiences),
                    "context_types": list(set(
                        exp.context.get("type", "general") 
                        for exp in self.experience_buffer.experiences
                    )),
                    "average_success_rate": statistics.mean([
                        exp.success_score for exp in self.experience_buffer.experiences
                    ]) if self.experience_buffer.experiences else 0.0
                }
            }
            
            export_file = data.get("file_path", files.get_abs_path("memory/learned_patterns_export.json"))
            
            with open(export_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            return Response(
                message=f"Exported {len(self.behavior_engine.behavioral_patterns)} learned patterns",
                data={
                    "export_file": export_file,
                    "patterns_exported": len(self.behavior_engine.behavioral_patterns),
                    "export_timestamp": export_data["timestamp"]
                },
                break_loop=False
            )
            
        except Exception as e:
            return self._create_response(f"Pattern export failed: {str(e)}")
    
    async def import_learned_patterns(self, data: Dict[str, Any]) -> Response:
        """Import learned behavioral patterns from another agent or backup."""
        try:
            import_file = data.get("file_path")
            if not import_file or not files.exists(import_file):
                return self._create_response("Import file not specified or not found")
            
            with open(import_file, 'r') as f:
                import_data = json.load(f)
            
            # Import behavioral patterns
            imported_count = 0
            for pattern_id, pattern_data in import_data.get("behavioral_patterns", {}).items():
                if data.get("merge_mode", False):
                    # Merge with existing patterns
                    if pattern_id in self.behavior_engine.behavioral_patterns:
                        existing = self.behavior_engine.behavioral_patterns[pattern_id]
                        # Update with weighted average
                        weight = 0.7  # Favor existing patterns
                        existing.success_rate = (weight * existing.success_rate + 
                                                (1 - weight) * pattern_data["success_rate"])
                        existing.usage_count += pattern_data["usage_count"]
                        existing.confidence_level = max(existing.confidence_level, 
                                                       pattern_data["confidence_level"])
                    else:
                        self.behavior_engine.behavioral_patterns[pattern_id] = BehavioralPattern(**pattern_data)
                        imported_count += 1
                else:
                    # Direct import (replace existing)
                    self.behavior_engine.behavioral_patterns[pattern_id] = BehavioralPattern(**pattern_data)
                    imported_count += 1
            
            self.save_learning_data()
            
            return Response(
                message=f"Imported {imported_count} behavioral patterns",
                data={
                    "patterns_imported": imported_count,
                    "total_patterns": len(self.behavior_engine.behavioral_patterns),
                    "import_source": import_file,
                    "merge_mode": data.get("merge_mode", False)
                },
                break_loop=False
            )
            
        except Exception as e:
            return self._create_response(f"Pattern import failed: {str(e)}")
    
    def _generate_context_signature(self, context: Dict[str, Any]) -> str:
        """Generate a signature for context matching."""
        key_features = sorted([
            f"{k}:{str(v)[:10]}" for k, v in context.items() 
            if k in ["type", "domain", "task", "difficulty"]
        ])
        return "_".join(key_features) if key_features else "general_context"
    
    def _calculate_learning_velocity(self, experiences: List[LearningExperience]) -> float:
        """Calculate learning velocity from recent experiences."""
        if len(experiences) < 5:
            return 0.0
        
        # Split experiences into early and late periods
        mid_point = len(experiences) // 2
        early_scores = [exp.success_score for exp in experiences[:mid_point]]
        late_scores = [exp.success_score for exp in experiences[mid_point:]]
        
        early_avg = statistics.mean(early_scores)
        late_avg = statistics.mean(late_scores)
        
        # Learning velocity is improvement rate per experience
        improvement = late_avg - early_avg
        return max(0.0, improvement / mid_point)
    
    def _assess_adaptation_readiness(self, experiences: List[LearningExperience]) -> float:
        """Assess readiness for behavioral adaptation."""
        if len(experiences) < self.learning_config["min_experiences_for_adaptation"]:
            return 0.0
        
        readiness_factors = []
        
        # Factor 1: Recent performance decline
        recent_scores = [exp.success_score for exp in experiences[-10:]]
        if len(recent_scores) >= 5:
            trend_slope = self.behavior_engine._calculate_trend_slope(recent_scores)
            if trend_slope < -0.05:  # Declining performance
                readiness_factors.append(0.8)
            elif trend_slope > 0.05:  # Improving performance
                readiness_factors.append(0.2)
            else:
                readiness_factors.append(0.5)
        
        # Factor 2: Pattern confidence levels
        if self.behavior_engine.behavioral_patterns:
            avg_confidence = statistics.mean([
                p.confidence_level for p in self.behavior_engine.behavioral_patterns.values()
            ])
            readiness_factors.append(avg_confidence)
        
        # Factor 3: Experience diversity
        context_types = len(set(exp.context.get("type", "general") for exp in experiences))
        diversity_factor = min(1.0, context_types / 5.0)  # Normalize to 5 contexts
        readiness_factors.append(diversity_factor)
        
        return statistics.mean(readiness_factors) if readiness_factors else 0.0
    
    def _find_relevant_patterns(self, context: Dict[str, Any]) -> List[BehavioralPattern]:
        """Find behavioral patterns relevant to current context."""
        context_signature = self._generate_context_signature(context)
        context_features = set(context.keys())
        
        relevant_patterns = []
        for pattern in self.behavior_engine.behavioral_patterns.values():
            # Calculate relevance based on feature overlap
            pattern_features = set(pattern.context_features)
            feature_overlap = len(context_features & pattern_features) / len(context_features | pattern_features)
            
            if feature_overlap > 0.3 or pattern.pattern_id == context_signature:
                relevant_patterns.append(pattern)
        
        # Sort by confidence and success rate
        relevant_patterns.sort(key=lambda p: p.confidence_level * p.success_rate, reverse=True)
        return relevant_patterns[:5]  # Top 5 relevant patterns
    
    def _generate_adaptation_strategy(self, patterns: List[BehavioralPattern], context: Dict[str, Any]) -> Dict[str, Any]:
        """Generate adaptation strategy from behavioral patterns."""
        if not patterns:
            return {"strategy": "explore_new_approaches", "confidence": 0.0}
        
        best_pattern = patterns[0]
        
        strategy = {
            "primary_approach": best_pattern.preferred_actions[0] if best_pattern.preferred_actions else "unknown",
            "alternative_approaches": best_pattern.preferred_actions[1:3],
            "expected_success_rate": best_pattern.success_rate,
            "confidence": best_pattern.confidence_level,
            "pattern_usage": best_pattern.usage_count,
            "recommendation": "apply_learned_pattern" if best_pattern.success_rate > 0.7 else "adapt_with_caution"
        }
        
        return strategy
    
    def _store_experience_atom(self, experience: LearningExperience):
        """Store experience as atoms in OpenCog AtomSpace."""
        if not self.atomspace:
            return
        
        try:
            # Create experience node
            exp_node = self.atomspace.add_node(types.ConceptNode, f"experience_{int(experience.timestamp)}")
            
            # Create context and outcome nodes
            context_node = self.atomspace.add_node(types.ConceptNode, f"context_{experience.context.get('type', 'general')}")
            action_node = self.atomspace.add_node(types.ConceptNode, f"action_{experience.action_taken}")
            
            # Create relationships
            self.atomspace.add_link(types.EvaluationLink, [
                self.atomspace.add_node(types.PredicateNode, "has_context"),
                exp_node, context_node
            ])
            
            self.atomspace.add_link(types.EvaluationLink, [
                self.atomspace.add_node(types.PredicateNode, "performed_action"),
                exp_node, action_node
            ])
            
            # Store success score as truth value
            exp_node.tv = (experience.success_score, 0.9)
            
        except Exception as e:
            print(f"Failed to store experience atom: {e}")
    
    def load_learning_data(self):
        """Load persistent learning data."""
        try:
            if files.exists(self.learning_data_file):
                with open(self.learning_data_file, 'r') as f:
                    data = json.load(f)
                
                # Load behavioral patterns
                for pattern_id, pattern_data in data.get("behavioral_patterns", {}).items():
                    self.behavior_engine.behavioral_patterns[pattern_id] = BehavioralPattern(**pattern_data)
                
                # Load recent experiences (last 100)
                for exp_data in data.get("recent_experiences", []):
                    experience = LearningExperience(**exp_data)
                    self.experience_buffer.add_experience(experience)
                
                print(f"Loaded {len(self.behavior_engine.behavioral_patterns)} behavioral patterns and {len(self.experience_buffer.experiences)} experiences")
        except Exception as e:
            print(f"Warning: Could not load learning data - {e}")
    
    def save_learning_data(self):
        """Save persistent learning data."""
        try:
            # Prepare data for saving
            save_data = {
                "timestamp": time.time(),
                "behavioral_patterns": {
                    pid: pattern.to_dict() 
                    for pid, pattern in self.behavior_engine.behavioral_patterns.items()
                },
                "recent_experiences": [
                    exp.to_dict() for exp in list(self.experience_buffer.experiences)[-100:]  # Last 100 experiences
                ],
                "learning_config": self.learning_config
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(self.learning_data_file), exist_ok=True)
            
            # Save to file
            with open(self.learning_data_file, 'w') as f:
                json.dump(save_data, f, indent=2, default=str)
                
        except Exception as e:
            print(f"Warning: Could not save learning data - {e}")


def register():
    """Register the cognitive learning tool with Agent-Zero."""
    return CognitiveLearningTool