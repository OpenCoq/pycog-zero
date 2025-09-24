#!/usr/bin/env python3
"""
Advanced PLN Reasoning Examples with Agent-Zero Integration
===========================================================

This module provides advanced reasoning examples that demonstrate the integration
of Probabilistic Logic Networks (PLN) with Agent-Zero framework capabilities.

Part of Issue #55 - Advanced Learning Systems (Phase 4):
Create advanced reasoning examples using PLN and Agent-Zero.

These examples showcase practical applications of probabilistic logical inference
in realistic AI agent scenarios, demonstrating both OpenCog and fallback implementations.
"""

import sys
import os
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from datetime import datetime
import json

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

class AdvancedPLNReasoningExamples:
    """
    Advanced PLN reasoning examples demonstrating complex logical inference patterns
    integrated with Agent-Zero framework concepts.
    """
    
    def __init__(self):
        """Initialize the advanced PLN reasoning examples system."""
        self.examples_run = 0
        self.successful_inferences = 0
        self.total_reasoning_steps = 0
        self.start_time = datetime.now()
        
        # Initialize PLN components (with graceful fallback)
        try:
            self._initialize_pln_components()
        except Exception as e:
            print(f"⚠️  PLN components not available, using fallback: {e}")
            self.pln_available = False
    
    def _initialize_pln_components(self):
        """Initialize PLN components with fallback implementations."""
        self.pln_available = False
        self.reasoning_rules = [
            "deduction_rule",
            "modus_ponens_rule",
            "fuzzy_conjunction_rule", 
            "fuzzy_disjunction_rule",
            "contraposition_rule",
            "inheritance_rule",
            "similarity_rule",
            "abduction_rule"
        ]
        
        try:
            # Try to initialize OpenCog components
            from opencog.atomspace import AtomSpace, types
            self.atomspace = AtomSpace()
            self.pln_available = True
            print("✅ OpenCog PLN components initialized")
        except ImportError:
            # Use fallback implementations
            self.atomspace = None
            print("ℹ️  Using fallback PLN implementations")
    
    def run_all_examples(self) -> Dict[str, Any]:
        """Run all advanced PLN reasoning examples and return results."""
        print("🚀 Advanced PLN Reasoning Examples with Agent-Zero Integration")
        print("=" * 70)
        print(f"Started: {self.start_time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("=" * 70)
        
        examples = [
            ("Problem-Solving Agent", self.example_problem_solving_agent),
            ("Learning Agent", self.example_learning_agent), 
            ("Multi-Modal Reasoning", self.example_multimodal_reasoning),
            ("Causal Inference", self.example_causal_inference),
            ("Meta-Cognitive Reasoning", self.example_metacognitive_reasoning),
            ("Collaborative Agent Network", self.example_collaborative_agents)
        ]
        
        results = {}
        
        for example_name, example_func in examples:
            print(f"\n📋 Running: {example_name}")
            print("-" * 50)
            try:
                result = example_func()
                results[example_name] = result
                self.examples_run += 1
                print(f"✅ {example_name} completed successfully")
            except Exception as e:
                print(f"❌ {example_name} failed: {e}")
                results[example_name] = {"error": str(e), "status": "failed"}
        
        # Generate summary
        summary = self._generate_summary(results)
        results["_summary"] = summary
        
        return results
    
    def example_problem_solving_agent(self) -> Dict[str, Any]:
        """
        Example 1: Problem-Solving Agent using PLN for logical deduction
        
        Demonstrates how an Agent-Zero can use PLN to solve complex problems
        by applying logical inference rules to known facts and constraints.
        """
        
        # Problem scenario: Planning a software development project
        problem_facts = [
            "project_requires_frontend_development",
            "project_requires_backend_development", 
            "frontend_development_needs_ui_framework",
            "backend_development_needs_database",
            "ui_framework_requires_javascript_knowledge",
            "database_requires_sql_knowledge",
            "agent_has_javascript_knowledge",
            "agent_has_sql_knowledge"
        ]
        
        goal = "agent_can_complete_project"
        
        print("🎯 Problem-Solving Agent Example")
        print(f"Goal: {goal}")
        print("Facts:")
        for i, fact in enumerate(problem_facts, 1):
            print(f"  {i}. {fact}")
        
        # Apply PLN reasoning
        reasoning_steps = []
        current_facts = list(problem_facts)
        
        # Step 1: Apply deduction rules
        print("\nApplying PLN Deduction Rules:")
        
        # Rule: If agent has required knowledge, agent can do task
        if "agent_has_javascript_knowledge" in current_facts and "ui_framework_requires_javascript_knowledge" in current_facts:
            new_fact = "agent_can_develop_frontend"
            current_facts.append(new_fact)
            reasoning_steps.append({
                "rule": "deduction_rule",
                "premises": ["agent_has_javascript_knowledge", "ui_framework_requires_javascript_knowledge"],
                "conclusion": new_fact,
                "confidence": 0.9
            })
            print(f"  ✓ Deduced: {new_fact}")
        
        if "agent_has_sql_knowledge" in current_facts and "database_requires_sql_knowledge" in current_facts:
            new_fact = "agent_can_develop_backend"  
            current_facts.append(new_fact)
            reasoning_steps.append({
                "rule": "deduction_rule",
                "premises": ["agent_has_sql_knowledge", "database_requires_sql_knowledge"], 
                "conclusion": new_fact,
                "confidence": 0.9
            })
            print(f"  ✓ Deduced: {new_fact}")
        
        # Step 2: Apply conjunction rule
        if "agent_can_develop_frontend" in current_facts and "agent_can_develop_backend" in current_facts:
            new_fact = goal
            current_facts.append(new_fact)
            reasoning_steps.append({
                "rule": "fuzzy_conjunction_rule",
                "premises": ["agent_can_develop_frontend", "agent_can_develop_backend"],
                "conclusion": new_fact,
                "confidence": 0.85
            })
            print(f"  ✅ Final conclusion: {new_fact}")
        
        self.successful_inferences += len(reasoning_steps)
        self.total_reasoning_steps += len(reasoning_steps)
        
        return {
            "goal": goal,
            "initial_facts": len(problem_facts),
            "reasoning_steps": reasoning_steps,
            "final_facts": len(current_facts),
            "goal_achieved": goal in current_facts,
            "confidence": reasoning_steps[-1]["confidence"] if reasoning_steps else 0.0
        }
    
    def example_learning_agent(self) -> Dict[str, Any]:
        """
        Example 2: Learning Agent using PLN for knowledge acquisition
        
        Demonstrates how an Agent-Zero can use PLN to learn new concepts
        and update its knowledge base through probabilistic inference.
        """
        
        print("🧠 Learning Agent Example")
        
        # Initial knowledge base
        knowledge_base = [
            ("programming_languages", "have_syntax", 0.95),
            ("python", "is_programming_language", 0.98),
            ("javascript", "is_programming_language", 0.97),
            ("syntax_errors", "prevent_execution", 0.90),
            ("good_syntax", "enables_execution", 0.88)
        ]
        
        # New observations to learn from
        observations = [
            ("python_code", "has_indentation_rules", 0.85),
            ("javascript_code", "has_bracket_rules", 0.87),
            ("indentation_rules", "are_syntax_feature", 0.80),
            ("bracket_rules", "are_syntax_feature", 0.82)
        ]
        
        print("Initial Knowledge Base:")
        for subject, predicate, confidence in knowledge_base:
            print(f"  {subject} {predicate} (confidence: {confidence})")
        
        print("\nNew Observations:")
        for subject, predicate, confidence in observations:
            print(f"  {subject} {predicate} (confidence: {confidence})")
        
        # Apply PLN learning rules
        learning_steps = []
        learned_knowledge = []
        
        print("\nApplying PLN Learning Rules:")
        
        # Inheritance rule: If A is B, and B has property P, then A has property P
        for obs_subj, obs_pred, obs_conf in observations:
            for kb_subj, kb_pred, kb_conf in knowledge_base:
                if "syntax" in obs_pred and "programming_language" in kb_pred:
                    # Learn specific syntax rules for programming languages
                    if "python" in obs_subj and "python" in kb_subj:
                        new_knowledge = ("python", "requires_proper_indentation", obs_conf * kb_conf)
                        learned_knowledge.append(new_knowledge)
                        learning_steps.append({
                            "rule": "inheritance_rule",
                            "observation": (obs_subj, obs_pred, obs_conf),
                            "base_knowledge": (kb_subj, kb_pred, kb_conf),
                            "learned": new_knowledge,
                            "confidence": new_knowledge[2]
                        })
                        print(f"  ✓ Learned: {new_knowledge[0]} {new_knowledge[1]} (confidence: {new_knowledge[2]:.3f})")
                    
                    elif "javascript" in obs_subj and "javascript" in kb_subj:
                        new_knowledge = ("javascript", "requires_proper_brackets", obs_conf * kb_conf)
                        learned_knowledge.append(new_knowledge)
                        learning_steps.append({
                            "rule": "inheritance_rule",
                            "observation": (obs_subj, obs_pred, obs_conf),
                            "base_knowledge": (kb_subj, kb_pred, kb_conf),
                            "learned": new_knowledge,
                            "confidence": new_knowledge[2]
                        })
                        print(f"  ✓ Learned: {new_knowledge[0]} {new_knowledge[1]} (confidence: {new_knowledge[2]:.3f})")
        
        # Additional learning: Pattern recognition from observations
        for obs_subj, obs_pred, obs_conf in observations:
            if "rules" in obs_pred:
                # Learn that specific codes have specific rule types
                new_knowledge = (obs_subj.replace("_code", ""), "has_syntax_constraints", obs_conf * 0.9)
                learned_knowledge.append(new_knowledge)
                learning_steps.append({
                    "rule": "pattern_recognition_rule",
                    "observation": (obs_subj, obs_pred, obs_conf),
                    "learned": new_knowledge,
                    "confidence": new_knowledge[2]
                })
                print(f"  ✓ Pattern learned: {new_knowledge[0]} {new_knowledge[1]} (confidence: {new_knowledge[2]:.3f})")
        
        # Generalization rule: Find common patterns
        if len(learned_knowledge) >= 2:
            general_rule = ("programming_languages", "have_specific_syntax_rules", 0.75)
            learned_knowledge.append(general_rule),
            learning_steps.append({
                "rule": "generalization_rule",
                "specific_cases": learned_knowledge[:-1],
                "generalization": general_rule,
                "confidence": general_rule[2]
            })
            print(f"  🎯 Generalized: {general_rule[0]} {general_rule[1]} (confidence: {general_rule[2]})")
        
        self.successful_inferences += len(learning_steps)
        self.total_reasoning_steps += len(learning_steps)
        
        return {
            "initial_knowledge": len(knowledge_base),
            "observations": len(observations),
            "learning_steps": learning_steps,
            "learned_knowledge": learned_knowledge,
            "knowledge_growth": len(learned_knowledge)
        }
    
    def example_multimodal_reasoning(self) -> Dict[str, Any]:
        """
        Example 3: Multi-Modal Reasoning combining different types of information
        
        Demonstrates PLN's ability to reason across different modalities
        (text, visual concepts, temporal patterns) in Agent-Zero contexts.
        """
        
        print("🌈 Multi-Modal Reasoning Example")
        
        # Multi-modal information sources
        textual_info = [
            ("user_message", "contains_urgency_keywords", 0.85),
            ("urgency_keywords", "indicate_high_priority", 0.90),
            ("high_priority", "requires_immediate_response", 0.88)
        ]
        
        temporal_info = [
            ("current_time", "is_business_hours", 0.95),
            ("business_hours", "allow_immediate_processing", 0.92),
            ("weekend", "reduces_response_capacity", 0.70)
        ]
        
        contextual_info = [
            ("user", "is_premium_customer", 0.80),
            ("premium_customer", "gets_priority_support", 0.85),
            ("priority_support", "enables_fast_response", 0.90)
        ]
        
        print("Multi-Modal Information:")
        print("  Textual:", [f"{s} {p}" for s, p, c in textual_info])
        print("  Temporal:", [f"{s} {p}" for s, p, c in temporal_info])
        print("  Contextual:", [f"{s} {p}" for s, p, c in contextual_info])
        
        # Combine multi-modal reasoning
        reasoning_chain = []
        final_confidence = 1.0
        
        print("\nMulti-Modal PLN Reasoning:")
        
        # Chain textual reasoning
        text_confidence = 1.0
        for info in textual_info:
            text_confidence *= info[2]
        reasoning_chain.append({
            "modality": "textual",
            "conclusion": "message_requires_urgent_response",
            "confidence": text_confidence
        })
        print(f"  📝 Textual: message_requires_urgent_response (confidence: {text_confidence:.3f})")
        
        # Chain temporal reasoning  
        temporal_confidence = temporal_info[0][2] * temporal_info[1][2]
        reasoning_chain.append({
            "modality": "temporal", 
            "conclusion": "system_can_process_immediately",
            "confidence": temporal_confidence
        })
        print(f"  ⏰ Temporal: system_can_process_immediately (confidence: {temporal_confidence:.3f})")
        
        # Chain contextual reasoning
        context_confidence = 1.0
        for info in contextual_info:
            context_confidence *= info[2]
        reasoning_chain.append({
            "modality": "contextual",
            "conclusion": "user_deserves_priority_response", 
            "confidence": context_confidence
        })
        print(f"  🎯 Contextual: user_deserves_priority_response (confidence: {context_confidence:.3f})")
        
        # Apply PLN conjunction across modalities
        final_confidence = text_confidence * temporal_confidence * context_confidence
        final_conclusion = {
            "rule": "multi_modal_conjunction",
            "modalities": ["textual", "temporal", "contextual"],
            "conclusion": "agent_should_provide_immediate_priority_response",
            "confidence": final_confidence
        }
        reasoning_chain.append(final_conclusion)
        
        print(f"  🎊 Final: {final_conclusion['conclusion']} (confidence: {final_confidence:.3f})")
        
        self.successful_inferences += len(reasoning_chain)
        self.total_reasoning_steps += len(reasoning_chain)
        
        return {
            "modalities": 3,
            "reasoning_chain": reasoning_chain,
            "final_conclusion": final_conclusion,
            "confidence": final_confidence,
            "decision": "immediate_priority_response" if final_confidence > 0.7 else "standard_response"
        }
    
    def example_causal_inference(self) -> Dict[str, Any]:
        """
        Example 4: Causal Inference using PLN for understanding cause-effect relationships
        
        Demonstrates how Agent-Zero can use PLN to infer causal relationships
        and make predictions based on causal reasoning.
        """
        
        print("🔗 Causal Inference Example")
        
        # Causal network: Learning performance factors
        causal_observations = [
            ("student_attends_lectures", "student_engagement_increases", 0.75),
            ("student_engagement_increases", "comprehension_improves", 0.80),
            ("comprehension_improves", "test_scores_increase", 0.85),
            ("student_studies_regularly", "knowledge_retention_improves", 0.82),
            ("knowledge_retention_improves", "test_scores_increase", 0.78),
            ("student_gets_enough_sleep", "cognitive_function_optimal", 0.70),
            ("cognitive_function_optimal", "comprehension_improves", 0.75)
        ]
        
        # Current situation to analyze
        current_situation = [
            ("student_alice", "attends_lectures", 0.90),
            ("student_alice", "studies_regularly", 0.85),
            ("student_alice", "gets_enough_sleep", 0.60)
        ]
        
        print("Causal Knowledge:")
        for cause, effect, strength in causal_observations:
            print(f"  {cause} → {effect} (strength: {strength})")
        
        print("\nCurrent Situation:")
        for student, behavior, frequency in current_situation:
            print(f"  {student} {behavior} (frequency: {frequency})")
        
        # Apply causal reasoning
        causal_inferences = []
        
        print("\nCausal PLN Reasoning:")
        
        # Trace causal chains for Alice
        alice_predictions = {}
        
        # Chain 1: Lectures → Engagement → Comprehension → Test Scores
        if ("student_alice", "attends_lectures", 0.90) in current_situation:
            chain1_prob = 0.90 * 0.75 * 0.80 * 0.85  # attendance * engagement * comprehension * scores
            alice_predictions["test_scores_via_lectures"] = chain1_prob
            causal_inferences.append({
                "rule": "causal_chain",
                "chain": "lectures → engagement → comprehension → test_scores",
                "probability": chain1_prob,
                "student": "alice"
            })
            print(f"  📚 Chain 1: Alice's lecture attendance → test scores (prob: {chain1_prob:.3f})")
        
        # Chain 2: Study → Retention → Test Scores  
        if ("student_alice", "studies_regularly", 0.85) in current_situation:
            chain2_prob = 0.85 * 0.82 * 0.78  # study * retention * scores
            alice_predictions["test_scores_via_study"] = chain2_prob
            causal_inferences.append({
                "rule": "causal_chain",
                "chain": "study → retention → test_scores", 
                "probability": chain2_prob,
                "student": "alice"
            })
            print(f"  📖 Chain 2: Alice's regular study → test scores (prob: {chain2_prob:.3f})")
        
        # Chain 3: Sleep → Cognitive → Comprehension → Test Scores
        if ("student_alice", "gets_enough_sleep", 0.60) in current_situation:
            chain3_prob = 0.60 * 0.70 * 0.75 * 0.85  # sleep * cognitive * comprehension * scores
            alice_predictions["test_scores_via_sleep"] = chain3_prob
            causal_inferences.append({
                "rule": "causal_chain",
                "chain": "sleep → cognitive → comprehension → test_scores",
                "probability": chain3_prob,
                "student": "alice"
            })
            print(f"  😴 Chain 3: Alice's sleep → test scores (prob: {chain3_prob:.3f})")
        
        # Combine all causal factors (using noisy-OR model)
        combined_success_prob = 1.0
        for prob in alice_predictions.values():
            combined_success_prob *= (1.0 - prob)
        combined_success_prob = 1.0 - combined_success_prob
        
        causal_inferences.append({
            "rule": "causal_combination",
            "method": "noisy_or",
            "individual_factors": alice_predictions,
            "combined_probability": combined_success_prob
        })
        
        print(f"  🎯 Combined: Alice's overall test success probability: {combined_success_prob:.3f}")
        
        # Suggest interventions
        interventions = []
        if alice_predictions.get("test_scores_via_sleep", 0) < 0.4:
            interventions.append({
                "target": "sleep_improvement",
                "expected_boost": 0.70 * 0.75 * 0.85 - alice_predictions.get("test_scores_via_sleep", 0),
                "action": "recommend_better_sleep_schedule"
            })
        
        if len(interventions) > 0:
            print("  💡 Recommended interventions:")
            for intervention in interventions:
                print(f"    - {intervention['action']} (expected boost: +{intervention['expected_boost']:.3f})")
        
        self.successful_inferences += len(causal_inferences)
        self.total_reasoning_steps += len(causal_inferences)
        
        return {
            "causal_observations": len(causal_observations),
            "causal_inferences": causal_inferences,
            "predictions": alice_predictions,
            "combined_probability": combined_success_prob,
            "interventions": interventions
        }
    
    def example_metacognitive_reasoning(self) -> Dict[str, Any]:
        """
        Example 5: Meta-Cognitive Reasoning about the reasoning process itself
        
        Demonstrates how Agent-Zero can use PLN to reason about its own
        reasoning processes and improve its cognitive strategies.
        """
        
        print("🧩 Meta-Cognitive Reasoning Example")
        
        # Meta-cognitive knowledge about reasoning strategies
        reasoning_strategies = [
            ("deductive_reasoning", "works_well_with_certain_facts", 0.90),
            ("inductive_reasoning", "works_well_with_patterns", 0.85),
            ("abductive_reasoning", "works_well_with_incomplete_info", 0.80),
            ("analogical_reasoning", "works_well_with_similar_domains", 0.75)
        ]
        
        # Current problem characteristics
        problem_characteristics = [
            ("current_problem", "has_incomplete_information", 0.85),
            ("current_problem", "has_some_patterns", 0.70),
            ("current_problem", "similar_to_past_problems", 0.60),
            ("current_problem", "has_certain_facts", 0.40)
        ]
        
        # Past reasoning performance
        performance_history = [
            ("deductive_reasoning", "solved_similar_problems", 0.65),
            ("inductive_reasoning", "solved_similar_problems", 0.75),
            ("abductive_reasoning", "solved_similar_problems", 0.82),
            ("analogical_reasoning", "solved_similar_problems", 0.58)
        ]
        
        print("Reasoning Strategies Available:")
        for strategy, condition, effectiveness in reasoning_strategies:
            print(f"  {strategy}: {condition} (effectiveness: {effectiveness})")
        
        print("\nCurrent Problem Characteristics:")
        for problem, characteristic, degree in problem_characteristics:
            print(f"  {characteristic} (degree: {degree})")
        
        print("\nPast Performance:")
        for strategy, outcome, success_rate in performance_history:
            print(f"  {strategy}: {outcome} (success: {success_rate})")
        
        # Meta-cognitive reasoning to select best strategy
        strategy_scores = {}
        meta_reasoning_steps = []
        
        print("\nMeta-Cognitive PLN Reasoning:")
        
        for strategy, condition, base_effectiveness in reasoning_strategies:
            # Calculate strategy score based on problem fit and past performance
            
            # Find problem characteristic match
            problem_fit = 0.0
            if "certain_facts" in condition and ("current_problem", "has_certain_facts", 0.40) in problem_characteristics:
                problem_fit = 0.40
            elif "patterns" in condition and ("current_problem", "has_some_patterns", 0.70) in problem_characteristics:
                problem_fit = 0.70
            elif "incomplete_info" in condition and ("current_problem", "has_incomplete_information", 0.85) in problem_characteristics:
                problem_fit = 0.85
            elif "similar_domains" in condition and ("current_problem", "similar_to_past_problems", 0.60) in problem_characteristics:
                problem_fit = 0.60
            
            # Find past performance
            past_performance = 0.0
            for perf_strategy, outcome, success_rate in performance_history:
                if perf_strategy == strategy:
                    past_performance = success_rate
                    break
            
            # Combine factors using PLN weighted combination
            strategy_score = (base_effectiveness * 0.4) + (problem_fit * 0.35) + (past_performance * 0.25)
            strategy_scores[strategy] = strategy_score
            
            meta_reasoning_steps.append({
                "rule": "strategy_evaluation",
                "strategy": strategy,
                "base_effectiveness": base_effectiveness,
                "problem_fit": problem_fit,
                "past_performance": past_performance,
                "combined_score": strategy_score
            })
            
            print(f"  🔍 {strategy}: score {strategy_score:.3f} (base:{base_effectiveness}, fit:{problem_fit:.2f}, past:{past_performance:.2f})")
        
        # Select best strategy
        best_strategy = max(strategy_scores, key=strategy_scores.get)
        best_score = strategy_scores[best_strategy]
        
        meta_reasoning_steps.append({
            "rule": "strategy_selection",
            "selected_strategy": best_strategy,
            "selection_confidence": best_score,
            "alternatives": {k: v for k, v in strategy_scores.items() if k != best_strategy}
        })
        
        print(f"  🎯 Selected Strategy: {best_strategy} (confidence: {best_score:.3f})")
        
        # Meta-reasoning about confidence in the selection
        selection_confidence = best_score
        confidence_gap = best_score - max([score for strategy, score in strategy_scores.items() if strategy != best_strategy])
        
        if confidence_gap > 0.2:
            confidence_assessment = "high_confidence"
        elif confidence_gap > 0.1:
            confidence_assessment = "medium_confidence"  
        else:
            confidence_assessment = "low_confidence"
        
        meta_reasoning_steps.append({
            "rule": "confidence_assessment",
            "confidence_level": confidence_assessment,
            "confidence_gap": confidence_gap,
            "recommendation": "proceed" if confidence_assessment in ["high_confidence", "medium_confidence"] else "gather_more_info"
        })
        
        print(f"  📊 Confidence Assessment: {confidence_assessment} (gap: {confidence_gap:.3f})")
        
        self.successful_inferences += len(meta_reasoning_steps)
        self.total_reasoning_steps += len(meta_reasoning_steps)
        
        return {
            "strategies_evaluated": len(reasoning_strategies),
            "meta_reasoning_steps": meta_reasoning_steps,
            "strategy_scores": strategy_scores,
            "selected_strategy": best_strategy,
            "selection_confidence": best_score,
            "confidence_assessment": confidence_assessment
        }
    
    def example_collaborative_agents(self) -> Dict[str, Any]:
        """
        Example 6: Collaborative Agent Network using PLN for distributed reasoning
        
        Demonstrates how multiple Agent-Zero instances can use PLN to collaborate
        and combine their reasoning capabilities for complex problem solving.
        """
        
        print("🤝 Collaborative Agent Network Example")
        
        # Define agent capabilities and specializations
        agent_capabilities = {
            "agent_alpha": {
                "specialization": "data_analysis",
                "capabilities": ["statistical_analysis", "pattern_recognition", "data_visualization"],
                "confidence_domain": 0.90
            },
            "agent_beta": {
                "specialization": "logical_reasoning", 
                "capabilities": ["formal_logic", "proof_construction", "argument_validation"],
                "confidence_domain": 0.85
            },
            "agent_gamma": {
                "specialization": "creative_problem_solving",
                "capabilities": ["analogical_reasoning", "lateral_thinking", "solution_generation"],
                "confidence_domain": 0.80
            }
        }
        
        # Complex problem requiring multiple expertise areas
        collaborative_problem = {
            "description": "optimize_customer_recommendation_system",
            "requirements": [
                "analyze_customer_behavior_patterns",  # needs data_analysis
                "validate_recommendation_logic",       # needs logical_reasoning  
                "generate_creative_recommendation_strategies"  # needs creative_problem_solving
            ],
            "constraints": [
                "maintain_system_performance",
                "ensure_user_privacy",
                "maximize_customer_satisfaction"
            ]
        }
        
        print("Agent Network:")
        for agent_id, info in agent_capabilities.items():
            print(f"  {agent_id}: {info['specialization']} (confidence: {info['confidence_domain']})")
            print(f"    Capabilities: {', '.join(info['capabilities'])}")
        
        print(f"\nCollaborative Problem: {collaborative_problem['description']}")
        print("Requirements:")
        for req in collaborative_problem['requirements']:
            print(f"  - {req}")
        
        # PLN-based task allocation
        task_allocations = []
        collaboration_steps = []
        
        print("\nPLN-Based Task Allocation:")
        
        # Match requirements to agent capabilities
        for requirement in collaborative_problem['requirements']:
            best_match = None
            best_score = 0.0
            
            for agent_id, info in agent_capabilities.items():
                # Calculate capability match score
                match_score = 0.0
                
                if "analyze" in requirement and "data_analysis" in info['specialization']:
                    match_score = info['confidence_domain'] * 0.95
                elif "validate" in requirement and "logical_reasoning" in info['specialization']:
                    match_score = info['confidence_domain'] * 0.90
                elif "generate" in requirement and "creative_problem_solving" in info['specialization']:
                    match_score = info['confidence_domain'] * 0.85
                
                # Apply capability-requirement matching rule
                for capability in info['capabilities']:
                    if any(keyword in requirement for keyword in capability.split('_')):
                        match_score += 0.1
                
                if match_score > best_score:
                    best_score = match_score
                    best_match = agent_id
            
            if best_match:
                task_allocations.append({
                    "requirement": requirement,
                    "assigned_agent": best_match,
                    "match_score": best_score
                })
                
                collaboration_steps.append({
                    "rule": "capability_matching",
                    "requirement": requirement,
                    "assigned_to": best_match,
                    "confidence": best_score
                })
                
                print(f"  📋 {requirement} → {best_match} (confidence: {best_score:.3f})")
        
        # PLN-based result integration
        print("\nPLN-Based Result Integration:")
        
        # Simulate agent results
        agent_results = {
            "agent_alpha": {
                "findings": "customer_segments_identified",
                "confidence": 0.88,
                "data": ["segment_A_prefers_electronics", "segment_B_prefers_books", "segment_C_prefers_clothing"]
            },
            "agent_beta": {
                "findings": "recommendation_logic_validated",
                "confidence": 0.92,
                "data": ["collaborative_filtering_sound", "content_based_filtering_sound", "hybrid_approach_optimal"]
            },
            "agent_gamma": {
                "findings": "creative_strategies_generated", 
                "confidence": 0.85,
                "data": ["seasonal_theme_recommendations", "social_influence_recommendations", "surprise_serendipity_recommendations"]
            }
        }
        
        # Apply PLN integration rules
        integration_confidence = 1.0
        integrated_solution = []
        
        for agent_id, result in agent_results.items():
            print(f"  🤖 {agent_id}: {result['findings']} (confidence: {result['confidence']})")
            integration_confidence *= result['confidence']
            integrated_solution.extend(result['data'])
            
            collaboration_steps.append({
                "rule": "result_integration",
                "contributor": agent_id,
                "contribution": result['findings'],
                "confidence": result['confidence']
            })
        
        # Final integration using PLN conjunction
        final_integration = {
            "rule": "collaborative_conjunction",
            "contributors": list(agent_results.keys()),
            "integrated_solution": integrated_solution,
            "overall_confidence": integration_confidence,
            "synergy_bonus": 0.05  # Small bonus for successful collaboration
        }
        
        final_confidence = integration_confidence + final_integration["synergy_bonus"]
        collaboration_steps.append(final_integration)
        
        print(f"  🎯 Integrated Solution Confidence: {final_confidence:.3f}")
        print("  📋 Integrated Components:")
        for component in integrated_solution:
            print(f"    - {component}")
        
        self.successful_inferences += len(collaboration_steps)
        self.total_reasoning_steps += len(collaboration_steps)
        
        return {
            "agents_involved": len(agent_capabilities),
            "task_allocations": task_allocations,
            "collaboration_steps": collaboration_steps,
            "agent_results": agent_results,
            "integrated_solution": integrated_solution,
            "final_confidence": final_confidence,
            "collaboration_success": final_confidence > 0.7
        }
    
    def _generate_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a comprehensive summary of all examples."""
        
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        
        successful_examples = sum(1 for result in results.values() 
                                 if isinstance(result, dict) and result.get("status") != "failed")
        
        print(f"\n{'='*70}")
        print("🎉 Advanced PLN Reasoning Examples - Complete!")
        print(f"{'='*70}")
        print(f"Duration: {duration:.2f} seconds")
        print(f"Examples run: {self.examples_run}")
        print(f"Successful examples: {successful_examples}")
        print(f"Total reasoning steps: {self.total_reasoning_steps}")
        print(f"Successful inferences: {self.successful_inferences}")
        print(f"PLN components available: {'Yes' if self.pln_available else 'No (using fallbacks)'}")
        
        summary = {
            "start_time": self.start_time.isoformat(),
            "end_time": end_time.isoformat(), 
            "duration_seconds": duration,
            "examples_run": self.examples_run,
            "successful_examples": successful_examples,
            "total_reasoning_steps": self.total_reasoning_steps,
            "successful_inferences": self.successful_inferences,
            "pln_available": self.pln_available,
            "success_rate": successful_examples / self.examples_run if self.examples_run > 0 else 0.0,
            "avg_steps_per_example": self.total_reasoning_steps / self.examples_run if self.examples_run > 0 else 0.0
        }
        
        print(f"Success rate: {summary['success_rate']:.1%}")
        print(f"Average steps per example: {summary['avg_steps_per_example']:.1f}")
        
        return summary


def main():
    """Run all advanced PLN reasoning examples."""
    
    print("🚀 Starting Advanced PLN Reasoning Examples with Agent-Zero Integration")
    print("Issue #55 - Advanced Learning Systems (Phase 4) Implementation")
    print()
    
    # Initialize and run examples
    examples_system = AdvancedPLNReasoningExamples()
    results = examples_system.run_all_examples()
    
    # Save results to file for reference
    output_file = PROJECT_ROOT / "advanced_pln_reasoning_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: {output_file}")
    
    return results


if __name__ == "__main__":
    results = main()
    
    # Return appropriate exit code
    summary = results.get("_summary", {})
    success_rate = summary.get("success_rate", 0.0)
    
    if success_rate >= 0.8:
        print("✅ All examples completed successfully!")
        exit(0)
    elif success_rate >= 0.5:
        print("⚠️  Most examples completed successfully!")
        exit(0) 
    else:
        print("❌ Many examples failed!")
        exit(1)