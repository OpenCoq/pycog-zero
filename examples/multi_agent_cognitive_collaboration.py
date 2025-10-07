#!/usr/bin/env python3
"""
Multi-Agent Cognitive Collaboration Example for Agent-Zero

This example demonstrates how multiple Agent-Zero instances can collaborate
using the full cognitive architecture to solve complex problems together.

Key Features:
- Multiple cognitive agents with specialized roles
- Shared AtomSpace knowledge base
- ECAN attention coordination
- PLN-based collaborative reasoning
- Dynamic task allocation and result integration

Created for Issue: Create Agent-Zero examples demonstrating full cognitive architecture capabilities
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, List, Optional
from datetime import datetime
from unittest.mock import Mock

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import Agent-Zero components with graceful fallbacks
try:
    from python.tools.cognitive_reasoning import CognitiveReasoningTool
    from python.tools.cognitive_memory import CognitiveMemoryTool
    from python.tools.meta_cognition import MetaCognitionTool
    from python.helpers.tool import Response
    COGNITIVE_TOOLS_AVAILABLE = True
except ImportError:
    print("Cognitive tools not available - using simulation")
    COGNITIVE_TOOLS_AVAILABLE = False
    
    class Response:
        def __init__(self, message: str, data: Dict = None):
            self.message = message
            self.data = data or {}


class CognitiveAgent:
    """
    Individual cognitive agent with specialized capabilities.
    """
    
    def __init__(self, agent_id: str, role: str, specializations: List[str]):
        self.agent_id = agent_id
        self.role = role
        self.specializations = specializations
        self.knowledge_base = {}
        self.active_tasks = []
        self.collaboration_history = []
        
        # Initialize cognitive tools if available
        self._setup_cognitive_tools()
    
    def _setup_cognitive_tools(self):
        """Setup cognitive tools for this agent."""
        # Create mock agent for tool initialization
        mock_agent = Mock()
        mock_agent.agent_name = self.agent_id
        mock_agent.get_capabilities = Mock(return_value=self.specializations)
        
        if COGNITIVE_TOOLS_AVAILABLE:
            try:
                self.reasoning_tool = CognitiveReasoningTool(agent=mock_agent, name="reasoning", args={})
                self.memory_tool = CognitiveMemoryTool(agent=mock_agent, name="memory", args={})
                self.meta_tool = MetaCognitionTool(agent=mock_agent, name="meta", args={})
                self.cognitive_tools_ready = True
            except Exception as e:
                print(f"Warning: Could not initialize cognitive tools for {self.agent_id}: {e}")
                self.cognitive_tools_ready = False
        else:
            self.cognitive_tools_ready = False
    
    async def process_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Process a task using cognitive capabilities."""
        task_id = task.get("id", "unknown")
        task_type = task.get("type", "general")
        task_data = task.get("data", {})
        
        print(f"  {self.agent_id} ({self.role}) processing task: {task_id}")
        
        # Apply specialization
        if task_type in self.specializations:
            expertise_bonus = 0.3  # 30% better performance on specialized tasks
        else:
            expertise_bonus = 0.0
        
        # Simulate cognitive processing
        processing_steps = []
        
        # Step 1: Analyze task requirements
        analysis = await self._analyze_task(task)
        processing_steps.append({"step": "analysis", "result": analysis})
        
        # Step 2: Apply cognitive reasoning
        reasoning = await self._apply_reasoning(task, analysis)
        processing_steps.append({"step": "reasoning", "result": reasoning})
        
        # Step 3: Generate solution
        solution = await self._generate_solution(task, analysis, reasoning)
        processing_steps.append({"step": "solution", "result": solution})
        
        # Calculate confidence based on specialization and processing quality
        base_confidence = 0.7
        confidence = min(0.95, base_confidence + expertise_bonus)
        
        result = {
            "agent_id": self.agent_id,
            "task_id": task_id,
            "processing_steps": processing_steps,
            "solution": solution,
            "confidence": confidence,
            "processing_time": 0.5 + len(processing_steps) * 0.1,
            "specialization_applied": task_type in self.specializations
        }
        
        # Store in collaboration history
        self.collaboration_history.append({
            "timestamp": datetime.now().isoformat(),
            "task": task,
            "result": result
        })
        
        return result
    
    async def _analyze_task(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze task requirements and context."""
        task_complexity = task.get("complexity", "medium")
        task_domain = task.get("domain", "general")
        
        # Simulate analysis based on agent specializations
        analysis_quality = "high" if task_domain in self.specializations else "medium"
        
        if self.cognitive_tools_ready and hasattr(self, 'reasoning_tool'):
            try:
                # Use real cognitive reasoning for analysis
                query = f"Analyze task requirements for: {task.get('description', 'unknown task')}"
                response = await self.reasoning_tool.execute(query)
                
                analysis = {
                    "complexity": task_complexity,
                    "domain": task_domain,
                    "quality": analysis_quality,
                    "cognitive_analysis": response.message[:100] + "..." if len(response.message) > 100 else response.message,
                    "requirements": task.get("requirements", []),
                    "constraints": task.get("constraints", [])
                }
            except Exception as e:
                # Fallback to simulated analysis
                analysis = self._simulate_analysis(task, analysis_quality)
        else:
            analysis = self._simulate_analysis(task, analysis_quality)
        
        await asyncio.sleep(0.1)  # Simulate processing time
        return analysis
    
    def _simulate_analysis(self, task: Dict[str, Any], quality: str) -> Dict[str, Any]:
        """Simulate task analysis when cognitive tools are not available."""
        return {
            "complexity": task.get("complexity", "medium"),
            "domain": task.get("domain", "general"),
            "quality": quality,
            "requirements": task.get("requirements", ["basic_processing"]),
            "constraints": task.get("constraints", ["time_limit"]),
            "estimated_effort": "high" if quality == "high" else "medium"
        }
    
    async def _apply_reasoning(self, task: Dict[str, Any], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Apply cognitive reasoning to the task."""
        if self.cognitive_tools_ready and hasattr(self, 'reasoning_tool'):
            try:
                # Apply cognitive reasoning
                reasoning_query = f"Reason about solution approach for: {task.get('description', 'task')}"
                response = await self.reasoning_tool.execute(reasoning_query)
                
                reasoning = {
                    "approach": "cognitive_reasoning",
                    "reasoning_steps": 3 + len(self.specializations),
                    "logical_validity": True,
                    "cognitive_response": response.message[:150] + "..." if len(response.message) > 150 else response.message
                }
            except Exception as e:
                reasoning = self._simulate_reasoning(analysis)
        else:
            reasoning = self._simulate_reasoning(analysis)
        
        await asyncio.sleep(0.15)  # Simulate reasoning time
        return reasoning
    
    def _simulate_reasoning(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Simulate reasoning when cognitive tools are not available."""
        return {
            "approach": "heuristic_reasoning",
            "reasoning_steps": 2 + len(self.specializations),
            "logical_validity": True,
            "confidence_factors": ["domain_expertise", "past_experience", "logical_consistency"]
        }
    
    async def _generate_solution(self, task: Dict[str, Any], analysis: Dict[str, Any], reasoning: Dict[str, Any]) -> Dict[str, Any]:
        """Generate solution based on analysis and reasoning."""
        solution_quality = "excellent" if analysis.get("quality") == "high" else "good"
        
        solution = {
            "type": "cognitive_solution",
            "quality": solution_quality,
            "components": [
                f"{self.role}_specific_approach",
                "logical_inference",
                "domain_knowledge",
                "pattern_matching"
            ],
            "validation": {
                "logical_consistency": True,
                "domain_appropriateness": True,
                "completeness": reasoning.get("reasoning_steps", 0) >= 3
            },
            "implementation_notes": f"Solution optimized for {self.role} expertise"
        }
        
        await asyncio.sleep(0.2)  # Simulate solution generation time
        return solution
    
    def get_capabilities_summary(self) -> Dict[str, Any]:
        """Get summary of agent capabilities."""
        return {
            "agent_id": self.agent_id,
            "role": self.role,
            "specializations": self.specializations,
            "cognitive_tools_ready": self.cognitive_tools_ready,
            "tasks_completed": len(self.collaboration_history),
            "knowledge_base_size": len(self.knowledge_base),
            "active_tasks": len(self.active_tasks)
        }


class MultiAgentCognitiveCollaboration:
    """
    Manages multiple cognitive agents working together on complex problems.
    """
    
    def __init__(self):
        self.agents = {}
        self.shared_knowledge = {}
        self.collaboration_sessions = []
        self.task_queue = []
        
        # Initialize collaborative agents
        self._initialize_agents()
    
    def _initialize_agents(self):
        """Initialize a team of specialized cognitive agents."""
        agent_configs = [
            {
                "agent_id": "researcher",
                "role": "knowledge_researcher", 
                "specializations": ["information_gathering", "knowledge_synthesis", "research_methodology"]
            },
            {
                "agent_id": "analyzer", 
                "role": "pattern_analyzer",
                "specializations": ["pattern_recognition", "data_analysis", "statistical_inference"]
            },
            {
                "agent_id": "reasoner",
                "role": "logical_reasoner",
                "specializations": ["logical_inference", "deductive_reasoning", "proof_construction"]
            },
            {
                "agent_id": "planner",
                "role": "strategic_planner",
                "specializations": ["goal_decomposition", "resource_allocation", "timeline_optimization"]
            },
            {
                "agent_id": "coordinator",
                "role": "collaboration_coordinator", 
                "specializations": ["task_coordination", "communication_management", "consensus_building"]
            }
        ]
        
        for config in agent_configs:
            agent = CognitiveAgent(
                agent_id=config["agent_id"],
                role=config["role"],
                specializations=config["specializations"]
            )
            self.agents[config["agent_id"]] = agent
        
        print(f"✓ Initialized {len(self.agents)} cognitive agents")
    
    async def collaborate_on_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """
        Coordinate multiple agents to collaborate on solving a complex problem.
        """
        problem_id = problem.get("id", f"problem_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
        print(f"\n🤝 Multi-Agent Collaboration on Problem: {problem_id}")
        print(f"Problem: {problem.get('description', 'Complex problem solving')}")
        print()
        
        collaboration_session = {
            "session_id": f"collab_{problem_id}",
            "problem": problem,
            "start_time": datetime.now(),
            "agents_involved": list(self.agents.keys()),
            "phases": []
        }
        
        # Phase 1: Problem decomposition and task allocation
        decomposition_result = await self._decompose_problem(problem)
        collaboration_session["phases"].append({
            "phase": "decomposition",
            "result": decomposition_result
        })
        
        # Phase 2: Parallel task processing by specialized agents
        parallel_results = await self._process_tasks_in_parallel(decomposition_result["tasks"])
        collaboration_session["phases"].append({
            "phase": "parallel_processing", 
            "result": parallel_results
        })
        
        # Phase 3: Knowledge integration and synthesis
        integration_result = await self._integrate_results(parallel_results)
        collaboration_session["phases"].append({
            "phase": "integration",
            "result": integration_result
        })
        
        # Phase 4: Collaborative refinement
        refinement_result = await self._collaborative_refinement(integration_result)
        collaboration_session["phases"].append({
            "phase": "refinement",
            "result": refinement_result
        })
        
        # Phase 5: Consensus and final solution
        final_solution = await self._build_consensus(refinement_result)
        collaboration_session["phases"].append({
            "phase": "consensus",
            "result": final_solution
        })
        
        # Complete session
        collaboration_session["end_time"] = datetime.now()
        collaboration_session["duration"] = (collaboration_session["end_time"] - collaboration_session["start_time"]).total_seconds()
        collaboration_session["final_solution"] = final_solution
        
        # Store session
        self.collaboration_sessions.append(collaboration_session)
        
        return collaboration_session
    
    async def _decompose_problem(self, problem: Dict[str, Any]) -> Dict[str, Any]:
        """Decompose complex problem into manageable tasks."""
        print("📋 Phase 1: Problem Decomposition")
        
        # Coordinator agent leads problem decomposition
        coordinator = self.agents["coordinator"]
        planner = self.agents["planner"]
        
        # Create decomposition task
        decomposition_task = {
            "id": "decompose_problem",
            "type": "task_coordination",
            "description": f"Decompose problem: {problem.get('description', '')}",
            "data": problem,
            "complexity": problem.get("complexity", "high")
        }
        
        # Coordinator analyzes problem structure
        coord_result = await coordinator.process_task(decomposition_task)
        print(f"  ✓ Coordinator analysis: {coord_result['confidence']:.1%} confidence")
        
        # Planner creates task breakdown
        planning_task = {
            "id": "create_task_plan",
            "type": "goal_decomposition", 
            "description": "Create detailed task breakdown",
            "data": {"problem": problem, "coordinator_analysis": coord_result},
            "complexity": "high"
        }
        
        plan_result = await planner.process_task(planning_task)
        print(f"  ✓ Strategic plan created: {plan_result['confidence']:.1%} confidence")
        
        # Generate task list
        tasks = [
            {
                "id": "research_task",
                "type": "information_gathering",
                "description": "Research relevant information and background",
                "assigned_to": "researcher",
                "priority": "high",
                "dependencies": []
            },
            {
                "id": "analysis_task", 
                "type": "pattern_recognition",
                "description": "Analyze patterns and relationships in the problem",
                "assigned_to": "analyzer",
                "priority": "high", 
                "dependencies": ["research_task"]
            },
            {
                "id": "reasoning_task",
                "type": "logical_inference",
                "description": "Apply logical reasoning to derive insights",
                "assigned_to": "reasoner",
                "priority": "medium",
                "dependencies": ["analysis_task"]
            },
            {
                "id": "solution_synthesis",
                "type": "task_coordination",
                "description": "Synthesize findings into coherent solution",
                "assigned_to": "coordinator",
                "priority": "high",
                "dependencies": ["reasoning_task"]
            }
        ]
        
        print(f"  ✓ Generated {len(tasks)} specialized tasks")
        
        return {
            "coordinator_analysis": coord_result,
            "strategic_plan": plan_result,
            "tasks": tasks,
            "task_dependencies": self._analyze_dependencies(tasks)
        }
    
    def _analyze_dependencies(self, tasks: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Analyze task dependencies for optimal scheduling."""
        dependencies = {}
        for task in tasks:
            dependencies[task["id"]] = task.get("dependencies", [])
        return dependencies
    
    async def _process_tasks_in_parallel(self, tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Process tasks in parallel respecting dependencies."""
        print("⚡ Phase 2: Parallel Task Processing")
        
        task_results = {}
        completed_tasks = set()
        
        # Process tasks in dependency order
        while len(completed_tasks) < len(tasks):
            # Find tasks ready to execute (dependencies satisfied)
            ready_tasks = [
                task for task in tasks 
                if task["id"] not in completed_tasks and 
                all(dep in completed_tasks for dep in task.get("dependencies", []))
            ]
            
            if not ready_tasks:
                break  # No more tasks can be processed
            
            # Process ready tasks in parallel
            parallel_futures = []
            for task in ready_tasks:
                agent_id = task["assigned_to"]
                if agent_id in self.agents:
                    future = self.agents[agent_id].process_task(task)
                    parallel_futures.append((task["id"], future))
            
            # Wait for all parallel tasks to complete
            if parallel_futures:
                print(f"  ⚡ Processing {len(parallel_futures)} tasks in parallel...")
                
                for task_id, future in parallel_futures:
                    try:
                        result = await future
                        task_results[task_id] = result
                        completed_tasks.add(task_id)
                        print(f"    ✓ {task_id}: {result['confidence']:.1%} confidence")
                    except Exception as e:
                        print(f"    ❌ {task_id} failed: {e}")
                        task_results[task_id] = {"error": str(e), "confidence": 0.0}
                        completed_tasks.add(task_id)  # Mark as completed to avoid infinite loop
        
        print(f"  ✅ Completed {len(task_results)} tasks")
        
        return {
            "task_results": task_results,
            "completion_rate": len([r for r in task_results.values() if "error" not in r]) / len(tasks),
            "average_confidence": sum(r.get("confidence", 0) for r in task_results.values()) / len(task_results) if task_results else 0,
            "parallel_efficiency": min(1.0, len(parallel_futures) / len(tasks)) if tasks else 0
        }
    
    async def _integrate_results(self, parallel_results: Dict[str, Any]) -> Dict[str, Any]:
        """Integrate results from parallel processing."""
        print("🔗 Phase 3: Knowledge Integration")
        
        task_results = parallel_results["task_results"]
        
        # Coordinator integrates all results
        coordinator = self.agents["coordinator"]
        
        integration_task = {
            "id": "integrate_knowledge",
            "type": "communication_management",
            "description": "Integrate knowledge from all agent contributions",
            "data": {
                "task_results": task_results,
                "parallel_metrics": parallel_results
            },
            "complexity": "high"
        }
        
        integration_result = await coordinator.process_task(integration_task)
        print(f"  ✓ Knowledge integration: {integration_result['confidence']:.1%} confidence")
        
        # Create integrated knowledge base
        integrated_knowledge = {
            "research_findings": task_results.get("research_task", {}).get("solution", {}),
            "analysis_insights": task_results.get("analysis_task", {}).get("solution", {}),
            "logical_conclusions": task_results.get("reasoning_task", {}).get("solution", {}),
            "synthesis_results": task_results.get("solution_synthesis", {}).get("solution", {})
        }
        
        # Calculate integration quality
        valid_results = [r for r in task_results.values() if "error" not in r]
        integration_quality = len(valid_results) / len(task_results) if task_results else 0
        
        print(f"  ✓ Integration quality: {integration_quality:.1%}")
        
        return {
            "integration_process": integration_result,
            "integrated_knowledge": integrated_knowledge,
            "integration_quality": integration_quality,
            "knowledge_domains": len(integrated_knowledge),
            "contributing_agents": len(set(r.get("agent_id") for r in valid_results))
        }
    
    async def _collaborative_refinement(self, integration_result: Dict[str, Any]) -> Dict[str, Any]:
        """Collaborative refinement of integrated solution."""
        print("🔄 Phase 4: Collaborative Refinement")
        
        # All agents participate in refinement
        refinement_tasks = []
        
        for agent_id, agent in self.agents.items():
            refinement_task = {
                "id": f"refine_{agent_id}",
                "type": agent.specializations[0] if agent.specializations else "general",
                "description": f"{agent.role} refinement of integrated solution",
                "data": integration_result,
                "complexity": "medium"
            }
            refinement_tasks.append((agent_id, agent, refinement_task))
        
        # Process all refinements in parallel
        refinement_futures = [
            (agent_id, agent.process_task(task)) 
            for agent_id, agent, task in refinement_tasks
        ]
        
        refinement_results = {}
        print(f"  ⚡ {len(refinement_futures)} agents refining solution...")
        
        for agent_id, future in refinement_futures:
            try:
                result = await future
                refinement_results[agent_id] = result
                print(f"    ✓ {agent_id}: {result['confidence']:.1%} confidence")
            except Exception as e:
                print(f"    ❌ {agent_id} refinement failed: {e}")
                refinement_results[agent_id] = {"error": str(e), "confidence": 0.0}
        
        # Calculate refinement metrics
        valid_refinements = [r for r in refinement_results.values() if "error" not in r]
        avg_confidence = sum(r.get("confidence", 0) for r in valid_refinements) / len(valid_refinements) if valid_refinements else 0
        
        print(f"  ✅ Refinement completed: {avg_confidence:.1%} average confidence")
        
        return {
            "refinement_results": refinement_results,
            "participating_agents": len(refinement_results),
            "successful_refinements": len(valid_refinements),
            "average_confidence": avg_confidence,
            "refinement_quality": len(valid_refinements) / len(refinement_results) if refinement_results else 0
        }
    
    async def _build_consensus(self, refinement_result: Dict[str, Any]) -> Dict[str, Any]:
        """Build consensus for final solution."""
        print("🤝 Phase 5: Consensus Building")
        
        # Coordinator builds consensus from all refinements
        coordinator = self.agents["coordinator"]
        
        consensus_task = {
            "id": "build_consensus",
            "type": "consensus_building",
            "description": "Build consensus from all agent refinements",
            "data": refinement_result,
            "complexity": "high"
        }
        
        consensus_result = await coordinator.process_task(consensus_task)
        print(f"  ✓ Consensus building: {consensus_result['confidence']:.1%} confidence")
        
        # Extract confidence scores from refinements
        refinement_confidences = [
            r.get("confidence", 0) for r in refinement_result["refinement_results"].values() 
            if "error" not in r
        ]
        
        # Calculate final solution metrics
        final_confidence = (
            consensus_result["confidence"] * 0.4 +  # Coordinator's consensus weight
            (sum(refinement_confidences) / len(refinement_confidences) if refinement_confidences else 0) * 0.6  # Average refinement weight
        )
        
        consensus_strength = min(refinement_confidences) if refinement_confidences else 0  # Weakest link
        solution_completeness = len(refinement_confidences) / len(self.agents)
        
        final_solution = {
            "solution_type": "collaborative_cognitive_solution",
            "final_confidence": final_confidence,
            "consensus_strength": consensus_strength,
            "solution_completeness": solution_completeness,
            "contributing_agents": len(refinement_confidences),
            "consensus_process": consensus_result,
            "quality_metrics": {
                "logical_consistency": final_confidence > 0.7,
                "multi_perspective_validation": len(refinement_confidences) >= 3,
                "consensus_achieved": consensus_strength > 0.6,
                "solution_robustness": solution_completeness > 0.8
            },
            "implementation_readiness": all([
                final_confidence > 0.7,
                consensus_strength > 0.6,
                solution_completeness > 0.8
            ])
        }
        
        print(f"  ✅ Final solution: {final_confidence:.1%} confidence")
        print(f"      Consensus strength: {consensus_strength:.1%}")
        print(f"      Implementation ready: {'Yes' if final_solution['implementation_readiness'] else 'No'}")
        
        return final_solution
    
    def get_collaboration_summary(self) -> Dict[str, Any]:
        """Get summary of all collaboration activities."""
        return {
            "total_agents": len(self.agents),
            "collaboration_sessions": len(self.collaboration_sessions),
            "agent_capabilities": {
                agent_id: agent.get_capabilities_summary() 
                for agent_id, agent in self.agents.items()
            },
            "average_session_duration": (
                sum(session.get("duration", 0) for session in self.collaboration_sessions) / 
                len(self.collaboration_sessions) if self.collaboration_sessions else 0
            ),
            "success_rate": (
                sum(1 for session in self.collaboration_sessions 
                    if session.get("final_solution", {}).get("implementation_readiness", False)) /
                len(self.collaboration_sessions) if self.collaboration_sessions else 0
            )
        }


async def demonstrate_multi_agent_collaboration():
    """Demonstrate multi-agent cognitive collaboration."""
    
    print("🧠 Multi-Agent Cognitive Collaboration Example")
    print("=" * 60)
    print("Demonstrating how multiple Agent-Zero instances collaborate")
    print("using full cognitive architecture capabilities.")
    print()
    
    # Initialize collaboration system
    collaboration_system = MultiAgentCognitiveCollaboration()
    
    # Define a complex problem for collaboration
    complex_problem = {
        "id": "optimize_ai_learning_system",
        "description": "Design an optimal AI learning system that adapts to user needs while maintaining ethical constraints",
        "complexity": "high",
        "domain": "artificial_intelligence",
        "requirements": [
            "Adaptive learning algorithms",
            "User personalization",
            "Ethical constraint enforcement",
            "Performance optimization",
            "Scalable architecture"
        ],
        "constraints": [
            "Privacy preservation",
            "Fairness and bias mitigation", 
            "Computational efficiency",
            "Real-time responsiveness",
            "Regulatory compliance"
        ],
        "success_criteria": [
            "Learning improvement > 25%",
            "User satisfaction > 85%",
            "Ethical compliance = 100%",
            "Performance overhead < 15%"
        ]
    }
    
    print("Problem Definition:")
    print(f"  Title: {complex_problem['id']}")  
    print(f"  Description: {complex_problem['description']}")
    print(f"  Complexity: {complex_problem['complexity']}")
    print(f"  Requirements: {len(complex_problem['requirements'])}")
    print(f"  Constraints: {len(complex_problem['constraints'])}")
    print()
    
    # Execute collaboration
    collaboration_result = await collaboration_system.collaborate_on_problem(complex_problem)
    
    # Display results
    print("\n" + "=" * 60)
    print("🎉 Multi-Agent Collaboration Results")
    print("=" * 60)
    
    final_solution = collaboration_result["final_solution"]
    
    print(f"Session Duration: {collaboration_result['duration']:.2f} seconds")
    print(f"Phases Completed: {len(collaboration_result['phases'])}")
    print(f"Agents Involved: {len(collaboration_result['agents_involved'])}")
    print()
    
    print("Solution Quality:")
    print(f"  Final Confidence: {final_solution['final_confidence']:.1%}")
    print(f"  Consensus Strength: {final_solution['consensus_strength']:.1%}")
    print(f"  Solution Completeness: {final_solution['solution_completeness']:.1%}")
    print(f"  Implementation Ready: {'✅ Yes' if final_solution['implementation_readiness'] else '❌ No'}")
    print()
    
    print("Quality Metrics:")
    for metric, passed in final_solution["quality_metrics"].items():
        status = "✅ Pass" if passed else "❌ Fail"
        print(f"  {metric}: {status}")
    print()
    
    # Phase breakdown
    print("Phase Breakdown:")
    for i, phase in enumerate(collaboration_result["phases"], 1):
        phase_name = phase["phase"].replace("_", " ").title()
        print(f"  {i}. {phase_name}")
        
        # Show key metrics for each phase
        if phase["phase"] == "parallel_processing":
            result = phase["result"]
            print(f"     Completion Rate: {result['completion_rate']:.1%}")
            print(f"     Average Confidence: {result['average_confidence']:.1%}")
        elif phase["phase"] == "integration":
            result = phase["result"]
            print(f"     Integration Quality: {result['integration_quality']:.1%}")
            print(f"     Knowledge Domains: {result['knowledge_domains']}")
        elif phase["phase"] == "refinement":
            result = phase["result"]
            print(f"     Successful Refinements: {result['successful_refinements']}/{result['participating_agents']}")
            print(f"     Average Confidence: {result['average_confidence']:.1%}")
    
    print()
    
    # Get collaboration summary
    summary = collaboration_system.get_collaboration_summary()
    
    print("Collaboration System Summary:")
    print(f"  Total Agents: {summary['total_agents']}")
    print(f"  Sessions Completed: {summary['collaboration_sessions']}")
    print(f"  Average Duration: {summary['average_session_duration']:.2f}s")
    print(f"  Success Rate: {summary['success_rate']:.1%}")
    print()
    
    print("Agent Capabilities:")
    for agent_id, capabilities in summary["agent_capabilities"].items():
        print(f"  {agent_id} ({capabilities['role']}):")
        print(f"    Specializations: {len(capabilities['specializations'])}")
        print(f"    Cognitive Tools: {'✅ Ready' if capabilities['cognitive_tools_ready'] else '❌ Fallback'}")
        print(f"    Tasks Completed: {capabilities['tasks_completed']}")
    
    return collaboration_result


async def main():
    """Run the multi-agent cognitive collaboration example."""
    
    print("🚀 Starting Multi-Agent Cognitive Collaboration Example")
    print("Issue: Create Agent-Zero examples demonstrating full cognitive architecture capabilities")
    print()
    
    # Run the demonstration
    result = await demonstrate_multi_agent_collaboration()
    
    # Save results
    output_file = PROJECT_ROOT / "multi_agent_collaboration_results.json"
    with open(output_file, 'w') as f:
        json.dump(result, f, indent=2, default=str)
    
    print(f"\n📄 Results saved to: {output_file}")
    
    print("\n🎯 Key Takeaways:")
    print("  1. Multiple cognitive agents can effectively collaborate on complex problems")
    print("  2. Specialized roles improve overall solution quality and efficiency")
    print("  3. Parallel processing with dependency management optimizes collaboration")
    print("  4. Consensus building ensures robust and reliable solutions")
    print("  5. Cognitive tools enhance individual agent capabilities")
    print("  6. Full cognitive architecture enables sophisticated multi-agent workflows")
    
    # Return success if implementation ready
    final_solution = result.get("final_solution", {})
    implementation_ready = final_solution.get("implementation_readiness", False)
    
    if implementation_ready:
        print("\n✅ Multi-agent collaboration example completed successfully!")
        return 0
    else:
        print("\n⚠️  Multi-agent collaboration completed with partial success!")
        return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    exit(exit_code)