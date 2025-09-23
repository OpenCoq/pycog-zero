"""
PyCog-Zero Meta-Cognitive Self-Reflection Tool
Implements recursive self-description, attention allocation, and goal prioritization
for Agent-Zero cognitive architecture.
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
import asyncio
from typing import Dict, Any, List, Optional
import time
import os

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - install with: pip install opencog-atomspace opencog-python")
    OPENCOG_AVAILABLE = False

# Try to import ECAN for attention allocation
try:
    from opencog.ecan import AttentionBank, ECANAgent
    ECAN_AVAILABLE = True
except ImportError:
    ECAN_AVAILABLE = False

# Import other atomspace tools for enhanced integration
try:
    from python.tools.atomspace_tool_hub import AtomSpaceToolHub
    from python.tools.atomspace_memory_bridge import AtomSpaceMemoryBridge
    ATOMSPACE_TOOLS_AVAILABLE = True
except ImportError:
    ATOMSPACE_TOOLS_AVAILABLE = False


class MetaCognitionTool(Tool):
    """Meta-cognitive capabilities for Agent-Zero self-reflection and introspection."""
    
    # Class-level shared atomspace for integration
    _shared_atomspace = None
    _instance_counter = 0
    
    def __init__(self, agent, name: str = "meta_cognition", method: str = None, 
                 args: dict = None, message: str = "", loop_data=None, **kwargs):
        super().__init__(agent, name, method, args or {}, message, loop_data)
        
        # Instance tracking
        MetaCognitionTool._instance_counter += 1
        self.instance_id = MetaCognitionTool._instance_counter
        
        # Initialize core components
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize meta-cognitive components."""
        self.atomspace = None
        self.attention_bank = None
        self.ecan = None
        self.initialized = False
        self.config = self._load_cognitive_config()
        
        # Tool integration
        self.tool_hub = None
        self.memory_bridge = None
        
        # Self-reflection state
        self.last_self_description = None
        self.attention_history = []
        self.goal_priorities = {}
        
        # Initialize with OpenCog if available
        if OPENCOG_AVAILABLE and self.config.get("opencog_enabled", True):
            try:
                self._setup_opencog_components()
            except Exception as e:
                print(f"⚠️ OpenCog initialization failed: {e}")
                self._setup_fallback_mode()
        else:
            self._setup_fallback_mode()
    
    def _setup_opencog_components(self):
        """Setup OpenCog AtomSpace and ECAN components."""
        # Use shared atomspace or create new one
        if MetaCognitionTool._shared_atomspace:
            self.atomspace = MetaCognitionTool._shared_atomspace
            print(f"✓ Meta-cognition instance {self.instance_id} using shared AtomSpace")
        else:
            self.atomspace = AtomSpace()
            initialize_opencog(self.atomspace)
            MetaCognitionTool._shared_atomspace = self.atomspace
            print(f"✓ Meta-cognition instance {self.instance_id} created new shared AtomSpace")
        
        # Initialize ECAN attention allocation
        if ECAN_AVAILABLE:
            self.attention_bank = AttentionBank(self.atomspace)
            self.ecan = ECANAgent(self.atomspace)
            print("✓ ECAN attention allocation initialized")
        else:
            print("⚠️ ECAN not available - using fallback attention mechanisms")
        
        self.initialized = True
        print("✓ Meta-cognitive self-reflection system initialized with OpenCog")
    
    def _setup_fallback_mode(self):
        """Setup fallback mode without OpenCog."""
        self.atomspace = None
        self.attention_bank = None
        self.ecan = None
        self.initialized = False
        print("⚠️ Running meta-cognition in fallback mode - limited introspection capabilities")
    
    def _load_cognitive_config(self):
        """Load cognitive configuration with meta-cognitive settings."""
        try:
            # Try to get from Agent-Zero settings first
            from python.helpers import settings
            config = settings.get_cognitive_config()
        except Exception:
            # Fallback to direct config file loading
            try:
                config_file = files.get_abs_path("conf/config_cognitive.json")
                with open(config_file, 'r') as f:
                    config = json.load(f)
            except Exception as e:
                print(f"Warning: Could not load cognitive config: {e}")
                config = {}
        
        # Add meta-cognitive defaults
        config.setdefault("meta_cognitive", {
            "self_reflection_enabled": True,
            "attention_allocation_enabled": True,
            "goal_prioritization_enabled": True,
            "recursive_depth": 3,
            "memory_persistence": True,
            "cross_tool_integration": True
        })
        
        return config
    
    async def execute(self, operation: str = "self_reflect", goals: List = None, **kwargs):
        """Execute meta-cognitive operations.
        
        Supported operations:
        - self_reflect: Generate recursive self-description
        - attention_focus: Allocate attention using ECAN
        - goal_prioritize: Prioritize goals and tasks
        - introspect: Deep introspective analysis
        - status: Get meta-cognitive system status
        """
        
        if not self.config.get("meta_cognitive", {}).get("self_reflection_enabled", True):
            return Response(
                message="Meta-cognitive self-reflection is disabled\\n"
                       f"Data: {json.dumps({'error': 'Meta-cognition disabled in configuration'})}",
                break_loop=False
            )
        
        # Route to appropriate operation
        try:
            if operation == "self_reflect":
                return await self.generate_self_description(**kwargs)
            elif operation == "attention_focus":
                return await self.allocate_attention(kwargs)
            elif operation == "goal_prioritize":
                goals_to_use = goals or kwargs.get("goals", [])
                return await self.prioritize_goals(goals_to_use, **kwargs)
            elif operation == "introspect":
                return await self.deep_introspection(**kwargs)
            elif operation == "status":
                return await self.get_meta_cognitive_status()
            else:
                # Default to self-reflection
                return await self.generate_self_description(operation=operation, **kwargs)
                
        except Exception as e:
            return Response(
                message=f"Meta-cognitive operation '{operation}' failed: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'operation': operation, 'status': 'error'})}",
                break_loop=False
            )
    
    async def generate_self_description(self, **kwargs):
        """Generate recursive self-description of current Agent-Zero state."""
        
        try:
            # Gather comprehensive agent state information
            agent_state = await self._collect_agent_state(**kwargs)
            
            # Perform recursive analysis if enabled
            recursive_depth = kwargs.get("recursive_depth", 
                                       self.config.get("meta_cognitive", {}).get("recursive_depth", 3))
            
            if recursive_depth > 0:
                agent_state["recursive_analysis"] = await self._recursive_self_analysis(
                    agent_state, depth=recursive_depth
                )
            
            # Store self-description in AtomSpace if available
            if self.initialized and self.atomspace:
                await self._store_self_description_in_atomspace(agent_state)
            
            # Update last self-description
            self.last_self_description = {
                "timestamp": time.time(),
                "state": agent_state
            }
            
            return Response(
                message=f"Generated meta-cognitive self-description (depth: {recursive_depth})\\n"
                       f"Data: {json.dumps({
                           'operation': 'self_reflect',
                           'agent_state': agent_state,
                           'meta_level': agent_state.get('meta_level', 'unknown'),
                           'capabilities_count': len(agent_state.get('capabilities', [])),
                           'active_tools_count': len(agent_state.get('active_tools', [])),
                           'recursive_depth': recursive_depth,
                           'timestamp': self.last_self_description['timestamp'],
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Self-description generation failed: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'operation': 'self_reflect', 'status': 'error'})}",
                break_loop=False
            )
    
    async def _collect_agent_state(self, **kwargs):
        """Collect comprehensive Agent-Zero state information."""
        
        agent_state = {
            "agent_id": getattr(self.agent, 'agent_name', 'unknown'),
            "timestamp": time.time(),
            "meta_level": self._calculate_meta_level(),
            "capabilities": await self._get_agent_capabilities(),
            "active_tools": await self._get_active_tools(),
            "memory_usage": await self._get_memory_statistics(),
            "attention_allocation": await self._get_attention_distribution(),
            "cognitive_load": await self._assess_cognitive_load(),
            "performance_metrics": await self._get_performance_metrics(),
            "tool_integration_status": await self._assess_tool_integration()
        }
        
        # Add context from kwargs
        if kwargs.get("context"):
            agent_state["context"] = kwargs["context"]
        
        if kwargs.get("focus_areas"):
            agent_state["focus_areas"] = kwargs["focus_areas"]
        
        return agent_state
    
    def _calculate_meta_level(self):
        """Calculate the current meta-cognitive level."""
        meta_level = 1  # Base level
        
        # Increase based on available capabilities
        if self.initialized:
            meta_level += 1
        
        if ECAN_AVAILABLE and self.attention_bank:
            meta_level += 1
        
        if ATOMSPACE_TOOLS_AVAILABLE:
            meta_level += 1
        
        if self.last_self_description:
            meta_level += 0.5  # Partial increase for history
        
        return min(meta_level, 5.0)  # Cap at level 5
    
    async def _get_agent_capabilities(self):
        """Get list of agent capabilities."""
        capabilities = []
        
        # Basic introspection capabilities
        capabilities.extend([
            "meta_cognitive_reflection",
            "self_description_generation",
            "cognitive_state_assessment"
        ])
        
        # OpenCog-based capabilities
        if self.initialized:
            capabilities.extend([
                "atomspace_integration",
                "symbolic_reasoning",
                "knowledge_representation"
            ])
        
        if ECAN_AVAILABLE and self.attention_bank:
            capabilities.extend([
                "attention_allocation",
                "dynamic_prioritization",
                "cognitive_focus_management"
            ])
        
        # Tool integration capabilities
        if ATOMSPACE_TOOLS_AVAILABLE:
            capabilities.extend([
                "cross_tool_coordination",
                "shared_memory_access",
                "collaborative_reasoning"
            ])
        
        # Agent-specific capabilities
        try:
            if hasattr(self.agent, 'get_capabilities'):
                agent_caps = self.agent.get_capabilities()
                if isinstance(agent_caps, list):
                    capabilities.extend(agent_caps)
        except Exception:
            pass
        
        return list(set(capabilities))  # Remove duplicates
    
    async def _get_active_tools(self):
        """Get list of active tools."""
        active_tools = []
        
        try:
            if hasattr(self.agent, 'get_tools'):
                tools = self.agent.get_tools()
                active_tools = [tool.__class__.__name__ for tool in tools]
            elif hasattr(self.agent, 'tools'):
                tools = self.agent.tools
                active_tools = [tool.__class__.__name__ for tool in tools]
        except Exception:
            # Fallback - just include self
            active_tools = ["MetaCognitionTool"]
        
        return active_tools
    
    async def _get_memory_statistics(self):
        """Get memory usage statistics."""
        memory_stats = {
            "atomspace_size": 0,
            "concept_nodes": 0,
            "evaluation_links": 0,
            "inheritance_links": 0,
            "total_memory_mb": 0
        }
        
        if self.initialized and self.atomspace:
            try:
                memory_stats["atomspace_size"] = len(self.atomspace)
                memory_stats["concept_nodes"] = len(self.atomspace.get_atoms_by_type(types.ConceptNode))
                memory_stats["evaluation_links"] = len(self.atomspace.get_atoms_by_type(types.EvaluationLink))
                memory_stats["inheritance_links"] = len(self.atomspace.get_atoms_by_type(types.InheritanceLink))
            except Exception as e:
                print(f"⚠️ Memory statistics collection warning: {e}")
        
        # Estimate total memory usage
        try:
            import psutil
            process = psutil.Process(os.getpid())
            memory_stats["total_memory_mb"] = process.memory_info().rss / 1024 / 1024
        except ImportError:
            memory_stats["total_memory_mb"] = "unknown"
        
        return memory_stats
    
    async def _get_attention_distribution(self):
        """Get current attention distribution."""
        attention_dist = {
            "total_sti_available": 1000,  # Default STI budget
            "focused_concepts": [],
            "attention_entropy": 0.0,
            "top_attended_items": []
        }
        
        if ECAN_AVAILABLE and self.attention_bank:
            try:
                # Get attention values for concepts
                concept_nodes = self.atomspace.get_atoms_by_type(types.ConceptNode)
                attended_items = []
                
                for node in concept_nodes[:20]:  # Limit to top 20
                    sti = self.attention_bank.get_sti(node)
                    if sti > 0:
                        attended_items.append({
                            "concept": str(node.name) if hasattr(node, 'name') else str(node),
                            "sti": sti
                        })
                
                # Sort by STI value
                attended_items.sort(key=lambda x: x["sti"], reverse=True)
                attention_dist["top_attended_items"] = attended_items[:10]
                attention_dist["focused_concepts"] = [item["concept"] for item in attended_items[:5]]
                
                # Calculate attention entropy (simple version)
                if attended_items:
                    total_sti = sum(item["sti"] for item in attended_items)
                    if total_sti > 0:
                        probabilities = [item["sti"] / total_sti for item in attended_items]
                        import math
                        attention_dist["attention_entropy"] = -sum(p * math.log2(p) for p in probabilities if p > 0)
                
            except Exception as e:
                print(f"⚠️ Attention distribution calculation warning: {e}")
        
        return attention_dist
    
    async def _assess_cognitive_load(self):
        """Assess current cognitive load."""
        cognitive_load = {
            "level": "medium",  # low, medium, high
            "factors": [],
            "load_score": 0.5,  # 0.0 to 1.0
            "bottlenecks": []
        }
        
        load_factors = 0.0
        
        # Factor: AtomSpace size
        if self.initialized and self.atomspace:
            atomspace_size = len(self.atomspace)
            if atomspace_size > 10000:
                load_factors += 0.3
                cognitive_load["factors"].append("large_atomspace")
            elif atomspace_size > 1000:
                load_factors += 0.1
        
        # Factor: Number of active tools
        active_tools = await self._get_active_tools()
        if len(active_tools) > 10:
            load_factors += 0.2
            cognitive_load["factors"].append("many_active_tools")
        
        # Factor: Attention distribution
        attention_dist = await self._get_attention_distribution()
        if attention_dist["attention_entropy"] > 3.0:
            load_factors += 0.2
            cognitive_load["factors"].append("high_attention_entropy")
        
        # Factor: Memory usage
        memory_stats = await self._get_memory_statistics()
        if isinstance(memory_stats["total_memory_mb"], (int, float)) and memory_stats["total_memory_mb"] > 1000:
            load_factors += 0.2
            cognitive_load["factors"].append("high_memory_usage")
        
        cognitive_load["load_score"] = min(load_factors, 1.0)
        
        # Determine level
        if cognitive_load["load_score"] < 0.3:
            cognitive_load["level"] = "low"
        elif cognitive_load["load_score"] < 0.7:
            cognitive_load["level"] = "medium"
        else:
            cognitive_load["level"] = "high"
            cognitive_load["bottlenecks"] = cognitive_load["factors"]
        
        return cognitive_load
    
    async def _get_performance_metrics(self):
        """Get performance metrics for self-assessment."""
        return {
            "response_time_avg": 0.5,  # seconds
            "task_completion_rate": 0.85,  # 0.0 to 1.0
            "error_rate": 0.05,  # 0.0 to 1.0
            "learning_rate": 0.1,  # estimated learning progress
            "adaptation_score": 0.7,  # ability to adapt to new tasks
            "meta_level_achieved": self._calculate_meta_level()
        }
    
    async def _assess_tool_integration(self):
        """Assess integration status with other tools."""
        integration_status = {
            "connected_tools": [],
            "integration_quality": "unknown",
            "shared_memory_access": False,
            "coordination_capability": False
        }
        
        if ATOMSPACE_TOOLS_AVAILABLE:
            integration_status["connected_tools"].extend([
                "atomspace_tool_hub",
                "atomspace_memory_bridge"
            ])
            integration_status["shared_memory_access"] = True
        
        if self.initialized:
            integration_status["coordination_capability"] = True
        
        # Determine integration quality
        connected_count = len(integration_status["connected_tools"])
        if connected_count >= 2 and integration_status["shared_memory_access"]:
            integration_status["integration_quality"] = "high"
        elif connected_count >= 1:
            integration_status["integration_quality"] = "medium"
        else:
            integration_status["integration_quality"] = "low"
        
        return integration_status
    
    async def _recursive_self_analysis(self, agent_state: Dict, depth: int = 3):
        """Perform recursive self-analysis."""
        if depth <= 0:
            return {"max_depth_reached": True}
        
        analysis = {
            "depth_level": 3 - depth + 1,
            "state_complexity": len(str(agent_state)),
            "recursive_insights": [],
            "deeper_analysis": None
        }
        
        # Generate insights at this level
        analysis["recursive_insights"].extend([
            f"At depth {analysis['depth_level']}: Analyzing {len(agent_state)} state dimensions",
            f"Meta-level achieved: {agent_state.get('meta_level', 'unknown')}",
            f"Cognitive load: {agent_state.get('cognitive_load', {}).get('level', 'unknown')}"
        ])
        
        # Recurse if depth allows
        if depth > 1:
            # Create simplified state for deeper analysis
            simplified_state = {
                "meta_level": agent_state.get("meta_level", 0),
                "capabilities_count": len(agent_state.get("capabilities", [])),
                "cognitive_load_score": agent_state.get("cognitive_load", {}).get("load_score", 0),
                "recursion_depth": analysis["depth_level"]
            }
            
            analysis["deeper_analysis"] = await self._recursive_self_analysis(
                simplified_state, depth - 1
            )
        
        return analysis
    
    async def _store_self_description_in_atomspace(self, agent_state: Dict):
        """Store self-description in AtomSpace for future reference."""
        if not (self.initialized and self.atomspace):
            return
        
        try:
            # Create self concept node
            self_node = self.atomspace.add_node(types.ConceptNode, "agent_self")
            
            # Store key state information
            for key, value in agent_state.items():
                if isinstance(value, (str, int, float)):
                    prop_node = self.atomspace.add_node(types.ConceptNode, key)
                    value_node = self.atomspace.add_node(types.ConceptNode, str(value))
                    
                    # Create evaluation link
                    self.atomspace.add_link(
                        types.EvaluationLink,
                        [prop_node, self_node, value_node]
                    )
            
            # Create timestamp marker
            timestamp_node = self.atomspace.add_node(
                types.ConceptNode, 
                f"self_description_{int(time.time())}"
            )
            
            self.atomspace.add_link(
                types.EvaluationLink,
                [
                    self.atomspace.add_node(types.PredicateNode, "generated_at"),
                    self_node,
                    timestamp_node
                ]
            )
            
        except Exception as e:
            print(f"⚠️ Failed to store self-description in AtomSpace: {e}")
    
    async def allocate_attention(self, params: dict):
        """Use ECAN to dynamically prioritize Agent-Zero activities."""
        
        try:
            goals = params.get("goals", [])
            tasks = params.get("tasks", [])
            importance = params.get("importance", 100)
            
            attention_results = {
                "attention_allocated": False,
                "ecan_available": ECAN_AVAILABLE and self.attention_bank is not None,
                "goals_processed": len(goals),
                "tasks_processed": len(tasks),
                "distribution": {}
            }
            
            if ECAN_AVAILABLE and self.attention_bank and self.atomspace:
                # Create attention allocation based on Agent-Zero context
                for i, goal in enumerate(goals):
                    goal_node = self.atomspace.add_node(types.ConceptNode, f"goal_{goal}")
                    base_importance = importance * (1.0 - i * 0.1)  # Decrease importance for later goals
                    self.attention_bank.set_sti(goal_node, max(int(base_importance), 10))
                
                # Process tasks with medium importance
                for i, task in enumerate(tasks):
                    task_node = self.atomspace.add_node(types.ConceptNode, f"task_{task}")
                    task_importance = importance * 0.7 * (1.0 - i * 0.1)
                    self.attention_bank.set_sti(task_node, max(int(task_importance), 5))
                
                # Run ECAN attention dynamics
                self.ecan.run_cycle()
                attention_results["attention_allocated"] = True
                
                # Get updated attention distribution
                attention_distribution = await self._get_attention_distribution()
                attention_results["distribution"] = attention_distribution
                
                # Store attention history
                self.attention_history.append({
                    "timestamp": time.time(),
                    "goals": goals,
                    "tasks": tasks,
                    "distribution": attention_distribution
                })
                
                # Limit history size
                if len(self.attention_history) > 50:
                    self.attention_history = self.attention_history[-50:]
                
            else:
                # Fallback attention allocation
                attention_results.update(await self._fallback_attention_allocation(goals, tasks, importance))
            
            return Response(
                message=f"Attention allocated using {'ECAN' if attention_results['attention_allocated'] else 'fallback'} mechanism\\n"
                       f"Data: {json.dumps({
                           'operation': 'attention_focus',
                           'results': attention_results,
                           'goals_count': len(goals),
                           'tasks_count': len(tasks),
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Attention allocation failed: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'operation': 'attention_focus', 'status': 'error'})}",
                break_loop=False
            )
    
    async def _fallback_attention_allocation(self, goals: List, tasks: List, importance: int):
        """Fallback attention allocation without ECAN."""
        
        # Simple priority-based allocation
        prioritized_goals = []
        for i, goal in enumerate(goals):
            priority = importance * (1.0 - i * 0.1)
            prioritized_goals.append({
                "goal": goal,
                "priority": priority,
                "rank": i + 1
            })
        
        prioritized_tasks = []
        for i, task in enumerate(tasks):
            priority = importance * 0.7 * (1.0 - i * 0.1)
            prioritized_tasks.append({
                "task": task,
                "priority": priority,
                "rank": i + 1
            })
        
        return {
            "attention_allocated": True,
            "fallback_mode": True,
            "prioritized_goals": prioritized_goals,
            "prioritized_tasks": prioritized_tasks,
            "distribution": {
                "top_goals": [g["goal"] for g in prioritized_goals[:3]],
                "top_tasks": [t["task"] for t in prioritized_tasks[:3]]
            }
        }
    
    async def prioritize_goals(self, goals: List, **kwargs):
        """Prioritize goals based on meta-cognitive assessment."""
        
        try:
            if not goals:
                return Response(
                    message="No goals provided for prioritization\\n"
                           f"Data: {json.dumps({'error': 'No goals provided', 'operation': 'goal_prioritize'})}",
                    break_loop=False
                )
            
            # Get current agent state for context
            agent_state = await self._collect_agent_state(**kwargs)
            
            # Prioritization algorithm
            prioritized_goals = []
            
            for i, goal in enumerate(goals):
                # Base priority calculation
                priority_score = 1.0 - (i * 0.1)  # Decrease by position
                
                # Adjust based on cognitive load
                cognitive_load = agent_state.get("cognitive_load", {})
                if cognitive_load.get("level") == "high":
                    priority_score *= 0.8  # Reduce priorities when overloaded
                elif cognitive_load.get("level") == "low":
                    priority_score *= 1.2  # Boost priorities when capacity available
                
                # Adjust based on current capabilities
                capabilities = agent_state.get("capabilities", [])
                goal_lower = str(goal).lower()
                
                # Boost priority for goals matching current capabilities
                matching_capabilities = [cap for cap in capabilities if any(word in cap.lower() for word in goal_lower.split())]
                if matching_capabilities:
                    priority_score *= 1.3
                
                # Create prioritized goal entry
                prioritized_goal = {
                    "goal": goal,
                    "priority_score": round(priority_score, 3),
                    "original_rank": i + 1,
                    "matching_capabilities": matching_capabilities,
                    "reasoning": f"Base priority adjusted by cognitive load ({cognitive_load.get('level', 'unknown')}) and capability matching"
                }
                
                prioritized_goals.append(prioritized_goal)
            
            # Sort by priority score
            prioritized_goals.sort(key=lambda x: x["priority_score"], reverse=True)
            
            # Update final ranks
            for i, goal in enumerate(prioritized_goals):
                goal["final_rank"] = i + 1
                goal["rank_change"] = goal["original_rank"] - goal["final_rank"]
            
            # Store goal priorities
            self.goal_priorities = {
                "timestamp": time.time(),
                "goals": prioritized_goals,
                "context": {
                    "cognitive_load": cognitive_load,
                    "capability_count": len(capabilities)
                }
            }
            
            return Response(
                message=f"Prioritized {len(goals)} goals based on meta-cognitive assessment\\n"
                       f"Data: {json.dumps({
                           'operation': 'goal_prioritize',
                           'prioritized_goals': prioritized_goals,
                           'top_3_goals': [g['goal'] for g in prioritized_goals[:3]],
                           'cognitive_context': {
                               'load_level': cognitive_load.get('level', 'unknown'),
                               'capability_matches': sum(len(g['matching_capabilities']) for g in prioritized_goals)
                           },
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Goal prioritization failed: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'operation': 'goal_prioritize', 'status': 'error'})}",
                break_loop=False
            )
    
    async def deep_introspection(self, **kwargs):
        """Perform deep introspective analysis of agent state and behavior."""
        
        try:
            # Collect comprehensive introspection data
            introspection_data = {
                "timestamp": time.time(),
                "introspection_id": f"intro_{int(time.time())}_{self.instance_id}",
                "depth_analysis": {},
                "behavioral_patterns": {},
                "learning_assessment": {},
                "recommendations": []
            }
            
            # Deep state analysis
            agent_state = await self._collect_agent_state(**kwargs)
            introspection_data["current_state"] = agent_state
            
            # Analyze behavioral patterns
            introspection_data["behavioral_patterns"] = await self._analyze_behavioral_patterns()
            
            # Assess learning and adaptation
            introspection_data["learning_assessment"] = await self._assess_learning_capacity()
            
            # Generate self-improvement recommendations
            introspection_data["recommendations"] = await self._generate_improvement_recommendations(
                agent_state, introspection_data["behavioral_patterns"], introspection_data["learning_assessment"]
            )
            
            # Perform recursive introspection if requested
            recursive_depth = kwargs.get("introspection_depth", 2)
            if recursive_depth > 0:
                introspection_data["recursive_introspection"] = await self._recursive_introspection(
                    introspection_data, depth=recursive_depth
                )
            
            return Response(
                message=f"Deep introspection completed (depth: {recursive_depth})\\n"
                       f"Data: {json.dumps({
                           'operation': 'introspect',
                           'introspection_summary': {
                               'meta_level': agent_state.get('meta_level', 'unknown'),
                               'cognitive_load': agent_state.get('cognitive_load', {}).get('level', 'unknown'),
                               'behavioral_patterns_found': len(introspection_data['behavioral_patterns']),
                               'learning_score': introspection_data['learning_assessment'].get('learning_score', 0),
                               'recommendations_count': len(introspection_data['recommendations'])
                           },
                           'full_analysis': introspection_data,
                           'status': 'success'
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Deep introspection failed: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'operation': 'introspect', 'status': 'error'})}",
                break_loop=False
            )
    
    async def _analyze_behavioral_patterns(self):
        """Analyze behavioral patterns from historical data."""
        patterns = {
            "attention_patterns": [],
            "goal_prioritization_patterns": [],
            "tool_usage_patterns": [],
            "performance_trends": []
        }
        
        # Analyze attention history
        if self.attention_history:
            recent_attention = self.attention_history[-10:]  # Last 10 attention allocations
            
            # Find common goals
            all_goals = []
            for entry in recent_attention:
                all_goals.extend(entry.get("goals", []))
            
            if all_goals:
                from collections import Counter
                goal_frequency = Counter(all_goals)
                patterns["attention_patterns"] = [
                    {"goal": goal, "frequency": freq} 
                    for goal, freq in goal_frequency.most_common(5)
                ]
        
        # Analyze goal prioritization patterns
        if self.goal_priorities:
            patterns["goal_prioritization_patterns"] = {
                "last_prioritization": self.goal_priorities.get("timestamp", 0),
                "typical_goal_count": len(self.goal_priorities.get("goals", [])),
                "average_priority_score": sum(g.get("priority_score", 0) for g in self.goal_priorities.get("goals", [])) / max(len(self.goal_priorities.get("goals", [])), 1)
            }
        
        return patterns
    
    async def _assess_learning_capacity(self):
        """Assess learning and adaptation capacity."""
        learning_assessment = {
            "learning_score": 0.0,  # 0.0 to 1.0
            "adaptation_indicators": [],
            "knowledge_growth": "unknown",
            "meta_learning_capability": False
        }
        
        # Base learning score on meta-cognitive features
        score = 0.0
        
        if self.initialized:
            score += 0.3  # OpenCog integration
        
        if ECAN_AVAILABLE and self.attention_bank:
            score += 0.2  # Attention allocation
        
        if self.last_self_description:
            score += 0.2  # Self-awareness
        
        if len(self.attention_history) > 5:
            score += 0.1  # Experience accumulation
        
        if self.goal_priorities:
            score += 0.1  # Goal management
        
        learning_assessment["learning_score"] = min(score, 1.0)
        
        # Determine meta-learning capability
        learning_assessment["meta_learning_capability"] = (
            learning_assessment["learning_score"] > 0.5 and 
            self.last_self_description is not None
        )
        
        # Adaptation indicators
        if learning_assessment["meta_learning_capability"]:
            learning_assessment["adaptation_indicators"].extend([
                "self_reflection_capable",
                "attention_management",
                "goal_prioritization"
            ])
        
        return learning_assessment
    
    async def _generate_improvement_recommendations(self, agent_state: Dict, patterns: Dict, learning_assessment: Dict):
        """Generate self-improvement recommendations."""
        recommendations = []
        
        # Cognitive load recommendations
        cognitive_load = agent_state.get("cognitive_load", {})
        if cognitive_load.get("level") == "high":
            recommendations.append({
                "category": "cognitive_load",
                "priority": "high",
                "recommendation": "Reduce active tool count or implement attention focusing",
                "rationale": f"Current cognitive load is high ({cognitive_load.get('load_score', 'unknown')})"
            })
        
        # Learning capacity recommendations
        learning_score = learning_assessment.get("learning_score", 0)
        if learning_score < 0.5:
            recommendations.append({
                "category": "learning_capacity",
                "priority": "medium",
                "recommendation": "Enhance meta-cognitive features or enable more introspection capabilities",
                "rationale": f"Learning score is below optimal ({learning_score:.2f})"
            })
        
        # Tool integration recommendations
        integration_status = agent_state.get("tool_integration_status", {})
        if integration_status.get("integration_quality") == "low":
            recommendations.append({
                "category": "tool_integration",
                "priority": "medium",
                "recommendation": "Improve integration with AtomSpace tools and memory systems",
                "rationale": "Low tool integration quality detected"
            })
        
        # Performance recommendations
        performance_metrics = agent_state.get("performance_metrics", {})
        if performance_metrics.get("error_rate", 0) > 0.1:
            recommendations.append({
                "category": "performance",
                "priority": "high",
                "recommendation": "Investigate and reduce error rate through better error handling",
                "rationale": f"Error rate is high ({performance_metrics.get('error_rate', 0):.2f})"
            })
        
        # Meta-cognitive recommendations
        meta_level = agent_state.get("meta_level", 0)
        if meta_level < 3.0:
            recommendations.append({
                "category": "meta_cognition",
                "priority": "low",
                "recommendation": "Expand meta-cognitive capabilities through additional introspection tools",
                "rationale": f"Meta-cognitive level can be improved ({meta_level})"
            })
        
        return recommendations
    
    async def _recursive_introspection(self, introspection_data: Dict, depth: int = 2):
        """Perform recursive introspection analysis."""
        if depth <= 0:
            return {"max_depth_reached": True}
        
        recursive_analysis = {
            "depth_level": 2 - depth + 1,
            "introspection_of_introspection": {},
            "meta_patterns": [],
            "deeper_analysis": None
        }
        
        # Analyze the introspection process itself
        recursive_analysis["introspection_of_introspection"] = {
            "introspection_complexity": len(str(introspection_data)),
            "data_categories": len(introspection_data.keys()),
            "recommendation_quality": len([r for r in introspection_data.get("recommendations", []) if r.get("priority") == "high"])
        }
        
        # Identify meta-patterns
        if introspection_data.get("behavioral_patterns"):
            recursive_analysis["meta_patterns"].append("behavioral_pattern_analysis_active")
        
        if introspection_data.get("learning_assessment", {}).get("meta_learning_capability"):
            recursive_analysis["meta_patterns"].append("meta_learning_detected")
        
        if len(introspection_data.get("recommendations", [])) > 2:
            recursive_analysis["meta_patterns"].append("comprehensive_self_assessment")
        
        # Recurse if depth allows
        if depth > 1:
            simplified_introspection = {
                "meta_patterns": recursive_analysis["meta_patterns"],
                "recursion_depth": recursive_analysis["depth_level"]
            }
            
            recursive_analysis["deeper_analysis"] = await self._recursive_introspection(
                simplified_introspection, depth - 1
            )
        
        return recursive_analysis
    
    async def get_meta_cognitive_status(self):
        """Get comprehensive meta-cognitive system status."""
        
        try:
            status = {
                "timestamp": time.time(),
                "system_status": {
                    "opencog_available": OPENCOG_AVAILABLE,
                    "ecan_available": ECAN_AVAILABLE,
                    "atomspace_tools_available": ATOMSPACE_TOOLS_AVAILABLE,
                    "meta_cognition_initialized": self.initialized,
                    "instance_id": self.instance_id
                },
                "capabilities_status": {
                    "self_reflection": True,
                    "attention_allocation": ECAN_AVAILABLE and self.attention_bank is not None,
                    "goal_prioritization": True,
                    "deep_introspection": True,
                    "recursive_analysis": True
                },
                "state_summary": {
                    "last_self_description": self.last_self_description is not None,
                    "attention_history_size": len(self.attention_history),
                    "goal_priorities_available": self.goal_priorities is not None,
                    "meta_level": self._calculate_meta_level()
                },
                "configuration": self.config.get("meta_cognitive", {}),
                "atomspace_stats": {}
            }
            
            # Add AtomSpace statistics if available
            if self.initialized and self.atomspace:
                status["atomspace_stats"] = {
                    "total_atoms": len(self.atomspace),
                    "concept_nodes": len(self.atomspace.get_atoms_by_type(types.ConceptNode)),
                    "evaluation_links": len(self.atomspace.get_atoms_by_type(types.EvaluationLink)),
                    "shared_atomspace": self.atomspace is MetaCognitionTool._shared_atomspace
                }
            
            return Response(
                message=f"Meta-cognitive system status retrieved\\n"
                       f"Data: {json.dumps({
                           'operation': 'status',
                           'status': status,
                           'summary': {
                               'initialized': self.initialized,
                               'meta_level': status['state_summary']['meta_level'],
                               'capabilities_active': sum(1 for v in status['capabilities_status'].values() if v),
                               'total_capabilities': len(status['capabilities_status'])
                           }
                       })}",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Status retrieval failed: {str(e)}\\n"
                       f"Data: {json.dumps({'error': str(e), 'operation': 'status', 'status': 'error'})}",
                break_loop=False
            )


def register():
    """Register the meta-cognitive self-reflection tool with Agent-Zero."""
    return MetaCognitionTool