"""
Multi-Agent Collaboration Tool for PyCog-Zero
Integrates multi-agent cognitive collaboration capabilities with Agent-Zero framework.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import multi-agent components first
try:
    from python.helpers.multi_agent import CognitiveMultiAgentSystem, create_cognitive_agent_network
    MULTI_AGENT_AVAILABLE = True
except ImportError as e:
    print(f"Multi-agent components not available: {e}")
    MULTI_AGENT_AVAILABLE = False

# Import Agent-Zero components with fallback
try:
    from python.helpers.tool import Tool, Response
    from python.helpers import files
    AGENT_ZERO_AVAILABLE = True
except ImportError as e:
    print(f"Agent-Zero components not available: {e}")
    AGENT_ZERO_AVAILABLE = False
    # Create fallback classes
    class Tool:
        def __init__(self, agent, name, method=None, args=None, message="", loop_data=None, **kwargs):
            self.agent = agent
            self.name = name
            self.method = method
            self.args = args or {}
            self.message = message
            self.loop_data = loop_data
    
    class Response:
        def __init__(self, message="", data=None, break_loop=False):
            self.message = message
            self.data = data
            self.break_loop = break_loop

# Try to import cognitive tools for integration
try:
    from python.tools.cognitive_reasoning import CognitiveReasoningTool
    from python.tools.cognitive_memory import CognitiveMemoryTool
    from python.tools.meta_cognition import MetaCognitionTool
    COGNITIVE_TOOLS_AVAILABLE = True
except ImportError:
    COGNITIVE_TOOLS_AVAILABLE = False


class MultiAgentCollaborationTool(Tool):
    """Tool for multi-agent cognitive collaboration within Agent-Zero framework."""
    
    # Class-level shared multi-agent system
    _shared_system: Optional[CognitiveMultiAgentSystem] = None
    
    def __init__(self, agent, name: str = "multi_agent_collaboration", method: str = None,
                 args: dict = None, message: str = "", loop_data=None, **kwargs):
        super().__init__(agent, name, method, args or {}, message, loop_data)
        
        # Initialize or reuse shared multi-agent system
        if not MultiAgentCollaborationTool._shared_system:
            MultiAgentCollaborationTool._shared_system = create_cognitive_agent_network(
                num_agents=kwargs.get('num_agents', 3)
            )
        
        self.multi_agent_system = MultiAgentCollaborationTool._shared_system
        self.config = self._load_config()
        
        # Register current agent with multi-agent system if not already registered
        self._register_current_agent()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load multi-agent configuration."""
        default_config = {
            "max_agents": 5,
            "collaboration_timeout": 300,  # 5 minutes
            "reasoning_integration": True,
            "memory_sharing": True,
            "metacognitive_feedback": True,
            "coordination_strategy": "distributed_cognitive_reasoning"
        }
        
        try:
            if AGENT_ZERO_AVAILABLE:
                config_path = files.get_abs_path("conf/config_multi_agent.json")
            else:
                config_path = Path(__file__).parent.parent.parent / "conf" / "config_multi_agent.json"
                
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    loaded_config = json.load(f)
                    default_config.update(loaded_config)
        except Exception:
            pass
        
        return default_config
    
    def _register_current_agent(self):
        """Register the current Agent-Zero instance with the multi-agent system."""
        try:
            agent_id = f"agent_{getattr(self.agent, 'number', 0)}_{getattr(self.agent, 'agent_name', 'unknown')}"
            
            # Determine agent capabilities based on available tools
            capabilities = ["general_reasoning", "task_execution"]
            
            if COGNITIVE_TOOLS_AVAILABLE:
                capabilities.extend(["cognitive_reasoning", "memory_management", "meta_cognition"])
            
            # Check if agent already registered
            if agent_id not in self.multi_agent_system.agents:
                self.multi_agent_system.register_agent(
                    agent_id=agent_id,
                    name=f"Agent-Zero-{getattr(self.agent, 'number', 0)}",
                    role="cognitive_agent",
                    capabilities=capabilities,
                    cognitive_config={
                        "agent_zero_integration": True,
                        "tool_integration": COGNITIVE_TOOLS_AVAILABLE
                    }
                )
        except Exception as e:
            print(f"Could not register agent with multi-agent system: {e}")
    
    async def execute(self, operation: str = "status", **kwargs) -> Response:
        """
        Execute multi-agent collaboration operations.
        
        Operations:
        - status: Get multi-agent system status
        - collaborate: Start collaborative reasoning task
        - coordinate: Coordinate specific task
        - agents: List registered agents
        - create_task: Create new collaboration task
        - workflow: Run end-to-end workflow
        """
        
        try:
            if operation == "status":
                return await self._get_system_status(**kwargs)
            elif operation == "collaborate":
                return await self._collaborative_reasoning(**kwargs)
            elif operation == "coordinate":
                return await self._coordinate_task(**kwargs)
            elif operation == "agents":
                return await self._list_agents(**kwargs)
            elif operation == "create_task":
                return await self._create_collaboration_task(**kwargs)
            elif operation == "workflow":
                return await self._run_end_to_end_workflow(**kwargs)
            else:
                return Response(
                    message=f"Unknown operation: {operation}. Available: status, collaborate, coordinate, agents, create_task, workflow",
                    break_loop=False
                )
        
        except Exception as e:
            return Response(
                message=f"Multi-agent collaboration error: {str(e)}",
                break_loop=False
            )
    
    async def _get_system_status(self, **kwargs) -> Response:
        """Get current multi-agent system status."""
        
        status = self.multi_agent_system.get_coordination_status()
        
        # Add tool-specific status information
        status.update({
            "cognitive_tools_available": COGNITIVE_TOOLS_AVAILABLE,
            "agent_zero_integration": AGENT_ZERO_AVAILABLE,
            "config": {
                "max_agents": self.config["max_agents"],
                "collaboration_timeout": self.config["collaboration_timeout"],
                "coordination_strategy": self.config["coordination_strategy"]
            }
        })
        
        message = f"Multi-Agent System Status:\n"
        message += f"• Total Agents: {status['total_agents']}\n"
        message += f"• Active Agents: {status['active_agents']}\n"
        message += f"• Active Tasks: {status['active_tasks']}\n"
        message += f"• Completed Tasks: {status['completed_tasks']}\n"
        message += f"• System Status: {status['system_status']}\n"
        message += f"• Cognitive Tools: {'✓' if COGNITIVE_TOOLS_AVAILABLE else '✗'}\n"
        message += f"• Agent-Zero Integration: {'✓' if AGENT_ZERO_AVAILABLE else '✗'}"
        
        return Response(
            message=message,
            data=status
        )
    
    async def _collaborative_reasoning(self, problem: str = None, **kwargs) -> Response:
        """Perform collaborative reasoning across multiple agents."""
        
        if not problem:
            problem = kwargs.get('query', 'Optimize multi-agent cognitive collaboration strategies')
        
        # Start collaborative reasoning
        reasoning_result = await self.multi_agent_system.collaborative_reasoning(
            problem=problem,
            context=kwargs.get('context', {})
        )
        
        message = f"Collaborative Reasoning Completed:\n"
        message += f"• Problem: {problem}\n"
        message += f"• Agent Contributions: {reasoning_result['agent_contributions']}\n"
        message += f"• Overall Confidence: {reasoning_result['overall_confidence']:.2f}\n"
        message += f"• Total Reasoning Steps: {reasoning_result['total_reasoning_steps']}\n"
        message += f"• Quality Score: {reasoning_result['quality_score']:.2f}\n"
        message += f"\nCognitive Insights:\n"
        
        for insight in reasoning_result['cognitive_insights'][:3]:  # Show first 3
            message += f"  - {insight}\n"
        
        return Response(
            message=message,
            data=reasoning_result
        )
    
    async def _coordinate_task(self, task_description: str = None, 
                             capabilities: List[str] = None, **kwargs) -> Response:
        """Coordinate a specific multi-agent task."""
        
        if not task_description:
            task_description = "Generic multi-agent coordination task"
        
        if not capabilities:
            capabilities = ["general_reasoning", "analysis"]
        
        # Create and coordinate task
        task = self.multi_agent_system.create_collaboration_task(
            description=task_description,
            required_capabilities=capabilities,
            priority=kwargs.get('priority', 1)
        )
        
        coordination_plan = await self.multi_agent_system.coordinate_task(task.task_id)
        
        message = f"Task Coordination Completed:\n"
        message += f"• Task ID: {task.task_id}\n"
        message += f"• Description: {task_description}\n"
        message += f"• Assigned Agents: {len(task.assigned_agents)}\n"
        message += f"• Required Capabilities: {', '.join(capabilities)}\n"
        message += f"• Status: {task.status}\n"
        message += f"• Coordination Strategy: {task.coordination_strategy}"
        
        return Response(
            message=message,
            data=coordination_plan
        )
    
    async def _list_agents(self, **kwargs) -> Response:
        """List all registered agents in the multi-agent system."""
        
        agents_info = []
        for agent_id, profile in self.multi_agent_system.agents.items():
            agent_info = {
                "agent_id": agent_id,
                "name": profile.name,
                "role": profile.role,
                "status": profile.status,
                "capabilities": profile.capabilities,
                "specializations": profile.specializations,
                "created_at": profile.created_at
            }
            agents_info.append(agent_info)
        
        message = f"Registered Agents ({len(agents_info)}):\n"
        for agent_info in agents_info:
            message += f"\n• {agent_info['name']} ({agent_info['agent_id']})\n"
            message += f"  Role: {agent_info['role']}\n"
            message += f"  Status: {agent_info['status']}\n"
            message += f"  Capabilities: {', '.join(agent_info['capabilities'])}\n"
        
        return Response(
            message=message,
            data={"agents": agents_info, "total_agents": len(agents_info)}
        )
    
    async def _create_collaboration_task(self, description: str = None,
                                       capabilities: List[str] = None, **kwargs) -> Response:
        """Create a new collaboration task."""
        
        if not description:
            return Response(
                message="Task description is required for creating collaboration task",
                break_loop=False
            )
        
        if not capabilities:
            capabilities = ["general_reasoning"]
        
        task = self.multi_agent_system.create_collaboration_task(
            description=description,
            required_capabilities=capabilities,
            priority=kwargs.get('priority', 1)
        )
        
        message = f"Collaboration Task Created:\n"
        message += f"• Task ID: {task.task_id}\n"
        message += f"• Description: {description}\n"
        message += f"• Required Capabilities: {', '.join(capabilities)}\n"
        message += f"• Assigned Agents: {len(task.assigned_agents)}\n"
        message += f"• Priority: {task.priority}\n"
        message += f"• Status: {task.status}"
        
        return Response(
            message=message,
            data={
                "task_id": task.task_id,
                "description": task.description,
                "assigned_agents": task.assigned_agents,
                "status": task.status
            }
        )
    
    async def _run_end_to_end_workflow(self, **kwargs) -> Response:
        """Run complete end-to-end multi-agent workflow."""
        
        workflow_result = await self.multi_agent_system.simulate_end_to_end_workflow()
        
        message = f"End-to-End Multi-Agent Workflow:\n"
        message += f"• Workflow ID: {workflow_result['workflow_id']}\n"
        message += f"• Steps Completed: {workflow_result['steps_completed']}/{workflow_result['total_steps']}\n"
        message += f"• Overall Success: {'✓' if workflow_result['overall_success'] else '✗'}\n"
        message += f"\nWorkflow Steps:\n"
        
        for i, step in enumerate(workflow_result['workflow_steps'], 1):
            step_name = step['step'].replace('_', ' ').title()
            status_icon = '✓' if step['status'] == 'completed' else '✗'
            message += f"  {i}. {status_icon} {step_name}\n"
        
        return Response(
            message=message,
            data=workflow_result
        )


def register():
    """Register the multi-agent collaboration tool with Agent-Zero."""
    return MultiAgentCollaborationTool