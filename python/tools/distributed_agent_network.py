"""
Distributed Agent Network Tool for Agent-Zero
Provides Agent-Zero integration with distributed cognitive agent networks and shared AtomSpace.
"""

import asyncio
import json
import time
from typing import Dict, List, Any, Optional
from pathlib import Path

# Import distributed components
try:
    from python.helpers.distributed_agent_network import DistributedAgentNetwork, create_distributed_agent_network
    from python.helpers.distributed_atomspace import NetworkAtomSpaceManager, create_distributed_atomspace_manager
    DISTRIBUTED_AVAILABLE = True
except ImportError as e:
    print(f"Distributed network components not available: {e}")
    DISTRIBUTED_AVAILABLE = False

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

# Import multi-agent components
try:
    from python.helpers.multi_agent import AgentProfile
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    print("Multi-agent components not available - using fallback")
    MULTI_AGENT_AVAILABLE = False
    # Create fallback AgentProfile
    from dataclasses import dataclass, field
    from typing import List, Dict, Any
    
    @dataclass
    class AgentProfile:
        agent_id: str
        name: str
        role: str
        capabilities: List[str] = field(default_factory=list)
        specializations: List[str] = field(default_factory=list)
        cognitive_config: Dict[str, Any] = field(default_factory=dict)
        status: str = "inactive"
        performance_metrics: Dict[str, float] = field(default_factory=dict)
        created_at: float = field(default_factory=time.time)


class DistributedAgentNetworkTool(Tool):
    """
    Agent-Zero tool for distributed cognitive agent networks with shared AtomSpace.
    Enables Agent-Zero to participate in and coordinate distributed agent networks.
    """
    
    def __init__(self, agent, **kwargs):
        # Handle Agent-Zero tool initialization with fallback
        try:
            super().__init__(agent, name="distributed_agent_network", **kwargs)
        except TypeError:
            # Fallback initialization for testing/development
            self.agent = agent
            self.name = kwargs.get("name", "distributed_agent_network")
        
        # Tool configuration
        self.config = self._load_config()
        
        # Distributed network components
        self.distributed_network: Optional[DistributedAgentNetwork] = None
        self.atomspace_manager: Optional[NetworkAtomSpaceManager] = None
        self.network_initialized = False
        
        # Agent registration
        self.local_agent_profile: Optional[AgentProfile] = None
        self.network_node_id = None
        
    def _load_config(self) -> Dict[str, Any]:
        """Load distributed network configuration."""
        config_path = Path("conf/config_distributed_network.json")
        
        default_config = {
            "enabled": True,
            "host": "localhost",
            "port": 17002,
            "atomspace_port": 18002,
            "bootstrap_nodes": [],  # List of [host, port] pairs
            "agent_discovery_interval": 10.0,
            "heartbeat_interval": 5.0,
            "sync_interval": 5.0,
            "max_agents_per_task": 5,
            "task_timeout": 300.0,
            "auto_join_network": True,
            "agent_capabilities": [
                "reasoning", "memory", "cognitive_processing", 
                "planning", "analysis", "pattern_recognition"
            ]
        }
        
        try:
            if config_path.exists():
                with open(config_path, 'r') as f:
                    user_config = json.load(f)
                    default_config.update(user_config)
        except Exception as e:
            print(f"Warning: Could not load distributed network config: {e}")
        
        return default_config
    
    async def execute(self, operation: str = "status", **kwargs) -> Response:
        """
        Execute distributed agent network operations.
        
        Available operations:
        - status: Get network status
        - start: Start/join distributed network
        - stop: Stop distributed network participation
        - discover: Discover available agents
        - create_task: Create distributed task
        - coordinate: Coordinate distributed task
        - reasoning: Execute distributed reasoning
        - sync: Synchronize AtomSpace across network
        """
        
        if not DISTRIBUTED_AVAILABLE:
            return Response(
                message="Distributed agent network functionality not available",
                break_loop=False
            )
        
        try:
            if operation == "status":
                return await self._get_network_status()
            elif operation == "start":
                return await self._start_network(**kwargs)
            elif operation == "stop":
                return await self._stop_network()
            elif operation == "discover":
                return await self._discover_agents()
            elif operation == "create_task":
                return await self._create_distributed_task(**kwargs)
            elif operation == "coordinate":
                return await self._coordinate_task(**kwargs)
            elif operation == "reasoning":
                return await self._distributed_reasoning(**kwargs)
            elif operation == "sync":
                return await self._sync_atomspace()
            else:
                return Response(
                    message=f"Unknown operation: {operation}. Available: status, start, stop, discover, create_task, coordinate, reasoning, sync",
                    break_loop=False
                )
                
        except Exception as e:
            return Response(
                message=f"Distributed network operation failed: {str(e)}",
                break_loop=False
            )
    
    async def _get_network_status(self) -> Response:
        """Get current distributed network status."""
        try:
            if not self.network_initialized:
                return Response(
                    message="Distributed network not initialized. Use 'start' operation to join network.",
                    data={
                        "initialized": False,
                        "config": self.config
                    },
                    break_loop=False
                )
            
            # Get comprehensive status
            network_status = self.distributed_network.get_network_status()
            atomspace_status = self.atomspace_manager.get_network_status() if self.atomspace_manager else {}
            
            status_data = {
                "initialized": True,
                "network": network_status,
                "atomspace": atomspace_status,
                "local_agent": {
                    "registered": self.local_agent_profile is not None,
                    "profile": self.local_agent_profile.__dict__ if self.local_agent_profile else None
                },
                "config": self.config
            }
            
            message = f"""Distributed Agent Network Status:
Network Node: {network_status['node_id']}
Address: {network_status['network_address']}
Status: {'Running' if network_status['running'] else 'Stopped'}
Connected Nodes: {network_status['connected_nodes']}
Total Agents: {network_status['total_agents']} (Local: {network_status['local_agents']}, Remote: {network_status['remote_agents']})
Active Tasks: {network_status['active_tasks']}
AtomSpace Network: {'Connected' if atomspace_status.get('connected_nodes', 0) > 0 else 'Isolated'}
AtomSpace Size: {atomspace_status.get('atomspace_size', 0)} atoms
"""
            
            return Response(
                message=message,
                data=status_data,
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Failed to get network status: {str(e)}",
                break_loop=False
            )
    
    async def _start_network(self, **kwargs) -> Response:
        """Start/join the distributed agent network."""
        try:
            if self.network_initialized:
                return Response(
                    message="Distributed network already initialized",
                    break_loop=False
                )
            
            # Extract parameters
            host = kwargs.get("host", self.config.get("host", "localhost"))
            port = kwargs.get("port", self.config.get("port", 17002))
            bootstrap_nodes = kwargs.get("bootstrap_nodes", self.config.get("bootstrap_nodes", []))
            
            # Create distributed network
            self.distributed_network = create_distributed_agent_network(
                host=host,
                port=port
            )
            
            self.network_node_id = self.distributed_network.node_id
            self.atomspace_manager = self.distributed_network.atomspace_manager
            
            # Start the network
            success = await self.distributed_network.start_distributed_network(
                bootstrap_nodes=[(n[0], n[1]) for n in bootstrap_nodes] if bootstrap_nodes else None
            )
            
            if not success:
                return Response(
                    message="Failed to start distributed network",
                    break_loop=False
                )
            
            # Register local agent
            if self.config.get("auto_join_network", True):
                await self._register_local_agent()
            
            self.network_initialized = True
            
            message = f"""Distributed Agent Network Started:
Node ID: {self.network_node_id}
Network Address: {host}:{port}
AtomSpace Address: {host}:{port + 1000}
Bootstrap Nodes: {len(bootstrap_nodes)}
Local Agent Registered: {self.local_agent_profile is not None}
"""
            
            return Response(
                message=message,
                data={
                    "node_id": self.network_node_id,
                    "network_address": f"{host}:{port}",
                    "atomspace_address": f"{host}:{port + 1000}",
                    "bootstrap_nodes": bootstrap_nodes,
                    "local_agent_registered": self.local_agent_profile is not None
                },
                break_loop=False
            )
            
        except Exception as e:
            self.network_initialized = False
            return Response(
                message=f"Failed to start distributed network: {str(e)}",
                break_loop=False
            )
    
    async def _stop_network(self) -> Response:
        """Stop distributed network participation."""
        try:
            if not self.network_initialized:
                return Response(
                    message="Distributed network not initialized",
                    break_loop=False
                )
            
            # Stop distributed network
            await self.distributed_network.stop_distributed_network()
            
            # Reset state
            self.distributed_network = None
            self.atomspace_manager = None
            self.local_agent_profile = None
            self.network_node_id = None
            self.network_initialized = False
            
            return Response(
                message="Distributed agent network stopped successfully",
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Failed to stop distributed network: {str(e)}",
                break_loop=False
            )
    
    async def _discover_agents(self) -> Response:
        """Discover available agents across the network."""
        try:
            if not self.network_initialized:
                return Response(
                    message="Network not initialized. Start network first.",
                    break_loop=False
                )
            
            # Discover network agents
            agents = await self.distributed_network.discover_network_agents()
            
            # Format agent information
            agent_info = []
            for agent in agents:
                agent_info.append({
                    "agent_id": agent.agent_id,
                    "node_id": agent.node_id,
                    "name": agent.profile.name,
                    "role": agent.profile.role,
                    "capabilities": agent.profile.capabilities,
                    "specializations": agent.profile.specializations,
                    "network_address": agent.network_address,
                    "is_local": agent.is_local,
                    "last_heartbeat": agent.last_heartbeat
                })
            
            local_count = sum(1 for a in agents if a.is_local)
            remote_count = len(agents) - local_count
            
            message = f"""Agent Discovery Results:
Total Agents: {len(agents)}
Local Agents: {local_count}
Remote Agents: {remote_count}
Network Nodes: {len(set(a.node_id for a in agents))}
"""
            
            return Response(
                message=message,
                data={
                    "total_agents": len(agents),
                    "local_agents": local_count,
                    "remote_agents": remote_count,
                    "agents": agent_info
                },
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Agent discovery failed: {str(e)}",
                break_loop=False
            )
    
    async def _create_distributed_task(self, **kwargs) -> Response:
        """Create a distributed task for network execution."""
        try:
            if not self.network_initialized:
                return Response(
                    message="Network not initialized. Start network first.",
                    break_loop=False
                )
            
            # Extract task parameters
            description = kwargs.get("description", "Distributed cognitive task")
            required_capabilities = kwargs.get("required_capabilities", ["reasoning", "cognitive_processing"])
            priority = kwargs.get("priority", 1)
            
            # Create distributed task
            task = await self.distributed_network.create_distributed_task(
                description=description,
                required_capabilities=required_capabilities,
                priority=priority
            )
            
            if not task:
                return Response(
                    message="Failed to create distributed task - no capable agents found",
                    break_loop=False
                )
            
            message = f"""Distributed Task Created:
Task ID: {task.task_id}
Description: {task.description}
Required Capabilities: {task.required_capabilities}
Assigned Agents: {len(task.assigned_agents)}
Participating Nodes: {len(task.participating_nodes)}
Priority: {task.priority}
Status: {task.status}
"""
            
            return Response(
                message=message,
                data={
                    "task_id": task.task_id,
                    "description": task.description,
                    "required_capabilities": task.required_capabilities,
                    "assigned_agents": task.assigned_agents,
                    "participating_nodes": task.participating_nodes,
                    "priority": task.priority,
                    "status": task.status,
                    "created_at": task.created_at
                },
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Failed to create distributed task: {str(e)}",
                break_loop=False
            )
    
    async def _coordinate_task(self, **kwargs) -> Response:
        """Coordinate execution of a distributed task."""
        try:
            task_id = kwargs.get("task_id")
            if not task_id:
                return Response(
                    message="Task ID required for coordination",
                    break_loop=False
                )
            
            if not self.network_initialized:
                return Response(
                    message="Network not initialized. Start network first.",
                    break_loop=False
                )
            
            # Coordinate the task
            result = await self.distributed_network.coordinate_distributed_task(task_id)
            
            if "error" in result:
                return Response(
                    message=f"Task coordination failed: {result['error']}",
                    break_loop=False
                )
            
            message = f"""Task Coordination Completed:
Task ID: {result['task_id']}
Status: {result['status']}
Participating Nodes: {result['participating_nodes']}
Assigned Agents: {result['assigned_agents']}
Coordination Success: {result['status'] == 'completed'}
"""
            
            return Response(
                message=message,
                data=result,
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Task coordination failed: {str(e)}",
                break_loop=False
            )
    
    async def _distributed_reasoning(self, **kwargs) -> Response:
        """Execute distributed reasoning across the network."""
        try:
            query = kwargs.get("query", "What is the optimal solution to this problem?")
            participating_agents = kwargs.get("participating_agents")
            
            if not self.network_initialized:
                return Response(
                    message="Network not initialized. Start network first.",
                    break_loop=False
                )
            
            # Execute distributed reasoning
            result = await self.distributed_network.execute_distributed_reasoning(
                query=query,
                participating_agents=participating_agents
            )
            
            if "error" in result:
                return Response(
                    message=f"Distributed reasoning failed: {result['error']}",
                    break_loop=False
                )
            
            message = f"""Distributed Reasoning Completed:
Query: {result['query']}
Task ID: {result['task_id']}
Participating Agents: {result['participating_agents']}
Participating Nodes: {result['participating_nodes']}
AtomSpace Synchronized: {result['atomspace_synchronized']}
Status: {result['reasoning_result']['status']}
"""
            
            return Response(
                message=message,
                data=result,
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"Distributed reasoning failed: {str(e)}",
                break_loop=False
            )
    
    async def _sync_atomspace(self) -> Response:
        """Synchronize AtomSpace across the network."""
        try:
            if not self.network_initialized or not self.atomspace_manager:
                return Response(
                    message="AtomSpace network not initialized",
                    break_loop=False
                )
            
            # Perform synchronization
            success = await self.atomspace_manager.synchronize_atomspace()
            
            # Get post-sync status
            status = self.atomspace_manager.get_network_status()
            
            message = f"""AtomSpace Synchronization {'Completed' if success else 'Failed'}:
Network Nodes: {status['connected_nodes']}
AtomSpace Size: {status['atomspace_size']} atoms
Last Sync: {time.ctime(status['last_sync']) if status['last_sync'] > 0 else 'Never'}
Pending Operations: {status['pending_operations']}
Completed Operations: {status['completed_operations']}
"""
            
            return Response(
                message=message,
                data={
                    "sync_success": success,
                    "atomspace_status": status
                },
                break_loop=False
            )
            
        except Exception as e:
            return Response(
                message=f"AtomSpace synchronization failed: {str(e)}",
                break_loop=False
            )
    
    async def _register_local_agent(self):
        """Register the current Agent-Zero instance as a local agent."""
        try:
            # Create agent profile
            agent_id = f"agent_zero_{self.network_node_id}"
            
            self.local_agent_profile = AgentProfile(
                agent_id=agent_id,
                name=f"Agent-Zero-{self.network_node_id[:8]}",
                role="cognitive_agent",
                capabilities=self.config.get("agent_capabilities", []),
                specializations=["agent_zero_integration", "tool_execution"],
                cognitive_config={
                    "distributed_reasoning": True,
                    "atomspace_sharing": True,
                    "network_coordination": True
                }
            )
            
            # Register with distributed network
            success = await self.distributed_network.register_local_agent(self.local_agent_profile)
            
            if success:
                print(f"Registered local agent: {agent_id}")
            else:
                print(f"Failed to register local agent: {agent_id}")
                
        except Exception as e:
            print(f"Error registering local agent: {e}")


# Global instance for easier access
_global_distributed_tool = None

def get_distributed_agent_network_tool(agent=None):
    """Get or create a global distributed agent network tool instance."""
    global _global_distributed_tool
    
    if _global_distributed_tool is None and agent is not None:
        _global_distributed_tool = DistributedAgentNetworkTool(agent)
    
    return _global_distributed_tool