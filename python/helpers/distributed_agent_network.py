"""
Distributed Agent Network Manager for PyCog-Zero
Manages distributed cognitive agent networks with shared AtomSpace across network boundaries.
"""

import asyncio
import json
import time
import uuid
import socket
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Import the distributed AtomSpace manager
from python.helpers.distributed_atomspace import NetworkAtomSpaceManager, AtomSpaceNode

# Import existing multi-agent components
try:
    from python.helpers.multi_agent import MultiAgentSystem, CognitiveMultiAgentSystem, AgentProfile, CollaborationTask
    MULTI_AGENT_AVAILABLE = True
except ImportError:
    print("Local multi-agent system not available")
    MULTI_AGENT_AVAILABLE = False

# Try to import Agent-Zero components
try:
    from agent import Agent, AgentConfig
    AGENT_ZERO_AVAILABLE = True
except ImportError:
    print("Agent-Zero components not available - using simulation mode")
    AGENT_ZERO_AVAILABLE = False


@dataclass
class NetworkAgent:
    """Represents an agent in the distributed network."""
    agent_id: str
    node_id: str  # Which network node hosts this agent
    profile: AgentProfile
    network_address: str  # host:port where agent can be reached
    last_heartbeat: float = field(default_factory=time.time)
    is_local: bool = False  # True if this agent runs on current node


@dataclass
class DistributedTask:
    """A task that can be distributed across multiple network nodes."""
    task_id: str
    description: str
    required_capabilities: List[str]
    assigned_agents: List[str] = field(default_factory=list)  # agent IDs
    participating_nodes: List[str] = field(default_factory=list)  # node IDs
    status: str = "pending"  # pending, distributed, active, completed, failed
    priority: int = 1
    created_at: float = field(default_factory=time.time)
    coordination_strategy: str = "distributed_consensus"
    results: Dict[str, Any] = field(default_factory=dict)


class DistributedAgentNetwork:
    """
    Manages a network of cognitive agents distributed across multiple nodes.
    Provides discovery, coordination, and task distribution capabilities.
    """
    
    def __init__(self, node_id: str = None, host: str = "localhost", port: int = 17002):
        self.node_id = node_id or str(uuid.uuid4())
        self.host = host
        self.port = port
        
        # Distributed AtomSpace manager
        atomspace_port = port + 1000  # Use different port for AtomSpace
        self.atomspace_manager = NetworkAtomSpaceManager(
            node_id=self.node_id, 
            host=host, 
            port=atomspace_port
        )
        
        # Local multi-agent system
        if MULTI_AGENT_AVAILABLE:
            self.local_agents = CognitiveMultiAgentSystem()
        else:
            self.local_agents = None
        
        # Network topology
        self.network_nodes: Dict[str, AtomSpaceNode] = {}  # Other network nodes
        self.distributed_agents: Dict[str, NetworkAgent] = {}  # All agents in network
        self.distributed_tasks: Dict[str, DistributedTask] = {}
        
        # Network communication
        self.server_socket: Optional[socket.socket] = None
        self.client_connections: Dict[str, socket.socket] = {}
        self.running = False
        
        # Settings
        self.agent_discovery_interval = 10.0  # seconds
        self.heartbeat_interval = 5.0
        self.task_timeout = 300.0  # 5 minutes
        
        self.logger = logging.getLogger(f"DistributedAgentNetwork.{self.node_id}")
    
    async def start_distributed_network(self, bootstrap_nodes: List[Tuple[str, int]] = None):
        """Start the distributed agent network."""
        self.logger.info(f"Starting distributed agent network on {self.host}:{self.port}")
        
        try:
            # Start AtomSpace networking
            atomspace_success = await self.atomspace_manager.start_network_service()
            if not atomspace_success:
                self.logger.warning("AtomSpace network service failed to start")
            
            # Start agent network service
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)
            
            self.running = True
            
            # Start background services
            asyncio.create_task(self._network_listener())
            asyncio.create_task(self._agent_discovery_service())
            asyncio.create_task(self._heartbeat_service())
            
            # Connect to bootstrap nodes if provided
            if bootstrap_nodes:
                for host, port in bootstrap_nodes:
                    await self.connect_to_network_node(host, port)
            
            self.logger.info("Distributed agent network started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start distributed network: {e}")
            return False
    
    async def stop_distributed_network(self):
        """Stop the distributed agent network."""
        self.logger.info("Stopping distributed agent network")
        self.running = False
        
        # Stop AtomSpace networking
        await self.atomspace_manager.stop_network_service()
        
        # Close network connections
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        for conn in self.client_connections.values():
            try:
                conn.close()
            except:
                pass
    
    async def connect_to_network_node(self, host: str, port: int) -> bool:
        """Connect to another distributed agent network node."""
        node_address = f"{host}:{port}"
        
        try:
            # Connect for agent networking
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))
            self.client_connections[node_address] = client_socket
            
            # Also connect AtomSpace networking (port + 1000)
            atomspace_success = await self.atomspace_manager.connect_to_network_node(host, port + 1000)
            if not atomspace_success:
                self.logger.warning(f"Failed to connect AtomSpace to {host}:{port + 1000}")
            
            # Register network node
            self.network_nodes[node_address] = AtomSpaceNode(
                node_id=node_address,
                host=host,
                port=port,
                status="connected",
                capabilities=["agent_hosting", "task_coordination", "distributed_reasoning"]
            )
            
            # Send initial handshake
            await self._send_agent_handshake(node_address)
            
            self.logger.info(f"Connected to network node: {node_address}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to network node {node_address}: {e}")
            return False
    
    async def register_local_agent(self, agent_profile: AgentProfile) -> bool:
        """Register a local agent in the distributed network."""
        try:
            # Create network agent entry
            network_agent = NetworkAgent(
                agent_id=agent_profile.agent_id,
                node_id=self.node_id,
                profile=agent_profile,
                network_address=f"{self.host}:{self.port}",
                is_local=True
            )
            
            self.distributed_agents[agent_profile.agent_id] = network_agent
            
            # Register in local multi-agent system if available
            if self.local_agents:
                self.local_agents.agents[agent_profile.agent_id] = agent_profile
            
            # Announce agent to network
            await self._announce_agent_to_network(network_agent)
            
            self.logger.info(f"Registered local agent: {agent_profile.agent_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to register local agent: {e}")
            return False
    
    async def discover_network_agents(self) -> List[NetworkAgent]:
        """Discover agents available across the network."""
        try:
            # Send discovery requests to all connected nodes
            discovery_results = []
            
            for node_address in self.network_nodes.keys():
                agents = await self._request_agent_list(node_address)
                if agents:
                    discovery_results.extend(agents)
            
            # Update distributed agents registry
            for agent_data in discovery_results:
                if agent_data["agent_id"] not in self.distributed_agents:
                    network_agent = NetworkAgent(
                        agent_id=agent_data["agent_id"],
                        node_id=agent_data["node_id"],
                        profile=AgentProfile(**agent_data["profile"]),
                        network_address=agent_data["network_address"],
                        is_local=False
                    )
                    self.distributed_agents[agent_data["agent_id"]] = network_agent
            
            return list(self.distributed_agents.values())
            
        except Exception as e:
            self.logger.error(f"Agent discovery failed: {e}")
            return []
    
    async def create_distributed_task(self, description: str, 
                                    required_capabilities: List[str],
                                    priority: int = 1) -> Optional[DistributedTask]:
        """Create a task for distributed processing across the network."""
        try:
            # Create distributed task
            task = DistributedTask(
                task_id=str(uuid.uuid4()),
                description=description,
                required_capabilities=required_capabilities,
                priority=priority
            )
            
            # Find capable agents across the network
            capable_agents = []
            participating_nodes = set()
            
            for agent in self.distributed_agents.values():
                agent_capabilities = set(agent.profile.capabilities)
                required_caps = set(required_capabilities)
                
                if required_caps.issubset(agent_capabilities):
                    capable_agents.append(agent.agent_id)
                    participating_nodes.add(agent.node_id)
            
            if not capable_agents:
                self.logger.warning(f"No capable agents found for task: {description}")
                return None
            
            # Assign agents and nodes
            task.assigned_agents = capable_agents[:5]  # Limit to 5 agents
            task.participating_nodes = list(participating_nodes)
            task.status = "distributed"
            
            self.distributed_tasks[task.task_id] = task
            
            self.logger.info(f"Created distributed task: {task.task_id}")
            return task
            
        except Exception as e:
            self.logger.error(f"Failed to create distributed task: {e}")
            return None
    
    async def coordinate_distributed_task(self, task_id: str) -> Dict[str, Any]:
        """Coordinate execution of a distributed task across network nodes."""
        if task_id not in self.distributed_tasks:
            return {"error": "Task not found"}
        
        task = self.distributed_tasks[task_id]
        
        try:
            task.status = "active"
            coordination_results = []
            
            # Group agents by their hosting nodes
            agents_by_node = {}
            for agent_id in task.assigned_agents:
                if agent_id in self.distributed_agents:
                    agent = self.distributed_agents[agent_id]
                    node_id = agent.node_id
                    if node_id not in agents_by_node:
                        agents_by_node[node_id] = []
                    agents_by_node[node_id].append(agent_id)
            
            # Coordinate with each participating node
            for node_id, agent_ids in agents_by_node.items():
                if node_id == self.node_id:
                    # Local coordination
                    result = await self._coordinate_local_agents(task, agent_ids)
                else:
                    # Remote coordination
                    result = await self._coordinate_remote_agents(task, node_id, agent_ids)
                
                coordination_results.append({
                    "node_id": node_id,
                    "agent_ids": agent_ids,
                    "result": result
                })
            
            # Update task results
            task.results["coordination"] = coordination_results
            task.results["timestamp"] = time.time()
            
            # Check if all coordination succeeded
            all_success = all(r["result"].get("success", False) for r in coordination_results)
            task.status = "completed" if all_success else "failed"
            
            return {
                "task_id": task_id,
                "status": task.status,
                "participating_nodes": len(agents_by_node),
                "assigned_agents": len(task.assigned_agents),
                "coordination_results": coordination_results
            }
            
        except Exception as e:
            self.logger.error(f"Task coordination failed for {task_id}: {e}")
            task.status = "failed"
            return {"error": str(e)}
    
    async def execute_distributed_reasoning(self, query: str, participating_agents: List[str] = None) -> Dict[str, Any]:
        """Execute distributed reasoning across network agents with shared AtomSpace."""
        try:
            # Use all agents if none specified
            if not participating_agents:
                participating_agents = list(self.distributed_agents.keys())[:5]  # Limit to 5
            
            # Ensure AtomSpace is synchronized before reasoning
            await self.atomspace_manager.synchronize_atomspace()
            
            # Create distributed reasoning task
            task = await self.create_distributed_task(
                description=f"Distributed reasoning: {query}",
                required_capabilities=["reasoning", "cognitive_processing"],
                priority=3
            )
            
            if not task:
                return {"error": "Failed to create reasoning task"}
            
            # Execute coordinated reasoning
            coordination_result = await self.coordinate_distributed_task(task.task_id)
            
            # Synchronize results back to AtomSpace
            await self.atomspace_manager.synchronize_atomspace()
            
            return {
                "query": query,
                "task_id": task.task_id,
                "participating_agents": len(task.assigned_agents),
                "participating_nodes": len(task.participating_nodes),
                "reasoning_result": coordination_result,
                "atomspace_synchronized": True,
                "timestamp": time.time()
            }
            
        except Exception as e:
            self.logger.error(f"Distributed reasoning failed: {e}")
            return {"error": str(e)}
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get comprehensive network status."""
        atomspace_status = self.atomspace_manager.get_network_status()
        
        local_agent_count = sum(1 for a in self.distributed_agents.values() if a.is_local)
        remote_agent_count = sum(1 for a in self.distributed_agents.values() if not a.is_local)
        
        active_tasks = sum(1 for t in self.distributed_tasks.values() if t.status in ["active", "distributed"])
        
        return {
            "node_id": self.node_id,
            "network_address": f"{self.host}:{self.port}",
            "running": self.running,
            "connected_nodes": len(self.network_nodes),
            "local_agents": local_agent_count,
            "remote_agents": remote_agent_count,
            "total_agents": len(self.distributed_agents),
            "active_tasks": active_tasks,
            "total_tasks": len(self.distributed_tasks),
            "atomspace_status": atomspace_status,
            "capabilities": ["agent_hosting", "task_coordination", "distributed_reasoning", "atomspace_sharing"]
        }
    
    async def _network_listener(self):
        """Background service to listen for network connections."""
        while self.running:
            try:
                if self.server_socket:
                    try:
                        client_socket, address = self.server_socket.accept()
                        asyncio.create_task(self._handle_agent_client(client_socket, address))
                    except socket.timeout:
                        continue
                    except Exception as e:
                        if self.running:
                            self.logger.warning(f"Network listener error: {e}")
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                if self.running:
                    self.logger.error(f"Agent network listener failed: {e}")
                    await asyncio.sleep(1.0)
    
    async def _agent_discovery_service(self):
        """Background service for agent discovery."""
        while self.running:
            try:
                await asyncio.sleep(self.agent_discovery_interval)
                await self.discover_network_agents()
                
            except Exception as e:
                if self.running:
                    self.logger.error(f"Agent discovery service error: {e}")
    
    async def _heartbeat_service(self):
        """Background service for agent heartbeats."""
        while self.running:
            try:
                await asyncio.sleep(self.heartbeat_interval)
                await self._send_heartbeats()
                await self._check_agent_health()
                
            except Exception as e:
                if self.running:
                    self.logger.error(f"Heartbeat service error: {e}")
    
    async def _handle_agent_client(self, client_socket: socket.socket, address: Tuple[str, int]):
        """Handle incoming agent network client connection."""
        client_id = f"{address[0]}:{address[1]}"
        
        try:
            self.logger.info(f"New agent client connection: {client_id}")
            
            while self.running:
                try:
                    # Implement agent message handling
                    await asyncio.sleep(0.1)  # Placeholder
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.warning(f"Error handling agent client {client_id}: {e}")
                    break
                    
        except Exception as e:
            self.logger.error(f"Agent client connection error {client_id}: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    async def _send_agent_handshake(self, target_node: str):
        """Send agent network handshake to target node."""
        handshake_msg = {
            "type": "agent_handshake",
            "node_id": self.node_id,
            "timestamp": time.time(),
            "capabilities": ["agent_hosting", "task_coordination", "distributed_reasoning"],
            "local_agent_count": sum(1 for a in self.distributed_agents.values() if a.is_local)
        }
        
        # Placeholder for actual message sending
        self.logger.debug(f"Would send handshake to {target_node}: {handshake_msg}")
        return True
    
    async def _announce_agent_to_network(self, network_agent: NetworkAgent):
        """Announce a local agent to the network."""
        announcement = {
            "type": "agent_announcement",
            "agent_id": network_agent.agent_id,
            "node_id": network_agent.node_id,
            "profile": {
                "name": network_agent.profile.name,
                "role": network_agent.profile.role,
                "capabilities": network_agent.profile.capabilities,
                "specializations": network_agent.profile.specializations
            },
            "timestamp": time.time()
        }
        
        # Broadcast to all connected nodes
        for node_address in self.network_nodes.keys():
            self.logger.debug(f"Would announce agent to {node_address}: {announcement}")
        
        return True
    
    async def _request_agent_list(self, node_address: str) -> List[Dict[str, Any]]:
        """Request agent list from a network node."""
        # Placeholder for actual network request
        return []
    
    async def _coordinate_local_agents(self, task: DistributedTask, agent_ids: List[str]) -> Dict[str, Any]:
        """Coordinate task execution with local agents."""
        try:
            if self.local_agents:
                # Use local multi-agent system
                local_task = CollaborationTask(
                    task_id=task.task_id,
                    description=task.description,
                    required_capabilities=task.required_capabilities,
                    assigned_agents=agent_ids,
                    priority=task.priority
                )
                
                # Execute with local system
                result = await self._simulate_local_coordination(local_task)
                return {"success": True, "result": result}
            else:
                # Simulate coordination
                return {
                    "success": True, 
                    "result": f"Simulated local coordination for {len(agent_ids)} agents",
                    "agent_ids": agent_ids
                }
                
        except Exception as e:
            self.logger.error(f"Local coordination failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _coordinate_remote_agents(self, task: DistributedTask, node_id: str, agent_ids: List[str]) -> Dict[str, Any]:
        """Coordinate task execution with remote agents."""
        try:
            coordination_request = {
                "type": "coordination_request",
                "task_id": task.task_id,
                "description": task.description,
                "agent_ids": agent_ids,
                "timestamp": time.time()
            }
            
            # Placeholder for actual remote coordination
            self.logger.debug(f"Would coordinate with remote node {node_id}: {coordination_request}")
            
            return {
                "success": True,
                "result": f"Simulated remote coordination with node {node_id}",
                "agent_ids": agent_ids
            }
            
        except Exception as e:
            self.logger.error(f"Remote coordination failed: {e}")
            return {"success": False, "error": str(e)}
    
    async def _simulate_local_coordination(self, task: CollaborationTask) -> Dict[str, Any]:
        """Simulate local task coordination."""
        return {
            "task_id": task.task_id,
            "assigned_agents": len(task.assigned_agents),
            "status": "completed",
            "execution_time": 0.1,
            "timestamp": time.time()
        }
    
    async def _send_heartbeats(self):
        """Send heartbeats to connected nodes."""
        heartbeat = {
            "type": "heartbeat",
            "node_id": self.node_id,
            "timestamp": time.time(),
            "agent_count": sum(1 for a in self.distributed_agents.values() if a.is_local),
            "task_count": len(self.distributed_tasks)
        }
        
        # Send to all connected nodes
        for node_address in self.network_nodes.keys():
            self.logger.debug(f"Would send heartbeat to {node_address}")
    
    async def _check_agent_health(self):
        """Check health of network agents."""
        current_time = time.time()
        stale_threshold = self.heartbeat_interval * 3  # 3 missed heartbeats
        
        stale_agents = []
        for agent in self.distributed_agents.values():
            if not agent.is_local and (current_time - agent.last_heartbeat) > stale_threshold:
                stale_agents.append(agent.agent_id)
        
        # Remove stale agents
        for agent_id in stale_agents:
            if agent_id in self.distributed_agents:
                del self.distributed_agents[agent_id]
                self.logger.warning(f"Removed stale agent: {agent_id}")


def create_distributed_agent_network(node_id: str = None, 
                                    host: str = "localhost", 
                                    port: int = 17002,
                                    bootstrap_nodes: List[Tuple[str, int]] = None) -> DistributedAgentNetwork:
    """Create a distributed agent network instance."""
    return DistributedAgentNetwork(node_id, host, port)