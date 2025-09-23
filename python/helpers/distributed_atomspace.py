"""
Distributed AtomSpace Manager for PyCog-Zero
Enables shared AtomSpace memory across distributed cognitive agent networks.
"""

import asyncio
import json
import time
import uuid
import socket
import pickle
import threading
from typing import Dict, List, Any, Optional, Set, Tuple, Callable
from dataclasses import dataclass, field
from pathlib import Path
import logging

# Try to import OpenCog components (graceful fallback if not installed)
try:
    from opencog.atomspace import AtomSpace, types, Atom, Handle
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("OpenCog not available - using fallback AtomSpace simulation")
    OPENCOG_AVAILABLE = False


@dataclass
class AtomSpaceNode:
    """Represents a node in the distributed AtomSpace network."""
    node_id: str
    host: str
    port: int
    status: str = "disconnected"  # connected, disconnected, synchronizing
    last_sync: float = field(default_factory=time.time)
    atomspace_version: int = 0
    capabilities: List[str] = field(default_factory=list)
    agent_count: int = 0


@dataclass
class SyncOperation:
    """Represents a synchronization operation between AtomSpaces."""
    operation_id: str
    operation_type: str  # add_atom, remove_atom, update_values, full_sync
    source_node: str
    target_nodes: List[str] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)
    data: Dict[str, Any] = field(default_factory=dict)
    status: str = "pending"  # pending, in_progress, completed, failed


class NetworkAtomSpaceManager:
    """
    Manages distributed AtomSpace synchronization across network nodes.
    Provides consistency and replication for shared cognitive memory.
    """
    
    def __init__(self, node_id: str = None, host: str = "localhost", port: int = 17001):
        self.node_id = node_id or str(uuid.uuid4())
        self.host = host
        self.port = port
        
        # Local AtomSpace
        if OPENCOG_AVAILABLE:
            self.local_atomspace = AtomSpace()
            initialize_opencog(self.local_atomspace)
        else:
            self.local_atomspace = {}  # Fallback dictionary
        
        # Network topology
        self.network_nodes: Dict[str, AtomSpaceNode] = {}
        self.pending_operations: Dict[str, SyncOperation] = {}
        self.completed_operations: List[SyncOperation] = []
        
        # Networking
        self.server_socket: Optional[socket.socket] = None
        self.client_connections: Dict[str, socket.socket] = {}
        self.running = False
        
        # Synchronization settings
        self.sync_interval = 5.0  # seconds
        self.conflict_resolution = "latest_wins"  # latest_wins, merge, custom
        self.max_sync_retries = 3
        
        self.logger = logging.getLogger(f"NetworkAtomSpace.{self.node_id}")
        
    async def start_network_service(self):
        """Start the network service for AtomSpace synchronization."""
        self.logger.info(f"Starting AtomSpace network service on {self.host}:{self.port}")
        
        try:
            self.server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            self.server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            self.server_socket.bind((self.host, self.port))
            self.server_socket.listen(5)
            self.server_socket.settimeout(1.0)  # Non-blocking accept
            
            self.running = True
            
            # Start background tasks
            asyncio.create_task(self._network_listener())
            asyncio.create_task(self._sync_scheduler())
            
            self.logger.info(f"AtomSpace network service started successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to start network service: {e}")
            return False
    
    async def stop_network_service(self):
        """Stop the network service."""
        self.logger.info("Stopping AtomSpace network service")
        self.running = False
        
        # Close server socket
        if self.server_socket:
            try:
                self.server_socket.close()
            except:
                pass
        
        # Close client connections
        for conn in self.client_connections.values():
            try:
                conn.close()
            except:
                pass
        
        self.client_connections.clear()
        
    async def connect_to_network_node(self, host: str, port: int) -> bool:
        """Connect to another AtomSpace network node."""
        node_id = f"{host}:{port}"
        
        try:
            # Create connection
            client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            client_socket.connect((host, port))
            
            # Register connection
            self.client_connections[node_id] = client_socket
            
            # Register network node
            self.network_nodes[node_id] = AtomSpaceNode(
                node_id=node_id,
                host=host,
                port=port,
                status="connected",
                capabilities=["atomspace_sync", "distributed_reasoning"]
            )
            
            # Send initial handshake
            await self._send_handshake(node_id)
            
            self.logger.info(f"Connected to network node: {node_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to connect to {node_id}: {e}")
            return False
    
    async def synchronize_atomspace(self, target_nodes: List[str] = None) -> bool:
        """Synchronize local AtomSpace with network nodes."""
        if not OPENCOG_AVAILABLE:
            return await self._sync_fallback_memory(target_nodes)
        
        target_nodes = target_nodes or list(self.network_nodes.keys())
        
        try:
            # Create sync operation
            sync_op = SyncOperation(
                operation_id=str(uuid.uuid4()),
                operation_type="full_sync",
                source_node=self.node_id,
                target_nodes=target_nodes,
                data={
                    "atom_count": len(self.local_atomspace),
                    "sync_timestamp": time.time()
                }
            )
            
            self.pending_operations[sync_op.operation_id] = sync_op
            
            # Perform synchronization with each target
            sync_results = []
            for target_node in target_nodes:
                if target_node in self.network_nodes:
                    result = await self._sync_with_node(target_node, sync_op)
                    sync_results.append(result)
            
            # Update operation status
            sync_op.status = "completed" if all(sync_results) else "failed"
            self.completed_operations.append(sync_op)
            
            self.logger.info(f"AtomSpace synchronization completed: {sync_op.operation_id}")
            return all(sync_results)
            
        except Exception as e:
            self.logger.error(f"AtomSpace synchronization failed: {e}")
            return False
    
    async def replicate_atom_operation(self, operation: str, atom_data: Dict[str, Any], 
                                     target_nodes: List[str] = None) -> bool:
        """Replicate an atom operation to network nodes."""
        target_nodes = target_nodes or list(self.network_nodes.keys())
        
        # Create replication operation
        repl_op = SyncOperation(
            operation_id=str(uuid.uuid4()),
            operation_type=operation,
            source_node=self.node_id,
            target_nodes=target_nodes,
            data=atom_data
        )
        
        self.pending_operations[repl_op.operation_id] = repl_op
        
        try:
            # Send operation to target nodes
            replication_results = []
            for target_node in target_nodes:
                if target_node in self.network_nodes:
                    result = await self._send_atom_operation(target_node, repl_op)
                    replication_results.append(result)
            
            repl_op.status = "completed" if all(replication_results) else "failed"
            self.completed_operations.append(repl_op)
            
            return all(replication_results)
            
        except Exception as e:
            self.logger.error(f"Atom replication failed: {e}")
            repl_op.status = "failed"
            return False
    
    def get_network_status(self) -> Dict[str, Any]:
        """Get current network and synchronization status."""
        connected_nodes = [n for n in self.network_nodes.values() if n.status == "connected"]
        
        return {
            "node_id": self.node_id,
            "host": self.host,
            "port": self.port,
            "running": self.running,
            "connected_nodes": len(connected_nodes),
            "total_nodes": len(self.network_nodes),
            "pending_operations": len(self.pending_operations),
            "completed_operations": len(self.completed_operations),
            "atomspace_size": len(self.local_atomspace) if OPENCOG_AVAILABLE else len(self.local_atomspace),
            "last_sync": max([n.last_sync for n in self.network_nodes.values()] + [0]),
            "opencog_available": OPENCOG_AVAILABLE
        }
    
    async def _network_listener(self):
        """Background task to listen for network connections."""
        while self.running:
            try:
                if self.server_socket:
                    try:
                        client_socket, address = self.server_socket.accept()
                        asyncio.create_task(self._handle_client_connection(client_socket, address))
                    except socket.timeout:
                        continue  # Normal timeout, continue listening
                    except Exception as e:
                        if self.running:  # Only log if we're still supposed to be running
                            self.logger.warning(f"Network listener error: {e}")
                
                await asyncio.sleep(0.1)
                
            except Exception as e:
                if self.running:
                    self.logger.error(f"Network listener failed: {e}")
                    await asyncio.sleep(1.0)
    
    async def _sync_scheduler(self):
        """Background task to schedule periodic synchronization."""
        while self.running:
            try:
                await asyncio.sleep(self.sync_interval)
                
                if self.network_nodes:
                    await self.synchronize_atomspace()
                
            except Exception as e:
                if self.running:
                    self.logger.error(f"Sync scheduler error: {e}")
    
    async def _handle_client_connection(self, client_socket: socket.socket, address: Tuple[str, int]):
        """Handle incoming client connection."""
        client_id = f"{address[0]}:{address[1]}"
        
        try:
            self.logger.info(f"New client connection: {client_id}")
            
            # Handle client messages
            while self.running:
                try:
                    data = client_socket.recv(4096)
                    if not data:
                        break
                    
                    message = pickle.loads(data)
                    await self._process_network_message(client_id, message)
                    
                except socket.timeout:
                    continue
                except Exception as e:
                    self.logger.warning(f"Error handling client {client_id}: {e}")
                    break
            
        except Exception as e:
            self.logger.error(f"Client connection error {client_id}: {e}")
        finally:
            try:
                client_socket.close()
            except:
                pass
    
    async def _process_network_message(self, sender: str, message: Dict[str, Any]):
        """Process incoming network message."""
        try:
            msg_type = message.get("type", "unknown")
            
            if msg_type == "handshake":
                await self._handle_handshake(sender, message)
            elif msg_type == "sync_request":
                await self._handle_sync_request(sender, message)
            elif msg_type == "atom_operation":
                await self._handle_atom_operation(sender, message)
            elif msg_type == "status_request":
                await self._handle_status_request(sender, message)
            else:
                self.logger.warning(f"Unknown message type from {sender}: {msg_type}")
                
        except Exception as e:
            self.logger.error(f"Error processing message from {sender}: {e}")
    
    async def _send_handshake(self, target_node: str):
        """Send handshake to target node."""
        handshake_msg = {
            "type": "handshake",
            "node_id": self.node_id,
            "timestamp": time.time(),
            "capabilities": ["atomspace_sync", "distributed_reasoning"],
            "atomspace_version": 0
        }
        
        return await self._send_message(target_node, handshake_msg)
    
    async def _handle_handshake(self, sender: str, message: Dict[str, Any]):
        """Handle incoming handshake."""
        # Register or update network node
        if sender not in self.network_nodes:
            self.network_nodes[sender] = AtomSpaceNode(
                node_id=message.get("node_id", sender),
                host=sender.split(":")[0],
                port=int(sender.split(":")[1]),
                status="connected",
                capabilities=message.get("capabilities", [])
            )
        
        # Send handshake response
        response = {
            "type": "handshake_response",
            "node_id": self.node_id,
            "status": "connected",
            "timestamp": time.time()
        }
        
        await self._send_message(sender, response)
    
    async def _sync_with_node(self, target_node: str, sync_op: SyncOperation) -> bool:
        """Synchronize AtomSpace with a specific node."""
        try:
            sync_message = {
                "type": "sync_request",
                "operation_id": sync_op.operation_id,
                "source_node": sync_op.source_node,
                "timestamp": sync_op.timestamp,
                "data": sync_op.data
            }
            
            success = await self._send_message(target_node, sync_message)
            
            if success:
                # Update node sync time
                if target_node in self.network_nodes:
                    self.network_nodes[target_node].last_sync = time.time()
            
            return success
            
        except Exception as e:
            self.logger.error(f"Failed to sync with node {target_node}: {e}")
            return False
    
    async def _send_atom_operation(self, target_node: str, operation: SyncOperation) -> bool:
        """Send atom operation to target node."""
        try:
            op_message = {
                "type": "atom_operation",
                "operation_id": operation.operation_id,
                "operation_type": operation.operation_type,
                "source_node": operation.source_node,
                "timestamp": operation.timestamp,
                "data": operation.data
            }
            
            return await self._send_message(target_node, op_message)
            
        except Exception as e:
            self.logger.error(f"Failed to send atom operation to {target_node}: {e}")
            return False
    
    async def _send_message(self, target_node: str, message: Dict[str, Any]) -> bool:
        """Send message to target node."""
        try:
            if target_node in self.client_connections:
                connection = self.client_connections[target_node]
                data = pickle.dumps(message)
                connection.send(data)
                return True
            else:
                self.logger.warning(f"No connection to target node: {target_node}")
                return False
                
        except Exception as e:
            self.logger.error(f"Failed to send message to {target_node}: {e}")
            return False
    
    async def _sync_fallback_memory(self, target_nodes: List[str] = None) -> bool:
        """Synchronize fallback memory when OpenCog is not available."""
        try:
            # In fallback mode, we simulate sync by sharing dictionary data
            sync_data = {
                "memory_snapshot": dict(self.local_atomspace),
                "timestamp": time.time(),
                "node_id": self.node_id
            }
            
            # Send to all connected nodes
            results = []
            for node in target_nodes or list(self.network_nodes.keys()):
                result = await self._send_message(node, {
                    "type": "fallback_sync",
                    "data": sync_data
                })
                results.append(result)
            
            return all(results) if results else True
            
        except Exception as e:
            self.logger.error(f"Fallback memory sync failed: {e}")
            return False
    
    async def _handle_sync_request(self, sender: str, message: Dict[str, Any]):
        """Handle incoming synchronization request."""
        # For now, acknowledge the sync request
        response = {
            "type": "sync_response",
            "operation_id": message.get("operation_id"),
            "status": "completed",
            "timestamp": time.time()
        }
        
        await self._send_message(sender, response)
    
    async def _handle_atom_operation(self, sender: str, message: Dict[str, Any]):
        """Handle incoming atom operation."""
        # For now, acknowledge the operation
        response = {
            "type": "operation_response",
            "operation_id": message.get("operation_id"),
            "status": "completed",
            "timestamp": time.time()
        }
        
        await self._send_message(sender, response)
    
    async def _handle_status_request(self, sender: str, message: Dict[str, Any]):
        """Handle status request."""
        status = self.get_network_status()
        response = {
            "type": "status_response",
            "timestamp": time.time(),
            "status": status
        }
        
        await self._send_message(sender, response)


def create_distributed_atomspace_manager(node_id: str = None, 
                                        host: str = "localhost", 
                                        port: int = 17001) -> NetworkAtomSpaceManager:
    """Create a new distributed AtomSpace manager instance."""
    return NetworkAtomSpaceManager(node_id, host, port)