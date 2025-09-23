# Distributed Agent Networks with Shared AtomSpace

This document describes the implementation of distributed cognitive agent networks with shared AtomSpace functionality for PyCog-Zero, enabling cognitive agents to collaborate across network boundaries while maintaining a synchronized knowledge base.

## Overview

The Distributed Agent Networks system provides:
- **Network-wide AtomSpace Sharing**: Synchronized cognitive memory across all network nodes
- **Distributed Agent Discovery**: Automatic discovery and registration of agents across the network
- **Cross-Network Task Coordination**: Distribute complex tasks across multiple network nodes
- **Distributed Reasoning**: Collaborative reasoning using shared AtomSpace knowledge
- **Network Resilience**: Fault-tolerant operations with graceful degradation
- **Agent-Zero Integration**: Seamless integration with existing Agent-Zero workflows

## Architecture Components

### Core Components

1. **NetworkAtomSpaceManager** (`python/helpers/distributed_atomspace.py`)
   - Manages AtomSpace synchronization across network nodes
   - Handles atom replication and consistency maintenance
   - Provides conflict resolution for distributed operations
   - Supports both OpenCog and fallback memory modes

2. **DistributedAgentNetwork** (`python/helpers/distributed_agent_network.py`)
   - Manages the network topology and node discovery
   - Handles agent registration and capability matching
   - Coordinates distributed task execution
   - Provides network-wide reasoning capabilities

3. **DistributedAgentNetworkTool** (`python/tools/distributed_agent_network.py`)
   - Agent-Zero tool interface for distributed operations
   - Provides high-level API for network interactions
   - Integrates with Agent-Zero's tool ecosystem
   - Handles configuration and lifecycle management

### Network Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Node A        │    │   Node B        │    │   Node C        │
│                 │    │                 │    │                 │
│ Agent Network   │◄──►│ Agent Network   │◄──►│ Agent Network   │
│ Port: 17002     │    │ Port: 17003     │    │ Port: 17004     │
│                 │    │                 │    │                 │
│ AtomSpace Sync  │◄──►│ AtomSpace Sync  │◄──►│ AtomSpace Sync  │
│ Port: 18002     │    │ Port: 18003     │    │ Port: 18004     │
│                 │    │                 │    │                 │
│ Local Agents:   │    │ Local Agents:   │    │ Local Agents:   │
│ - Reasoner A1   │    │ - Analyzer B1   │    │ - Planner C1    │
│ - Memory A2     │    │ - Validator B2  │    │ - Executor C2   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Shared AtomSpace│
                    │   Knowledge     │
                    │     Base        │
                    └─────────────────┘
```

## Configuration

### Network Configuration (`conf/config_distributed_network.json`)

```json
{
  "enabled": true,
  "network": {
    "host": "localhost",
    "port": 17002,
    "atomspace_port": 18002,
    "bootstrap_nodes": [
      ["192.168.1.100", 17002],
      ["192.168.1.101", 17002]
    ]
  },
  "agents": {
    "auto_join_network": true,
    "max_agents_per_task": 5,
    "capabilities": [
      "reasoning", "memory", "cognitive_processing",
      "planning", "analysis", "pattern_recognition"
    ]
  },
  "atomspace": {
    "sync_interval": 5.0,
    "conflict_resolution": "latest_wins",
    "replication_strategy": "full_mesh"
  }
}
```

### Key Configuration Options

- **network.port**: Base port for agent network communication (AtomSpace uses port + 1000)
- **bootstrap_nodes**: List of initial nodes to connect to when joining the network
- **agents.auto_join_network**: Automatically register local Agent-Zero as network participant
- **atomspace.sync_interval**: How often to synchronize AtomSpace across nodes (seconds)
- **atomspace.conflict_resolution**: Strategy for resolving conflicting updates ("latest_wins", "merge", "custom")

## Usage Examples

### Basic Network Setup

```python
from python.tools.distributed_agent_network import DistributedAgentNetworkTool

# Create tool instance (normally done by Agent-Zero automatically)
tool = DistributedAgentNetworkTool(agent)

# Start and join distributed network
response = await tool.execute(operation="start", 
                             bootstrap_nodes=[["192.168.1.100", 17002]])
print(response.message)

# Get network status
status_response = await tool.execute(operation="status")
print(f"Network status: {status_response.data}")
```

### Agent Discovery and Task Distribution

```python
# Discover available agents across the network
agents_response = await tool.execute(operation="discover")
print(f"Found {agents_response.data['total_agents']} agents across network")

# Create distributed task
task_response = await tool.execute(
    operation="create_task",
    description="Analyze complex dataset patterns",
    required_capabilities=["analysis", "pattern_recognition"],
    priority=2
)

task_id = task_response.data["task_id"]

# Coordinate task execution across network
coordination_response = await tool.execute(
    operation="coordinate", 
    task_id=task_id
)
print(f"Task coordination: {coordination_response.data['status']}")
```

### Distributed Reasoning with Shared AtomSpace

```python
# Execute distributed reasoning across network agents
reasoning_response = await tool.execute(
    operation="reasoning",
    query="What is the optimal approach to multi-agent coordination?",
    participating_agents=None  # Use all available agents
)

print(f"Reasoning result: {reasoning_response.data}")
print(f"Participating agents: {reasoning_response.data['participating_agents']}")
print(f"AtomSpace synchronized: {reasoning_response.data['atomspace_synchronized']}")
```

### AtomSpace Synchronization

```python
# Manually trigger AtomSpace synchronization
sync_response = await tool.execute(operation="sync")
print(f"Sync status: {sync_response.data['sync_success']}")
print(f"Network nodes: {sync_response.data['atomspace_status']['connected_nodes']}")
print(f"AtomSpace size: {sync_response.data['atomspace_status']['atomspace_size']} atoms")
```

## Programmatic API Usage

### Direct Network Management

```python
from python.helpers.distributed_agent_network import create_distributed_agent_network
from python.helpers.multi_agent import AgentProfile

# Create distributed network
network = create_distributed_agent_network(
    node_id="my_node",
    host="localhost", 
    port=17002
)

# Start network
await network.start_distributed_network(
    bootstrap_nodes=[("192.168.1.100", 17002)]
)

# Register local agent
agent_profile = AgentProfile(
    agent_id="my_agent",
    name="Cognitive Reasoner",
    role="reasoner",
    capabilities=["reasoning", "analysis", "planning"]
)

await network.register_local_agent(agent_profile)

# Create and coordinate distributed task
task = await network.create_distributed_task(
    description="Distributed problem solving",
    required_capabilities=["reasoning", "planning"]
)

if task:
    result = await network.coordinate_distributed_task(task.task_id)
    print(f"Task result: {result}")
```

### AtomSpace Network Management

```python
from python.helpers.distributed_atomspace import create_distributed_atomspace_manager

# Create AtomSpace manager
atomspace_manager = create_distributed_atomspace_manager(
    node_id="atomspace_node",
    host="localhost",
    port=18002
)

# Start AtomSpace network service
await atomspace_manager.start_network_service()

# Connect to other AtomSpace nodes
await atomspace_manager.connect_to_network_node("192.168.1.100", 18002)

# Synchronize AtomSpace across network
sync_success = await atomspace_manager.synchronize_atomspace()

# Replicate specific atom operations
await atomspace_manager.replicate_atom_operation(
    operation="add_atom",
    atom_data={"type": "ConceptNode", "name": "distributed_knowledge"}
)
```

## Network Topology and Scaling

### Node Types

1. **Coordinator Nodes**: Manage network topology and task distribution
2. **Worker Nodes**: Host cognitive agents and execute tasks
3. **Storage Nodes**: Specialized for AtomSpace storage and replication
4. **Bridge Nodes**: Connect different network segments

### Scaling Considerations

- **Horizontal Scaling**: Add more nodes to increase capacity
- **Load Balancing**: Distribute tasks based on agent capabilities and node load
- **Network Partitioning**: Handle network splits gracefully with local operations
- **Resource Management**: Monitor and manage computational resources across nodes

### Performance Optimization

```python
# Configure for large-scale deployment
config = {
    "network": {
        "connection_pool_size": 20,
        "message_buffer_size": 8192
    },
    "atomspace": {
        "sync_interval": 10.0,  # Reduce frequency for large networks
        "batch_sync_size": 1000  # Sync atoms in batches
    },
    "tasks": {
        "max_concurrent_tasks": 50,
        "task_queue_size": 200
    }
}
```

## Security and Authentication

### Basic Security

```python
# Configure basic security settings
security_config = {
    "security": {
        "enable_authentication": true,
        "allowed_nodes": [
            "192.168.1.100", "192.168.1.101", "192.168.1.102"
        ],
        "message_encryption": true,
        "max_connections_per_node": 10
    }
}
```

### Advanced Security (Future Enhancement)

- **Node Authentication**: Certificate-based node authentication
- **Message Encryption**: End-to-end encryption of network messages
- **Access Control**: Role-based access control for different operations
- **Audit Logging**: Comprehensive logging of network operations

## Monitoring and Diagnostics

### Network Monitoring

```python
# Get comprehensive network status
status = await tool.execute(operation="status")
network_data = status.data

print(f"Network Health:")
print(f"- Connected Nodes: {network_data['network']['connected_nodes']}")
print(f"- Total Agents: {network_data['network']['total_agents']}")
print(f"- Active Tasks: {network_data['network']['active_tasks']}")
print(f"- AtomSpace Sync Status: {network_data['atomspace']['last_sync']}")
```

### Performance Metrics

- **Network Latency**: Measure communication delays between nodes
- **Task Completion Rate**: Track successful task coordination
- **AtomSpace Sync Performance**: Monitor synchronization speed and success rate
- **Agent Availability**: Track agent online/offline status

## Error Handling and Resilience

### Network Failures

- **Node Disconnection**: Gracefully handle node failures
- **Partial Network Splits**: Continue operations with available nodes
- **AtomSpace Consistency**: Maintain data consistency during network issues
- **Automatic Reconnection**: Automatically reconnect when nodes come back online

### Fault Tolerance

```python
# Configure fault tolerance
fault_tolerance_config = {
    "resilience": {
        "max_retry_attempts": 3,
        "retry_backoff": 2.0,
        "node_timeout": 30.0,
        "health_check_interval": 15.0
    }
}
```

## Testing and Validation

### Running Tests

```bash
# Run comprehensive distributed network tests
python3 test_distributed_agent_networks.py

# Test specific components
python3 -c "from python.helpers.distributed_atomspace import create_distributed_atomspace_manager; print('AtomSpace networking available')"
python3 -c "from python.helpers.distributed_agent_network import create_distributed_agent_network; print('Agent networking available')"
```

### Test Coverage

- **AtomSpace Synchronization**: Test data consistency across nodes
- **Agent Discovery**: Test network-wide agent discovery
- **Task Distribution**: Test distributed task creation and coordination
- **Network Resilience**: Test behavior under node failures
- **End-to-End Workflows**: Test complete distributed reasoning workflows

## Troubleshooting

### Common Issues

1. **Connection Failures**
   - Check network connectivity between nodes
   - Verify port availability (default: 17002 for agents, 18002 for AtomSpace)
   - Ensure firewall rules allow connections

2. **Synchronization Problems**
   - Check AtomSpace service status on all nodes
   - Verify network connectivity for AtomSpace ports
   - Review sync_interval configuration

3. **Agent Registration Issues**
   - Ensure agent capabilities are properly defined
   - Check agent profile format and required fields
   - Verify network connectivity

4. **Performance Issues**
   - Monitor network bandwidth usage
   - Check CPU and memory utilization
   - Review task distribution patterns

### Debug Mode

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Create tool with detailed logging
tool = DistributedAgentNetworkTool(agent)
tool.config["logging"]["level"] = "DEBUG"
```

## Integration with Existing Systems

### Agent-Zero Integration

The distributed agent network seamlessly integrates with existing Agent-Zero workflows:

```python
# In Agent-Zero tool execution
def execute_with_network_support(self):
    # Use local processing if network unavailable
    if not self.has_network_connectivity():
        return self.execute_locally()
    
    # Use distributed processing when network available
    distributed_tool = get_distributed_agent_network_tool(self.agent)
    return await distributed_tool.execute(operation="reasoning", 
                                         query=self.query)
```

### Multi-Agent Framework Compatibility

The system extends the existing local multi-agent framework without breaking changes:

```python
from python.helpers.multi_agent import CognitiveMultiAgentSystem
from python.helpers.distributed_agent_network import DistributedAgentNetwork

# Use local multi-agent system
local_system = CognitiveMultiAgentSystem(num_agents=3)

# Extend to distributed network
distributed_system = DistributedAgentNetwork()
await distributed_system.start_distributed_network()

# Both systems work together seamlessly
```

## Future Enhancements

### Planned Features

1. **Advanced Consensus Mechanisms**: Implement distributed consensus algorithms
2. **Dynamic Load Balancing**: Intelligent task distribution based on node capacity
3. **Hierarchical Networks**: Support for multi-level network topologies  
4. **Cross-Cloud Deployment**: Support for deployment across multiple cloud providers
5. **Advanced Security**: Certificate-based authentication and encryption
6. **Machine Learning Integration**: ML-based optimization of network operations

### Roadmap Integration

This implementation completes the "Distributed cognitive agent networks with shared AtomSpace" task from the PyCog-Zero Genesis roadmap. It provides:

- ✅ Network-distributed agent collaboration
- ✅ Shared AtomSpace across network boundaries  
- ✅ Distributed task coordination
- ✅ Network-wide cognitive reasoning
- ✅ Agent-Zero tool integration
- ✅ Comprehensive testing and documentation

---

*This implementation establishes PyCog-Zero as a leading platform for distributed cognitive agent networks, enabling scalable and resilient AI agent collaboration across network boundaries.*