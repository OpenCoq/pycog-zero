# ECAN Cross-Tool Integration Documentation

## Overview

This document describes the ECAN (Economic Attention Networks) integration implemented across PyCog-Zero cognitive tools, providing centralized attention management and coordination.

## Architecture

### Core Components

1. **ECAN Coordinator** (`python/helpers/ecan_coordinator.py`)
   - Centralized attention management across all cognitive tools
   - Priority-based attention allocation system
   - Cross-tool attention synchronization
   - Real-time metrics and monitoring
   - Fallback mechanisms for development environments

2. **Enhanced Cognitive Tools**
   - **Cognitive Reasoning** (`python/tools/cognitive_reasoning.py`): Attention-guided reasoning
   - **Cognitive Memory** (`python/tools/cognitive_memory.py`): Importance-based memory operations
   - **Meta-Cognition** (`python/tools/meta_cognition.py`): Enhanced with coordinator integration

### Integration Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    ECAN Coordinator                         │
│  ┌─────────────────┐ ┌─────────────────┐ ┌───────────────┐  │
│  │   Attention     │ │   Priority      │ │   Metrics     │  │
│  │   Allocation    │ │   Management    │ │   Monitoring  │  │
│  └─────────────────┘ └─────────────────┘ └───────────────┘  │
└─────────────────────────┬───────────────────────────────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
   ┌────▼─────┐      ┌────▼─────┐      ┌───▼──────┐
   │Cognitive │      │Cognitive │      │   Meta   │
   │Reasoning │      │ Memory   │      │Cognition │
   └──────────┘      └──────────┘      └──────────┘
```

## Usage

### Basic Integration

All cognitive tools automatically register with the ECAN coordinator during initialization:

```python
# Automatic registration in cognitive tools
if ECAN_COORDINATOR_AVAILABLE:
    register_tool_with_ecan("cognitive_reasoning", default_priority=1.5)
```

### Attention Requests

Tools request attention allocation for specific operations:

```python
# Request attention for reasoning task
if ECAN_COORDINATOR_AVAILABLE:
    attention_requested = request_attention_for_tool(
        tool_name="cognitive_reasoning",
        priority=2.0,
        context="Complex logical reasoning",
        concepts=["logic", "inference", "deduction"],
        importance_multiplier=1.2
    )
```

### Cross-Tool Synchronization

The coordinator provides synchronized attention allocation:

```python
from python.helpers.ecan_coordinator import get_ecan_coordinator

coordinator = get_ecan_coordinator()
sync_data = coordinator.synchronize_attention()
```

## Configuration

### ECAN Settings

The system supports both OpenCog ECAN and fallback mechanisms:

```python
# OpenCog ECAN (when available)
- Real OpenCog AttentionBank and ECANAgent
- STI (Short-Term Importance) allocation
- Dynamic attention dynamics

# Fallback Mechanisms (development environments)  
- Priority-based weighting system
- Normalized attention allocation
- Cross-tool coordination without OpenCog
```

### Priority Levels

Default priority levels for cognitive tools:

- **Meta-Cognition**: 2.0 (highest priority for self-reflection)
- **Cognitive Reasoning**: 1.5 (high priority for reasoning tasks)
- **Cognitive Memory**: 1.0 (standard priority for memory operations)

## Testing

### Integration Tests

Run the comprehensive test suite:

```bash
python3 test_ecan_integration.py
```

### Demo and Validation

Experience the full integration:

```bash
python3 demo_ecan_integration.py
```

## Performance Metrics

The coordinator tracks various performance metrics:

- **Total allocations**: Number of attention allocations performed
- **Average entropy**: Measure of attention distribution diversity
- **Cross-tool interactions**: Number of coordination events
- **Attention conflicts**: Conflicts resolved during allocation

### Monitoring

```python
coordinator = get_ecan_coordinator()
metrics = coordinator.get_metrics()
print(f"Active tools: {metrics['active_tools']}")
print(f"Total allocations: {metrics['total_allocations']}")
print(f"Average entropy: {metrics['average_entropy']:.3f}")
```

## Development Environment Support

### Fallback Mechanisms

When OpenCog is not available, the system automatically uses fallback mechanisms:

1. **Priority-based allocation**: Uses tool priorities and importance multipliers
2. **Normalized weighting**: Ensures consistent attention distribution
3. **Cross-tool coordination**: Maintains synchronization without OpenCog
4. **Metrics collection**: Provides monitoring even in fallback mode

### Production Deployment

For production environments with OpenCog:

1. Install OpenCog dependencies:
   ```bash
   pip install opencog-atomspace opencog-python opencog-ecan
   ```

2. The system automatically detects and uses real ECAN components
3. Full STI allocation and attention dynamics become available
4. Enhanced performance and cognitive capabilities activate

## Best Practices

### Tool Integration

1. **Register Early**: Register tools with ECAN coordinator during initialization
2. **Request Appropriately**: Request attention with appropriate priority and context
3. **Extract Concepts**: Provide meaningful concept extraction for attention allocation
4. **Monitor Performance**: Use metrics to optimize attention allocation

### Attention Management

1. **Priority Setting**: Set tool priorities based on cognitive importance
2. **Context Provision**: Provide clear context for attention requests
3. **Concept Limiting**: Limit concept lists to 6-8 items for efficient processing
4. **Synchronization**: Use synchronization for coordinated multi-tool operations

## Troubleshooting

### Common Issues

1. **Import Errors**: Ensure all tools import the ECAN coordinator correctly
2. **Registration Failures**: Verify tools register before requesting attention
3. **OpenCog Warnings**: OpenCog unavailability warnings are expected in development
4. **Test Failures**: Run tests to validate integration functionality

### Debug Information

Enable debug output to monitor ECAN activity:

```python
coordinator = get_ecan_coordinator()
allocation = coordinator.get_attention_allocation()
print(f"Attention distribution: {allocation.concept_allocations}")
```

## Future Enhancements

Planned improvements for ECAN integration:

1. **Advanced Attention Models**: Machine learning-based attention prediction
2. **Dynamic Priority Adjustment**: Adaptive priority based on performance metrics
3. **Attention Persistence**: Long-term attention pattern storage and learning
4. **Multi-Agent Coordination**: Attention coordination across multiple agents
5. **Performance Optimization**: Enhanced algorithms for large-scale deployments

---

## API Reference

### Core Classes

#### `ECANCoordinator`
Main coordinator class for attention management.

#### `AttentionRequest`  
Data class representing attention allocation requests.

#### `AttentionAllocation`
Data class containing attention allocation results.

### Core Functions

#### `get_ecan_coordinator(shared_atomspace=None)`
Get or create the global ECAN coordinator instance.

#### `register_tool_with_ecan(tool_name, default_priority=1.0)`
Register a cognitive tool with the ECAN coordinator.

#### `request_attention_for_tool(tool_name, priority, context, concepts, importance_multiplier=1.0)`
Request attention allocation for a specific tool.

---

*This documentation covers the complete ECAN cross-tool integration implementation in PyCog-Zero.*