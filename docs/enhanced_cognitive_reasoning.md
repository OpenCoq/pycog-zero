# Enhanced Cognitive Reasoning Tool Documentation

## Overview

The Enhanced Cognitive Reasoning Tool is a significant upgrade to PyCog-Zero's cognitive architecture, providing new atomspace bindings and advanced cross-tool integration capabilities. This tool bridges OpenCog's symbolic reasoning with Agent-Zero's framework, enabling sophisticated cognitive operations.

## Key Features

### 🧠 Core Enhancements

1. **Shared AtomSpace Integration**
   - Uses shared atomspace instance from AtomSpaceToolHub
   - Enables cross-tool data sharing and coordination
   - Maintains consistency across cognitive operations

2. **Multiple Operation Modes**
   - `reason`: Core cognitive reasoning with enhanced context
   - `analyze_patterns`: Advanced pattern detection and analysis
   - `cross_reference`: Integration with other atomspace tools
   - `status`: Comprehensive system status reporting
   - `share_knowledge`: Knowledge sharing with tool hub

3. **Enhanced Fallback Mode**
   - Comprehensive reasoning without OpenCog dependencies
   - Pattern-based analysis using linguistic features
   - Graceful degradation with full functionality

4. **Cross-Tool Integration**
   - Seamless integration with atomspace_tool_hub.py
   - Data sharing with atomspace_memory_bridge.py
   - Coordinated execution across cognitive tools

## Usage Examples

### Basic Cognitive Reasoning

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool

# Initialize tool
tool = CognitiveReasoningTool(agent, 'cognitive_reasoning', None, {}, '', None)

# Basic reasoning
response = await tool.execute(
    "What is the relationship between learning and memory?"
)
print(response.message)
```

### Pattern Analysis

```python
# Analyze patterns in queries
response = await tool.execute(
    "How do neural networks learn patterns?",
    operation="analyze_patterns"
)

# Parse results
data = json.loads(response.message.split("Data: ")[1])
patterns = data["analysis"]
print(f"Patterns found: {patterns}")
```

### Status Monitoring

```python
# Get comprehensive system status
response = await tool.execute("", operation="status")

# Parse status information
data = json.loads(response.message.split("Data: ")[1])
status = data["status"]

print(f"OpenCog Available: {status['opencog_available']}")
print(f"Fallback Mode: {status['fallback_mode']}")
print(f"Cross-tool Integration: {status['cross_tool_integration']}")
```

### Cross-Tool Integration

```python
# Cross-reference with other tools
response = await tool.execute(
    "machine learning concepts",
    operation="cross_reference",
    supporting_tools=["memory_bridge", "search_engine"]
)
```

### Knowledge Sharing

```python
# Share knowledge with atomspace hub
response = await tool.execute(
    "cognitive reasoning patterns",
    operation="share_knowledge"
)
```

## Configuration

The tool uses enhanced configuration from `conf/config_cognitive.json`:

```json
{
  "cognitive_mode": true,
  "opencog_enabled": true,
  "neural_symbolic_bridge": true,
  "reasoning_config": {
    "pln_enabled": true,
    "pattern_matching": true,
    "forward_chaining": false,
    "backward_chaining": true
  },
  "atomspace_config": {
    "persistence_backend": "memory",
    "cross_tool_sharing": true,
    "attention_allocation": "basic"
  }
}
```

## Advanced Features

### Enhanced Reasoning Strategies

1. **Pattern Matching Reasoning**
   - Creates inheritance and similarity relationships
   - Context-aware pattern detection
   - Memory association integration

2. **PLN (Probabilistic Logic Networks) Reasoning**
   - Enhanced evaluation links with truth values
   - Context-based confidence assessment
   - Cross-tool integration markers

3. **Backward Chaining Reasoning**
   - Goal-directed reasoning implementation
   - Step-by-step reasoning chain construction
   - Achievement tracking

### Context Building

The tool builds rich reasoning context using:
- Cross-tool data sharing
- Memory associations
- Reasoning hints from parameters
- Related concept extraction

### Error Handling

Comprehensive error handling includes:
- Graceful degradation when OpenCog unavailable
- Fallback reasoning with linguistic analysis
- Cross-tool integration error recovery
- Configuration loading fallbacks

## Integration with Existing Tools

### AtomSpace Tool Hub Integration

```python
# Automatic integration with shared atomspace
shared_atomspace = AtomSpaceToolHub.get_shared_atomspace()
if shared_atomspace:
    self.atomspace = shared_atomspace
```

### Memory Bridge Integration

```python
# Cross-reference with memory bridge
memory_data_response = await self.tool_hub.retrieve_shared_data(
    tool_name="memory_bridge"
)
```

### Tool Registration

```python
# Register with hub for coordination
registration_data = {
    "tool_type": "cognitive_reasoning",
    "capabilities": ["reasoning", "pattern_matching", "inference"],
    "atomspace_operations": ["create_atoms", "query_patterns", "inference_chains"],
    "status": "active"
}

await self.tool_hub.share_tool_data(
    tool_name="cognitive_reasoning",
    data_type="registration",
    data=registration_data
)
```

## Testing

### Running Tests

```bash
# Run comprehensive tests
python3 tests/test_enhanced_cognitive_reasoning.py

# Run integration tests
python3 integration_test_cognitive_reasoning.py
```

### Test Coverage

- ✅ Tool initialization and configuration
- ✅ All reasoning operations (reason, analyze_patterns, etc.)
- ✅ Fallback mode functionality
- ✅ Error handling and edge cases
- ✅ Cross-tool integration scenarios
- ✅ Response formatting and data structures

## Performance Characteristics

### Fallback Mode Performance
- Fast linguistic pattern analysis
- Minimal memory footprint
- No external dependencies

### OpenCog Mode Performance (when available)
- Rich symbolic reasoning capabilities
- Shared atomspace efficiency
- Cross-tool data sharing overhead

## Troubleshooting

### Common Issues

1. **OpenCog Not Available**
   - Expected behavior: Tool runs in fallback mode
   - Solution: Install opencog-atomspace, opencog-python for full features
   - Impact: Limited to linguistic pattern analysis

2. **Tool Hub Not Available**
   - Symptom: Cross-tool operations return "hub_unavailable"
   - Solution: Ensure atomspace tools are properly imported
   - Workaround: Individual tool operations still function

3. **Configuration Issues**
   - Symptom: "Could not load cognitive config" warnings
   - Solution: Check conf/config_cognitive.json exists and is valid
   - Fallback: Tool uses default configuration

### Debugging

Enable verbose logging:
```python
# Check tool initialization
tool._initialize_if_needed()
print(f"Initialized: {tool.initialized}")
print(f"Config: {tool.config}")
print(f"AtomSpace available: {tool.atomspace is not None}")
```

## Future Enhancements

### Planned Features
- Neural-symbolic bridge integration
- ECAN attention allocation
- PLN inference engine integration
- Distributed atomspace coordination
- Advanced pattern learning

### Development Roadmap
1. **Phase 1**: ✅ Enhanced atomspace bindings (Current)
2. **Phase 2**: Neural-symbolic integration
3. **Phase 3**: Advanced learning capabilities
4. **Phase 4**: Distributed cognitive networks

## Contributing

To contribute to the enhanced cognitive reasoning tool:

1. Follow existing code patterns and error handling
2. Add comprehensive tests for new features
3. Maintain backward compatibility
4. Update documentation for new capabilities
5. Consider both OpenCog and fallback modes

## API Reference

### CognitiveReasoningTool Class

#### Methods

- `execute(query, operation="reason", **kwargs)`: Main execution method
- `_initialize_if_needed()`: Initialize tool and configuration
- `parse_query_to_atoms(query, context)`: Convert queries to atomspace atoms
- `enhanced_pattern_matching_reasoning(atoms, context)`: Advanced pattern matching
- `enhanced_pln_reasoning(atoms, context)`: PLN reasoning with context
- `backward_chaining_reasoning(atoms, context)`: Goal-directed reasoning

#### Configuration Properties

- `config`: Loaded cognitive configuration
- `atomspace`: OpenCog AtomSpace instance or None
- `initialized`: Boolean indicating OpenCog availability
- `tool_hub`: Reference to AtomSpace tool hub

#### Response Format

All operations return Response objects with:
```python
Response(
    message="Operation completed for: query\nData: {json_data}",
    break_loop=False
)
```

Where `json_data` contains structured results including:
- `query`: Original query
- `operation`: Operation performed
- `status`: Success/error/fallback status
- Operation-specific data fields

---

*This enhanced cognitive reasoning tool represents a significant advancement in PyCog-Zero's cognitive architecture, providing robust atomspace bindings while maintaining compatibility and graceful fallback behavior.*