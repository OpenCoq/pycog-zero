# URE (Unified Rule Engine) Python Bindings Integration

## Overview

The PyCog-Zero URE integration provides Python bindings for OpenCog's Unified Rule Engine, enabling forward and backward chaining inference within the Agent-Zero cognitive framework.

## Features

### Core URE Operations
- **Forward Chaining**: Derive new knowledge from existing facts using rules
- **Backward Chaining**: Prove goals by working backwards through rule chains  
- **Rule Management**: Create and manage rulebases with different rule sets
- **Cross-Tool Integration**: Seamless integration with other PyCog-Zero cognitive tools

### Integration Benefits
- **Shared AtomSpace**: Uses shared memory across cognitive tools for efficiency
- **Agent-Zero Compatible**: Designed specifically for Agent-Zero agent framework
- **Graceful Fallback**: Works without OpenCog installation using fallback reasoning
- **Performance Optimized**: Minimal overhead for rule-based inference operations

## Usage

### Basic URE Operations

```python
# In Agent-Zero context
from python.tools.ure_tool import UREChainTool

# Backward chaining to prove a goal
response = await ure_tool.execute(
    "if A implies B and B implies C, prove A implies C", 
    operation="backward_chain"
)

# Forward chaining from known facts
response = await ure_tool.execute(
    "given A and A implies B, derive B",
    operation="forward_chain"
)
```

### Integration with Cognitive Reasoning

```python
# Use URE through cognitive reasoning tool
from python.tools.cognitive_reasoning import CognitiveReasoningTool

# Delegate to URE for backward chaining
response = await cognitive_tool.execute(
    "prove logical statement using rules",
    operation="ure_backward_chain"
)

# Delegate to URE for forward chaining  
response = await cognitive_tool.execute(
    "derive conclusions from facts",
    operation="ure_forward_chain"
)
```

### Advanced Operations

```python
# Create custom rulebase
response = await ure_tool.execute(
    "create specialized rulebase",
    operation="create_rulebase", 
    rulebase_name="logic_rules",
    rules=["deduction", "modus_ponens", "syllogism"]
)

# List available rules
response = await ure_tool.execute("", operation="list_rules")

# Get URE system status
response = await ure_tool.execute("", operation="status")
```

## Configuration

### URE Configuration (`conf/config_cognitive.json`)

```json
{
  "ure_config": {
    "ure_enabled": true,
    "forward_chaining": true, 
    "backward_chaining": true,
    "max_iterations": 1000,
    "complexity_penalty": 0.01,
    "trace_enabled": false,
    "default_rulebase": "default_rulebase",
    "available_rules": [
      "deduction",
      "modus_ponens",
      "syllogism", 
      "abduction",
      "induction"
    ]
  }
}
```

### Configuration Options

- **`ure_enabled`**: Enable/disable URE functionality
- **`forward_chaining`**: Enable forward chaining operations
- **`backward_chaining`**: Enable backward chaining operations  
- **`max_iterations`**: Maximum iterations for chaining operations
- **`complexity_penalty`**: Penalty factor for complex rule chains
- **`trace_enabled`**: Enable inference tracing for debugging
- **`default_rulebase`**: Name of default rulebase to use
- **`available_rules`**: List of available rule types

## Architecture

### Class Structure

```
UREChainTool
├── Forward Chaining (_perform_forward_chaining)
├── Backward Chaining (_perform_backward_chaining) 
├── Rulebase Management (_get_or_create_rulebase)
├── Query Parsing (_parse_query_to_target, _parse_query_to_source)
├── Result Formatting (_format_ure_results)
├── Cross-Tool Integration (_setup_cross_tool_integration)
└── Fallback Mode (_fallback_reasoning)
```

### Integration Points

1. **AtomSpace Sharing**: Uses shared AtomSpace from AtomSpaceToolHub or CognitiveReasoningTool
2. **Result Sharing**: Shares inference results with other tools via tool hub
3. **Cognitive Delegation**: CognitiveReasoningTool can delegate operations to URE
4. **Configuration Integration**: Uses Agent-Zero configuration system

## Implementation Details

### Forward Chaining Process

1. Parse query to extract source atoms
2. Create or get rulebase with specified rules
3. Initialize ForwardChainer with source, rulebase, and configuration
4. Execute chaining (`do_chain()`)
5. Retrieve and format results (`get_results()`)
6. Share results with other cognitive tools

### Backward Chaining Process

1. Parse query to extract target/goal atoms
2. Create or get rulebase with specified rules  
3. Initialize BackwardChainer with target, rulebase, and configuration
4. Execute chaining (`do_chain()`)
5. Retrieve and format results (`get_results()`)
6. Share results with other cognitive tools

### Fallback Mode

When OpenCog/URE is not available:
- Performs basic logical pattern recognition
- Detects implication, conjunction, disjunction, negation patterns
- Provides meaningful responses without full inference capabilities
- Maintains API compatibility for development/testing

## Error Handling

### Graceful Degradation
- Falls back to pattern recognition if OpenCog unavailable
- Handles missing dependencies transparently
- Provides informative error messages
- Maintains system stability

### Error Types
- **Import Errors**: OpenCog/URE modules not available
- **Initialization Errors**: AtomSpace or URE setup failures
- **Reasoning Errors**: Issues during inference operations
- **Configuration Errors**: Invalid or missing configuration

## Testing

### Test Coverage

```bash
# Run URE integration tests
python3 tests/test_ure_integration.py

# Run specific test class
python3 -m unittest tests.test_ure_integration.TestUREChainTool

# Run performance tests
python3 -m unittest tests.test_ure_integration.TestUREPerformance
```

### Test Categories
- **Basic Functionality**: Tool initialization, configuration loading
- **Chaining Operations**: Forward/backward chaining with fallback
- **Integration**: Cross-tool communication, shared AtomSpace usage
- **Performance**: Instantiation speed, reasoning performance
- **Error Handling**: Graceful degradation, error recovery

## Development

### Extension Points

1. **Rule Addition**: Add new rules to `available_rules` configuration
2. **Rulebase Customization**: Implement custom rulebase creation logic
3. **Query Parsing**: Enhance query parsing for complex logical expressions
4. **Result Processing**: Add domain-specific result formatting
5. **Integration**: Add integration with additional cognitive tools

### Best Practices

1. **Shared AtomSpace**: Always use shared AtomSpace for cross-tool consistency
2. **Configuration**: Use configuration system for all URE settings
3. **Error Handling**: Implement graceful fallback for missing dependencies
4. **Testing**: Test both OpenCog-enabled and fallback modes
5. **Documentation**: Document custom rules and rulebase configurations

## Examples

### Example 1: Logical Deduction

```python
# Agent-Zero agent using URE for logical reasoning
query = "if all humans are mortal and Socrates is human, is Socrates mortal?"

response = await ure_tool.execute(query, "backward_chain")
# Returns proof chain showing Socrates is mortal
```

### Example 2: Knowledge Derivation

```python
# Forward chaining from facts
facts = "A implies B, B implies C, A is true"
response = await ure_tool.execute(facts, "forward_chain")
# Derives that C is true through rule chain
```

### Example 3: Cross-Tool Reasoning

```python
# Cognitive reasoning with URE delegation
response = await cognitive_tool.execute(
    "use rule-based logic to analyze this problem",
    operation="ure_backward_chain",
    rulebase="logic_rules"
)
```

## Troubleshooting

### Common Issues

1. **OpenCog Not Found**: Install with `pip install opencog-atomspace opencog-python`
2. **URE Import Error**: Ensure URE bindings are properly compiled
3. **Shared AtomSpace Issues**: Check cross-tool integration configuration  
4. **Performance Problems**: Adjust `max_iterations` and `complexity_penalty`

### Debug Mode

Enable tracing for detailed inference information:

```json
{
  "ure_config": {
    "trace_enabled": true
  }
}
```

## Roadmap

### Near Term
- [ ] Enhanced query parsing for complex logical expressions
- [ ] Additional rule types (temporal, modal logic)
- [ ] Performance optimizations for large rule sets
- [ ] Integration with PLN (Probabilistic Logic Networks)

### Long Term  
- [ ] Visual rule execution tracing
- [ ] Machine learning for rule selection
- [ ] Distributed reasoning across multiple agents
- [ ] Natural language to logic translation

---

*This URE integration provides the foundation for advanced logical reasoning in PyCog-Zero agents, enabling sophisticated inference capabilities while maintaining compatibility with the existing Agent-Zero framework.*