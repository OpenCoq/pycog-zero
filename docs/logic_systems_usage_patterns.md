# Logic Systems Usage Patterns for Agent-Zero Integration

## Overview

This document provides comprehensive usage patterns for integrating OpenCog logic systems (Phase 2: Unify and URE) with the Agent-Zero framework in PyCog-Zero. These patterns enable sophisticated cognitive reasoning, logical inference, and goal-directed problem-solving capabilities.

## Table of Contents

1. [Core Concepts](#core-concepts)
2. [URE (Unified Rule Engine) Patterns](#ure-unified-rule-engine-patterns)
3. [Pattern Unification](#pattern-unification)
4. [Integration with Agent-Zero Tools](#integration-with-agent-zero-tools)
5. [Configuration Patterns](#configuration-patterns)
6. [Common Use Cases](#common-use-cases)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## Core Concepts

### Logic Systems in PyCog-Zero

Phase 2 logic systems provide two key capabilities:

1. **URE (Unified Rule Engine)**: Forward and backward chaining for logical inference
2. **Unify System**: Pattern matching and term unification

These systems integrate with Agent-Zero's cognitive architecture through:
- Shared AtomSpace for knowledge representation
- Cross-tool integration for collaborative reasoning
- Fallback mechanisms for graceful degradation

### Architecture Overview

```
┌─────────────────────────────────────────────────┐
│         Agent-Zero Framework                    │
├─────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌──────────────────────┐   │
│  │ URE Chain Tool│  │ Cognitive Reasoning  │   │
│  │               │◄─┤      Tool            │   │
│  └───────┬───────┘  └──────────┬───────────┘   │
│          │                     │                │
│          ▼                     ▼                │
│  ┌─────────────────────────────────────────┐   │
│  │     Shared AtomSpace (Knowledge Base)   │   │
│  └─────────────────────────────────────────┘   │
└─────────────────────────────────────────────────┘
```

## URE (Unified Rule Engine) Patterns

### Pattern 1: Forward Chaining for Inference

**Use Case**: Derive new facts from existing knowledge

**Implementation**:

```python
from python.tools.ure_tool import UREChainTool

# Initialize tool
ure_tool = UREChainTool(agent)

# Define facts/premises
query = "given A is true and A implies B, derive consequences"

# Execute forward chaining
response = await ure_tool.execute(query, "forward_chain")

# Access results
try:
    # Parse response data safely
    if "Data: " in response.message:
        data = json.loads(response.message.split("Data: ")[1])
        print(f"Inferences: {data['results']}")
    else:
        # Fallback: access data directly
        print(f"Inferences: {response.data.get('results', [])}")
except (json.JSONDecodeError, KeyError) as e:
    print(f"Error parsing response: {e}")
```

**Configuration**:

```json
{
  "ure_config": {
    "forward_chaining": true,
    "max_iterations": 1000,
    "complexity_penalty": 0.01
  }
}
```

**When to Use**:
- Deriving consequences from known facts
- Expanding knowledge through logical inference
- Automatic fact discovery

### Pattern 2: Backward Chaining for Goal Proving

**Use Case**: Prove a goal by finding supporting evidence

**Implementation**:

```python
# Define goal to prove
query = "if A implies B and B implies C, then prove A implies C"

# Execute backward chaining
response = await ure_tool.execute(query, "backward_chain")

# Extract proof steps
try:
    if "Data: " in response.message:
        data = json.loads(response.message.split("Data: ")[1])
        proof_steps = data['results']
    else:
        proof_steps = response.data.get('results', [])
    print(f"Proof completed in {len(proof_steps)} steps")
except (json.JSONDecodeError, KeyError) as e:
    print(f"Error parsing response: {e}")
```

**Configuration**:

```json
{
  "ure_config": {
    "backward_chaining": true,
    "max_iterations": 1000,
    "trace_enabled": true
  }
}
```

**When to Use**:
- Goal-directed reasoning
- Proving theorems or hypotheses
- Planning with subgoal decomposition

### Pattern 3: Hybrid Reasoning (Forward + Backward)

**Use Case**: Combine forward and backward chaining for complex reasoning

**Implementation**:

```python
# Step 1: Forward chain from facts
facts_query = "facts: X and Y, rule: X and Y implies Z"
forward_result = await ure_tool.execute(facts_query, "forward_chain")

# Step 2: Backward chain to prove goal
goal_query = "prove goal G given derived facts"
backward_result = await ure_tool.execute(goal_query, "backward_chain")

# Step 3: Synthesize results
def calculate_confidence(forward_result, backward_result):
    """Calculate combined confidence from forward and backward reasoning."""
    forward_count = len(forward_result.data.get('results', []))
    backward_count = len(backward_result.data.get('results', []))
    
    # Confidence increases with successful results from both directions
    forward_conf = min(1.0, forward_count / 10.0)
    backward_conf = min(1.0, backward_count / 10.0)
    
    # Weighted average (backward chaining more reliable for proofs)
    return (forward_conf * 0.4 + backward_conf * 0.6)

combined_insights = {
    "forward_inferences": forward_result.data.get('results', []),
    "backward_proofs": backward_result.data.get('results', []),
    "reasoning_confidence": calculate_confidence(forward_result, backward_result)
}
```

**When to Use**:
- Complex multi-step reasoning tasks
- When both exploration (forward) and validation (backward) are needed
- Strategic problem-solving

### Pattern 4: Custom Rulebase Creation

**Use Case**: Define domain-specific reasoning rules

**Implementation**:

```python
# Create custom rulebase
response = await ure_tool.execute(
    query="create reasoning rules for task planning",
    operation="create_rulebase",
    rulebase_name="task_planning_rules",
    rules=["task_decomposition", "capability_matching", "resource_allocation"]
)

# Use custom rulebase
result = await ure_tool.execute(
    query="plan task with custom rules",
    operation="backward_chain",
    rulebase="task_planning_rules"
)
```

**Configuration**:

```json
{
  "ure_config": {
    "default_rulebase": "task_planning_rules",
    "available_rules": [
      "task_decomposition",
      "capability_matching",
      "resource_allocation"
    ]
  }
}
```

**When to Use**:
- Domain-specific reasoning
- Custom inference patterns
- Specialized problem domains

### Pattern 5: URE Status Monitoring

**Use Case**: Monitor URE system health and performance

**Implementation**:

```python
# Get URE status
response = await ure_tool.execute("", "status")

# Parse status safely
try:
    if "Data: " in response.message:
        status_data = json.loads(response.message.split("Data: ")[1])
    else:
        status_data = response.data
except (json.JSONDecodeError, AttributeError) as e:
    print(f"Error parsing status: {e}")
    status_data = {"status": {"ure_initialized": False}}

# Check status
if status_data.get('status', {}).get('ure_initialized'):
    print("URE fully operational")
    print(f"AtomSpace size: {status_data['status']['atomspace_size']}")
else:
    print("URE in fallback mode")
    print("Limited functionality available")
```

**When to Use**:
- Debugging reasoning issues
- Performance monitoring
- System diagnostics

## Pattern Unification

### Pattern 6: Concept Matching

**Use Case**: Match similar concepts across different representations

**Implementation**:

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool

cognitive_tool = CognitiveReasoningTool(agent)

# Match patterns
pattern_a = "agent has skill programming"
pattern_b = "entity possesses capability coding"

response = await cognitive_tool.execute(
    f"unify patterns: {pattern_a} and {pattern_b}",
    operation="pattern_matching"
)

# Access unification results
matches = response.data.get('pattern_matches', [])
for match in matches:
    print(f"Unified: {match['unified_concept']} (confidence: {match['confidence']})")
```

**When to Use**:
- Semantic similarity detection
- Knowledge integration from multiple sources
- Concept alignment across ontologies

### Pattern 7: Query-Goal Unification

**Use Case**: Map user queries to system goals

**Implementation**:

```python
# User query
user_query = "I want to learn Python programming"

# System goals
available_goals = [
    "teach_programming_skill",
    "provide_learning_resources",
    "create_practice_exercises"
]

# Unify query with goals
response = await cognitive_tool.execute(
    f"match query '{user_query}' with goals: {', '.join(available_goals)}",
    operation="goal_unification"
)

# Get best matching goal
best_match = response.data.get('best_goal_match')
print(f"Matched goal: {best_match['goal']} (confidence: {best_match['confidence']})")
```

**When to Use**:
- Natural language query processing
- Intent recognition
- Task routing

## Integration with Agent-Zero Tools

### Pattern 8: Cognitive Reasoning + URE Integration

**Use Case**: Delegate complex reasoning to URE from cognitive tools

**Implementation**:

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool

cognitive_tool = CognitiveReasoningTool(agent)

# Cognitive tool automatically delegates to URE when needed
response = await cognitive_tool.execute(
    "use logical rules to solve: if all humans are mortal and Socrates is human, is Socrates mortal?",
    operation="logical_reasoning"
)

# The cognitive tool will:
# 1. Detect logical reasoning requirement
# 2. Delegate to URE backward chaining
# 3. Return synthesized results

print(f"Answer: {response.data['conclusion']}")
print(f"Reasoning steps: {response.data['reasoning_steps']}")
```

**Configuration**:

```json
{
  "cross_tool_integration": {
    "cognitive_reasoning": true,
    "ure_chain": true,
    "shared_atomspace": true
  }
}
```

**Benefits**:
- Automatic tool selection
- Seamless integration
- Shared knowledge base

### Pattern 9: AtomSpace Tool Hub Integration

**Use Case**: Share URE results across multiple cognitive tools

**Implementation**:

```python
from python.tools.atomspace_tool_hub import AtomSpaceToolHub

# URE tool shares results automatically
ure_response = await ure_tool.execute(query, "forward_chain")

# Other tools can access shared results
cognitive_tool = CognitiveReasoningTool(agent)
memory_tool = CognitiveMemoryTool(agent)

# Both tools can now access URE-derived facts
# from the shared AtomSpace
```

**Configuration**:

```json
{
  "atomspace_config": {
    "cross_tool_sharing": true,
    "ure_integration": true
  }
}
```

**Benefits**:
- Knowledge sharing between tools
- Reduced redundant computation
- Consistent reasoning across tools

### Pattern 10: Meta-Cognition with URE

**Use Case**: Self-reflection on reasoning processes

**Implementation**:

```python
from python.tools.meta_cognition import MetaCognitionTool

meta_tool = MetaCognitionTool(agent)

# Analyze reasoning performance
response = await meta_tool.execute(
    operation="analyze_reasoning",
    reasoning_task="recent_ure_chains"
)

# Get insights
insights = response.data['meta_insights']
print(f"Average reasoning depth: {insights['avg_chain_depth']}")
print(f"Success rate: {insights['success_rate']}")
print(f"Common patterns: {insights['common_patterns']}")
```

**When to Use**:
- Optimizing reasoning strategies
- Identifying reasoning bottlenecks
- Adaptive learning

## Configuration Patterns

### Pattern 11: Development vs Production Configuration

**Development Configuration**:

```json
{
  "ure_config": {
    "ure_enabled": true,
    "trace_enabled": true,
    "max_iterations": 100,
    "complexity_penalty": 0.05
  }
}
```

**Production Configuration**:

```json
{
  "ure_config": {
    "ure_enabled": true,
    "trace_enabled": false,
    "max_iterations": 1000,
    "complexity_penalty": 0.01
  }
}
```

### Pattern 12: Fallback Mode Configuration

**Graceful Degradation**:

```json
{
  "ure_config": {
    "ure_enabled": true,
    "fallback_mode": "pattern_recognition"
  }
}
```

When OpenCog URE is not available, the system automatically falls back to:
- Pattern recognition
- Simple logical analysis
- Heuristic-based reasoning

### Pattern 13: Performance Tuning

**Optimize for Speed**:

```json
{
  "ure_config": {
    "max_iterations": 100,
    "complexity_penalty": 0.1,
    "trace_enabled": false
  }
}
```

**Optimize for Accuracy**:

```json
{
  "ure_config": {
    "max_iterations": 10000,
    "complexity_penalty": 0.001,
    "trace_enabled": true
  }
}
```

## Common Use Cases

### Use Case 1: Task Planning with URE

```python
def calculate_plan_confidence(plan_steps, validation_data):
    """Calculate confidence in the generated plan."""
    if not plan_steps:
        return 0.0
    
    # Base confidence on plan completeness
    plan_confidence = min(1.0, len(plan_steps) / 5.0)
    
    # Adjust based on validation results
    validation_results = validation_data.get('results', [])
    if validation_results:
        validation_confidence = min(1.0, len(validation_results) / 3.0)
        # Weighted average
        return plan_confidence * 0.6 + validation_confidence * 0.4
    
    return plan_confidence * 0.7  # Reduce confidence if no validation

async def plan_task_with_ure(task_description: str):
    """Plan task execution using URE backward chaining."""
    
    # Step 1: Define goal
    goal_query = f"achieve: {task_description}"
    
    # Step 2: Backward chain to find requirements
    ure_response = await ure_tool.execute(goal_query, "backward_chain")
    
    # Step 3: Extract plan steps safely
    plan_steps = ure_response.data.get('results', [])
    
    # Step 4: Validate plan
    validation_query = f"validate plan: {plan_steps}"
    validation = await ure_tool.execute(validation_query, "forward_chain")
    
    return {
        "task": task_description,
        "plan": plan_steps,
        "validation": validation.data.get('results', []),
        "confidence": calculate_plan_confidence(plan_steps, validation.data)
    }
```

### Use Case 2: Knowledge Base Expansion

```python
async def expand_knowledge_base(seed_facts: list):
    """Expand knowledge base using forward chaining."""
    
    # Create query from facts
    facts_query = "given facts: " + ", ".join(seed_facts)
    
    # Forward chain to derive new knowledge
    response = await ure_tool.execute(facts_query, "forward_chain")
    
    # Store new facts in cognitive memory
    new_facts = response.data['results']
    for fact in new_facts:
        await memory_tool.execute(
            operation="store",
            data={"concept": fact, "source": "ure_inference"}
        )
    
    return {
        "seed_facts": len(seed_facts),
        "derived_facts": len(new_facts),
        "expansion_ratio": len(new_facts) / len(seed_facts)
    }
```

### Use Case 3: Multi-Agent Collaborative Reasoning

```python
def measure_synergy(individual_results, synthesis):
    """Measure collaboration benefit from multi-agent reasoning."""
    # Count total individual insights
    total_individual = sum(len(r.get('insights', [])) for r in individual_results)
    
    # Count synthesized insights
    synthesized_count = len(synthesis.data.get('results', []))
    
    # Synergy metric: synthesized insights beyond simple combination
    if total_individual == 0:
        return 0.0
    
    # Positive synergy when synthesis produces more value than sum of parts
    synergy_ratio = synthesized_count / total_individual
    
    # Synergy is highest when synthesis creates new insights (ratio > 1)
    return min(1.0, synergy_ratio)

async def collaborative_reasoning(problem: str, agents: list):
    """Multiple agents collaborate using shared URE reasoning."""
    
    results = []
    
    # Each agent contributes reasoning
    for agent in agents:
        agent_ure = UREChainTool(agent)
        
        # Forward chain from agent's perspective
        response = await agent_ure.execute(
            f"analyze: {problem} from perspective: {agent.name}",
            "forward_chain"
        )
        
        results.append({
            "agent": agent.name,
            "insights": response.data.get('results', [])
        })
    
    # Synthesize insights using backward chaining
    synthesis_query = "synthesize insights from all agents"
    synthesis = await ure_tool.execute(synthesis_query, "backward_chain")
    
    return {
        "individual_insights": results,
        "synthesized_solution": synthesis.data.get('results', []),
        "collaboration_benefit": measure_synergy(results, synthesis)
    }
```

### Use Case 4: Automated Debugging with Logic

```python
async def debug_with_logic(code_issue: str, symptoms: list):
    """Debug code issues using logical reasoning."""
    
    # Step 1: Forward chain from symptoms to potential causes
    symptoms_query = f"symptoms: {', '.join(symptoms)}"
    causes_response = await ure_tool.execute(symptoms_query, "forward_chain")
    potential_causes = causes_response.data['results']
    
    # Step 2: Backward chain to verify each cause
    verified_causes = []
    for cause in potential_causes:
        verify_query = f"verify cause: {cause} explains symptoms"
        verify_response = await ure_tool.execute(verify_query, "backward_chain")
        
        if verify_response.data['status'] == 'success':
            verified_causes.append({
                "cause": cause,
                "confidence": len(verify_response.data['results']) / 10.0
            })
    
    # Step 3: Rank causes by confidence
    verified_causes.sort(key=lambda x: x['confidence'], reverse=True)
    
    return {
        "issue": code_issue,
        "symptoms": symptoms,
        "potential_causes": len(potential_causes),
        "verified_causes": verified_causes,
        "recommended_fix": verified_causes[0] if verified_causes else None
    }
```

### Use Case 5: Learning Rule Discovery

```python
async def discover_learning_rules(training_examples: list):
    """Discover learning rules from examples using URE."""
    
    # Create facts from examples
    facts = [f"example_{i}: {ex}" for i, ex in enumerate(training_examples)]
    
    # Use forward chaining to identify patterns
    pattern_query = f"identify patterns in: {', '.join(facts)}"
    patterns = await ure_tool.execute(pattern_query, "forward_chain")
    
    # Generalize patterns into rules
    rules = []
    for pattern in patterns.data['results']:
        rule_query = f"generalize pattern: {pattern} into rule"
        rule_response = await ure_tool.execute(rule_query, "backward_chain")
        
        if rule_response.data['status'] == 'success':
            rules.extend(rule_response.data['results'])
    
    # Create custom rulebase with discovered rules
    await ure_tool.execute(
        query="learned rules",
        operation="create_rulebase",
        rulebase_name="learned_rules",
        rules=rules
    )
    
    return {
        "training_examples": len(training_examples),
        "patterns_found": len(patterns.data['results']),
        "rules_discovered": len(rules),
        "rulebase": "learned_rules"
    }
```

## Best Practices

### 1. Always Check URE Availability

```python
# Check status before intensive operations
status = await ure_tool.execute("", "status")
status_data = json.loads(status.message.split("Data: ")[1])

if not status_data['status']['ure_initialized']:
    print("Warning: URE in fallback mode - results may be limited")
```

### 2. Use Appropriate Chaining Direction

- **Forward Chaining**: When you have facts and want to explore consequences
- **Backward Chaining**: When you have a goal and need to find supporting evidence
- **Hybrid**: For complex problems requiring both exploration and validation

### 3. Optimize Iteration Limits

```python
# For quick queries
simple_query_config = {"max_iterations": 100}

# For complex reasoning
complex_query_config = {"max_iterations": 5000}

# Adjust based on query complexity
response = await ure_tool.execute(
    query,
    operation="forward_chain",
    **query_config
)
```

### 4. Leverage Shared AtomSpace

```python
# Store intermediate results for reuse
await memory_tool.execute(
    operation="store",
    data={"concept": "reasoning_cache", "properties": ure_results}
)

# Retrieve in subsequent reasoning
cached = await memory_tool.execute(
    operation="retrieve",
    data={"concept": "reasoning_cache"}
)
```

### 5. Monitor Performance

```python
import time

start_time = time.time()
response = await ure_tool.execute(query, operation)
elapsed = time.time() - start_time

if elapsed > 5.0:
    print(f"Warning: Reasoning took {elapsed:.2f}s - consider optimization")
```

### 6. Handle Fallback Gracefully

```python
response = await ure_tool.execute(query, "forward_chain")
data = json.loads(response.message.split("Data: ")[1])

if data.get('status') == 'fallback_success':
    print("Using pattern-based reasoning (OpenCog URE not available)")
    # Adjust expectations for fallback results
```

### 7. Use Trace Mode for Debugging

```json
{
  "ure_config": {
    "trace_enabled": true
  }
}
```

Trace mode provides detailed reasoning steps but impacts performance.

### 8. Validate Reasoning Results

```python
# Always validate critical reasoning
if len(response.data['results']) == 0:
    print("Warning: No results from reasoning - check query format")
elif response.data.get('status') == 'error':
    print(f"Error in reasoning: {response.data.get('error')}")
```

## Troubleshooting

### Issue: URE Not Available

**Symptoms**:
- Status shows `ure_initialized: false`
- Operations return fallback results

**Solutions**:
1. Check OpenCog installation: `pip list | grep opencog`
2. Verify configuration: Check `conf/config_cognitive.json`
3. Review initialization logs for errors

### Issue: Slow Reasoning Performance

**Symptoms**:
- Operations take >5 seconds
- High memory usage

**Solutions**:
1. Reduce `max_iterations` in configuration
2. Increase `complexity_penalty`
3. Use more specific queries
4. Clear AtomSpace periodically

### Issue: Unexpected Results

**Symptoms**:
- Results don't match expectations
- Empty result sets

**Solutions**:
1. Enable trace mode to see reasoning steps
2. Verify query format
3. Check available rules: `await ure_tool.execute("", "list_rules")`
4. Validate input atoms in AtomSpace

### Issue: Memory Growth

**Symptoms**:
- Increasing memory usage over time
- System slowdown

**Solutions**:
1. Clear unused atoms from AtomSpace
2. Reduce `max_iterations`
3. Use attention mechanisms to focus on relevant atoms
4. Implement periodic garbage collection

### Issue: Cross-Tool Integration Failures

**Symptoms**:
- Results not shared between tools
- Inconsistent AtomSpace state

**Solutions**:
1. Verify shared AtomSpace configuration
2. Check tool hub initialization
3. Ensure all tools use same AtomSpace instance
4. Review cross-tool integration logs

## Advanced Patterns

### Helper Functions

The following helper functions are used in the patterns above and can be customized for your specific needs:

```python
def calculate_confidence(forward_result, backward_result):
    """Calculate combined confidence from forward and backward reasoning."""
    forward_count = len(forward_result.data.get('results', []))
    backward_count = len(backward_result.data.get('results', []))
    
    # Confidence increases with successful results from both directions
    forward_conf = min(1.0, forward_count / 10.0)
    backward_conf = min(1.0, backward_count / 10.0)
    
    # Weighted average (backward chaining more reliable for proofs)
    return (forward_conf * 0.4 + backward_conf * 0.6)

def calculate_plan_confidence(plan_steps, validation_data):
    """Calculate confidence in a generated plan."""
    if not plan_steps:
        return 0.0
    
    # Base confidence on plan completeness
    plan_confidence = min(1.0, len(plan_steps) / 5.0)
    
    # Adjust based on validation results
    validation_results = validation_data.get('results', [])
    if validation_results:
        validation_confidence = min(1.0, len(validation_results) / 3.0)
        return plan_confidence * 0.6 + validation_confidence * 0.4
    
    return plan_confidence * 0.7

def measure_synergy(individual_results, synthesis):
    """Measure collaboration benefit from multi-agent reasoning."""
    total_individual = sum(len(r.get('insights', [])) for r in individual_results)
    synthesized_count = len(synthesis.data.get('results', []))
    
    if total_individual == 0:
        return 0.0
    
    synergy_ratio = synthesized_count / total_individual
    return min(1.0, synergy_ratio)

def safe_parse_response(response):
    """Safely parse URE tool response data."""
    try:
        if hasattr(response, 'data') and response.data:
            return response.data
        elif "Data: " in str(response.message):
            return json.loads(response.message.split("Data: ")[1])
        else:
            return {"error": "No data available", "results": []}
    except (json.JSONDecodeError, AttributeError, KeyError) as e:
        print(f"Error parsing response: {e}")
        return {"error": str(e), "results": []}
```

### Pattern 14: Attention-Guided URE

```python
from python.tools.meta_cognition import MetaCognitionTool

# Allocate attention to important atoms
await meta_tool.execute(
    operation="attention_focus",
    goals=["critical_reasoning_goal"],
    tasks=["ure_forward_chain"]
)

# URE will prioritize high-attention atoms
response = await ure_tool.execute(query, "forward_chain")
```

### Pattern 15: Incremental Reasoning

```python
# Build up reasoning incrementally
facts = ["fact1", "fact2", "fact3"]
accumulated_knowledge = []

for fact in facts:
    response = await ure_tool.execute(
        f"given {accumulated_knowledge} and {fact}",
        "forward_chain"
    )
    accumulated_knowledge.extend(response.data['results'])

print(f"Final knowledge base: {len(accumulated_knowledge)} facts")
```

### Pattern 16: Probabilistic Logic Integration

```python
# Combine URE with PLN for probabilistic reasoning
from python.tools.cognitive_reasoning import CognitiveReasoningTool

# Use URE for logical structure
logical_structure = await ure_tool.execute(query, "backward_chain")

# Use PLN for uncertainty handling
probabilistic_result = await cognitive_tool.execute(
    f"evaluate probability of: {logical_structure.data['results']}",
    operation="pln_reasoning"
)

# Combine for robust reasoning
final_result = {
    "logical_proof": logical_structure.data['results'],
    "probability": probabilistic_result.data['truth_value'],
    "confidence": probabilistic_result.data['confidence']
}
```

## Conclusion

Logic systems integration in PyCog-Zero provides powerful reasoning capabilities for Agent-Zero. By following these usage patterns, you can:

- Implement sophisticated logical reasoning
- Build goal-directed intelligent agents
- Create adaptive problem-solving systems
- Integrate multiple reasoning strategies

For more information:
- See `demo_ure_integration.py` for working examples
- Check `python/tools/ure_tool.py` for implementation details
- Review `conf/config_cognitive.json` for configuration options
- Read Phase 2 implementation guide: `docs/phase2_logic_systems_implementation.md`

## References

- OpenCog URE Documentation: https://wiki.opencog.org/w/URE
- Agent-Zero Framework: https://github.com/agent0ai/agent-zero
- PyCog-Zero Genesis Roadmap: `AGENT-ZERO-GENESIS.md`
- Implementation Summary: `IMPLEMENTATION_SUMMARY.md`
