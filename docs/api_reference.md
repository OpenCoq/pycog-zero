# PyCog-Zero API Reference Documentation

Complete API reference for all cognitive tools and components in the PyCog-Zero system.

## 📋 Table of Contents

### Core Cognitive Tools
- [CognitiveReasoningTool](#cognitivereasoningtool)
- [CognitiveMemoryTool](#cognitivememorytool)  
- [MetaCognitionTool](#metacognitiontool)

### Advanced Cognitive Tools
- [NeuralSymbolicAgent](#neuralsymbolicagent)
- [SelfModifyingArchitecture](#selfmodifyingarchitecture)
- [DistributedAgentNetwork](#distributedagentnetwork)

### Memory and Storage Tools
- [AtomSpaceMemoryBridge](#atomspacememorybridge)
- [CognitiveMemory](#cognitivememory)
- [AtomSpaceRocksOptimizer](#atomspacerocksoptimizer)

### Reasoning and Logic Tools
- [UREReasoningTool](#urereasoningtool)
- [AdvancedPatternRecognition](#advancedpatternrecognition)
- [ConceptLearning](#conceptlearning)

### Attention and Performance Tools
- [AttentionAllocation](#attentionallocation)
- [PerformanceMonitor](#performancemonitor)
- [CognitiveLearning](#cognitivelearning)

### Utility and Integration Tools
- [AtomSpaceToolHub](#atomspacetoolhub)
- [AtomSpaceSearchEngine](#atomspacesearchengine)
- [MultiAgentCollaboration](#multiagentcollaboration)

---

## Core Cognitive Tools

### CognitiveReasoningTool

Advanced reasoning tool with OpenCog PLN integration for sophisticated cognitive processing.

#### Class Definition

```python
class CognitiveReasoningTool:
    """Advanced cognitive reasoning with PLN integration."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize cognitive reasoning tool.
        
        Args:
            config: Optional configuration dictionary containing:
                - pln_enabled: bool - Enable PLN reasoning (default: True)
                - max_reasoning_steps: int - Maximum reasoning steps (default: 100)
                - attention_allocation: bool - Enable attention allocation (default: True)
                - reasoning_timeout: int - Timeout in seconds (default: 30)
        """
```

#### Methods

##### execute(args: Dict) -> Dict

```python
async def execute(self, args: Dict) -> Dict:
    """Execute cognitive reasoning operation.
    
    Args:
        args: Dictionary containing:
            - query: str - The reasoning query to process
            - reasoning_mode: str - Mode of reasoning:
                * "logical" - Logical inference and deduction
                * "probabilistic" - Probabilistic reasoning with uncertainty
                * "analogical" - Analogical reasoning across domains
                * "causal" - Causal reasoning and analysis
                * "meta" - Meta-reasoning about reasoning processes
            - context: Optional[str] - Context for reasoning
            - evidence: Optional[List[str]] - Evidence to consider
            - max_steps: Optional[int] - Maximum reasoning steps
            - confidence_threshold: Optional[float] - Minimum confidence (0.0-1.0)
    
    Returns:
        Dict containing:
            - reasoning_result: str - Main reasoning conclusion
            - confidence: float - Confidence level (0.0-1.0)
            - reasoning_steps: List[Dict] - Step-by-step reasoning process
            - evidence_used: List[str] - Evidence incorporated in reasoning
            - alternative_conclusions: List[Dict] - Alternative possibilities
            - reasoning_time: float - Processing time in seconds
    
    Raises:
        ReasoningError: If reasoning process fails
        TimeoutError: If reasoning exceeds timeout
    """
```

##### analyze_reasoning_quality(reasoning_result: Dict) -> Dict

```python
async def analyze_reasoning_quality(self, reasoning_result: Dict) -> Dict:
    """Analyze the quality and reliability of reasoning results.
    
    Args:
        reasoning_result: Result from execute() method
        
    Returns:
        Dict containing quality metrics:
            - logical_consistency: float - Consistency score (0.0-1.0)
            - evidence_strength: float - Evidence support strength
            - conclusion_reliability: float - Reliability assessment
            - potential_biases: List[str] - Identified potential biases
            - improvement_suggestions: List[str] - Suggestions for improvement
    """
```

#### Usage Examples

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool

# Initialize tool
reasoning_tool = CognitiveReasoningTool({
    "pln_enabled": True,
    "max_reasoning_steps": 50,
    "reasoning_timeout": 60
})

# Logical reasoning
logical_result = await reasoning_tool.execute({
    "query": "If AI systems become more capable, what are the implications?",
    "reasoning_mode": "logical",
    "context": "ai_safety",
    "max_steps": 20
})

# Probabilistic reasoning
probabilistic_result = await reasoning_tool.execute({
    "query": "What is the likelihood of AGI by 2030?",
    "reasoning_mode": "probabilistic",
    "evidence": ["current_progress", "expert_opinions", "technical_challenges"],
    "confidence_threshold": 0.7
})

# Quality analysis
quality_analysis = await reasoning_tool.analyze_reasoning_quality(logical_result)
```

---

### CognitiveMemoryTool

Persistent cognitive memory system with AtomSpace integration for knowledge storage and retrieval.

#### Class Definition

```python
class CognitiveMemoryTool:
    """Persistent cognitive memory with AtomSpace integration."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize cognitive memory tool.
        
        Args:
            config: Optional configuration dictionary containing:
                - persistence_enabled: bool - Enable persistent storage (default: True)
                - atomspace_backend: str - Backend type ("memory", "rocks") (default: "rocks")
                - memory_capacity: int - Maximum memory items (default: 1000000)
                - auto_cleanup: bool - Enable automatic cleanup (default: True)
                - indexing_enabled: bool - Enable knowledge indexing (default: True)
        """
```

#### Methods

##### store_knowledge(knowledge: str, context: str = None, metadata: Dict = None) -> Dict

```python
async def store_knowledge(
    self, 
    knowledge: str, 
    context: str = None, 
    metadata: Dict = None
) -> Dict:
    """Store knowledge in cognitive memory.
    
    Args:
        knowledge: Knowledge content to store
        context: Optional context for knowledge
        metadata: Optional metadata dictionary containing:
            - source: str - Source of knowledge
            - confidence: float - Confidence level (0.0-1.0)
            - timestamp: str - ISO timestamp
            - tags: List[str] - Knowledge tags
            - importance: float - Importance score (0.0-1.0)
    
    Returns:
        Dict containing:
            - knowledge_id: str - Unique identifier for stored knowledge
            - storage_success: bool - Whether storage was successful
            - atomspace_handle: str - AtomSpace handle reference
            - indexed_concepts: List[str] - Extracted and indexed concepts
            - storage_time: float - Storage processing time
    """
```

##### retrieve_knowledge(query: str, context: str = None, filters: Dict = None) -> Dict

```python
async def retrieve_knowledge(
    self, 
    query: str, 
    context: str = None, 
    filters: Dict = None
) -> Dict:
    """Retrieve relevant knowledge from memory.
    
    Args:
        query: Search query for knowledge retrieval
        context: Optional context to narrow search
        filters: Optional filters dictionary containing:
            - min_confidence: float - Minimum confidence threshold
            - tags: List[str] - Required tags
            - source: str - Knowledge source filter
            - time_range: Dict - Time range for knowledge
            - max_results: int - Maximum results to return (default: 10)
    
    Returns:
        Dict containing:
            - retrieved_knowledge: List[Dict] - Retrieved knowledge items
            - relevance_scores: List[float] - Relevance scores for each item
            - total_matches: int - Total number of matches found
            - search_time: float - Search processing time
            - query_analysis: Dict - Analysis of the search query
    """
```

##### update_knowledge(knowledge_id: str, new_content: str, metadata: Dict = None) -> Dict

```python
async def update_knowledge(
    self, 
    knowledge_id: str, 
    new_content: str, 
    metadata: Dict = None
) -> Dict:
    """Update existing knowledge in memory.
    
    Args:
        knowledge_id: Unique identifier of knowledge to update
        new_content: New content to replace or merge
        metadata: Optional metadata updates
    
    Returns:
        Dict containing update status and details
    """
```

##### delete_knowledge(knowledge_id: str) -> Dict

```python
async def delete_knowledge(self, knowledge_id: str) -> Dict:
    """Delete knowledge from memory.
    
    Args:
        knowledge_id: Unique identifier of knowledge to delete
        
    Returns:
        Dict containing deletion status
    """
```

#### Usage Examples

```python
from python.tools.cognitive_memory import CognitiveMemoryTool

# Initialize memory tool
memory_tool = CognitiveMemoryTool({
    "persistence_enabled": True,
    "atomspace_backend": "rocks",
    "indexing_enabled": True
})

# Store knowledge
storage_result = await memory_tool.store_knowledge(
    knowledge="Python is excellent for AI development due to its extensive libraries",
    context="programming",
    metadata={
        "source": "expert_knowledge",
        "confidence": 0.9,
        "tags": ["python", "ai", "programming"],
        "importance": 0.8
    }
)

# Retrieve knowledge
retrieval_result = await memory_tool.retrieve_knowledge(
    query="best programming languages for AI",
    context="programming",
    filters={
        "min_confidence": 0.7,
        "tags": ["programming", "ai"],
        "max_results": 5
    }
)

# Update knowledge
update_result = await memory_tool.update_knowledge(
    knowledge_id=storage_result["knowledge_id"],
    new_content="Python and R are excellent for AI development...",
    metadata={"confidence": 0.95}
)
```

---

### MetaCognitionTool

Meta-cognitive reasoning and self-reflection capabilities for cognitive architecture improvement.

#### Class Definition

```python
class MetaCognitionTool:
    """Meta-cognitive reasoning and self-reflection capabilities."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize meta-cognition tool.
        
        Args:
            config: Optional configuration dictionary containing:
                - reflection_depth: int - Depth of self-reflection (default: 3)
                - learning_enabled: bool - Enable learning from reflection (default: True)
                - performance_tracking: bool - Track performance metrics (default: True)
                - adaptation_threshold: float - Threshold for strategy adaptation (default: 0.6)
        """
```

#### Methods

##### reflect_on_performance(task_history: List[Dict], performance_metrics: Dict = None) -> Dict

```python
async def reflect_on_performance(
    self, 
    task_history: List[Dict], 
    performance_metrics: Dict = None
) -> Dict:
    """Analyze and reflect on past performance.
    
    Args:
        task_history: List of past tasks with outcomes
        performance_metrics: Optional performance metrics to consider
    
    Returns:
        Dict containing:
            - performance_analysis: Dict - Detailed performance analysis
            - identified_patterns: List[Dict] - Patterns in performance
            - strengths: List[str] - Identified strengths
            - weaknesses: List[str] - Areas for improvement
            - improvement_recommendations: List[Dict] - Specific recommendations
            - confidence_in_analysis: float - Confidence in analysis (0.0-1.0)
    """
```

##### adapt_strategy(current_strategy: str, feedback: Dict, context: str = None) -> Dict

```python
async def adapt_strategy(
    self, 
    current_strategy: str, 
    feedback: Dict, 
    context: str = None
) -> Dict:
    """Adapt reasoning strategy based on feedback.
    
    Args:
        current_strategy: Current strategy description
        feedback: Feedback on strategy effectiveness
        context: Optional context for adaptation
    
    Returns:
        Dict containing:
            - adapted_strategy: str - New adapted strategy
            - adaptation_rationale: str - Explanation for changes
            - expected_improvements: List[str] - Expected benefits
            - implementation_steps: List[str] - Steps to implement
            - confidence_in_adaptation: float - Confidence in new strategy
    """
```

##### self_evaluate_capabilities(evaluation_criteria: List[str] = None) -> Dict

```python
async def self_evaluate_capabilities(
    self, 
    evaluation_criteria: List[str] = None
) -> Dict:
    """Evaluate own cognitive capabilities.
    
    Args:
        evaluation_criteria: Optional specific criteria to evaluate
    
    Returns:
        Dict containing capability evaluation results
    """
```

#### Usage Examples

```python
from python.tools.meta_cognition import MetaCognitionTool

# Initialize meta-cognition tool
meta_tool = MetaCognitionTool({
    "reflection_depth": 4,
    "learning_enabled": True,
    "performance_tracking": True
})

# Reflect on performance
task_history = [
    {"task": "reasoning", "outcome": "success", "time": 2.5, "confidence": 0.8},
    {"task": "memory_retrieval", "outcome": "partial", "time": 1.2, "confidence": 0.6}
]

reflection_result = await meta_tool.reflect_on_performance(
    task_history=task_history,
    performance_metrics={"average_time": 1.85, "success_rate": 0.75}
)

# Adapt strategy
adaptation_result = await meta_tool.adapt_strategy(
    current_strategy="sequential_processing",
    feedback={"effectiveness": 0.6, "efficiency": 0.4, "accuracy": 0.8},
    context="time_critical_tasks"
)

# Self-evaluate capabilities
capability_evaluation = await meta_tool.self_evaluate_capabilities([
    "reasoning_accuracy",
    "processing_speed",
    "knowledge_integration",
    "adaptation_ability"
])
```

---

## Advanced Cognitive Tools

### NeuralSymbolicAgent

Advanced neural-symbolic integration tool bridging neural networks and symbolic reasoning.

#### Class Definition

```python
class NeuralSymbolicAgent:
    """Neural-symbolic bridge for integrated AI processing."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize neural-symbolic agent.
        
        Args:
            config: Configuration containing neural and symbolic settings
        """
```

#### Methods

##### process_multimodal(input_data: Dict) -> Dict

```python
async def process_multimodal(self, input_data: Dict) -> Dict:
    """Process multi-modal input using neural-symbolic integration.
    
    Args:
        input_data: Dictionary containing:
            - text: Optional[str] - Text input
            - image: Optional[str] - Image path or data
            - audio: Optional[str] - Audio path or data
            - context: Optional[str] - Processing context
            - reasoning_type: str - Type of reasoning to apply
    
    Returns:
        Dict containing integrated processing results
    """
```

#### Usage Examples

```python
from python.tools.neural_symbolic_agent import NeuralSymbolicAgent

# Initialize neural-symbolic agent
ns_agent = NeuralSymbolicAgent({
    "neural_backend": "pytorch",
    "symbolic_backend": "opencog",
    "integration_mode": "tight_coupling"
})

# Process multi-modal input
result = await ns_agent.process_multimodal({
    "text": "Analyze this image and explain the relationships",
    "image": "/path/to/image.jpg",
    "context": "visual_reasoning",
    "reasoning_type": "analogical"
})
```

---

### SelfModifyingArchitecture

Self-modifying architecture capabilities for dynamic system adaptation.

#### Class Definition

```python
class SelfModifyingArchitecture:
    """Self-modifying architecture for dynamic adaptation."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize self-modifying architecture tool."""
```

#### Methods

##### modify_architecture(modification_plan: Dict) -> Dict

```python
async def modify_architecture(self, modification_plan: Dict) -> Dict:
    """Modify system architecture based on plan.
    
    Args:
        modification_plan: Plan for architecture modifications
    
    Returns:
        Dict containing modification results and status
    """
```

##### evaluate_modification_impact(proposed_changes: Dict) -> Dict

```python
async def evaluate_modification_impact(self, proposed_changes: Dict) -> Dict:
    """Evaluate potential impact of architecture modifications.
    
    Args:
        proposed_changes: Proposed architectural changes
    
    Returns:
        Dict containing impact analysis
    """
```

#### Usage Examples

```python
from python.tools.self_modifying_architecture import SelfModifyingArchitecture

# Initialize self-modifying architecture
sma_tool = SelfModifyingArchitecture({
    "safety_checks": True,
    "rollback_enabled": True,
    "impact_analysis": True
})

# Evaluate modification impact
impact_analysis = await sma_tool.evaluate_modification_impact({
    "add_tool": "advanced_pattern_matcher",
    "modify_reasoning": "increase_depth",
    "optimize_memory": "enable_compression"
})

# Apply modifications if safe
if impact_analysis["safety_score"] > 0.8:
    modification_result = await sma_tool.modify_architecture({
        "changes": impact_analysis["recommended_changes"],
        "safety_mode": True
    })
```

---

## Memory and Storage Tools

### AtomSpaceMemoryBridge

Bridge between Agent-Zero memory and OpenCog AtomSpace for persistent cognitive storage.

#### Class Definition

```python
class AtomSpaceMemoryBridge:
    """Bridge between Agent-Zero and AtomSpace memory systems."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize AtomSpace memory bridge."""
```

#### Methods

##### sync_agent_memory(agent_memory: Dict) -> Dict

```python
async def sync_agent_memory(self, agent_memory: Dict) -> Dict:
    """Synchronize Agent-Zero memory with AtomSpace.
    
    Args:
        agent_memory: Agent-Zero memory content
    
    Returns:
        Dict containing synchronization results
    """
```

##### create_knowledge_graph(knowledge_items: List[Dict]) -> Dict

```python
async def create_knowledge_graph(self, knowledge_items: List[Dict]) -> Dict:
    """Create knowledge graph from structured knowledge.
    
    Args:
        knowledge_items: List of knowledge items to graph
    
    Returns:
        Dict containing knowledge graph creation results
    """
```

---

## Reasoning and Logic Tools

### UREReasoningTool

Unified Rule Engine integration for forward and backward chaining inference.

#### Class Definition

```python
class UREReasoningTool:
    """URE-based reasoning with forward and backward chaining."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize URE reasoning tool."""
```

#### Methods

##### forward_chain(premises: List[str], rules: List[str] = None) -> Dict

```python
async def forward_chain(
    self, 
    premises: List[str], 
    rules: List[str] = None
) -> Dict:
    """Perform forward chaining inference.
    
    Args:
        premises: Starting premises for inference
        rules: Optional specific rules to use
    
    Returns:
        Dict containing forward chaining results
    """
```

##### backward_chain(goal: str, knowledge_base: List[str] = None) -> Dict

```python
async def backward_chain(
    self, 
    goal: str, 
    knowledge_base: List[str] = None
) -> Dict:
    """Perform backward chaining inference.
    
    Args:
        goal: Goal to prove or derive
        knowledge_base: Optional knowledge base to use
    
    Returns:
        Dict containing backward chaining results
    """
```

#### Usage Examples

```python
from python.tools.ure_tool import UREReasoningTool

# Initialize URE reasoning tool
ure_tool = UREReasoningTool({
    "max_iterations": 100,
    "confidence_threshold": 0.8,
    "rule_selection": "adaptive"
})

# Forward chaining
forward_result = await ure_tool.forward_chain(
    premises=[
        "All humans are mortal",
        "Socrates is human"
    ],
    rules=["modus_ponens", "universal_instantiation"]
)

# Backward chaining
backward_result = await ure_tool.backward_chain(
    goal="Socrates is mortal",
    knowledge_base=[
        "All humans are mortal",
        "Socrates is human"
    ]
)
```

---

## Attention and Performance Tools

### AttentionAllocation

Economic Cognitive Attention Networks (ECAN) integration for attention management.

#### Class Definition

```python
class AttentionAllocation:
    """ECAN-based attention allocation and management."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize attention allocation tool."""
```

#### Methods

##### allocate_attention(tasks: List[Dict], constraints: Dict = None) -> Dict

```python
async def allocate_attention(
    self, 
    tasks: List[Dict], 
    constraints: Dict = None
) -> Dict:
    """Allocate attention across multiple tasks.
    
    Args:
        tasks: List of tasks requiring attention
        constraints: Optional attention constraints
    
    Returns:
        Dict containing attention allocation results
    """
```

##### update_attention_values(feedback: Dict) -> Dict

```python
async def update_attention_values(self, feedback: Dict) -> Dict:
    """Update attention values based on feedback.
    
    Args:
        feedback: Performance feedback for attention updates
    
    Returns:
        Dict containing updated attention values
    """
```

#### Usage Examples

```python
from python.tools.attention_allocation import AttentionAllocation

# Initialize attention allocation
attention_tool = AttentionAllocation({
    "attention_budget": 100,
    "allocation_strategy": "economic",
    "update_frequency": "adaptive"
})

# Allocate attention
allocation_result = await attention_tool.allocate_attention([
    {"task": "reasoning", "priority": 0.8, "estimated_cost": 30},
    {"task": "memory_search", "priority": 0.6, "estimated_cost": 15},
    {"task": "pattern_matching", "priority": 0.7, "estimated_cost": 20}
])

# Update based on performance
update_result = await attention_tool.update_attention_values({
    "task_performance": {"reasoning": 0.9, "memory_search": 0.7},
    "efficiency_scores": {"reasoning": 0.8, "memory_search": 0.9}
})
```

---

## Utility and Integration Tools

### AtomSpaceToolHub

Central hub for AtomSpace-integrated tools and utilities.

#### Class Definition

```python
class AtomSpaceToolHub:
    """Central hub for AtomSpace-integrated cognitive tools."""
    
    def __init__(self, config: Optional[Dict] = None):
        """Initialize AtomSpace tool hub."""
```

#### Methods

##### register_tool(tool_name: str, tool_class: type, config: Dict = None) -> Dict

```python
async def register_tool(
    self, 
    tool_name: str, 
    tool_class: type, 
    config: Dict = None
) -> Dict:
    """Register a new cognitive tool with AtomSpace integration.
    
    Args:
        tool_name: Name for the tool
        tool_class: Tool class to register
        config: Optional tool configuration
    
    Returns:
        Dict containing registration status
    """
```

##### execute_tool_chain(tool_chain: List[Dict]) -> Dict

```python
async def execute_tool_chain(self, tool_chain: List[Dict]) -> Dict:
    """Execute a chain of cognitive tools in sequence.
    
    Args:
        tool_chain: List of tools and parameters to execute
    
    Returns:
        Dict containing chain execution results
    """
```

---

## Error Handling and Exceptions

### Common Exceptions

```python
class CognitiveToolError(Exception):
    """Base exception for cognitive tool errors."""
    pass

class ReasoningError(CognitiveToolError):
    """Exception raised during reasoning operations."""
    pass

class MemoryError(CognitiveToolError):
    """Exception raised during memory operations."""
    pass

class AttentionAllocationError(CognitiveToolError):
    """Exception raised during attention allocation."""
    pass

class IntegrationError(CognitiveToolError):
    """Exception raised during system integration."""
    pass
```

### Error Handling Patterns

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool, ReasoningError

try:
    reasoning_tool = CognitiveReasoningTool()
    result = await reasoning_tool.execute({
        "query": "Complex reasoning query",
        "reasoning_mode": "probabilistic"
    })
except ReasoningError as e:
    print(f"Reasoning failed: {e}")
    # Handle specific reasoning error
except CognitiveToolError as e:
    print(f"General cognitive tool error: {e}")
    # Handle general tool error
except Exception as e:
    print(f"Unexpected error: {e}")
    # Handle unexpected errors
```

---

## Configuration Reference

### Global Configuration Structure

```python
{
    "cognitive_mode": True,
    "opencog_enabled": True,
    "performance_monitoring": True,
    "distributed_agents": False,
    
    "atomspace": {
        "backend": "rocks",
        "persistence_enabled": True,
        "memory_limit": "1GB",
        "auto_cleanup": True
    },
    
    "reasoning": {
        "pln_enabled": True,
        "ure_enabled": True,
        "max_reasoning_steps": 100,
        "reasoning_timeout": 30,
        "confidence_threshold": 0.7
    },
    
    "attention": {
        "ecan_enabled": True,
        "attention_budget": 100,
        "allocation_strategy": "economic",
        "update_frequency": "adaptive"
    },
    
    "memory": {
        "indexing_enabled": True,
        "auto_cleanup": True,
        "capacity_limit": 1000000,
        "persistence_backend": "atomspace_rocks"
    },
    
    "performance": {
        "monitoring_enabled": True,
        "benchmarking": True,
        "optimization_enabled": True,
        "target_response_time": 2.0
    }
}
```

### Tool-Specific Configurations

Each tool supports specific configuration options as documented in their respective sections above.

---

## Version Information

- **API Version**: 1.0.0
- **PyCog-Zero Version**: Genesis Phase 5
- **OpenCog Integration**: Complete
- **Agent-Zero Compatibility**: Full
- **Last Updated**: October 2024

---

## See Also

- [Comprehensive Integration Documentation](./COMPREHENSIVE_INTEGRATION_DOCUMENTATION.md)
- [Usage Examples](./usage_examples.md)
- [Performance Benchmarking](./performance_benchmarking.md)
- [Troubleshooting Guide](./troubleshooting.md)
- [Contributing Guidelines](./contribution.md)

---

*This API reference provides complete coverage of all cognitive tools and components in the PyCog-Zero system. For additional examples and usage patterns, refer to the comprehensive documentation and example code in the repository.*