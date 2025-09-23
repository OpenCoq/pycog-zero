# Phase 2 Logic Systems Implementation Guide

## Overview

Phase 2 of the PyCog-Zero development roadmap focuses on integrating OpenCog logic systems (unify and URE) with the Agent-Zero framework. This guide provides step-by-step implementation instructions and integration patterns.

## Phase 2 Components

### 1. Unify System
- **Repository**: https://github.com/opencog/unify  
- **Purpose**: Pattern unification and term matching
- **Priority**: HIGH
- **Dependencies**: atomspace

### 2. URE (Unified Rule Engine)
- **Repository**: https://github.com/opencog/ure
- **Purpose**: Forward and backward chaining inference
- **Priority**: HIGH  
- **Dependencies**: atomspace, unify

## Implementation Steps

### Step 1: Clone and Setup Components

```bash
# Navigate to PyCog-Zero directory
cd /path/to/pycog-zero

# Clone unify component
python3 scripts/cpp2py_conversion_pipeline.py clone unify

# Clone URE component  
python3 scripts/cpp2py_conversion_pipeline.py clone ure

# Validate dependencies
python3 scripts/cpp2py_conversion_pipeline.py validate unify
python3 scripts/cpp2py_conversion_pipeline.py validate ure
```

### Step 2: Component Analysis

#### Unify System Analysis
- **Core Files**: `opencog/unify/Unifier.h`, `Unifier.cc`
- **Key Classes**: `Unifier`, `UnifierLink`, `UnifyReduceLink`
- **Python Bindings**: Located in `opencog/cython/opencog/`
- **Examples**: Available in `examples/` directory

#### URE System Analysis  
- **Core Files**: `opencog/ure/forwardchainer/`, `opencog/ure/backwardchainer/`
- **Key Classes**: `ForwardChainer`, `BackwardChainer`, `Rule`
- **Python Bindings**: Scheme integration via `URESCM.cc`
- **Examples**: Rule execution examples in `examples/ure/`

### Step 3: Python Integration Architecture

```python
# Integration architecture for Phase 2 logic systems
from opencog.atomspace import AtomSpace, types
from opencog.unify import Unifier  # To be implemented
from opencog.ure import ForwardChainer, BackwardChainer  # To be implemented

class Phase2LogicIntegration:
    """Integration manager for Phase 2 logic systems."""
    
    def __init__(self):
        self.atomspace = AtomSpace()
        self.unifier = None
        self.forward_chainer = None
        self.backward_chainer = None
        self._initialize_components()
    
    def _initialize_components(self):
        """Initialize logic system components."""
        try:
            # Initialize unification system
            self.unifier = Unifier(self.atomspace)
            
            # Initialize rule engine
            self.forward_chainer = ForwardChainer(self.atomspace)
            self.backward_chainer = BackwardChainer(self.atomspace)
            
            print("✓ Phase 2 logic systems initialized successfully")
            
        except ImportError as e:
            print(f"⚠ Logic systems not yet available: {e}")
            print("  Run Python binding generation to enable full functionality")
    
    def test_integration(self):
        """Test integration functionality."""
        # Test unification
        if self.unifier:
            print("✓ Unification system ready")
        
        # Test rule engines
        if self.forward_chainer and self.backward_chainer:
            print("✓ Rule engines ready")
        
        return bool(self.unifier and self.forward_chainer and self.backward_chainer)
```

### Step 4: Agent-Zero Tool Creation

#### Create Unification Tool

```bash
# Create unification tool for Agent-Zero
touch python/tools/unification_tool.py
```

#### Create URE Tool

```bash
# Create URE reasoning tool for Agent-Zero
touch python/tools/ure_reasoning_tool.py
```

### Step 5: Python Binding Generation

#### Unify Python Bindings

```python
# python/tools/unification_tool.py
"""
Agent-Zero tool for OpenCog unification system.
Implements pattern unification for cognitive reasoning.
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
from typing import Dict, Any, List, Optional

# Try to import unification components
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.unify import Unifier  # Will be available after binding generation
    UNIFY_AVAILABLE = True
except ImportError:
    print("Unify system not available - run Python binding generation")
    UNIFY_AVAILABLE = False

class UnificationTool(Tool):
    """Agent-Zero tool for pattern unification using OpenCog unify system."""
    
    def __init__(self, agent):
        super().__init__(agent)
        self.atomspace = AtomSpace() if UNIFY_AVAILABLE else None
        self.unifier = Unifier(self.atomspace) if UNIFY_AVAILABLE else None
        self.initialized = UNIFY_AVAILABLE
    
    async def execute(self, pattern_a: str, pattern_b: str, **kwargs):
        """Execute pattern unification."""
        
        if not self.initialized:
            return Response(
                message="Unification system not available",
                data={
                    "error": "OpenCog unify system not installed",
                    "suggestion": "Run Python binding generation for unify component"
                }
            )
        
        try:
            # Convert patterns to atoms
            atom_a = self._pattern_to_atom(pattern_a)
            atom_b = self._pattern_to_atom(pattern_b)
            
            # Perform unification
            result = self.unifier.unify(atom_a, atom_b)
            
            return Response(
                message=f"Unification completed: {len(result) if result else 0} results",
                data={
                    "pattern_a": pattern_a,
                    "pattern_b": pattern_b,
                    "unification_results": self._results_to_dict(result),
                    "success": bool(result)
                }
            )
            
        except Exception as e:
            return Response(
                message=f"Unification failed: {str(e)}",
                data={"error": str(e), "success": False}
            )
    
    def _pattern_to_atom(self, pattern: str):
        """Convert pattern string to AtomSpace atom."""
        # Implementation depends on pattern format
        # This is a placeholder for actual conversion logic
        return self.atomspace.add_node(types.ConceptNode, pattern)
    
    def _results_to_dict(self, results) -> list:
        """Convert unification results to dictionary format."""
        if not results:
            return []
        
        return [{"result": str(result), "type": "unification"} for result in results]
```

#### URE Python Bindings

```python  
# python/tools/ure_reasoning_tool.py
"""
Agent-Zero tool for OpenCog Unified Rule Engine (URE).
Implements forward and backward chaining for cognitive reasoning.
"""

from python.helpers.tool import Tool, Response
from python.helpers import files
import json
from typing import Dict, Any, List, Optional

# Try to import URE components
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.ure import ForwardChainer, BackwardChainer  # Will be available after binding generation
    URE_AVAILABLE = True
except ImportError:
    print("URE system not available - run Python binding generation")  
    URE_AVAILABLE = False

class UREReasoningTool(Tool):
    """Agent-Zero tool for rule-based reasoning using OpenCog URE."""
    
    def __init__(self, agent):
        super().__init__(agent)
        self.atomspace = AtomSpace() if URE_AVAILABLE else None
        self.forward_chainer = ForwardChainer(self.atomspace) if URE_AVAILABLE else None
        self.backward_chainer = BackwardChainer(self.atomspace) if URE_AVAILABLE else None
        self.initialized = URE_AVAILABLE
    
    async def execute(self, premises: List[str], operation: str = "forward", **kwargs):
        """Execute rule-based reasoning."""
        
        if not self.initialized:
            return Response(
                message="URE system not available",
                data={
                    "error": "OpenCog URE system not installed",
                    "suggestion": "Run Python binding generation for URE component"
                }
            )
        
        try:
            # Convert premises to atoms
            premise_atoms = [self._premise_to_atom(premise) for premise in premises]
            
            if operation == "forward":
                # Forward chaining
                results = self.forward_chainer.do_chain(premise_atoms)
                message = f"Forward chaining completed: {len(results)} inferences"
                
            elif operation == "backward":
                # Backward chaining (premises treated as goals)
                results = self.backward_chainer.do_chain(premise_atoms) 
                message = f"Backward chaining completed: {len(results)} proofs"
                
            else:
                raise ValueError(f"Unknown operation: {operation}")
            
            return Response(
                message=message,
                data={
                    "operation": operation,
                    "premises": premises,
                    "reasoning_results": self._results_to_dict(results),
                    "inference_count": len(results) if results else 0,
                    "success": bool(results)
                }
            )
            
        except Exception as e:
            return Response(
                message=f"URE reasoning failed: {str(e)}",
                data={"error": str(e), "success": False}
            )
    
    def _premise_to_atom(self, premise: str):
        """Convert premise string to AtomSpace atom."""
        # Implementation depends on premise format
        # This is a placeholder for actual conversion logic
        return self.atomspace.add_node(types.ConceptNode, premise)
    
    def _results_to_dict(self, results) -> list:
        """Convert reasoning results to dictionary format."""
        if not results:
            return []
        
        return [
            {
                "inference": str(result),
                "type": "reasoning_result",
                "confidence": getattr(result, 'confidence', 0.8)
            }
            for result in results
        ]
```

### Step 6: Integration Testing

#### Create Integration Tests

```python
# tests/integration/test_phase2_logic_systems.py
"""
Integration tests for Phase 2 logic systems (unify and URE).
"""

import pytest
import asyncio
from unittest.mock import MagicMock

# Import Phase 2 tools
try:
    from python.tools.unification_tool import UnificationTool
    from python.tools.ure_reasoning_tool import UREReasoningTool
    TOOLS_AVAILABLE = True
except ImportError:
    TOOLS_AVAILABLE = False

class TestPhase2LogicSystems:
    """Test Phase 2 logic systems integration."""
    
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.config = {"cognitive_mode": True, "phase2_logic": True}
        return agent
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Phase 2 tools not yet available")
    def test_unification_tool_creation(self, mock_agent):
        """Test that unification tool can be created."""
        tool = UnificationTool(mock_agent)
        assert tool is not None
        assert hasattr(tool, 'execute')
    
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Phase 2 tools not yet available") 
    def test_ure_tool_creation(self, mock_agent):
        """Test that URE tool can be created."""
        tool = UREReasoningTool(mock_agent)
        assert tool is not None
        assert hasattr(tool, 'execute')
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Phase 2 tools not yet available")
    async def test_unification_execution(self, mock_agent):
        """Test unification tool execution."""
        tool = UnificationTool(mock_agent)
        
        result = await tool.execute("pattern A", "pattern B")
        assert result is not None
        assert hasattr(result, 'data')
        assert 'pattern_a' in result.data
        assert 'pattern_b' in result.data
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Phase 2 tools not yet available")
    async def test_ure_forward_chaining(self, mock_agent):
        """Test URE forward chaining."""
        tool = UREReasoningTool(mock_agent)
        
        premises = ["fact A", "fact B"]
        result = await tool.execute(premises, operation="forward")
        assert result is not None
        assert result.data['operation'] == "forward"
        assert 'reasoning_results' in result.data
    
    @pytest.mark.asyncio
    @pytest.mark.skipif(not TOOLS_AVAILABLE, reason="Phase 2 tools not yet available")
    async def test_ure_backward_chaining(self, mock_agent):
        """Test URE backward chaining.""" 
        tool = UREReasoningTool(mock_agent)
        
        goals = ["goal A"]
        result = await tool.execute(goals, operation="backward")
        assert result is not None
        assert result.data['operation'] == "backward"
        assert 'reasoning_results' in result.data

class TestPhase2Integration:
    """Test integration between Phase 2 components."""
    
    def test_component_cloning_status(self):
        """Test that Phase 2 components can be cloned."""
        import os
        
        # Check if components directory exists
        components_dir = "components"
        assert os.path.exists(components_dir), "Components directory should exist"
        
        # Check for unify component
        unify_dir = os.path.join(components_dir, "unify")
        if os.path.exists(unify_dir):
            assert os.path.exists(os.path.join(unify_dir, "README.md"))
            print("✓ Unify component cloned successfully")
        
        # Check for URE component
        ure_dir = os.path.join(components_dir, "ure")
        if os.path.exists(ure_dir):
            assert os.path.exists(os.path.join(ure_dir, "README.md"))
            print("✓ URE component cloned successfully")
    
    def test_dependency_validation(self):
        """Test dependency validation for Phase 2 components."""
        # This test validates that atomspace (Phase 1) is available for Phase 2
        atomspace_dir = "components/atomspace"
        if os.path.exists(atomspace_dir):
            print("✓ AtomSpace dependency available for Phase 2")
        else:
            print("⚠ AtomSpace dependency not found - may affect Phase 2 integration")
    
    def test_pipeline_configuration(self):
        """Test that pipeline recognizes Phase 2 components."""
        try:
            from scripts.cpp2py_conversion_pipeline import CPP2PyConversionPipeline
            
            pipeline = CPP2PyConversionPipeline()
            components = pipeline._load_component_definitions()
            
            # Check Phase 2 components are defined
            assert "unify" in components, "Unify component should be defined"
            assert "ure" in components, "URE component should be defined"
            
            # Check phase assignments
            assert components["unify"].phase.value == "phase_2_logic_systems"
            assert components["ure"].phase.value == "phase_2_logic_systems"
            
            print("✓ Phase 2 components properly configured in pipeline")
            
        except ImportError:
            print("⚠ Pipeline not available for testing")

if __name__ == "__main__":
    # Run tests directly if executed as script
    pytest.main([__file__, "-v"])
```

### Step 7: Documentation Update

#### Update AGENT-ZERO-GENESIS.md

Add Phase 2 specific usage examples to the existing file:

```python
# Addition to AGENT-ZERO-GENESIS.md Usage Examples section

### 4. Phase 2 Logic Systems Integration

```python
# Phase 2: Unification and Rule-based Reasoning with Agent-Zero
from opencog.atomspace import AtomSpace, types
from opencog.unify import Unifier
from opencog.ure import ForwardChainer, BackwardChainer

class Phase2CognitiveAgent:
    def __init__(self):
        self.atomspace = AtomSpace()
        self.agent_zero = Agent()
        self.unifier = Unifier(self.atomspace)
        self.forward_chainer = ForwardChainer(self.atomspace)
        self.backward_chainer = BackwardChainer(self.atomspace)
        
        # Setup Phase 2 capabilities
        self.setup_logic_systems()
    
    def setup_logic_systems(self):
        """Setup Phase 2 logic system integration."""
        
        # Add inference rules for Agent-Zero reasoning
        self.setup_agent_reasoning_rules()
        
        # Configure unification patterns
        self.setup_unification_patterns()
    
    def setup_agent_reasoning_rules(self):
        """Setup inference rules for Agent-Zero task reasoning."""
        
        # Rule: Task capability inference
        # If agent has skill X and skill X enables capability Y, then agent can do Y
        capability_rule = self.atomspace.add_link(
            types.RuleLink,
            [
                self.atomspace.add_link(
                    types.VariableList,
                    [
                        self.atomspace.add_node(types.VariableNode, "$agent"),
                        self.atomspace.add_node(types.VariableNode, "$skill"), 
                        self.atomspace.add_node(types.VariableNode, "$capability")
                    ]
                ),
                self.atomspace.add_link(
                    types.AndLink,
                    [
                        self.atomspace.add_link(
                            types.EvaluationLink,
                            [
                                self.atomspace.add_node(types.PredicateNode, "has_skill"),
                                self.atomspace.add_link(
                                    types.ListLink,
                                    [
                                        self.atomspace.add_node(types.VariableNode, "$agent"),
                                        self.atomspace.add_node(types.VariableNode, "$skill")
                                    ]
                                )
                            ]
                        ),
                        self.atomspace.add_link(
                            types.EvaluationLink,
                            [
                                self.atomspace.add_node(types.PredicateNode, "enables"),
                                self.atomspace.add_link(
                                    types.ListLink,
                                    [
                                        self.atomspace.add_node(types.VariableNode, "$skill"),
                                        self.atomspace.add_node(types.VariableNode, "$capability")
                                    ]
                                )
                            ]
                        )
                    ]
                ),
                self.atomspace.add_link(
                    types.EvaluationLink,
                    [
                        self.atomspace.add_node(types.PredicateNode, "can_do"),
                        self.atomspace.add_link(
                            types.ListLink,
                            [
                                self.atomspace.add_node(types.VariableNode, "$agent"),
                                self.atomspace.add_node(types.VariableNode, "$capability")
                            ]
                        )
                    ]
                )
            ]
        )
        
        # Add rule to forward chainer
        self.forward_chainer.add_rule(capability_rule)
    
    def setup_unification_patterns(self):
        """Setup unification patterns for Agent-Zero query matching."""
        
        # Pattern for task-goal unification
        task_pattern = self.atomspace.add_node(types.ConceptNode, "task_template")
        goal_pattern = self.atomspace.add_node(types.ConceptNode, "goal_template")
        
        # Create unification template
        self.task_goal_unifier = self.atomspace.add_link(
            types.UnifierLink,
            [
                self.atomspace.add_link(
                    types.LambdaLink,
                    [
                        self.atomspace.add_node(types.VariableNode, "$task"),
                        task_pattern
                    ]
                ),
                self.atomspace.add_link(
                    types.LambdaLink,
                    [
                        self.atomspace.add_node(types.VariableNode, "$goal"),
                        goal_pattern  
                    ]
                ),
                self.atomspace.add_node(types.ConceptNode, "unified_task_goal")
            ]
        )
    
    async def reason_about_task(self, task_description: str):
        """Use Phase 2 logic systems for task reasoning."""
        
        # Step 1: Parse task using Agent-Zero NLP
        parsed_task = await self.agent_zero.process_natural_language(task_description)
        
        # Step 2: Convert to AtomSpace representation
        task_atom = self.atomspace.add_node(types.ConceptNode, f"task_{parsed_task['intent']}")
        
        # Step 3: Use unification to match with known patterns
        unification_results = self.unifier.unify(task_atom, self.task_goal_unifier)
        
        # Step 4: Apply forward chaining to infer capabilities
        inference_results = self.forward_chainer.do_chain([task_atom])
        
        # Step 5: Use backward chaining for goal achievement
        goal_atom = self.atomspace.add_node(types.ConceptNode, f"achieve_{parsed_task['intent']}")
        proof_results = self.backward_chainer.do_chain([goal_atom])
        
        return {
            "task": task_description,
            "parsed_intent": parsed_task['intent'],
            "unification_matches": len(unification_results) if unification_results else 0,
            "inferred_capabilities": len(inference_results) if inference_results else 0,
            "goal_proofs": len(proof_results) if proof_results else 0,
            "reasoning_confidence": self._calculate_reasoning_confidence(
                unification_results, inference_results, proof_results
            )
        }
    
    def _calculate_reasoning_confidence(self, unify_results, inference_results, proof_results):
        """Calculate overall confidence in reasoning results."""
        scores = []
        
        if unify_results:
            scores.append(0.8)  # High confidence in pattern matching
        
        if inference_results:
            scores.append(0.9)  # Very high confidence in logical inference
            
        if proof_results:
            scores.append(0.95)  # Highest confidence in formal proofs
        
        return sum(scores) / len(scores) if scores else 0.0

# Example usage
phase2_agent = Phase2CognitiveAgent()
result = await phase2_agent.reason_about_task(
    "I need to write a Python script to analyze data"
)
print(result)
```

### Step 8: Performance Validation

#### Benchmark Phase 2 Components

```python
# Create performance benchmark for Phase 2
# tests/performance/benchmark_phase2_logic.py

import time
import statistics
from typing import List, Dict

class Phase2LogicBenchmark:
    """Benchmark Phase 2 logic systems performance."""
    
    def __init__(self):
        self.results = {
            "unification": [],
            "forward_chaining": [],
            "backward_chaining": [],
            "integration": []
        }
    
    def benchmark_unification(self, pattern_pairs: List[tuple], iterations: int = 100):
        """Benchmark unification performance."""
        
        times = []
        for _ in range(iterations):
            start_time = time.time()
            
            # Simulate unification operations
            for pattern_a, pattern_b in pattern_pairs:
                self._simulate_unification(pattern_a, pattern_b)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        self.results["unification"] = {
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "iterations": iterations,
            "patterns": len(pattern_pairs)
        }
    
    def benchmark_forward_chaining(self, premises: List[str], iterations: int = 50):
        """Benchmark forward chaining performance."""
        
        times = []
        for _ in range(iterations):
            start_time = time.time()
            
            # Simulate forward chaining
            self._simulate_forward_chaining(premises)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        self.results["forward_chaining"] = {
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "iterations": iterations,
            "premises": len(premises)
        }
    
    def benchmark_backward_chaining(self, goals: List[str], iterations: int = 50):
        """Benchmark backward chaining performance."""
        
        times = []
        for _ in range(iterations):
            start_time = time.time()
            
            # Simulate backward chaining
            self._simulate_backward_chaining(goals)
            
            end_time = time.time()
            times.append(end_time - start_time)
        
        self.results["backward_chaining"] = {
            "mean": statistics.mean(times),
            "stdev": statistics.stdev(times) if len(times) > 1 else 0,
            "min": min(times),
            "max": max(times),
            "iterations": iterations,
            "goals": len(goals)
        }
    
    def _simulate_unification(self, pattern_a: str, pattern_b: str):
        """Simulate unification operation (placeholder)."""
        # Placeholder for actual unification
        time.sleep(0.001)  # Simulate processing time
    
    def _simulate_forward_chaining(self, premises: List[str]):
        """Simulate forward chaining operation (placeholder)."""
        # Placeholder for actual forward chaining
        time.sleep(0.005)  # Simulate processing time
    
    def _simulate_backward_chaining(self, goals: List[str]):
        """Simulate backward chaining operation (placeholder)."""
        # Placeholder for actual backward chaining
        time.sleep(0.005)  # Simulate processing time
    
    def generate_report(self) -> Dict:
        """Generate performance report."""
        return {
            "benchmark_results": self.results,
            "summary": {
                "total_operations": sum([
                    r.get("iterations", 0) for r in self.results.values()
                ]),
                "average_unification_time": self.results["unification"].get("mean", 0),
                "average_forward_chain_time": self.results["forward_chaining"].get("mean", 0), 
                "average_backward_chain_time": self.results["backward_chaining"].get("mean", 0),
            },
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate performance recommendations."""
        recommendations = []
        
        if self.results["unification"].get("mean", 0) > 0.1:
            recommendations.append("Consider optimizing unification patterns for better performance")
        
        if self.results["forward_chaining"].get("mean", 0) > 0.05:
            recommendations.append("Forward chaining may benefit from rule optimization")
            
        if self.results["backward_chaining"].get("mean", 0) > 0.05:
            recommendations.append("Backward chaining could be optimized with better goal decomposition")
        
        return recommendations

# Usage example
if __name__ == "__main__":
    benchmark = Phase2LogicBenchmark()
    
    # Benchmark unification
    patterns = [("pattern A", "pattern B"), ("concept X", "concept Y")]
    benchmark.benchmark_unification(patterns)
    
    # Benchmark chaining
    premises = ["fact 1", "fact 2", "fact 3"]
    goals = ["goal 1", "goal 2"]
    benchmark.benchmark_forward_chaining(premises)
    benchmark.benchmark_backward_chaining(goals)
    
    # Generate report
    report = benchmark.generate_report()
    print("Phase 2 Logic Systems Performance Report:")
    print(json.dumps(report, indent=2))
```

## Validation and Testing

### Integration Validation Checklist

- [ ] Unify component cloned successfully
- [ ] URE component cloned successfully  
- [ ] Dependencies validated (atomspace available)
- [ ] Python binding generation completed
- [ ] Agent-Zero tools created and functional
- [ ] Integration tests passing
- [ ] Performance benchmarks completed
- [ ] Documentation updated

### Test Commands

```bash
# Run Phase 2 integration tests
python3 -m pytest tests/integration/test_phase2_logic_systems.py -v

# Run performance benchmarks
python3 tests/performance/benchmark_phase2_logic.py

# Validate components
python3 scripts/cpp2py_conversion_pipeline.py validate unify
python3 scripts/cpp2py_conversion_pipeline.py validate ure

# Check overall status
python3 scripts/cpp2py_conversion_pipeline.py status --phase phase_2_logic_systems
```

## Next Steps After Implementation

### Phase 2 Completion Tasks

1. **Validate All Tools**: Test unification and URE tools with real queries
2. **Performance Optimization**: Optimize logic system performance based on benchmarks
3. **Documentation Enhancement**: Add usage examples and troubleshooting guides
4. **Integration Testing**: Test with existing Agent-Zero cognitive tools
5. **Roadmap Update**: Mark Phase 2 tasks as complete in roadmap

### Preparation for Phase 3

Phase 2 completion enables Phase 3 (Cognitive Systems):
- Attention system integration
- ECAN (Economic Attention Networks) 
- Advanced cognitive capabilities

The logic systems from Phase 2 provide the foundation for attention-guided reasoning in Phase 3.

## Troubleshooting

### Common Issues

**Issue**: Unify component clone fails
**Solution**: Check network connectivity and repository access

**Issue**: Python bindings not generating
**Solution**: Ensure CMake and Python development headers are installed

**Issue**: Agent-Zero tools not finding logic systems  
**Solution**: Verify PYTHONPATH includes OpenCog Python modules

**Issue**: Performance benchmarks showing slow execution
**Solution**: Check AtomSpace size and consider optimization techniques

### Support Resources

- Component documentation: `components/unify/README.md`, `components/ure/README.md`
- Integration tests: `tests/integration/test_phase2_logic_systems.py`
- Performance benchmarks: `tests/performance/benchmark_phase2_logic.py`
- Usage examples: [Logic Systems Integration Patterns](logic_systems_integration_patterns.md)