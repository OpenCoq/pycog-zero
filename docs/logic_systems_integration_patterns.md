# Logic Systems Integration Patterns for Agent-Zero

## Overview

This document provides comprehensive guidance for integrating OpenCog logic systems (unify and URE) with the Agent-Zero framework in PyCog-Zero. These patterns enable advanced cognitive reasoning capabilities within Agent-Zero's tool ecosystem.

## Logic Systems Architecture

### Phase 2 Logic Systems Components

PyCog-Zero integrates two primary logic systems in Phase 2 of development:

1. **Unify System**: Pattern unification and term matching
2. **URE (Unified Rule Engine)**: Forward and backward chaining inference

```
Agent-Zero Framework
├── Cognitive Reasoning Tool
│   ├── Pattern Matching (Unify Integration)
│   ├── Rule-based Inference (URE Integration)  
│   ├── Neural-Symbolic Bridge
│   └── AtomSpace Memory Bridge
├── Logic System Tools
│   ├── UnificationTool
│   ├── ForwardChainingTool
│   ├── BackwardChainingTool
│   └── RuleManagementTool
└── Integration Patterns
    ├── Query-to-Logic Translation
    ├── Logic-to-Response Translation
    ├── Multi-step Reasoning Chains
    └── Attention-guided Logic Processing
```

## Pattern 1: Unification-based Agent Tools

### Basic Unification Pattern

The unification system enables pattern matching and term unification for Agent-Zero queries:

```python
from python.helpers.tool import Tool, Response
from opencog.atomspace import AtomSpace, types
from opencog.unify import Unifier

class UnificationTool(Tool):
    """Agent-Zero tool for pattern unification and matching."""
    
    def __init__(self, agent):
        super().__init__(agent)
        self.atomspace = AtomSpace()
        self.unifier = Unifier()
        self._setup_unification_patterns()
    
    def _setup_unification_patterns(self):
        """Setup common unification patterns for Agent-Zero tasks."""
        # Pattern 1: Task-Goal Unification
        task_pattern = self.atomspace.add_node(
            types.ConceptNode, "task_pattern"
        )
        goal_pattern = self.atomspace.add_node(
            types.ConceptNode, "goal_pattern"
        )
        
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
                self.atomspace.add_node(types.ConceptNode, "unified_result")
            ]
        )
    
    async def execute(self, pattern_a: str, pattern_b: str, **kwargs):
        """Unify two patterns and return results."""
        try:
            # Convert Agent-Zero queries to AtomSpace patterns
            atom_a = self._query_to_atom(pattern_a)
            atom_b = self._query_to_atom(pattern_b)
            
            # Perform unification
            unification_result = self._unify_patterns(atom_a, atom_b)
            
            # Convert back to Agent-Zero response
            response = self._unification_to_response(unification_result)
            
            return Response(
                message=f"Unification completed: {len(response)} results",
                data={
                    "unification_results": response,
                    "pattern_a": pattern_a,
                    "pattern_b": pattern_b,
                    "success": len(response) > 0
                }
            )
            
        except Exception as e:
            return Response(
                message=f"Unification failed: {str(e)}",
                data={"error": str(e), "success": False}
            )
    
    def _query_to_atom(self, query: str):
        """Convert Agent-Zero query string to AtomSpace atom."""
        # Parse natural language query into structured atom
        concepts = self._extract_concepts(query)
        relations = self._extract_relations(query)
        
        # Create composite atom structure
        if relations:
            return self.atomspace.add_link(
                types.EvaluationLink,
                [
                    self.atomspace.add_node(types.PredicateNode, relations[0]),
                    self.atomspace.add_link(
                        types.ListLink,
                        [self.atomspace.add_node(types.ConceptNode, c) for c in concepts]
                    )
                ]
            )
        else:
            return self.atomspace.add_node(types.ConceptNode, concepts[0] if concepts else query)
    
    def _unify_patterns(self, atom_a, atom_b):
        """Perform actual unification using OpenCog unify system."""
        # Create unifier expression
        unifier_expr = self.atomspace.add_link(
            types.UnifierLink,
            [atom_a, atom_b, self.atomspace.add_node(types.VariableNode, "$result")]
        )
        
        # Execute unification
        return self.atomspace.execute(unifier_expr)
    
    def _extract_concepts(self, query: str) -> list:
        """Extract key concepts from natural language query."""
        # Simple keyword extraction (can be enhanced with NLP)
        words = query.lower().split()
        concepts = [word for word in words if len(word) > 3 and word.isalpha()]
        return concepts[:3]  # Limit to top 3 concepts
    
    def _extract_relations(self, query: str) -> list:
        """Extract relations/predicates from query."""
        relation_keywords = ["is", "has", "can", "does", "relates", "connects"]
        words = query.lower().split()
        relations = [word for word in words if word in relation_keywords]
        return relations
    
    def _unification_to_response(self, unification_result) -> list:
        """Convert unification result to Agent-Zero compatible format."""
        if not unification_result:
            return []
        
        # Extract unified bindings and convert to readable format
        results = []
        # This would need actual implementation based on OpenCog unification results
        results.append({
            "binding": "unified_pattern",
            "confidence": 0.8,
            "explanation": "Patterns successfully unified"
        })
        
        return results
```

### Advanced Unification Pattern with Variables

```python
class AdvancedUnificationTool(UnificationTool):
    """Enhanced unification with variable binding and complex pattern matching."""
    
    async def unify_with_variables(self, pattern: str, data: str, variables: list = None):
        """Unify patterns with explicit variable declarations."""
        
        # Create variable declarations
        if variables:
            var_list = self.atomspace.add_link(
                types.VariableList,
                [self.atomspace.add_node(types.VariableNode, f"${var}") for var in variables]
            )
        else:
            # Auto-extract variables from pattern
            var_list = self._auto_extract_variables(pattern)
        
        # Create lambda expressions with proper variable scoping
        pattern_lambda = self.atomspace.add_link(
            types.LambdaLink,
            [var_list, self._query_to_atom(pattern)]
        )
        
        data_lambda = self.atomspace.add_link(
            types.LambdaLink,
            [var_list, self._query_to_atom(data)]
        )
        
        # Create rewrite template
        rewrite_template = self.atomspace.add_link(
            types.ListLink,
            [self.atomspace.add_node(types.VariableNode, f"${var}") for var in (variables or ["result"])]
        )
        
        # Perform advanced unification
        unifier = self.atomspace.add_link(
            types.UnifierLink,
            [pattern_lambda, data_lambda, rewrite_template]
        )
        
        result = self.atomspace.execute(unifier)
        
        return Response(
            message="Advanced unification completed",
            data={
                "unification_result": str(result),
                "variables": variables or ["result"],
                "pattern": pattern,
                "data": data
            }
        )
    
    def _auto_extract_variables(self, pattern: str):
        """Automatically extract variables from pattern string."""
        import re
        # Find $Variable patterns
        variables = re.findall(r'\$(\w+)', pattern)
        if variables:
            return self.atomspace.add_link(
                types.VariableList,
                [self.atomspace.add_node(types.VariableNode, f"${var}") for var in variables]
            )
        else:
            # Create default variable
            return self.atomspace.add_node(types.VariableNode, "$X")
```

## Pattern 2: URE-based Reasoning Chains

### Forward Chaining Pattern

The URE system enables sophisticated forward and backward chaining for Agent-Zero:

```python
from opencog.ure import ForwardChainer, BackwardChainer

class ForwardChainingTool(Tool):
    """Agent-Zero tool for forward chaining inference using URE."""
    
    def __init__(self, agent):
        super().__init__(agent)
        self.atomspace = AtomSpace()
        self.forward_chainer = ForwardChainer(self.atomspace)
        self._setup_inference_rules()
    
    def _setup_inference_rules(self):
        """Setup common inference rules for Agent-Zero reasoning."""
        
        # Rule 1: Inheritance Transitivity
        # If A inherits B and B inherits C, then A inherits C
        inheritance_rule = self.atomspace.add_link(
            types.RuleLink,
            [
                # Rule variables
                self.atomspace.add_link(
                    types.VariableList,
                    [
                        self.atomspace.add_node(types.VariableNode, "$A"),
                        self.atomspace.add_node(types.VariableNode, "$B"),
                        self.atomspace.add_node(types.VariableNode, "$C")
                    ]
                ),
                # Premises
                self.atomspace.add_link(
                    types.AndLink,
                    [
                        self.atomspace.add_link(
                            types.InheritanceLink,
                            [
                                self.atomspace.add_node(types.VariableNode, "$A"),
                                self.atomspace.add_node(types.VariableNode, "$B")
                            ]
                        ),
                        self.atomspace.add_link(
                            types.InheritanceLink,
                            [
                                self.atomspace.add_node(types.VariableNode, "$B"),
                                self.atomspace.add_node(types.VariableNode, "$C")
                            ]
                        )
                    ]
                ),
                # Conclusion
                self.atomspace.add_link(
                    types.InheritanceLink,
                    [
                        self.atomspace.add_node(types.VariableNode, "$A"),
                        self.atomspace.add_node(types.VariableNode, "$C")
                    ]
                )
            ]
        )
        
        # Rule 2: Capability Inference
        # If entity has skill and skill enables capability, then entity can capability
        capability_rule = self.atomspace.add_link(
            types.RuleLink,
            [
                self.atomspace.add_link(
                    types.VariableList,
                    [
                        self.atomspace.add_node(types.VariableNode, "$entity"),
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
                                        self.atomspace.add_node(types.VariableNode, "$entity"),
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
                                self.atomspace.add_node(types.VariableNode, "$entity"),
                                self.atomspace.add_node(types.VariableNode, "$capability")
                            ]
                        )
                    ]
                )
            ]
        )
        
        self.inference_rules = [inheritance_rule, capability_rule]
    
    async def execute(self, premises: list, max_steps: int = 10, **kwargs):
        """Perform forward chaining from given premises."""
        try:
            # Add premises to AtomSpace
            premise_atoms = []
            for premise in premises:
                atom = self._premise_to_atom(premise)
                premise_atoms.append(atom)
            
            # Configure forward chainer
            self.forward_chainer.max_iteration = max_steps
            
            # Add inference rules to chainer
            for rule in self.inference_rules:
                self.forward_chainer.add_rule(rule)
            
            # Run forward chaining from premises
            results = []
            for premise_atom in premise_atoms:
                chain_results = self.forward_chainer.do_chain([premise_atom])
                results.extend(chain_results)
            
            # Convert results to Agent-Zero response
            response_data = self._chain_results_to_response(results)
            
            return Response(
                message=f"Forward chaining completed: {len(results)} inferences",
                data={
                    "inferences": response_data,
                    "premise_count": len(premises),
                    "inference_count": len(results),
                    "max_steps": max_steps
                }
            )
            
        except Exception as e:
            return Response(
                message=f"Forward chaining failed: {str(e)}",
                data={"error": str(e), "success": False}
            )
    
    def _premise_to_atom(self, premise: str):
        """Convert premise string to AtomSpace atom."""
        # Parse premise into structured format
        # Example: "agent has skill programming" -> EvaluationLink
        parts = premise.lower().split()
        
        if "has" in parts and len(parts) >= 4:
            subject = parts[0]
            predicate = "has_" + parts[2]
            object_part = " ".join(parts[3:])
            
            return self.atomspace.add_link(
                types.EvaluationLink,
                [
                    self.atomspace.add_node(types.PredicateNode, predicate),
                    self.atomspace.add_link(
                        types.ListLink,
                        [
                            self.atomspace.add_node(types.ConceptNode, subject),
                            self.atomspace.add_node(types.ConceptNode, object_part)
                        ]
                    )
                ]
            )
        elif "is" in parts and len(parts) >= 3:
            subject = parts[0]
            object_part = " ".join(parts[2:])
            
            return self.atomspace.add_link(
                types.InheritanceLink,
                [
                    self.atomspace.add_node(types.ConceptNode, subject),
                    self.atomspace.add_node(types.ConceptNode, object_part)
                ]
            )
        else:
            # Default to concept
            return self.atomspace.add_node(types.ConceptNode, premise)
    
    def _chain_results_to_response(self, chain_results) -> list:
        """Convert chaining results to Agent-Zero format."""
        response = []
        for result in chain_results:
            response.append({
                "inference": str(result),
                "type": result.type_name if hasattr(result, 'type_name') else "unknown",
                "confidence": 0.8,  # Could be extracted from actual confidence values
                "reasoning_step": len(response) + 1
            })
        return response
```

### Backward Chaining Pattern

```python
class BackwardChainingTool(Tool):
    """Agent-Zero tool for backward chaining inference using URE."""
    
    def __init__(self, agent):
        super().__init__(agent)
        self.atomspace = AtomSpace()
        self.backward_chainer = BackwardChainer(self.atomspace)
        self._setup_goal_directed_rules()
    
    def _setup_goal_directed_rules(self):
        """Setup rules optimized for goal-directed reasoning."""
        # Similar rule setup as forward chainer but optimized for goals
        pass
    
    async def execute(self, goal: str, max_steps: int = 10, **kwargs):
        """Perform backward chaining to achieve a goal."""
        try:
            # Convert goal to AtomSpace representation
            goal_atom = self._goal_to_atom(goal)
            
            # Configure backward chainer
            self.backward_chainer.max_iteration = max_steps
            
            # Run backward chaining
            proof_results = self.backward_chainer.do_chain([goal_atom])
            
            # Extract reasoning path
            reasoning_path = self._extract_reasoning_path(proof_results)
            
            return Response(
                message=f"Backward chaining completed for goal: {goal}",
                data={
                    "goal": goal,
                    "reasoning_path": reasoning_path,
                    "proof_found": len(proof_results) > 0,
                    "steps_taken": len(reasoning_path)
                }
            )
            
        except Exception as e:
            return Response(
                message=f"Backward chaining failed: {str(e)}",
                data={"error": str(e), "success": False}
            )
    
    def _goal_to_atom(self, goal: str):
        """Convert goal string to AtomSpace atom."""
        # Similar to premise parsing but goal-oriented
        return self.atomspace.add_node(types.ConceptNode, goal)
    
    def _extract_reasoning_path(self, proof_results) -> list:
        """Extract step-by-step reasoning path from proof."""
        path = []
        for i, step in enumerate(proof_results):
            path.append({
                "step": i + 1,
                "operation": str(step),
                "justification": f"Applied inference rule to derive {step}"
            })
        return path
```

## Pattern 3: Integrated Logic System Agent Tool

### Comprehensive Logic Reasoning Tool

```python
class LogicSystemsTool(Tool):
    """Comprehensive Agent-Zero tool integrating both unification and URE."""
    
    def __init__(self, agent):
        super().__init__(agent)
        self.atomspace = AtomSpace()
        self.unification_tool = UnificationTool(agent)
        self.forward_chainer_tool = ForwardChainingTool(agent)
        self.backward_chainer_tool = BackwardChainingTool(agent)
        
        # Share atomspace across tools
        self.unification_tool.atomspace = self.atomspace
        self.forward_chainer_tool.atomspace = self.atomspace
        self.backward_chainer_tool.atomspace = self.atomspace
    
    async def execute(self, query: str, operation: str = "auto", **kwargs):
        """Execute logic operations based on query analysis."""
        
        # Analyze query to determine best logic approach
        operation_type = self._analyze_query_for_logic(query) if operation == "auto" else operation
        
        if operation_type == "unify":
            # Extract patterns for unification
            patterns = self._extract_patterns_for_unification(query)
            if len(patterns) >= 2:
                return await self.unification_tool.execute(patterns[0], patterns[1], **kwargs)
        
        elif operation_type == "forward_chain":
            # Extract premises for forward chaining
            premises = self._extract_premises(query)
            return await self.forward_chainer_tool.execute(premises, **kwargs)
        
        elif operation_type == "backward_chain":
            # Extract goal for backward chaining
            goal = self._extract_goal(query)
            return await self.backward_chainer_tool.execute(goal, **kwargs)
        
        elif operation_type == "multi_step":
            # Perform multi-step reasoning combining techniques
            return await self._multi_step_reasoning(query, **kwargs)
        
        else:
            return Response(
                message=f"Unknown logic operation: {operation_type}",
                data={"error": "Unsupported operation", "query": query}
            )
    
    def _analyze_query_for_logic(self, query: str) -> str:
        """Analyze query to determine appropriate logic operation."""
        query_lower = query.lower()
        
        # Pattern matching keywords
        if any(word in query_lower for word in ["match", "unify", "similar", "pattern"]):
            return "unify"
        
        # Goal-directed keywords
        if any(word in query_lower for word in ["achieve", "goal", "want", "need", "prove"]):
            return "backward_chain"
        
        # Forward reasoning keywords
        if any(word in query_lower for word in ["infer", "deduce", "conclude", "given"]):
            return "forward_chain"
        
        # Multi-step reasoning keywords
        if any(word in query_lower for word in ["complex", "multi-step", "reasoning", "analyze"]):
            return "multi_step"
        
        # Default to forward chaining
        return "forward_chain"
    
    def _extract_patterns_for_unification(self, query: str) -> list:
        """Extract patterns for unification from query."""
        # Simple pattern extraction (can be enhanced)
        sentences = query.split("and")
        return [sentence.strip() for sentence in sentences if sentence.strip()]
    
    def _extract_premises(self, query: str) -> list:
        """Extract premises for forward chaining."""
        # Look for premise indicators
        indicators = ["given", "fact", "premise", "assume"]
        premises = []
        
        for indicator in indicators:
            if indicator in query.lower():
                parts = query.lower().split(indicator)
                if len(parts) > 1:
                    premise_text = parts[1].split(".")[0].strip()
                    premises.append(premise_text)
        
        if not premises:
            # Default: treat whole query as premise
            premises.append(query)
        
        return premises
    
    def _extract_goal(self, query: str) -> str:
        """Extract goal for backward chaining."""
        # Look for goal indicators
        indicators = ["prove", "show", "demonstrate", "achieve", "goal"]
        
        for indicator in indicators:
            if indicator in query.lower():
                parts = query.lower().split(indicator)
                if len(parts) > 1:
                    return parts[1].split(".")[0].strip()
        
        # Default: treat whole query as goal
        return query
    
    async def _multi_step_reasoning(self, query: str, **kwargs):
        """Perform complex multi-step reasoning combining multiple logic systems."""
        
        reasoning_steps = []
        
        # Step 1: Extract knowledge through unification
        patterns = self._extract_patterns_for_unification(query)
        if len(patterns) >= 2:
            unify_result = await self.unification_tool.execute(patterns[0], patterns[1])
            reasoning_steps.append({
                "step": 1,
                "operation": "unification",
                "result": unify_result.data
            })
        
        # Step 2: Forward chain from premises
        premises = self._extract_premises(query)
        forward_result = await self.forward_chainer_tool.execute(premises, max_steps=5)
        reasoning_steps.append({
            "step": 2,
            "operation": "forward_chaining",
            "result": forward_result.data
        })
        
        # Step 3: Backward chain to goal if identifiable
        goal = self._extract_goal(query)
        if goal != query:  # Only if goal is different from original query
            backward_result = await self.backward_chainer_tool.execute(goal, max_steps=5)
            reasoning_steps.append({
                "step": 3,
                "operation": "backward_chaining", 
                "result": backward_result.data
            })
        
        # Synthesize results
        final_conclusion = self._synthesize_reasoning_results(reasoning_steps)
        
        return Response(
            message="Multi-step reasoning completed",
            data={
                "query": query,
                "reasoning_steps": reasoning_steps,
                "final_conclusion": final_conclusion,
                "total_steps": len(reasoning_steps)
            }
        )
    
    def _synthesize_reasoning_results(self, reasoning_steps: list) -> dict:
        """Synthesize results from multiple reasoning steps."""
        conclusion = {
            "unified_insights": [],
            "inferred_facts": [],
            "proven_goals": [],
            "confidence": 0.0
        }
        
        total_confidence = 0
        step_count = 0
        
        for step in reasoning_steps:
            step_count += 1
            result = step["result"]
            
            if step["operation"] == "unification" and result.get("success"):
                conclusion["unified_insights"].extend(result.get("unification_results", []))
                total_confidence += 0.7
            
            elif step["operation"] == "forward_chaining":
                conclusion["inferred_facts"].extend(result.get("inferences", []))
                total_confidence += 0.8
            
            elif step["operation"] == "backward_chaining" and result.get("proof_found"):
                conclusion["proven_goals"].append(result.get("goal"))
                total_confidence += 0.9
        
        conclusion["confidence"] = total_confidence / step_count if step_count > 0 else 0.0
        
        return conclusion
```

## Pattern 4: Agent-Zero Query Processing Pipeline

### Logic System Query Router

```python
class LogicQueryRouter(Tool):
    """Routes Agent-Zero queries to appropriate logic systems based on analysis."""
    
    def __init__(self, agent):
        super().__init__(agent)
        self.logic_systems = LogicSystemsTool(agent)
        self.query_analyzer = QueryAnalyzer()
        self.response_formatter = ResponseFormatter()
    
    async def execute(self, query: str, **kwargs):
        """Route query to appropriate logic system and format response."""
        
        # Analyze query complexity and type
        analysis = self.query_analyzer.analyze(query)
        
        # Route to appropriate logic system
        if analysis["complexity"] == "simple":
            result = await self._handle_simple_logic(query, analysis)
        elif analysis["complexity"] == "moderate":
            result = await self._handle_moderate_logic(query, analysis)
        else:
            result = await self._handle_complex_logic(query, analysis)
        
        # Format response for Agent-Zero compatibility
        formatted_response = self.response_formatter.format_logic_response(result, analysis)
        
        return formatted_response
    
    async def _handle_simple_logic(self, query: str, analysis: dict):
        """Handle simple logic queries with single operation."""
        operation = analysis["primary_operation"]
        return await self.logic_systems.execute(query, operation=operation)
    
    async def _handle_moderate_logic(self, query: str, analysis: dict):
        """Handle moderate complexity with 2-3 operations."""
        # Use targeted multi-step approach
        return await self.logic_systems.execute(query, operation="multi_step", max_steps=3)
    
    async def _handle_complex_logic(self, query: str, analysis: dict):
        """Handle complex logic with full reasoning pipeline."""
        # Use comprehensive multi-step reasoning
        return await self.logic_systems.execute(query, operation="multi_step", max_steps=10)

class QueryAnalyzer:
    """Analyze Agent-Zero queries for logic system routing."""
    
    def analyze(self, query: str) -> dict:
        """Analyze query and return routing information."""
        
        complexity_score = self._calculate_complexity(query)
        primary_operation = self._identify_primary_operation(query)
        required_systems = self._identify_required_systems(query)
        
        return {
            "complexity": "simple" if complexity_score < 3 else "moderate" if complexity_score < 7 else "complex",
            "complexity_score": complexity_score,
            "primary_operation": primary_operation,
            "required_systems": required_systems,
            "query_length": len(query),
            "sentence_count": len(query.split(".")),
            "concept_count": len(self._extract_concepts(query))
        }
    
    def _calculate_complexity(self, query: str) -> int:
        """Calculate complexity score for query."""
        score = 0
        
        # Length factor
        score += min(len(query) // 50, 3)
        
        # Sentence count factor
        score += len(query.split("."))
        
        # Logic keyword factor
        logic_keywords = ["if", "then", "because", "therefore", "since", "implies", "given", "assume"]
        score += len([word for word in query.lower().split() if word in logic_keywords])
        
        # Question complexity
        question_words = ["what", "why", "how", "when", "where", "which", "who"]
        score += len([word for word in query.lower().split() if word in question_words])
        
        return score
    
    def _identify_primary_operation(self, query: str) -> str:
        """Identify the primary logic operation needed."""
        query_lower = query.lower()
        
        # Operation indicators with priority
        operations = [
            ("unify", ["match", "similar", "pattern", "unify", "compare"]),
            ("backward_chain", ["prove", "show", "demonstrate", "goal", "achieve"]),
            ("forward_chain", ["infer", "deduce", "conclude", "given", "fact"])
        ]
        
        for operation, keywords in operations:
            if any(keyword in query_lower for keyword in keywords):
                return operation
        
        return "auto"
    
    def _identify_required_systems(self, query: str) -> list:
        """Identify which logic systems are needed."""
        systems = []
        query_lower = query.lower()
        
        if any(word in query_lower for word in ["match", "unify", "pattern"]):
            systems.append("unify")
        
        if any(word in query_lower for word in ["infer", "chain", "rule", "conclude"]):
            systems.append("ure")
        
        if any(word in query_lower for word in ["complex", "multi", "step", "reasoning"]):
            systems.extend(["unify", "ure"])
        
        return systems or ["auto"]
    
    def _extract_concepts(self, query: str) -> list:
        """Extract key concepts from query."""
        words = query.lower().split()
        concepts = [word for word in words if len(word) > 3 and word.isalpha()]
        return list(set(concepts))

class ResponseFormatter:
    """Format logic system responses for Agent-Zero compatibility."""
    
    def format_logic_response(self, result: Response, analysis: dict) -> Response:
        """Format logic system response for Agent-Zero."""
        
        formatted_data = {
            "logic_analysis": analysis,
            "reasoning_type": analysis["primary_operation"],
            "complexity": analysis["complexity"],
            "original_result": result.data,
            "formatted_insights": self._extract_insights(result.data),
            "actionable_items": self._extract_actionable_items(result.data),
            "confidence_score": self._calculate_confidence(result.data)
        }
        
        formatted_message = self._format_message(result, analysis)
        
        return Response(
            message=formatted_message,
            data=formatted_data
        )
    
    def _extract_insights(self, result_data: dict) -> list:
        """Extract key insights from logic system results."""
        insights = []
        
        if "unification_results" in result_data:
            insights.extend([f"Pattern match: {ur}" for ur in result_data["unification_results"]])
        
        if "inferences" in result_data:
            insights.extend([f"Inference: {inf['inference']}" for inf in result_data["inferences"]])
        
        if "reasoning_path" in result_data:
            insights.extend([f"Reasoning step: {step['operation']}" for step in result_data["reasoning_path"]])
        
        return insights
    
    def _extract_actionable_items(self, result_data: dict) -> list:
        """Extract actionable items from logic results."""
        actions = []
        
        if result_data.get("proof_found"):
            actions.append("Goal can be achieved through identified reasoning path")
        
        if result_data.get("inference_count", 0) > 0:
            actions.append(f"Apply {result_data['inference_count']} new inferences")
        
        if result_data.get("unification_results"):
            actions.append("Use unified patterns for further processing")
        
        return actions
    
    def _calculate_confidence(self, result_data: dict) -> float:
        """Calculate overall confidence score."""
        base_confidence = result_data.get("confidence", 0.5)
        
        # Boost confidence based on result quality
        if result_data.get("proof_found"):
            base_confidence += 0.2
        
        if result_data.get("inference_count", 0) > 0:
            base_confidence += 0.1
        
        return min(base_confidence, 1.0)
    
    def _format_message(self, result: Response, analysis: dict) -> str:
        """Format human-readable message."""
        operation = analysis["primary_operation"]
        complexity = analysis["complexity"]
        
        if operation == "unify":
            return f"Pattern unification completed ({complexity} complexity): {result.message}"
        elif operation == "forward_chain":
            return f"Forward reasoning completed ({complexity} complexity): {result.message}"
        elif operation == "backward_chain":
            return f"Goal-directed reasoning completed ({complexity} complexity): {result.message}"
        else:
            return f"Logic processing completed ({complexity} complexity): {result.message}"
```

## Integration Examples

### Example 1: Task Planning with Logic Systems

```python
# Agent-Zero tool that uses logic systems for task planning
class TaskPlanningTool(Tool):
    """Uses logic systems to plan and reason about task execution."""
    
    def __init__(self, agent):
        super().__init__(agent)
        self.logic_router = LogicQueryRouter(agent)
    
    async def execute(self, task_description: str, **kwargs):
        """Plan task using logic reasoning."""
        
        # Step 1: Analyze task requirements
        analysis_query = f"What are the requirements and dependencies for: {task_description}"
        requirements = await self.logic_router.execute(analysis_query, operation="forward_chain")
        
        # Step 2: Generate plan options  
        planning_query = f"Generate execution plans for task with requirements: {requirements.data}"
        plans = await self.logic_router.execute(planning_query, operation="backward_chain")
        
        # Step 3: Validate plans
        validation_query = f"Validate feasibility of plans: {plans.data}"
        validation = await self.logic_router.execute(validation_query, operation="multi_step")
        
        return Response(
            message=f"Task planning completed for: {task_description}",
            data={
                "task": task_description,
                "requirements": requirements.data,
                "plans": plans.data,
                "validation": validation.data,
                "recommended_plan": self._select_best_plan(plans.data, validation.data)
            }
        )
    
    def _select_best_plan(self, plans_data: dict, validation_data: dict) -> dict:
        """Select best plan based on logic analysis."""
        # Implementation would analyze confidence scores and feasibility
        return {
            "plan_id": 1,
            "confidence": 0.85,
            "reasoning": "Selected based on highest feasibility and confidence scores"
        }
```

### Example 2: Knowledge Integration

```python
# Agent-Zero tool for integrating new knowledge using logic systems
class KnowledgeIntegrationTool(Tool):
    """Integrate new knowledge using unification and reasoning."""
    
    async def execute(self, new_knowledge: str, existing_context: str = "", **kwargs):
        """Integrate new knowledge with existing context."""
        
        # Use unification to find relationships
        unification_query = f"Find relationships between '{new_knowledge}' and existing knowledge"
        relationships = await self.logic_router.execute(unification_query, operation="unify")
        
        # Use forward chaining to derive implications
        inference_query = f"Given new knowledge: {new_knowledge}, what can be inferred?"
        implications = await self.logic_router.execute(inference_query, operation="forward_chain")
        
        # Integrate results
        return Response(
            message="Knowledge integration completed",
            data={
                "new_knowledge": new_knowledge,
                "relationships": relationships.data,
                "implications": implications.data,
                "integration_confidence": (relationships.data.get("confidence", 0) + implications.data.get("confidence", 0)) / 2
            }
        )
```

## Testing Patterns

### Integration Test Pattern

```python
# Test pattern for logic system integration
import pytest
from unittest.mock import MagicMock

class TestLogicSystemsIntegration:
    """Test logic systems integration with Agent-Zero."""
    
    @pytest.fixture
    def mock_agent(self):
        agent = MagicMock()
        agent.config = {"cognitive_mode": True}
        return agent
    
    @pytest.fixture
    def logic_router(self, mock_agent):
        return LogicQueryRouter(mock_agent)
    
    @pytest.mark.asyncio
    async def test_unification_pattern(self, logic_router):
        """Test unification pattern integration."""
        query = "Match pattern A with pattern B"
        result = await logic_router.execute(query)
        
        assert result.data["reasoning_type"] == "unify"
        assert "logic_analysis" in result.data
        assert result.data["confidence_score"] >= 0.0
    
    @pytest.mark.asyncio
    async def test_forward_chaining_pattern(self, logic_router):
        """Test forward chaining integration."""
        query = "Given fact A, infer conclusions"
        result = await logic_router.execute(query)
        
        assert result.data["reasoning_type"] == "forward_chain"
        assert "formatted_insights" in result.data
    
    @pytest.mark.asyncio
    async def test_complex_reasoning_pattern(self, logic_router):
        """Test complex multi-step reasoning."""
        query = "Analyze complex scenario with multiple facts and goals"
        result = await logic_router.execute(query)
        
        assert result.data["complexity"] in ["moderate", "complex"]
        assert "actionable_items" in result.data
    
    def test_query_analysis(self):
        """Test query analysis functionality."""
        analyzer = QueryAnalyzer()
        
        simple_query = "Match A and B"
        analysis = analyzer.analyze(simple_query)
        assert analysis["complexity"] == "simple"
        
        complex_query = "Given multiple premises, prove the goal through multi-step reasoning while considering various alternatives and implications"
        analysis = analyzer.analyze(complex_query)
        assert analysis["complexity"] == "complex"
```

## Configuration and Setup

### Logic Systems Configuration

```json
{
  "logic_systems": {
    "enabled": true,
    "unify_system": {
      "enabled": true,
      "max_unification_depth": 5,
      "variable_extraction": "auto",
      "pattern_matching_threshold": 0.7
    },
    "ure_system": {
      "enabled": true,
      "max_forward_steps": 10,
      "max_backward_steps": 10,
      "rule_confidence_threshold": 0.6,
      "chaining_timeout": 30
    },
    "integration_settings": {
      "shared_atomspace": true,
      "cross_tool_communication": true,
      "result_caching": true,
      "performance_optimization": true
    }
  },
  "agent_zero_integration": {
    "auto_route_queries": true,
    "format_responses": true,
    "extract_actionable_items": true,
    "confidence_threshold": 0.5
  }
}
```

## Performance Considerations

### Optimization Patterns

1. **Shared AtomSpace**: Use single AtomSpace instance across logic tools
2. **Rule Caching**: Cache frequently used inference rules
3. **Result Memoization**: Cache results for repeated queries
4. **Incremental Processing**: Process large knowledge bases incrementally
5. **Attention-guided Processing**: Use ECAN for focusing reasoning

### Monitoring and Metrics

```python
class LogicSystemsMonitor:
    """Monitor performance and usage of logic systems."""
    
    def __init__(self):
        self.metrics = {
            "unification_count": 0,
            "forward_chain_count": 0,
            "backward_chain_count": 0,
            "average_response_time": 0.0,
            "success_rate": 0.0,
            "error_count": 0
        }
    
    def record_operation(self, operation_type: str, duration: float, success: bool):
        """Record operation metrics."""
        self.metrics[f"{operation_type}_count"] += 1
        
        # Update average response time
        total_ops = sum([count for key, count in self.metrics.items() if key.endswith("_count")])
        self.metrics["average_response_time"] = (
            (self.metrics["average_response_time"] * (total_ops - 1) + duration) / total_ops
        )
        
        # Update success rate
        if success:
            success_count = total_ops - self.metrics["error_count"]
            self.metrics["success_rate"] = success_count / total_ops
        else:
            self.metrics["error_count"] += 1
            self.metrics["success_rate"] = (total_ops - self.metrics["error_count"]) / total_ops
```

## Conclusion

These logic system integration patterns provide a comprehensive framework for integrating OpenCog's unification and URE systems with Agent-Zero. The patterns enable:

1. **Flexible Query Processing**: Automatic routing of queries to appropriate logic systems
2. **Multi-step Reasoning**: Complex reasoning chains combining multiple techniques  
3. **Agent-Zero Compatibility**: Seamless integration with existing Agent-Zero tools
4. **Performance Optimization**: Efficient shared resources and caching strategies
5. **Extensible Architecture**: Easy addition of new logic capabilities

The patterns support both simple single-operation queries and complex multi-step reasoning scenarios, making them suitable for a wide range of cognitive agent applications.