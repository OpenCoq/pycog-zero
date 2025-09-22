#!/usr/bin/env python3
"""
Logic Systems Integration Example for PyCog-Zero

This example demonstrates how to use the Phase 2 logic systems (unify and URE) 
with Agent-Zero for advanced cognitive reasoning.

Usage:
    python3 examples/logic_systems_example.py
"""

import asyncio
import json
from typing import Dict, Any, List

# Mock imports for demonstration (actual imports depend on Phase 2 implementation)
try:
    from python.tools.unification_tool import UnificationTool
    from python.tools.ure_reasoning_tool import UREReasoningTool
    from python.tools.logic_systems_tool import LogicSystemsTool
    from python.helpers.tool import Tool, Response
    LOGIC_TOOLS_AVAILABLE = True
except ImportError:
    print("Logic tools not yet available - showing simulation")
    LOGIC_TOOLS_AVAILABLE = False

class MockAgent:
    """Mock Agent-Zero for demonstration purposes."""
    def __init__(self):
        self.config = {
            "cognitive_mode": True,
            "phase2_logic": True,
            "opencog_enabled": True
        }

class LogicSystemsExample:
    """Example demonstrating Phase 2 logic systems integration patterns."""
    
    def __init__(self):
        self.agent = MockAgent()
        self.setup_tools()
    
    def setup_tools(self):
        """Setup Phase 2 logic system tools."""
        if LOGIC_TOOLS_AVAILABLE:
            self.unification_tool = UnificationTool(self.agent)
            self.ure_tool = UREReasoningTool(self.agent)
            self.logic_systems_tool = LogicSystemsTool(self.agent)
        else:
            # Create mock tools for demonstration
            self.unification_tool = self.create_mock_unification_tool()
            self.ure_tool = self.create_mock_ure_tool()
            self.logic_systems_tool = self.create_mock_logic_systems_tool()
    
    def create_mock_unification_tool(self):
        """Create mock unification tool for demonstration."""
        class MockUnificationTool:
            async def execute(self, pattern_a: str, pattern_b: str, **kwargs):
                return type('Response', (), {
                    'message': f"Mock unification: {pattern_a} <-> {pattern_b}",
                    'data': {
                        'pattern_a': pattern_a,
                        'pattern_b': pattern_b,
                        'unification_results': [
                            {'result': 'unified_pattern_1', 'confidence': 0.8},
                            {'result': 'unified_pattern_2', 'confidence': 0.6}
                        ],
                        'success': True
                    }
                })()
        return MockUnificationTool()
    
    def create_mock_ure_tool(self):
        """Create mock URE tool for demonstration."""
        class MockURETools:
            async def execute(self, premises: List[str], operation: str = "forward", **kwargs):
                return type('Response', (), {
                    'message': f"Mock {operation} chaining with {len(premises)} premises",
                    'data': {
                        'operation': operation,
                        'premises': premises,
                        'reasoning_results': [
                            {'inference': f'inference_from_{premise}', 'confidence': 0.9}
                            for premise in premises[:2]  # Limit for demo
                        ],
                        'inference_count': len(premises),
                        'success': True
                    }
                })()
        return MockURETools()
    
    def create_mock_logic_systems_tool(self):
        """Create mock integrated logic systems tool."""
        class MockLogicSystemsTool:
            async def execute(self, query: str, operation: str = "auto", **kwargs):
                return type('Response', (), {
                    'message': f"Mock logic processing: {operation} operation",
                    'data': {
                        'query': query,
                        'operation': operation,
                        'reasoning_steps': [
                            {'step': 1, 'operation': 'unification', 'result': 'patterns_unified'},
                            {'step': 2, 'operation': 'forward_chaining', 'result': 'inferences_derived'},
                            {'step': 3, 'operation': 'synthesis', 'result': 'conclusion_reached'}
                        ],
                        'final_conclusion': {
                            'unified_insights': ['insight_1', 'insight_2'],
                            'inferred_facts': ['fact_1', 'fact_2'],
                            'confidence': 0.85
                        }
                    }
                })()
        return MockLogicSystemsTool()

    async def run_unification_example(self):
        """Example 1: Pattern unification for concept matching."""
        print("\n" + "="*60)
        print("EXAMPLE 1: Pattern Unification")
        print("="*60)
        
        # Define patterns to unify
        pattern_a = "agent has skill programming"
        pattern_b = "entity possesses capability coding"
        
        print(f"Pattern A: {pattern_a}")
        print(f"Pattern B: {pattern_b}")
        print("\nPerforming unification...")
        
        # Perform unification
        result = await self.unification_tool.execute(pattern_a, pattern_b)
        
        print(f"\nResult: {result.message}")
        print("\nUnification Results:")
        for i, ur in enumerate(result.data['unification_results'], 1):
            print(f"  {i}. {ur['result']} (confidence: {ur['confidence']})")
        
        return result.data

    async def run_forward_chaining_example(self):
        """Example 2: Forward chaining for inference."""
        print("\n" + "="*60)
        print("EXAMPLE 2: Forward Chaining Inference")
        print("="*60)
        
        # Define premises
        premises = [
            "agent has skill programming",
            "programming skill enables web development",
            "web development requires problem solving",
            "agent has experience with Python"
        ]
        
        print("Premises:")
        for i, premise in enumerate(premises, 1):
            print(f"  {i}. {premise}")
        
        print("\nPerforming forward chaining...")
        
        # Perform forward chaining
        result = await self.ure_tool.execute(premises, operation="forward")
        
        print(f"\nResult: {result.message}")
        print(f"Inferences derived: {result.data['inference_count']}")
        print("\nReasoning Results:")
        for i, inference in enumerate(result.data['reasoning_results'], 1):
            print(f"  {i}. {inference['inference']} (confidence: {inference['confidence']})")
        
        return result.data

    async def run_backward_chaining_example(self):
        """Example 3: Backward chaining for goal achievement."""
        print("\n" + "="*60)
        print("EXAMPLE 3: Backward Chaining for Goal Achievement")
        print("="*60)
        
        # Define goals
        goals = [
            "agent can build web application",
            "agent can solve complex problems"
        ]
        
        print("Goals to achieve:")
        for i, goal in enumerate(goals, 1):
            print(f"  {i}. {goal}")
        
        print("\nPerforming backward chaining...")
        
        # Perform backward chaining
        result = await self.ure_tool.execute(goals, operation="backward")
        
        print(f"\nResult: {result.message}")
        print(f"Proofs found: {result.data['inference_count']}")
        print("\nReasoning Path:")
        for i, step in enumerate(result.data['reasoning_results'], 1):
            print(f"  {i}. {step['inference']} (confidence: {step['confidence']})")
        
        return result.data

    async def run_integrated_reasoning_example(self):
        """Example 4: Integrated multi-step reasoning."""
        print("\n" + "="*60)
        print("EXAMPLE 4: Integrated Multi-Step Reasoning")
        print("="*60)
        
        # Complex query requiring multiple reasoning techniques
        complex_query = """
        Given that an agent has programming skills and wants to build a web application,
        analyze what capabilities are needed, what can be inferred about the agent's
        potential, and determine the best approach for achieving the goal.
        """
        
        print("Complex Query:")
        print(complex_query)
        print("\nPerforming integrated reasoning...")
        
        # Use integrated logic systems tool
        result = await self.logic_systems_tool.execute(complex_query, operation="multi_step")
        
        print(f"\nResult: {result.message}")
        print(f"Total reasoning steps: {len(result.data['reasoning_steps'])}")
        
        print("\nReasoning Process:")
        for step in result.data['reasoning_steps']:
            print(f"  Step {step['step']}: {step['operation']} -> {step['result']}")
        
        print("\nFinal Conclusion:")
        conclusion = result.data['final_conclusion']
        print(f"  Confidence: {conclusion['confidence']}")
        print(f"  Unified insights: {len(conclusion['unified_insights'])}")
        print(f"  Inferred facts: {len(conclusion['inferred_facts'])}")
        
        if conclusion['unified_insights']:
            print("  Key insights:")
            for insight in conclusion['unified_insights']:
                print(f"    - {insight}")
        
        return result.data

    async def run_task_planning_example(self):
        """Example 5: Task planning using logic systems."""
        print("\n" + "="*60)
        print("EXAMPLE 5: Task Planning with Logic Systems")
        print("="*60)
        
        task_description = "Create a data analysis dashboard for sales metrics"
        
        print(f"Task: {task_description}")
        print("\nAnalyzing task requirements and generating plan...")
        
        # Step 1: Analyze requirements
        requirements_query = f"What are the requirements and dependencies for: {task_description}"
        requirements_result = await self.logic_systems_tool.execute(
            requirements_query, operation="forward_chain"
        )
        
        # Step 2: Generate plan
        planning_query = f"Generate execution plan for task with identified requirements"
        plan_result = await self.logic_systems_tool.execute(
            planning_query, operation="backward_chain"
        )
        
        # Step 3: Validate plan
        validation_query = f"Validate feasibility and optimize the proposed plan"
        validation_result = await self.logic_systems_tool.execute(
            validation_query, operation="multi_step"
        )
        
        print("\nTask Planning Results:")
        print(f"  Requirements analysis: {requirements_result.message}")
        print(f"  Plan generation: {plan_result.message}")
        print(f"  Plan validation: {validation_result.message}")
        
        # Synthesize results
        planning_data = {
            "task": task_description,
            "requirements": requirements_result.data,
            "plan": plan_result.data,
            "validation": validation_result.data,
            "overall_confidence": (
                requirements_result.data.get('final_conclusion', {}).get('confidence', 0.5) +
                plan_result.data.get('final_conclusion', {}).get('confidence', 0.5) +
                validation_result.data.get('final_conclusion', {}).get('confidence', 0.5)
            ) / 3
        }
        
        print(f"\nOverall planning confidence: {planning_data['overall_confidence']:.2f}")
        
        return planning_data

    async def run_all_examples(self):
        """Run all logic systems examples."""
        print("Phase 2 Logic Systems Integration Examples")
        print("PyCog-Zero with Agent-Zero Framework")
        print("="*60)
        
        if not LOGIC_TOOLS_AVAILABLE:
            print("NOTE: Using mock tools for demonstration")
            print("Install Phase 2 components for full functionality\n")
        
        # Run examples
        results = {}
        
        try:
            results['unification'] = await self.run_unification_example()
            results['forward_chaining'] = await self.run_forward_chaining_example()
            results['backward_chaining'] = await self.run_backward_chaining_example()
            results['integrated_reasoning'] = await self.run_integrated_reasoning_example()
            results['task_planning'] = await self.run_task_planning_example()
            
        except Exception as e:
            print(f"\nError running examples: {e}")
            return None
        
        # Summary
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60)
        print("Successfully demonstrated Phase 2 logic systems integration patterns:")
        print("  ✓ Pattern unification for concept matching")
        print("  ✓ Forward chaining for logical inference")
        print("  ✓ Backward chaining for goal-directed reasoning")
        print("  ✓ Integrated multi-step reasoning")
        print("  ✓ Task planning with logic systems")
        print("\nThese patterns enable Agent-Zero to perform sophisticated")
        print("cognitive reasoning using OpenCog logic systems.")
        
        return results

def main():
    """Main function to run the examples."""
    example = LogicSystemsExample()
    
    try:
        # Run all examples
        results = asyncio.run(example.run_all_examples())
        
        # Save results to file for reference
        if results:
            with open('/tmp/logic_systems_example_results.json', 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"\nResults saved to: /tmp/logic_systems_example_results.json")
    
    except KeyboardInterrupt:
        print("\nExamples interrupted by user")
    except Exception as e:
        print(f"\nError running examples: {e}")

if __name__ == "__main__":
    main()