from python.helpers import files, memory
from python.helpers.tool import Tool, Response
from agent import Agent
from python.helpers.log import LogItem
import json
import time
from typing import Dict, Any, Optional

# Import cognitive learning for learning-based behavior adjustment
try:
    from python.tools.cognitive_learning import CognitiveLearningTool, LearningExperience
    COGNITIVE_LEARNING_AVAILABLE = True
except ImportError:
    COGNITIVE_LEARNING_AVAILABLE = False


class UpdateBehaviour(Tool):
    """Enhanced behavior adjustment tool with cognitive learning integration."""
    
    def _initialize_learning_integration(self):
        """Initialize integration with cognitive learning system."""
        if hasattr(self, '_behavior_initialized'):
            return
        
        self._behavior_initialized = True
        self.learning_tool: Optional[CognitiveLearningTool] = None
        if COGNITIVE_LEARNING_AVAILABLE:
            try:
                self.learning_tool = CognitiveLearningTool(self.agent, "cognitive_learning", None, {}, "", None)
            except Exception as e:
                print(f"Warning: Could not initialize cognitive learning integration - {e}")

    async def execute(self, adjustments="", **kwargs):
        """Execute behavior adjustment with learning-based enhancement."""
        
        # Initialize if needed
        self._initialize_learning_integration()
        
        # stringify adjustments if needed
        if not isinstance(adjustments, str):
            adjustments = str(adjustments)

        # Record the current behavior adjustment as a learning experience
        context = {
            "type": "behavior_adjustment",
            "domain": kwargs.get("domain", "general"),
            "trigger": kwargs.get("trigger", "manual_adjustment"),
            "adjustment_type": self._classify_adjustment_type(adjustments)
        }
        
        # Get learning-based recommendations if available
        learning_recommendations = await self._get_learning_recommendations(context)
        if learning_recommendations:
            adjustments = self._integrate_learning_recommendations(adjustments, learning_recommendations)
        
        # Perform the behavior update
        await update_behaviour(self.agent, self.log, adjustments)
        
        # Record this as a learning experience for future reference
        await self._record_adjustment_experience(context, adjustments, kwargs.get("expected_success", 0.7))
        
        return Response(
            message=self.agent.read_prompt("behaviour.updated.md"), 
            data={
                "adjustment_enhanced": learning_recommendations is not None,
                "learning_integrated": COGNITIVE_LEARNING_AVAILABLE,
                "context": context
            },
            break_loop=False
        )
    
    def _classify_adjustment_type(self, adjustments: str) -> str:
        """Classify the type of behavioral adjustment being made."""
        adjustments_lower = adjustments.lower()
        
        if any(word in adjustments_lower for word in ["error", "mistake", "fix", "correct"]):
            return "error_correction"
        elif any(word in adjustments_lower for word in ["improve", "optimize", "enhance", "better"]):
            return "performance_improvement"  
        elif any(word in adjustments_lower for word in ["add", "new", "introduce", "expand"]):
            return "capability_extension"
        elif any(word in adjustments_lower for word in ["remove", "delete", "stop", "eliminate"]):
            return "behavior_reduction"
        else:
            return "general_modification"
    
    async def _get_learning_recommendations(self, context: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Get recommendations from cognitive learning system."""
        if not self.learning_tool:
            return None
        
        try:
            response = await self.learning_tool.get_adaptation_recommendations({"context": context})
            if response.data and response.data.get("recommendations"):
                return response.data
        except Exception as e:
            print(f"Warning: Could not get learning recommendations - {e}")
        
        return None
    
    def _integrate_learning_recommendations(self, original_adjustments: str, learning_data: Dict[str, Any]) -> str:
        """Integrate learning recommendations with original adjustments."""
        recommendations = learning_data.get("recommendations", [])
        if not recommendations:
            return original_adjustments
        
        # Filter high-priority recommendations
        high_priority_recs = [rec for rec in recommendations if rec.get("priority") == "high"]
        
        if high_priority_recs:
            learning_additions = "\n\n# Learning-Based Enhancements:\n"
            for rec in high_priority_recs:
                learning_additions += f"- {rec['recommendation']} (Reason: {rec['rationale']})\n"
            
            return original_adjustments + learning_additions
        
        return original_adjustments
    
    async def _record_adjustment_experience(self, context: Dict[str, Any], adjustments: str, expected_success: float):
        """Record the behavior adjustment as a learning experience."""
        if not self.learning_tool:
            return
        
        try:
            experience_data = {
                "context": context,
                "action": f"behavior_adjustment_{context['adjustment_type']}",
                "outcome": {
                    "adjustment_content": adjustments[:200],  # First 200 chars
                    "timestamp": time.time()
                },
                "success_score": expected_success,  # Will be updated later with actual outcome
                "learning_type": "behavioral_adaptation"
            }
            
            await self.learning_tool.record_experience(experience_data)
            
        except Exception as e:
            print(f"Warning: Could not record adjustment experience - {e}")

    # async def before_execution(self, **kwargs):
    #     pass

    # async def after_execution(self, response, **kwargs):
    #     pass

    # async def before_execution(self, **kwargs):
    #     pass

    # async def after_execution(self, response, **kwargs):
    #     pass


async def update_behaviour(agent: Agent, log_item: LogItem, adjustments: str):

    # get system message and current ruleset
    system = agent.read_prompt("behaviour.merge.sys.md")
    current_rules = read_rules(agent)

    # log query streamed by LLM
    async def log_callback(content):
        log_item.stream(ruleset=content)

    msg = agent.read_prompt(
        "behaviour.merge.msg.md", current_rules=current_rules, adjustments=adjustments
    )

    # call util llm to find solutions in history
    adjustments_merge = await agent.call_utility_model(
        system=system,
        message=msg,
        callback=log_callback,
    )

    # update rules file
    rules_file = get_custom_rules_file(agent)
    files.write_file(rules_file, adjustments_merge)
    log_item.update(result="Behaviour updated")


def get_custom_rules_file(agent: Agent):
    return memory.get_memory_subdir_abs(agent) + f"/behaviour.md"


def read_rules(agent: Agent):
    rules_file = get_custom_rules_file(agent)
    if files.exists(rules_file):
        rules = files.read_file(rules_file)
        return agent.read_prompt("agent.system.behaviour.md", rules=rules)
    else:
        rules = agent.read_prompt("agent.system.behaviour_default.md")
        return agent.read_prompt("agent.system.behaviour.md", rules=rules)
