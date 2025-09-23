from python.helpers.api import ApiHandler
from python.helpers import settings
import json
import os
import traceback


class CognitiveStatusHandler(ApiHandler):
    """API handler for cognitive system status monitoring."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.path = "/api/cognitive/status"
        self.method = "GET"

    async def handle_request(self, data=None):
        try:
            # Load cognitive configuration
            config = settings.get_cognitive_config()
            
            # Initialize stats
            stats = {
                "cognitive_mode": config.get("cognitive_mode", False),
                "opencog_enabled": config.get("opencog_enabled", False),
                "neural_symbolic_bridge": config.get("neural_symbolic_bridge", False),
                "ecan_attention": config.get("ecan_attention", False),
                "pln_reasoning": config.get("pln_reasoning", False),
                "atomspace_persistence": config.get("atomspace_persistence", False),
                "atomspace_stats": {
                    "atoms": 0,
                    "memory": 0,
                    "tools_active": []
                }
            }
            
            # Try to get AtomSpace statistics if available
            try:
                from python.tools.cognitive_reasoning import CognitiveReasoningTool
                tool = CognitiveReasoningTool()
                if hasattr(tool, 'get_stats'):
                    atomspace_stats = tool.get_stats()
                    stats["atomspace_stats"].update(atomspace_stats)
            except ImportError:
                # Cognitive tools not available, use defaults
                pass
            except Exception:
                # OpenCog not available or other error, use defaults  
                pass
            
            # Check for active cognitive tool usage
            try:
                # Look for recent cognitive tool usage in logs
                cognitive_tools = [
                    "cognitive_reasoning",
                    "cognitive_memory", 
                    "meta_cognition",
                    "neural_symbolic_agent",
                    "ure_tool",
                    "atomspace_tool_hub"
                ]
                
                active_tools = []
                log_file = "logs/agent.log"
                if os.path.exists(log_file):
                    # Read last 100 lines of log file
                    with open(log_file, 'r') as f:
                        lines = f.readlines()[-100:]
                        
                    # Look for cognitive tool usage in recent logs
                    for line in lines:
                        for tool in cognitive_tools:
                            if tool in line and "executing" in line.lower():
                                if tool not in active_tools:
                                    active_tools.append(tool)
                
                stats["atomspace_stats"]["tools_active"] = active_tools
                
            except Exception:
                # Log parsing failed, use defaults
                pass
            
            return stats
            
        except Exception as e:
            traceback.print_exc()
            return {
                "error": f"Failed to get cognitive status: {str(e)}",
                "cognitive_mode": False,
                "opencog_enabled": False,
                "neural_symbolic_bridge": False,
                "ecan_attention": False,
                "pln_reasoning": False,
                "atomspace_persistence": False,
                "atomspace_stats": {
                    "atoms": 0,
                    "memory": 0,
                    "tools_active": []
                }
            }