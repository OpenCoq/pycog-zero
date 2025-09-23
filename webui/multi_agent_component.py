"""
Multi-Agent Collaboration Web Interface Component
Provides web UI integration for multi-agent cognitive collaboration framework.
"""

def multi_agent_dashboard():
    """Create multi-agent dashboard component for the web UI."""
    
    return """
    <div class="multi-agent-dashboard" style="margin: 20px 0; padding: 20px; border: 1px solid #ddd; border-radius: 8px; background-color: #f9f9f9;">
        <h3>🤖 Multi-Agent Cognitive Collaboration</h3>
        
        <div class="agent-status" style="margin: 15px 0;">
            <h4>System Status</h4>
            <div id="agent-status-content">
                <p>Loading multi-agent system status...</p>
            </div>
        </div>
        
        <div class="agent-controls" style="margin: 15px 0;">
            <h4>Collaboration Controls</h4>
            <div style="display: flex; gap: 10px; flex-wrap: wrap;">
                <button onclick="getAgentStatus()" class="btn btn-info btn-sm">
                    📊 System Status
                </button>
                <button onclick="listAgents()" class="btn btn-primary btn-sm">
                    👥 List Agents
                </button>
                <button onclick="startCollaboration()" class="btn btn-success btn-sm">
                    🧠 Collaborative Reasoning
                </button>
                <button onclick="runWorkflow()" class="btn btn-warning btn-sm">
                    🔄 End-to-End Workflow
                </button>
            </div>
        </div>
        
        <div class="collaboration-results" style="margin: 15px 0;">
            <h4>Results</h4>
            <div id="collaboration-results" style="background: white; padding: 10px; border-radius: 4px; min-height: 100px; font-family: monospace; font-size: 12px; overflow-y: auto;">
                <p style="color: #666;">Multi-agent collaboration results will appear here...</p>
            </div>
        </div>
    </div>
    
    <script>
        // Multi-Agent Collaboration JavaScript Functions
        
        function updateResults(content, type = 'info') {
            const resultsDiv = document.getElementById('collaboration-results');
            const timestamp = new Date().toLocaleTimeString();
            const colorMap = {
                'info': '#0066cc',
                'success': '#28a745', 
                'error': '#dc3545',
                'warning': '#ffc107'
            };
            
            const newContent = `
                <div style="margin: 5px 0; padding: 8px; background: #f8f9fa; border-left: 3px solid ${colorMap[type] || '#0066cc'};">
                    <div style="font-size: 10px; color: #666; margin-bottom: 4px;">[${timestamp}] Multi-Agent System</div>
                    <div>${content}</div>
                </div>
            `;
            
            resultsDiv.innerHTML = newContent + resultsDiv.innerHTML;
        }
        
        async function callMultiAgentTool(operation, params = {}) {
            try {
                // This would integrate with the Agent-Zero tool calling system
                // For now, we'll simulate the response based on the operation
                
                updateResults(`🔄 Executing multi-agent operation: ${operation}...`, 'info');
                
                // Simulate API call delay
                await new Promise(resolve => setTimeout(resolve, 1000));
                
                // Mock responses based on operation type
                let mockResponse;
                switch(operation) {
                    case 'status':
                        mockResponse = {
                            message: "Multi-Agent System Status:\\n• Total Agents: 3\\n• Active Agents: 3\\n• Active Tasks: 0\\n• Completed Tasks: 5\\n• System Status: operational",
                            data: { total_agents: 3, active_agents: 3, system_status: "operational" }
                        };
                        break;
                    case 'agents':
                        mockResponse = {
                            message: "Registered Agents (3):\\n\\n• CognitiveReasoner (cognitive_reasoner_1)\\n  Role: reasoning_specialist\\n  Status: active\\n  Capabilities: logical_reasoning, pattern_matching\\n\\n• CognitiveAnalyzer (cognitive_analyzer_2)\\n  Role: analysis_specialist\\n  Status: active\\n  Capabilities: data_analysis, pattern_recognition",
                            data: { agents: [], total_agents: 3 }
                        };
                        break;
                    case 'collaborate':
                        mockResponse = {
                            message: "Collaborative Reasoning Completed:\\n• Problem: Multi-agent cognitive collaboration optimization\\n• Agent Contributions: 3\\n• Overall Confidence: 0.92\\n• Total Reasoning Steps: 12\\n• Quality Score: 0.95",
                            data: { agent_contributions: 3, overall_confidence: 0.92 }
                        };
                        break;
                    case 'workflow':
                        mockResponse = {
                            message: "End-to-End Multi-Agent Workflow:\\n• Workflow ID: workflow_demo123\\n• Steps Completed: 6/6\\n• Overall Success: ✓\\n\\nWorkflow Steps:\\n  1. ✓ Initialize Environment\\n  2. ✓ Register Agents\\n  3. ✓ Establish Communication\\n  4. ✓ Collaborative Reasoning\\n  5. ✓ Integrate Results\\n  6. ✓ Coordinate Next Actions",
                            data: { steps_completed: 6, total_steps: 6, overall_success: true }
                        };
                        break;
                    default:
                        mockResponse = { message: `Unknown operation: ${operation}`, data: {} };
                }
                
                updateResults(`✅ ${mockResponse.message.replace(/\\n/g, '<br>')}`, 'success');
                return mockResponse;
                
            } catch (error) {
                updateResults(`❌ Error: ${error.message}`, 'error');
                throw error;
            }
        }
        
        async function getAgentStatus() {
            try {
                const response = await callMultiAgentTool('status');
                
                // Update status display
                const statusDiv = document.getElementById('agent-status-content');
                statusDiv.innerHTML = `
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 4px;">
                            <div style="font-size: 24px; font-weight: bold; color: #0066cc;">${response.data.total_agents}</div>
                            <div style="font-size: 12px; color: #666;">Total Agents</div>
                        </div>
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 4px;">
                            <div style="font-size: 24px; font-weight: bold; color: #28a745;">${response.data.active_agents}</div>
                            <div style="font-size: 12px; color: #666;">Active Agents</div>
                        </div>
                        <div style="text-align: center; padding: 10px; background: white; border-radius: 4px;">
                            <div style="font-size: 16px; font-weight: bold; color: #28a745;">●</div>
                            <div style="font-size: 12px; color: #666;">${response.data.system_status}</div>
                        </div>
                    </div>
                `;
                
            } catch (error) {
                console.error('Error getting agent status:', error);
            }
        }
        
        async function listAgents() {
            await callMultiAgentTool('agents');
        }
        
        async function startCollaboration() {
            const problem = prompt("Enter a problem for collaborative reasoning:", 
                                 "How can we improve multi-agent coordination and cognitive collaboration?");
            if (problem) {
                await callMultiAgentTool('collaborate', { problem: problem });
            }
        }
        
        async function runWorkflow() {
            await callMultiAgentTool('workflow');
        }
        
        // Initialize dashboard on load
        document.addEventListener('DOMContentLoaded', function() {
            setTimeout(() => {
                getAgentStatus();
                updateResults('🚀 Multi-Agent Cognitive Collaboration Framework initialized', 'success');
            }, 500);
        });
        
    </script>
    """


def get_multi_agent_status_widget():
    """Get a simple status widget for multi-agent system."""
    
    return """
    <div style="display: inline-block; padding: 8px 12px; background: #e3f2fd; border-radius: 20px; margin: 5px;">
        <span style="color: #1976d2; font-size: 12px; font-weight: 500;">
            🤖 Multi-Agent: 
            <span id="ma-agent-count">3</span> agents • 
            <span id="ma-status" style="color: #4caf50;">●</span> active
        </span>
    </div>
    """


if __name__ == "__main__":
    # Demo the web component
    print("Multi-Agent Web Component Demo")
    print("=" * 40)
    print(multi_agent_dashboard())
    print("\nStatus Widget:")
    print(get_multi_agent_status_widget())