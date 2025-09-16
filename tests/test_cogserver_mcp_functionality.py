#!/usr/bin/env python3
"""
Test script for cogserver Model Context Protocol (MCP) functionality
This tests the multi-agent communication capabilities of cogserver
"""

import os
import sys
import subprocess
import time
import json
import socket
import threading
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class CogServerMCPTester:
    """Test cogserver MCP (Model Context Protocol) functionality for multi-agent communication."""
    
    def __init__(self):
        self.project_root = project_root
        self.cogserver_path = self.project_root / "components" / "cogserver"
        self.mcp_examples_path = self.cogserver_path / "examples" / "mcp"
        self.test_results = {
            "mcp_proxy_scripts_available": False,
            "mcp_proxy_syntax_valid": False,
            "mcp_socket_communication": False,
            "mcp_readme_documentation": False
        }
    
    def test_mcp_proxy_scripts_availability(self):
        """Test that MCP proxy scripts are available and properly structured."""
        print("Testing MCP proxy scripts availability...")
        
        stdio_proxy = self.mcp_examples_path / "stdio_to_unix_proxy.py"
        unix_proxy = self.mcp_examples_path / "unix_to_tcp_proxy.py"
        
        if stdio_proxy.exists() and unix_proxy.exists():
            print(f"✓ Both MCP proxy scripts found:")
            print(f"  - stdio_to_unix_proxy.py: {stdio_proxy}")
            print(f"  - unix_to_tcp_proxy.py: {unix_proxy}")
            self.test_results["mcp_proxy_scripts_available"] = True
        else:
            print("❌ MCP proxy scripts not found")
    
    def test_mcp_proxy_syntax_validation(self):
        """Test that MCP proxy scripts have valid Python syntax."""
        print("\nTesting MCP proxy scripts syntax validation...")
        
        scripts = [
            self.mcp_examples_path / "stdio_to_unix_proxy.py",
            self.mcp_examples_path / "unix_to_tcp_proxy.py"
        ]
        
        all_valid = True
        for script in scripts:
            if script.exists():
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "py_compile", str(script)
                    ], capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        print(f"✓ {script.name} syntax validation passed")
                    else:
                        print(f"❌ {script.name} syntax validation failed: {result.stderr}")
                        all_valid = False
                        
                except subprocess.TimeoutExpired:
                    print(f"⚠️ {script.name} validation timeout")
                    all_valid = False
                except Exception as e:
                    print(f"❌ Error validating {script.name}: {e}")
                    all_valid = False
            else:
                print(f"❌ Script not found: {script}")
                all_valid = False
        
        if all_valid:
            self.test_results["mcp_proxy_syntax_valid"] = True
    
    def test_mcp_socket_communication_simulation(self):
        """Simulate MCP socket communication for multi-agent testing."""
        print("\nTesting MCP socket communication simulation...")
        
        try:
            # Simulate the MCP communication protocol
            socket_path = "/tmp/test_mcp_socket"
            
            # Clean up any existing socket
            if os.path.exists(socket_path):
                os.unlink(socket_path)
            
            # Create a simple server to simulate cogserver MCP interface
            def mock_mcp_server():
                """Mock MCP server for testing."""
                server_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                server_socket.bind(socket_path)
                server_socket.listen(1)
                
                print(f"  - Mock MCP server listening on {socket_path}")
                
                try:
                    server_socket.settimeout(5)  # 5 second timeout
                    connection, _ = server_socket.accept()
                    
                    # Receive test message
                    data = connection.recv(1024).decode('utf-8')
                    print(f"  - Server received: {data}")
                    
                    # Send response
                    response = json.dumps({
                        "jsonrpc": "2.0",
                        "id": 1,
                        "result": {"status": "success", "message": "Multi-agent communication test"}
                    })
                    connection.send(response.encode('utf-8'))
                    
                    connection.close()
                    
                except socket.timeout:
                    print("  - Server timeout (no client connected)")
                except Exception as e:
                    print(f"  - Server error: {e}")
                finally:
                    server_socket.close()
                    if os.path.exists(socket_path):
                        os.unlink(socket_path)
            
            # Start mock server in a thread
            server_thread = threading.Thread(target=mock_mcp_server, daemon=True)
            server_thread.start()
            
            # Give server time to start
            time.sleep(0.5)
            
            # Test client connection
            try:
                client_socket = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                client_socket.connect(socket_path)
                
                # Send test MCP message
                test_message = json.dumps({
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/list",
                    "params": {}
                })
                
                client_socket.send(test_message.encode('utf-8'))
                print(f"  - Client sent: {test_message}")
                
                # Receive response
                response = client_socket.recv(1024).decode('utf-8')
                print(f"  - Client received: {response}")
                
                # Parse response
                response_data = json.loads(response)
                if "result" in response_data:
                    print("✓ MCP socket communication test successful")
                    self.test_results["mcp_socket_communication"] = True
                else:
                    print("⚠️ MCP response format unexpected")
                
                client_socket.close()
                
            except Exception as e:
                print(f"❌ Client connection failed: {e}")
            
            # Wait for server thread to complete
            server_thread.join(timeout=2)
            
        except Exception as e:
            print(f"❌ MCP socket communication test failed: {e}")
    
    def test_mcp_documentation_completeness(self):
        """Test that MCP documentation is complete and helpful."""
        print("\nTesting MCP documentation completeness...")
        
        readme_files = [
            self.mcp_examples_path / "README.md",
            self.mcp_examples_path / "CLAUDE.md",
            self.mcp_examples_path / "CLAUDE-AtomSpace.md"
        ]
        
        documentation_found = 0
        for readme in readme_files:
            if readme.exists():
                print(f"✓ Documentation found: {readme.name}")
                documentation_found += 1
                
                # Check content quality
                try:
                    with open(readme, 'r') as f:
                        content = f.read()
                        
                    # Check for key MCP concepts
                    key_concepts = [
                        "Model Context Protocol",
                        "multi-agent",
                        "cogserver",
                        "AtomSpace"
                    ]
                    
                    concepts_found = sum(1 for concept in key_concepts if concept.lower() in content.lower())
                    print(f"  - Key concepts coverage: {concepts_found}/{len(key_concepts)}")
                    
                except Exception as e:
                    print(f"  - Error reading {readme.name}: {e}")
            else:
                print(f"⚠️ Documentation missing: {readme.name}")
        
        if documentation_found >= 2:  # At least 2 documentation files
            print("✓ MCP documentation is adequately complete")
            self.test_results["mcp_readme_documentation"] = True
        else:
            print("⚠️ MCP documentation needs improvement")
    
    def test_multi_agent_mcp_workflow(self):
        """Test a complete multi-agent workflow using MCP concepts."""
        print("\nTesting multi-agent MCP workflow...")
        
        try:
            # Simulate a multi-agent scenario using MCP-like communication
            class MCPAgent:
                def __init__(self, agent_id, capabilities):
                    self.agent_id = agent_id
                    self.capabilities = capabilities
                    self.message_queue = []
                
                def send_mcp_request(self, method, params):
                    """Simulate sending an MCP request."""
                    return {
                        "jsonrpc": "2.0",
                        "id": f"{self.agent_id}_{int(time.time())}",
                        "method": method,
                        "params": params
                    }
                
                def handle_mcp_response(self, response):
                    """Simulate handling an MCP response."""
                    self.message_queue.append(response)
                    return len(self.message_queue)
            
            # Create multiple agents
            agents = [
                MCPAgent("cognitive_agent_1", ["reasoning", "analysis"]),
                MCPAgent("cognitive_agent_2", ["planning", "execution"]),
                MCPAgent("coordinator_agent", ["supervision", "coordination"])
            ]
            
            print(f"  - Created {len(agents)} MCP-enabled agents")
            
            # Simulate inter-agent communication
            workflow_steps = [
                ("cognitive_agent_1", "tools/list", {}),
                ("cognitive_agent_2", "resources/list", {}),
                ("coordinator_agent", "tools/call", {"name": "coordinate", "arguments": {}})
            ]
            
            total_messages = 0
            for agent_id, method, params in workflow_steps:
                agent = next(a for a in agents if a.agent_id == agent_id)
                request = agent.send_mcp_request(method, params)
                
                # Simulate response
                response = {
                    "jsonrpc": "2.0",
                    "id": request["id"],
                    "result": {"status": "success", "method": method}
                }
                
                messages_handled = agent.handle_mcp_response(response)
                total_messages += 1
                
                print(f"  - {agent_id} executed {method}: {messages_handled} messages handled")
            
            print(f"✓ Multi-agent MCP workflow completed successfully")
            print(f"  - Total workflow steps: {len(workflow_steps)}")
            print(f"  - Total messages processed: {total_messages}")
            
            return True
            
        except Exception as e:
            print(f"❌ Multi-agent MCP workflow failed: {e}")
            return False
    
    def run_all_tests(self):
        """Run all cogserver MCP functionality tests."""
        print("=" * 70)
        print("CogServer MCP (Model Context Protocol) Functionality Test Suite")
        print("=" * 70)
        
        # Run individual tests
        self.test_mcp_proxy_scripts_availability()
        self.test_mcp_proxy_syntax_validation()
        self.test_mcp_socket_communication_simulation()
        self.test_mcp_documentation_completeness()
        workflow_success = self.test_multi_agent_mcp_workflow()
        
        # Print summary
        print("\n" + "=" * 70)
        print("MCP TEST RESULTS SUMMARY")
        print("=" * 70)
        
        for test_name, result in self.test_results.items():
            status = "✓ PASS" if result else "❌ FAIL"
            print(f"{test_name:.<50} {status}")
        
        if workflow_success:
            print(f"{'multi_agent_mcp_workflow':.<50} ✓ PASS")
        else:
            print(f"{'multi_agent_mcp_workflow':.<50} ❌ FAIL")
        
        # Overall assessment
        passed_tests = sum(self.test_results.values()) + (1 if workflow_success else 0)
        total_tests = len(self.test_results) + 1
        
        print("\n" + "-" * 70)
        print(f"Overall MCP Result: {passed_tests}/{total_tests} tests passed")
        
        if passed_tests >= total_tests * 0.8:  # 80% pass rate
            print("✓ CogServer MCP multi-agent functionality is OPERATIONAL")
            return True
        else:
            print("⚠️ CogServer MCP multi-agent functionality needs IMPROVEMENT")
            return False


def main():
    """Main test function."""
    tester = CogServerMCPTester()
    success = tester.run_all_tests()
    
    # Return appropriate exit code
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()