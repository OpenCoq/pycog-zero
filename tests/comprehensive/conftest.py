"""
Pytest configuration for comprehensive PyCog-Zero test suite.
"""

import pytest
import asyncio
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Test configuration
os.environ["PYCOG_ZERO_TEST_MODE"] = "1"


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture(scope="session")
def test_environment():
    """Setup test environment for comprehensive tests."""
    # Create test results directory
    os.makedirs(PROJECT_ROOT / "test_results", exist_ok=True)
    
    # Configure test environment
    test_config = {
        "cognitive_mode": True,
        "opencog_enabled": False,  # Default to false for CI/testing
        "test_mode": True,
        "performance_tests": os.environ.get("PERFORMANCE_TESTS", "false").lower() == "true"
    }
    
    return test_config


@pytest.fixture
def mock_agent():
    """Create a mock Agent-Zero instance for testing."""
    class MockAgent:
        def __init__(self):
            self.agent_name = "test_agent"
            self.capabilities = ["cognitive_reasoning", "memory", "metacognition"]
            self.tools = []
        
        def get_capabilities(self):
            return self.capabilities
        
        def get_tools(self):
            return self.tools
    
    return MockAgent()


# Pytest markers
pytest_plugins = []

# Configure markers
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", 
        "asyncio: mark test as async"
    )
    config.addinivalue_line(
        "markers",
        "performance: mark test as performance benchmark"
    )
    config.addinivalue_line(
        "markers",
        "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers",
        "system: mark test as system test"
    )


# Test collection configuration
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add appropriate markers."""
    for item in items:
        # Add asyncio marker to async tests
        if asyncio.iscoroutinefunction(item.function):
            item.add_marker(pytest.mark.asyncio)
        
        # Add markers based on test file
        if "test_performance" in str(item.fspath):
            item.add_marker(pytest.mark.performance)
        elif "test_integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "test_system" in str(item.fspath):
            item.add_marker(pytest.mark.system)


# Test reporting hooks
def pytest_runtest_setup(item):
    """Setup hook for each test."""
    # Skip performance tests if not explicitly enabled
    if "performance" in [mark.name for mark in item.iter_markers()]:
        if not os.environ.get("PERFORMANCE_TESTS", "false").lower() == "true":
            pytest.skip("Performance tests disabled (set PERFORMANCE_TESTS=true to enable)")


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    """Add custom terminal summary."""
    if hasattr(terminalreporter, 'stats'):
        passed = len(terminalreporter.stats.get('passed', []))
        failed = len(terminalreporter.stats.get('failed', []))
        skipped = len(terminalreporter.stats.get('skipped', []))
        
        terminalreporter.write_sep("=", "PyCog-Zero Comprehensive Test Summary")
        terminalreporter.write_line(f"Passed: {passed}")
        terminalreporter.write_line(f"Failed: {failed}")
        terminalreporter.write_line(f"Skipped: {skipped}")
        
        if failed == 0 and passed > 0:
            terminalreporter.write_line("✅ All comprehensive tests passed!")
        elif failed > 0:
            terminalreporter.write_line("❌ Some tests failed - check individual test reports")