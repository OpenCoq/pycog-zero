"""
Integration Tests for PyCog-Zero cpp2py Conversion Pipeline
===========================================================

Test suite for validating OpenCog component integration and conversion.
"""

import pytest
import asyncio
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Add project root to Python path
import sys
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.cpp2py_conversion_pipeline import CPP2PyConversionPipeline, Phase


class TestCPP2PyConversionPipeline:
    """Test class for conversion pipeline functionality."""
    
    @pytest.fixture(scope="class")
    def pipeline(self, tmp_path_factory):
        """Create test pipeline instance with temporary directory."""
        test_root = tmp_path_factory.mktemp("cpp2py_test")
        return CPP2PyConversionPipeline(str(test_root))
    
    def test_pipeline_initialization(self, pipeline):
        """Test that pipeline initializes correctly."""
        assert pipeline.root_dir.exists()
        assert pipeline.components_dir.exists()
        assert pipeline.tests_dir.exists()
        assert pipeline.docs_dir.exists()
        
        # Check that required directories were created
        required_dirs = [
            "components/core",
            "components/logic", 
            "components/cognitive",
            "components/advanced",
            "components/language",
            "tests/integration",
            "tests/performance",
            "tests/end_to_end",
            "docs/components",
            "docs/integration"
        ]
        
        for dir_path in required_dirs:
            full_path = pipeline.root_dir / dir_path
            assert full_path.exists(), f"Missing directory: {dir_path}"
    
    def test_component_definitions_loaded(self, pipeline):
        """Test that component definitions are properly loaded."""
        assert len(pipeline.components) > 0
        
        # Check for key components
        key_components = ["cogutil", "atomspace", "ure", "pln", "opencog"]
        for comp in key_components:
            assert comp in pipeline.components, f"Missing component: {comp}"
    
    def test_component_phase_assignments(self, pipeline):
        """Test that components are assigned to correct phases."""
        # Phase 0: Foundation
        cogutil = pipeline.components["cogutil"]
        assert cogutil.phase == Phase.PHASE_0_FOUNDATION
        
        # Phase 1: Core Extensions
        atomspace = pipeline.components["atomspace"] 
        assert atomspace.phase == Phase.PHASE_1_CORE_EXTENSIONS
        
        # Phase 5: Language & Integration
        opencog = pipeline.components["opencog"]
        assert opencog.phase == Phase.PHASE_5_LANGUAGE_INTEGRATION
    
    def test_dependency_validation(self, pipeline):
        """Test dependency validation logic."""
        # cogutil should have no dependencies
        assert pipeline.components["cogutil"].dependencies == []
        
        # atomspace should depend on cogutil
        assert "cogutil" in pipeline.components["atomspace"].dependencies
        
        # ure should depend on atomspace and unify
        ure_deps = pipeline.components["ure"].dependencies
        assert "atomspace" in ure_deps
        assert "unify" in ure_deps
    
    def test_generate_phase_report(self, pipeline):
        """Test phase report generation."""
        report = pipeline.generate_phase_report(Phase.PHASE_0_FOUNDATION)
        
        assert "phase" in report
        assert "total_components" in report
        assert "components" in report
        assert report["phase"] == Phase.PHASE_0_FOUNDATION.value
        assert report["total_components"] >= 1  # At least cogutil
        
        # Check cogutil is in the report
        assert "cogutil" in report["components"]


class TestComponentIntegration:
    """Test integration of individual components."""
    
    def test_cogutil_integration_ready(self):
        """Test that cogutil integration requirements are met."""
        # This would test actual cogutil integration when component is cloned
        # For now, just test the definition
        pipeline = CPP2PyConversionPipeline()
        cogutil = pipeline.components["cogutil"]
        
        assert cogutil.repository == "https://github.com/opencog/cogutil"
        assert cogutil.priority == "HIGH"
        assert len(cogutil.tasks) > 0
        assert len(cogutil.deliverables) > 0
    
    def test_atomspace_integration_ready(self):
        """Test that atomspace integration requirements are met."""
        pipeline = CPP2PyConversionPipeline()
        atomspace = pipeline.components["atomspace"]
        
        assert atomspace.repository == "https://github.com/opencog/atomspace"
        assert "cogutil" in atomspace.dependencies
        assert atomspace.priority == "HIGH"
    
    def test_pln_integration_ready(self):
        """Test that PLN integration requirements are met."""
        pipeline = CPP2PyConversionPipeline()
        pln = pipeline.components["pln"]
        
        assert pln.repository == "https://github.com/opencog/pln"
        assert "atomspace" in pln.dependencies
        assert "ure" in pln.dependencies
        assert pln.phase == Phase.PHASE_4_ADVANCED_LEARNING


class TestPyCogZeroIntegration:
    """Test integration with existing PyCog-Zero cognitive tools."""
    
    def test_cognitive_reasoning_tool_exists(self):
        """Test that cognitive reasoning tool is available."""
        tool_path = project_root / "python" / "tools" / "cognitive_reasoning.py"
        assert tool_path.exists(), "Cognitive reasoning tool not found"
        
        # Test that we can import it
        try:
            from python.tools.cognitive_reasoning import CognitiveReasoningTool
            assert CognitiveReasoningTool is not None
        except ImportError as e:
            pytest.skip(f"Cannot import cognitive reasoning tool: {e}")
    
    def test_cognitive_config_exists(self):
        """Test that cognitive configuration exists."""
        config_path = project_root / "conf" / "config_cognitive.json"
        assert config_path.exists(), "Cognitive configuration not found"
        
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        assert config.get("cognitive_mode") is True
        assert config.get("opencog_enabled") is True
        assert "cognitive_tools" in config
    
    @pytest.mark.asyncio
    async def test_cognitive_tool_basic_execution(self):
        """Test basic execution of cognitive reasoning tool."""
        try:
            from python.tools.cognitive_reasoning import CognitiveReasoningTool
            
            # Create mock agent
            class MockAgent:
                pass
            
            tool = CognitiveReasoningTool(MockAgent())
            
            # Test basic execution (may fail if OpenCog not installed)
            result = await tool.execute("test query")
            assert result is not None
            assert hasattr(result, 'message')
            
        except ImportError:
            pytest.skip("OpenCog dependencies not available")
        except Exception as e:
            # Expected if OpenCog not properly installed
            assert "OpenCog" in str(e) or "cognitive reasoning not available" in str(e)


class TestBuildSystem:
    """Test build system integration."""
    
    def test_requirements_cognitive_exists(self):
        """Test that cognitive requirements file exists."""
        req_path = project_root / "requirements-cognitive.txt"
        assert req_path.exists(), "Cognitive requirements file not found"
        
        with open(req_path, 'r') as f:
            content = f.read()
        
        # Check for key cognitive dependencies
        assert "opencog-atomspace" in content
        assert "torch" in content
        assert "pytest" in content
    
    def test_conversion_pipeline_executable(self):
        """Test that the conversion pipeline script is executable."""
        script_path = project_root / "scripts" / "cpp2py_conversion_pipeline.py"
        assert script_path.exists(), "Conversion pipeline script not found"
        
        # Test that it can be executed
        result = subprocess.run([
            "python", str(script_path), "--help"
        ], capture_output=True, text=True)
        
        assert result.returncode == 0
        assert "PyCog-Zero cpp2py Conversion Pipeline" in result.stdout


class TestEndToEndWorkflow:
    """Test end-to-end conversion workflow."""
    
    def test_pipeline_status_command(self):
        """Test pipeline status command."""
        script_path = project_root / "scripts" / "cpp2py_conversion_pipeline.py"
        
        result = subprocess.run([
            "python", str(script_path), "status"
        ], capture_output=True, text=True, cwd=project_root)
        
        assert result.returncode == 0
        assert "Overall Status" in result.stdout
        assert "phase_0_foundation" in result.stdout
    
    def test_component_validation_command(self):
        """Test component validation command."""
        script_path = project_root / "scripts" / "cpp2py_conversion_pipeline.py"
        
        result = subprocess.run([
            "python", str(script_path), "validate", "cogutil"
        ], capture_output=True, text=True, cwd=project_root)
        
        # Should fail since components aren't cloned yet, but command should work
        assert result.returncode in [0, 1]  # 0 if deps found, 1 if missing (expected)


@pytest.mark.integration
class TestActualComponentCloning:
    """Test actual component cloning (requires network access)."""
    
    @pytest.mark.slow
    def test_clone_cogutil_component(self, tmp_path):
        """Test cloning cogutil component (slow test requiring network)."""
        pipeline = CPP2PyConversionPipeline(str(tmp_path))
        
        # This is a slow test that requires network access
        # Skip in CI unless specifically requested
        if not os.getenv("RUN_SLOW_TESTS"):
            pytest.skip("Slow test skipped (set RUN_SLOW_TESTS=1 to run)")
        
        success = pipeline.clone_component("cogutil")
        assert success
        
        # Check that component was cloned
        cogutil_dir = pipeline.components_dir / "cogutil"
        assert cogutil_dir.exists()
        
        # Check that git headers were removed
        git_dir = cogutil_dir / ".git"
        assert not git_dir.exists()
        
        # Check that status file was created
        status_file = cogutil_dir / "conversion_status.json"
        assert status_file.exists()
        
        with open(status_file, 'r') as f:
            status = json.load(f)
        
        assert status["component"] == "cogutil"
        assert status["status"] == "cloned"


if __name__ == "__main__":
    pytest.main([__file__])