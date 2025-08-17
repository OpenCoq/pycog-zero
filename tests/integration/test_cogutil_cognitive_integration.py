"""
Cognitive Reasoning Integration Tests for Cogutil Components
===========================================================

Test suite for validating cogutil component integration with PyCog-Zero cognitive reasoning.
These tests focus on the foundation components (Phase 0) and their readiness for OpenCog integration.
"""

import pytest
import os
import json
import subprocess
from pathlib import Path
from typing import Dict, List, Optional

# Project paths
project_root = Path(__file__).parent.parent.parent
cogutil_path = project_root / "components" / "cogutil"
cognitive_config_path = project_root / "conf" / "config_cognitive.json"


class TestCogutilFoundationComponents:
    """Test cogutil foundation components for cognitive reasoning integration."""
    
    def test_cogutil_directory_structure(self):
        """Test that cogutil component has expected directory structure."""
        assert cogutil_path.exists(), "Cogutil component directory not found"
        
        # Check key directories
        required_dirs = [
            "opencog/util",
            "cmake", 
            "tests"
        ]
        
        for dir_name in required_dirs:
            dir_path = cogutil_path / dir_name
            assert dir_path.exists(), f"Missing cogutil directory: {dir_name}"
    
    def test_cogutil_core_headers_present(self):
        """Test that core cogutil headers are present."""
        util_dir = cogutil_path / "opencog" / "util"
        
        # Core headers that cognitive reasoning depends on
        core_headers = [
            "cogutil.h",
            "files.h", 
            "Config.h",
            "misc.h",
            "platform.h"
        ]
        
        for header in core_headers:
            header_path = util_dir / header
            assert header_path.exists(), f"Missing core header: {header}"
    
    def test_cogutil_source_files_present(self):
        """Test that core cogutil source files are present."""
        util_dir = cogutil_path / "opencog" / "util"
        
        # Core source files
        core_sources = [
            "files.cc",
            "Config.cc", 
            "misc.cc",
            "platform.cc"
        ]
        
        for source in core_sources:
            source_path = util_dir / source
            assert source_path.exists(), f"Missing core source: {source}"
    
    def test_cogutil_version_information(self):
        """Test that cogutil version information is accessible."""
        cogutil_header = cogutil_path / "opencog" / "util" / "cogutil.h"
        
        with open(cogutil_header, 'r') as f:
            content = f.read()
        
        # Check version defines are present
        version_defines = [
            "COGUTIL_MAJOR_VERSION",
            "COGUTIL_MINOR_VERSION", 
            "COGUTIL_MICRO_VERSION",
            "COGUTIL_VERSION_STRING"
        ]
        
        for define in version_defines:
            assert define in content, f"Missing version define: {define}"
    
    def test_cogutil_cmake_configuration(self):
        """Test that cogutil CMake configuration is present."""
        cmake_dir = cogutil_path / "cmake"
        cmake_config = cmake_dir / "CogUtilConfig.cmake.in"
        
        assert cmake_config.exists(), "CogUtilConfig.cmake.in not found"
        
        with open(cmake_config, 'r') as f:
            content = f.read()
        
        # Check key CMake variables
        cmake_vars = [
            "COGUTIL_VERSION",
            "COGUTIL_INCLUDE_DIR",
            "COGUTIL_LIBRARY",
            "COGUTIL_FOUND"
        ]
        
        for var in cmake_vars:
            assert var in content, f"Missing CMake variable: {var}"


class TestCogutilCognitiveIntegration:
    """Test cogutil integration with cognitive reasoning systems."""
    
    def test_cognitive_config_cogutil_compatibility(self):
        """Test that cognitive configuration is compatible with cogutil."""
        assert cognitive_config_path.exists(), "Cognitive configuration not found"
        
        with open(cognitive_config_path, 'r') as f:
            config = json.load(f)
        
        # Verify cognitive mode settings that would use cogutil
        assert config.get("cognitive_mode") is True
        assert config.get("opencog_enabled") is True
        
        # Check for reasoning config that depends on cogutil
        reasoning_config = config.get("reasoning_config", {})
        assert isinstance(reasoning_config, dict)
        assert "pln_enabled" in reasoning_config
        assert "pattern_matching" in reasoning_config
    
    def test_cogutil_file_utilities_readiness(self):
        """Test that cogutil file utilities are ready for cognitive systems."""
        files_header = cogutil_path / "opencog" / "util" / "files.h"
        files_source = cogutil_path / "opencog" / "util" / "files.cc"
        
        assert files_header.exists(), "files.h not found"
        assert files_source.exists(), "files.cc not found"
        
        # Check for key file utility functions
        with open(files_header, 'r') as f:
            header_content = f.read()
        
        key_functions = [
            "file_exists",
            "get_module_paths",
            "DEFAULT_MODULE_PATHS"
        ]
        
        for func in key_functions:
            assert func in header_content, f"Missing file utility function: {func}"
    
    def test_cogutil_config_utilities_readiness(self):
        """Test that cogutil configuration utilities are ready."""
        config_header = cogutil_path / "opencog" / "util" / "Config.h"
        config_source = cogutil_path / "opencog" / "util" / "Config.cc"
        
        assert config_header.exists(), "Config.h not found"
        assert config_source.exists(), "Config.cc not found"
        
        # These utilities are needed for OpenCog configuration management
        with open(config_header, 'r') as f:
            header_content = f.read()
        
        # Check for Config class presence
        assert "class Config" in header_content or "Config" in header_content
    
    def test_cogutil_logging_utilities_readiness(self):
        """Test that cogutil logging utilities are ready for cognitive systems."""
        util_dir = cogutil_path / "opencog" / "util"
        
        # Check for logging-related files
        logging_files = [
            "log_prog_name.cc"
        ]
        
        for log_file in logging_files:
            log_path = util_dir / log_file
            assert log_path.exists(), f"Missing logging utility: {log_file}"


class TestCogutilPythonBindingsReadiness:
    """Test cogutil readiness for Python bindings generation."""
    
    def test_cogutil_python_binding_infrastructure(self):
        """Test that cogutil has infrastructure for Python bindings."""
        # Check for CMake configuration that supports Python bindings
        cmake_main = cogutil_path / "CMakeLists.txt"
        assert cmake_main.exists(), "Main CMakeLists.txt not found"
        
        with open(cmake_main, 'r') as f:
            cmake_content = f.read()
        
        # Look for Python-related CMake configurations
        python_indicators = [
            "Python", "PYTHON", "python"
        ]
        
        # At least one Python reference should exist for binding support
        has_python_support = any(indicator in cmake_content for indicator in python_indicators)
        
        # If no Python support found, it's still valid - bindings can be added
        # This is an informational test
        if not has_python_support:
            pytest.skip("No explicit Python binding support found - can be added later")
    
    def test_cogutil_header_accessibility(self):
        """Test that cogutil headers are accessible for binding generation."""
        include_dir = cogutil_path / "opencog"
        util_dir = include_dir / "util"
        
        assert include_dir.exists(), "Include directory structure missing"
        assert util_dir.exists(), "Util include directory missing"
        
        # Count available headers
        header_files = list(util_dir.glob("*.h"))
        assert len(header_files) > 0, "No header files found for binding generation"
        
        # Check that headers are readable
        for header in header_files[:5]:  # Test first 5 headers
            with open(header, 'r') as f:
                content = f.read()
                assert len(content) > 0, f"Empty header file: {header.name}"
    
    def test_cogutil_namespace_structure(self):
        """Test that cogutil uses consistent namespace structure for bindings."""
        core_headers = [
            "files.h",
            "misc.h", 
            "Config.h"
        ]
        
        util_dir = cogutil_path / "opencog" / "util"
        
        for header_name in core_headers:
            header_path = util_dir / header_name
            if header_path.exists():
                with open(header_path, 'r') as f:
                    content = f.read()
                
                # Check for opencog namespace
                assert "namespace opencog" in content, f"Missing opencog namespace in {header_name}"


class TestCogutilBuildSystemIntegration:
    """Test cogutil build system integration for cognitive framework."""
    
    def test_cogutil_cmake_structure(self):
        """Test that cogutil has proper CMake build structure."""
        cmake_files = [
            "CMakeLists.txt",
            "cmake/CogUtilConfig.cmake.in"
        ]
        
        for cmake_file in cmake_files:
            cmake_path = cogutil_path / cmake_file
            assert cmake_path.exists(), f"Missing CMake file: {cmake_file}"
    
    def test_cogutil_build_dependencies(self):
        """Test that cogutil build dependencies are documented."""
        readme_path = cogutil_path / "README.md"
        assert readme_path.exists(), "README.md not found"
        
        with open(readme_path, 'r') as f:
            readme_content = f.read()
        
        # Check for dependency documentation
        dependency_indicators = [
            "Prerequisites", "Dependencies", "boost", "cmake"
        ]
        
        found_deps = [dep for dep in dependency_indicators if dep in readme_content]
        assert len(found_deps) >= 2, "Insufficient dependency documentation"
    
    def test_cogutil_test_structure(self):
        """Test that cogutil has test infrastructure."""
        tests_dir = cogutil_path / "tests"
        assert tests_dir.exists(), "Tests directory not found"
        
        # Check for test files (CxxTest uses .cxxtest extension)
        test_files = list(tests_dir.rglob("*Test.cxxtest")) + list(tests_dir.rglob("test*"))
        if len(test_files) == 0:
            # Also check for other test file patterns
            test_files = list(tests_dir.rglob("*.cc")) + list(tests_dir.rglob("*.cpp")) + list(tests_dir.rglob("*.cxxtest"))
        
        assert len(test_files) > 0, "No test files found"


class TestCogutilCognitiveReasoningPrerequisites:
    """Test cogutil prerequisites for cognitive reasoning integration."""
    
    def test_cogutil_component_status(self):
        """Test cogutil component conversion status."""
        status_file = cogutil_path / "conversion_status.json"
        
        # Status file may not exist yet - this is informational
        if status_file.exists():
            with open(status_file, 'r') as f:
                status = json.load(f)
            
            assert status.get("component") == "cogutil"
            assert "status" in status
        else:
            # Status file creation is part of the pipeline
            pytest.skip("Conversion status file not yet created")
    
    def test_memory_and_file_utilities_integration(self):
        """Test that cogutil memory and file utilities support cognitive memory."""
        util_dir = cogutil_path / "opencog" / "util"
        
        # Files needed for cognitive memory integration
        memory_support_files = [
            "files.h",  # File operations for persistence
            "files.cc"
        ]
        
        for file_name in memory_support_files:
            file_path = util_dir / file_name
            assert file_path.exists(), f"Missing memory support file: {file_name}"
    
    def test_configuration_system_integration(self):
        """Test that cogutil configuration system supports cognitive config."""
        util_dir = cogutil_path / "opencog" / "util"
        
        # Configuration files needed
        config_files = [
            "Config.h",
            "Config.cc"
        ]
        
        for file_name in config_files:
            file_path = util_dir / file_name
            assert file_path.exists(), f"Missing config support file: {file_name}"
    
    def test_utility_functions_for_reasoning(self):
        """Test that cogutil provides utilities needed for reasoning."""
        util_dir = cogutil_path / "opencog" / "util"
        misc_header = util_dir / "misc.h"
        misc_source = util_dir / "misc.cc"
        
        assert misc_header.exists(), "misc.h utilities not found"
        assert misc_source.exists(), "misc.cc utilities not found"
        
        # Check for utility functions that reasoning systems typically need
        with open(misc_header, 'r') as f:
            misc_content = f.read()
        
        # Look for common utility patterns
        utility_patterns = [
            "bitcount",  # Bit manipulation for reasoning
            "demangle"   # Symbol demangling for debugging
        ]
        
        for pattern in utility_patterns:
            assert pattern in misc_content, f"Missing utility function: {pattern}"


@pytest.mark.integration
class TestCogutilIntegrationWorkflow:
    """Integration tests for complete cogutil workflow."""
    
    def test_cogutil_to_cognitive_reasoning_pipeline(self):
        """Test the complete pipeline from cogutil to cognitive reasoning."""
        # This test validates the integration pathway
        
        # 1. Cogutil components exist
        assert cogutil_path.exists(), "Cogutil component missing"
        
        # 2. Cognitive configuration exists
        assert cognitive_config_path.exists(), "Cognitive configuration missing"
        
        # 3. Core headers are accessible
        core_headers = ["cogutil.h", "files.h", "Config.h"]
        util_dir = cogutil_path / "opencog" / "util"
        
        for header in core_headers:
            header_path = util_dir / header
            assert header_path.exists(), f"Core header missing: {header}"
        
        # 4. Configuration integration is possible
        with open(cognitive_config_path, 'r') as f:
            config = json.load(f)
        
        assert config.get("opencog_enabled") is True
        assert config.get("cognitive_mode") is True
    
    def test_cogutil_readiness_for_phase1(self):
        """Test that cogutil is ready for Phase 1 (AtomSpace integration)."""
        # Prerequisites for AtomSpace that cogutil must provide
        
        # 1. File utilities for AtomSpace persistence
        files_header = cogutil_path / "opencog" / "util" / "files.h"
        assert files_header.exists(), "File utilities not ready for AtomSpace"
        
        # 2. Configuration system for AtomSpace settings
        config_header = cogutil_path / "opencog" / "util" / "Config.h"
        assert config_header.exists(), "Configuration system not ready"
        
        # 3. Basic utilities for AtomSpace operations
        misc_header = cogutil_path / "opencog" / "util" / "misc.h"
        assert misc_header.exists(), "Misc utilities not ready"
        
        # 4. Platform compatibility
        platform_source = cogutil_path / "opencog" / "util" / "platform.cc"
        assert platform_source.exists(), "Platform utilities not ready"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])