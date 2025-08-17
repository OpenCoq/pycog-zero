#!/usr/bin/env python3
"""
PyCog-Zero cpp2py Conversion Pipeline
=====================================

Main script for managing the OpenCog C++ to Python conversion pipeline.
Implements the 20-week roadmap for systematic component integration.
"""

import os
import sys
import json
import subprocess
import shutil
import logging
from pathlib import Path
from typing import Dict, List, Optional
from dataclasses import dataclass
from enum import Enum

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase(Enum):
    """Development phases for OpenCog component integration."""
    PHASE_0_FOUNDATION = "phase_0_foundation" 
    PHASE_1_CORE_EXTENSIONS = "phase_1_core_extensions"
    PHASE_2_LOGIC_SYSTEMS = "phase_2_logic_systems"
    PHASE_3_COGNITIVE_SYSTEMS = "phase_3_cognitive_systems"
    PHASE_4_ADVANCED_LEARNING = "phase_4_advanced_learning"
    PHASE_5_LANGUAGE_INTEGRATION = "phase_5_language_integration"

@dataclass
class Component:
    """OpenCog component definition."""
    name: str
    repository: str
    phase: Phase
    dependencies: List[str]
    priority: str  # HIGH, MEDIUM, LOW, CRITICAL
    tasks: List[str]
    deliverables: List[str]

class CPP2PyConversionPipeline:
    """Main conversion pipeline manager."""
    
    def __init__(self, root_dir: str = None):
        self.root_dir = Path(root_dir or os.getcwd())
        self.components_dir = self.root_dir / "components"
        self.scripts_dir = self.root_dir / "scripts"  
        self.tests_dir = self.root_dir / "tests"
        self.docs_dir = self.root_dir / "docs"
        self.config_file = self.root_dir / "cpp2py_config.json"
        
        # Load component definitions
        self.components = self._load_component_definitions()
        
        # Create required directories
        self._ensure_directories()
    
    def _load_component_definitions(self) -> Dict[str, Component]:
        """Load OpenCog component definitions based on the roadmap."""
        return {
            # Phase 0: Foundation Layer
            "cogutil": Component(
                name="cogutil",
                repository="https://github.com/opencog/cogutil",
                phase=Phase.PHASE_0_FOUNDATION,
                dependencies=[],
                priority="HIGH",
                tasks=[
                    "Clone cogutil repository",
                    "Analyze build dependencies and requirements", 
                    "Create Python bindings for core utilities",
                    "Integrate into PyCog-Zero build system",
                    "Create utility integration tests",
                    "Documentation and validation"
                ],
                deliverables=[
                    "cogutil integrated into build system",
                    "Python utility wrappers functional", 
                    "Core utility tests passing",
                    "Updated dependency configuration"
                ]
            ),
            
            # Phase 1: Core Extensions
            "atomspace": Component(
                name="atomspace", 
                repository="https://github.com/opencog/atomspace",
                phase=Phase.PHASE_1_CORE_EXTENSIONS,
                dependencies=["cogutil"],
                priority="HIGH",
                tasks=[
                    "Clone atomspace repository",
                    "Analyze AtomSpace core architecture",
                    "Create Python AtomSpace bindings", 
                    "Integrate with PyCog-Zero memory system",
                    "Build comprehensive AtomSpace tests",
                    "Performance optimization and validation"
                ],
                deliverables=[
                    "AtomSpace integrated into PyCog-Zero",
                    "Hypergraph storage functional",
                    "Python bindings working", 
                    "Memory integration complete"
                ]
            ),
            
            "atomspace-rocks": Component(
                name="atomspace-rocks",
                repository="https://github.com/opencog/atomspace-rocks", 
                phase=Phase.PHASE_1_CORE_EXTENSIONS,
                dependencies=["atomspace"],
                priority="HIGH",
                tasks=[
                    "Clone atomspace-rocks repository",
                    "Analyze RocksDB storage backend",
                    "Configure RocksDB dependencies",
                    "Integrate into build system with atomspace dependency",
                    "Build and resolve compilation issues",
                    "Create integration tests and documentation"
                ],
                deliverables=[
                    "atomspace-rocks integrated into build system",
                    "RocksDB storage backend functional", 
                    "Integration tests passing",
                    "Updated CMake configuration"
                ]
            ),
            
            "cogserver": Component(
                name="cogserver", 
                repository="https://github.com/opencog/cogserver",
                phase=Phase.PHASE_1_CORE_EXTENSIONS,
                dependencies=["atomspace"],
                priority="HIGH",
                tasks=[
                    "Clone cogserver repository",
                    "Analyze cognitive server architecture",
                    "Create Python server bindings",
                    "Integrate with Agent-Zero framework", 
                    "Test multi-agent server functionality",
                    "Server deployment and validation"
                ],
                deliverables=[
                    "cogserver integrated with atomspace",
                    "Multi-agent server functional",
                    "Agent-Zero integration complete",
                    "Server deployment tests passing"
                ]
            ),
            
            # Phase 2: Logic Systems  
            "unify": Component(
                name="unify",
                repository="https://github.com/opencog/unify",
                phase=Phase.PHASE_2_LOGIC_SYSTEMS, 
                dependencies=["atomspace"],
                priority="HIGH",
                tasks=[
                    "Clone unify repository",
                    "Analyze unification algorithm implementation",
                    "Configure pattern matching dependencies",
                    "Integrate into build system",
                    "Test unification algorithms", 
                    "Create comprehensive unification test suite"
                ],
                deliverables=[
                    "unify integrated into build system",
                    "Pattern unification working",
                    "Unification test suite passing",
                    "Integration with atomspace validated"
                ]
            ),
            
            "ure": Component(
                name="ure",
                repository="https://github.com/opencog/ure", 
                phase=Phase.PHASE_2_LOGIC_SYSTEMS,
                dependencies=["atomspace", "unify"],
                priority="HIGH", 
                tasks=[
                    "Clone ure (Unified Rule Engine) repository", 
                    "Analyze backward/forward chaining dependencies",
                    "Configure rule engine components",
                    "Integrate with unify dependency",
                    "Test rule engine functionality",
                    "Create rule execution test cases"
                ],
                deliverables=[
                    "ure integrated with unify dependency",
                    "Forward/backward chaining functional", 
                    "Rule execution tests passing",
                    "Integration with reasoning systems"
                ]
            ),
            
            # Phase 3: Cognitive Systems
            "attention": Component(
                name="attention",
                repository="https://github.com/opencog/attention",
                phase=Phase.PHASE_3_COGNITIVE_SYSTEMS,
                dependencies=["atomspace", "cogserver"], 
                priority="HIGH",
                tasks=[
                    "Clone attention repository", 
                    "Analyze attention allocation algorithms",
                    "Configure ECAN (Economic Attention Networks)",
                    "Integrate with cogserver dependency", 
                    "Test attention spreading algorithms",
                    "Validate attention allocation mechanisms"
                ],
                deliverables=[
                    "attention integrated with cogserver",
                    "ECAN algorithms functional", 
                    "Attention spreading working",
                    "Resource allocation tests passing"
                ]
            ),
            
            # Phase 4: Advanced Systems
            "pln": Component(
                name="pln", 
                repository="https://github.com/opencog/pln",
                phase=Phase.PHASE_4_ADVANCED_LEARNING,
                dependencies=["atomspace", "ure"],
                priority="HIGH",
                tasks=[
                    "Clone pln (Probabilistic Logic Networks) repository",
                    "Analyze probabilistic reasoning dependencies", 
                    "Configure PLN inference engines", 
                    "Integrate with ure dependency",
                    "Test probabilistic inference",
                    "Validate PLN reasoning chains"
                ],
                deliverables=[
                    "pln integrated with ure dependency",
                    "Probabilistic inference working",
                    "PLN reasoning tests passing", 
                    "Integration with reasoning validated"
                ]
            ),
            
            # Phase 5: Language & Integration  
            "opencog": Component(
                name="opencog",
                repository="https://github.com/opencog/opencog",
                phase=Phase.PHASE_5_LANGUAGE_INTEGRATION,
                dependencies=["atomspace", "cogserver", "attention", "ure"],
                priority="CRITICAL",
                tasks=[
                    "Clone opencog main integration repository", 
                    "Analyze final integration requirements",
                    "Configure all component dependencies", 
                    "Integrate into unified build system",
                    "Test complete system integration",
                    "Validate end-to-end functionality"
                ],
                deliverables=[
                    "opencog main integration complete",
                    "All components working together", 
                    "End-to-end system tests passing",
                    "Complete OpenCog stack functional"
                ]
            )
        }
    
    def _ensure_directories(self):
        """Create required directory structure."""
        directories = [
            self.components_dir,
            self.components_dir / "core", 
            self.components_dir / "logic",
            self.components_dir / "cognitive", 
            self.components_dir / "advanced",
            self.components_dir / "language",
            self.tests_dir / "integration",
            self.tests_dir / "performance", 
            self.tests_dir / "end_to_end",
            self.docs_dir / "components",
            self.docs_dir / "integration"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
            logger.info(f"Ensured directory: {directory}")
    
    def clone_component(self, component_name: str, remove_git: bool = True) -> bool:
        """Clone a component repository and optionally remove git headers."""
        if component_name not in self.components:
            logger.error(f"Unknown component: {component_name}")
            return False
        
        component = self.components[component_name]
        target_dir = self.components_dir / component.name
        
        try:
            # Clone repository
            logger.info(f"Cloning {component.name} from {component.repository}")
            subprocess.run([
                "git", "clone", component.repository, str(target_dir)
            ], check=True, capture_output=True)
            
            # Remove git headers for monorepo approach
            if remove_git:
                git_dir = target_dir / ".git"
                if git_dir.exists():
                    shutil.rmtree(git_dir)
                    logger.info(f"Removed git headers from {component.name}")
            
            # Create component status file
            status_file = target_dir / "conversion_status.json"
            with open(status_file, 'w') as f:
                json.dump({
                    "component": component.name,
                    "phase": component.phase.value,
                    "cloned_at": str(subprocess.check_output(["date"], text=True).strip()),
                    "status": "cloned",
                    "tasks_completed": [],
                    "dependencies": component.dependencies
                }, f, indent=2)
            
            logger.info(f"Successfully cloned {component.name}")
            return True
            
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to clone {component.name}: {e}")
            return False
        except Exception as e:
            logger.error(f"Error cloning {component.name}: {e}")
            return False
    
    def clone_phase_components(self, phase: Phase) -> Dict[str, bool]:
        """Clone all components for a specific phase."""
        results = {}
        phase_components = [
            comp for comp in self.components.values() 
            if comp.phase == phase
        ]
        
        logger.info(f"Cloning {len(phase_components)} components for {phase.value}")
        
        for component in phase_components:
            results[component.name] = self.clone_component(component.name)
        
        return results
    
    def validate_dependencies(self, component_name: str) -> bool:
        """Validate that all dependencies are available."""
        if component_name not in self.components:
            return False
        
        component = self.components[component_name]
        missing_deps = []
        
        for dep in component.dependencies:
            dep_dir = self.components_dir / dep
            if not dep_dir.exists():
                missing_deps.append(dep)
        
        if missing_deps:
            logger.warning(f"Missing dependencies for {component_name}: {missing_deps}")
            return False
        
        logger.info(f"All dependencies satisfied for {component_name}")
        return True
    
    def validate_python_bindings(self, component_name: str) -> bool:
        """Validate Python bindings for a component."""
        if component_name not in self.components:
            logger.error(f"Unknown component: {component_name}")
            return False
        
        component = self.components[component_name]
        component_dir = self.components_dir / component_name
        
        if not component_dir.exists():
            logger.error(f"Component {component_name} not cloned yet")
            return False
        
        logger.info(f"Validating Python bindings for {component_name}")
        
        # Check for Python binding prerequisites
        validation_results = []
        
        # 1. Check CMake Python configuration
        cmake_check = self._validate_cmake_python_config(component_dir)
        validation_results.append(("CMake Python Config", cmake_check))
        
        # 2. Check Python interpreter availability
        python_check = self._validate_python_interpreter()
        validation_results.append(("Python Interpreter", python_check))
        
        # 3. Check component-specific Python readiness
        component_check = self._validate_component_python_readiness(component_name, component_dir)
        validation_results.append(("Component Python Readiness", component_check))
        
        # 4. Check build system compatibility
        build_check = self._validate_build_system_compatibility(component_dir)
        validation_results.append(("Build System Compatibility", build_check))
        
        # Print detailed results
        print(f"\nPython Bindings Validation Results for {component_name}:")
        print("=" * 60)
        all_passed = True
        for check_name, result in validation_results:
            status = "✓ PASS" if result else "✗ FAIL"
            print(f"  {status}: {check_name}")
            if not result:
                all_passed = False
        
        if all_passed:
            logger.info(f"✓ Python bindings validation passed for {component_name}")
            self._update_component_status(component_name, "python_bindings_validated")
        else:
            logger.error(f"✗ Python bindings validation failed for {component_name}")
        
        return all_passed
    
    def _validate_cmake_python_config(self, component_dir: Path) -> bool:
        """Check if CMake Python configuration is present and valid."""
        try:
            cmake_files = list(component_dir.glob("**/CMakeLists.txt"))
            python_cmake_files = list(component_dir.glob("**/OpenCogFindPython.cmake"))
            
            # Check if main CMakeLists.txt exists
            main_cmake = component_dir / "CMakeLists.txt"
            if not main_cmake.exists():
                logger.warning("No main CMakeLists.txt found")
                return False
            
            # Check for Python-related CMake configuration
            has_python_config = False
            for cmake_file in cmake_files + python_cmake_files:
                try:
                    with open(cmake_file, 'r') as f:
                        content = f.read().lower()
                        if any(keyword in content for keyword in ['python', 'cython', 'find_package(python']):
                            has_python_config = True
                            break
                except Exception:
                    continue
            
            if has_python_config:
                logger.info("✓ CMake Python configuration found")
                return True
            else:
                logger.warning("No CMake Python configuration detected")
                return False
                
        except Exception as e:
            logger.error(f"Error checking CMake Python config: {e}")
            return False
    
    def _validate_python_interpreter(self) -> bool:
        """Check if Python interpreter is available and compatible."""
        try:
            result = subprocess.run([
                "python3", "--version"
            ], capture_output=True, text=True, check=True)
            
            version_output = result.stdout.strip()
            logger.info(f"✓ {version_output} found")
            
            # Check if Python development headers are available
            try:
                import sysconfig
                include_dir = sysconfig.get_path('include')
                logger.info(f"✓ Python development headers available at {include_dir}")
                return True
            except Exception:
                logger.warning("Python development headers may not be available")
                return True  # Still count as success for basic validation
                
        except subprocess.CalledProcessError:
            logger.error("Python3 interpreter not found")
            return False
        except Exception as e:
            logger.error(f"Error checking Python interpreter: {e}")
            return False
    
    def _validate_component_python_readiness(self, component_name: str, component_dir: Path) -> bool:
        """Validate component-specific Python binding readiness."""
        try:
            if component_name == "cogutil":
                return self._validate_cogutil_python_readiness(component_dir)
            elif component_name == "atomspace":
                return self._validate_atomspace_python_readiness(component_dir)
            else:
                # Generic validation for other components
                return self._validate_generic_python_readiness(component_dir)
                
        except Exception as e:
            logger.error(f"Error validating component Python readiness: {e}")
            return False
    
    def _validate_cogutil_python_readiness(self, component_dir: Path) -> bool:
        """Validate cogutil-specific Python binding readiness."""
        try:
            # Check for key utility headers that will be used by Python bindings
            util_dir = component_dir / "opencog" / "util"
            if not util_dir.exists():
                logger.warning("cogutil utility directory not found")
                return False
            
            # Check for key utility classes
            key_files = ["cogutil.h", "Config.h", "Logger.h"]
            missing_files = []
            for key_file in key_files:
                if not (util_dir / key_file).exists():
                    missing_files.append(key_file)
            
            if missing_files:
                logger.warning(f"Missing key cogutil files: {missing_files}")
                return False
            
            # Check for Python binding configuration
            python_cmake = component_dir / "cmake" / "OpenCogFindPython.cmake"
            if python_cmake.exists():
                logger.info("✓ cogutil Python CMake configuration found")
                return True
            else:
                logger.warning("cogutil Python CMake configuration not found")
                return False
                
        except Exception as e:
            logger.error(f"Error validating cogutil Python readiness: {e}")
            return False
    
    def _validate_atomspace_python_readiness(self, component_dir: Path) -> bool:
        """Validate atomspace-specific Python binding readiness."""
        # Placeholder for atomspace validation
        logger.info("atomspace Python validation not yet implemented")
        return True
    
    def _validate_generic_python_readiness(self, component_dir: Path) -> bool:
        """Generic Python binding readiness validation."""
        try:
            # Check for basic structure
            has_headers = len(list(component_dir.glob("**/*.h"))) > 0
            has_sources = len(list(component_dir.glob("**/*.cc"))) > 0 or len(list(component_dir.glob("**/*.cpp"))) > 0
            
            if has_headers and has_sources:
                logger.info("✓ Basic C++ structure found for Python binding generation")
                return True
            else:
                logger.warning("Insufficient C++ structure for Python bindings")
                return False
                
        except Exception as e:
            logger.error(f"Error in generic Python validation: {e}")
            return False
    
    def _validate_build_system_compatibility(self, component_dir: Path) -> bool:
        """Check if the build system is compatible with Python bindings."""
        try:
            # Check for CMake compatibility
            cmake_file = component_dir / "CMakeLists.txt"
            if not cmake_file.exists():
                logger.warning("No CMakeLists.txt found")
                return False
            
            with open(cmake_file, 'r') as f:
                content = f.read()
            
            # Check for minimum CMake version compatibility
            if "CMAKE_MINIMUM_REQUIRED" in content:
                logger.info("✓ CMake version requirements specified")
                return True
            else:
                logger.warning("No CMake version requirements found")
                return False
                
        except Exception as e:
            logger.error(f"Error checking build system compatibility: {e}")
            return False
    
    def _update_component_status(self, component_name: str, new_status: str):
        """Update component status file with new validation status."""
        try:
            component_dir = self.components_dir / component_name
            status_file = component_dir / "conversion_status.json"
            
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
                
                if "tasks_completed" not in status:
                    status["tasks_completed"] = []
                
                if new_status not in status["tasks_completed"]:
                    status["tasks_completed"].append(new_status)
                
                status["last_updated"] = str(subprocess.check_output(["date"], text=True).strip())
                
                with open(status_file, 'w') as f:
                    json.dump(status, f, indent=2)
                
                logger.info(f"Updated status for {component_name}: {new_status}")
            
        except Exception as e:
            logger.warning(f"Could not update component status: {e}")
    
    def generate_phase_report(self, phase: Phase) -> Dict:
        """Generate status report for a specific phase."""
        phase_components = [
            comp for comp in self.components.values()
            if comp.phase == phase
        ]
        
        report = {
            "phase": phase.value,
            "total_components": len(phase_components),
            "components": {}
        }
        
        for component in phase_components:
            comp_dir = self.components_dir / component.name
            status_file = comp_dir / "conversion_status.json"
            
            if status_file.exists():
                with open(status_file, 'r') as f:
                    status = json.load(f)
                report["components"][component.name] = status
            else:
                report["components"][component.name] = {
                    "status": "not_cloned",
                    "dependencies": component.dependencies
                }
        
        return report
    
    def run_integration_tests(self, component_name: str = None) -> bool:
        """Run integration tests for a component or all components."""
        if component_name:
            # Run tests for specific component
            test_script = self.tests_dir / "integration" / f"test_{component_name}.py"
            if test_script.exists():
                try:
                    subprocess.run([sys.executable, "-m", "pytest", str(test_script)], check=True)
                    logger.info(f"Integration tests passed for {component_name}")
                    return True
                except subprocess.CalledProcessError:
                    logger.error(f"Integration tests failed for {component_name}")
                    return False
            else:
                logger.warning(f"No integration tests found for {component_name}")
                return False
        else:
            # Run all integration tests
            try:
                subprocess.run([sys.executable, "-m", "pytest", str(self.tests_dir / "integration")], check=True)
                logger.info("All integration tests passed")
                return True
            except subprocess.CalledProcessError:
                logger.error("Some integration tests failed")
                return False

def main():
    """Main CLI interface for the cpp2py conversion pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description="PyCog-Zero cpp2py Conversion Pipeline")
    parser.add_argument("--root", default=".", help="Root directory for the project")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Clone command
    clone_parser = subparsers.add_parser("clone", help="Clone component repositories")
    clone_parser.add_argument("component", nargs="?", help="Component name to clone (or 'all' for all)")
    clone_parser.add_argument("--phase", choices=[p.value for p in Phase], help="Clone all components for a phase")
    clone_parser.add_argument("--keep-git", action="store_true", help="Keep git headers (default: remove for monorepo)")
    
    # Status command
    status_parser = subparsers.add_parser("status", help="Show conversion status")
    status_parser.add_argument("--phase", choices=[p.value for p in Phase], help="Show status for specific phase")
    
    # Test command
    test_parser = subparsers.add_parser("test", help="Run integration tests")
    test_parser.add_argument("component", nargs="?", help="Component to test (default: all)")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate dependencies and Python bindings")
    validate_parser.add_argument("component", help="Component to validate")
    validate_parser.add_argument("--deps-only", action="store_true", help="Only validate dependencies, skip Python bindings")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    pipeline = CPP2PyConversionPipeline(args.root)
    
    if args.command == "clone":
        if args.phase:
            phase = Phase(args.phase)
            results = pipeline.clone_phase_components(phase)
            print(f"\nCloning results for {args.phase}:")
            for comp, success in results.items():
                status = "✓" if success else "✗"
                print(f"  {status} {comp}")
        elif args.component:
            if args.component == "all":
                all_results = {}
                for phase in Phase:
                    results = pipeline.clone_phase_components(phase)
                    all_results.update(results)
                print("\nCloning results for all components:")
                for comp, success in all_results.items():
                    status = "✓" if success else "✗" 
                    print(f"  {status} {comp}")
            else:
                success = pipeline.clone_component(args.component, remove_git=not args.keep_git)
                status = "✓" if success else "✗"
                print(f"{status} {args.component}")
        else:
            clone_parser.print_help()
    
    elif args.command == "status":
        if args.phase:
            phase = Phase(args.phase)
            report = pipeline.generate_phase_report(phase)
            print(f"\nPhase: {report['phase']}")
            print(f"Total components: {report['total_components']}")
            print("\nComponent status:")
            for name, status in report['components'].items():
                print(f"  {name}: {status.get('status', 'unknown')}")
        else:
            print("\nOverall Status:")
            for phase in Phase:
                report = pipeline.generate_phase_report(phase) 
                cloned = sum(1 for s in report['components'].values() if s.get('status') == 'cloned')
                total = report['total_components']
                print(f"  {phase.value}: {cloned}/{total} components cloned")
    
    elif args.command == "test":
        success = pipeline.run_integration_tests(args.component)
        sys.exit(0 if success else 1)
    
    elif args.command == "validate":
        if args.deps_only:
            success = pipeline.validate_dependencies(args.component)
        else:
            # Full validation includes dependencies + Python bindings
            deps_success = pipeline.validate_dependencies(args.component)
            if deps_success:
                bindings_success = pipeline.validate_python_bindings(args.component)
                success = deps_success and bindings_success
            else:
                success = False
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()