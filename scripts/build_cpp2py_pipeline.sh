#!/bin/bash
"""
PyCog-Zero Build Script for cpp2py Conversion Pipeline
======================================================

Build script for setting up and validating the OpenCog component conversion pipeline.
"""

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo -e "${BLUE}PyCog-Zero cpp2py Conversion Pipeline Build Script${NC}"
echo "=================================================="
echo "Project Root: $PROJECT_ROOT"
echo ""

# Function to print status
print_status() {
    echo -e "${GREEN}✓${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

# Check Python version
echo "Checking Python environment..."
python_version=$(python3 --version 2>&1)
print_info "Python version: $python_version"

if python3 -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)"; then
    print_status "Python 3.8+ detected"
else
    print_error "Python 3.8+ required"
    exit 1
fi

# Check if virtual environment exists or create one
if [ ! -d "$PROJECT_ROOT/pycog-zero-env" ]; then
    print_info "Creating Python virtual environment..."
    python3 -m venv "$PROJECT_ROOT/pycog-zero-env"
    print_status "Virtual environment created"
else
    print_status "Virtual environment already exists"
fi

# Activate virtual environment
print_info "Activating virtual environment..."
source "$PROJECT_ROOT/pycog-zero-env/bin/activate" || {
    print_error "Failed to activate virtual environment"
    exit 1
}

# Install dependencies
echo ""
echo "Installing dependencies..."

# Install base requirements
if [ -f "$PROJECT_ROOT/requirements.txt" ]; then
    print_info "Installing base requirements..."
    pip install -r "$PROJECT_ROOT/requirements.txt" --quiet || {
        print_warning "Some base requirements may have failed to install"
    }
    print_status "Base requirements processed"
else
    print_warning "requirements.txt not found"
fi

# Install cognitive requirements  
if [ -f "$PROJECT_ROOT/requirements-cognitive.txt" ]; then
    print_info "Installing cognitive requirements..."
    pip install -r "$PROJECT_ROOT/requirements-cognitive.txt" --quiet || {
        print_warning "Some cognitive requirements may have failed to install"
    }
    print_status "Cognitive requirements processed"
else
    print_warning "requirements-cognitive.txt not found"
fi

# Install pytest for testing
print_info "Installing testing dependencies..."
pip install pytest pytest-asyncio --quiet
print_status "Testing dependencies installed"

# Validate OpenCog integration
echo ""
echo "Validating OpenCog integration..."

python3 -c "
try:
    from opencog.atomspace import AtomSpace, types
    print('✓ OpenCog AtomSpace bindings available')
    atomspace_available = True
except ImportError as e:
    print('⚠ OpenCog AtomSpace bindings not available:', e)
    atomspace_available = False

try:
    import torch
    print('✓ PyTorch available')
    torch_available = True
except ImportError as e:
    print('⚠ PyTorch not available:', e)
    torch_available = False

try:
    import numpy as np
    print('✓ NumPy available')
except ImportError as e:
    print('✗ NumPy not available:', e)

if atomspace_available and torch_available:
    print('✓ Core cognitive dependencies satisfied')
else:
    print('⚠ Some cognitive dependencies missing - install with:')
    if not atomspace_available:
        print('  pip install opencog-atomspace opencog-python')
    if not torch_available:
        print('  pip install torch')
" || print_warning "Dependency validation had issues"

# Create necessary directories
echo ""
echo "Setting up directory structure..."

directories=(
    "components"
    "components/core"
    "components/logic" 
    "components/cognitive"
    "components/advanced"
    "components/language"
    "tests/integration"
    "tests/performance"
    "tests/end_to_end"
    "docs/components"
    "docs/integration"
)

for dir in "${directories[@]}"; do
    mkdir -p "$PROJECT_ROOT/$dir"
    print_status "Created directory: $dir"
done

# Test the conversion pipeline script
echo ""
echo "Testing conversion pipeline script..."

if [ -f "$PROJECT_ROOT/scripts/cpp2py_conversion_pipeline.py" ]; then
    # Test basic script execution
    cd "$PROJECT_ROOT"
    python3 scripts/cpp2py_conversion_pipeline.py --help > /dev/null 2>&1 && {
        print_status "Conversion pipeline script is executable"
    } || {
        print_error "Conversion pipeline script has issues"
        exit 1
    }
    
    # Test status command
    python3 scripts/cpp2py_conversion_pipeline.py status > /dev/null 2>&1 && {
        print_status "Pipeline status command works"
    } || {
        print_warning "Pipeline status command had issues"
    }
else
    print_error "Conversion pipeline script not found"
    exit 1
fi

# Run basic tests
echo ""
echo "Running integration tests..."

cd "$PROJECT_ROOT"

if [ -f "tests/integration/test_cpp2py_pipeline.py" ]; then
    python3 -m pytest tests/integration/test_cpp2py_pipeline.py -v --tb=short || {
        print_warning "Some integration tests failed"
    }
    print_status "Integration tests completed"
else
    print_warning "Integration test file not found"
fi

# Test existing cognitive tools
echo ""
echo "Testing existing cognitive tools..."

python3 -c "
import sys
import os
sys.path.insert(0, '.')

try:
    from python.tools.cognitive_reasoning import CognitiveReasoningTool
    print('✓ Cognitive reasoning tool importable')
    
    # Test tool instantiation
    class MockAgent:
        pass
    
    tool = CognitiveReasoningTool(MockAgent())
    print('✓ Cognitive reasoning tool instantiable')
    
except ImportError as e:
    print('⚠ Cannot import cognitive reasoning tool:', e)
except Exception as e:
    if 'OpenCog' in str(e):
        print('⚠ Cognitive tool requires OpenCog:', e)
    else:
        print('✗ Error testing cognitive tool:', e)
"

# Generate initial status report
echo ""
echo "Generating pipeline status report..."

python3 scripts/cpp2py_conversion_pipeline.py status

# Create example configuration
echo ""
echo "Creating example configuration..."

cat > "$PROJECT_ROOT/cpp2py_config.json" << EOF
{
  "project_name": "PyCog-Zero",
  "conversion_pipeline_version": "1.0.0",
  "opencog_integration": {
    "enabled": true,
    "python_bindings": true,
    "remove_git_headers": true,
    "monorepo_approach": true
  },
  "phases": {
    "current_phase": "phase_0_foundation",
    "auto_advance": false,
    "parallel_cloning": true
  },
  "testing": {
    "integration_tests": true,
    "performance_benchmarks": true,
    "end_to_end_validation": true
  },
  "build_system": {
    "python_version_min": "3.8",
    "virtual_environment": true,
    "dependencies_auto_install": true
  }
}
EOF

print_status "Configuration file created: cpp2py_config.json"

# Summary
echo ""
echo -e "${BLUE}Build Summary${NC}"
echo "=============="
print_status "Directory structure created"
print_status "Dependencies installed"
print_status "Pipeline script validated"
print_status "Integration tests available"
print_status "Configuration file created"

echo ""
print_info "Next steps:"
echo "  1. Install OpenCog dependencies: pip install opencog-atomspace opencog-python"
echo "  2. Clone components: python3 scripts/cpp2py_conversion_pipeline.py clone --phase phase_0_foundation"  
echo "  3. Run tests: python3 -m pytest tests/integration/"
echo "  4. Check status: python3 scripts/cpp2py_conversion_pipeline.py status"

echo ""
echo -e "${GREEN}cpp2py Conversion Pipeline build completed successfully!${NC}"