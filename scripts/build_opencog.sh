#!/bin/bash
#
# PyCog-Zero OpenCog Build Script
# ================================
# 
# Builds OpenCog Python bindings from the C++ components in the components/ directory.
# This script automates the process described in AGENT-ZERO-GENESIS.md
#

set -e  # Exit on any error

echo "🧠 PyCog-Zero: Building OpenCog Python Bindings"
echo "=============================================="
echo

# Get the directory where this script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "📍 Project root: $PROJECT_ROOT"
echo "📍 Components: $PROJECT_ROOT/components"

# Check if components directory exists
if [ ! -d "$PROJECT_ROOT/components" ]; then
    echo "❌ Error: components/ directory not found"
    echo "   Please run from the PyCog-Zero project root directory"
    exit 1
fi

# Check for required components
REQUIRED_COMPONENTS=("cogutil" "atomspace")
for component in "${REQUIRED_COMPONENTS[@]}"; do
    if [ ! -d "$PROJECT_ROOT/components/$component" ]; then
        echo "❌ Error: Component $component not found in components/"
        exit 1
    fi
done

echo "✅ Required OpenCog components found"
echo

# Install system dependencies
echo "📦 Installing system dependencies..."
sudo apt update -qq
sudo apt install -y \
    build-essential \
    cmake \
    libboost-all-dev \
    guile-3.0-dev \
    cython3 \
    python3-dev \
    python3-pip

# Install Cython if not available
if ! python3 -c "import cython" 2>/dev/null; then
    echo "📦 Installing Cython..."
    pip install cython
fi

echo "✅ System dependencies installed"
echo

# Build cogutil (foundation dependency)
echo "🔧 Building cogutil..."
cd "$PROJECT_ROOT/components/cogutil"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig

echo "✅ cogutil built and installed"
echo

# Build atomspace (with Python bindings)
echo "🔧 Building atomspace with Python bindings..."
cd "$PROJECT_ROOT/components/atomspace"
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig

echo "✅ atomspace built and installed"
echo

# Test the installation
echo "🧪 Testing OpenCog Python bindings..."
python3 -c "
from opencog.atomspace import AtomSpace, types
from opencog.utilities import initialize_opencog
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

print('Testing basic functionality...')
atomspace = AtomSpace()
initialize_opencog(atomspace)
node = atomspace.add_node(types.ConceptNode, 'test')
print(f'Created node: {node}')
print(f'AtomSpace size: {atomspace.size()}')
print('✅ OpenCog Python bindings working!')
"

echo
echo "🎉 OpenCog Python bindings successfully built and installed!"
echo
echo "📋 Next steps:"
echo "   1. Test Agent-Zero cognitive tools: python3 -c \"from python.tools.cognitive_reasoning import CognitiveReasoningTool\""
echo "   2. Start Agent-Zero: python3 agent.py"
echo "   3. Use cognitive features in the Agent-Zero interface"
echo