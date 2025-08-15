# OpenCog Python Bindings Installation

This document describes how to install and configure OpenCog Python bindings for the PyCog-Zero Agent-Zero framework.

## Quick Installation

Use the automated build script:

```bash
./scripts/build_opencog.sh
```

This script will:
1. Install system dependencies (cmake, build-essential, boost, guile, etc.)
2. Build cogutil from `components/cogutil/`
3. Build atomspace with Python bindings from `components/atomspace/` 
4. Install both to system locations
5. Test the installation

## Manual Installation

If you prefer to build manually:

### 1. Install Dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake libboost-all-dev guile-3.0-dev python3-dev
pip install cython
```

### 2. Build cogutil

```bash
cd components/cogutil
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig
```

### 3. Build atomspace

```bash
cd components/atomspace  
mkdir -p build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
make -j$(nproc)
sudo make install
sudo ldconfig
```

## Verification

Test the installation:

```bash
python3 -c "from opencog.atomspace import AtomSpace, types; print('OpenCog working!')"
```

Test Agent-Zero integration:

```bash
PYTHONPATH=. python3 -c "from python.tools.cognitive_reasoning import CognitiveReasoningTool; print('Cognitive tool ready!')"
```

## Usage in Agent-Zero

Once installed, the `cognitive_reasoning.py` tool will automatically detect and use OpenCog:

```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool

# The tool will initialize with OpenCog if available
tool = CognitiveReasoningTool(agent)
result = await tool.execute("reason about concepts and relationships")
```

## Troubleshooting

### Import Errors

If you get import errors, ensure:
1. OpenCog was built and installed correctly (`sudo make install` step)
2. Library cache was updated (`sudo ldconfig`)
3. Python path includes the installed modules

### Build Errors

Common issues:
- Missing dependencies: Run `sudo apt install build-essential cmake libboost-all-dev guile-3.0-dev`
- Old CMake version: Requires CMake 3.12+
- Missing Cython: Run `pip install cython`

### Permission Issues

If you get permission errors:
- Use `sudo` for `make install` and `ldconfig` commands
- Or install to a local prefix with `-DCMAKE_INSTALL_PREFIX=$HOME/.local`

## Architecture

The integration works as follows:

```
Agent-Zero
    ↓
python/tools/cognitive_reasoning.py  
    ↓
OpenCog Python Bindings (built from C++)
    ↓
OpenCog AtomSpace + cogutil (C++ libraries)
```

The `CognitiveReasoningTool` provides a bridge between Agent-Zero's tool system and OpenCog's cognitive architecture, enabling symbolic reasoning capabilities within the agent framework.