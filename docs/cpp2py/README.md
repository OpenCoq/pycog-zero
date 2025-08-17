# PyCog-Zero cpp2py Conversion Pipeline Documentation

## Overview

The PyCog-Zero cpp2py conversion pipeline is a comprehensive system for systematically converting OpenCog C++ components to Python and integrating them into the PyCog-Zero cognitive architecture framework. This pipeline implements the 20-week roadmap for unified OpenCog component integration.

## Architecture

### Pipeline Components

```
PyCog-Zero cpp2py Pipeline
├── Conversion Pipeline Manager (scripts/cpp2py_conversion_pipeline.py)
├── Component Definitions (OpenCog repositories mapped to phases)
├── Build System (scripts/build_cpp2py_pipeline.sh)  
├── Testing Infrastructure (tests/integration/)
├── Documentation (docs/cpp2py/)
└── Configuration (cpp2py_config.json)
```

### Phase-Based Integration

The pipeline follows a 5-phase approach based on dependency relationships:

#### Phase 0: Foundation Layer
- **cogutil**: Core utilities and data structures
- **Dependencies**: None
- **Timeline**: Weeks 0-1

#### Phase 1: Core Extensions  
- **atomspace**: Hypergraph knowledge representation
- **cogserver**: Multi-agent cognitive server
- **atomspace-rocks**: RocksDB storage backend
- **moses**: Evolutionary learning algorithms
- **Dependencies**: cogutil
- **Timeline**: Weeks 1-4

#### Phase 2: Logic Systems
- **unify**: Pattern unification algorithms
- **ure**: Unified Rule Engine (forward/backward chaining)
- **language-learning**: Language processing and learning
- **Dependencies**: atomspace, unify
- **Timeline**: Weeks 5-8

#### Phase 3: Cognitive Systems
- **attention**: Economic Cognitive Attention Networks (ECAN)
- **spacetime**: Spatial-temporal reasoning
- **Dependencies**: atomspace, cogserver
- **Timeline**: Weeks 9-12

#### Phase 4: Advanced & Learning Systems
- **pln**: Probabilistic Logic Networks
- **miner**: Pattern mining and knowledge discovery
- **asmoses**: AtomSpace MOSES integration
- **Dependencies**: atomspace, ure, spacetime
- **Timeline**: Weeks 13-16

#### Phase 5: Language & Final Integration
- **lg-atomese**: Link Grammar to AtomSpace conversion
- **learn**: Unsupervised learning algorithms
- **opencog**: Final unified integration
- **Dependencies**: All previous components
- **Timeline**: Weeks 17-20

## Installation & Setup

### Prerequisites

- Python 3.8+
- Git
- Virtual environment support
- Network access for repository cloning

### Quick Setup

```bash
# Clone the PyCog-Zero repository
git clone https://github.com/OpenCoq/pycog-zero.git
cd pycog-zero

# Run the automated build script
bash scripts/build_cpp2py_pipeline.sh

# Install OpenCog dependencies (optional but recommended)
pip install opencog-atomspace opencog-python cogutil
```

### Manual Setup

```bash
# Create virtual environment
python3 -m venv pycog-zero-env
source pycog-zero-env/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-cognitive.txt

# Install testing dependencies
pip install pytest pytest-asyncio

# Create directory structure
python3 scripts/cpp2py_conversion_pipeline.py status
```

## Usage

### Command Line Interface

The main pipeline script provides a comprehensive CLI:

```bash
python3 scripts/cpp2py_conversion_pipeline.py --help
```

#### Key Commands

**Check Status**
```bash
# Overall pipeline status
python3 scripts/cpp2py_conversion_pipeline.py status

# Status for specific phase
python3 scripts/cpp2py_conversion_pipeline.py status --phase phase_0_foundation
```

**Clone Components**
```bash
# Clone single component
python3 scripts/cpp2py_conversion_pipeline.py clone cogutil

# Clone entire phase
python3 scripts/cpp2py_conversion_pipeline.py clone --phase phase_0_foundation

# Clone all components
python3 scripts/cpp2py_conversion_pipeline.py clone all
```

**Validate Dependencies**
```bash
# Check if dependencies are satisfied for a component
python3 scripts/cpp2py_conversion_pipeline.py validate atomspace

# Validate dependencies only (skip Python bindings validation)
python3 scripts/cpp2py_conversion_pipeline.py validate cogutil --deps-only
```

**Validate Python Bindings**
```bash
# Full validation including Python bindings readiness
python3 scripts/cpp2py_conversion_pipeline.py validate cogutil

# This checks:
# - CMake Python configuration
# - Python interpreter compatibility
# - Component-specific Python readiness
# - Build system compatibility
```

**Run Tests**
```bash
# Run all integration tests
python3 scripts/cpp2py_conversion_pipeline.py test

# Test specific component
python3 scripts/cpp2py_conversion_pipeline.py test cogutil
```

### Component Cloning Process

When components are cloned, the pipeline:

1. **Clones Repository**: Downloads the OpenCog component from GitHub
2. **Removes Git Headers**: Strips `.git` directory for monorepo approach
3. **Creates Status File**: Adds `conversion_status.json` tracking file
4. **Validates Structure**: Ensures component structure is as expected

### Monorepo Approach

The pipeline implements a monorepo strategy:

- **No Submodules**: All components become part of single repository
- **Git Headers Removed**: Components lose independent git history
- **Unified Build**: All components share build system and configuration
- **Centralized Testing**: Integrated test suite for all components

## Integration with PyCog-Zero

### Existing Cognitive Tools

The pipeline integrates with existing PyCog-Zero cognitive capabilities:

```python
# Existing cognitive reasoning tool
from python.tools.cognitive_reasoning import CognitiveReasoningTool

# Cognitive configuration
config = {
    "cognitive_mode": true,
    "opencog_enabled": true,
    "neural_symbolic_bridge": true,
    "ecan_attention": true,
    "pln_reasoning": true
}
```

### Agent-Zero Framework Integration

Components are designed to integrate as Agent-Zero tools:

```python
# Example: PLN reasoning as Agent-Zero tool
class PLNReasoningTool(Tool):
    async def execute(self, query: str, **kwargs):
        # Convert Agent-Zero query to AtomSpace
        atoms = self.parse_query_to_atoms(query)
        
        # Apply PLN reasoning
        results = self.pln_chainer.forward_chain(atoms)
        
        # Return Agent-Zero compatible response
        return Response(message="PLN reasoning completed", data=results)
```

### Neural-Symbolic Bridge

The pipeline supports neural-symbolic integration:

```python
# Bridge PyTorch and OpenCog
from python.helpers.neural_symbolic_bridge import NeuralSymbolicBridge

bridge = NeuralSymbolicBridge(atomspace)
embeddings = bridge.atomspace_to_tensor(atoms)
processed = neural_network(embeddings)
new_atoms = bridge.tensor_to_atomspace(processed)
```

## Testing Infrastructure

### Integration Tests

Comprehensive test suite for validating component integration:

```bash
# Run all integration tests
python3 -m pytest tests/integration/ -v

# Run specific test categories
python3 -m pytest tests/integration/test_cpp2py_pipeline.py::TestComponentIntegration -v
```

### Test Categories

1. **Pipeline Tests**: Validate conversion pipeline functionality
2. **Component Tests**: Test individual component integration  
3. **PyCog-Zero Tests**: Test integration with existing cognitive tools
4. **Build System Tests**: Validate build and dependency management
5. **End-to-End Tests**: Complete workflow validation

### Performance Benchmarks

Performance testing for cognitive operations:

```python
# Benchmark AtomSpace operations
from tests.performance.benchmark_cognitive import CognitiveBenchmark

benchmark = CognitiveBenchmark()
benchmark.benchmark_atomspace_operations(1000)
benchmark.benchmark_neural_symbolic_bridge(100)
```

## Configuration

### Pipeline Configuration (cpp2py_config.json)

```json
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
  }
}
```

### Cognitive Configuration (conf/config_cognitive.json)

```json
{
  "cognitive_mode": true,
  "opencog_enabled": true,
  "neural_symbolic_bridge": true,
  "ecan_attention": true,
  "pln_reasoning": true,
  "atomspace_config": {
    "persistence_backend": "file",
    "persistence_path": "memory/cognitive_atomspace.pkl"
  }
}
```

## Development Workflow

### Phase-Based Development

1. **Phase Planning**: Review component definitions and dependencies
2. **Component Cloning**: Clone repositories for current phase
3. **Integration Development**: Create Python bindings and Agent-Zero tools
4. **Testing**: Comprehensive integration and performance testing
5. **Validation**: End-to-end workflow validation
6. **Documentation**: Update documentation and examples
7. **Phase Advancement**: Move to next phase

### Component Integration Steps

For each OpenCog component:

1. **Clone Repository**
   ```bash
   python3 scripts/cpp2py_conversion_pipeline.py clone <component>
   ```

2. **Analyze Structure**
   - Review C++ source code
   - Identify key classes and functions
   - Map dependencies and interfaces

3. **Create Python Bindings**
   - Implement Python wrapper classes
   - Create Agent-Zero tool interfaces
   - Add neural-symbolic bridge support

4. **Integration Testing**
   - Unit tests for Python bindings
   - Integration tests with existing tools
   - Performance benchmarking

5. **Documentation**
   - API documentation
   - Usage examples
   - Integration guides

### Quality Assurance

- **Code Review**: All integration code reviewed
- **Testing**: Comprehensive test coverage required
- **Performance**: Benchmarking for cognitive operations
- **Documentation**: Complete documentation for all components
- **Validation**: End-to-end workflow testing

## Troubleshooting

### Common Issues

**OpenCog Dependencies Missing**
```bash
pip install opencog-atomspace opencog-python cogutil
```

**Component Cloning Fails**
- Check network connectivity
- Verify repository URLs
- Ensure git is installed

**Integration Tests Fail**
- Check Python environment
- Verify dependencies installed
- Review error logs

**Performance Issues**
- Profile cognitive operations
- Optimize AtomSpace usage
- Consider parallel processing

### Debug Mode

Enable detailed logging:

```bash
export PYCOG_DEBUG=1
python3 scripts/cpp2py_conversion_pipeline.py status
```

### Support

- **Documentation**: [docs/cpp2py/](.)
- **Issues**: GitHub Issues tracker
- **Community**: Discord/Forums
- **Examples**: [examples/](../examples/)

## Roadmap

### Current Status (Phase 0)

- [x] Pipeline infrastructure created
- [x] Component definitions mapped
- [x] Testing framework established
- [x] Build system implemented
- [ ] Foundation components cloned and integrated

### Next Steps

1. **Phase 0 Completion**: cogutil integration
2. **Phase 1 Initiation**: atomspace and cogserver integration
3. **Neural-Symbolic Bridge**: Enhanced PyTorch integration
4. **Performance Optimization**: Large-scale cognitive processing
5. **Production Deployment**: Scalable cognitive agent systems

### Long-Term Goals

- Complete OpenCog C++ to Python conversion
- Seamless Agent-Zero cognitive tool ecosystem
- Advanced neural-symbolic reasoning capabilities
- Production-ready cognitive agent deployment
- Community-driven cognitive architecture development

---

*This documentation provides comprehensive guidance for using and extending the PyCog-Zero cpp2py conversion pipeline. For additional support, refer to the examples and community resources.*