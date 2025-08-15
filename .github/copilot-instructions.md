# PyCog-Zero - Cognitive Agent Framework

PyCog-Zero is a cognitive architecture that integrates OpenCog's hypergraph-based memory (AtomSpace) with Agent-Zero's autonomous capabilities, creating a Python-native ecosystem for advanced cognitive agent development.

**Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.**

## Working Effectively

### Quick Setup - Docker (Recommended)
- `docker pull agent0ai/agent-zero:latest` -- takes 1.5 minutes. NEVER CANCEL. Set timeout to 5+ minutes.
- `docker run -p 50001:80 agent0ai/agent-zero` -- starts container with web UI
- Visit http://localhost:50001 for web interface

### Local Development Setup
- **Requirements:** Python 3.12+ (tested with 3.12.3)
- **CRITICAL INSTALLATION TIMING:** Main dependencies take 7+ minutes. NEVER CANCEL. Set timeout to 15+ minutes minimum.
- Install dependencies: `pip install -r requirements.txt` -- takes 7-10 minutes. NEVER CANCEL.
- Optional cognitive features: `pip install -r requirements-cognitive.txt` -- will fail without OpenCog binaries. This is expected.
- Install Playwright browsers: `playwright install` -- may fail due to network issues. Use Docker if needed.
- Start web UI: `python3 run_ui.py --host 0.0.0.0 --port 50001` -- takes 15 seconds to start
- Start command-line agent: `python3 agent.py`

### Docker Development Build (Local Files)
- **Local development build:** `docker build -f DockerfileLocal -t agent-zero-local --build-arg CACHE_DATE=$(date +%Y-%m-%d:%H:%M:%S) .` -- takes 10+ minutes. NEVER CANCEL. Set timeout to 20+ minutes.
- **Run local build:** `docker run -p 50001:80 agent-zero-local`

## Validation Scenarios

**ALWAYS test complete end-to-end scenarios after making changes:**

### Web UI Validation
- Start web UI with `python3 run_ui.py`
- Verify server starts on http://localhost:50001 
- Check HTTP response: `curl -I http://localhost:50001/` should return HTTP 200
- Verify no critical errors in startup logs (warnings about missing models are expected)

### Agent Functionality Validation  
- Test Python imports: `python3 -c "from initialize import initialize_agent; import agent; print('Agent framework working')"`
- Test cognitive tool import: `python3 -c "from python.tools.cognitive_reasoning import CognitiveReasoningTool; print('Cognitive tools available')"`
- Verify core dependencies: `python3 -c "import torch, transformers, faiss, playwright; print('Core deps working')"`

### Docker Validation
- Test Docker functionality: `docker --version && docker images | grep agent`
- Verify container runs: `docker run --rm agent0ai/agent-zero:latest echo "Container working"`

## Build and Test Commands

### Dependencies and Setup
- **Main installation:** `pip install -r requirements.txt` -- 7-10 minutes. NEVER CANCEL. Set timeout to 15+ minutes.
- **Cognitive dependencies (optional):** `pip install -r requirements-cognitive.txt` -- will fail, this is expected without OpenCog
- **Playwright browsers:** `playwright install` -- may fail due to network issues

### Testing Framework
- **Basic framework test:** `python3 -c "import agent; print('Success')"`
- **Web UI test:** `curl -I http://localhost:50001/` after starting server
- **Docker test:** `docker pull agent0ai/agent-zero:latest` -- 1.5 minutes. Set timeout to 5+ minutes.

### Validation Before Commits
- **ALWAYS run these before committing changes:**
- `python3 -c "from initialize import initialize_agent; import agent; print('✓ Framework imports work')"`
- `python3 -c "from python.tools.cognitive_reasoning import CognitiveReasoningTool; from python.tools.code_execution_tool import CodeExecution; print('✓ Core tools import work')"` -- may show OpenCog warnings, that's expected
- If modifying web UI: start `python3 run_ui.py` and verify `curl -I http://localhost:50001/` returns HTTP 200

## Critical Timing Information

**NEVER CANCEL these operations - they take significant time:**
- **pip install -r requirements.txt:** 7-10 minutes (includes PyTorch, transformers, FAISS)
- **docker pull agent0ai/agent-zero:latest:** 1.5-3 minutes
- **docker build (local):** 10-20 minutes  
- **Web UI startup:** 15-30 seconds
- **Playwright browser install:** 2-5 minutes (may fail due to network)

**Always set timeouts of at least:**
- pip installs: 15+ minutes
- Docker operations: 10+ minutes  
- Web UI startup: 2+ minutes

## Key Repository Structure

### Core Components
```
agent.py                    # Main agent entry point
run_ui.py                   # Web UI server (Flask on port 50001)
initialize.py               # Agent initialization and configuration
requirements.txt            # Main dependencies (~200 packages, 7+ min install)
requirements-cognitive.txt  # OpenCog dependencies (require special setup)
```

### Important Directories
```
python/tools/               # Agent tools (cognitive_reasoning.py, etc.)
python/helpers/             # Framework utilities  
conf/                       # Configuration (model_providers.yaml, config_cognitive.json)
webui/                      # Web UI components (HTML, CSS, JS)
prompts/                    # Agent behavior and tool prompts
docker/                     # Container build files and scripts
agents/                     # Agent profiles and configurations
memory/                     # Persistent storage for agent memory
```

### Cognitive Architecture Files
```
conf/config_cognitive.json     # Cognitive features configuration
python/tools/cognitive_reasoning.py  # OpenCog integration tool
AGENT-ZERO-GENESIS.md         # Complete cognitive architecture roadmap
```

## Common Development Tasks

### Working with Agent Tools
- **Tool location:** `/python/tools/` directory contains all agent tools
- **Key tools:** code_execution_tool.py, cognitive_reasoning.py, search_engine.py, memory_*.py
- **Tool registration:** Tools auto-register when placed in tools directory
- **Testing tools:** Import test - `python3 -c "from python.tools.TOOLNAME import ToolClass; print('Working')"`

### Cognitive Features
- **OpenCog integration:** Requires `pip install opencog-atomspace opencog-python` (not available via pip)
- **Cognitive config:** Located in `conf/config_cognitive.json`  
- **Status check:** `python3 -c "from python.tools.cognitive_reasoning import CognitiveReasoningTool; print('Available but not initialized without OpenCog')"`
- **Docker includes OpenCog:** Use Docker for full cognitive functionality

### Configuration Management
- **Model providers:** `conf/model_providers.yaml` 
- **Cognitive settings:** `conf/config_cognitive.json`
- **Agent behavior:** `prompts/` directory contains all agent prompts
- **User settings:** Web UI Settings page or environment variables

## Troubleshooting

### Common Issues
- **OpenCog not available:** Expected in local setup. Use Docker for full cognitive features.
- **Playwright install fails:** Use Docker or skip browser functionality
- **Web UI startup warnings:** Model download warnings are expected, server still works
- **Long installation times:** This is normal, PyTorch and ML libraries are large

### Network-Related Issues  
- **Model download failures:** Expected in restricted environments
- **Playwright browser download fails:** Use existing browsers or Docker
- **HuggingFace connection errors:** Set offline mode or use cached models

### Performance Notes
- **First startup slow:** Model loading and initialization 
- **Docker preferred:** For full functionality including OpenCog cognitive features
- **Memory usage:** High due to PyTorch, transformers, and ML models loaded

## Development Workflow

1. **Test basic functionality:** `python3 -c "import agent; print('Framework OK')"`
2. **Make your changes** to relevant files in python/, webui/, conf/, or prompts/
3. **Test imports work:** Re-run import tests for affected components
4. **Test web UI:** If UI changes made, start server and verify functionality
5. **Validate core features:** Run validation scenarios appropriate to your changes
6. **Full system test:** If major changes, test both local and Docker setups

**Remember:** This is a complex AI agent framework with cognitive architecture. Test thoroughly and use Docker for the most reliable experience.