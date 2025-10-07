# PyCog-Zero Troubleshooting Guide

Comprehensive troubleshooting guide for common issues, debugging techniques, and problem resolution in PyCog-Zero.

## 📋 Table of Contents

### Quick Start Issues
- [Installation and Setup Problems](#installation-and-setup-problems)
- [Docker and Container Issues](#docker-and-container-issues)
- [Web UI Access Problems](#web-ui-access-problems)

### Cognitive System Issues
- [OpenCog Integration Problems](#opencog-integration-problems)
- [Reasoning Tool Errors](#reasoning-tool-errors)
- [Memory and AtomSpace Issues](#memory-and-atomspace-issues)
- [Performance and Timeout Issues](#performance-and-timeout-issues)

### Frequently Asked Questions
- [Common Usage Questions](#common-usage-questions)
- [Configuration Questions](#configuration-questions)
- [Integration Questions](#integration-questions)

### Advanced Troubleshooting
- [Diagnostic Tools](#diagnostic-tools)
- [Log Analysis](#log-analysis)
- [System Recovery](#system-recovery)

---

## Installation and Setup Problems

### Python Dependencies Missing

**Problem:** `ModuleNotFoundError` for core dependencies like `nest_asyncio`, `torch`, or cognitive tools.

**Solution:**
```bash
# Install dependencies with extended timeout (7-10 minutes)
pip install -r requirements.txt --timeout 900

# For cognitive features (may fail without OpenCog binaries)
pip install -r requirements-cognitive.txt

# Verify installation
python3 -c "import agent; print('✅ Framework available')"
python3 -c "from python.tools.cognitive_reasoning import CognitiveReasoningTool; print('✅ Cognitive tools available')"
```

### OpenCog Not Available

**Problem:** `ImportError: No module named 'opencog'` or OpenCog-related errors.

**Solutions:**

1. **Use Docker (Recommended):**
```bash
# Pull and run with cognitive features
docker pull agent0ai/agent-zero:latest
docker run -p 50001:80 agent0ai/agent-zero:latest
```

2. **Disable OpenCog for Testing:**
```bash
export OPENCOG_ENABLED=false
python3 agent.py
```

3. **Build Local Docker with OpenCog:**
```bash
docker build -f DockerfileLocal -t pycog-zero-local \
  --build-arg CACHE_DATE=$(date +%Y-%m-%d:%H:%M:%S) .
```

### Long Installation Times

**Problem:** `pip install` taking extremely long or timing out.

**Expected Behavior:** Installation takes 7-10 minutes for main dependencies. This is normal due to PyTorch, transformers, and FAISS.

**Solutions:**
```bash
# Use extended timeout - NEVER CANCEL
pip install -r requirements.txt --timeout 900

# Alternative: Use Docker
docker pull agent0ai/agent-zero:latest
```

---

## Docker and Container Issues

### Docker Build Failures

**Problem:** Docker build fails with wheel building errors or timeouts.

**Solutions:**

1. **Use Pre-built Image:**
```bash
docker pull agent0ai/agent-zero:latest
docker run -p 50001:80 agent0ai/agent-zero:latest
```

2. **Build with More Resources:**
```bash
docker build -f DockerfileLocal \
  --memory=4g \
  --timeout=1800 \
  -t pycog-zero-local .
```

3. **Clean Docker Cache:**
```bash
docker system prune -a
docker build --no-cache -f DockerfileLocal -t pycog-zero-local .
```

### Container Won't Start

**Problem:** Docker container exits immediately or fails to start.

**Debugging Steps:**
```bash
# Check container logs
docker logs <container_id>

# Run in interactive mode
docker run -it --entrypoint /bin/bash agent0ai/agent-zero:latest

# Check port conflicts
netstat -an | grep 50001
```

---

## Web UI Access Problems

### Cannot Access http://localhost:50001

**Problem:** Browser shows "connection refused" or timeout when accessing the web UI.

**Solutions:**

1. **Check if Server is Running:**
```bash
# Start web UI manually
python3 run_ui.py --host 0.0.0.0 --port 50001

# Check if port is open
curl -I http://localhost:50001/
```

2. **Docker Port Mapping:**
```bash
# Ensure proper port mapping
docker run -p 50001:80 agent0ai/agent-zero:latest

# Check Docker port status
docker port <container_name>
```

3. **Firewall/Network Issues:**
```bash
# Test local connectivity
telnet localhost 50001

# Check if bound to correct interface
netstat -an | grep 50001
```

### Web UI Loads but Features Don't Work

**Problem:** Web UI loads but cognitive features, chat, or tools don't respond.

**Debugging:**
```bash
# Check browser console for JavaScript errors
# Open Developer Tools → Console

# Check server logs
tail -f logs/*.log

# Verify API endpoints
curl -X POST http://localhost:50001/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "test"}'
```

---

## OpenCog Integration Problems

### AtomSpace Initialization Failed

**Problem:** `RuntimeError: AtomSpace initialization failed` or similar AtomSpace errors.

**Debugging Script:**
```python
#!/usr/bin/env python3
"""Debug AtomSpace issues"""

def debug_atomspace():
    print("🔍 Debugging AtomSpace Integration")
    
    try:
        from opencog.atomspace import AtomSpace
        atomspace = AtomSpace()
        print("✅ AtomSpace initialized successfully")
        
        # Test basic operations
        from opencog.type_constructors import ConceptNode
        test_concept = ConceptNode("test")
        print(f"✅ Created concept: {test_concept}")
        
    except ImportError as e:
        print(f"❌ OpenCog import failed: {e}")
        print("💡 Solution: Use Docker deployment")
        print("   docker run -p 50001:80 agent0ai/agent-zero:latest")
        
    except Exception as e:
        print(f"❌ AtomSpace error: {e}")
        print("💡 Try restarting or using memory backend")

if __name__ == "__main__":
    debug_atomspace()
```

**Solutions:**
1. Use Docker deployment for full OpenCog support
2. Disable OpenCog: `export OPENCOG_ENABLED=false`
3. Use memory-only AtomSpace backend

---

## Reasoning Tool Errors

### Reasoning Timeout Exceeded

**Problem:** `TimeoutError: Reasoning operation exceeded timeout`

**Solutions:**

1. **Increase Timeout:**
```python
from python.tools.cognitive_reasoning import CognitiveReasoningTool

reasoning_tool = CognitiveReasoningTool({
    "reasoning_timeout": 120  # 2 minutes instead of 30 seconds
})
```

2. **Reduce Complexity:**
```python
result = await reasoning_tool.execute({
    "query": "your question",
    "max_steps": 10,  # Reduce from default 100
    "confidence_threshold": 0.7  # Stop early if confident
})
```

3. **Enable Caching:**
```python
reasoning_tool = CognitiveReasoningTool({
    "reasoning_cache": True,
    "cache_size": 1000
})
```

### Reasoning Returns Empty Results

**Problem:** Reasoning tool returns empty or nonsensical results.

**Debugging:**
```python
# Test with simple query first
simple_result = await reasoning_tool.execute({
    "query": "What is 2 + 2?",
    "reasoning_mode": "logical",
    "max_steps": 5
})

print(f"Simple test: {simple_result}")

# Check reasoning steps
if 'reasoning_steps' in simple_result:
    for step in simple_result['reasoning_steps']:
        print(f"Step: {step}")
```

---

## Memory and AtomSpace Issues

### Memory Storage Failed

**Problem:** `StorageError: Failed to store knowledge in AtomSpace`

**Solutions:**

1. **Check Storage Backend:**
```python
from python.tools.cognitive_memory import CognitiveMemoryTool

# Fallback to memory backend
memory_tool = CognitiveMemoryTool({
    "atomspace_backend": "memory",  # Instead of "rocks"
    "persistence_enabled": False
})
```

2. **Clear Storage Space:**
```bash
# Check disk space
df -h

# Clean up old memory files
rm -rf memory/*.tmp
```

3. **Reset AtomSpace:**
```python
# Create fresh AtomSpace
from opencog.atomspace import AtomSpace
atomspace = AtomSpace()
```

### High Memory Usage

**Problem:** Memory usage grows continuously during operation.

**Monitoring Script:**
```python
#!/usr/bin/env python3
"""Monitor memory usage"""

import psutil
import time
from python.tools.cognitive_memory import CognitiveMemoryTool

def monitor_memory():
    process = psutil.Process()
    memory_tool = CognitiveMemoryTool()
    
    baseline = process.memory_info().rss / 1024 / 1024  # MB
    print(f"Baseline memory: {baseline:.1f} MB")
    
    # Monitor during operations
    for i in range(10):
        # Perform memory operation
        memory_tool.store_knowledge(f"Test knowledge {i}")
        
        current = process.memory_info().rss / 1024 / 1024
        print(f"After operation {i+1}: {current:.1f} MB (+{current-baseline:.1f})")
        
        time.sleep(1)

if __name__ == "__main__":
    monitor_memory()
```

**Solutions:**
- Enable automatic cleanup
- Implement periodic garbage collection
- Use memory-efficient backends

---

## Performance and Timeout Issues

### Slow Response Times

**Problem:** Cognitive operations taking much longer than expected (>2 seconds).

**Performance Testing:**
```python
#!/usr/bin/env python3
"""Test cognitive performance"""

import time
import asyncio
from python.tools.cognitive_reasoning import CognitiveReasoningTool

async def test_performance():
    reasoning_tool = CognitiveReasoningTool()
    
    test_queries = [
        "What is the capital of France?",
        "Explain machine learning in simple terms",
        "What are the benefits of renewable energy?"
    ]
    
    for query in test_queries:
        start_time = time.time()
        
        result = await reasoning_tool.execute({
            "query": query,
            "reasoning_mode": "logical",
            "max_steps": 20
        })
        
        duration = time.time() - start_time
        print(f"Query: {query}")
        print(f"Time: {duration:.2f}s")
        print(f"Steps: {len(result.get('reasoning_steps', []))}")
        print("---")

if __name__ == "__main__":
    asyncio.run(test_performance())
```

**Optimization Strategies:**
1. Reduce `max_steps` parameter
2. Set `confidence_threshold` for early termination
3. Enable result caching
4. Use simpler reasoning modes

---

## Common Usage Questions

### Q: How do I enable cognitive features?

**A:** Set the cognitive mode configuration:

```python
# In your code
agent_config = {
    "cognitive_mode": True,
    "opencog_enabled": True
}

# Or as environment variable
export COGNITIVE_MODE=true
export OPENCOG_ENABLED=true
```

### Q: Can I use PyCog-Zero without OpenCog?

**A:** Yes, many features work without OpenCog:

```python
# Disable OpenCog
agent_config = {
    "cognitive_mode": True,
    "opencog_enabled": False
}
```

You'll lose AtomSpace integration but keep reasoning and memory tools.

### Q: How do I work with files and directories?

**A:** Place files in the accessible directory structure:

```bash
# Files in these directories are accessible
work_dir/          # Working directory
memory/           # Persistent memory
knowledge/        # Knowledge files
examples/         # Example files
```

### Q: How do I retain memory between sessions?

**A:** Enable persistent storage:

```python
memory_config = {
    "persistence_enabled": True,
    "atomspace_backend": "rocks",  # Persistent backend
    "backup_enabled": True
}
```

### Q: My reasoning takes too long. How do I speed it up?

**A:** Use these optimization techniques:

```python
# Quick reasoning configuration
quick_config = {
    "max_steps": 10,           # Reduce steps
    "reasoning_timeout": 30,   # Shorter timeout
    "confidence_threshold": 0.8,  # Stop when confident
    "reasoning_cache": True    # Enable caching
}
```

---

## Diagnostic Tools

### System Health Check

```python
#!/usr/bin/env python3
"""Comprehensive system health check"""

import asyncio
import psutil
from pathlib import Path

async def health_check():
    print("🏥 PyCog-Zero Health Check")
    print("=" * 50)
    
    checks = [
        ("System Resources", check_system_resources),
        ("Python Environment", check_python_environment),
        ("PyCog Components", check_pycog_components),
        ("OpenCog Integration", check_opencog),
        ("File Permissions", check_file_permissions)
    ]
    
    results = {}
    
    for check_name, check_func in checks:
        print(f"\n🔍 {check_name}...")
        try:
            result = await check_func()
            results[check_name] = result
            print(f"✅ {check_name}: {result['status']}")
        except Exception as e:
            results[check_name] = {"status": "failed", "error": str(e)}
            print(f"❌ {check_name}: {e}")
    
    # Generate summary
    print(f"\n📊 Health Check Summary:")
    healthy_count = sum(1 for r in results.values() if r.get('status') == 'healthy')
    total_count = len(results)
    
    print(f"  Healthy components: {healthy_count}/{total_count}")
    
    if healthy_count == total_count:
        print("  🎉 All systems healthy!")
    else:
        print("  ⚠️ Some issues detected - see details above")

async def check_system_resources():
    """Check system resource availability."""
    cpu_usage = psutil.cpu_percent(interval=1)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Determine health
    issues = []
    if cpu_usage > 90:
        issues.append(f"High CPU usage: {cpu_usage:.1f}%")
    if memory.percent > 90:
        issues.append(f"High memory usage: {memory.percent:.1f}%")
    if (disk.used / disk.total) > 0.9:
        issues.append(f"Low disk space: {(disk.used/disk.total)*100:.1f}% used")
    
    return {
        "status": "healthy" if not issues else "warning",
        "cpu_usage": cpu_usage,
        "memory_usage": memory.percent,
        "disk_usage": (disk.used / disk.total) * 100,
        "issues": issues
    }

async def check_python_environment():
    """Check Python environment and dependencies."""
    import sys
    
    critical_modules = [
        "agent", "torch", "transformers", "numpy", 
        "pandas", "requests", "aiohttp"
    ]
    
    missing_modules = []
    for module in critical_modules:
        try:
            __import__(module)
        except ImportError:
            missing_modules.append(module)
    
    return {
        "status": "healthy" if not missing_modules else "failed",
        "python_version": sys.version,
        "missing_modules": missing_modules
    }

async def check_pycog_components():
    """Check PyCog-Zero specific components."""
    components = {
        "agent": "agent",
        "cognitive_reasoning": "python.tools.cognitive_reasoning",
        "cognitive_memory": "python.tools.cognitive_memory",
        "meta_cognition": "python.tools.meta_cognition"
    }
    
    failed_components = []
    for name, module_path in components.items():
        try:
            __import__(module_path)
        except ImportError:
            failed_components.append(name)
    
    return {
        "status": "healthy" if not failed_components else "failed",
        "total_components": len(components),
        "working_components": len(components) - len(failed_components),
        "failed_components": failed_components
    }

async def check_opencog():
    """Check OpenCog integration."""
    try:
        from opencog.atomspace import AtomSpace
        atomspace = AtomSpace()
        return {"status": "healthy", "atomspace_available": True}
    except ImportError:
        return {"status": "warning", "atomspace_available": False, "message": "OpenCog not installed"}
    except Exception as e:
        return {"status": "failed", "atomspace_available": False, "error": str(e)}

async def check_file_permissions():
    """Check file and directory permissions."""
    critical_dirs = ["python/tools", "memory", "logs", "conf"]
    permission_issues = []
    
    for dir_path in critical_dirs:
        path = Path(dir_path)
        if not path.exists():
            permission_issues.append(f"{dir_path} does not exist")
        elif not path.is_dir():
            permission_issues.append(f"{dir_path} is not a directory")
        else:
            # Check read/write permissions
            test_file = path / ".permission_test"
            try:
                test_file.write_text("test")
                test_file.unlink()
            except:
                permission_issues.append(f"{dir_path} is not writable")
    
    return {
        "status": "healthy" if not permission_issues else "failed",
        "issues": permission_issues
    }

if __name__ == "__main__":
    asyncio.run(health_check())
```

### Quick Fix Script

```bash
#!/bin/bash
# quick_fix.sh - Automated fixes for common issues

echo "🔧 PyCog-Zero Quick Fix"
echo "======================"

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    echo "⚠️ Warning: Running as root"
fi

# Fix 1: Install missing dependencies
echo "1. Checking Python dependencies..."
if ! python3 -c "import agent" 2>/dev/null; then
    echo "   Installing missing dependencies..."
    pip install -r requirements.txt --timeout 600
else
    echo "   ✅ Dependencies OK"
fi

# Fix 2: Check Docker
echo "2. Checking Docker..."
if ! docker --version &>/dev/null; then
    echo "   ❌ Docker not installed"
    echo "   💡 Install Docker: https://docs.docker.com/get-docker/"
else
    echo "   ✅ Docker OK"
fi

# Fix 3: Create necessary directories
echo "3. Creating necessary directories..."
mkdir -p memory logs conf knowledge work_dir
chmod 755 memory logs conf knowledge work_dir
echo "   ✅ Directories created"

# Fix 4: Clean temporary files
echo "4. Cleaning temporary files..."
find . -name "*.tmp" -delete 2>/dev/null
find . -name "__pycache__" -type d -exec rm -rf {} + 2>/dev/null
echo "   ✅ Cleanup complete"

# Fix 5: Check port conflicts
echo "5. Checking port conflicts..."
if netstat -an 2>/dev/null | grep -q ":50001"; then
    echo "   ⚠️ Port 50001 in use"
    echo "   💡 Kill existing process or use different port"
else
    echo "   ✅ Port 50001 available"
fi

# Fix 6: Test basic functionality
echo "6. Testing basic functionality..."
if python3 -c "from python.tools.cognitive_reasoning import CognitiveReasoningTool; print('✅ Cognitive tools work')" 2>/dev/null; then
    echo "   ✅ Cognitive tools working"
else
    echo "   ⚠️ Cognitive tools need attention"
    echo "   💡 Try Docker deployment for full features"
fi

echo ""
echo "🎉 Quick fix complete!"
echo "💡 If issues persist, run: python3 docs/troubleshooting_health_check.py"
```

---

## Getting Help

### Community Support

- **GitHub Issues**: https://github.com/OpenCoq/pycog-zero/issues
- **Documentation**: Complete guides in `/docs/`
- **Examples**: Working code in repository root and `/examples/`

### Creating Bug Reports

Use this template for effective bug reports:

```markdown
## Bug Report

**Environment:**
- OS: [e.g., Ubuntu 20.04, macOS 12, Windows 10]
- Python version: [e.g., 3.12.3]
- Installation method: [Docker/pip/source]

**Issue:**
[Clear description of the problem]

**Expected behavior:**
[What you expected to happen]

**Actual behavior:**
[What actually happened]

**Steps to reproduce:**
1. [First step]
2. [Second step]  
3. [Third step]

**Error messages:**
```
[Paste any error messages here]
```

**Additional context:**
[Any other relevant information]
```

### Emergency Contact

For critical system failures:

1. **Stop all processes:**
   ```bash
   pkill -f "python.*agent"
   docker stop $(docker ps -q)
   ```

2. **Create emergency backup:**
   ```bash
   cp -r memory/ backup_$(date +%Y%m%d_%H%M%S)/
   ```

3. **Use safe mode:**
   ```bash
   export SAFE_MODE=true
   export OPENCOG_ENABLED=false
   python3 agent.py
   ```

---

This comprehensive troubleshooting guide should help resolve most common issues with PyCog-Zero. Keep it handy for quick problem resolution!

---

*Last Updated: October 2024 - PyCog-Zero Genesis Phase 5 Complete*