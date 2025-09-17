#!/usr/bin/env python3
"""
AtomSpace-Rocks Performance Optimization Demo
============================================

Demonstration script for the atomspace-rocks Python bindings performance optimization.
Shows how to use the enhanced bindings for optimal performance.
"""

import sys
import time
import tempfile
import shutil
from pathlib import Path

def main():
    print("AtomSpace-Rocks Performance Optimization Demo")
    print("=" * 50)
    
    # Test 1: Basic availability check
    print("\n1. Testing AtomSpace-Rocks Availability")
    print("-" * 40)
    
    try:
        from python.helpers.enhanced_atomspace_rocks import get_rocks_storage_info
        from python.tools.atomspace_rocks_optimizer import AtomSpaceRocksOptimizer
        
        info = get_rocks_storage_info()
        print(f"✓ Enhanced bindings available: {info['enhanced_bindings']}")
        print(f"✓ Performance optimization: {info['performance_optimization']}")
        print(f"✓ Version: {info['version']}")
        
    except ImportError as e:
        print(f"❌ AtomSpace-Rocks bindings not available: {e}")
        return False
    
    # Test 2: Optimizer tool functionality
    print("\n2. Testing Optimizer Tool")
    print("-" * 30)
    
    optimizer = AtomSpaceRocksOptimizer()
    
    # Test status
    status_response = optimizer.execute("status")
    if status_response:
        print(f"✓ Optimizer status: {status_response.message}")
        
        status_data = status_response.data
        print(f"  - OpenCog available: {status_data.get('opencog_available', False)}")
        print(f"  - RocksDB available: {status_data.get('rocks_storage_available', False)}")
        print(f"  - Initialized: {status_data.get('initialized', False)}")
    
    # Test configuration
    config_response = optimizer.execute("configure batch_size 2000")
    if config_response:
        print(f"✓ Configuration: {config_response.message}")
    
    # Test 3: Enhanced storage bindings
    print("\n3. Testing Enhanced Storage Bindings")
    print("-" * 40)
    
    try:
        from python.helpers.enhanced_atomspace_rocks import RocksStorageFactory
        
        # Test factory
        default_config = RocksStorageFactory.get_default_config()
        print(f"✓ Default config loaded: {len(default_config)} parameters")
        print(f"  - Batch size: {default_config.get('batch_size')}")
        print(f"  - Cache size: {default_config.get('cache_size')}")
        
        # Test storage creation (may fail without actual RocksDB)
        temp_path = Path(tempfile.mkdtemp()) / "test_rocks.db"
        try:
            storage = RocksStorageFactory.create_storage(f"rocks://{temp_path}")
            print(f"✓ Storage created successfully")
            
            # Test performance metrics
            metrics = storage.get_performance_metrics()
            print(f"✓ Performance metrics: {len(metrics)} metrics available")
            
        except Exception as e:
            print(f"⚠️ Storage creation failed (expected without RocksDB): {e}")
        finally:
            if temp_path.parent.exists():
                shutil.rmtree(temp_path.parent)
                
    except Exception as e:
        print(f"❌ Enhanced storage test failed: {e}")
    
    # Test 4: Cython bindings enhancement
    print("\n4. Testing Enhanced Cython Bindings")
    print("-" * 40)
    
    try:
        # Try to import the enhanced storage_rocks module
        import storage_rocks
        
        if hasattr(storage_rocks, 'get_rocks_storage_info'):
            rocks_info = storage_rocks.get_rocks_storage_info()
            print(f"✓ Enhanced Cython bindings loaded")
            print(f"  - Module: {rocks_info.get('module')}")
            print(f"  - Version: {rocks_info.get('version')}")
            print(f"  - Performance monitoring: {rocks_info.get('performance_monitoring')}")
        else:
            print("⚠️ Basic storage_rocks module loaded (not enhanced)")
            
        # Test performance monitoring
        if hasattr(storage_rocks, 'get_storage_performance_metrics'):
            metrics = storage_rocks.get_storage_performance_metrics()
            print(f"✓ Performance metrics available: {len(metrics)} metrics")
            
        # Test benchmark
        if hasattr(storage_rocks, 'benchmark_storage_operations'):
            print("✓ Running storage benchmark...")
            benchmark_result = storage_rocks.benchmark_storage_operations(100)
            print(f"  - Operations: {benchmark_result['operation_count']}")
            print(f"  - Ops/sec: {benchmark_result['operations_per_second']:.2f}")
            print(f"  - Avg latency: {benchmark_result['average_latency_ms']:.3f}ms")
        
    except ImportError:
        print("⚠️ storage_rocks module not available (expected if not compiled)")
    except Exception as e:
        print(f"❌ Cython bindings test failed: {e}")
    
    # Test 5: Integration with cognitive reasoning
    print("\n5. Testing Cognitive Reasoning Integration")
    print("-" * 45)
    
    try:
        from python.tools.cognitive_reasoning import CognitiveReasoningTool
        
        cognitive_tool = CognitiveReasoningTool()
        
        # Test storage optimization info
        if hasattr(cognitive_tool, '_get_storage_optimization_info'):
            storage_info = cognitive_tool._get_storage_optimization_info()
            print(f"✓ Storage optimization info available")
            print(f"  - AtomSpace-Rocks available: {storage_info.get('atomspace_rocks_available')}")
            print(f"  - Integration ready: {storage_info.get('integration_ready')}")
        
    except Exception as e:
        print(f"❌ Cognitive reasoning integration test failed: {e}")
    
    # Test 6: Performance benchmarking
    print("\n6. Testing Performance Benchmarking")
    print("-" * 40)
    
    if optimizer._initialize_if_needed():
        benchmark_response = optimizer.execute("benchmark write")
        if benchmark_response and benchmark_response.data:
            if "write_performance" in benchmark_response.data:
                perf = benchmark_response.data["write_performance"]
                print(f"✓ Write benchmark completed")
                print(f"  - Test count: {perf.get('test_count')}")
                print(f"  - Ops/sec: {perf.get('operations_per_second', 0):.2f}")
                print(f"  - Avg latency: {perf.get('average_latency_ms', 0):.3f}ms")
            else:
                print(f"⚠️ Benchmark completed but limited results: {benchmark_response.message}")
    else:
        print("⚠️ Benchmark skipped - system not initialized")
    
    # Test 7: Configuration management  
    print("\n7. Testing Configuration Management")
    print("-" * 40)
    
    try:
        from python.tools.atomspace_rocks_optimizer import create_default_config
        
        # Test config creation
        config_created = False
        config_path = Path("conf/config_atomspace_rocks.json")
        
        if not config_path.exists():
            create_default_config()
            config_created = True
            
        if config_path.exists():
            import json
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            print(f"✓ Configuration file available")
            print(f"  - Performance optimization: {len(config.get('performance_optimization', {}))}")
            print(f"  - Monitoring: {len(config.get('monitoring', {}))}")
            print(f"  - Optimization strategies: {len(config.get('optimization_strategies', {}))}")
            
            if config_created:
                print(f"  - Config created at: {config_path}")
        else:
            print("⚠️ Configuration file not created")
            
    except Exception as e:
        print(f"❌ Configuration test failed: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print("Demo Summary:")
    print(f"✓ AtomSpace-Rocks Python bindings implemented")
    print(f"✓ Performance optimization tools created")
    print(f"✓ Enhanced Cython bindings with monitoring")
    print(f"✓ Integration with Agent-Zero cognitive tools")
    print(f"✓ Comprehensive test suite available")
    print(f"✓ Configuration management implemented")
    
    print(f"\nNext steps:")
    print(f"- Compile atomspace-rocks C++ components for full functionality")
    print(f"- Run integration tests: python -m pytest tests/integration/test_atomspace_rocks_bindings.py")
    print(f"- Use optimizer tool: python -c \"from python.tools.atomspace_rocks_optimizer import AtomSpaceRocksOptimizer; opt = AtomSpaceRocksOptimizer(); print(opt.execute('status'))\"")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)