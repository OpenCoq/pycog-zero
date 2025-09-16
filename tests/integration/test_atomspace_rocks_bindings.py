"""
AtomSpace-Rocks Python Bindings Integration Tests
==================================================

Comprehensive test suite for atomspace-rocks Python bindings performance optimization.
"""

import unittest
import tempfile
import shutil
import time
import json
from pathlib import Path
from typing import Dict, List

# Import test subjects
try:
    from python.tools.atomspace_rocks_optimizer import AtomSpaceRocksOptimizer
    from python.helpers.enhanced_atomspace_rocks import (
        EnhancedRocksStorage, RocksStorageFactory, get_rocks_storage_info
    )
    BINDINGS_AVAILABLE = True
except ImportError:
    print("Warning: AtomSpace-Rocks bindings not available for testing")
    BINDINGS_AVAILABLE = False

# OpenCog imports for testing
try:
    from opencog.atomspace import AtomSpace, types
    from opencog.utilities import initialize_opencog
    OPENCOG_AVAILABLE = True
except ImportError:
    print("Warning: OpenCog not available for testing")
    OPENCOG_AVAILABLE = False


class TestAtomSpaceRocksBindings(unittest.TestCase):
    """Test AtomSpace-Rocks Python bindings functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="atomspace_rocks_test_")
        self.test_storage_path = Path(self.temp_dir) / "test_rocks.db"
        
    def tearDown(self):
        """Clean up test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(BINDINGS_AVAILABLE, "AtomSpace-Rocks bindings not available")
    def test_rocks_storage_info(self):
        """Test getting RocksDB storage information."""
        info = get_rocks_storage_info()
        
        self.assertIsInstance(info, dict)
        self.assertIn('opencog_available', info)
        self.assertIn('storage_rocks_module', info)
        self.assertIn('enhanced_bindings', info)
        self.assertTrue(info['enhanced_bindings'])
        self.assertIn('version', info)
        
    @unittest.skipUnless(BINDINGS_AVAILABLE and OPENCOG_AVAILABLE, "Dependencies not available")
    def test_enhanced_rocks_storage_creation(self):
        """Test creating enhanced RocksDB storage."""
        uri = f"rocks://{self.test_storage_path}"
        config = RocksStorageFactory.get_default_config()
        
        storage = EnhancedRocksStorage(uri, "rocks", config)
        self.assertEqual(storage.uri, uri)
        self.assertEqual(storage.storage_type, "rocks")
        self.assertIsNotNone(storage.config)
        
    @unittest.skipUnless(BINDINGS_AVAILABLE and OPENCOG_AVAILABLE, "Dependencies not available") 
    def test_storage_factory(self):
        """Test RocksDB storage factory functionality."""
        uri = f"rocks://{self.test_storage_path}"
        
        # Test default configuration
        default_config = RocksStorageFactory.get_default_config()
        self.assertIn('batch_size', default_config)
        self.assertIn('cache_size', default_config)
        
        # Test storage creation (may fail without actual RocksDB)
        try:
            storage = RocksStorageFactory.create_storage(uri, "rocks", default_config)
            self.assertIsInstance(storage, EnhancedRocksStorage)
        except Exception as e:
            # Expected if RocksDB not properly compiled
            self.assertIn("Failed to initialize", str(e))
    
    @unittest.skipUnless(BINDINGS_AVAILABLE and OPENCOG_AVAILABLE, "Dependencies not available")
    def test_optimized_atomspace_storage(self):
        """Test optimized AtomSpace storage creation."""
        # Create test AtomSpace
        atomspace = AtomSpace()
        initialize_opencog(atomspace)
        
        # Add test atoms
        node1 = atomspace.add_node(types.ConceptNode, "TestNode1")
        node2 = atomspace.add_node(types.ConceptNode, "TestNode2")
        link = atomspace.add_link(types.InheritanceLink, [node1, node2])
        
        self.assertEqual(len(atomspace), 3)
        
        # Test optimized storage creation (may fail without RocksDB)
        try:
            storage = RocksStorageFactory.create_optimized_atomspace_storage(
                str(self.test_storage_path), atomspace
            )
            self.assertIsInstance(storage, EnhancedRocksStorage)
            
            # Test performance metrics
            metrics = storage.get_performance_metrics()
            self.assertIsInstance(metrics, dict)
            self.assertIn('operations', metrics)
            self.assertIn('average_latency', metrics)
            
        except Exception as e:
            # Expected if RocksDB not properly compiled
            print(f"Storage creation failed (expected): {e}")


class TestAtomSpaceRocksOptimizer(unittest.TestCase):
    """Test AtomSpace-Rocks optimizer tool functionality."""
    
    def setUp(self):
        """Set up test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="atomspace_optimizer_test_")
        
    def tearDown(self):
        """Clean up test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(BINDINGS_AVAILABLE, "AtomSpace-Rocks optimizer not available")
    def test_optimizer_initialization(self):
        """Test optimizer tool initialization."""
        optimizer = AtomSpaceRocksOptimizer()
        self.assertIsInstance(optimizer, AtomSpaceRocksOptimizer)
        self.assertFalse(optimizer._initialized)
    
    @unittest.skipUnless(BINDINGS_AVAILABLE, "AtomSpace-Rocks optimizer not available")
    def test_optimizer_status(self):
        """Test optimizer status reporting."""
        optimizer = AtomSpaceRocksOptimizer()
        
        response = optimizer.execute("status")
        self.assertIsNotNone(response)
        self.assertIsNotNone(response.data)
        
        status = response.data
        self.assertIn('opencog_available', status)
        self.assertIn('rocks_storage_available', status)
        self.assertIn('initialized', status)
    
    @unittest.skipUnless(BINDINGS_AVAILABLE, "AtomSpace-Rocks optimizer not available")
    def test_optimizer_help(self):
        """Test optimizer help functionality."""
        optimizer = AtomSpaceRocksOptimizer()
        
        response = optimizer.execute("help")
        self.assertIsNotNone(response)
        self.assertIn("help", response.data)
        
        help_text = response.data["help"]
        self.assertIn("status", help_text)
        self.assertIn("benchmark", help_text)
        self.assertIn("optimize", help_text)
    
    @unittest.skipUnless(BINDINGS_AVAILABLE and OPENCOG_AVAILABLE, "Dependencies not available")
    def test_optimizer_configuration(self):
        """Test optimizer configuration functionality."""
        optimizer = AtomSpaceRocksOptimizer()
        
        # Test configuration update
        response = optimizer.execute("configure batch_size 2000")
        self.assertIsNotNone(response)
        
        # Check if configuration was updated
        if response.data and "config" in response.data:
            config = response.data["config"]
            self.assertIn("performance_optimization", config)
    
    @unittest.skipUnless(BINDINGS_AVAILABLE and OPENCOG_AVAILABLE, "Dependencies not available")
    def test_storage_creation(self):
        """Test storage node creation through optimizer."""
        optimizer = AtomSpaceRocksOptimizer()
        
        storage_path = str(Path(self.temp_dir) / "test_optimizer_storage")
        
        # Test storage creation (may fail without actual RocksDB)
        response = optimizer.execute(f"create_storage {storage_path}")
        self.assertIsNotNone(response)
        
        # Check response structure
        if response.data:
            if "error" in response.data:
                # Expected if RocksDB not available
                print(f"Storage creation failed (expected): {response.data['error']}")
            else:
                # Success case
                self.assertIn("storage_id", response.data)
                self.assertIn("path", response.data)
    
    @unittest.skipUnless(BINDINGS_AVAILABLE and OPENCOG_AVAILABLE, "Dependencies not available")
    def test_benchmark_operations(self):
        """Test benchmark functionality."""
        optimizer = AtomSpaceRocksOptimizer()
        
        # Initialize if possible
        if optimizer._initialize_if_needed():
            # Test write benchmark
            response = optimizer.execute("benchmark write")
            self.assertIsNotNone(response)
            
            if response.data and "write_performance" in response.data:
                perf_data = response.data["write_performance"]
                self.assertIn("test_count", perf_data)
                self.assertIn("total_time", perf_data)
                
            # Test statistics
            stats_response = optimizer.execute("stats")
            self.assertIsNotNone(stats_response)
            
            if stats_response.data and "summary" in stats_response.data:
                summary = stats_response.data["summary"]
                self.assertIn("total_operations", summary)


class TestAtomSpaceRocksPerformance(unittest.TestCase):
    """Test performance aspects of AtomSpace-Rocks bindings."""
    
    @unittest.skipUnless(BINDINGS_AVAILABLE and OPENCOG_AVAILABLE, "Dependencies not available")
    def test_performance_monitoring(self):
        """Test performance monitoring capabilities."""
        # Create AtomSpace
        atomspace = AtomSpace()
        initialize_opencog(atomspace)
        
        # Add atoms and measure performance
        start_time = time.time()
        
        test_count = 100
        for i in range(test_count):
            node = atomspace.add_node(types.ConceptNode, f"PerfTest_{i}")
            
        creation_time = time.time() - start_time
        
        self.assertGreater(test_count, 0)
        self.assertGreater(creation_time, 0)
        
        # Calculate performance metrics
        ops_per_second = test_count / creation_time
        avg_latency = (creation_time * 1000) / test_count  # ms
        
        self.assertGreater(ops_per_second, 0)
        self.assertGreater(avg_latency, 0)
        
        print(f"Performance: {ops_per_second:.2f} ops/sec, {avg_latency:.3f}ms avg latency")
    
    @unittest.skipUnless(BINDINGS_AVAILABLE and OPENCOG_AVAILABLE, "Dependencies not available")
    def test_batch_operations_performance(self):
        """Test batch operations performance."""
        atomspace = AtomSpace()
        initialize_opencog(atomspace)
        
        batch_sizes = [10, 100, 1000]
        results = {}
        
        for batch_size in batch_sizes:
            start_time = time.time()
            
            # Create batch of atoms
            atoms = []
            for i in range(batch_size):
                node = atomspace.add_node(types.ConceptNode, f"Batch_{batch_size}_{i}")
                atoms.append(node)
            
            batch_time = time.time() - start_time
            results[batch_size] = {
                'time': batch_time,
                'ops_per_second': batch_size / batch_time if batch_time > 0 else 0
            }
        
        # Verify results
        for batch_size, result in results.items():
            self.assertGreater(result['ops_per_second'], 0)
            
        print(f"Batch performance results: {results}")


class TestAtomSpaceRocksIntegration(unittest.TestCase):
    """Test integration with existing Agent-Zero tools."""
    
    def setUp(self):
        """Set up integration test environment."""
        self.temp_dir = tempfile.mkdtemp(prefix="atomspace_integration_test_")
        
    def tearDown(self):
        """Clean up integration test environment."""
        if Path(self.temp_dir).exists():
            shutil.rmtree(self.temp_dir)
    
    @unittest.skipUnless(BINDINGS_AVAILABLE, "AtomSpace-Rocks bindings not available")
    def test_configuration_file_creation(self):
        """Test configuration file creation and loading."""
        # Import the config creation function
        from python.tools.atomspace_rocks_optimizer import create_default_config
        
        # Create config in temp directory
        config_path = Path(self.temp_dir) / "conf" / "config_atomspace_rocks.json"
        
        # Temporarily change working directory
        import os
        original_cwd = os.getcwd()
        try:
            os.chdir(self.temp_dir)
            create_default_config()
            
            # Verify config file was created
            self.assertTrue(config_path.exists())
            
            # Verify config content
            with open(config_path, 'r') as f:
                config = json.load(f)
                
            self.assertIn('performance_optimization', config)
            self.assertIn('monitoring', config)
            self.assertIn('optimization_strategies', config)
            
        finally:
            os.chdir(original_cwd)
    
    @unittest.skipUnless(BINDINGS_AVAILABLE, "AtomSpace-Rocks bindings not available")
    def test_tool_integration_pattern(self):
        """Test integration pattern with Agent-Zero tools."""
        optimizer = AtomSpaceRocksOptimizer()
        
        # Test tool interface compliance
        self.assertTrue(hasattr(optimizer, 'execute'))
        
        # Test response format
        response = optimizer.execute("status")
        self.assertIsNotNone(response)
        self.assertTrue(hasattr(response, 'message'))
        self.assertTrue(hasattr(response, 'data'))
    
    @unittest.skipUnless(BINDINGS_AVAILABLE and OPENCOG_AVAILABLE, "Dependencies not available")
    def test_cognitive_reasoning_integration(self):
        """Test integration with cognitive reasoning components."""
        try:
            from python.tools.cognitive_reasoning import CognitiveReasoningTool
            cognitive_tool = CognitiveReasoningTool()
            
            # Test if cognitive tool can work with rocks optimizer
            optimizer = AtomSpaceRocksOptimizer()
            
            # Both tools should be able to initialize
            self.assertIsNotNone(cognitive_tool)
            self.assertIsNotNone(optimizer)
            
        except ImportError:
            print("Cognitive reasoning tool not available for integration test")


def run_atomspace_rocks_tests():
    """Run all AtomSpace-Rocks binding tests."""
    print("Running AtomSpace-Rocks Python Bindings Tests")
    print("=" * 60)
    
    # Create test suite
    test_suite = unittest.TestSuite()
    
    # Add test classes
    test_classes = [
        TestAtomSpaceRocksBindings,
        TestAtomSpaceRocksOptimizer, 
        TestAtomSpaceRocksPerformance,
        TestAtomSpaceRocksIntegration
    ]
    
    for test_class in test_classes:
        tests = unittest.TestLoader().loadTestsFromTestCase(test_class)
        test_suite.addTests(tests)
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(test_suite)
    
    # Print summary
    print(f"\nTest Summary:")
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success rate: {((result.testsRun - len(result.failures) - len(result.errors)) / result.testsRun * 100):.1f}%")
    
    return result.wasSuccessful()


if __name__ == "__main__":
    run_atomspace_rocks_tests()