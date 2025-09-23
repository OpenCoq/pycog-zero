#!/usr/bin/env python3
"""
Comprehensive Test Suite Runner for PyCog-Zero
Runs all comprehensive tests and generates a unified report.
"""

import asyncio
import json
import time
import sys
import os
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import all comprehensive test suites
from tests.comprehensive.test_cognitive_functions import run_comprehensive_cognitive_tests
from tests.comprehensive.test_integration import run_comprehensive_integration_tests
from tests.comprehensive.test_performance import run_comprehensive_performance_benchmarks
from tests.comprehensive.test_validation import run_comprehensive_validation_tests
from tests.comprehensive.test_system import run_comprehensive_system_tests


class ComprehensiveTestRunner:
    """Comprehensive test suite runner and coordinator."""
    
    def __init__(self):
        self.test_suites = [
            ("Cognitive Functions", run_comprehensive_cognitive_tests),
            ("Integration", run_comprehensive_integration_tests),
            ("Performance Benchmarks", run_comprehensive_performance_benchmarks),
            ("Validation", run_comprehensive_validation_tests),
            ("System Tests", run_comprehensive_system_tests)
        ]
        self.results = {}
        
    async def run_all_comprehensive_tests(self):
        """Run all comprehensive test suites."""
        print("\n🧠 PYCOG-ZERO COMPREHENSIVE TESTING AND VALIDATION SUITE")
        print("=" * 80)
        print("Implementing Medium-term Roadmap Requirements (Month 2-3)")
        print("Complete cognitive testing and validation suite")
        print("=" * 80)
        
        overall_start_time = time.time()
        
        # Run each test suite
        for suite_name, test_function in self.test_suites:
            print(f"\n🚀 Starting {suite_name} Test Suite")
            print("-" * 50)
            
            suite_start_time = time.time()
            
            try:
                result = await test_function()
                suite_end_time = time.time()
                suite_duration = suite_end_time - suite_start_time
                
                self.results[suite_name] = {
                    "success": True,
                    "result": result,
                    "duration": suite_duration,
                    "error": None
                }
                
                print(f"✅ {suite_name} completed successfully in {suite_duration:.2f} seconds")
                
            except Exception as e:
                suite_end_time = time.time()
                suite_duration = suite_end_time - suite_start_time
                
                self.results[suite_name] = {
                    "success": False,
                    "result": None,
                    "duration": suite_duration,
                    "error": str(e)
                }
                
                print(f"❌ {suite_name} failed after {suite_duration:.2f} seconds: {e}")
        
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        # Generate comprehensive summary report
        summary_report = self.generate_comprehensive_summary(overall_duration)
        
        # Save and display results
        self.save_comprehensive_report(summary_report)
        self.display_comprehensive_summary(summary_report)
        
        return summary_report
    
    def generate_comprehensive_summary(self, total_duration: float) -> Dict[str, Any]:
        """Generate comprehensive summary of all test results."""
        successful_suites = sum(1 for r in self.results.values() if r["success"])
        failed_suites = len(self.results) - successful_suites
        
        # Collect detailed metrics from each suite
        detailed_metrics = {}
        
        for suite_name, result in self.results.items():
            if result["success"] and result["result"]:
                suite_result = result["result"]
                
                # Extract key metrics based on suite type
                if suite_name == "Cognitive Functions":
                    detailed_metrics[suite_name] = {
                        "total_tests": suite_result.get("total_tests", 0),
                        "passed_tests": suite_result.get("passed_tests", 0),
                        "success_rate": suite_result.get("success_rate", 0.0),
                        "duration": result["duration"]
                    }
                    
                elif suite_name == "Integration":
                    detailed_metrics[suite_name] = {
                        "total_tests": suite_result.get("total_tests", 0),
                        "passed_tests": suite_result.get("passed_tests", 0),
                        "success_rate": suite_result.get("success_rate", 0.0),
                        "duration": result["duration"]
                    }
                    
                elif suite_name == "Performance Benchmarks":
                    detailed_metrics[suite_name] = {
                        "total_benchmarks": suite_result.get("total_benchmarks", 0),
                        "benchmarks_completed": len(suite_result.get("benchmarks_executed", [])),
                        "system_info": suite_result.get("system_info", {}),
                        "duration": result["duration"]
                    }
                    
                elif suite_name == "Validation":
                    detailed_metrics[suite_name] = {
                        "total_tests": suite_result.get("total_validation_tests", 0),
                        "passed_tests": suite_result.get("passed_tests", 0),
                        "success_rate": suite_result.get("success_rate", 0.0),
                        "duration": result["duration"]
                    }
                    
                elif suite_name == "System Tests":
                    detailed_metrics[suite_name] = {
                        "total_tests": suite_result.get("total_system_tests", 0),
                        "passed_tests": suite_result.get("passed_tests", 0),
                        "success_rate": suite_result.get("success_rate", 0.0),
                        "duration": result["duration"]
                    }
                    
            else:
                detailed_metrics[suite_name] = {
                    "status": "failed",
                    "error": result["error"],
                    "duration": result["duration"]
                }
        
        # Calculate overall statistics
        total_individual_tests = sum(
            metrics.get("total_tests", metrics.get("total_benchmarks", 0)) 
            for metrics in detailed_metrics.values() 
            if isinstance(metrics, dict) and "status" not in metrics
        )
        
        total_passed_tests = sum(
            metrics.get("passed_tests", metrics.get("benchmarks_completed", 0))
            for metrics in detailed_metrics.values()
            if isinstance(metrics, dict) and "status" not in metrics
        )
        
        overall_success_rate = total_passed_tests / total_individual_tests if total_individual_tests > 0 else 0.0
        
        return {
            "comprehensive_test_suite": "PyCog-Zero Complete Validation",
            "roadmap_milestone": "Medium-term (Month 2-3)",
            "timestamp": time.time(),
            "total_duration": total_duration,
            "test_suites_run": len(self.test_suites),
            "successful_suites": successful_suites,
            "failed_suites": failed_suites,
            "suite_success_rate": successful_suites / len(self.test_suites),
            "total_individual_tests": total_individual_tests,
            "total_passed_tests": total_passed_tests,
            "overall_success_rate": overall_success_rate,
            "detailed_metrics": detailed_metrics,
            "suite_results": self.results,
            "roadmap_completion_status": {
                "cognitive_testing_implemented": successful_suites >= 4,
                "validation_suite_complete": "Validation" in [name for name, result in self.results.items() if result["success"]],
                "performance_benchmarking_done": "Performance Benchmarks" in [name for name, result in self.results.items() if result["success"]],
                "integration_testing_verified": "Integration" in [name for name, result in self.results.items() if result["success"]],
                "system_testing_validated": "System Tests" in [name for name, result in self.results.items() if result["success"]],
                "ready_for_production": successful_suites == len(self.test_suites) and overall_success_rate >= 0.8
            }
        }
    
    def save_comprehensive_report(self, summary_report: Dict[str, Any]):
        """Save comprehensive test report."""
        try:
            # Ensure test results directory exists
            os.makedirs(PROJECT_ROOT / "test_results", exist_ok=True)
            
            report_path = PROJECT_ROOT / "test_results" / "comprehensive_summary.json"
            with open(report_path, 'w') as f:
                json.dump(summary_report, f, indent=2, default=str)
            
            print(f"\n📄 Comprehensive report saved to: {report_path}")
            
        except Exception as e:
            print(f"❌ Failed to save comprehensive report: {e}")
    
    def display_comprehensive_summary(self, summary_report: Dict[str, Any]):
        """Display comprehensive test summary."""
        print(f"\n{'='*80}")
        print(f"🎯 COMPREHENSIVE TEST SUITE SUMMARY")
        print(f"{'='*80}")
        
        print(f"📋 Test Execution Summary:")
        print(f"   • Total Test Suites: {summary_report['test_suites_run']}")
        print(f"   • Successful Suites: {summary_report['successful_suites']}")
        print(f"   • Failed Suites: {summary_report['failed_suites']}")
        print(f"   • Suite Success Rate: {summary_report['suite_success_rate']:.1%}")
        print(f"   • Total Duration: {summary_report['total_duration']:.2f} seconds")
        
        print(f"\n📊 Detailed Test Results:")
        print(f"   • Total Individual Tests: {summary_report['total_individual_tests']}")
        print(f"   • Total Passed Tests: {summary_report['total_passed_tests']}")
        print(f"   • Overall Success Rate: {summary_report['overall_success_rate']:.1%}")
        
        print(f"\n🛠️ Test Suite Breakdown:")
        for suite_name, metrics in summary_report['detailed_metrics'].items():
            if "status" in metrics and metrics["status"] == "failed":
                print(f"   ❌ {suite_name}: FAILED ({metrics.get('error', 'Unknown error')})")
            else:
                tests = metrics.get('total_tests', metrics.get('total_benchmarks', 0))
                passed = metrics.get('passed_tests', metrics.get('benchmarks_completed', 0))
                rate = passed / tests if tests > 0 else 0.0
                print(f"   ✅ {suite_name}: {passed}/{tests} tests passed ({rate:.1%})")
        
        print(f"\n🗺️ Roadmap Completion Status:")
        roadmap_status = summary_report['roadmap_completion_status']
        for status_item, completed in roadmap_status.items():
            status_symbol = "✅" if completed else "❌"
            status_name = status_item.replace('_', ' ').title()
            print(f"   {status_symbol} {status_name}: {'COMPLETE' if completed else 'INCOMPLETE'}")
        
        # Overall assessment
        overall_success = summary_report['roadmap_completion_status']['ready_for_production']
        print(f"\n🎖️ OVERALL ASSESSMENT:")
        if overall_success:
            print(f"   🌟 COMPREHENSIVE COGNITIVE TESTING SUITE: ✅ COMPLETE")
            print(f"   🚀 System is ready for production deployment")
            print(f"   📈 All medium-term roadmap requirements validated")
        else:
            print(f"   ⚠️  COMPREHENSIVE COGNITIVE TESTING SUITE: ❌ INCOMPLETE")
            print(f"   🔧 Additional work needed before production deployment")
            print(f"   📋 Review failed tests and roadmap requirements")
        
        print(f"\n💾 Individual test reports available in: test_results/")
        print(f"   • cognitive_functions_report.json")
        print(f"   • integration_report.json")
        print(f"   • performance_report.json")
        print(f"   • validation_report.json")
        print(f"   • system_test_report.json")
        
        print(f"{'='*80}")


# Test configuration and environment setup
def setup_comprehensive_test_environment():
    """Setup environment for comprehensive testing."""
    os.environ["PYCOG_ZERO_TEST_MODE"] = "1"
    os.environ["PERFORMANCE_TESTS"] = "true"
    
    # Create test results directory
    os.makedirs(PROJECT_ROOT / "test_results", exist_ok=True)
    
    print("🔧 Comprehensive test environment configured")


# Main execution function
async def main():
    """Main function to run comprehensive test suite."""
    setup_comprehensive_test_environment()
    
    runner = ComprehensiveTestRunner()
    summary_report = await runner.run_all_comprehensive_tests()
    
    # Exit with appropriate code based on results
    if summary_report['roadmap_completion_status']['ready_for_production']:
        print("\n🎉 All comprehensive tests completed successfully!")
        return 0  # Success
    else:
        print("\n⚠️  Some comprehensive tests failed. Review the results.")
        return 1  # Failure


# Command-line interface
if __name__ == "__main__":
    exit_code = asyncio.run(main())