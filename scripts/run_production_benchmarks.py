#!/usr/bin/env python3
"""
Production Readiness Benchmark Runner for PyCog-Zero.

This script orchestrates comprehensive production readiness testing including
performance benchmarks, load testing, stability analysis, and deployment validation.
"""

import asyncio
import argparse
import json
import sys
import os
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import benchmark suites
try:
    from tests.comprehensive.test_performance import run_comprehensive_performance_benchmarks
    from tests.production_readiness.test_production_benchmarks import run_production_readiness_benchmarks
except ImportError as e:
    print(f"❌ Failed to import benchmark modules: {e}")
    sys.exit(1)


class ProductionBenchmarkOrchestrator:
    """Orchestrates all production readiness testing."""
    
    def __init__(self):
        self.results = {
            "timestamp": None,
            "performance_benchmarks": None,
            "production_readiness": None,
            "combined_score": 0,
            "overall_ready": False
        }
    
    async def run_comprehensive_benchmarks(self, skip_ui_tests=False):
        """Run all comprehensive benchmark suites."""
        print("🎯 PYCOG-ZERO COMPREHENSIVE PRODUCTION BENCHMARKING")
        print("=" * 80)
        print("Running complete production readiness validation suite")
        print("=" * 80)
        
        # Set environment variables
        os.environ["PRODUCTION_READINESS_TESTS"] = "true"
        os.environ["PERFORMANCE_TESTS"] = "true"
        
        if skip_ui_tests:
            os.environ["SKIP_UI_TESTS"] = "true"
        
        # Run performance benchmarks first
        print("\n🚀 PHASE 1: Performance Benchmarks")
        print("-" * 50)
        
        try:
            perf_result = await run_comprehensive_performance_benchmarks()
            self.results["performance_benchmarks"] = {
                "completed": True,
                "success": perf_result is not None,
                "result": perf_result
            }
            print("✅ Performance benchmarks completed")
        except Exception as e:
            print(f"❌ Performance benchmarks failed: {e}")
            self.results["performance_benchmarks"] = {
                "completed": False,
                "success": False,
                "error": str(e)
            }
        
        # Run production readiness benchmarks
        print("\n🏭 PHASE 2: Production Readiness Testing")
        print("-" * 50)
        
        try:
            prod_result = await run_production_readiness_benchmarks()
            self.results["production_readiness"] = {
                "completed": True,
                "success": prod_result,
                "result": prod_result
            }
            print("✅ Production readiness testing completed")
        except Exception as e:
            print(f"❌ Production readiness testing failed: {e}")
            self.results["production_readiness"] = {
                "completed": False,
                "success": False,
                "error": str(e)
            }
        
        # Calculate combined score
        self._calculate_combined_score()
        
        # Generate final report
        await self._generate_final_report()
        
        return self.results["overall_ready"]
    
    def _calculate_combined_score(self):
        """Calculate combined production readiness score."""
        perf_score = 0
        prod_score = 0
        
        # Performance benchmark scoring
        if self.results["performance_benchmarks"]["success"]:
            perf_score = 50  # 50% weight for performance
        
        # Production readiness scoring  
        if self.results["production_readiness"]["success"]:
            prod_score = 50  # 50% weight for production readiness
        
        self.results["combined_score"] = perf_score + prod_score
        self.results["overall_ready"] = self.results["combined_score"] >= 75
    
    async def _generate_final_report(self):
        """Generate comprehensive final report."""
        report_path = PROJECT_ROOT / "test_results" / "PRODUCTION_READINESS_FINAL_REPORT.json"
        
        # Load detailed reports if available
        detailed_reports = {}
        
        # Load performance report
        perf_report_path = PROJECT_ROOT / "test_results" / "performance_report.json"
        if perf_report_path.exists():
            try:
                with open(perf_report_path, 'r') as f:
                    detailed_reports["performance"] = json.load(f)
            except:
                pass
        
        # Load production readiness report
        prod_report_path = PROJECT_ROOT / "test_results" / "production" / "production_readiness_report.json"
        if prod_report_path.exists():
            try:
                with open(prod_report_path, 'r') as f:
                    detailed_reports["production_readiness"] = json.load(f)
            except:
                pass
        
        # Generate executive summary
        executive_summary = self._generate_executive_summary(detailed_reports)
        
        final_report = {
            "pycog_zero_production_readiness_final_report": {
                "timestamp": self.results.get("timestamp"),
                "executive_summary": executive_summary,
                "benchmark_results": self.results,
                "detailed_reports": detailed_reports,
                "production_deployment_ready": self.results["overall_ready"],
                "next_steps": self._generate_next_steps()
            }
        }
        
        # Save final report
        os.makedirs(report_path.parent, exist_ok=True)
        with open(report_path, 'w') as f:
            json.dump(final_report, f, indent=2)
        
        print(f"\n📋 Final report saved to: {report_path}")
        return str(report_path)
    
    def _generate_executive_summary(self, detailed_reports):
        """Generate executive summary of all benchmarks."""
        summary = {
            "overall_score": self.results["combined_score"],
            "production_ready": self.results["overall_ready"],
            "key_findings": [],
            "critical_issues": [],
            "strengths": []
        }
        
        # Analyze performance benchmarks
        if "performance" in detailed_reports:
            perf_data = detailed_reports["performance"]
            if perf_data.get("all_benchmarks_completed"):
                summary["strengths"].append("All core performance benchmarks completed successfully")
                summary["key_findings"].append(f"System tested with {perf_data.get('total_benchmarks', 0)} performance benchmarks")
            
            # Check specific performance metrics
            for result in perf_data.get("benchmark_results", []):
                if result["benchmark_name"] == "scalability_under_load":
                    optimal_concurrency = result["metrics"].get("optimal_concurrency", 0)
                    summary["key_findings"].append(f"Optimal concurrency level: {optimal_concurrency} tasks")
                
                if result["benchmark_name"] == "reasoning_speed":
                    qps = result["metrics"].get("queries_per_second", 0)
                    summary["key_findings"].append(f"Reasoning performance: {qps:.2f} queries per second")
        
        # Analyze production readiness
        if "production_readiness" in detailed_reports:
            prod_data = detailed_reports["production_readiness"]["production_readiness_report"]
            
            prod_score = prod_data.get("production_readiness_score", 0)
            summary["key_findings"].append(f"Production readiness score: {prod_score:.1f}%")
            
            if prod_data.get("production_ready"):
                summary["strengths"].append("System meets production readiness criteria")
            else:
                summary["critical_issues"].append("System requires improvements before production deployment")
            
            # Add specific benchmark results
            for result in prod_data.get("benchmark_results", []):
                benchmark_name = result["benchmark_name"]
                metrics = result["metrics"]
                
                if benchmark_name == "multi_user_load" and not metrics.get("error"):
                    max_users = metrics.get("max_concurrent_users_80_success", 0)
                    summary["key_findings"].append(f"Maximum concurrent users (80% success): {max_users}")
                
                if benchmark_name == "resource_limits" and not metrics.get("error"):
                    max_memory = metrics.get("max_stable_memory_mb", 0)
                    peak_cpu = metrics.get("peak_cpu_utilization", 0)
                    summary["key_findings"].append(f"Resource limits: {max_memory}MB memory, {peak_cpu:.1f}% CPU")
                
                if benchmark_name == "end_to_end_integration" and not metrics.get("error"):
                    success_rate = metrics.get("integration_success_rate", 0) * 100
                    summary["key_findings"].append(f"Integration success rate: {success_rate:.1f}%")
        
        # Overall assessment
        if self.results["overall_ready"]:
            summary["key_findings"].append("🎯 SYSTEM IS READY FOR PRODUCTION DEPLOYMENT")
        else:
            summary["critical_issues"].append("⚠️ SYSTEM REQUIRES OPTIMIZATION BEFORE PRODUCTION")
        
        return summary
    
    def _generate_next_steps(self):
        """Generate next steps based on benchmark results."""
        next_steps = []
        
        if self.results["overall_ready"]:
            next_steps.extend([
                "✅ System is ready for production deployment",
                "🔧 Set up production monitoring and alerting",
                "🛡️ Configure authentication and security measures",
                "📊 Implement performance monitoring dashboards",
                "🔄 Set up automated backup and recovery procedures",
                "🚀 Plan gradual rollout and user onboarding"
            ])
        else:
            next_steps.extend([
                "⚠️ Address failed benchmarks before deployment",
                "🔧 Optimize system performance and stability",
                "🧪 Re-run benchmarks after improvements",
                "📋 Review detailed benchmark reports for specific issues"
            ])
        
        # Add specific recommendations based on failures
        if not self.results["performance_benchmarks"]["success"]:
            next_steps.append("🚀 Focus on core performance optimization")
        
        if not self.results["production_readiness"]["success"]:
            next_steps.append("🏭 Address production readiness concerns")
        
        next_steps.extend([
            "📚 Update documentation with benchmark results",
            "🤝 Share results with development and operations teams",
            "📅 Schedule regular benchmark runs for continuous validation"
        ])
        
        return next_steps


async def main():
    """Main entry point for production benchmark runner."""
    parser = argparse.ArgumentParser(description="PyCog-Zero Production Readiness Benchmark Suite")
    parser.add_argument("--skip-ui-tests", action="store_true", 
                       help="Skip UI server tests (useful for headless environments)")
    parser.add_argument("--performance-only", action="store_true",
                       help="Run only performance benchmarks")
    parser.add_argument("--production-only", action="store_true", 
                       help="Run only production readiness tests")
    parser.add_argument("--report-only", action="store_true",
                       help="Generate report from existing benchmark results")
    
    args = parser.parse_args()
    
    orchestrator = ProductionBenchmarkOrchestrator()
    
    if args.report_only:
        # Generate report from existing results
        await orchestrator._generate_final_report()
        return 0
    
    if args.performance_only:
        # Run only performance benchmarks
        os.environ["PERFORMANCE_TESTS"] = "true"
        try:
            result = await run_comprehensive_performance_benchmarks()
            return 0 if result else 1
        except Exception as e:
            print(f"❌ Performance benchmarks failed: {e}")
            return 1
    
    if args.production_only:
        # Run only production readiness tests
        os.environ["PRODUCTION_READINESS_TESTS"] = "true"
        try:
            result = await run_production_readiness_benchmarks()
            return 0 if result else 1
        except Exception as e:
            print(f"❌ Production readiness tests failed: {e}")
            return 1
    
    # Run comprehensive benchmarks
    try:
        success = await orchestrator.run_comprehensive_benchmarks(skip_ui_tests=args.skip_ui_tests)
        
        # Print final summary
        print("\n" + "=" * 80)
        print("🎯 FINAL PRODUCTION READINESS ASSESSMENT")
        print("=" * 80)
        print(f"Combined Score: {orchestrator.results['combined_score']}/100")
        print(f"Production Ready: {'✅ YES' if success else '❌ NO'}")
        
        if success:
            print("\n🚀 CONGRATULATIONS! PyCog-Zero is ready for production deployment!")
            print("📋 Review the final report for deployment guidelines and next steps.")
        else:
            print("\n⚠️ PyCog-Zero requires optimization before production deployment.")
            print("📋 Review the detailed benchmark reports to address specific issues.")
        
        return 0 if success else 1
        
    except Exception as e:
        print(f"❌ Comprehensive benchmarking failed: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)