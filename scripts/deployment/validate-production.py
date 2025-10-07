#!/usr/bin/env python3

"""
PyCog-Zero Production Validation Tool

This script validates the production deployment of PyCog-Zero cognitive Agent-Zero systems.
It performs comprehensive checks to ensure the system is ready for production use.
"""

import os
import sys
import json
import time
import requests
import subprocess
from typing import Dict, Any, List, Tuple
from datetime import datetime

# Add project root to path
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
sys.path.insert(0, os.path.join(project_root, 'python'))

try:
    from python.helpers.print_style import PrintStyle
except ImportError:
    # Fallback if not available
    class PrintStyle:
        @staticmethod
        def success(msg): print(f"✓ {msg}")
        @staticmethod
        def error(msg): print(f"✗ {msg}")
        @staticmethod
        def warning(msg): print(f"⚠ {msg}")
        @staticmethod
        def standard(msg): print(f"• {msg}")


class ProductionValidator:
    """Validates PyCog-Zero production deployment."""
    
    def __init__(self, base_url: str = "http://localhost:80"):
        self.base_url = base_url
        self.validation_results = []
        self.start_time = datetime.now()
    
    def validate_all(self) -> bool:
        """Run all validation checks."""
        PrintStyle.standard("Starting PyCog-Zero production validation...")
        PrintStyle.standard(f"Target URL: {self.base_url}")
        PrintStyle.standard(f"Validation started at: {self.start_time}")
        print()
        
        validation_checks = [
            ("Docker Environment", self.validate_docker_environment),
            ("Service Availability", self.validate_service_availability),
            ("Health Checks", self.validate_health_checks),
            ("API Endpoints", self.validate_api_endpoints),
            ("Cognitive Features", self.validate_cognitive_features),  
            ("Security Configuration", self.validate_security_configuration),
            ("Performance Metrics", self.validate_performance_metrics),
            ("Monitoring Systems", self.validate_monitoring_systems),
            ("Backup Systems", self.validate_backup_systems),
            ("Resource Limits", self.validate_resource_limits)
        ]
        
        all_passed = True
        
        for check_name, check_function in validation_checks:
            PrintStyle.standard(f"Running {check_name} validation...")
            try:
                result = check_function()
                if result["passed"]:
                    PrintStyle.success(f"{check_name}: {result['message']}")
                else:
                    PrintStyle.error(f"{check_name}: {result['message']}")
                    all_passed = False
                
                self.validation_results.append({
                    "check": check_name,
                    "passed": result["passed"],
                    "message": result["message"],
                    "details": result.get("details", {}),
                    "timestamp": datetime.now().isoformat()
                })
                
            except Exception as e:
                PrintStyle.error(f"{check_name}: Validation failed with error: {str(e)}")
                all_passed = False
                self.validation_results.append({
                    "check": check_name,
                    "passed": False,
                    "message": f"Validation error: {str(e)}",
                    "timestamp": datetime.now().isoformat()
                })
        
        print()
        self.print_summary()
        
        return all_passed
    
    def validate_docker_environment(self) -> Dict[str, Any]:
        """Validate Docker environment and containers."""
        try:
            # Check if Docker is running
            result = subprocess.run(["docker", "ps"], capture_output=True, text=True)
            if result.returncode != 0:
                return {"passed": False, "message": "Docker is not running or not accessible"}
            
            # Check for PyCog-Zero containers
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=pycog-zero", "--format", "table {{.Names}}\t{{.Status}}"],
                capture_output=True, text=True
            )
            
            containers = result.stdout.strip().split('\n')[1:] if len(result.stdout.strip().split('\n')) > 1 else []
            
            if not containers:
                return {"passed": False, "message": "No PyCog-Zero containers found"}
            
            running_containers = [c for c in containers if "Up" in c]
            
            return {
                "passed": len(running_containers) > 0,
                "message": f"Found {len(running_containers)} running PyCog-Zero containers",
                "details": {"containers": containers}
            }
            
        except Exception as e:
            return {"passed": False, "message": f"Docker validation failed: {str(e)}"}
    
    def validate_service_availability(self) -> Dict[str, Any]:
        """Validate that core services are available."""
        services_to_check = [
            (f"{self.base_url}/", "Main web interface"),
            (f"{self.base_url}/health", "Health check endpoint"),
        ]
        
        results = []
        all_available = True
        
        for url, description in services_to_check:
            try:
                response = requests.get(url, timeout=10)
                if response.status_code == 200:
                    results.append(f"✓ {description}")
                else:
                    results.append(f"✗ {description} (Status: {response.status_code})")
                    all_available = False
            except Exception as e:
                results.append(f"✗ {description} (Error: {str(e)})")
                all_available = False
        
        return {
            "passed": all_available,
            "message": f"Service availability: {len([r for r in results if r.startswith('✓')])}/{len(results)} services available",
            "details": {"results": results}
        }
    
    def validate_health_checks(self) -> Dict[str, Any]:
        """Validate health check endpoints."""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=10)
            
            if response.status_code != 200:
                return {"passed": False, "message": f"Health check returned status {response.status_code}"}
            
            health_data = response.json()
            overall_status = health_data.get("overall_status", "unknown")
            
            if overall_status == "healthy":
                return {
                    "passed": True,
                    "message": "All health checks passed",
                    "details": health_data
                }
            else:
                return {
                    "passed": False,
                    "message": f"Health check status: {overall_status}",
                    "details": health_data
                }
                
        except Exception as e:
            return {"passed": False, "message": f"Health check validation failed: {str(e)}"}
    
    def validate_api_endpoints(self) -> Dict[str, Any]:
        """Validate API endpoints."""
        api_endpoints = [
            ("/health", "Health check"),
            ("/metrics", "Metrics endpoint"),
            ("/cognitive/metrics", "Cognitive metrics")
        ]
        
        working_endpoints = []
        failed_endpoints = []
        
        for endpoint, description in api_endpoints:
            try:
                response = requests.get(f"{self.base_url}{endpoint}", timeout=10)
                if response.status_code in [200, 503]:  # 503 acceptable for unavailable features
                    working_endpoints.append(f"{endpoint} ({description})")
                else:
                    failed_endpoints.append(f"{endpoint} (Status: {response.status_code})")
            except Exception as e:
                failed_endpoints.append(f"{endpoint} (Error: {str(e)})")
        
        return {
            "passed": len(failed_endpoints) == 0,
            "message": f"API validation: {len(working_endpoints)}/{len(api_endpoints)} endpoints working",
            "details": {
                "working": working_endpoints,
                "failed": failed_endpoints
            }
        }
    
    def validate_cognitive_features(self) -> Dict[str, Any]:
        """Validate cognitive system features."""
        try:
            response = requests.get(f"{self.base_url}/cognitive/metrics", timeout=15)
            
            if response.status_code == 200:
                cognitive_data = response.json()
                if cognitive_data and not cognitive_data.get("error"):
                    return {
                        "passed": True,
                        "message": "Cognitive features are operational",
                        "details": cognitive_data
                    }
                else:
                    return {
                        "passed": False,
                        "message": "Cognitive features not fully operational",
                        "details": cognitive_data
                    }
            elif response.status_code == 503:
                return {
                    "passed": False,
                    "message": "Cognitive monitoring not available (expected in basic deployment)",
                    "details": {"note": "This may be acceptable depending on deployment configuration"}
                }
            else:
                return {
                    "passed": False,
                    "message": f"Cognitive metrics endpoint returned status {response.status_code}"
                }
                
        except Exception as e:
            return {
                "passed": False,
                "message": f"Cognitive validation failed: {str(e)}",
                "details": {"note": "This may be acceptable if cognitive features are not enabled"}
            }
    
    def validate_security_configuration(self) -> Dict[str, Any]:
        """Validate security configuration."""
        security_checks = []
        
        # Check for security headers
        try:
            response = requests.get(f"{self.base_url}/", timeout=10)
            headers = response.headers
            
            security_headers = [
                ("X-Frame-Options", "Clickjacking protection"),
                ("X-Content-Type-Options", "MIME type sniffing protection"),
                ("X-XSS-Protection", "XSS protection")
            ]
            
            for header, description in security_headers:
                if header in headers:
                    security_checks.append(f"✓ {description}")
                else:
                    security_checks.append(f"⚠ {description} header missing")
            
        except Exception as e:
            security_checks.append(f"✗ Could not check security headers: {str(e)}")
        
        # Basic security validation passed if no critical failures
        passed = not any(check.startswith("✗") for check in security_checks)
        
        return {
            "passed": passed,
            "message": f"Security validation: {len([c for c in security_checks if c.startswith('✓')])}/{len(security_headers)} security headers present",
            "details": {"checks": security_checks}
        }
    
    def validate_performance_metrics(self) -> Dict[str, Any]:
        """Validate performance metrics collection."""
        try:
            start_time = time.time()
            response = requests.get(f"{self.base_url}/metrics", timeout=10)
            response_time = (time.time() - start_time) * 1000
            
            if response.status_code == 200:
                metrics_data = response.json()
                
                performance_ok = (
                    response_time < 5000 and  # Response time under 5 seconds
                    not metrics_data.get("error")
                )
                
                return {
                    "passed": performance_ok,
                    "message": f"Performance validation: Response time {response_time:.0f}ms",
                    "details": {
                        "response_time_ms": response_time,
                        "metrics_available": not metrics_data.get("error"),
                        "metrics_data": metrics_data
                    }
                }
            else:
                return {
                    "passed": False,
                    "message": f"Metrics endpoint not available (Status: {response.status_code})"
                }
                
        except Exception as e:
            return {"passed": False, "message": f"Performance validation failed: {str(e)}"}
    
    def validate_monitoring_systems(self) -> Dict[str, Any]:
        """Validate monitoring systems."""
        try:
            # Check if Prometheus is available (might be on different port)
            monitoring_endpoints = [
                ("http://localhost:9090/", "Prometheus monitoring"),
                (f"{self.base_url}/metrics", "Application metrics")
            ]
            
            available_monitoring = []
            unavailable_monitoring = []
            
            for url, description in monitoring_endpoints:
                try:
                    response = requests.get(url, timeout=5)
                    if response.status_code == 200:
                        available_monitoring.append(description)
                    else:
                        unavailable_monitoring.append(f"{description} (Status: {response.status_code})")
                except:
                    unavailable_monitoring.append(f"{description} (Connection failed)")
            
            return {
                "passed": len(available_monitoring) > 0,
                "message": f"Monitoring: {len(available_monitoring)}/{len(monitoring_endpoints)} systems available",
                "details": {
                    "available": available_monitoring,
                    "unavailable": unavailable_monitoring
                }
            }
            
        except Exception as e:
            return {"passed": False, "message": f"Monitoring validation failed: {str(e)}"}
    
    def validate_backup_systems(self) -> Dict[str, Any]:
        """Validate backup systems."""
        try:
            # Check if backup container is running
            result = subprocess.run(
                ["docker", "ps", "--filter", "name=pycog-zero-backup", "--format", "{{.Names}}"],
                capture_output=True, text=True
            )
            
            backup_containers = result.stdout.strip().split('\n') if result.stdout.strip() else []
            backup_running = len([c for c in backup_containers if c]) > 0
            
            return {
                "passed": backup_running,
                "message": f"Backup system: {'Running' if backup_running else 'Not found'}",
                "details": {"backup_containers": backup_containers}
            }
            
        except Exception as e:
            return {"passed": False, "message": f"Backup validation failed: {str(e)}"}
    
    def validate_resource_limits(self) -> Dict[str, Any]:
        """Validate resource limits and constraints."""
        try:
            # Check Docker container resource usage
            result = subprocess.run([
                "docker", "stats", "--no-stream", "--format", 
                "table {{.Name}}\t{{.CPUPerc}}\t{{.MemUsage}}\t{{.MemPerc}}"
            ], capture_output=True, text=True)
            
            if result.returncode == 0 and result.stdout:
                stats_lines = result.stdout.strip().split('\n')[1:]  # Skip header
                pycog_stats = [line for line in stats_lines if 'pycog-zero' in line]
                
                if pycog_stats:
                    return {
                        "passed": True,
                        "message": f"Resource monitoring: {len(pycog_stats)} PyCog-Zero containers monitored",
                        "details": {"container_stats": pycog_stats}
                    }
                else:
                    return {
                        "passed": False,
                        "message": "No PyCog-Zero containers found in resource monitoring"
                    }
            else:
                return {
                    "passed": False,
                    "message": "Could not retrieve Docker resource stats"
                }
                
        except Exception as e:
            return {"passed": False, "message": f"Resource validation failed: {str(e)}"}
    
    def print_summary(self):
        """Print validation summary."""
        total_checks = len(self.validation_results)
        passed_checks = len([r for r in self.validation_results if r["passed"]])
        failed_checks = total_checks - passed_checks
        
        print("=" * 60)
        PrintStyle.standard("PRODUCTION VALIDATION SUMMARY")
        print("=" * 60)
        PrintStyle.standard(f"Total checks: {total_checks}")
        PrintStyle.success(f"Passed: {passed_checks}")
        if failed_checks > 0:
            PrintStyle.error(f"Failed: {failed_checks}")
        else:
            PrintStyle.success("Failed: 0")
        
        success_rate = (passed_checks / total_checks) * 100 if total_checks > 0 else 0
        PrintStyle.standard(f"Success rate: {success_rate:.1f}%")
        
        validation_time = datetime.now() - self.start_time
        PrintStyle.standard(f"Validation time: {validation_time}")
        
        if failed_checks == 0:
            print()
            PrintStyle.success("🎉 PyCog-Zero production deployment validation PASSED!")
            PrintStyle.success("System is ready for production use.")
        else:
            print()
            PrintStyle.error("❌ PyCog-Zero production deployment validation FAILED!")
            PrintStyle.error("Please review failed checks before deploying to production.")
            
            print("\nFailed checks:")
            for result in self.validation_results:
                if not result["passed"]:
                    PrintStyle.error(f"  • {result['check']}: {result['message']}")
    
    def save_report(self, filename: str = None):
        """Save validation report to file."""
        if filename is None:
            filename = f"pycog-zero-validation-{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        report = {
            "validation_timestamp": self.start_time.isoformat(),
            "target_url": self.base_url,
            "total_checks": len(self.validation_results),
            "passed_checks": len([r for r in self.validation_results if r["passed"]]),
            "failed_checks": len([r for r in self.validation_results if not r["passed"]]),
            "success_rate": (len([r for r in self.validation_results if r["passed"]]) / len(self.validation_results)) * 100,
            "validation_results": self.validation_results
        }
        
        with open(filename, 'w') as f:
            json.dump(report, f, indent=2)
        
        PrintStyle.success(f"Validation report saved to: {filename}")


def main():
    """Main validation function.""" 
    import argparse
    
    parser = argparse.ArgumentParser(description="PyCog-Zero Production Validation Tool")
    parser.add_argument("--url", default="http://localhost:80", help="Base URL to validate (default: http://localhost:80)")
    parser.add_argument("--report", help="Save validation report to file")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    validator = ProductionValidator(args.url)
    
    try:
        validation_passed = validator.validate_all()
        
        if args.report:
            validator.save_report(args.report)
        
        sys.exit(0 if validation_passed else 1)
        
    except KeyboardInterrupt:
        PrintStyle.error("\nValidation interrupted by user")
        sys.exit(130)
    except Exception as e:
        PrintStyle.error(f"Validation failed with error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()