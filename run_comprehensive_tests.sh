#!/bin/bash
# PyCog-Zero Comprehensive Testing and Validation Suite Runner
# Implements Medium-term Roadmap Requirements (Month 2-3)

echo "🧠 PyCog-Zero Comprehensive Testing and Validation Suite"
echo "======================================================="
echo "Implementing Medium-term Roadmap Requirements (Month 2-3)"
echo ""

# Set environment variables
export PYCOG_ZERO_TEST_MODE=1
export PERFORMANCE_TESTS=true

# Create test results directory
mkdir -p test_results

echo "🚀 Starting comprehensive test execution..."
echo ""

# Option 1: Run via pytest (recommended for CI/CD)
if command -v pytest >/dev/null 2>&1; then
    echo "📋 Running via pytest..."
    pytest tests/comprehensive/ -v --tb=short
    pytest_exit_code=$?
    
    if [ $pytest_exit_code -eq 0 ]; then
        echo "✅ Pytest execution completed successfully"
    else
        echo "⚠️  Pytest execution completed with issues"
    fi
    echo ""
fi

# Option 2: Run comprehensive test runner (always runs)
echo "🧪 Running comprehensive test suite..."
python3 tests/comprehensive/run_comprehensive_tests.py
runner_exit_code=$?

echo ""
echo "📊 Test Execution Complete"
echo "=========================="

if [ $runner_exit_code -eq 0 ]; then
    echo "✅ Comprehensive test suite completed successfully"
    echo "📄 Detailed reports available in test_results/ directory"
    echo ""
    echo "📋 Available Reports:"
    echo "   • comprehensive_summary.json - Overall summary"
    echo "   • cognitive_functions_report.json - Cognitive functions tests"
    echo "   • integration_report.json - Integration tests"
    echo "   • performance_report.json - Performance benchmarks"
    echo "   • validation_report.json - Validation tests"
    echo "   • system_test_report.json - System tests"
    echo ""
    echo "🎯 Medium-term roadmap validation: IN PROGRESS"
else
    echo "❌ Comprehensive test suite completed with issues"
    echo "🔍 Check individual test reports for details"
    echo ""
fi

# Display summary if available
if [ -f "test_results/comprehensive_summary.json" ]; then
    echo "📈 Quick Summary:"
    python3 -c "
import json
with open('test_results/comprehensive_summary.json', 'r') as f:
    data = json.load(f)
    print(f'   • Test Suites: {data[\"successful_suites\"]}/{data[\"test_suites_run\"]} successful')
    print(f'   • Individual Tests: {data[\"total_passed_tests\"]}/{data[\"total_individual_tests\"]} passed')
    print(f'   • Overall Success Rate: {data[\"overall_success_rate\"]:.1%}')
    print(f'   • Total Duration: {data[\"total_duration\"]:.1f} seconds')
    if data['roadmap_completion_status']['ready_for_production']:
        print('   🎉 Ready for production!')
    else:
        print('   🔧 Additional work needed')
"
fi

echo ""
echo "🔗 For detailed analysis, examine the JSON reports in test_results/"
echo "📚 See tests/comprehensive/README.md for more information"

exit $runner_exit_code