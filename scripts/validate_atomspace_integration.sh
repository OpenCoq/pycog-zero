#!/bin/bash
# AtomSpace Integration Validation Script
# ========================================
# This script validates the atomspace integration for PyCog-Zero
# It runs the cpp2py conversion pipeline validation and reports results.

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

echo "=================================="
echo "AtomSpace Integration Validation"
echo "=================================="
echo ""
echo "Project Root: $PROJECT_ROOT"
echo "Validation Date: $(date)"
echo ""

# Change to project root
cd "$PROJECT_ROOT"

# Run the validation
echo "Running atomspace validation..."
echo ""

python3 scripts/cpp2py_conversion_pipeline.py validate atomspace
# Capture exit code immediately to avoid it being overwritten
validation_result=$?

# Check exit code
if [ $validation_result -eq 0 ]; then
    echo ""
    echo "=================================="
    echo "✅ VALIDATION SUCCESSFUL"
    echo "=================================="
    echo ""
    echo "AtomSpace component is properly integrated with:"
    echo "  ✓ Core C++ headers present"
    echo "  ✓ Cython binding files available"
    echo "  ✓ Python module structure correct"
    echo "  ✓ CMake configuration valid"
    echo "  ✓ Test infrastructure present"
    echo ""
    echo "The atomspace component is ready for use in PyCog-Zero."
    echo ""
    echo "Documentation: docs/cpp2py/atomspace_validation_report.md"
    echo ""
    exit 0
else
    echo ""
    echo "=================================="
    echo "❌ VALIDATION FAILED"
    echo "=================================="
    echo ""
    echo "Please check the error messages above and ensure:"
    echo "  1. atomspace is properly cloned in components/atomspace"
    echo "  2. Python 3.8+ is available"
    echo "  3. Required dependencies are installed"
    echo ""
    echo "For troubleshooting, see: docs/cpp2py/README.md"
    echo ""
    exit 1
fi
