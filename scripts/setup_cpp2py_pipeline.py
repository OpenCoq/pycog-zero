#!/usr/bin/env python3
"""
PyCog-Zero cpp2py Conversion Pipeline Setup Script
==================================================

Quick setup script for initializing the OpenCog component conversion pipeline.
"""

import os
import sys
import json
import subprocess
from pathlib import Path

def main():
    """Main setup function."""
    print("🧠 PyCog-Zero cpp2py Conversion Pipeline Setup")
    print("==============================================")
    print()
    
    # Get current directory
    project_root = Path.cwd()
    print(f"Project root: {project_root}")
    print()
    
    # Check if we're in the right directory
    if not (project_root / "scripts" / "cpp2py_conversion_pipeline.py").exists():
        print("❌ Error: This script should be run from the PyCog-Zero project root directory")
        print("Please navigate to the project root and run: python3 scripts/setup_cpp2py_pipeline.py")
        sys.exit(1)
    
    # Run the build script
    print("🔧 Running automated build script...")
    build_script = project_root / "scripts" / "build_cpp2py_pipeline.sh"
    
    if build_script.exists():
        try:
            result = subprocess.run(["bash", str(build_script)], check=True)
            print("✅ Build script completed successfully")
        except subprocess.CalledProcessError as e:
            print(f"⚠️  Build script completed with warnings (exit code: {e.returncode})")
    else:
        print("⚠️  Build script not found, creating minimal setup...")
        
        # Create minimal directory structure
        directories = [
            "components", "components/core", "components/logic", 
            "components/cognitive", "components/advanced", "components/language",
            "tests/integration", "tests/performance", "tests/end_to_end",
            "docs/components", "docs/integration"
        ]
        
        for directory in directories:
            (project_root / directory).mkdir(parents=True, exist_ok=True)
        
        print("✅ Directory structure created")
    
    print()
    print("📋 Next Steps:")
    print("==============")
    print()
    print("1. Check pipeline status:")
    print("   python3 scripts/cpp2py_conversion_pipeline.py status")
    print()
    print("2. Clone foundation components:")
    print("   python3 scripts/cpp2py_conversion_pipeline.py clone --phase phase_0_foundation")
    print()
    print("3. Run integration tests:")
    print("   python3 -m pytest tests/integration/ -v")
    print()
    print("4. Install OpenCog dependencies (optional but recommended):")
    print("   pip install opencog-atomspace opencog-python cogutil")
    print()
    print("5. Read documentation:")
    print("   docs/cpp2py/README.md")
    print()
    
    # Show current status
    print("📊 Current Pipeline Status:")
    print("===========================")
    try:
        result = subprocess.run([
            "python3", "scripts/cpp2py_conversion_pipeline.py", "status"
        ], capture_output=True, text=True, cwd=project_root)
        
        if result.returncode == 0:
            # Extract just the summary lines
            lines = result.stdout.split('\n')
            summary_started = False
            for line in lines:
                if "Overall Status:" in line:
                    summary_started = True
                if summary_started and line.strip():
                    print(line)
        else:
            print("⚠️  Could not retrieve pipeline status")
    except Exception as e:
        print(f"⚠️  Error retrieving status: {e}")
    
    print()
    print("🎉 PyCog-Zero cpp2py Conversion Pipeline setup complete!")
    print()
    print("For help with any command:")
    print("   python3 scripts/cpp2py_conversion_pipeline.py --help")

if __name__ == "__main__":
    main()