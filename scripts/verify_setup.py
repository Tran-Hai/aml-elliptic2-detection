#!/usr/bin/env python3
"""
Setup verification script for AML Detection Project
Checks if all required packages and files are available
"""

import sys
from pathlib import Path

# Check if running from correct directory
current_dir = Path.cwd()
expected_file = current_dir / "src" / "data_processing" / "config.py"

if not expected_file.exists():
    print("❌ ERROR: Please run this script from the aml_project root directory!")
    print(f"   Current directory: {current_dir}")
    print(f"   Expected file not found: {expected_file}")
    sys.exit(1)

print("=" * 60)
print("AML Detection Project - Setup Verification")
print("=" * 60)

# Check required packages
print("\n📦 Checking required packages...")
required_packages = ['pandas', 'numpy', 'psutil', 'sklearn']
missing_packages = []

for package in required_packages:
    try:
        __import__(package)
        print(f"  ✓ {package}")
    except ImportError:
        print(f"  ✗ {package} - NOT INSTALLED")
        missing_packages.append(package)

if missing_packages:
    print(f"\n⚠️  Missing packages: {', '.join(missing_packages)}")
    print("\n🔧 Installation options:")
    print("   Option 1: Create virtual environment and install:")
    print("      ./scripts/setup_env.sh")
    print("\n   Option 2: Install system-wide (requires sudo):")
    print(f"      sudo apt-get install python3-pip")
    print(f"      pip3 install {' '.join(missing_packages)}")
    print("\n   Option 3: Use conda:")
    print(f"      conda install {' '.join(missing_packages)}")
    sys.exit(1)

print("\n✅ All required packages are installed!")

# Check data files
print("\n📁 Checking data files...")
data_files = [
    'data/raw/nodes.csv',
    'data/raw/edges.csv',
    'data/raw/connected_components.csv'
]

all_data_exist = True
for file_path in data_files:
    full_path = current_dir / file_path
    if full_path.exists():
        size_mb = full_path.stat().st_size / 1024 / 1024
        print(f"  ✓ {file_path} ({size_mb:.2f} MB)")
    else:
        print(f"  ✗ {file_path} - NOT FOUND")
        all_data_exist = False

if not all_data_exist:
    print("\n❌ Some data files are missing!")
    print("   Please ensure all CSV files are in data/raw/")
    sys.exit(1)

print("\n✅ All data files are present!")

# Check created files
print("\n📝 Checking Phase 1 files...")
phase1_files = [
    'src/data_processing/config.py',
    'src/data_processing/utils.py',
    'src/data_processing/phase1_build_index.py'
]

all_files_exist = True
for file_path in phase1_files:
    full_path = current_dir / file_path
    if full_path.exists():
        print(f"  ✓ {file_path}")
    else:
        print(f"  ✗ {file_path} - NOT FOUND")
        all_files_exist = False

if not all_files_exist:
    print("\n❌ Some script files are missing!")
    sys.exit(1)

print("\n✅ All Phase 1 files are created!")

# Summary
print("\n" + "=" * 60)
print("SETUP VERIFICATION COMPLETE")
print("=" * 60)
print("\n🚀 Ready to run Phase 1!")
print("\nCommand:")
print("   python3 src/data_processing/phase1_build_index.py")
print("\nOr use the run script:")
print("   ./scripts/run_processing.sh")
print("\nExpected output:")
print("   - data/processed/index/node_to_idx.pkl")
print("   - data/processed/index/idx_to_node.pkl")
print("   - data/processed/index/node_labels.pkl")
print("   - data/processed/index/component_to_nodes.pkl")
print("   - data/processed/index/edges_index.pkl")
print("\n⏱️  Estimated time: 5-10 minutes")
print("💾 Estimated RAM: < 100 MB")
