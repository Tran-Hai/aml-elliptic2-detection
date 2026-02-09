#!/bin/bash
# Setup script for AML Detection Project

set -e

echo "=========================================="
echo "AML Detection Project Setup"
echo "=========================================="
echo ""

# Check Python version
echo "Checking Python version..."
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "Python version: $python_version"

# Check available disk space
echo ""
echo "Checking disk space..."
available_space=$(df -h . | awk 'NR==2 {print $4}')
echo "Available disk space: $available_space"

# Check RAM
echo ""
echo "Checking available RAM..."
available_ram=$(free -h | awk 'NR==2 {print $7}')
echo "Available RAM: $available_ram"

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
echo "Upgrading pip..."
pip install --upgrade pip

# Install requirements
echo ""
echo "Installing requirements..."
pip install -r requirements.txt

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "To activate the environment:"
echo "  source venv/bin/activate"
echo ""
echo "To run data processing:"
echo "  python src/data_processing/phase1_build_index.py"
echo ""
