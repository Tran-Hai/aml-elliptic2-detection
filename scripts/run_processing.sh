#!/bin/bash
# Master script to run all data processing phases
# Optimized for 4GB RAM

set -e

echo "=========================================="
echo "AML Detection - Data Processing Pipeline"
echo "=========================================="
echo ""
echo "This script will run all 4 phases:"
echo "  Phase 1: Build Index (5-10 min)"
echo "  Phase 2: Extract Features (2-3 hours)"
echo "  Phase 3: Build Sequences (30-60 min)"
echo "  Phase 4: Build Graph (5-10 min)"
echo ""
echo "Estimated total time: 3-4 hours"
echo "RAM usage: < 400 MB throughout"
echo ""

# Check if we're in the right directory
if [ ! -f "src/data_processing/config.py" ]; then
    echo "Error: Please run this script from the aml_project root directory"
    exit 1
fi

# Activate virtual environment if it exists
if [ -d "venv" ]; then
    echo "Activating virtual environment..."
    source venv/bin/activate
fi

# Create necessary directories
echo "Creating directories..."
mkdir -p logs checkpoints/phase2_checkpoints checkpoints/phase3_checkpoints

# Function to run a phase
run_phase() {
    local phase_name=$1
    local script_name=$2
    
    echo ""
    echo "=========================================="
    echo "Starting $phase_name"
    echo "=========================================="
    echo ""
    
    python "src/data_processing/$script_name"
    
    if [ $? -eq 0 ]; then
        echo ""
        echo "✓ $phase_name completed successfully!"
    else
        echo ""
        echo "✗ $phase_name failed!"
        exit 1
    fi
}

# Ask user if they want to run all phases or select specific ones
echo "Select mode:"
echo "1. Run all phases (1-4)"
echo "2. Run specific phase"
echo "3. Resume from checkpoint"
echo ""
read -p "Enter choice (1-3): " choice

case $choice in
    1)
        echo "Running all phases..."
        run_phase "Phase 1: Build Index" "phase1_build_index.py"
        run_phase "Phase 2: Extract Features" "phase2_extract_features.py"
        run_phase "Phase 3: Build Sequences" "phase3_build_sequences.py"
        run_phase "Phase 4: Build Graph" "phase4_build_graph.py"
        ;;
    
    2)
        echo ""
        echo "Select phase to run:"
        echo "1. Phase 1: Build Index"
        echo "2. Phase 2: Extract Features"
        echo "3. Phase 3: Build Sequences"
        echo "4. Phase 4: Build Graph"
        echo ""
        read -p "Enter phase (1-4): " phase
        
        case $phase in
            1) run_phase "Phase 1: Build Index" "phase1_build_index.py" ;;
            2) run_phase "Phase 2: Extract Features" "phase2_extract_features.py" ;;
            3) run_phase "Phase 3: Build Sequences" "phase3_build_sequences.py" ;;
            4) run_phase "Phase 4: Build Graph" "phase4_build_graph.py" ;;
            *) echo "Invalid phase!" ; exit 1 ;;
        esac
        ;;
    
    3)
        echo ""
        echo "Select phase to resume:"
        echo "2. Phase 2: Extract Features"
        echo "3. Phase 3: Build Sequences"
        echo ""
        read -p "Enter phase (2-3): " resume_phase
        
        case $resume_phase in
            2) 
                echo "Resuming Phase 2..."
                python src/data_processing/phase2_extract_features.py --resume
                ;;
            3)
                echo "Resuming Phase 3..."
                python src/data_processing/phase3_build_sequences.py --resume
                ;;
            *) echo "Invalid phase!" ; exit 1 ;;
        esac
        ;;
    
    *)
        echo "Invalid choice!"
        exit 1
        ;;
esac

echo ""
echo "=========================================="
echo "All done!"
echo "=========================================="
echo ""
echo "Processed data is available in: data/processed/"
echo "Logs are available in: logs/"
echo ""
echo "Next steps:"
echo "  1. Review processed data"
echo "  2. Start model training"
echo ""
