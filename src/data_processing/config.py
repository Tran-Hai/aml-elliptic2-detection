"""
Configuration file for AML Detection Project
Data processing parameters optimized for 4GB RAM
"""

import os
from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================

BASE_DIR = Path(__file__).parent.parent.parent

# Raw data paths
RAW_DATA_DIR = BASE_DIR / "data" / "raw"
NODES_CSV = RAW_DATA_DIR / "nodes.csv"
EDGES_CSV = RAW_DATA_DIR / "edges.csv"
CONNECTED_COMPONENTS_CSV = RAW_DATA_DIR / "connected_components.csv"
BACKGROUND_NODES_CSV = RAW_DATA_DIR / "background_nodes.csv"
BACKGROUND_EDGES_CSV = RAW_DATA_DIR / "background_edges.csv"

# Processed data paths
PROCESSED_DIR = BASE_DIR / "data" / "processed"
INDEX_DIR = PROCESSED_DIR / "index"
FEATURES_DIR = PROCESSED_DIR / "features"
SEQUENCES_DIR = PROCESSED_DIR / "sequences"
GRAPH_DIR = PROCESSED_DIR / "graph"

# Output paths
LOGS_DIR = BASE_DIR / "logs"
CHECKPOINTS_DIR = BASE_DIR / "checkpoints"

# ============================================================================
# DATA PROCESSING PARAMETERS
# ============================================================================

# Phase 2: Feature Extraction
CHUNK_SIZE = 50000  # Số rows đọc mỗi lần từ background_edges.csv
                    # 50k rows ~ 50-100 MB RAM
FEATURE_EXTRACTION_BATCH_SIZE = 1000  # Số nodes xử lý trước khi log

# Phase 3: Sequence Building
SEQUENCE_LENGTH = 50  # Số transactions tối đa mỗi sequence
SEQUENCE_BUILDING_BATCH_SIZE = 10000  # Số nodes xử lý trước khi checkpoint

# Feature dimensions
NUM_EDGE_FEATURES = 95  # Từ background_edges.csv
TXID_FEATURE = 1  # Transaction ID
timestamp_FEATURE = 1  # Normalized txId as timestamp proxy
TOTAL_FEATURE_DIM = NUM_EDGE_FEATURES + TXID_FEATURE + timestamp_FEATURE  # 97

# ============================================================================
# DATASET STATISTICS (estimated)
# ============================================================================

NUM_NODES = 444521
NUM_EDGES = 367138
NUM_CONNECTED_COMPONENTS = 121811
BACKGROUND_NODES_COUNT = 49299866
BACKGROUND_EDGES_COUNT = 196215607

# Class distribution
CLASS_LICIT_RATIO = 0.977
CLASS_SUSPICIOUS_RATIO = 0.023

# ============================================================================
# TRAIN/VAL/TEST SPLIT
# ============================================================================

TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Stratified split để giữ tỷ lệ class imbalance
STRATIFIED_SPLIT = True

# ============================================================================
# MEMORY OPTIMIZATION
# ============================================================================

# Garbage collection frequency
GC_FREQUENCY = 100  # Collect garbage mỗi N batches

# Checkpointing
CHECKPOINT_FREQUENCY_PHASE2 = 1000  # Mỗi 1000 chunks
CHECKPOINT_FREQUENCY_PHASE3 = 10000  # Mỗi 10000 nodes

# ============================================================================
# LOGGING
# ============================================================================

LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
LOG_FILE_PROCESSING = LOGS_DIR / "processing.log"

# ============================================================================
# TEMPORAL SETTINGS
# ============================================================================

# txId được sử dụng như proxy cho timestamp
# Giả định: txId càng lớn = transaction càng mới
USE_TXID_AS_TIMESTAMP = True
NORMALIZE_TIMESTAMP = True  # Normalize txId về [0, 1]

# ============================================================================
# FEATURE SELECTION
# ============================================================================

# Các cột trong background_edges.csv
EDGE_FEATURE_COLUMNS = [f"feat#{i}" for i in range(1, 96)]  # feat#1 đến feat#95
REQUIRED_COLUMNS = ['clId1', 'clId2', 'txId'] + EDGE_FEATURE_COLUMNS

# ============================================================================
# VALIDATION
# ============================================================================

def validate_paths():
    """Kiểm tra các paths tồn tại"""
    required_files = [
        NODES_CSV,
        EDGES_CSV,
        CONNECTED_COMPONENTS_CSV,
        BACKGROUND_EDGES_CSV
    ]
    
    missing = []
    for file_path in required_files:
        if not file_path.exists():
            missing.append(str(file_path))
    
    if missing:
        raise FileNotFoundError(f"Missing required files:\n" + "\n".join(missing))
    
    print("✓ All required paths validated successfully!")
    return True

# ============================================================================
# CREATE DIRECTORIES
# ============================================================================

def create_directories():
    """Tạo tất cả directories cần thiết"""
    dirs_to_create = [
        RAW_DATA_DIR,
        INDEX_DIR,
        FEATURES_DIR,
        SEQUENCES_DIR,
        GRAPH_DIR,
        LOGS_DIR,
        CHECKPOINTS_DIR
    ]
    
    for dir_path in dirs_to_create:
        dir_path.mkdir(parents=True, exist_ok=True)
    
    print(f"✓ Created {len(dirs_to_create)} directories")

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    create_directories()
    validate_paths()
    print("\nConfiguration loaded successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"Sequence length: {SEQUENCE_LENGTH}")
    print(f"Total feature dimension: {TOTAL_FEATURE_DIM}")
