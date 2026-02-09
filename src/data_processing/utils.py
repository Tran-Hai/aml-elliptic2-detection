"""
Utility functions for data processing pipeline
AML Detection Project - Elliptic2 Dataset
"""

import os
import sys
import time
import pickle
import logging
import psutil
import gc
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict

import pandas as pd
import numpy as np

# Setup logging
def setup_logger(name: str, log_file: Path, level=logging.INFO) -> logging.Logger:
    """Setup logger with both file and console handlers"""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Avoid duplicate handlers
    if not logger.handlers:
        # File handler
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        
        # Console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        logger.addHandler(fh)
        logger.addHandler(ch)
    
    return logger


class Timer:
    """Context manager for timing code blocks"""
    def __init__(self, name: str = "Operation", logger: Optional[logging.Logger] = None):
        self.name = name
        self.logger = logger
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = time.time()
        if self.logger:
            self.logger.info(f"Starting {self.name}...")
        else:
            print(f"⏱️  Starting {self.name}...")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = time.time()
        elapsed = self.end_time - self.start_time
        
        if self.logger:
            self.logger.info(f"✓ {self.name} completed in {elapsed:.2f} seconds")
        else:
            print(f"✓ {self.name} completed in {elapsed:.2f} seconds")
    
    def elapsed(self) -> float:
        """Get elapsed time"""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        elif self.start_time:
            return time.time() - self.start_time
        return 0.0


class MemoryMonitor:
    """Monitor RAM usage"""
    def __init__(self, logger: Optional[logging.Logger] = None):
        self.logger = logger
        self.process = psutil.Process(os.getpid())
    
    def get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return self.process.memory_info().rss / 1024 / 1024
    
    def log_memory(self, label: str = "Current"):
        """Log current memory usage"""
        mem_mb = self.get_memory_usage()
        msg = f"{label} memory usage: {mem_mb:.2f} MB"
        
        if self.logger:
            self.logger.info(msg)
        else:
            print(f"💾 {msg}")
    
    @staticmethod
    def collect_garbage():
        """Force garbage collection"""
        gc.collect()
        print("🧹 Garbage collection completed")


def save_pickle(data: Any, filepath: Path, logger: Optional[logging.Logger] = None):
    """Save data to pickle file"""
    filepath.parent.mkdir(parents=True, exist_ok=True)
    
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size = filepath.stat().st_size / 1024  # KB
    msg = f"Saved {filepath.name} ({file_size:.2f} KB)"
    
    if logger:
        logger.info(msg)
    else:
        print(f"💾 {msg}")


def load_pickle(filepath: Path, logger: Optional[logging.Logger] = None) -> Any:
    """Load data from pickle file"""
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")
    
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    
    file_size = filepath.stat().st_size / 1024  # KB
    msg = f"Loaded {filepath.name} ({file_size:.2f} KB)"
    
    if logger:
        logger.info(msg)
    else:
        print(f"📂 {msg}")
    
    return data


def check_file_exists(filepath: Path, logger: Optional[logging.Logger] = None) -> bool:
    """Check if file exists and log result"""
    exists = filepath.exists()
    
    if logger:
        if exists:
            logger.info(f"✓ File exists: {filepath}")
        else:
            logger.warning(f"✗ File not found: {filepath}")
    
    return exists


def print_dataframe_info(df: pd.DataFrame, name: str = "DataFrame", 
                         logger: Optional[logging.Logger] = None):
    """Print information about a DataFrame"""
    info_lines = [
        f"\n{'='*50}",
        f"{name} Information",
        f"{'='*50}",
        f"Shape: {df.shape}",
        f"Columns: {list(df.columns)}",
        f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
        f"Missing values: {df.isnull().sum().sum()}",
        f"{'='*50}\n"
    ]
    
    info_str = "\n".join(info_lines)
    
    if logger:
        logger.info(info_str)
    else:
        print(info_str)


def create_adjacency_list(edges_df: pd.DataFrame, 
                          logger: Optional[logging.Logger] = None) -> Dict[int, List[Tuple[int, int]]]:
    """
    Create adjacency list from edges DataFrame
    
    Args:
        edges_df: DataFrame with columns ['clId1', 'clId2', 'txId']
        logger: Optional logger
    
    Returns:
        Dict mapping node_id -> [(neighbor_id, txId), ...]
    """
    adjacency_list = defaultdict(list)
    
    total_edges = len(edges_df)
    
    for idx, row in edges_df.iterrows():
        src, dst, tx_id = row['clId1'], row['clId2'], row['txId']
        
        # Add both directions (undirected for adjacency, but keep txId)
        adjacency_list[src].append((dst, tx_id))
        adjacency_list[dst].append((src, tx_id))
        
        # Log progress every 100k edges
        if (idx + 1) % 100000 == 0:
            progress = (idx + 1) / total_edges * 100
            msg = f"Processing edges: {idx + 1:,} / {total_edges:,} ({progress:.1f}%)"
            
            if logger:
                logger.info(msg)
            else:
                print(f"  {msg}")
    
    # Convert defaultdict to regular dict
    result = dict(adjacency_list)
    
    msg = f"Created adjacency list with {len(result)} nodes"
    if logger:
        logger.info(msg)
    else:
        print(f"✓ {msg}")
    
    return result


def validate_processed_data(processed_dir: Path, 
                           expected_files: List[str],
                           logger: Optional[logging.Logger] = None) -> bool:
    """
    Validate that all expected processed files exist
    
    Args:
        processed_dir: Directory containing processed files
        expected_files: List of expected filenames
        logger: Optional logger
    
    Returns:
        True if all files exist, False otherwise
    """
    all_exist = True
    
    for filename in expected_files:
        filepath = processed_dir / filename
        exists = filepath.exists()
        
        status = "✓" if exists else "✗"
        msg = f"{status} {filename}"
        
        if logger:
            if exists:
                logger.info(msg)
            else:
                logger.error(msg)
                all_exist = False
        else:
            print(msg)
            if not exists:
                all_exist = False
    
    return all_exist


# Constants
SEED = 42
EPSILON = 1e-8

# Label mapping
LABEL_MAP = {
    'licit': 0,
    'suspicious': 1,
    'ilicit': 1,  # Alternative spelling
    0: 0,
    1: 1
}

if __name__ == "__main__":
    # Test utilities
    print("Testing utility functions...")
    
    # Test logger
    logger = setup_logger("test", Path("logs/test.log"))
    logger.info("Test log message")
    
    # Test timer
    with Timer("Test operation"):
        time.sleep(0.1)
    
    # Test memory monitor
    monitor = MemoryMonitor()
    monitor.log_memory("Test")
    
    print("✓ All utility functions working!")
