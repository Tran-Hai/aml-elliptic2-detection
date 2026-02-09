#!/usr/bin/env python3
"""
Phase 2: Extract Features from Background Edges
AML Detection Project - Elliptic2 Dataset

This script extracts features from background_edges.csv (78GB) using streaming:
1. Read background_edges.csv in chunks (50k rows at a time)
2. Filter edges related to nodes in nodes.csv (444k nodes)
3. Extract txId + 95 features for each node
4. Save to individual files: node_{clId}_in.csv and node_{clId}_out.csv

Memory: ~200 MB
Time: 2-3 hours
Supports: Resume from checkpoint
"""

import sys
import csv
import gzip
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Set

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logger, Timer, MemoryMonitor,
    save_pickle, load_pickle, check_file_exists
)
from config import (
    BACKGROUND_EDGES_CSV, FEATURES_DIR,
    LOG_FILE_PROCESSING, CHUNK_SIZE, CHECKPOINT_FREQUENCY_PHASE2
)


def load_node_set(index_dir: Path) -> Set[int]:
    """Load set of node IDs that we need to process"""
    node_to_idx = load_pickle(index_dir / 'node_to_idx.pkl')
    return set(node_to_idx.keys())


def get_checkpoint(features_dir: Path) -> Tuple[int, Dict]:
    """
    Get checkpoint info if exists
    
    Returns:
        (last_processed_chunk, processed_nodes_stats)
    """
    checkpoint_file = features_dir / '.checkpoint.pkl'
    
    if checkpoint_file.exists():
        checkpoint = load_pickle(checkpoint_file)
        return checkpoint['last_chunk'], checkpoint['stats']
    
    return 0, {'processed_chunks': 0, 'total_edges_extracted': 0}


def save_checkpoint(features_dir: Path, chunk_num: int, stats: Dict):
    """Save checkpoint for resume capability"""
    checkpoint_file = features_dir / '.checkpoint.pkl'
    checkpoint = {
        'last_chunk': chunk_num,
        'stats': stats
    }
    save_pickle(checkpoint, checkpoint_file)


def process_chunk(chunk: pd.DataFrame, target_nodes: Set[int], 
                  features_dir: Path, logger) -> Tuple[int, int]:
    """
    Process one chunk of background_edges.csv
    
    Args:
        chunk: DataFrame with 50k rows
        target_nodes: Set of node IDs to keep
        features_dir: Output directory
        logger: Logger instance
    
    Returns:
        (edges_written, nodes_affected)
    """
    edges_written = 0
    nodes_affected = set()
    
    # Filter rows where either clId1 or clId2 is in target_nodes
    mask = chunk['clId1'].isin(target_nodes) | chunk['clId2'].isin(target_nodes)
    filtered_chunk = chunk[mask]
    
    if len(filtered_chunk) == 0:
        return 0, 0
    
    # Get feature columns (feat#1 to feat#95)
    feature_cols = [col for col in chunk.columns if col.startswith('feat#')]
    
    # Process each row
    for _, row in filtered_chunk.iterrows():
        src_id = int(row['clId1'])
        dst_id = int(row['clId2'])
        tx_id = int(row['txId'])
        
        # Extract features
        features = [row[col] for col in feature_cols]
        
        # Write to src node's out-flow file
        if src_id in target_nodes:
            out_file = features_dir / f'node_{src_id}_out.csv'
            with open(out_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([tx_id] + features)
            edges_written += 1
            nodes_affected.add(src_id)
        
        # Write to dst node's in-flow file
        if dst_id in target_nodes:
            in_file = features_dir / f'node_{dst_id}_in.csv'
            with open(in_file, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([tx_id] + features)
            edges_written += 1
            nodes_affected.add(dst_id)
    
    return edges_written, len(nodes_affected)


def initialize_feature_files(features_dir: Path, target_nodes: Set[int], 
                             feature_cols: List[str], logger):
    """
    Initialize empty CSV files with headers for all nodes
    This is done once at the beginning
    """
    logger.info("Initializing feature files...")
    
    # Create headers
    headers = ['txId'] + feature_cols
    
    count = 0
    for node_id in target_nodes:
        # Create empty files with headers
        in_file = features_dir / f'node_{node_id}_in.csv'
        out_file = features_dir / f'node_{node_id}_out.csv'
        
        if not in_file.exists():
            with open(in_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        if not out_file.exists():
            with open(out_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(headers)
        
        count += 1
        if count % 10000 == 0:
            logger.info(f"  Initialized {count:,} / {len(target_nodes):,} nodes")
    
    logger.info(f"✓ Initialized {count:,} nodes")


def main():
    """Main function for Phase 2"""
    
    # Setup
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase2", LOG_FILE_PROCESSING)
    
    logger.info("="*60)
    logger.info("PHASE 2: EXTRACT FEATURES")
    logger.info("="*60)
    
    monitor = MemoryMonitor(logger)
    monitor.log_memory("Initial")
    
    try:
        # Check if background_edges.csv exists
        if not check_file_exists(BACKGROUND_EDGES_CSV, logger):
            logger.error(f"Background edges file not found: {BACKGROUND_EDGES_CSV}")
            return 1
        
        # Load target nodes
        with Timer("Loading target nodes", logger):
            target_nodes = load_node_set(FEATURES_DIR.parent / 'index')
            logger.info(f"Target nodes: {len(target_nodes):,}")
        
        # Get checkpoint info
        start_chunk, stats = get_checkpoint(FEATURES_DIR)
        if start_chunk > 0:
            logger.info(f"Resuming from chunk {start_chunk:,}")
            logger.info(f"Previously processed: {stats['processed_chunks']:,} chunks")
            logger.info(f"Previously extracted: {stats['total_edges_extracted']:,} edges")
        
        # Get total file size for progress calculation
        file_size = BACKGROUND_EDGES_CSV.stat().st_size
        logger.info(f"Background edges file size: {file_size / 1024**3:.2f} GB")
        
        # Read first chunk to get feature columns
        logger.info("Reading first chunk to get feature columns...")
        first_chunk = pd.read_csv(BACKGROUND_EDGES_CSV, nrows=5)
        feature_cols = [col for col in first_chunk.columns if col.startswith('feat#')]
        logger.info(f"Feature columns: {len(feature_cols)} (feat#1 to feat#{len(feature_cols)})")
        
        # Initialize feature files (only if not resuming)
        if start_chunk == 0:
            with Timer("Initializing feature files", logger):
                initialize_feature_files(FEATURES_DIR, target_nodes, feature_cols, logger)
        
        # Process background_edges.csv in chunks
        logger.info(f"Processing background edges in chunks of {CHUNK_SIZE:,} rows...")
        logger.info(f"Starting from chunk {start_chunk:,}")
        
        chunk_num = start_chunk
        total_edges_extracted = stats.get('total_edges_extracted', 0)
        
        with Timer("Processing all chunks", logger):
            for chunk in pd.read_csv(BACKGROUND_EDGES_CSV, chunksize=CHUNK_SIZE):
                chunk_num += 1
                
                # Skip chunks before checkpoint
                if chunk_num <= start_chunk:
                    continue
                
                # Process this chunk
                edges_written, nodes_affected = process_chunk(
                    chunk, target_nodes, FEATURES_DIR, logger
                )
                
                total_edges_extracted += edges_written
                
                # Log progress every 100 chunks
                if chunk_num % 100 == 0:
                    progress = chunk_num * CHUNK_SIZE * 100 / (file_size / 100)  # Approximate
                    logger.info(f"Chunk {chunk_num:,}: Extracted {edges_written:,} edges, "
                              f"Affected {nodes_affected:,} nodes. "
                              f"Total extracted: {total_edges_extracted:,}")
                    monitor.log_memory(f"After chunk {chunk_num}")
                
                # Save checkpoint every N chunks
                if chunk_num % CHECKPOINT_FREQUENCY_PHASE2 == 0:
                    stats = {
                        'processed_chunks': chunk_num,
                        'total_edges_extracted': total_edges_extracted
                    }
                    save_checkpoint(FEATURES_DIR, chunk_num, stats)
                    logger.info(f"✓ Checkpoint saved at chunk {chunk_num:,}")
                
                # Clear chunk from memory
                del chunk
        
        # Final checkpoint
        stats = {
            'processed_chunks': chunk_num,
            'total_edges_extracted': total_edges_extracted
        }
        save_checkpoint(FEATURES_DIR, chunk_num, stats)
        
        # Final stats
        logger.info("\n" + "="*60)
        logger.info("EXTRACTION COMPLETE")
        logger.info("="*60)
        logger.info(f"Total chunks processed: {chunk_num:,}")
        logger.info(f"Total edges extracted: {total_edges_extracted:,}")
        
        # Count created files
        in_files = list(FEATURES_DIR.glob('node_*_in.csv'))
        out_files = list(FEATURES_DIR.glob('node_*_out.csv'))
        logger.info(f"In-flow files created: {len(in_files):,}")
        logger.info(f"Out-flow files created: {len(out_files):,}")
        
        logger.info("\n" + "="*60)
        logger.info("✓ PHASE 2 COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Feature files saved to: {FEATURES_DIR}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in Phase 2: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
