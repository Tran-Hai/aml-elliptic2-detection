#!/usr/bin/env python3
"""
Phase 3: Build Temporal Sequences
AML Detection Project - Elliptic2 Dataset

This script builds temporal sequences for LAS-Mamba-GNN:
1. Read feature files (node_{id}_in/out.csv) from Phase 2
2. Sort by txId (temporal ordering)
3. Keep K=50 most recent transactions
4. Pre-pad with zeros if needed
5. Create timestamp_proxy (txId / max_txId)
6. Save as numpy arrays (.npz)

Memory: ~100-200 MB
Time: 30-60 minutes
Supports: Resume from checkpoint
"""

import sys
import gc
import pickle
from pathlib import Path
from typing import Dict, Tuple, List
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np

sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logger, Timer, MemoryMonitor,
    save_pickle, load_pickle
)
from config import (
    FEATURES_DIR, INDEX_DIR, SEQUENCES_DIR,
    LOG_FILE_PROCESSING, SEQUENCE_LENGTH,
    SEQUENCE_BUILDING_BATCH_SIZE
)


def get_max_txid(features_dir: Path, logger) -> int:
    """
    Find maximum txId across all feature files
    This is needed for normalization
    """
    logger.info("Finding maximum txId for normalization...")
    
    # Sample some files to estimate max
    sample_files = list(features_dir.glob('node_*_in.csv'))[:100]
    max_txids = []
    
    for file in sample_files:
        try:
            df = pd.read_csv(file, usecols=['txId'])
            if len(df) > 0:
                max_txids.append(df['txId'].max())
        except:
            continue
    
    if max_txids:
        estimated_max = max(max_txids)
        logger.info(f"Estimated max txId from sample: {estimated_max:,}")
        # Add some buffer for safety
        return int(estimated_max * 1.1)
    else:
        # Fallback value
        return 800_000_000


def get_checkpoint(sequences_dir: Path) -> Tuple[int, List[int]]:
    """
    Get checkpoint info if exists
    
    Returns:
        (last_processed_idx, processed_node_indices)
    """
    checkpoint_file = sequences_dir / '.checkpoint_phase3.pkl'
    
    if checkpoint_file.exists():
        checkpoint = load_pickle(checkpoint_file)
        return checkpoint['last_idx'], checkpoint['processed_nodes']
    
    return 0, []


def save_checkpoint(sequences_dir: Path, idx: int, processed_nodes: List[int]):
    """Save checkpoint for resume capability"""
    checkpoint_file = sequences_dir / '.checkpoint_phase3.pkl'
    checkpoint = {
        'last_idx': idx,
        'processed_nodes': processed_nodes
    }
    save_pickle(checkpoint, checkpoint_file)


def process_node_sequence(node_id: int, node_idx: int, features_dir: Path, 
                         max_txid: int, logger) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """
    Process sequences for a single node
    
    Args:
        node_id: Original clId
        node_idx: Index (0, 1, 2...)
        features_dir: Directory with CSV files
        max_txid: Maximum txId for normalization
        logger: Logger instance
    
    Returns:
        (in_sequence, out_sequence, n_in_original, n_out_original)
    """
    K = SEQUENCE_LENGTH  # 50
    n_features = 95  # feat#1 to feat#95
    
    # File paths
    in_file = features_dir / f'node_{node_id}_in.csv'
    out_file = features_dir / f'node_{node_id}_out.csv'
    
    # Initialize with zeros (pre-padding)
    in_sequence = np.zeros((K, n_features + 1))  # +1 for timestamp_proxy
    out_sequence = np.zeros((K, n_features + 1))
    
    n_in_original = 0
    n_out_original = 0
    
    # Process in-flow
    if in_file.exists():
        try:
            df_in = pd.read_csv(in_file)
            if len(df_in) > 0:
                n_in_original = len(df_in)
                
                # Sort by txId (temporal ordering)
                df_in = df_in.sort_values('txId', ascending=True)
                
                # Keep K most recent (highest txId)
                df_in = df_in.tail(K)
                
                # Create timestamp_proxy
                df_in['timestamp_proxy'] = df_in['txId'] / max_txid
                
                # Drop txId, keep features + timestamp
                feature_cols = [col for col in df_in.columns if col.startswith('feat#')]
                df_in = df_in[feature_cols + ['timestamp_proxy']]
                
                # Convert to numpy
                data = df_in.values  # Shape: (n, 96) where n <= K
                
                # Place at the end (pre-padding: zeros at beginning)
                start_idx = K - len(data)
                in_sequence[start_idx:] = data
                
        except Exception as e:
            logger.warning(f"Error processing in-flow for node {node_id}: {e}")
    
    # Process out-flow
    if out_file.exists():
        try:
            df_out = pd.read_csv(out_file)
            if len(df_out) > 0:
                n_out_original = len(df_out)
                
                # Sort by txId (temporal ordering)
                df_out = df_out.sort_values('txId', ascending=True)
                
                # Keep K most recent
                df_out = df_out.tail(K)
                
                # Create timestamp_proxy
                df_out['timestamp_proxy'] = df_out['txId'] / max_txid
                
                # Drop txId
                feature_cols = [col for col in df_out.columns if col.startswith('feat#')]
                df_out = df_out[feature_cols + ['timestamp_proxy']]
                
                # Convert to numpy
                data = df_out.values
                
                # Place at the end
                start_idx = K - len(data)
                out_sequence[start_idx:] = data
                
        except Exception as e:
            logger.warning(f"Error processing out-flow for node {node_id}: {e}")
    
    return in_sequence, out_sequence, n_in_original, n_out_original


def main():
    """Main function for Phase 3"""
    
    # Setup
    SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase3", LOG_FILE_PROCESSING)
    
    logger.info("="*60)
    logger.info("PHASE 3: BUILD SEQUENCES")
    logger.info("="*60)
    logger.info(f"Sequence length K: {SEQUENCE_LENGTH}")
    logger.info(f"Padding: Pre-padding (zeros at beginning)")
    
    monitor = MemoryMonitor(logger)
    monitor.log_memory("Initial")
    
    try:
        # Load mappings from Phase 1
        with Timer("Loading mappings", logger):
            node_to_idx = load_pickle(INDEX_DIR / 'node_to_idx.pkl')
            idx_to_node = load_pickle(INDEX_DIR / 'idx_to_node.pkl')
            node_labels = load_pickle(INDEX_DIR / 'node_labels.pkl')
            
            total_nodes = len(node_to_idx)
            logger.info(f"Total nodes to process: {total_nodes:,}")
        
        # Get max txId for normalization
        max_txid = get_max_txid(FEATURES_DIR, logger)
        logger.info(f"Max txId for normalization: {max_txid:,}")
        
        # Get checkpoint info
        start_idx, processed_nodes = get_checkpoint(SEQUENCES_DIR)
        if start_idx > 0:
            logger.info(f"Resuming from node index {start_idx:,}")
            logger.info(f"Previously processed: {len(processed_nodes):,} nodes")
        
        # Create list of (node_id, node_idx) tuples
        node_list = [(idx_to_node[idx], idx) for idx in range(total_nodes)]
        
        # Statistics
        stats = {
            'total_nodes': total_nodes,
            'processed': 0,
            'failed': 0,
            'empty_in': 0,
            'empty_out': 0
        }
        
        # Process nodes
        with Timer("Building sequences", logger):
            for i, (node_id, node_idx) in enumerate(node_list[start_idx:], start=start_idx):
                # Skip if already processed (resume)
                if node_idx in processed_nodes:
                    continue
                
                # Process this node
                try:
                    in_seq, out_seq, n_in, n_out = process_node_sequence(
                        node_id, node_idx, FEATURES_DIR, max_txid, logger
                    )
                    
                    # Get label
                    label = node_labels.get(node_id, 0)
                    
                    # Save as NPZ
                    output_file = SEQUENCES_DIR / f'node_{node_idx:06d}.npz'
                    np.savez(
                        output_file,
                        in_flow=in_seq,           # Shape: (50, 96)
                        out_flow=out_seq,         # Shape: (50, 96)
                        label=label,              # int
                        node_id=node_id,          # Original clId
                        n_in=n_in,                # Original count
                        n_out=n_out               # Original count
                    )
                    
                    stats['processed'] += 1
                    if n_in == 0:
                        stats['empty_in'] += 1
                    if n_out == 0:
                        stats['empty_out'] += 1
                    
                    processed_nodes.append(node_idx)
                    
                except Exception as e:
                    logger.error(f"Error processing node {node_id} (idx {node_idx}): {e}")
                    stats['failed'] += 1
                    continue
                
                # Log progress
                if (i + 1) % 10000 == 0:
                    progress = (i + 1) / total_nodes * 100
                    logger.info(f"Processed {i + 1:,} / {total_nodes:,} nodes "
                              f"({progress:.1f}%) - "
                              f"Success: {stats['processed']}, "
                              f"Failed: {stats['failed']}")
                    monitor.log_memory(f"After {i + 1} nodes")
                
                # Save checkpoint every N nodes
                if (i + 1) % SEQUENCE_BUILDING_BATCH_SIZE == 0:
                    save_checkpoint(SEQUENCES_DIR, i + 1, processed_nodes)
                    logger.info(f"✓ Checkpoint saved at node {i + 1:,}")
                    gc.collect()  # Clear memory
        
        # Final checkpoint
        save_checkpoint(SEQUENCES_DIR, total_nodes, processed_nodes)
        
        # Final stats
        logger.info("\n" + "="*60)
        logger.info("SEQUENCE BUILDING COMPLETE")
        logger.info("="*60)
        logger.info(f"Total nodes: {stats['total_nodes']:,}")
        logger.info(f"Successfully processed: {stats['processed']:,}")
        logger.info(f"Failed: {stats['failed']:,}")
        logger.info(f"Empty in-flow: {stats['empty_in']:,}")
        logger.info(f"Empty out-flow: {stats['empty_out']:,}")
        
        # Count output files
        output_files = list(SEQUENCES_DIR.glob('node_*.npz'))
        logger.info(f"NPZ files created: {len(output_files):,}")
        
        # Calculate disk usage
        total_size = sum(f.stat().st_size for f in output_files)
        logger.info(f"Total disk usage: {total_size / 1024**3:.2f} GB")
        
        logger.info("\n" + "="*60)
        logger.info("✓ PHASE 3 COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Sequence files saved to: {SEQUENCES_DIR}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error in Phase 3: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
