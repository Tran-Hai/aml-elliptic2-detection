#!/usr/bin/env python3
"""
Phase 4: Build Graph Structure
AML Detection Project - Elliptic2 Dataset

This script creates the final graph structure for training:
1. Create train/val/test split (stratified)
2. Build edge_index and edge_attr for PyTorch Geometric
3. Save metadata for training

Memory: ~100 MB
Time: 5-10 minutes
"""

import sys
import pickle
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit

sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logger, Timer, MemoryMonitor,
    save_pickle, load_pickle, validate_processed_data
)
from config import (
    INDEX_DIR, GRAPH_DIR, SEQUENCES_DIR,
    LOG_FILE_PROCESSING,
    TRAIN_RATIO, VAL_RATIO, TEST_RATIO,
    RANDOM_SEED, STRATIFIED_SPLIT
)


def create_splits(node_labels: Dict[int, int], 
                  idx_to_node: Dict[int, int],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_state: int = 42,
                  logger = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create stratified train/val/test splits
    
    Args:
        node_labels: Dict mapping node_id (clId) to label
        idx_to_node: Dict mapping index to node_id (clId)
        train_ratio: Proportion for training
        val_ratio: Proportion for validation
        test_ratio: Proportion for testing
        random_state: Random seed
        logger: Logger instance
    
    Returns:
        (train_indices, val_indices, test_indices)
    """
    if logger:
        logger.info("Creating train/val/test splits...")
    
    # Convert to arrays
    num_nodes = len(idx_to_node)
    all_indices = np.arange(num_nodes)
    # Get labels by converting index -> node_id -> label
    all_labels = np.array([node_labels[idx_to_node[idx]] for idx in range(num_nodes)])
    
    if logger:
        logger.info(f"Total nodes: {len(all_indices):,}")
        logger.info(f"Class distribution: {np.bincount(all_labels)}")
    
    # First split: train vs (val + test)
    val_test_ratio = val_ratio + test_ratio
    
    sss1 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=val_test_ratio,
        random_state=random_state
    )
    
    train_idx, val_test_idx = next(sss1.split(all_indices, all_labels))
    
    # Second split: val vs test
    val_test_labels = all_labels[val_test_idx]
    val_ratio_adjusted = val_ratio / val_test_ratio
    
    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=(1 - val_ratio_adjusted),
        random_state=random_state
    )
    
    val_idx_local, test_idx_local = next(sss2.split(val_test_idx, val_test_labels))
    
    # Convert local indices to global
    val_idx = val_test_idx[val_idx_local]
    test_idx = val_test_idx[test_idx_local]
    
    if logger:
        logger.info(f"Train: {len(train_idx):,} ({len(train_idx)/len(all_indices)*100:.1f}%)")
        logger.info(f"Val: {len(val_idx):,} ({len(val_idx)/len(all_indices)*100:.1f}%)")
        logger.info(f"Test: {len(test_idx):,} ({len(test_idx)/len(all_indices)*100:.1f}%)")
        
        # Verify class distribution
        train_labels = all_labels[train_idx]
        val_labels = all_labels[val_idx]
        test_labels = all_labels[test_idx]
        
        logger.info("Class distribution:")
        logger.info(f"  Train: {np.bincount(train_labels)} ({np.bincount(train_labels)/len(train_labels)*100})")
        logger.info(f"  Val: {np.bincount(val_labels)} ({np.bincount(val_labels)/len(val_labels)*100})")
        logger.info(f"  Test: {np.bincount(test_labels)} ({np.bincount(test_labels)/len(test_labels)*100})")
    
    return train_idx, val_idx, test_idx


def build_graph_structure(edges_index: Dict[int, List[Tuple[int, int]]],
                          node_to_idx: Dict[int, int],
                          max_txid: int,
                          logger) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build edge_index and edge_attr for PyTorch Geometric
    
    Args:
        edges_index: Adjacency list from Phase 1
        node_to_idx: Mapping from clId to index
        max_txid: Maximum txId for normalization
        logger: Logger instance
    
    Returns:
        (edge_index, edge_attr)
        edge_index: np.array shape [2, num_edges]
        edge_attr: np.array shape [num_edges, 1] (timestamp)
    """
    if logger:
        logger.info("Building graph structure...")
    
    edge_list = []
    edge_attrs = []
    
    for src_clId, neighbors in edges_index.items():
        if src_clId not in node_to_idx:
            continue
        
        src_idx = node_to_idx[src_clId]
        
        for dst_clId, tx_id in neighbors:
            if dst_clId not in node_to_idx:
                continue
            
            dst_idx = node_to_idx[dst_clId]
            
            # Normalize timestamp
            timestamp = tx_id / max_txid
            
            edge_list.append([src_idx, dst_idx])
            edge_attrs.append([timestamp])
    
    edge_index = np.array(edge_list).T  # Shape: [2, num_edges]
    edge_attr = np.array(edge_attrs)    # Shape: [num_edges, 1]
    
    if logger:
        logger.info(f"Number of edges: {edge_index.shape[1]:,}")
        logger.info(f"Edge index shape: {edge_index.shape}")
        logger.info(f"Edge attr shape: {edge_attr.shape}")
    
    return edge_index, edge_attr


def main():
    """Main function for Phase 4"""
    
    # Setup
    GRAPH_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase4", LOG_FILE_PROCESSING)
    
    logger.info("="*60)
    logger.info("PHASE 4: BUILD GRAPH")
    logger.info("="*60)
    logger.info(f"Train ratio: {TRAIN_RATIO}")
    logger.info(f"Val ratio: {VAL_RATIO}")
    logger.info(f"Test ratio: {TEST_RATIO}")
    logger.info(f"Stratified: {STRATIFIED_SPLIT}")
    
    monitor = MemoryMonitor(logger)
    monitor.log_memory("Initial")
    
    try:
        # Load data from previous phases
        with Timer("Loading data", logger):
            logger.info("Loading mappings from Phase 1...")
            node_to_idx = load_pickle(INDEX_DIR / 'node_to_idx.pkl')
            idx_to_node = load_pickle(INDEX_DIR / 'idx_to_node.pkl')
            node_labels = load_pickle(INDEX_DIR / 'node_labels.pkl')
            edges_index = load_pickle(INDEX_DIR / 'edges_index.pkl')
            
            logger.info(f"Loaded {len(node_to_idx):,} nodes")
            logger.info(f"Loaded {len(edges_index):,} nodes with edges")
            
            monitor.log_memory("After loading mappings")
        
        # Create train/val/test splits
        with Timer("Creating splits", logger):
            train_idx, val_idx, test_idx = create_splits(
                node_labels,
                idx_to_node,
                train_ratio=TRAIN_RATIO,
                val_ratio=VAL_RATIO,
                test_ratio=TEST_RATIO,
                random_state=RANDOM_SEED,
                logger=logger
            )
            
            # Save splits
            splits = {
                'train': train_idx,
                'val': val_idx,
                'test': test_idx
            }
            save_pickle(splits, GRAPH_DIR / 'train_val_test_split.pkl', logger)
            
            monitor.log_memory("After creating splits")
        
        # Build graph structure
        with Timer("Building graph structure", logger):
            # Get max txId (same as Phase 3)
            max_txid = 856_732_323  # From Phase 3
            
            edge_index, edge_attr = build_graph_structure(
                edges_index, node_to_idx, max_txid, logger
            )
            
            # Save graph structure
            np.save(GRAPH_DIR / 'edge_index.npy', edge_index)
            np.save(GRAPH_DIR / 'edge_attr.npy', edge_attr)
            
            logger.info(f"Saved edge_index.npy ({edge_index.nbytes / 1024**2:.2f} MB)")
            logger.info(f"Saved edge_attr.npy ({edge_attr.nbytes / 1024**2:.2f} MB)")
            
            monitor.log_memory("After building graph")
        
        # Create metadata
        with Timer("Creating metadata", logger):
            metadata = {
                'num_nodes': len(node_to_idx),
                'num_edges': edge_index.shape[1],
                'num_features': 96,  # 95 + 1 timestamp
                'sequence_length': 50,
                'num_classes': 2,
                'class_distribution': {
                    'train': np.bincount([node_labels[idx_to_node[int(i)]] for i in train_idx]).tolist(),
                    'val': np.bincount([node_labels[idx_to_node[int(i)]] for i in val_idx]).tolist(),
                    'test': np.bincount([node_labels[idx_to_node[int(i)]] for i in test_idx]).tolist()
                },
                'split_sizes': {
                    'train': len(train_idx),
                    'val': len(val_idx),
                    'test': len(test_idx)
                }
            }
            
            save_pickle(metadata, GRAPH_DIR / 'metadata.pkl', logger)
        
        # Validate outputs
        expected_files = [
            'train_val_test_split.pkl',
            'edge_index.npy',
            'edge_attr.npy',
            'metadata.pkl'
        ]
        
        logger.info("\nValidating output files...")
        all_valid = validate_processed_data(GRAPH_DIR, expected_files, logger)
        
        if all_valid:
            logger.info("\n" + "="*60)
            logger.info("✓ PHASE 4 COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Graph files saved to: {GRAPH_DIR}")
            logger.info("\nReady for training!")
            logger.info("Next steps:")
            logger.info("  1. Create PyTorch Dataset class")
            logger.info("  2. Implement LAS-Mamba-GNN model")
            logger.info("  3. Start training!")
            return 0
        else:
            logger.error("✗ Some files are missing!")
            return 1
            
    except Exception as e:
        logger.error(f"Error in Phase 4: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
