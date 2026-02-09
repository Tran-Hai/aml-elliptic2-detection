#!/usr/bin/env python3
"""
Phase 1: Build Index
AML Detection Project - Elliptic2 Dataset

This script creates lookup tables for efficient data access:
1. node_to_idx.pkl - Maps clId to index (0, 1, 2...)
2. idx_to_node.pkl - Reverse mapping
3. node_labels.pkl - Maps clId to label (0=licit, 1=suspicious)
4. component_to_nodes.pkl - Maps ccId to list of clIds
5. edges_index.pkl - Adjacency list

Memory: ~100 MB
Time: 5-10 minutes
"""

import sys
import pickle
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np

# Add parent directory to path
sys.path.append(str(Path(__file__).parent))
from utils import (
    setup_logger, Timer, MemoryMonitor, 
    save_pickle, print_dataframe_info,
    create_adjacency_list, validate_processed_data
)
from config import (
    NODES_CSV, EDGES_CSV, CONNECTED_COMPONENTS_CSV,
    INDEX_DIR, LOG_FILE_PROCESSING, RANDOM_SEED
)


def build_node_mappings(nodes_df: pd.DataFrame, components_df: pd.DataFrame, logger):
    """
    Build node-to-index and index-to-node mappings
    
    Args:
        nodes_df: Nodes DataFrame
        components_df: Components DataFrame  
        logger: Logger instance
    
    Returns:
        node_to_idx, idx_to_node, node_labels, component_to_nodes
    """
    logger.info("Building node mappings...")
    
    # Merge nodes with components to get labels
    logger.info("Merging nodes with component labels...")
    nodes_labeled = nodes_df.merge(components_df, on='ccId', how='left')
    
    # Check for missing labels
    missing_labels = nodes_labeled['ccLabel'].isnull().sum()
    if missing_labels > 0:
        logger.warning(f"Found {missing_labels} nodes without labels!")
    else:
        logger.info("✓ All nodes have labels")
    
    # Convert labels to binary (0=licit, 1=suspicious)
    nodes_labeled['label'] = (nodes_labeled['ccLabel'] == 'suspicious').astype(int)
    
    # Print label distribution
    label_counts = nodes_labeled['label'].value_counts().sort_index()
    logger.info(f"Label distribution:")
    logger.info(f"  Licit (0):      {label_counts[0]:,} ({label_counts[0]/len(nodes_labeled)*100:.2f}%)")
    logger.info(f"  Suspicious (1): {label_counts[1]:,} ({label_counts[1]/len(nodes_labeled)*100:.2f}%)")
    
    # Create node to index mapping
    logger.info("Creating node-to-index mapping...")
    unique_nodes = sorted(nodes_labeled['clId'].unique())
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    logger.info(f"Total unique nodes: {len(node_to_idx):,}")
    
    # Create node labels mapping
    node_labels = dict(zip(nodes_labeled['clId'], nodes_labeled['label']))
    
    # Create component to nodes mapping
    logger.info("Creating component-to-nodes mapping...")
    component_to_nodes = defaultdict(list)
    for _, row in nodes_labeled.iterrows():
        component_to_nodes[row['ccId']].append(row['clId'])
    
    component_to_nodes = dict(component_to_nodes)
    logger.info(f"Total components: {len(component_to_nodes):,}")
    
    # Log component size statistics
    comp_sizes = [len(nodes) for nodes in component_to_nodes.values()]
    logger.info(f"Component size statistics:")
    logger.info(f"  Mean: {np.mean(comp_sizes):.2f}")
    logger.info(f"  Median: {np.median(comp_sizes):.2f}")
    logger.info(f"  Max: {np.max(comp_sizes)}")
    logger.info(f"  Min: {np.min(comp_sizes)}")
    
    return node_to_idx, idx_to_node, node_labels, component_to_nodes


def build_edges_index(edges_df: pd.DataFrame, logger):
    """
    Build adjacency list from edges
    
    Args:
        edges_df: Edges DataFrame
        logger: Logger instance
    
    Returns:
        Adjacency list dictionary
    """
    logger.info("Building edges index (adjacency list)...")
    
    # Use utility function
    adjacency_list = create_adjacency_list(edges_df, logger)
    
    # Log statistics
    num_nodes_with_edges = len(adjacency_list)
    total_edges = sum(len(neighbors) for neighbors in adjacency_list.values()) // 2  # Undirected
    
    logger.info(f"Adjacency list statistics:")
    logger.info(f"  Nodes with edges: {num_nodes_with_edges:,}")
    logger.info(f"  Total unique edges: {total_edges:,}")
    
    # Calculate degree statistics
    degrees = [len(neighbors) for neighbors in adjacency_list.values()]
    logger.info(f"Degree statistics:")
    logger.info(f"  Mean: {np.mean(degrees):.2f}")
    logger.info(f"  Median: {np.median(degrees):.2f}")
    logger.info(f"  Max: {np.max(degrees)}")
    
    return adjacency_list


def main():
    """Main function for Phase 1"""
    
    # Setup
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    logger = setup_logger("Phase1", LOG_FILE_PROCESSING)
    
    logger.info("="*60)
    logger.info("PHASE 1: BUILD INDEX")
    logger.info("="*60)
    
    monitor = MemoryMonitor(logger)
    monitor.log_memory("Initial")
    
    try:
        # Step 1: Load data
        with Timer("Loading data", logger):
            logger.info(f"Loading nodes from {NODES_CSV}")
            nodes_df = pd.read_csv(NODES_CSV)
            print_dataframe_info(nodes_df, "Nodes", logger)
            
            logger.info(f"Loading edges from {EDGES_CSV}")
            edges_df = pd.read_csv(EDGES_CSV)
            print_dataframe_info(edges_df, "Edges", logger)
            
            logger.info(f"Loading components from {CONNECTED_COMPONENTS_CSV}")
            components_df = pd.read_csv(CONNECTED_COMPONENTS_CSV)
            print_dataframe_info(components_df, "Components", logger)
            
            monitor.log_memory("After loading")
        
        # Step 2: Build node mappings
        with Timer("Building node mappings", logger):
            node_to_idx, idx_to_node, node_labels, component_to_nodes = \
                build_node_mappings(nodes_df, components_df, logger)
            monitor.log_memory("After node mappings")
        
        # Step 3: Build edges index
        with Timer("Building edges index", logger):
            edges_index = build_edges_index(edges_df, logger)
            monitor.log_memory("After edges index")
        
        # Step 4: Save all indices
        with Timer("Saving index files", logger):
            save_pickle(node_to_idx, INDEX_DIR / 'node_to_idx.pkl', logger)
            save_pickle(idx_to_node, INDEX_DIR / 'idx_to_node.pkl', logger)
            save_pickle(node_labels, INDEX_DIR / 'node_labels.pkl', logger)
            save_pickle(component_to_nodes, INDEX_DIR / 'component_to_nodes.pkl', logger)
            save_pickle(edges_index, INDEX_DIR / 'edges_index.pkl', logger)
        
        # Step 5: Validate
        expected_files = [
            'node_to_idx.pkl',
            'idx_to_node.pkl',
            'node_labels.pkl',
            'component_to_nodes.pkl',
            'edges_index.pkl'
        ]
        
        logger.info("\nValidating output files...")
        all_valid = validate_processed_data(INDEX_DIR, expected_files, logger)
        
        if all_valid:
            logger.info("\n" + "="*60)
            logger.info("✓ PHASE 1 COMPLETED SUCCESSFULLY!")
            logger.info("="*60)
            logger.info(f"Index files saved to: {INDEX_DIR}")
            return 0
        else:
            logger.error("✗ Some files are missing!")
            return 1
            
    except Exception as e:
        logger.error(f"Error in Phase 1: {str(e)}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
