#!/usr/bin/env python3

import sys
import pickle
import time
from pathlib import Path
from collections import defaultdict

import pandas as pd
import numpy as np


def save_pickle(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    file_size = filepath.stat().st_size / 1024
    print(f"  Saved {filepath.name} ({file_size:.2f} KB)")


def create_adjacency_list(edges_df):
    adjacency_list = defaultdict(list)
    total_edges = len(edges_df)
    
    for idx, row in edges_df.iterrows():
        src, dst, tx_id = row['clId1'], row['clId2'], row['txId']
        adjacency_list[src].append((dst, tx_id))
        adjacency_list[dst].append((src, tx_id))
        
        if (idx + 1) % 100000 == 0:
            progress = (idx + 1) / total_edges * 100
            print(f"  Processing edges: {idx + 1:,} / {total_edges:,} ({progress:.1f}%)")
    
    return dict(adjacency_list)


def validate_output_files(index_dir, expected_files):
    all_valid = True
    for fname in expected_files:
        fpath = index_dir / fname
        if fpath.exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} - MISSING!")
            all_valid = False
    return all_valid


def build_node_mappings(nodes_df, components_df):
    
    print("Building node mappings...")
    
    # Merge nodes with components to get labels
    print("Merging nodes with component labels...")
    nodes_labeled = nodes_df.merge(components_df, on='ccId', how='left')
    
    # Check for missing labels
    missing_labels = nodes_labeled['ccLabel'].isnull().sum()
    if missing_labels > 0:
        print(f"WARNING: Found {missing_labels} nodes without labels!")
    else:
        print("All nodes have labels")
    
    # Convert labels to binary (0=licit, 1=suspicious)
    nodes_labeled['label'] = (nodes_labeled['ccLabel'] == 'suspicious').astype(int)
    
    # Print label distribution
    label_counts = nodes_labeled['label'].value_counts().sort_index()
    print(f"Label distribution:")
    print(f"  Licit (0):      {label_counts[0]:,} ({label_counts[0]/len(nodes_labeled)*100:.2f}%)")
    print(f"  Suspicious (1): {label_counts[1]:,} ({label_counts[1]/len(nodes_labeled)*100:.2f}%)")
    
    # Create node to index mapping
    print("Creating node-to-index mapping...")
    unique_nodes = sorted(nodes_labeled['clId'].unique())
    node_to_idx = {node: idx for idx, node in enumerate(unique_nodes)}
    idx_to_node = {idx: node for node, idx in node_to_idx.items()}
    
    print(f"Total unique nodes: {len(node_to_idx):,}")
    
    # Create node labels mapping
    node_labels = dict(zip(nodes_labeled['clId'], nodes_labeled['label']))
    
    # Create component to nodes mapping
    print("Creating component-to-nodes mapping...")
    component_to_nodes = defaultdict(list)
    for _, row in nodes_labeled.iterrows():
        component_to_nodes[row['ccId']].append(row['clId'])
    
    component_to_nodes = dict(component_to_nodes)
    print(f"Total components: {len(component_to_nodes):,}")
    
    # Log component size statistics
    comp_sizes = [len(nodes) for nodes in component_to_nodes.values()]
    print(f"Component size statistics:")
    print(f"  Mean: {np.mean(comp_sizes):.2f}")
    print(f"  Median: {np.median(comp_sizes):.2f}")
    print(f"  Max: {np.max(comp_sizes)}")
    print(f"  Min: {np.min(comp_sizes)}")
    
    return node_to_idx, idx_to_node, node_labels, component_to_nodes


def build_edges_index(edges_df):
    
    print("Building edges index (adjacency list)...")
    
    adjacency_list = create_adjacency_list(edges_df)
    
    # Log statistics
    num_nodes_with_edges = len(adjacency_list)
    total_edges = sum(len(neighbors) for neighbors in adjacency_list.values()) // 2
    
    print(f"Adjacency list statistics:")
    print(f"  Nodes with edges: {num_nodes_with_edges:,}")
    print(f"  Total unique edges: {total_edges:,}")
    
    # Calculate degree statistics
    degrees = [len(neighbors) for neighbors in adjacency_list.values()]
    print(f"Degree statistics:")
    print(f"  Mean: {np.mean(degrees):.2f}")
    print(f"  Median: {np.median(degrees):.2f}")
    print(f"  Max: {np.max(degrees)}")
    
    return adjacency_list


def main():
    """Main function for Phase 1"""
    
    BASE_DIR = Path(__file__).parent.parent.parent
    NODES_CSV = BASE_DIR / "data" / "raw" / "nodes.csv"
    EDGES_CSV = BASE_DIR / "data" / "raw" / "edges.csv"
    CONNECTED_COMPONENTS_CSV = BASE_DIR / "data" / "raw" / "connected_components.csv"
    INDEX_DIR = BASE_DIR / "data" / "processed" / "index"
    

    print(f"BUILD INDEX - Phase 1")
    
    start_time = time.time()
    
    try:
        # Step 1: Load data
        print("Loading data...")
        
        print(f"  Loading nodes from {NODES_CSV}")
        nodes_df = pd.read_csv(NODES_CSV)
        print(f"    Nodes shape: {nodes_df.shape}")
        
        print(f"  Loading edges from {EDGES_CSV}")
        edges_df = pd.read_csv(EDGES_CSV)
        print(f"    Edges shape: {edges_df.shape}")
        
        print(f"  Loading components from {CONNECTED_COMPONENTS_CSV}")
        components_df = pd.read_csv(CONNECTED_COMPONENTS_CSV)
        print(f"    Components shape: {components_df.shape}")
        
        # Step 2: Build node mappings
        print("Building node mappings...")
        node_to_idx, idx_to_node, node_labels, component_to_nodes = \
            build_node_mappings(nodes_df, components_df)
        
        # Step 3: Build edges index
        print("Building edges index...")
        edges_index = build_edges_index(edges_df)
        
        # Step 4: Save all indices
        print("Saving index files...")
        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        save_pickle(node_to_idx, INDEX_DIR / 'node_to_idx.pkl')
        save_pickle(idx_to_node, INDEX_DIR / 'idx_to_node.pkl')
        save_pickle(node_labels, INDEX_DIR / 'node_labels.pkl')
        save_pickle(component_to_nodes, INDEX_DIR / 'component_to_nodes.pkl')
        save_pickle(edges_index, INDEX_DIR / 'edges_index.pkl')
        
        # Step 5: Validate
        expected_files = [
            'node_to_idx.pkl',
            'idx_to_node.pkl',
            'node_labels.pkl',
            'component_to_nodes.pkl',
            'edges_index.pkl'
        ]
        
        print("\nValidating output files...")
        all_valid = validate_output_files(INDEX_DIR, expected_files)
        
        elapsed = time.time() - start_time
        
        if all_valid:
            print(f"COMPLETED SUCCESSFULLY - Phase 1")
            print(f"Index files saved to: {INDEX_DIR}")
            print(f"Total time: {elapsed:.2f} seconds")
            return 0
        else:
            print("ERROR: Some files are missing!")
            return 1
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1
            
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
