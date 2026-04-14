#!/usr/bin/env python3

import sys
import pickle
import time
from pathlib import Path
from typing import Dict, List, Tuple
import warnings
warnings.filterwarnings('ignore')

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit



def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def save_pickle(data, filepath):
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
    file_size = filepath.stat().st_size / 1024
    print(f"  Saved {filepath.name} ({file_size:.2f} KB)")


def create_splits(node_labels: Dict[int, int], 
                  idx_to_node: Dict[int, int],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  random_state: int = 42):
    
    print("Creating train/val/test splits...")
    
    # Convert to arrays
    num_nodes = len(idx_to_node)
    all_indices = np.arange(num_nodes)
    # Get labels by converting index -> node_id -> label
    all_labels = np.array([node_labels[idx_to_node[idx]] for idx in range(num_nodes)])
    
    print(f"Total nodes: {len(all_indices):,}")
    print(f"Class distribution: {np.bincount(all_labels)}")
    
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
    
    print(f"Train: {len(train_idx):,} ({len(train_idx)/len(all_indices)*100:.1f}%)")
    print(f"Val: {len(val_idx):,} ({len(val_idx)/len(all_indices)*100:.1f}%)")
    print(f"Test: {len(test_idx):,} ({len(test_idx)/len(all_indices)*100:.1f}%)")
    
    # Verify class distribution
    train_labels = all_labels[train_idx]
    val_labels = all_labels[val_idx]
    test_labels = all_labels[test_idx]
    
    print("Class distribution:")
    print(f"  Train: {np.bincount(train_labels)} ({np.bincount(train_labels)/len(train_labels)*100})")
    print(f"  Val: {np.bincount(val_labels)} ({np.bincount(val_labels)/len(val_labels)*100})")
    print(f"  Test: {np.bincount(test_labels)} ({np.bincount(test_labels)/len(test_labels)*100})")
    
    return train_idx, val_idx, test_idx


def build_graph_structure(edges_index: Dict[int, List[Tuple[int, int]]],
                          node_to_idx: Dict[int, int],
                          max_txid: int):
    
    print("Building graph structure...")
    
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
    
    print(f"Number of edges: {edge_index.shape[1]:,}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge attr shape: {edge_attr.shape}")
    
    return edge_index, edge_attr


def validate_output_files(graph_dir, expected_files):
    all_valid = True
    for fname in expected_files:
        fpath = graph_dir / fname
        if fpath.exists():
            print(f"  ✓ {fname}")
        else:
            print(f"  ✗ {fname} - MISSING!")
            all_valid = False
    return all_valid


def main():
    """Main function for Phase 4"""
    
    BASE_DIR = Path(__file__).parent.parent.parent
    INDEX_DIR = BASE_DIR / "data" / "processed" / "index"
    GRAPH_DIR = BASE_DIR / "data" / "processed" / "graph"
    
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    RANDOM_SEED = 42
    

    print(f"BUILD GRAPH - Phase 4")
    print(f"Train ratio: {TRAIN_RATIO}")
    print(f"Val ratio: {VAL_RATIO}")
    print(f"Test ratio: {TEST_RATIO}")
    print(f"Stratified: True")
    
    start_time = time.time()
    
    try:
        # Load data from previous phases
        print("Loading mappings from Phase 1...")
        node_to_idx = load_pickle(INDEX_DIR / 'node_to_idx.pkl')
        idx_to_node = load_pickle(INDEX_DIR / 'idx_to_node.pkl')
        node_labels = load_pickle(INDEX_DIR / 'node_labels.pkl')
        edges_index = load_pickle(INDEX_DIR / 'edges_index.pkl')
        
        print(f"Loaded {len(node_to_idx):,} nodes")
        print(f"Loaded {len(edges_index):,} nodes with edges")
        
        # Create train/val/test splits
        train_idx, val_idx, test_idx = create_splits(
            node_labels,
            idx_to_node,
            train_ratio=TRAIN_RATIO,
            val_ratio=VAL_RATIO,
            test_ratio=TEST_RATIO,
            random_state=RANDOM_SEED
        )
        
        # Save splits
        GRAPH_DIR.mkdir(parents=True, exist_ok=True)
        splits = {
            'train': train_idx,
            'val': val_idx,
            'test': test_idx
        }
        save_pickle(splits, GRAPH_DIR / 'train_val_test_split.pkl')
        
        # Build graph structure
        max_txid = 856_732_323  # From Phase 3
        
        edge_index, edge_attr = build_graph_structure(
            edges_index, node_to_idx, max_txid
        )
        
        # Save graph structure
        np.save(GRAPH_DIR / 'edge_index.npy', edge_index)
        np.save(GRAPH_DIR / 'edge_attr.npy', edge_attr)
        
        print(f"Saved edge_index.npy ({edge_index.nbytes / 1024**2:.2f} MB)")
        print(f"Saved edge_attr.npy ({edge_attr.nbytes / 1024**2:.2f} MB)")
        
        # Create metadata
        print("Creating metadata...")
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
        
        save_pickle(metadata, GRAPH_DIR / 'metadata.pkl')
        
        # Validate outputs
        expected_files = [
            'train_val_test_split.pkl',
            'edge_index.npy',
            'edge_attr.npy',
            'metadata.pkl'
        ]
        
        print("\nValidating output files...")
        all_valid = validate_output_files(GRAPH_DIR, expected_files)
        
        elapsed = time.time() - start_time
        
        if all_valid:
            print(f"COMPLETED SUCCESSFULLY - Phase 4")
            print(f"Graph files saved to: {GRAPH_DIR}")
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


if __name__ == "__main__":
    sys.exit(main())
