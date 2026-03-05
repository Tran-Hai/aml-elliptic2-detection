#!/usr/bin/env python3

import sys
import csv
import time
from pathlib import Path
from collections import defaultdict
from typing import Set

import pandas as pd


def load_pickle(filepath):
    import pickle
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def load_node_set(index_dir: Path) -> set:
    node_to_idx = load_pickle(index_dir / 'node_to_idx.pkl')
    return set(node_to_idx.keys())


def process_chunk(chunk: pd.DataFrame, target_nodes: set, features_dir: Path):
    
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


def initialize_feature_files(features_dir: Path, target_nodes: set, feature_cols: list):
    print("Initializing feature files...")
    
    headers = ['txId'] + feature_cols
    
    count = 0
    for node_id in target_nodes:
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
            print(f"  Initialized {count:,} / {len(target_nodes):,} nodes")
    
    print(f"Initialized {count:,} nodes")


def main():
    """Main function for Phase 2"""
    
    BASE_DIR = Path(__file__).parent.parent.parent
    BACKGROUND_EDGES_CSV = BASE_DIR / "data" / "raw" / "background_edges.csv"
    FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"
    INDEX_DIR = BASE_DIR / "data" / "processed" / "index"
    CHUNK_SIZE = 50000
    
    print(f"\n{'='*60}")
    print(f"EXTRACT FEATURES - Phase 2")
    print('='*60)
    
    start_time = time.time()
    
    try:
        # Check if background_edges.csv exists
        if not BACKGROUND_EDGES_CSV.exists():
            print(f"ERROR: Background edges file not found: {BACKGROUND_EDGES_CSV}")
            return 1
        
        # Load target nodes
        print("Loading target nodes...")
        target_nodes = load_node_set(INDEX_DIR)
        print(f"Target nodes: {len(target_nodes):,}")
        
        # Get total file size
        file_size = BACKGROUND_EDGES_CSV.stat().st_size
        print(f"Background edges file size: {file_size / 1024**3:.2f} GB")
        
        # Read first chunk to get feature columns
        print("Reading first chunk to get feature columns...")
        first_chunk = pd.read_csv(BACKGROUND_EDGES_CSV, nrows=5)
        feature_cols = [col for col in first_chunk.columns if col.startswith('feat#')]
        print(f"Feature columns: {len(feature_cols)} (feat#1 to feat#{len(feature_cols)})")
        
        # Initialize feature files
        FEATURES_DIR.mkdir(parents=True, exist_ok=True)
        initialize_feature_files(FEATURES_DIR, target_nodes, feature_cols)
        
        # Process background_edges.csv in chunks
        print(f"Processing background edges in chunks of {CHUNK_SIZE:,} rows...")
        
        chunk_num = 0
        total_edges_extracted = 0
        
        for chunk in pd.read_csv(BACKGROUND_EDGES_CSV, chunksize=CHUNK_SIZE):
            chunk_num += 1
            
            # Process this chunk
            edges_written, nodes_affected = process_chunk(
                chunk, target_nodes, FEATURES_DIR
            )
            
            total_edges_extracted += edges_written
            
            # Log progress every 100 chunks
            if chunk_num % 100 == 0:
                print(f"Chunk {chunk_num:,}: Extracted {edges_written:,} edges, "
                          f"Affected {nodes_affected:,} nodes. "
                          f"Total extracted: {total_edges_extracted:,}")
            
            # Clear chunk from memory
            del chunk
        
        # Final stats
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"EXTRACTION COMPLETE - Phase 2")
        print('='*60)
        print(f"Total chunks processed: {chunk_num:,}")
        print(f"Total edges extracted: {total_edges_extracted:,}")
        
        # Count created files
        in_files = list(FEATURES_DIR.glob('node_*_in.csv'))
        out_files = list(FEATURES_DIR.glob('node_*_out.csv'))
        print(f"In-flow files created: {len(in_files):,}")
        print(f"Out-flow files created: {len(out_files):,}")
        
        print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
        print(f"\n{'='*60}")
        print(f"COMPLETED SUCCESSFULLY - Phase 2")
        print('='*60)
        print(f"Feature files saved to: {FEATURES_DIR}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
