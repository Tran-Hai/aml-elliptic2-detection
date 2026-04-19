#!/usr/bin/env python3

import sys
import time
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import pickle


def load_pickle(filepath):
    with open(filepath, 'rb') as f:
        return pickle.load(f)


def get_max_txid(features_dir: Path) -> int:
    print("Finding maximum txId for normalization...")
    
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
        print(f"Estimated max txId from sample: {estimated_max:,}")
        return int(estimated_max * 1.1)
    else:
        return 800_000_000


def process_node_sequence(node_id: int, features_dir: Path, max_txid: int, K: int = 50):
    
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
                data = df_in.values
                
                # Place at the end (pre-padding: zeros at beginning)
                start_idx = K - len(data)
                in_sequence[start_idx:] = data
                
        except Exception as e:
            print(f"Warning: Error processing in-flow for node {node_id}: {e}")
    
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
            print(f"Warning: Error processing out-flow for node {node_id}: {e}")
    
    return in_sequence, out_sequence, n_in_original, n_out_original


def main():
    """Main function for Phase 3"""
    
    BASE_DIR = Path(__file__).parent.parent.parent
    FEATURES_DIR = BASE_DIR / "data" / "processed" / "features"
    INDEX_DIR = BASE_DIR / "data" / "processed" / "index"
    SEQUENCES_DIR = BASE_DIR / "data" / "processed" / "sequences"
    SEQUENCE_LENGTH = 50
    
    print(f"\n{'='*60}")
    print(f"BUILD SEQUENCES - Phase 3")
    print('='*60)
    print(f"Sequence length K: {SEQUENCE_LENGTH}")
    print(f"Padding: Pre-padding (zeros at beginning)")
    
    start_time = time.time()
    
    try:
        # Load mappings from Phase 1
        print("Loading mappings...")
        node_to_idx = load_pickle(INDEX_DIR / 'node_to_idx.pkl')
        idx_to_node = load_pickle(INDEX_DIR / 'idx_to_node.pkl')
        node_labels = load_pickle(INDEX_DIR / 'node_labels.pkl')
        
        total_nodes = len(node_to_idx)
        print(f"Total nodes to process: {total_nodes:,}")
        
        # Get max txId for normalization
        max_txid = get_max_txid(FEATURES_DIR)
        print(f"Max txId for normalization: {max_txid:,}")
        
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
        
        # Create output directory
        SEQUENCES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Process nodes
        print("Building sequences...")
        
        for i, (node_id, node_idx) in enumerate(node_list):
            try:
                in_seq, out_seq, n_in, n_out = process_node_sequence(
                    node_id, FEATURES_DIR, max_txid, SEQUENCE_LENGTH
                )
                
                # Get label
                label = node_labels.get(node_id, 0)
                
                # Save as NPZ
                output_file = SEQUENCES_DIR / f'node_{node_idx:06d}.npz'
                np.savez(
                    output_file,
                    in_flow=in_seq,
                    out_flow=out_seq,
                    label=label,
                    node_id=node_id,
                    n_in=n_in,
                    n_out=n_out
                )
                
                stats['processed'] += 1
                if n_in == 0:
                    stats['empty_in'] += 1
                if n_out == 0:
                    stats['empty_out'] += 1
                
            except Exception as e:
                print(f"Error processing node {node_id} (idx {node_idx}): {e}")
                stats['failed'] += 1
                continue
            
            # Log progress
            if (i + 1) % 10000 == 0:
                progress = (i + 1) / total_nodes * 100
                print(f"Processed {i + 1:,} / {total_nodes:,} nodes ({progress:.1f}%) - "
                          f"Success: {stats['processed']}, Failed: {stats['failed']}")
        
        # Final stats
        elapsed = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"SEQUENCE BUILDING COMPLETE - Phase 3")
        print('='*60)
        print(f"Total nodes: {stats['total_nodes']:,}")
        print(f"Successfully processed: {stats['processed']:,}")
        print(f"Failed: {stats['failed']:,}")
        print(f"Empty in-flow: {stats['empty_in']:,}")
        print(f"Empty out-flow: {stats['empty_out']:,}")
        
        # Count output files
        output_files = list(SEQUENCES_DIR.glob('node_*.npz'))
        print(f"NPZ files created: {len(output_files):,}")
        
        # Calculate disk usage
        total_size = sum(f.stat().st_size for f in output_files)
        print(f"Total disk usage: {total_size / 1024**3:.2f} GB")
        
        print(f"Total time: {elapsed:.2f} seconds ({elapsed/60:.1f} minutes)")
        print(f"\n{'='*60}")
        print(f"COMPLETED SUCCESSFULLY - Phase 3")
        print('='*60)
        print(f"Sequence files saved to: {SEQUENCES_DIR}")
        
        return 0
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
