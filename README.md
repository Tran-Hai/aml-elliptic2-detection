# AML Detection Project - Elliptic2 Dataset Processing

Dự án phát hiện rửa tiền (Anti-Money Laundering) sử dụng dataset Elliptic2 với kiến trúc LAS-Mamba-GNN.

---

## 📁 Cấu trúc Thư mục

### `/data/`
Chứa toàn bộ dữ liệu

#### `/data/raw/`
Dữ liệu gốc từ Elliptic2 dataset:
- `nodes.csv` - 444,521 nodes với clId và ccId
- `edges.csv` - 367,137 edges với clId1, clId2, txId
- `connected_components.csv` - 121,811 components với labels
- `background_nodes.csv` - 49M nodes với 43 features
- `background_edges.csv` - 196M edges với 95 features (78GB)

#### `/data/processed/`
Dữ liệu đã xử lý, được tạo qua 4 phases:

**`/data/processed/index/`** (Phase 1)
- Lookup tables để tra cứu nhanh
- Size: ~10 MB

**`/data/processed/features/`** (Phase 2)
- Individual feature files cho từng node
- Format: `node_{clId}_in.csv` và `node_{clId}_out.csv`
- Size: ~2-3 GB

**`/data/processed/sequences/`** (Phase 3)
- Temporal sequences đã xây dựng (numpy arrays)
- Format: `node_{idx}.npz` chứa in_flow và out_flow
- Shape: (2, 50, 96) = [in/out, K=50 transactions, 96 features]
- Size: ~3-4 GB

**`/data/processed/graph/`** (Phase 4)
- Graph structure cho GNN training
- edge_index.pt, edge_attr.pt
- Train/val/test split indices
- Size: ~100 MB

---

### `/src/`
Mã nguồn chính

#### `/src/data_processing/`
Scripts xử lý data (4 phases):
- `phase1_build_index.py` - Build lookup tables
- `phase2_extract_features.py` - Extract features từ background data
- `phase3_build_sequences.py` - Build temporal sequences
- `phase4_build_graph.py` - Build graph structure
- `config.py` - Configuration parameters
- `utils.py` - Helper functions

#### `/src/models/`
Kiến trúc model:
- `las_mamba_gnn.py` - LAS-Mamba-GNN implementation
- `mamba_block.py` - Mamba SSM block
- `layers.py` - Custom layers (Signed Message Passing, etc.)

#### `/src/utils/`
Tiện ích:
- `logger.py` - Logging utilities
- `metrics.py` - Evaluation metrics
- `visualization.py` - Plotting functions

---

### `/notebooks/`
Jupyter notebooks cho:
- Exploratory Data Analysis (EDA)
- Experiment tracking
- Visualization
- Debugging

---

### `/docs/`
Tài liệu:
- Báo cáo đồ án
- Papers (LAs-GNN, Mamba)
- Meeting notes
- Literature review

---

### `/logs/`
Logs training và processing:
- `processing_logs/` - Phase 1-4 execution logs
- `training_logs/` - Model training logs
- `tensorboard/` - TensorBoard logs

---

### `/checkpoints/`
Model checkpoints:
- `phase2_checkpoints/` - Resume capability cho feature extraction
- `phase3_checkpoints/` - Resume capability cho sequence building
- `model_checkpoints/` - Trained model weights

---

### `/tests/`
Unit tests cho các modules

---

### `/scripts/`
Scripts hỗ trợ:
- `setup_env.sh` - Setup environment
- `run_processing.sh` - Run all 4 phases
- `monitor_ram.sh` - Monitor RAM usage
- `cleanup.sh` - Clean temporary files

---

## 🚀 Quy trình Xử lý

### Phase 1: Build Index
```bash
python src/data_processing/phase1_build_index.py
```
- Input: data/raw/*.csv
- Output: data/processed/index/
- Time: 5-10 phút
- RAM: ~100 MB

### Phase 2: Extract Features
```bash
python src/data_processing/phase2_extract_features.py
```
- Input: data/raw/background_edges.csv (78GB)
- Output: data/processed/features/
- Time: 2-3 giờ
- RAM: ~200 MB (streaming)
- Checkpoint: Mỗi 1000 chunks

### Phase 3: Build Sequences
```bash
python src/data_processing/phase3_build_sequences.py
```
- Input: data/processed/features/
- Output: data/processed/sequences/
- Time: 30-60 phút
- RAM: ~100 MB
- Checkpoint: Mỗi 10k nodes

### Phase 4: Build Graph
```bash
python src/data_processing/phase4_build_graph.py
```
- Input: data/raw/edges.csv + index
- Output: data/processed/graph/
- Time: 5-10 phút
- RAM: ~100 MB

---

## 📊 Thông số Kỹ thuật

### Dataset
- **Nodes**: 444,521
- **Edges**: 367,137
- **Labels**: 97.7% licit, 2.3% suspicious
- **Sequence length (K)**: 50 transactions
- **Features**: 95 edge features + txId + timestamp_proxy = 97 dims

### Resources
- **RAM available**: 4 GB
- **Disk space needed**: ~8-10 GB cho processed data
- **Processing time**: ~3-4 giờ tổng cộng

---

## 📝 Ghi chú

- Tất cả các phase đều có thể pause/resume nếu bị gián đoạn
- Logs được lưu chi tiết trong /logs/ để debug
- Checkpoints giúp tiếp tục từ chỗ dừng

---

## 🔗 Tham khảo

- LAS-GNN Paper: LAs-GNN: A Graph Neural Network for Temporal Money Laundering Motif Detection
- Mamba Paper: Mamba: Linear-Time Sequence Modeling with Selective State Spaces
- Dataset: Elliptic2 Bitcoin Transaction Dataset
