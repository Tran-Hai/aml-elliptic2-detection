# Data Processing Workflow Guide



---

## 🎯 Tổng quan

Quy trình gồm **4 phases**, mỗi phase có thể chạy độc lập hoặc liên tiếp. Tất cả phases đều tối ưu cho 4GB RAM.

**Thứ tự thực hiện**: Phase 1 → Phase 2 → Phase 3 → Phase 4

---

## 📋 Phase 1: Build Index

**Thời gian**: 5-10 phút  
**RAM**: ~100 MB  
**Disk I/O**: Low

### Mục tiêu
Tạo các lookup tables để tra cứu nhanh, tránh đọc lại CSV lớn nhiều lần.

### Input
- `data/raw/nodes.csv` (7.2 MB)
- `data/raw/edges.csv` (11 MB)
- `data/raw/connected_components.csv` (1.6 MB)

### Output
```
data/processed/index/
├── node_to_idx.pkl          # Dict: clId → index
├── idx_to_node.pkl          # Dict: index → clId
├── node_labels.pkl          # Dict: clId → label (0/1)
├── component_to_nodes.pkl   # Dict: ccId → [clIds]
└── edges_index.pkl          # Adjacency list
```

### Chạy
```bash
python src/data_processing/phase1_build_index.py
```

### Kiểm tra
```bash
ls -lh data/processed/index/
```

---

## 📋 Phase 2: Extract Features

**ThờI gian**: 2-3 giờ  
**RAM**: ~200 MB  
**Disk I/O**: VERY HIGH (đọc 78GB, ghi 2-3GB)

### Mục tiêu
Trích xuất 95 features từ `background_edges.csv` (78GB) và lưu thành individual files.


### Solution
**Streaming với chunks**:
- Đọc 50,000 rows mỗi lần (~50-100 MB RAM)
- Filter chỉ giữ edges liên quan đến 444k nodes
- Extract và lưu ngay ra disk
- Xóa chunk khỏi RAM

### Input
- `data/raw/background_edges.csv` (78 GB)
- `data/processed/index/node_to_idx.pkl`

### Output
```
data/processed/features/
├── node_12345_in.csv       # Incoming transactions
├── node_12345_out.csv      # Outgoing transactions
├── ... (444k files)
```

**Format mỗi file**:
```csv
txId,feat#1,feat#2,...,feat#95,timestamp_proxy
50679415,40,68,...,51,0.456
589133991,53,68,...,59,0.789
```

### Chạy
```bash
# Lần đầu
python src/data_processing/phase2_extract_features.py

# Hoặc resume nếu bị gián đoạn
python src/data_processing/phase2_extract_features.py --resume
```

### Theo dõi tiến độ
```bash
# Terminal 1: Chạy processing
python src/data_processing/phase2_extract_features.py

# Terminal 2: Monitor RAM
./scripts/monitor_ram.sh

# Terminal 3: Xem số files đã tạo
watch -n 10 'ls data/processed/features/ | wc -l'
```

### Checkpointing
- Tự động lưu checkpoint mỗi 1000 chunks
- Checkpoint file: `checkpoints/phase2_checkpoint.pkl`
- Có thể resume từ checkpoint bằng flag `--resume`

---

## 📋 Phase 3: Build Sequences

**ThờI gian**: 30-60 phút  
**RAM**: ~100 MB  
**Disk I/O**: Medium

### Mục tiêu
Xây dựng temporal sequences (in-flow và out-flow) cho từng node.

### Input
- `data/processed/features/` (2-3 GB)
- `data/processed/index/`

### Process
1. Đọc `node_{clId}_in.csv` và `node_{clId}_out.csv`
2. Sort by txId (ascending) - thứ tự thờI gian
3. Giữ K=50 transactions gần nhất
4. Padding nếu thiếu
5. Lưu thành numpy array

### Output
```
data/processed/sequences/
├── node_000000.npz          # Shape: (2, 50, 97)
├── node_000001.npz          #   Dim 0: [in_flow, out_flow]
├── ...                      #   Dim 1: 50 transactions
└── metadata.json            #   Dim 2: 97 features
```

### Chạy
```bash
python src/data_processing/phase3_build_sequences.py
```

### Kiểm tra
```python
import numpy as np

# Load 1 sample
data = np.load('data/processed/sequences/node_000000.npz')
print(data['in_flow'].shape)   # (50, 97)
print(data['out_flow'].shape)  # (50, 97)
```

---

## 📋 Phase 4: Build Graph

**ThờI gian**: 5-10 phút  
**RAM**: ~100 MB  
**Disk I/O**: Low

### Mục tiêu
Tạo graph structure cho GNN training.

### Input
- `data/raw/edges.csv` (11 MB)
- `data/processed/index/`

### Output
```
data/processed/graph/
├── edge_index.pt            # torch.Tensor [2, num_edges]
├── edge_attr.pt             # torch.Tensor [num_edges, 3]
├── adjacency_list.pkl       # Dict: node_idx → [neighbors]
└── train_val_test_split.pkl # Indices for splits
```

### Train/Val/Test Split
- **Train**: 70% (stratified)
- **Val**: 15% (stratified)
- **Test**: 15% (stratified)
- Giữ nguyên tỷ lệ class imbalance (97.7:2.3)

### Chạy
```bash
python src/data_processing/phase4_build_graph.py
```

---

## 🚀 Cách Chạy

### Option 1: Chạy từng phase riêng lẻ

```bash
cd aml_project

# Phase 1
python src/data_processing/phase1_build_index.py

# Phase 2 (có thể pause/resume)
python src/data_processing/phase2_extract_features.py

# Nếu bị gián đoạn, resume
python src/data_processing/phase2_extract_features.py --resume

# Phase 3
python src/data_processing/phase3_build_sequences.py

# Phase 4
python src/data_processing/phase4_build_graph.py
```

### Option 2: Chạy tất cả bằng script

```bash
cd aml_project
./scripts/run_processing.sh
```

### Option 3: Monitor RAM trong lúc chạy

```bash
# Terminal 1: Start processing
python src/data_processing/phase2_extract_features.py

# Terminal 2: Monitor RAM
./scripts/monitor_ram.sh

# Logs will be saved to: logs/ram_usage.log
```


---

## 📊 Expected Results

Sau khi hoàn thành 4 phases:

```
data/processed/
├── index/          ~10 MB
├── features/       ~2-3 GB (444k files)
├── sequences/      ~3-4 GB (444k files)
└── graph/          ~100 MB
```

**Total**: ~5-8 GB processed data

---



## ✅ Verification Checklist

Sau mỗi phase, kiểm tra:

- [ ] Phase 1: `ls data/processed/index/` có 5 files
- [ ] Phase 2: `ls data/processed/features/ | wc -l` ≈ 888k (444k × 2)
- [ ] Phase 3: `ls data/processed/sequences/ | wc -l` ≈ 444k
- [ ] Phase 4: `ls data/processed/graph/` có 4 files

---



