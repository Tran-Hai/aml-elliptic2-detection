# Project Structure Summary



---

## 📁 Cấu trúc Chi tiết

```
aml_project/
│
├── 📄 README.md                          # Tổng quan dự án
├── 📄 requirements.txt                   # Python dependencies
├── 📄 .gitignore                        # Git ignore rules
│
├── 📂 data/
│   ├── 📂 raw/                          
│   │   ├── nodes.csv (7.2 MB)
│   │   ├── edges.csv (11 MB)
│   │   ├── connected_components.csv (1.6 MB)
│   │   ├── background_nodes.csv (5.0 GB)
│   │   └── background_edges.csv (78 GB)
│   │
│   └── 📂 processed/                    # 🔄 Sẽ tạo qua 4 phases
│       ├── index/                       # Phase 1 output
│       ├── features/                    # Phase 2 output
│       ├── sequences/                   # Phase 3 output
│       └── graph/                       # Phase 4 output
│
├── 📂 src/
│   ├── 📂 data_processing/              
│   │   ├── config.py                    
│   │   ├── utils.py                     
│   │   ├── phase1_build_index.py        
│   │   ├── phase2_extract_features.py   
│   │   ├── phase3_build_sequences.py    
│   │   └── phase4_build_graph.py        
│   │
│   ├── 📂 models/                       
│   │   └── (sẽ tạo sau)
│   │
│   └── 📂 utils/                        
│       └── (sẽ tạo sau)
│
├── 📂 docs/                             
│   ├── project_report.docx
│   ├── las_gnn_paper.pdf
│   └── mamba_paper.pdf
│
├── 📂 scripts/                          # ✅ Utility scripts
│   ├── setup_env.sh                     # Setup environment
│   ├── run_processing.sh                # Run all phases
│   └── monitor_ram.sh                   # Monitor RAM
│
├── 📂 notebooks/                        # 📊 Cho EDA và visualization
│
├── 📂 tests/                            # 🧪 Unit tests
│
├── 📂 logs/                             # 📝 Execution logs
│
└── 📂 checkpoints/                      # 💾 Resume checkpoints
    ├── phase2_checkpoints/
    └── phase3_checkpoints/
```

---

## 📊 Disk Usage

### Hiện tại:
- Raw data: ~83 GB 
- Source code & docs: < 10 MB
- **Total hiện tại**: ~83 GB

### Sau processing (dự kiến):
- Raw data: ~83 GB
- Processed data: ~5-8 GB
- **Total sau processing**: ~90 GB

---


