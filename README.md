# Guided Topic Detection System

A guided topic detection system based on group feature evolution, implementing a complete machine learning pipeline for identifying guided topics in social media.



##  Project Structure

```
├── src/                          # Core source code
│   ├── main.py                   # Main pipeline entry point
│   ├── data_slicer.py           # Data slicing module
│   ├── feature_extraction.py    # Feature extraction module
│   ├── ta_louvain.py           # TA-Louvain group partitioning algorithm
│   ├── ag2vec.py               # AG2vec group feature representation
│   ├── mut_inf.py              # Group position mutual influence quantification
│   └── gc_gru.py               # GC-GRU topic detection model
├── scripts/                     # Execution scripts
│   ├── run_pipeline.py         # Main execution script
│   └── train.py                # Training script
├── configs/                     # Configuration files
│   └── pipeline_config.json    # Main configuration file
├── data/                        # Data directory
│   ├── raw/                    # Raw data
│   └── processed/              # Processed data
├── tests/                       # Test files
├── requirements.txt             # Dependencies list
└── setup.py                    # Installation configuration
```


### Requirements

- Python >= 3.8
- PyTorch >= 1.9.0 (optional, for deep learning models)

### Install Dependencies

```bash
# Clone the project
git https://github.com/ATues/GC-GRU.git

# Install dependencies
pip install -r requirements.txt

```



### 1. Train

```bash
python src/main.py \
    --dataset weibo \
    --mode train \
    --label_path data/processed/weibo/gc_gru/labels.csv \
    --model_path models/gc_gru_model.pth
```

### 2. Test

```bash
python src/main.py \
    --dataset weibo \
    --mode predict \
    --model_path models/gc_gru_model.pth
```

### 3. Programming Interface


##  Configuration Parameters

Main configuration parameters (`configs/pipeline_config.json`):

### Data Slicing Parameters
- `time_window_hours`: Time window size (hours)
- `overlap_ratio`: Slice overlap ratio
- `min_users_per_slice`: Minimum users per slice

### Feature Extraction Parameters
- `time_decay_factor`: Time decay factor
- `beta`: Short-term dynamic performance weight
- `alpha1`, `alpha2`: Topic awareness weight parameters

### Model Parameters
- `hidden_dim`: Hidden layer dimension
- `num_layers`: Number of GRU layers
- `sequence_length`: Time series length
- `epochs`: Number of training epochs


## Dataset

The datasets used in this paper are **not included** in this repository due to licensing restrictions.  
However, they are **publicly available** and can be accessed from the following sources:

| Dataset Name | Description | Access Link                                                                      |
|---------------|-------------|----------------------------------------------------------------------------------|
| **Weibo** | Chinese social media dataset for rumor detection | [https://github.com/thunlp/CED](https://github.com/thunlp/CED)   |                                             |
| **FakeNewsNet** | Multidomain dataset (Politifact & Gossipcop) | [https://github.com/KaiDMML/FakeNewsNet](https://github.com/KaiDMML/FakeNewsNet) |



