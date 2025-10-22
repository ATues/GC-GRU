# GC-GRU Training and Validation Guide

This guide explains the proper training and validation workflow for the GC-GRU model.

## Overview

The project now follows a proper machine learning workflow with clear separation between:
- **Training**: Model training with train/validation/test splits
- **Validation**: Independent evaluation of trained models
- **Prediction**: Inference on new data

## File Structure

```
src/
├── main.py              # Main pipeline (simplified)
├── gc_gru.py            # GC-GRU model implementation
├── data_slicer.py       # Data slicing
├── feature_extraction.py # Feature extraction
├── ta_louvain.py        # Community detection
├── ag2vec.py            # Group embeddings
└── mut_inf.py           # Position quantification

scripts/
├── train_model.py       # Dedicated training script
└── validate_model.py    # Independent validation script

configs/
└── gc_gru_config.json   # Model hyperparameters
```

## Training Workflow

### 1. Prepare Data and Labels

First, ensure you have:
- Raw data in `data/raw/` directory
- Labels file (`.npy` or `.csv` format) with ground truth

Example labels file (`labels.csv`):
```csv
label
0
1
0
1
...
```

### 2. Train Model

Use the dedicated training script:

```bash
python scripts/train_model.py \
    --dataset weibo \
    --label_path data/processed/weibo/gc_gru/labels.csv \
    --model_path models/gc_gru_weibo.pth \
    --config configs/gc_gru_config.json
```

This will:
- Run the complete pipeline (slicing → features → communities → embeddings → positions)
- Split data into train/validation/test sets
- Train GC-GRU model with validation monitoring
- Save the best model based on validation loss
- Evaluate on test set
- Save training summary

### 3. Validate Model

After training, validate the model independently:

```bash
python scripts/validate_model.py \
    --model_path models/gc_gru_weibo.pth \
    --dataset weibo \
    --label_path data/processed/weibo/gc_gru/labels.csv
```

This provides:
- Detailed performance metrics
- Classification report
- Saved validation results

## Training Process Details

### Data Splits
- **Training set**: 60% of data (used for model training)
- **Validation set**: 20% of data (used for monitoring and early stopping)
- **Test set**: 20% of data (used for final evaluation only)

### Model Configuration
Key hyperparameters (configurable via `--config`):
- `hidden_dim`: 128 (GRU hidden size)
- `num_layers`: 2 (GRU layers)
- `dropout_rate`: 0.5 (dropout rate)
- `learning_rate`: 0.004
- `epochs`: 32 (training epochs)
- `sequence_length`: 10 (time window)

### Training Monitoring
- Early stopping based on validation loss
- Best model saved automatically
- Training/validation curves tracked

## Output Files

### Training Outputs
- `models/gc_gru_weibo.pth`: Trained model
- `training_summary_weibo_YYYYMMDD_HHMMSS.json`: Training summary

### Validation Outputs
- `validation_results_weibo.json`: Detailed validation results

## Quick Start Example

```bash
# 1. Train model
python scripts/train_model.py \
    --dataset weibo \
    --label_path data/processed/weibo/gc_gru/labels.csv \
    --model_path models/gc_gru_weibo.pth

# 2. Validate model
python scripts/validate_model.py \
    --model_path models/gc_gru_weibo.pth \
    --dataset weibo \
    --label_path data/processed/weibo/gc_gru/labels.csv

# 3. Use main.py for prediction on new data
python src/main.py \
    --dataset weibo \
    --mode predict \
    --model_path models/gc_gru_weibo.pth
```

## Key Improvements

1. **Proper Train/Val/Test Split**: Clear separation of data for training, validation, and testing
2. **Independent Validation**: Separate script for model evaluation
3. **Early Stopping**: Prevents overfitting using validation loss
4. **Model Persistence**: Automatic saving of best models
5. **Comprehensive Metrics**: Detailed performance evaluation
6. **Configurable Hyperparameters**: Easy parameter tuning via JSON config

## Troubleshooting

### Common Issues

1. **No sequences found**: Check that processed data exists in `data/processed/<dataset>/slicer/`
2. **Label mismatch**: Ensure labels file has same length as processed sequences
3. **Model not found**: Verify model path exists before validation
4. **Memory issues**: Reduce batch size or sequence length in config

### Data Requirements

- Raw data must have time column (`created_at`)
- Labels must be binary (0/1) for guided topic detection
- Sufficient data for train/val/test splits (recommend >100 samples)
