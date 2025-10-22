"""
Main pipeline entry for dataset-configurable guided topic detection (processed-data workflow).
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List

from data_slicer import DataSlicer
from feature_extraction import FeatureExtractor, FeatureConfig
from ta_louvain import TALouvain
from ag2vec import AG2vec
from mut_inf import PositionQuantifier, PositionConfig
from gc_gru import GCGRUPredictor

def output_metrics(preds: List[int], labels: List[int], split_name: str = ""):
    """Compute and print core metrics: acc/precision/recall/F1."""
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    acc = accuracy_score(labels, preds)
    pre = precision_score(labels, preds, average='weighted', zero_division=0)
    rec = recall_score(labels, preds, average='weighted', zero_division=0)
    f1 = f1_score(labels, preds, average='weighted', zero_division=0)
    
    prefix = f"{split_name} " if split_name else ""
    print(f"{prefix}Accuracy: {acc:.4f}")
    print(f"{prefix}Precision: {pre:.4f}")
    print(f"{prefix}Recall: {rec:.4f}")
    print(f"{prefix}F1: {f1:.4f}")
    return {"accuracy": acc, "precision": pre, "recall": rec, "f1": f1}
def main():
    parser = argparse.ArgumentParser(description='End-to-end: slicing, features, communities, embeddings, positions, GC-GRU')
    parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'politifact', 'gossipcop'])
    parser.add_argument('--config', type=str, default=None, help='Optional JSON config file')
    parser.add_argument('--mode', type=str, choices=['train', 'predict'], default='predict')
    parser.add_argument('--label_path', type=str, default=None, help='Label file (npy/csv) for training')
    parser.add_argument('--model_path', type=str, default='models/gc_gru_model.pth', help='Model path for save/load')
    parser.add_argument('--processed_root', type=str, default='data/processed')
    parser.add_argument('--params', type=str, default=None, help='Inline JSON to override config')
    args = parser.parse_args()
    
    # Load optional config overrides
    config = {}
    if args.config and os.path.exists(args.config):
        with open(args.config, 'r', encoding='utf-8') as f:
            config = json.load(f)
    if args.params:
        config.update(json.loads(args.params))

    # Slicing (idempotent if already processed)
    ds = DataSlicer(input_dir=os.path.join('data', 'raw'), output_dir=args.processed_root)
    # ds.process(...)  # optional first-time slicing

    # Feature extraction per slice
    feat_extractor = FeatureExtractor()
    feat_extractor.process_dataset_slices(args.dataset, args.processed_root)

    # Community detection
    TALouvain().process_dataset(args.dataset, args.processed_root)

    # Embeddings and position quantification
    AG2vec().process_dataset(args.dataset, args.processed_root)
    PositionQuantifier().process_dataset(args.dataset, args.processed_root)

    # GC-GRU: prepare/train/predict
    predictor = GCGRUPredictor()
    cache_dir = os.path.join(args.processed_root, args.dataset, 'gc_gru')
    
    if args.mode == 'train':
        print("=== Training Mode ===")
        # Load labels from npy or csv (with column 'label')
        lbl_path_npy = args.label_path or os.path.join(cache_dir, 'labels.npy')
        labels = None
        if os.path.isfile(lbl_path_npy):
            labels = list(np.load(lbl_path_npy).tolist())
        else:
            lbl_csv = args.label_path or os.path.join(cache_dir, 'labels.csv')
            if os.path.isfile(lbl_csv):
                try:
                    df_lbl = pd.read_csv(lbl_csv)
                    labels = df_lbl['label'].astype(int).tolist()
                except Exception:
                    labels = None
        if not labels:
            raise RuntimeError('Training labels (labels.npy or labels.csv) not found.')
        
        print(f"Loaded {len(labels)} labels for training")
        
        # Build sequences from processed data
        gf_seq, pos_seq = predictor.build_sequences_from_dataset(args.dataset, args.processed_root)
        print(f"Built {len(gf_seq)} group feature sequences")
        
        # Train model with proper train/val/test split
        print("Starting training with train/validation/test split...")
        result = predictor.train(gf_seq, pos_seq, labels, save_path=args.model_path)
        
        # Output training results
        print("\n=== Training Results ===")
        train_results = result['train_results']
        test_results = result['test_results']
        
        print(f"Best validation loss: {train_results['best_val_loss']:.4f}")
        print(f"Final training accuracy: {train_results['train_accuracies'][-1]:.2f}%")
        print(f"Final validation accuracy: {train_results['val_accuracies'][-1]:.2f}%")
        
        print("\n=== Test Set Evaluation ===")
        test_preds = test_results['predictions']
        test_labels = test_results['targets']
        test_metrics = output_metrics(test_preds, test_labels, "Test")
        
        print(f"\nModel saved to: {args.model_path}")
        
    elif args.mode == 'predict':
        print("=== Prediction Mode ===")
        # Load trained model
        if not os.path.exists(args.model_path):
            raise FileNotFoundError(f"Model file not found: {args.model_path}")
        
        predictor.trainer.load_model(args.model_path)
        predictor.is_trained = True
        print(f"Loaded model from: {args.model_path}")
        
        # Build sequences and predict
        gf_seq, pos_seq = predictor.build_sequences_from_dataset(args.dataset, args.processed_root)
        print(f"Built {len(gf_seq)} group feature sequences")
        
        preds, probs = predictor.predict(gf_seq, pos_seq)
        print(f"Generated {len(preds)} predictions")
        
        # Try to load labels for evaluation if available
        labels = None
        lbl_path_npy = args.label_path or os.path.join(cache_dir, 'labels.npy')
        if os.path.isfile(lbl_path_npy):
            labels = list(np.load(lbl_path_npy).tolist())
        else:
            lbl_csv = args.label_path or os.path.join(cache_dir, 'labels.csv')
            if os.path.isfile(lbl_csv):
                try:
                    df_lbl = pd.read_csv(lbl_csv)
                    labels = df_lbl['label'].astype(int).tolist()
                except Exception:
                    labels = None
        
        if labels and len(preds) <= len(labels):
            # Align predictions with corresponding labels
            aligned_labels = labels[-len(preds):]
            print("\n=== Prediction Evaluation ===")
            output_metrics(preds, aligned_labels, "Prediction")
        else:
            print("\n=== Predictions (No Ground Truth) ===")
            print(f"Predictions: {preds}")
            print(f"Probabilities: {[f'{p:.3f}' for p in probs]}")

if __name__ == "__main__":
    main()
