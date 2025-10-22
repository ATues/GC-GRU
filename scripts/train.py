#!/usr/bin/env python3
"""
Training script for GC-GCN
"""

import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from gc_gcn import GCGCN, GraphDataset, GraphDataProcessor, Trainer


def load_config(config_path: str) -> dict:
    """Load configuration from YAML file"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def set_seed(seed: int):
    """Set random seed for reproducibility"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device(device_config: str) -> torch.device:
    """Get device based on configuration"""
    if device_config == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        return torch.device(device_config)


def main():
    parser = argparse.ArgumentParser(description="Train GC-GCN model")
    parser.add_argument("--config", type=str, default="configs/default.yaml",
                       help="Path to configuration file")
    parser.add_argument("--data", type=str, default=None,
                       help="Path to dataset (overrides config)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output directory (overrides config)")
    parser.add_argument("--epochs", type=int, default=None,
                       help="Number of epochs (overrides config)")
    parser.add_argument("--lr", type=float, default=None,
                       help="Learning rate (overrides config)")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.data:
        config['data']['data_path'] = args.data
    if args.output:
        config['experiment']['save_dir'] = args.output
    if args.epochs:
        config['training']['num_epochs'] = args.epochs
    if args.lr:
        config['training']['learning_rate'] = args.lr
    
    # Set random seed
    set_seed(config['experiment']['seed'])
    
    # Get device
    device = get_device(config['experiment']['device'])
    print(f"Using device: {device}")
    
    # Load data
    print("Loading data...")
    # TODO: Implement actual data loading based on your dataset
    # For now, this is a placeholder
    print("Data loading not implemented yet. Please implement load_graph_data function.")
    return
    
    # Create model
    model_config = config['model']
    model = GCGCN(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        output_dim=model_config['output_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )
    
    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    
    # Create data loaders
    # TODO: Implement data loading and preprocessing
    train_loader = None
    val_loader = None
    test_loader = None
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        log_dir=config['experiment']['log_dir'],
        save_dir=config['experiment']['save_dir']
    )
    
    # Train model
    training_config = config['training']
    history = trainer.train(
        num_epochs=training_config['num_epochs'],
        learning_rate=training_config['learning_rate'],
        weight_decay=training_config['weight_decay'],
        patience=training_config['patience'],
        save_best=training_config['save_best']
    )
    
    print("Training completed!")
    print(f"Best validation accuracy: {trainer.best_val_acc:.2f}%")
    print(f"Best validation loss: {trainer.best_val_loss:.4f}")
    
    # Evaluate on test set if available
    if test_loader is not None:
        print("Evaluating on test set...")
        test_results = trainer.evaluate()
        print(f"Test accuracy: {test_results['test_accuracy']:.2f}%")
        print(f"Test loss: {test_results['test_loss']:.4f}")


if __name__ == "__main__":
    main()



