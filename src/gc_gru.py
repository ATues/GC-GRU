"""
GC-GRU model: build sequences from processed slices, train and predict, and persist results per slice.
"""

import numpy as np
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. GC-GRU model will not work.")
from typing import Dict, List, Tuple, Optional, Union
from dataclasses import dataclass
from collections import defaultdict
import math
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import pickle
import os
import json
import pandas as pd

 

# Processed slice IO helpers
def list_slice_directories(dataset_name: str,
                           processed_root: str = os.path.join('data', 'processed')) -> List[str]:
    base_dir = os.path.join(processed_root, dataset_name, 'slicer')
    if not os.path.isdir(base_dir):
        return []
    def sort_key(x: str):
        return (len(x), x)
    return [os.path.join(base_dir, d) for d in sorted(os.listdir(base_dir), key=sort_key)
            if os.path.isdir(os.path.join(base_dir, d))]


def load_features_and_communities(slice_dir: str) -> Tuple[Optional[Dict], Optional[Dict[int, List[str]]]]:
    features = None
    communities = None
    feat_path = os.path.join(slice_dir, 'features.json')
    if os.path.isfile(feat_path):
        try:
            with open(feat_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
                features = obj.get('features', None)
        except Exception:
            features = None
    com_path = os.path.join(slice_dir, 'communities.json')
    if os.path.isfile(com_path):
        try:
            with open(com_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                communities = {int(k): v for k, v in raw.items()}
        except Exception:
            communities = None
    return features, communities


def load_position_matrix(slice_dir: str) -> Optional[np.ndarray]:
    npy_path = os.path.join(slice_dir, 'position_matrix.npy')
    if os.path.isfile(npy_path):
        try:
            return np.load(npy_path)
        except Exception:
            return None
    # fallback: build from positions.json if present
    pos_json = os.path.join(slice_dir, 'positions.json')
    if os.path.isfile(pos_json):
        try:
            with open(pos_json, 'r', encoding='utf-8') as f:
                pos = json.load(f)
            rows = []
            for _, pdata in pos.items():
                mi = pdata.get('mutual_influences', {})
                rows.append([float(mi.get('support', 0.0)), float(mi.get('oppose', 0.0)), float(mi.get('neutral', 0.0))])
            if rows:
                return np.array(rows, dtype=float)
        except Exception:
            return None
    return None


def build_group_features_summary(features: Dict, communities: Dict[int, List[str]]) -> Dict:
    return {
        'communities': communities or {},
        'avg_influence': float(np.mean(list((features or {}).get('user_influences', {}).values())) if features and features.get('user_influences') else 0.0),
        'avg_awareness': float(np.mean(list((features or {}).get('user_topic_awareness', {}).values())) if features and features.get('user_topic_awareness') else 0.0),
        'topic_heat': float((features or {}).get('topic_heat', 0.0))
    }


def dataset_cache_dir(dataset_name: str, processed_root: str) -> str:
    d = os.path.join(processed_root, dataset_name, 'gc_gru')
    os.makedirs(d, exist_ok=True)
    return d

@dataclass
class GCGRUConfig:
    """Model and training hyperparameters."""
    # model
    input_dim: int = 131  # 128 (group features) + 3 (position summary)
    hidden_dim: int = 128
    num_layers: int = 2
    dropout_rate: float = 0.5
    fc_dim: int = 256
    # training
    learning_rate: float = 0.004
    batch_size: int = 32
    epochs: int = 32
    early_stopping_patience: int = 10
    # data
    sequence_length: int = 10
    test_size: float = 0.2
    validation_size: float = 0.2
    random_state: int = 42
    # optimization
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0

if TORCH_AVAILABLE:
    class TopicDataset(Dataset):
        """话题检测数据集"""
        
        def __init__(self, 
                     sequences: List[np.ndarray], 
                     labels: List[int],
                     transform=None):
            """
            初始化数据集
            
            Args:
                sequences: 时间序列数据
                labels: 标签数据
                transform: 数据变换（可选）
            """
            self.sequences = sequences
            self.labels = labels
            self.transform = transform
            
            # 确保数据长度一致
            assert len(sequences) == len(labels), "Sequences and labels must have same length"
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            sequence = self.sequences[idx]
            label = self.labels[idx]
            
            if self.transform:
                sequence = self.transform(sequence)
            
            return torch.FloatTensor(sequence), torch.LongTensor([label])
else:
    class TopicDataset:
        """话题检测数据集（PyTorch不可用时的占位符）"""
        def __init__(self, sequences, labels, transform=None):
            self.sequences = sequences
            self.labels = labels
            self.transform = transform
        
        def __len__(self):
            return len(self.sequences)
        
        def __getitem__(self, idx):
            return self.sequences[idx], self.labels[idx]

if TORCH_AVAILABLE:
    class GCGRUModel(nn.Module):
        """GRU + attention + classifier."""
        
        def __init__(self, config: GCGRUConfig):
            """
            初始化GC-GRU模型
            
            Args:
                config: 模型配置
            """
            super(GCGRUModel, self).__init__()
            self.config = config
            
            # GRU encoder
            self.gru = nn.GRU(
                input_size=config.input_dim,
                hidden_size=config.hidden_dim,
                num_layers=config.num_layers,
                dropout=config.dropout_rate if config.num_layers > 1 else 0,
                batch_first=True,
                bidirectional=False
            )
            
            # Self-attention over time
            self.attention = nn.MultiheadAttention(
                embed_dim=config.hidden_dim,
                num_heads=4,
                dropout=config.dropout_rate,
                batch_first=True
            )
            
            # Classifier
            self.classifier = nn.Sequential(
                nn.Linear(config.hidden_dim, config.fc_dim),
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.fc_dim, 2),
                nn.Softmax(dim=1)
            )
            
            # Init
            self._init_weights()
else:
    class GCGRUModel:
        """Placeholder when PyTorch is unavailable."""
        def __init__(self, config: GCGRUConfig):
            self.config = config
    
    def _init_weights(self):
        """Initialize weights."""
        for name, param in self.named_parameters():
            if 'weight' in name:
                if len(param.shape) >= 2:
                    nn.init.xavier_uniform_(param)
                else:
                    nn.init.normal_(param, 0, 0.01)
            elif 'bias' in name:
                nn.init.constant_(param, 0)
    
    def forward(self, x):
        """Forward pass: [B, T, D] -> probs [B, 2]."""
        batch_size, seq_len, _ = x.size()
        
        gru_out, hidden = self.gru(x)
        
        attn_out, _ = self.attention(gru_out, gru_out, gru_out)
        
        pooled = torch.mean(attn_out, dim=1)
        
        output = self.classifier(pooled)
        
        return output

class GCGRUTrainer:
    """Trainer wrapper."""
    
    def __init__(self, config: GCGRUConfig):
        self.config = config
        self.model = None
        self.optimizer = None
        self.criterion = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
    def setup_model(self):
        self.model = GCGRUModel(self.config).to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        self.criterion = nn.CrossEntropyLoss()
    
    def train_epoch(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (data, target) in enumerate(dataloader):
            data, target = data.to(self.device), target.to(self.device)
            target = target.squeeze()
            
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.gradient_clip_norm
            )
            
            self.optimizer.step()
            
            # 统计
            total_loss += loss.item()
            pred = output.argmax(dim=1)
            correct += pred.eq(target).sum().item()
            total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def validate(self, dataloader: DataLoader) -> Tuple[float, float]:
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                output = self.model(data)
                loss = self.criterion(output, target)
                
                total_loss += loss.item()
                pred = output.argmax(dim=1)
                correct += pred.eq(target).sum().item()
                total += target.size(0)
        
        avg_loss = total_loss / len(dataloader)
        accuracy = 100.0 * correct / total
        
        return avg_loss, accuracy
    
    def train(self, 
              train_loader: DataLoader, 
              val_loader: DataLoader,
              save_path: str = None) -> Dict:
        if self.model is None:
            self.setup_model()
        
        best_val_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(self.config.epochs):
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate(val_loader)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                if save_path:
                    self.save_model(save_path)
            else:
                patience_counter += 1
            if patience_counter >= self.config.early_stopping_patience:
                break
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss
        }
    
    def evaluate(self, dataloader: DataLoader) -> Dict:
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        all_predictions = []
        all_targets = []
        all_probabilities = []
        
        with torch.no_grad():
            for data, target in dataloader:
                data, target = data.to(self.device), target.to(self.device)
                target = target.squeeze()
                
                output = self.model(data)
                probabilities = output.cpu().numpy()
                predictions = output.argmax(dim=1).cpu().numpy()
                
                all_predictions.extend(predictions)
                all_targets.extend(target.cpu().numpy())
                all_probabilities.extend(probabilities)
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, average='weighted')
        recall = recall_score(all_targets, all_predictions, average='weighted')
        f1 = f1_score(all_targets, all_predictions, average='weighted')
        
        try:
            auc = roc_auc_score(all_targets, [prob[1] for prob in all_probabilities])
        except ValueError:
            auc = 0.0
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'auc': auc,
            'predictions': all_predictions,
            'targets': all_targets,
            'probabilities': all_probabilities
        }
    
    def predict(self, sequences: List[np.ndarray]) -> Tuple[List[int], List[float]]:
        if self.model is None:
            raise ValueError("Model not initialized")
        
        self.model.eval()
        predictions = []
        probabilities = []
        
        with torch.no_grad():
            for sequence in sequences:
                data = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                output = self.model(data)
                prob = output.cpu().numpy()[0]
                pred = np.argmax(prob)
                
                predictions.append(pred)
                probabilities.append(prob[1])
        
        return predictions, probabilities
    
    def save_model(self, filepath: str):
        if self.model is None:
            raise ValueError("Model not initialized")
        
        save_dict = {
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies
        }
        
        torch.save(save_dict, filepath)
        
    
    def load_model(self, filepath: str):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Model file not found: {filepath}")
        
        checkpoint = torch.load(filepath, map_location=self.device)
        
        self.config = checkpoint['config']
        
        self.setup_model()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        self.train_accuracies = checkpoint.get('train_accuracies', [])
        self.val_accuracies = checkpoint.get('val_accuracies', [])

class GCGRUPredictor:
    """High-level predictor API."""
    
    def __init__(self, config: GCGRUConfig = None):
        self.config = config or GCGRUConfig()
        self.trainer = GCGRUTrainer(self.config)
        self.is_trained = False
    
    def prepare_data(self, 
                    group_features_sequence: List[Dict],
                    position_matrices_sequence: List[np.ndarray],
                    labels: List[int]) -> Tuple[List[np.ndarray], List[int]]:
        """Align sequences and labels into fixed-length windows."""
        sequences = []
        processed_labels = []
        
        for i in range(len(group_features_sequence)):
            if i < self.config.sequence_length - 1:
                continue
            
            # build one window
            sequence = []
            for j in range(i - self.config.sequence_length + 1, i + 1):
                if j < len(group_features_sequence) and j < len(position_matrices_sequence):
                    # concat group features with position summary
                    group_features = group_features_sequence[j]
                    position_matrix = position_matrices_sequence[j]
                    
                    # vectorize group features
                    feature_vector = self._group_features_to_vector(group_features)
                    
                    # summarize positions to 3 dims (avg across groups)
                    if position_matrix.size == 0:
                        position_vector = np.array([0.0, 0.0, 0.0])
                    else:
                        if len(position_matrix.shape) == 2 and position_matrix.shape[1] == 3:
                            position_vector = position_matrix.mean(axis=0)
                        elif len(position_matrix.shape) == 1 and position_matrix.shape[0] == 3:
                            position_vector = position_matrix
                        else:
                            # fallback pad
                            flat = position_matrix.flatten()
                            pad = np.zeros(3)
                            pad[:min(3, flat.shape[0])] = flat[:min(3, flat.shape[0])]
                            position_vector = pad
                    
                    # concat
                    combined_vector = np.concatenate([feature_vector, position_vector])
                    sequence.append(combined_vector)
            
            if len(sequence) == self.config.sequence_length:
                sequences.append(np.array(sequence))
                processed_labels.append(labels[i])
        
        return sequences, processed_labels

    # ----------------------------
    # 从已处理数据集构建序列并可选持久化
    # ----------------------------
    def build_sequences_from_dataset(self, dataset_name: str = 'weibo', processed_root: str = os.path.join('data', 'processed')) -> Tuple[List[Dict], List[np.ndarray]]:
        group_features_seq: List[Dict] = []
        position_mats_seq: List[np.ndarray] = []
        for slice_dir in list_slice_directories(dataset_name, processed_root):
            features, communities = load_features_and_communities(slice_dir)
            if features is None or communities is None:
                continue
            pos_mat = load_position_matrix(slice_dir)
            if pos_mat is None:
                continue
            gf = build_group_features_summary(features, communities)
            group_features_seq.append(gf)
            position_mats_seq.append(pos_mat)
        return group_features_seq, position_mats_seq

    def save_prepared_sequences(self, dataset_name: str, processed_root: str,
                                sequences: List[np.ndarray], labels: List[int] = None) -> Dict[str, str]:
        out_dir = dataset_cache_dir(dataset_name, processed_root)
        seq_path = os.path.join(out_dir, 'sequences.npz')
        np.savez_compressed(seq_path, sequences=np.array(sequences, dtype=object))
        paths = {'sequences': seq_path}
        if labels is not None:
            lbl_path = os.path.join(out_dir, 'labels.npy')
            np.save(lbl_path, np.array(labels))
            paths['labels'] = lbl_path
        return paths
    
    def _group_features_to_vector(self, group_features: Dict) -> np.ndarray:
        """将群体特征转换为向量"""
        # 提取数值特征
        features = []
        
        # 群体数量
        features.append(len(group_features.get('communities', {})))
        
        # 平均群体大小
        communities = group_features.get('communities', {})
        if communities:
            avg_size = np.mean([len(members) for members in communities.values()])
            features.append(avg_size)
        else:
            features.append(0.0)
        
        # 平均影响力
        avg_influence = group_features.get('avg_influence', 0.0)
        features.append(avg_influence)
        
        # 平均话题意识度
        avg_awareness = group_features.get('avg_awareness', 0.0)
        features.append(avg_awareness)
        
        # 话题热度
        topic_heat = group_features.get('topic_heat', 0.0)
        features.append(topic_heat)
        
        # 填充到固定维度
        target_dim = 128  # 与AG2vec嵌入维度一致
        while len(features) < target_dim:
            features.append(0.0)
        
        return np.array(features[:target_dim])
    
    def train(self, 
              group_features_sequence: List[Dict],
              position_matrices_sequence: List[np.ndarray],
              labels: List[int],
              save_path: str = None) -> Dict:
        """
        训练模型
        
        Args:
            group_features_sequence: 群体特征序列
            position_matrices_sequence: 立场矩阵序列
            labels: 标签序列
            save_path: 模型保存路径
            
        Returns:
            训练结果
        """
        
        
        # 准备数据
        sequences, processed_labels = self.prepare_data(
            group_features_sequence, position_matrices_sequence, labels
        )
        
        if len(sequences) == 0:
            raise ValueError("No valid sequences found")
        
        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(
            sequences, processed_labels, 
            test_size=self.config.test_size, 
            random_state=self.config.random_state,
            stratify=processed_labels
        )
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train,
            test_size=self.config.validation_size,
            random_state=self.config.random_state,
            stratify=y_train
        )
        
        # 创建数据加载器
        train_dataset = TopicDataset(X_train, y_train)
        val_dataset = TopicDataset(X_val, y_val)
        test_dataset = TopicDataset(X_test, y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=True
        )
        val_loader = DataLoader(
            val_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=self.config.batch_size, 
            shuffle=False
        )
        
        
        
        # Train model with validation monitoring
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")
        print(f"Test samples: {len(X_test)}")
        
        train_results = self.trainer.train(train_loader, val_loader, save_path)
        
        # Evaluate on test set
        test_results = self.trainer.evaluate(test_loader)
        
        self.is_trained = True
        
        return {
            'train_results': train_results,
            'test_results': test_results
        }
    
    def predict(self, 
                group_features_sequence: List[Dict],
                position_matrices_sequence: List[np.ndarray]) -> Tuple[List[int], List[float]]:
        """
        预测
        
        Args:
            group_features_sequence: 群体特征序列
            position_matrices_sequence: 立场矩阵序列
            
        Returns:
            预测标签和概率
        """
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        # 准备数据
        sequences, _ = self.prepare_data(
            group_features_sequence, position_matrices_sequence, [0] * len(group_features_sequence)
        )
        
        if len(sequences) == 0:
            return [], []
        
        # 预测
        predictions, probabilities = self.trainer.predict(sequences)
        
        return predictions, probabilities
    
    def load_model(self, filepath: str):
        """加载模型"""
        self.trainer.load_model(filepath)
        self.is_trained = True
        


def main():
    """GC-GRU数据管线入口：基于已保存产物构建序列、训练或预测。"""
    import argparse
    parser = argparse.ArgumentParser(description='GC-GRU pipeline over processed slices.')
    parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'politifact', 'gossipcop'])
    parser.add_argument('--processed_root', type=str, default=os.path.join('data', 'processed'))
    parser.add_argument('--mode', type=str, default='prepare', choices=['prepare', 'train', 'predict'])
    parser.add_argument('--model_path', type=str, default=os.path.join('models', 'gc_gru_model.pth'))
    args = parser.parse_args()

    os.makedirs('models', exist_ok=True)

    config = GCGRUConfig()
    predictor = GCGRUPredictor(config)

    # 构建序列
    gf_seq, pos_seq = predictor.build_sequences_from_dataset(args.dataset, args.processed_root)
    if len(gf_seq) == 0:
        print('No valid slices found to build sequences.')
        return

    if args.mode == 'prepare':
        # 准备并持久化序列（无标签）
        seqs, _ = predictor.prepare_data(gf_seq, pos_seq, [0] * len(gf_seq))
        paths = predictor.save_prepared_sequences(args.dataset, args.processed_root, seqs)
        print(f"[{args.dataset}] prepared {len(seqs)} sequences -> {paths['sequences']}")
        return

    if args.mode == 'train':
        # 寻找标签文件 data/processed/<dataset>/gc_gru/labels.npy 或 labels.csv
        cache_dir = dataset_cache_dir(args.dataset, args.processed_root)
        labels_path_npy = os.path.join(cache_dir, 'labels.npy')
        labels = None
        if os.path.isfile(labels_path_npy):
            labels = list(np.load(labels_path_npy).tolist())
        else:
            labels_csv = os.path.join(cache_dir, 'labels.csv')
            if os.path.isfile(labels_csv):
                try:
                    df_lbl = pd.read_csv(labels_csv)
                    labels = df_lbl['label'].astype(int).tolist()
                except Exception:
                    labels = None
        if labels is None:
            raise ValueError('Training labels not found. Provide labels.npy or labels.csv in dataset cache directory.')
        results = predictor.train(gf_seq, pos_seq, labels, save_path=args.model_path)
        print(f"Trained model saved to {args.model_path}")
        print(f"Test Accuracy: {results['test_results']['accuracy']:.4f}")
        print(f"Test F1 Score: {results['test_results']['f1_score']:.4f}")
        return

    if args.mode == 'predict':
        predictor.trainer.load_model(args.model_path)
        predictor.is_trained = True
        preds, probs = predictor.predict(gf_seq, pos_seq)
        # 将预测保存回各切片目录
        slice_dirs = list_slice_directories(args.dataset, args.processed_root)
        # 对齐序列数量到 slice 数量可不一致（序列长度需求），只保存有预测的后缀切片
        offset = len(slice_dirs) - len(preds)
        for i, (pred, prob) in enumerate(zip(preds, probs)):
            idx = offset + i if offset >= 0 else i
            if 0 <= idx < len(slice_dirs):
                out_path = os.path.join(slice_dirs[idx], 'gc_gru_pred.json')
                try:
                    with open(out_path, 'w', encoding='utf-8') as f:
                        json.dump({'prediction': int(pred), 'probability': float(prob)}, f, ensure_ascii=False, indent=2)
                except Exception:
                    pass
        print(f"[{args.dataset}] predictions saved for {len(preds)} slices.")


if __name__ == "__main__":
    main()
