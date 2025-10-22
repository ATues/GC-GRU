"""
AG2vec: group representation learning based on node2vec and attention.
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Optional, Set
import logging
from dataclasses import dataclass
from collections import defaultdict
import random
import math
from sklearn.decomposition import PCA
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features may not work.")

logger = logging.getLogger(__name__)

# Processed slice discovery
def list_slice_directories(dataset_name: str,
                           processed_root: str = os.path.join('data', 'processed')) -> List[str]:
    base_dir = os.path.join(processed_root, dataset_name, 'slicer')
    if not os.path.isdir(base_dir):
        return []
    # sort by numeric id if possible
    def sort_key(x: str):
        return (len(x), x)
    return [os.path.join(base_dir, d) for d in sorted(os.listdir(base_dir), key=sort_key)
            if os.path.isdir(os.path.join(base_dir, d))]


def load_slice_df_and_meta(slice_dir: str) -> Tuple[Optional[pd.DataFrame], Optional[Dict]]:
    data_csv = os.path.join(slice_dir, 'data.csv')
    meta_json = os.path.join(slice_dir, 'meta.json')
    if not (os.path.isfile(data_csv) and os.path.isfile(meta_json)):
        return None, None
    try:
        df = pd.read_csv(data_csv)
        with open(meta_json, 'r', encoding='utf-8') as f:
            meta = json.load(f)
        return df, meta
    except Exception:
        return None, None


def build_network_from_df(df: pd.DataFrame,
                          user_col: str = 'user_id',
                          target_col: str = 'target_user_id') -> nx.Graph:
    G = nx.Graph()
    if user_col in df.columns:
        for uid in df[user_col].dropna().astype(str).unique().tolist():
            G.add_node(uid)
    if target_col in df.columns and user_col in df.columns:
        for _, row in df.dropna(subset=[user_col, target_col]).iterrows():
            u = str(row[user_col])
            v = str(row[target_col])
            if u and v:
                G.add_edge(u, v)
    return G


def save_network(slice_dir: str, graph: nx.Graph) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    # Save as edge list CSV
    edges_path = os.path.join(slice_dir, 'network.edgelist.csv')
    try:
        import csv
        with open(edges_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['source', 'target'])
            for u, v in graph.edges():
                writer.writerow([u, v])
        paths['edgelist'] = edges_path
    except Exception:
        pass

    # Save as pickle for direct Python loading
    try:
        import pickle
        pkl_path = os.path.join(slice_dir, 'network.pkl')
        with open(pkl_path, 'wb') as f:
            pickle.dump(graph, f)
        paths['pickle'] = pkl_path
    except Exception:
        pass

    return paths


def build_slice_view_from_df(df: pd.DataFrame, meta: Dict,
                             time_col: str = 'created_at', user_col: str = 'user_id',
                             target_col: str = 'target_user_id', type_col: str = 'interaction_type'):
    """Build minimal slice view for feature extraction."""
    class SliceView:
        pass

    sv = SliceView()
    end_time_iso = meta.get('end_time')
    try:
        sv.timestamp = pd.to_datetime(end_time_iso).to_pydatetime() if end_time_iso else pd.Timestamp.utcnow().to_pydatetime()
    except Exception:
        sv.timestamp = pd.Timestamp.utcnow().to_pydatetime()

    if user_col in df.columns:
        sv.users = list({str(u) for u in df[user_col].dropna().astype(str).tolist()})
    else:
        sv.users = []

    comments: List[Dict] = []
    if type_col in df.columns and time_col in df.columns and user_col in df.columns:
        mask = df[type_col].astype(str).str.lower() == 'comment'
        for _, row in df[mask].iterrows():
            try:
                comments.append({
                    'user_id': str(row[user_col]),
                    'timestamp': pd.to_datetime(row[time_col], errors='coerce').to_pydatetime().isoformat()
                })
            except Exception:
                continue
    sv.comments = comments

    interactions: List[Dict] = []
    if time_col in df.columns and user_col in df.columns:
        for _, row in df.iterrows():
            try:
                interactions.append({
                    'user_id': str(row.get(user_col, '')),
                    'target_user_id': str(row.get(target_col, '')) if target_col in df.columns and pd.notna(row.get(target_col)) else None,
                    'interaction_type': str(row.get(type_col, 'like')) if type_col in df.columns else 'like',
                    'timestamp': pd.to_datetime(row[time_col], errors='coerce').to_pydatetime().isoformat()
                })
            except Exception:
                continue
    sv.interactions = interactions

    user_attributes: Dict[str, Dict] = {}
    if user_col in df.columns:
        grouped = df.groupby(df[user_col].astype(str))
        for uid, group in grouped:
            user_attributes[uid] = {
                'total_likes': int(((group[type_col].astype(str).str.lower() == 'like').sum()) if type_col in df.columns else 0),
                'total_retweets': int(((group[type_col].astype(str).str.lower() == 'retweet').sum()) if type_col in df.columns else 0),
                'total_comments': int(((group[type_col].astype(str).str.lower() == 'comment').sum()) if type_col in df.columns else 0),
                'total_posts': int(((group[type_col].astype(str).str.lower() == 'post').sum()) if type_col in df.columns else 0),
                'follower_count': int(group.get('follower_count', pd.Series([0])).iloc[0]) if 'follower_count' in df.columns else 0,
                'follow_count': int(group.get('follow_count', pd.Series([0])).iloc[0]) if 'follow_count' in df.columns else 0,
                'gender': str(group.get('gender', pd.Series(['unknown'])).iloc[0]) if 'gender' in df.columns else 'unknown',
                'age': int(group.get('age', pd.Series([30])).iloc[0]) if 'age' in df.columns else 30,
                'followed_users': []
            }
    sv.user_attributes = user_attributes

    # 网络由外部构建并赋值
    sv.network = nx.Graph()
    return sv

@dataclass
class AG2vecConfig:
    """AG2vec config."""
    walk_length: int = 80
    num_walks: int = 10
    p: float = 1.0
    q: float = 1.0
    
    # 嵌入维度
    embedding_dim: int = 128
    
    # 训练参数
    window_size: int = 10
    negative_samples: int = 5
    learning_rate: float = 0.01
    epochs: int = 100
    batch_size: int = 1024
    
    # 注意力机制参数
    attention_dim: int = 64
    dropout_rate: float = 0.1

class Node2vecEmbedding:
    """Node2vec embedding learner."""
    
    def __init__(self, config: AG2vecConfig):
        self.config = config
        self.graph = None
        self.alias_nodes = {}
        self.alias_edges = {}
        self.node_embeddings = {}
        
    def fit(self, graph: nx.Graph, weighted_adjacency: np.ndarray = None):
        """Fit node2vec on a graph (optionally weighted)."""
        self.graph = graph
        self._preprocess_transition_probs(weighted_adjacency)
        self._learn_embeddings()
    
    def _preprocess_transition_probs(self, weighted_adjacency: np.ndarray = None):
        """Pre-compute transition probabilities."""
        nodes = list(self.graph.nodes())
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # 计算转移概率
        for node in nodes:
            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                continue
            
            # 计算未归一化概率
            unnormalized_probs = []
            for neighbor in neighbors:
                if weighted_adjacency is not None:
                    weight = weighted_adjacency[node_to_idx[node]][node_to_idx[neighbor]]
                else:
                    weight = self.graph[node][neighbor].get('weight', 1.0)
                unnormalized_probs.append(weight)
            
            # 归一化概率
            norm_const = sum(unnormalized_probs)
            normalized_probs = [prob / norm_const for prob in unnormalized_probs]
            
            # 创建别名表
            self.alias_nodes[node] = self._alias_setup(normalized_probs)
        
        # 计算边的转移概率
        for edge in self.graph.edges():
            src, dst = edge
            if src in self.alias_nodes and dst in self.alias_nodes:
                self.alias_edges[(src, dst)] = self._get_edge_prob(src, dst)
    
    def _alias_setup(self, probs: List[float]) -> Tuple[List[int], List[float]]:
        """Alias sampling setup."""
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=np.int)
        
        smaller = []
        larger = []
        
        for kk, prob in enumerate(probs):
            q[kk] = K * prob
            if q[kk] < 1.0:
                smaller.append(kk)
            else:
                larger.append(kk)
        
        while len(smaller) > 0 and len(larger) > 0:
            small = smaller.pop()
            large = larger.pop()
            
            J[small] = large
            q[large] = q[large] + q[small] - 1.0
            if q[large] < 1.0:
                smaller.append(large)
            else:
                larger.append(large)
        
        return J, q
    
    def _alias_draw(self, J: List[int], q: List[float]) -> int:
        """Alias draw."""
        K = len(J)
        kk = int(np.floor(np.random.rand() * K))
        if np.random.rand() < q[kk]:
            return kk
        else:
            return J[kk]
    
    def _get_edge_prob(self, src: str, dst: str) -> float:
        """Edge transition probability."""
        neighbors = list(self.graph.neighbors(src))
        if not neighbors:
            return 0.0
        
        # 计算到dst的概率
        for i, neighbor in enumerate(neighbors):
            if neighbor == dst:
                J, q = self.alias_nodes[src]
                prob = q[i] / len(neighbors)
                return prob
        
        return 0.0
    
    def _node2vec_walk(self, start_node: str) -> List[str]:
        """Perform a single node2vec walk."""
        walk = [start_node]
        
        while len(walk) < self.config.walk_length:
            cur = walk[-1]
            cur_neighbors = list(self.graph.neighbors(cur))
            
            if len(cur_neighbors) > 0:
                if len(walk) == 1:
                    # 第一步，均匀选择邻居
                    next_node = random.choice(cur_neighbors)
                else:
                    # 使用p和q参数选择下一个节点
                    prev = walk[-2]
                    next_node = self._get_next_node(prev, cur, cur_neighbors)
                
                walk.append(next_node)
            else:
                break
        
        return walk
    
    def _get_next_node(self, prev: str, cur: str, neighbors: List[str]) -> str:
        """Select next node using p/q parameters."""
        if not neighbors:
            return None
        
        # 计算转移概率
        probs = []
        for neighbor in neighbors:
            if neighbor == prev:
                prob = 1.0 / self.config.p
            elif self.graph.has_edge(cur, neighbor):
                prob = 1.0
            else:
                prob = 1.0 / self.config.q
            probs.append(prob)
        
        # 归一化概率
        norm_const = sum(probs)
        probs = [prob / norm_const for prob in probs]
        
        # 采样
        return np.random.choice(neighbors, p=probs)
    
    def _learn_embeddings(self):
        """Learn embeddings via random walks + skip-gram."""
        nodes = list(self.graph.nodes())
        n_nodes = len(nodes)
        
        # 生成随机游走序列
        walks = []
        for _ in range(self.config.num_walks):
            random.shuffle(nodes)
            for node in nodes:
                walk = self._node2vec_walk(node)
                walks.append(walk)
        
        # 使用Skip-gram模型学习嵌入
        self._skip_gram_learning(walks, nodes)
    
    def _skip_gram_learning(self, walks: List[List[str]], nodes: List[str]):
        """Skip-gram optimization loop."""
        # 创建词汇表
        vocab = {node: i for i, node in enumerate(nodes)}
        vocab_size = len(vocab)
        
        # 初始化嵌入矩阵
        embeddings = np.random.normal(0, 0.1, (vocab_size, self.config.embedding_dim))
        
        # 训练嵌入
        for epoch in range(self.config.epochs):
            total_loss = 0.0
            
            for walk in walks:
                for i, center_node in enumerate(walk):
                    if center_node not in vocab:
                        continue
                    
                    center_idx = vocab[center_node]
                    
                    # 获取上下文窗口
                    context_start = max(0, i - self.config.window_size)
                    context_end = min(len(walk), i + self.config.window_size + 1)
                    context = walk[context_start:context_end]
                    context.remove(center_node)
                    
                    # 更新嵌入
                    for context_node in context:
                        if context_node not in vocab:
                            continue
                        
                        context_idx = vocab[context_node]
                        
                        # 计算梯度并更新
                        loss, grad_center, grad_context = self._compute_gradients(
                            embeddings[center_idx], embeddings[context_idx]
                        )
                        
                        embeddings[center_idx] -= self.config.learning_rate * grad_center
                        embeddings[context_idx] -= self.config.learning_rate * grad_context
                        
                        total_loss += loss
            
            if epoch % 10 == 0:
                pass
        
        # save embeddings
        for node, idx in vocab.items():
            self.node_embeddings[node] = embeddings[idx]

if TORCH_AVAILABLE:
    class AttentionAggregator(nn.Module):
        """Attention aggregator."""
        
        def __init__(self, config: AG2vecConfig):
            super(AttentionAggregator, self).__init__()
            self.config = config
            
            # attention layer
            self.attention = nn.Sequential(
                nn.Linear(config.embedding_dim + 1, config.attention_dim),  # +1 for influence
                nn.ReLU(),
                nn.Dropout(config.dropout_rate),
                nn.Linear(config.attention_dim, 1)
            )
            
            # output projection
            self.output = nn.Linear(config.embedding_dim, config.embedding_dim)
else:
    class AttentionAggregator:
        """Attention aggregator placeholder when PyTorch is unavailable."""
        def __init__(self, config: AG2vecConfig):
            self.config = config
        
        def forward(self, node_embeddings, influence_scores):
            """Forward: aggregate node embeddings with influence weights."""
            # 拼接嵌入和影响力分数
            combined = torch.cat([node_embeddings, influence_scores], dim=1)
            
            # 计算注意力权重
            attention_weights = self.attention(combined)
            attention_weights = torch.softmax(attention_weights, dim=0)
            
            # 加权聚合
            aggregated = torch.sum(attention_weights * node_embeddings, dim=0)
            
            # 输出变换
            output = self.output(aggregated)
            
            return output

class AG2vec:
    """AG2vec group representation."""
    
    def __init__(self, config: AG2vecConfig = None):
        """Init with config."""
        self.config = config or AG2vecConfig()
        self.node2vec = Node2vecEmbedding(self.config)
        self.attention_aggregator = AttentionAggregator(self.config)
        self.group_embeddings = {}
        self.last_network: Optional[nx.Graph] = None
        
    def learn_group_representations(self, 
                                  communities: Dict[int, List[str]],
                                  network: nx.Graph,
                                  user_influences: Dict[str, float],
                                  weighted_adjacency: np.ndarray = None) -> Dict[int, np.ndarray]:
        """Learn group representations per community."""
        
        
        # 学习节点嵌入
        self.node2vec.fit(network, weighted_adjacency)
        self.last_network = network
        
        # 为每个群体学习表示
        for community_id, members in communities.items():
            if not members:
                continue
            
            # collect member embeddings
            member_embeddings = []
            member_influences = []
            
            for member in members:
                if member in self.node2vec.node_embeddings:
                    member_embeddings.append(self.node2vec.node_embeddings[member])
                    member_influences.append(user_influences.get(member, 0.0))
            
            if not member_embeddings:
                continue
            
            # to tensors
            embeddings_tensor = torch.tensor(member_embeddings, dtype=torch.float32)
            influences_tensor = torch.tensor(member_influences, dtype=torch.float32).unsqueeze(1)
            
            # attention aggregation
            if TORCH_AVAILABLE:
                with torch.no_grad():
                    group_embedding = self.attention_aggregator(embeddings_tensor, influences_tensor)
                    self.group_embeddings[community_id] = group_embedding.numpy()
            else:
                # simple average if no torch
                weighted_embeddings = embeddings_tensor * influences_tensor
                group_embedding = np.mean(weighted_embeddings, axis=0)
                self.group_embeddings[community_id] = group_embedding
        
        
        return self.group_embeddings

    # Dataset-level processing
    def process_dataset(self,
                        dataset_name: str = 'weibo',
                        processed_root: str = os.path.join('data', 'processed'),
                        time_col: str = 'created_at') -> List[str]:
        """Process all slices: build network, features, communities, and persist."""
        from feature_extraction import FeatureExtractor, FeatureConfig
        from ta_louvain import TALouvain

        created_feature_paths: List[str] = []
        extractor = FeatureExtractor(FeatureConfig())
        ta_louvain = TALouvain()

        for slice_dir in list_slice_directories(dataset_name, processed_root):
            df, meta = load_slice_df_and_meta(slice_dir)
            if df is None or meta is None:
                continue

            # normalize time col
            if time_col in df.columns:
                df[time_col] = pd.to_datetime(df[time_col], errors='coerce')
                df = df.dropna(subset=[time_col])
                if df.empty:
                    continue

            # build and save network
            G = build_network_from_df(df)
            save_network(slice_dir, G)

            # build slice view
            slice_view = build_slice_view_from_df(df, meta, time_col=time_col)
            slice_view.network = G  # 覆盖网络为构建结果
            features = extractor.extract_all_features(slice_view)

            # communities
            communities = ta_louvain.detect_communities(G, slice_view.user_attributes, features)

            # learn group representations
            self.learn_group_representations(communities, G, features['user_influences'])

            # persist features.json with metadata
            unique_id = f"{dataset_name}_{meta.get('slice_id', os.path.basename(slice_dir))}_{meta.get('start_time', '')}_{meta.get('end_time', '')}"
            output = {
                'id': unique_id,
                'dataset': dataset_name,
                'slice_id': meta.get('slice_id', os.path.basename(slice_dir)),
                'start_time': meta.get('start_time'),
                'end_time': meta.get('end_time'),
                'communities': {str(k): v for k, v in communities.items()},
                'features': features
            }
            feat_path = os.path.join(slice_dir, 'features.json')
            try:
                with open(feat_path, 'w', encoding='utf-8') as f:
                    json.dump(output, f, ensure_ascii=False, indent=2)
                created_feature_paths.append(feat_path)
            except Exception:
                continue

            # persist group embeddings
            try:
                import pickle
                emb_path = os.path.join(slice_dir, 'group_embeddings.pkl')
                with open(emb_path, 'wb') as f:
                    pickle.dump(self.group_embeddings, f)
            except Exception:
                pass

        return created_feature_paths
    
    def get_group_embedding(self, community_id: int) -> Optional[np.ndarray]:
        """Get group embedding by id."""
        return self.group_embeddings.get(community_id)
    
    def get_node_embedding(self, node: str) -> Optional[np.ndarray]:
        """Get node embedding."""
        return self.node2vec.node_embeddings.get(node)
    
    def compute_group_similarity(self, 
                               community_id1: int, 
                               community_id2: int) -> float:
        """Cosine similarity between two groups."""
        embedding1 = self.get_group_embedding(community_id1)
        embedding2 = self.get_group_embedding(community_id2)
        
        if embedding1 is None or embedding2 is None:
            return 0.0
        
        # cosine
        dot_product = np.dot(embedding1, embedding2)
        norm1 = np.linalg.norm(embedding1)
        norm2 = np.linalg.norm(embedding2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def get_group_features(self, 
                          community_id: int,
                          communities: Dict[int, List[str]],
                          user_attributes: Dict[str, Dict],
                          features: Dict) -> Dict:
        """
        获取群体特征
        
        Args:
            community_id: 社区ID
            communities: 社区划分
            user_attributes: 用户属性
            features: 特征字典
            
        Returns:
            群体特征字典
        """
        if community_id not in communities:
            return {}
        
        members = communities[community_id]
        if not members:
            return {}
        
        # 基础统计特征
        group_features = {
            'size': len(members),
            'embedding': self.get_group_embedding(community_id),
            'avg_influence': 0.0,
            'avg_topic_awareness': 0.0,
            'diversity_score': 0.0,
            'activity_score': 0.0
        }
        
        # 计算平均影响力
        influences = [features['user_influences'].get(member, 0.0) for member in members]
        if influences:
            group_features['avg_influence'] = np.mean(influences)
        
        # 计算平均话题意识度
        awareness = [features['user_topic_awareness'].get(member, 0.0) for member in members]
        if awareness:
            group_features['avg_topic_awareness'] = np.mean(awareness)
        
        # 计算多样性得分
        diversity_scores = []
        for member in members:
            user_attr = user_attributes.get(member, {})
            # 基于属性的多样性计算
            attr_values = [str(v) for v in user_attr.values()]
            diversity = len(set(attr_values)) / max(len(attr_values), 1)
            diversity_scores.append(diversity)
        
        if diversity_scores:
            group_features['diversity_score'] = np.mean(diversity_scores)
        
        # 计算活跃度得分
        activities = [features['group_activities'].get(member, 0.0) for member in members]
        if activities:
            group_features['activity_score'] = np.sum(activities)
        
        return group_features
    
    def reduce_dimensionality(self, 
                            target_dim: int = 64,
                            method: str = 'pca') -> Dict[int, np.ndarray]:
        """
        降维处理
        
        Args:
            target_dim: 目标维度
            method: 降维方法
            
        Returns:
            降维后的群体嵌入
        """
        if not self.group_embeddings:
            return {}
        
        # 收集所有嵌入
        embeddings = []
        community_ids = []
        
        for community_id, embedding in self.group_embeddings.items():
            embeddings.append(embedding)
            community_ids.append(community_id)
        
        embeddings = np.array(embeddings)
        
        # 降维
        if method == 'pca':
            pca = PCA(n_components=target_dim)
            reduced_embeddings = pca.fit_transform(embeddings)
        else:
            # 其他降维方法可以在这里添加
            reduced_embeddings = embeddings[:, :target_dim]
        
        # 更新嵌入
        reduced_group_embeddings = {}
        for i, community_id in enumerate(community_ids):
            reduced_group_embeddings[community_id] = reduced_embeddings[i]
        
        return reduced_group_embeddings
    
    def save_embeddings(self, filepath: str):
        """保存嵌入"""
        import pickle
        
        embedding_data = {
            'node_embeddings': self.node2vec.node_embeddings,
            'group_embeddings': self.group_embeddings,
            'config': self.config
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(embedding_data, f)
        
        logger.info(f"Embeddings saved to {filepath}")
    
    def load_embeddings(self, filepath: str):
        """加载嵌入"""
        import pickle
        
        with open(filepath, 'rb') as f:
            embedding_data = pickle.load(f)
        
        self.node2vec.node_embeddings = embedding_data['node_embeddings']
        self.group_embeddings = embedding_data['group_embeddings']
        self.config = embedding_data['config']
        
        logger.info(f"Embeddings loaded from {filepath}")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Process processed slices and run AG2vec pipeline.')
    parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'politifact', 'gossipcop'], help='Dataset name')
    parser.add_argument('--processed_root', type=str, default=os.path.join('data', 'processed'))
    parser.add_argument('--time_col', type=str, default='created_at')
    args = parser.parse_args()

    config = AG2vecConfig()
    ag2vec = AG2vec(config)
    created = ag2vec.process_dataset(dataset_name=args.dataset, processed_root=args.processed_root, time_col=args.time_col)
    print(f"[{args.dataset}] saved {len(created)} features.json files under slicer directories.")


if __name__ == "__main__":
    main()
