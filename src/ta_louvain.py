"""
TA-Louvain: topology + attribute-aware community detection with persistence.
"""

import os
import json
import pandas as pd
import numpy as np
import networkx as nx
from typing import Dict, List, Tuple, Set, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import math
from datetime import datetime
from sklearn.metrics.pairwise import cosine_similarity

logger = logging.getLogger(__name__)

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


def load_network_from_slice(slice_dir: str) -> Optional[nx.Graph]:
    pkl_path = os.path.join(slice_dir, 'network.pkl')
    if os.path.isfile(pkl_path):
        try:
            import pickle
            with open(pkl_path, 'rb') as f:
                return pickle.load(f)
        except Exception:
            pass
    edge_csv = os.path.join(slice_dir, 'network.edgelist.csv')
    if os.path.isfile(edge_csv):
        try:
            df = pd.read_csv(edge_csv)
            G = nx.Graph()
            for _, row in df.iterrows():
                G.add_edge(str(row['source']), str(row['target']))
            return G
        except Exception:
            pass
    return None


def load_meta_and_data(slice_dir: str) -> Tuple[Optional[Dict], Optional[pd.DataFrame]]:
    meta = None
    df = None
    meta_path = os.path.join(slice_dir, 'meta.json')
    data_path = os.path.join(slice_dir, 'data.csv')
    try:
        if os.path.isfile(meta_path):
            with open(meta_path, 'r', encoding='utf-8') as f:
                meta = json.load(f)
    except Exception:
        meta = None
    try:
        if os.path.isfile(data_path):
            df = pd.read_csv(data_path)
    except Exception:
        df = None
    return meta, df


def build_user_attributes_from_df(df: pd.DataFrame, user_col: str = 'user_id', type_col: str = 'interaction_type') -> Dict[str, Dict]:
    attrs: Dict[str, Dict] = {}
    if user_col not in df.columns:
        return attrs
    grouped = df.groupby(df[user_col].astype(str))
    for uid, group in grouped:
        attrs[uid] = {
            'total_likes': int(((group[type_col].astype(str).str.lower() == 'like').sum()) if type_col in df.columns else 0),
            'total_retweets': int(((group[type_col].astype(str).str.lower() == 'retweet').sum()) if type_col in df.columns else 0),
            'total_comments': int(((group[type_col].astype(str).str.lower() == 'comment').sum()) if type_col in df.columns else 0),
            'total_posts': int(((group[type_col].astype(str).str.lower() == 'post').sum()) if type_col in df.columns else 0),
            'follower_count': int(group.get('follower_count', pd.Series([0])).iloc[0]) if 'follower_count' in df.columns else 0,
            'follow_count': int(group.get('follow_count', pd.Series([0])).iloc[0]) if 'follow_count' in df.columns else 0,
            'age': int(group.get('age', pd.Series([30])).iloc[0]) if 'age' in df.columns else 30,
            'verified': bool(group.get('verified', pd.Series([False])).iloc[0]) if 'verified' in df.columns else False,
        }
    return attrs


def save_communities(slice_dir: str, communities: Dict[int, List[str]]) -> Dict[str, str]:
    paths: Dict[str, str] = {}
    # JSON
    com_json = os.path.join(slice_dir, 'communities.json')
    try:
        with open(com_json, 'w', encoding='utf-8') as f:
            json.dump({str(k): v for k, v in communities.items()}, f, ensure_ascii=False, indent=2)
        paths['json'] = com_json
    except Exception:
        pass
    # CSV
    try:
        import csv
        com_csv = os.path.join(slice_dir, 'memberships.csv')
        with open(com_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['user_id', 'community_id'])
            for cid, members in communities.items():
                for uid in members:
                    writer.writerow([uid, cid])
        paths['csv'] = com_csv
    except Exception:
        pass
    return paths

@dataclass
class TALouvainConfig:
    """Algorithm configuration."""
    lambda1: float = 0.6
    lambda2: float = 0.4
    gamma: float = 0.3
    jaccard_threshold: float = 0.5
    max_iterations: int = 100
    convergence_threshold: float = 1e-6

class TALouvain:
    """Community detection."""
    
    def __init__(self, config: TALouvainConfig = None):
        self.config = config or TALouvainConfig()
        self.communities = {}
        self.community_history = []
        
    def detect_communities(self, 
                          network: nx.Graph,
                          user_attributes: Dict[str, Dict],
                          features: Dict = None) -> Dict[int, List[str]]:
        """Run detection and return mapping {community_id: [users]}"""
        
        # similarity matrix
        similarity_matrix = self._calculate_similarity_matrix(network, user_attributes)
        
        # weighted adjacency
        weighted_adjacency = self._build_weighted_adjacency(network, similarity_matrix)
        
        # Louvain iterations
        communities = self._louvain_algorithm(network, weighted_adjacency, similarity_matrix)
        
        # organize results
        community_dict = defaultdict(list)
        for node, community_id in communities.items():
            community_dict[community_id].append(node)
        
        self.communities = dict(community_dict)
        
        
        return self.communities

    # Dataset-level batch processing
    def process_dataset(self,
                        dataset_name: str = 'weibo',
                        processed_root: str = os.path.join('data', 'processed'),
                        time_col: str = 'created_at') -> List[str]:
        saved_paths: List[str] = []
        for slice_dir in list_slice_directories(dataset_name, processed_root):
            G = load_network_from_slice(slice_dir)
            if G is None or G.number_of_nodes() == 0:
                continue
            meta, df = load_meta_and_data(slice_dir)
            # attributes from data.csv if available
            user_attrs = build_user_attributes_from_df(df) if df is not None else {}
            # features if available
            features = None
            feat_path = os.path.join(slice_dir, 'features.json')
            if os.path.isfile(feat_path):
                try:
                    with open(feat_path, 'r', encoding='utf-8') as f:
                        feat_obj = json.load(f)
                        features = feat_obj.get('features', None)
                except Exception:
                    features = None

            communities = self.detect_communities(G, user_attrs, features)
            out_map = save_communities(slice_dir, communities)
            if 'json' in out_map:
                saved_paths.append(out_map['json'])
        return saved_paths
    
    def _calculate_similarity_matrix(self, 
                                   network: nx.Graph,
                                   user_attributes: Dict[str, Dict]) -> np.ndarray:
        """Compute combined similarity matrix."""
        nodes = list(network.nodes())
        n = len(nodes)
        similarity_matrix = np.zeros((n, n))
        
        # 构建节点到索引的映射
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # 计算拓扑相似性
        topological_sim = self._calculate_topological_similarity(network, nodes)
        
        # 计算属性相似性
        attribute_sim = self._calculate_attribute_similarity(nodes, user_attributes)
        
        # 综合相似性
        for i, node_i in enumerate(nodes):
            for j, node_j in enumerate(nodes):
                if i != j:
                    # direct
                    direct_sim = topological_sim[i][j]
                    
                    # indirect
                    indirect_sim = self._calculate_indirect_similarity(
                        node_i, node_j, network, topological_sim, node_to_idx
                    )
                    
                    # attributes
                    attr_sim = attribute_sim[i][j]
                    
                    # combined
                    similarity_matrix[i][j] = (
                        self.config.lambda1 * direct_sim +
                        (1 - self.config.lambda1) * indirect_sim +
                        self.config.lambda2 * attr_sim
                    )
        
        return similarity_matrix
    
    def _calculate_topological_similarity(self, 
                                        network: nx.Graph,
                                        nodes: List[str]) -> np.ndarray:
        """Compute topological similarity."""
        n = len(nodes)
        similarity_matrix = np.zeros((n, n))
        
        for i, node_i in enumerate(nodes):
            neighbors_i = set(network.neighbors(node_i))
            degree_i = len(neighbors_i)
            
            for j, node_j in enumerate(nodes):
                if i != j:
                    neighbors_j = set(network.neighbors(node_j))
                    degree_j = len(neighbors_j)
                    
                    # common neighbors
                    common_neighbors = len(neighbors_i & neighbors_j)
                    
                    # direct similarity S_N
                    if degree_i + degree_j > 0:
                        similarity_matrix[i][j] = common_neighbors / (degree_i + degree_j)
        
        return similarity_matrix
    
    def _calculate_indirect_similarity(self, 
                                     node_i: str,
                                     node_j: str,
                                     network: nx.Graph,
                                     topological_sim: np.ndarray,
                                     node_to_idx: Dict[str, int]) -> float:
        """Compute indirect similarity via common neighbors."""
        neighbors_i = list(network.neighbors(node_i))
        neighbors_j = list(network.neighbors(node_j))
        
        max_indirect_sim = 0.0
        
        # 通过共同中介节点计算间接相似性
        for neighbor in neighbors_i:
            if neighbor in neighbors_j:
                idx_i = node_to_idx[node_i]
                idx_neighbor = node_to_idx[neighbor]
                idx_j = node_to_idx[node_j]
                
                # S_NN = S_N(i, neighbor) × S_N(neighbor, j)
                indirect_sim = (topological_sim[idx_i][idx_neighbor] * 
                              topological_sim[idx_neighbor][idx_j])
                max_indirect_sim = max(max_indirect_sim, indirect_sim)
        
        return max_indirect_sim
    
    def _calculate_attribute_similarity(self, 
                                      nodes: List[str],
                                      user_attributes: Dict[str, Dict]) -> np.ndarray:
        """Compute attribute-based similarity (cosine)."""
        n = len(nodes)
        
        # 构建属性向量矩阵
        attribute_vectors = []
        for node in nodes:
            user_attr = user_attributes.get(node, {})
            # 提取数值属性
            vector = [
                user_attr.get('age', 0),
                user_attr.get('follow_count', 0),
                user_attr.get('follower_count', 0),
                user_attr.get('account_age_days', 0),
                1.0 if user_attr.get('verified', False) else 0.0
            ]
            attribute_vectors.append(vector)
        
        attribute_matrix = np.array(attribute_vectors)
        
        # 计算余弦相似性
        similarity_matrix = cosine_similarity(attribute_matrix)
        
        return similarity_matrix
    
    def _build_weighted_adjacency(self, 
                                network: nx.Graph,
                                similarity_matrix: np.ndarray) -> np.ndarray:
        """Build weighted adjacency from topology + attributes."""
        nodes = list(network.nodes())
        n = len(nodes)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # 初始化邻接矩阵
        adjacency_matrix = np.zeros((n, n))
        
        # 填充基础邻接矩阵
        for edge in network.edges():
            i = node_to_idx[edge[0]]
            j = node_to_idx[edge[1]]
            weight = network[edge[0]][edge[1]].get('weight', 1.0)
            adjacency_matrix[i][j] = weight
            adjacency_matrix[j][i] = weight
        
        # 构建加权邻接矩阵
        weighted_adjacency = adjacency_matrix + self.config.gamma * similarity_matrix
        
        return weighted_adjacency
    
    def _louvain_algorithm(self, 
                          network: nx.Graph,
                          weighted_adjacency: np.ndarray,
                          similarity_matrix: np.ndarray) -> Dict[str, int]:
        """Run simplified Louvain and return {node: community_id}."""
        nodes = list(network.nodes())
        n = len(nodes)
        
        # 初始化：每个节点自成一个社区
        communities = {node: i for i, node in enumerate(nodes)}
        community_sizes = {i: 1 for i in range(n)}
        
        # 计算总边数
        total_edges = np.sum(weighted_adjacency) / 2
        
        iteration = 0
        improved = True
        
        while improved and iteration < self.config.max_iterations:
            improved = False
            iteration += 1
            
            # shuffle nodes per iteration
            node_order = nodes.copy()
            np.random.shuffle(node_order)
            
            for node in node_order:
                node_idx = nodes.index(node)
                current_community = communities[node]
                
                # current gain
                current_gain = self._calculate_modularity_gain(
                    node_idx, current_community, communities, community_sizes,
                    weighted_adjacency, similarity_matrix, total_edges
                )
                
                # try moving to neighbor communities
                best_community = current_community
                best_gain = current_gain
                
                neighbors = list(network.neighbors(node))
                neighbor_communities = set(communities[neighbor] for neighbor in neighbors)
                
                for neighbor_community in neighbor_communities:
                    if neighbor_community != current_community:
                        # gain if moved
                        gain = self._calculate_modularity_gain(
                            node_idx, neighbor_community, communities, community_sizes,
                            weighted_adjacency, similarity_matrix, total_edges
                        )
                        
                        if gain > best_gain:
                            best_gain = gain
                            best_community = neighbor_community
                
                # move if improvement
                if best_community != current_community and best_gain > current_gain:
                    # update assignment
                    communities[node] = best_community
                    community_sizes[current_community] -= 1
                    community_sizes[best_community] += 1
                    
                    # cleanup empty community ids
                    if community_sizes[current_community] == 0:
                        del community_sizes[current_community]
                    
                    improved = True
            
            
        
        return communities
    
    def _calculate_modularity_gain(self, 
                                 node_idx: int,
                                 target_community: int,
                                 communities: Dict[str, int],
                                 community_sizes: Dict[int, int],
                                 weighted_adjacency: np.ndarray,
                                 similarity_matrix: np.ndarray,
                                 total_edges: float) -> float:
        """Compute modularity gain if node moved to target community."""
        if total_edges == 0:
            return 0.0
        
        # node degree
        node_degree = np.sum(weighted_adjacency[node_idx])
        
        # connection strength to target community
        community_connection = 0.0
        for other_node, other_community in communities.items():
            if other_community == target_community:
                other_idx = list(communities.keys()).index(other_node)
                community_connection += weighted_adjacency[node_idx][other_idx]
        
        # total degree of target community
        community_degree = 0.0
        for other_node, other_community in communities.items():
            if other_community == target_community:
                other_idx = list(communities.keys()).index(other_node)
                community_degree += np.sum(weighted_adjacency[other_idx])
        
        # similarity term
        similarity_term = 0.0
        for other_node, other_community in communities.items():
            if other_community == target_community:
                other_idx = list(communities.keys()).index(other_node)
                similarity_term += similarity_matrix[node_idx][other_idx]
        
        # modularity gain
        gain = (community_connection - 
                (community_degree * node_degree) / (2 * total_edges)) * similarity_term
        
        return gain
    
    def track_community_evolution(self, 
                                current_communities: Dict[int, List[str]],
                                timestamp: datetime) -> Dict[int, int]:
        """Track community evolution between consecutive slices."""
        if not self.community_history:
            # first slice, just record
            self.community_history.append({
                'timestamp': timestamp,
                'communities': current_communities.copy()
            })
            return {}
        
        # previous communities
        previous_communities = self.community_history[-1]['communities']
        
        # jaccard matrix
        jaccard_matrix = self._calculate_jaccard_matrix(
            current_communities, previous_communities
        )
        
        # best matching mapping
        community_mapping = {}
        used_previous = set()
        
        for current_id, current_members in current_communities.items():
            best_match = None
            best_jaccard = 0.0
            
            for previous_id, previous_members in previous_communities.items():
                if previous_id not in used_previous:
                    jaccard = jaccard_matrix.get((current_id, previous_id), 0.0)
                    if jaccard > best_jaccard and jaccard >= self.config.jaccard_threshold:
                        best_jaccard = jaccard
                        best_match = previous_id
            
            if best_match is not None:
                community_mapping[current_id] = best_match
                used_previous.add(best_match)
        
        # append snapshot
        self.community_history.append({
            'timestamp': timestamp,
            'communities': current_communities.copy(),
            'mapping': community_mapping
        })
        
        return community_mapping
    
    def _calculate_jaccard_matrix(self, 
                                current_communities: Dict[int, List[str]],
                                previous_communities: Dict[int, List[str]]) -> Dict[Tuple[int, int], float]:
        """Compute pairwise Jaccard between community sets."""
        jaccard_matrix = {}
        
        for current_id, current_members in current_communities.items():
            current_set = set(current_members)
            
            for previous_id, previous_members in previous_communities.items():
                previous_set = set(previous_members)
                
                # 计算Jaccard系数
                intersection = len(current_set & previous_set)
                union = len(current_set | previous_set)
                
                if union > 0:
                    jaccard = intersection / union
                    jaccard_matrix[(current_id, previous_id)] = jaccard
        
        return jaccard_matrix
    
    def get_community_features(self, 
                             communities: Dict[int, List[str]],
                             user_attributes: Dict[str, Dict],
                             features: Dict) -> Dict[int, Dict]:
        """Aggregate per-community statistics."""
        community_features = {}
        
        for community_id, members in communities.items():
            if not members:
                continue
            
            # 计算社区统计特征
            community_attr = {
                'size': len(members),
                'avg_influence': 0.0,
                'avg_topic_awareness': 0.0,
                'total_activity': 0.0,
                'diversity_score': 0.0
            }
            
            # 计算平均影响力
            influences = [features['user_influences'].get(user, 0.0) for user in members]
            if influences:
                community_attr['avg_influence'] = np.mean(influences)
            
            # 计算平均话题意识度
            awareness = [features['user_topic_awareness'].get(user, 0.0) for user in members]
            if awareness:
                community_attr['avg_topic_awareness'] = np.mean(awareness)
            
            # 计算总活跃度
            activities = [features['group_activities'].get(user, 0.0) for user in members]
            if activities:
                community_attr['total_activity'] = np.sum(activities)
            
            # 计算多样性得分（基于用户属性）
            diversity_scores = []
            for user in members:
                user_attr = user_attributes.get(user, {})
                # 简化的多样性计算
                diversity = len(set(user_attr.values()))
                diversity_scores.append(diversity)
            
            if diversity_scores:
                community_attr['diversity_score'] = np.mean(diversity_scores)
            
            community_features[community_id] = community_attr
        
        return community_features


def main():
    """Batch entry: run TA-Louvain over saved networks."""
    import argparse
    parser = argparse.ArgumentParser(description='Run TA-Louvain on saved networks and persist communities.')
    parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'politifact', 'gossipcop'])
    parser.add_argument('--processed_root', type=str, default=os.path.join('data', 'processed'))
    parser.add_argument('--time_col', type=str, default='created_at')
    args = parser.parse_args()

    ta = TALouvain(TALouvainConfig())
    saved = ta.process_dataset(dataset_name=args.dataset, processed_root=args.processed_root, time_col=args.time_col)
    print(f"[{args.dataset}] saved {len(saved)} communities.json files under slicer directories.")


if __name__ == "__main__":
    main()
