"""
Mutual influence quantification over communities; dataset batch processing and persistence.
"""

import os
import json
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import logging
from dataclasses import dataclass
from collections import defaultdict
import math
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

logger = logging.getLogger(__name__)


def list_slice_directories(dataset_name: str,
                           processed_root: str = os.path.join('data', 'processed')) -> List[str]:
    base_dir = os.path.join(processed_root, dataset_name, 'slicer')
    if not os.path.isdir(base_dir):
        return []
    def sort_key(x: str):
        return (len(x), x)
    return [os.path.join(base_dir, d) for d in sorted(os.listdir(base_dir), key=sort_key)
            if os.path.isdir(os.path.join(base_dir, d))]


def load_slice_artifacts(slice_dir: str) -> Tuple[Optional[Dict[int, List[str]]], Optional[Dict], Optional[pd.DataFrame]]:
    """Load communities, features and raw slice data if present."""
    communities: Optional[Dict[int, List[str]]] = None
    features: Optional[Dict] = None
    df: Optional[pd.DataFrame] = None

    # communities
    com_path = os.path.join(slice_dir, 'communities.json')
    if os.path.isfile(com_path):
        try:
            with open(com_path, 'r', encoding='utf-8') as f:
                raw = json.load(f)
                communities = {int(k): v for k, v in raw.items()}
        except Exception:
            communities = None

    # features (from features.json if exists)
    feat_path = os.path.join(slice_dir, 'features.json')
    if os.path.isfile(feat_path):
        try:
            with open(feat_path, 'r', encoding='utf-8') as f:
                feat_obj = json.load(f)
                features = feat_obj.get('features', None)
        except Exception:
            features = None

    # raw slice data
    data_csv = os.path.join(slice_dir, 'data.csv')
    if os.path.isfile(data_csv):
        try:
            df = pd.read_csv(data_csv)
        except Exception:
            df = None

    return communities, features, df


@dataclass
class PositionConfig:
    label_smoothing_epsilon: float = 0.1
    omega1: float = 0.5
    omega2: float = 0.5
    test_size: float = 0.2
    random_state: int = 42
    epsilon: float = 1e-6


class SoftLabelProcessor:
    def __init__(self, config: PositionConfig):
        self.config = config

    def create_soft_labels(self, comments: List[Dict], stance_labels: Dict[str, str] = None) -> Dict[str, np.ndarray]:
        soft_labels: Dict[str, np.ndarray] = {}
        for comment in comments:
            comment_id = comment.get('id', str(hash(comment.get('content', ''))))
            if stance_labels and comment_id in stance_labels:
                original_stance = stance_labels[comment_id]
            else:
                original_stance = self._extract_stance_from_comment(comment)
            soft_label = self._create_single_soft_label(original_stance)
            soft_labels[comment_id] = soft_label
        return soft_labels

    def _extract_stance_from_comment(self, comment: Dict) -> str:
        content = str(comment.get('content', '')).lower()
        positive_words = ['支持', '同意', '赞成', '好', '棒', 'support', 'agree', 'good']
        negative_words = ['反对', '不同意', '不赞成', '坏', '差', 'oppose', 'disagree', 'bad']
        positive_count = sum(1 for w in positive_words if w in content)
        negative_count = sum(1 for w in negative_words if w in content)
        if positive_count > negative_count:
            return 'support'
        elif negative_count > positive_count:
            return 'oppose'
        return 'neutral'

    def _create_single_soft_label(self, original_stance: str) -> np.ndarray:
        stance_to_idx = {'support': 0, 'oppose': 1, 'neutral': 2}
        idx = stance_to_idx.get(original_stance, 2)
        soft_label = np.zeros(3)
        soft_label[idx] = 1 - self.config.label_smoothing_epsilon
        soft_label[(idx + 1) % 3] = self.config.label_smoothing_epsilon / 2
        soft_label[(idx + 2) % 3] = self.config.label_smoothing_epsilon / 2
        return soft_label

    def calculate_confidence_weights(self, soft_labels: Dict[str, np.ndarray]) -> Dict[str, float]:
        return {cid: float(np.max(sl)) for cid, sl in soft_labels.items()}

    def calculate_group_stance_proportions(self,
                                           comments: List[Dict],
                                           soft_labels: Dict[str, np.ndarray],
                                           confidence_weights: Dict[str, float],
                                           group_members: List[str]) -> np.ndarray:
        group_comments = [c for c in comments if str(c.get('user_id')) in set(group_members)]
        if not group_comments:
            return np.array([0.33, 0.33, 0.34])
        weighted = np.zeros(3)
        total_w = 0.0
        for c in group_comments:
            cid = c.get('id', str(hash(c.get('content', ''))))
            if cid in soft_labels and cid in confidence_weights:
                w = confidence_weights[cid]
                weighted += w * soft_labels[cid]
                total_w += w
        if total_w > 0:
            return weighted / total_w
        return np.array([0.33, 0.33, 0.34])


class PositionInfluenceCalculator:
    def __init__(self, config: PositionConfig):
        self.config = config
        self.regression_models: Dict[str, LinearRegression] = {}
        self.scalers: Dict[str, StandardScaler] = {}

    def calculate_internal_factors(self, user_id: str, stance_proportions: np.ndarray,
                                   topic_awareness: float, influence: float) -> Dict[str, float]:
        stances = ['support', 'oppose', 'neutral']
        return {s: float(stance_proportions[i] * topic_awareness * influence) for i, s in enumerate(stances)}

    def calculate_external_factors(self, user_id: str, stance_proportions: np.ndarray,
                                   group_activity: float, topic_heat: float) -> Dict[str, float]:
        stances = ['support', 'oppose', 'neutral']
        return {s: float(stance_proportions[i] * group_activity * topic_heat) for i, s in enumerate(stances)}

    def calculate_position_influence(self, internal_factors: Dict[str, float],
                                     external_factors: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for s in ['support', 'oppose', 'neutral']:
            influence = 0.5 + 0.3 * internal_factors.get(s, 0.0) + 0.2 * external_factors.get(s, 0.0)
            out[s] = max(0.0, min(1.0, float(influence)))
        return out


class GroupPositionAggregator:
    def __init__(self, config: PositionConfig):
        self.config = config

    def aggregate_user_influences(self, group_members: List[str], user_influences: Dict[str, float],
                                  position_influences: Dict[str, Dict[str, float]]) -> Dict[str, float]:
        total = sum(user_influences.get(m, 0.0) for m in group_members)
        if total == 0:
            return {'support': 0.0, 'oppose': 0.0, 'neutral': 0.0}
        result = {'support': 0.0, 'oppose': 0.0, 'neutral': 0.0}
        for m in group_members:
            w = user_influences.get(m, 0.0) / total
            pi = position_influences.get(m, {})
            for k in result:
                result[k] += w * pi.get(k, 0.0)
        return result


class GameTheoryPositionCalculator:
    def __init__(self, config: PositionConfig):
        self.config = config

    def _sigmoid(self, x: float) -> float:
        try:
            return 1.0 / (1.0 + math.exp(-x))
        except OverflowError:
            return 1.0 if x > 0 else 0.0

    def calculate_mutual_influence(self, group_position_influences: Dict[str, float]) -> Dict[str, float]:
        s = group_position_influences.get('support', 0.0)
        o = group_position_influences.get('oppose', 0.0)
        n = group_position_influences.get('neutral', 0.0)
        return {
            'support': self._sigmoid(s + n - o),
            'oppose': self._sigmoid(o + n - s),
            'neutral': self._sigmoid(n + self.config.omega1 * s + self.config.omega2 * o)
        }


class PositionQuantifier:
    def __init__(self, config: PositionConfig = None):
        self.config = config or PositionConfig()
        self.soft_label_processor = SoftLabelProcessor(self.config)
        self.influence_calculator = PositionInfluenceCalculator(self.config)
        self.group_aggregator = GroupPositionAggregator(self.config)
        self.game_theory_calculator = GameTheoryPositionCalculator(self.config)

    def quantify_group_positions(self,
                                 communities: Dict[int, List[str]],
                                 comments: List[Dict],
                                 features: Dict,
                                 user_attributes: Dict[str, Dict],
                                 topic_heat: float,
                                 stance_labels: Dict[str, str] = None) -> Dict[int, Dict]:
        soft_labels = self.soft_label_processor.create_soft_labels(comments, stance_labels)
        confidence_weights = self.soft_label_processor.calculate_confidence_weights(soft_labels)

        group_positions: Dict[int, Dict] = {}
        for community_id, members in communities.items():
            if not members:
                continue
            stance_proportions = self.soft_label_processor.calculate_group_stance_proportions(
                comments, soft_labels, confidence_weights, members
            )
            group_activity = self._calculate_group_activity(members, features)

            user_position_influences: Dict[str, Dict[str, float]] = {}
            for member in members:
                topic_awareness = features.get('user_topic_awareness', {}).get(member, 0.0)
                influence = features.get('user_influences', {}).get(member, 0.0)
                internal_factors = self.influence_calculator.calculate_internal_factors(
                    member, stance_proportions, topic_awareness, influence
                )
                external_factors = self.influence_calculator.calculate_external_factors(
                    member, stance_proportions, group_activity, features.get('topic_heat', topic_heat)
                )
                user_position_influences[member] = self.influence_calculator.calculate_position_influence(
                    internal_factors, external_factors
                )

            group_position_influences = self.group_aggregator.aggregate_user_influences(
                members, features.get('user_influences', {}), user_position_influences
            )
            mutual_influences = self.game_theory_calculator.calculate_mutual_influence(group_position_influences)

            group_positions[community_id] = {
                'stance_proportions': stance_proportions.tolist(),
                'group_activity': float(group_activity),
                'position_influences': group_position_influences,
                'mutual_influences': mutual_influences,
                'members': members
            }

        return group_positions

    def _calculate_group_activity(self, members: List[str], features: Dict) -> float:
        activities = [features.get('group_activities', {}).get(m, 0.0) for m in members]
        return float(np.mean(activities)) if activities else 0.0

    def create_position_matrix(self, group_positions: Dict[int, Dict]) -> np.ndarray:
        if not group_positions:
            return np.array([])
        n = len(group_positions)
        mat = np.zeros((n, 3))
        for i, (_, pdata) in enumerate(group_positions.items()):
            mi = pdata['mutual_influences']
            mat[i, 0] = mi['support']
            mat[i, 1] = mi['oppose']
            mat[i, 2] = mi['neutral']
        return mat

    def process_dataset(self, dataset_name: str = 'weibo',
                        processed_root: str = os.path.join('data', 'processed')) -> List[str]:
        saved_paths: List[str] = []
        for slice_dir in list_slice_directories(dataset_name, processed_root):
            communities, features, df = load_slice_artifacts(slice_dir)
            if communities is None or features is None:
                continue

            # Build minimal comments list if available
            comments: List[Dict] = []
            if df is not None and 'user_id' in df.columns:
                # If content column exists, use it; else empty content
                content_col = 'content' if 'content' in df.columns else None
                for _, row in df.iterrows():
                    try:
                        comments.append({
                            'id': row.get('comment_id', None) or str(hash(str(row.get(content_col, '')) + str(row.get('user_id')))),
                            'user_id': str(row['user_id']),
                            'content': str(row.get(content_col, ''))
                        })
                    except Exception:
                        continue

            # user_attributes optional; not used directly now
            user_attributes: Dict[str, Dict] = {}
            topic_heat = float(features.get('topic_heat', 0.0))

            group_positions = self.quantify_group_positions(
                communities, comments, features, user_attributes, topic_heat
            )
            matrix = self.create_position_matrix(group_positions)

            # Save
            out_json = os.path.join(slice_dir, 'positions.json')
            try:
                with open(out_json, 'w', encoding='utf-8') as f:
                    json.dump({str(k): v for k, v in group_positions.items()}, f, ensure_ascii=False, indent=2)
                saved_paths.append(out_json)
            except Exception:
                pass

            try:
                npy_path = os.path.join(slice_dir, 'position_matrix.npy')
                np.save(npy_path, matrix)
            except Exception:
                pass

        return saved_paths


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Mutual influence quantification over processed slices.')
    parser.add_argument('--dataset', type=str, default='weibo', choices=['weibo', 'politifact', 'gossipcop'])
    parser.add_argument('--processed_root', type=str, default=os.path.join('data', 'processed'))
    args = parser.parse_args()

    pq = PositionQuantifier(PositionConfig())
    saved = pq.process_dataset(dataset_name=args.dataset, processed_root=args.processed_root)
    print(f"[{args.dataset}] saved {len(saved)} positions.json files under slicer directories.")


if __name__ == "__main__":
    main()


