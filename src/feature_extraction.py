import os
import json
import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, List, Tuple, Optional, Union
from datetime import datetime, timedelta
from dataclasses import dataclass
from collections import defaultdict
import math


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    time_decay_factor: float = 0.1
    w_l: float = 0.2
    w_r: float = 1.5
    w_c: float = 0.3
    beta: float = 0.4

    alpha1: float =1.0
    alpha2: float = 1.0
    epsilon: float = 1e-6


class FeatureExtractor:
    """Feature extractor implementing core metrics."""

    def __init__(self, config: FeatureConfig = None):
        self.config = config or FeatureConfig()

    def extract_user_interaction(self, user_i: str, user_j: str, interactions: List[Dict],
                                 network: nx.Graph, current_time: datetime) -> float:
        """Compute user interaction score."""
        try:
            try:
                path_length = nx.shortest_path_length(network, user_i, user_j)
            except nx.NetworkXNoPath:
                path_length = len(network.nodes())

            if path_length == 0:
                return 0.0

            user_interactions = [
                i for i in interactions
                if ((i['user_id'] == user_i and i['target_user_id'] == user_j) or
                    (i['user_id'] == user_j and i['target_user_id'] == user_i))
            ]

            if not user_interactions:
                return 0.0

            total_interaction = 0.0
            message_count = 0

            for i in user_interactions:
                itype = i.get('interaction_type', 'like')
                if itype == 'comment':
                    I_ma = 1.0
                elif itype == 'retweet':
                    I_ma = 0.8
                elif itype == 'like':
                    I_ma = 0.5
                else:
                    I_ma = 0.0

                interaction_time = datetime.fromisoformat(i['timestamp'])
                time_diff = (current_time - interaction_time).total_seconds() / 3600
                time_decay = math.exp(-self.config.time_decay_factor * time_diff)

                total_interaction += I_ma * time_decay
                message_count += 1

            interaction_score = total_interaction / path_length
            return min(interaction_score, 2.3 * message_count)

        except Exception:
            return 0.0

    def extract_user_influence(self, user_id: str, user_attributes: Dict,
                               interactions: List[Dict], current_time: datetime,
                               time_window_hours: int = 24) -> float:
        """Compute user influence index."""
        try:
            user_attr = user_attributes.get(user_id, {})
            w_i = self._calculate_historical_credibility(user_attr)

            time_start = current_time - timedelta(hours=time_window_hours)
            recent = [
                i for i in interactions
                if (i.get('target_user_id') == user_id and
                    datetime.fromisoformat(i['timestamp']) >= time_start)
            ]

            L_i = sum(1 for i in recent if i.get('interaction_type') == 'like')
            R_i = sum(1 for i in recent if i.get('interaction_type') == 'retweet')
            C_i = sum(1 for i in recent if i.get('interaction_type') == 'comment')

            L_all = user_attr.get('total_likes', 0)
            R_all = user_attr.get('total_retweets', 0)
            C_all = user_attr.get('total_comments', 0)

            follower_count = user_attr.get('follower_count', 1)
            follow_count = user_attr.get('follow_count', 1)
            follow_ratio = follower_count / (follow_count + self.config.epsilon)

            dynamic_part = (L_i + self.config.w_r * R_i + self.config.w_c * C_i) / (
                L_all + self.config.w_r * R_all + self.config.w_c * C_all + self.config.epsilon)

            historical_part = w_i * follow_ratio
            influence = self.config.beta * dynamic_part + (1 - self.config.beta) * historical_part
            return min(influence, 10.0)

        except Exception:
            return 0.0

    def _calculate_historical_credibility(self, user_attr: Dict) -> float:
        """Compute historical credibility."""
        L_all = user_attr.get('total_likes', 0)
        R_all = user_attr.get('total_retweets', 0)
        C_all = user_attr.get('total_comments', 0)
        numerator = L_all + self.config.w_r * R_all + self.config.w_c * C_all
        denominator = numerator + self.config.epsilon
        return numerator / denominator

    def extract_topic_awareness(self, user_id: str, user_attributes: Dict,
                                interactions: List[Dict], comments: List[Dict],
                                current_time: datetime, time_window_hours: int = 24) -> float:
        """Compute topic awareness score."""
        try:
            time_start = current_time - timedelta(hours=time_window_hours)

            user_interactions = [
                i for i in interactions
                if (i.get('user_id') == user_id and
                    datetime.fromisoformat(i['timestamp']) >= time_start)
            ]
            user_comments = [
                c for c in comments
                if (c.get('user_id') == user_id and
                    datetime.fromisoformat(c['timestamp']) >= time_start)
            ]

            N_orig = sum(1 for i in user_interactions if i.get('interaction_type') == 'post')
            N_ret = sum(1 for i in user_interactions if i.get('interaction_type') == 'retweet')
            N_like = sum(1 for i in user_interactions if i.get('interaction_type') == 'like')
            N_com = len(user_comments)
            total_act_user = N_orig + N_ret + N_like + N_com
            self_behavior_ratio = (N_orig + N_ret + N_like + N_com) / total_act_user if total_act_user > 0 else 0.0

            user_attr = user_attributes.get(user_id, {})
            followed_users = user_attr.get('followed_users', [])

            if followed_users:
                followed_interactions = [
                    i for i in interactions
                    if (i.get('user_id') in followed_users and
                        datetime.fromisoformat(i['timestamp']) >= time_start)
                ]
                followed_comments = [
                    c for c in comments
                    if (c.get('user_id') in followed_users and
                        datetime.fromisoformat(c['timestamp']) >= time_start)
                ]
                N_orig_f = sum(1 for i in followed_interactions if i.get('interaction_type') == 'post')
                N_ret_f = sum(1 for i in followed_interactions if i.get('interaction_type') == 'retweet')
                N_like_f = sum(1 for i in followed_interactions if i.get('interaction_type') == 'like')
                N_com_f = len(followed_comments)
                total_act_f = N_orig_f + N_ret_f + N_like_f + N_com_f
                followed_behavior_ratio = (N_orig_f + N_ret_f + N_like_f + N_com_f) / (total_act_f + 1) if total_act_f > 0 else 0.0
            else:
                followed_behavior_ratio = 0.0

            topic_awareness = (self.config.alpha1 * self_behavior_ratio +
                               self.config.alpha2 * followed_behavior_ratio)
            return min(topic_awareness, 1.0)

        except Exception:
            return 0.0

    def extract_topic_heat(self, comments: List[Dict], current_time: datetime,
                           time_window_hours: int = 24) -> float:
        """Compute topic heat value."""
        try:
            time_start = current_time - timedelta(hours=time_window_hours)
            recent_comments = [c for c in comments if datetime.fromisoformat(c['timestamp']) >= time_start]
            if not recent_comments:
                return 0.0

            hourly_counts = defaultdict(int)
            for c in recent_comments:
                hour_key = datetime.fromisoformat(c['timestamp']).replace(minute=0, second=0, microsecond=0)
                hourly_counts[hour_key] += 1

            counts = list(hourly_counts.values())
            if len(counts) < 2:
                return len(counts)

            peaks = 0
            for i in range(1, len(counts) - 1):
                if counts[i] > counts[i - 1] and counts[i] > counts[i + 1]:
                    peaks += 1
            return peaks

        except Exception:
            return 0.0

    def extract_group_activity(self, group_users: List[str], user_attributes: Dict) -> float:
        """Compute group activity."""
        try:
            total_retweets = 0
            total_originals = 0
            for uid in group_users:
                ua = user_attributes.get(uid, {})
                total_retweets += ua.get('total_retweets', 0)
                total_originals += ua.get('total_posts', 0)

            if total_retweets + total_originals == 0:
                return 0.0
            activity = total_retweets / (total_retweets + total_originals)
            return min(activity, 1.0)

        except Exception:
            return 0.0

    def extract_user_properties(self, user_id: str, user_attributes: Dict) -> List[float]:
        """Extract basic user properties."""
        try:
            ua = user_attributes.get(user_id, {})
            age = ua.get('age', 30)
            normalized_age = min(age / 100.0, 1.0)

            gender = ua.get('gender', 'unknown')
            sex = 1.0 if gender == 'male' else 0.0 if gender == 'female' else 0.5

            follow_count = ua.get('follow_count', 0)
            normalized_follow = min(math.log(follow_count + 1) / 10.0, 1.0)

            follower_count = ua.get('follower_count', 0)
            normalized_fans = min(math.log(follower_count + 1) / 12.0, 1.0)

            return [normalized_age, sex, normalized_follow, normalized_fans]

        except Exception:
            return [0.0, 0.5, 0.0, 0.0]

    def extract_all_features(self, data_slice, user_pairs: List[Tuple[str, str]] = None) -> Dict:
        """Extract all features."""
        features = {
            'user_interactions': {},
            'user_influences': {},
            'user_topic_awareness': {},
            'topic_heat': 0.0,
            'group_activities': {},
            'user_properties': {}
        }

        features['topic_heat'] = self.extract_topic_heat(data_slice.comments, data_slice.timestamp)

        for user_id in data_slice.users:
            features['user_influences'][user_id] = self.extract_user_influence(
                user_id, data_slice.user_attributes, data_slice.interactions, data_slice.timestamp)
            features['user_topic_awareness'][user_id] = self.extract_topic_awareness(
                user_id, data_slice.user_attributes, data_slice.interactions,
                data_slice.comments, data_slice.timestamp)
            features['user_properties'][user_id] = self.extract_user_properties(
                user_id, data_slice.user_attributes)

        if user_pairs:
            for user_i, user_j in user_pairs:
                if user_i in data_slice.users and user_j in data_slice.users:
                    key = f"{user_i}_{user_j}"
                    features['user_interactions'][key] = self.extract_user_interaction(
                        user_i, user_j, data_slice.interactions, data_slice.network, data_slice.timestamp)

        for user_id in data_slice.users:
            features['group_activities'][user_id] = self.extract_group_activity([user_id], data_slice.user_attributes)

        return features

def save_features_to_file(features: Dict, output_dir: str = "output"):
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    file_path = os.path.join(output_dir, f"features_{timestamp}.json")
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(features, f, indent=4, ensure_ascii=False)
    print(f"Features saved to {file_path}")
    return file_path

def main():
    """Example: use FeatureExtractor."""
    from data_slicer import DataSlicer

    slicer = DataSlicer()
    raw_data = slicer.load_raw_data("data/raw")
    slices = slicer.create_time_slices(raw_data)

    config = FeatureConfig()
    extractor = FeatureExtractor(config)

    if slices:
        features = extractor.extract_all_features(slices[0])
        save_features_to_file(features)
        print(f"Extracted features for {len(features['user_influences'])} users")
        print(f"Topic heat: {features['topic_heat']}")
        print(f"Sample user influence: {list(features['user_influences'].values())[:5]}")


if __name__ == "__main__":
    main()
