"""
Tests for Guided Topic Detection System
"""

import pytest
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from data_slicer import DataSlicer
from feature_extraction import FeatureExtractor, FeatureConfig
from ta_louvain import TALouvain, TALouvainConfig
from ag2vec import AG2vec, AG2vecConfig
from mut_inf import PositionQuantifier, PositionConfig
from gc_gru import GCGRUPredictor, GCGRUConfig


class TestDataSlicer:
    """Test cases for DataSlicer"""
    
    def test_data_slicer_initialization(self):
        """Test DataSlicer initialization"""
        slicer = DataSlicer(
            input_dir='data/raw',
            output_dir='data/processed',
            interval_days=7,
            min_slice_size=50
        )
        
        assert slicer.input_dir == 'data/raw'
        assert slicer.output_dir == 'data/processed'
        assert slicer.interval_days == 7
        assert slicer.min_slice_size == 50


class TestFeatureExtractor:
    """Test cases for FeatureExtractor"""
    
    def test_feature_extractor_initialization(self):
        """Test FeatureExtractor initialization"""
        config = FeatureConfig()
        extractor = FeatureExtractor(config)
        
        assert extractor.config == config
        assert extractor.config.time_decay_factor == 0.1
        assert extractor.config.beta == 0.4


class TestTALouvain:
    """Test cases for TALouvain"""
    
    def test_ta_louvain_initialization(self):
        """Test TALouvain initialization"""
        config = TALouvainConfig()
        ta_louvain = TALouvain(config)
        
        assert ta_louvain.config == config
        assert ta_louvain.config.lambda1 == 0.6
        assert ta_louvain.config.lambda2 == 0.4


class TestAG2vec:
    """Test cases for AG2vec"""
    
    def test_ag2vec_initialization(self):
        """Test AG2vec initialization"""
        config = AG2vecConfig()
        ag2vec = AG2vec(config)
        
        assert ag2vec.config == config
        assert ag2vec.config.embedding_dim == 128
        assert ag2vec.config.walk_length == 80


class TestPositionQuantifier:
    """Test cases for PositionQuantifier"""
    
    def test_position_quantifier_initialization(self):
        """Test PositionQuantifier initialization"""
        config = PositionConfig()
        quantifier = PositionQuantifier(config)
        
        assert quantifier.config == config
        assert quantifier.config.label_smoothing_epsilon == 0.1


class TestGCGRUPredictor:
    """Test cases for GCGRUPredictor"""
    
    def test_gc_gru_predictor_initialization(self):
        """Test GCGRUPredictor initialization"""
        config = GCGRUConfig()
        predictor = GCGRUPredictor(config)
        
        assert predictor.config == config
        assert predictor.config.hidden_dim == 128
        assert predictor.config.sequence_length == 10
        assert not predictor.is_trained


if __name__ == "__main__":
    pytest.main([__file__])



