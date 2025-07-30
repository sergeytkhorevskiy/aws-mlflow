"""Unit tests for train_text_classifier.py."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch, MagicMock
import tempfile
import os

# Import the functions to test
import sys
sys.path.append('scripts')
from train_text_classifier import (
    download_from_s3,
    load_and_prepare_data,
    train_text_classifier,
    save_model_and_artifacts
)


class TestDownloadFromS3:
    """Test S3 download functionality."""
    
    def test_download_from_s3_success(self, mock_s3_client, temp_file):
        """Test successful S3 download."""
        # Arrange
        bucket = "test-bucket"
        key = "test-file.csv"
        dest = temp_file
        
        # Act
        download_from_s3(bucket, key, dest)
        
        # Assert
        mock_s3_client.download_file.assert_called_once_with(bucket, key, dest)
    
    def test_download_from_s3_error(self, mock_s3_client):
        """Test S3 download error handling."""
        # Arrange
        mock_s3_client.download_file.side_effect = Exception("S3 error")
        
        # Act & Assert
        with pytest.raises(Exception):
            download_from_s3("bucket", "key", "dest")


class TestLoadAndPrepareData:
    """Test data loading and preparation."""
    
    def test_load_text_data(self, sample_dataset, temp_file):
        """Test loading data with text column."""
        # Arrange
        sample_dataset.to_csv(temp_file, index=False)
        
        # Act
        X_train, X_test, y_train, y_test, use_text = load_and_prepare_data(temp_file)
        
        # Assert
        assert use_text is True
        assert len(X_train) > 0
        assert len(X_test) > 0
        assert len(y_train) > 0
        assert len(y_test) > 0
    
    def test_load_feature_data(self, temp_file):
        """Test loading data with feature columns."""
        # Arrange
        df = pd.DataFrame({
            'feature_0': np.random.randn(100),
            'feature_1': np.random.randn(100),
            'target': np.random.choice([0, 1, 2, 3], 100)
        })
        df.to_csv(temp_file, index=False)
        
        # Act
        X_train, X_test, y_train, y_test, use_text = load_and_prepare_data(temp_file)
        
        # Assert
        assert use_text is False
        assert X_train.shape[1] == 2  # 2 feature columns
        assert len(y_train) > 0
    
    def test_load_data_file_not_found(self):
        """Test loading data from non-existent file."""
        # Act & Assert
        with pytest.raises(FileNotFoundError):
            load_and_prepare_data("non_existent_file.csv")


class TestTrainTextClassifier:
    """Test text classifier training."""
    
    @patch('scripts.train_text_classifier.mlflow')
    def test_train_text_classifier_text_mode(self, mock_mlflow, sample_dataset):
        """Test training with text data."""
        # Arrange
        X_train = sample_dataset['text'].iloc[:80]
        X_test = sample_dataset['text'].iloc[80:]
        y_train = sample_dataset['category'].iloc[:80]
        y_test = sample_dataset['category'].iloc[80:]
        
        # Act
        pipeline, accuracy, y_pred = train_text_classifier(
            X_train, y_train, X_test, y_test, use_text=True
        )
        
        # Assert
        assert pipeline is not None
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert len(y_pred) == len(y_test)
    
    @patch('scripts.train_text_classifier.mlflow')
    def test_train_text_classifier_feature_mode(self, mock_mlflow):
        """Test training with feature data."""
        # Arrange
        X_train = np.random.randn(80, 10)
        X_test = np.random.randn(20, 10)
        y_train = np.random.choice([0, 1, 2, 3], 80)
        y_test = np.random.choice([0, 1, 2, 3], 20)
        
        # Act
        pipeline, accuracy, y_pred = train_text_classifier(
            X_train, y_train, X_test, y_test, use_text=False
        )
        
        # Assert
        assert pipeline is not None
        assert isinstance(accuracy, float)
        assert 0 <= accuracy <= 1
        assert len(y_pred) == len(y_test)


class TestSaveModelAndArtifacts:
    """Test model and artifacts saving."""
    
    def test_save_model_and_artifacts(self, temp_file):
        """Test saving model and artifacts."""
        # Arrange
        pipeline = Mock()
        accuracy = 0.85
        y_pred = np.array([0, 1, 2, 3])
        y_test = np.array([0, 1, 2, 3])
        model_path = temp_file
        use_text = True
        
        # Act
        result = save_model_and_artifacts(
            pipeline, accuracy, y_pred, y_test, model_path, use_text
        )
        
        # Assert
        assert result is True
        assert os.path.exists(model_path)


class TestIntegration:
    """Integration tests."""
    
    @patch('scripts.train_text_classifier.mlflow')
    def test_full_training_pipeline(self, mock_mlflow, sample_dataset, temp_file):
        """Test complete training pipeline."""
        # Arrange
        sample_dataset.to_csv(temp_file, index=False)
        
        # Act
        X_train, X_test, y_train, y_test, use_text = load_and_prepare_data(temp_file)
        pipeline, accuracy, y_pred = train_text_classifier(
            X_train, y_train, X_test, y_test, use_text
        )
        
        # Assert
        assert pipeline is not None
        assert isinstance(accuracy, float)
        assert len(y_pred) == len(y_test) 