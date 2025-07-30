"""Pytest configuration and fixtures."""

import pytest
import tempfile
import os
import pandas as pd
import numpy as np
from unittest.mock import Mock, patch


@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    np.random.seed(42)
    n_samples = 100
    
    # Generate sample text data
    categories = ['World', 'Sports', 'Business', 'Sci/Tech']
    texts = [
        f"Sample news article about {cat.lower()} {i}" 
        for i in range(n_samples) 
        for cat in categories
    ][:n_samples]
    
    # Generate categories
    categories_list = np.random.choice([0, 1, 2, 3], n_samples)
    
    # Generate features
    features = np.random.randn(n_samples, 10)
    feature_cols = [f'feature_{i}' for i in range(10)]
    
    df = pd.DataFrame({
        'text': texts,
        'category': categories_list,
        'target': categories_list
    })
    
    # Add feature columns
    for i, col in enumerate(feature_cols):
        df[col] = features[:, i]
    
    return df


@pytest.fixture
def temp_file():
    """Create a temporary file."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        yield f.name
    os.unlink(f.name)


@pytest.fixture
def mock_s3_client():
    """Mock S3 client for testing."""
    with patch('boto3.client') as mock_client:
        mock_s3 = Mock()
        mock_client.return_value = mock_s3
        yield mock_s3


@pytest.fixture
def mock_mlflow():
    """Mock MLflow for testing."""
    with patch('mlflow.start_run') as mock_start_run, \
         patch('mlflow.log_metric') as mock_log_metric, \
         patch('mlflow.sklearn.log_model') as mock_log_model:
        
        mock_run = Mock()
        mock_start_run.return_value.__enter__.return_value = mock_run
        
        yield {
            'start_run': mock_start_run,
            'log_metric': mock_log_metric,
            'log_model': mock_log_model,
            'run': mock_run
        }


@pytest.fixture
def mock_environment():
    """Mock environment variables."""
    env_vars = {
        'AWS_ACCESS_KEY_ID': 'test_key',
        'AWS_SECRET_ACCESS_KEY': 'test_secret',
        'AWS_DEFAULT_REGION': 'us-east-1',
        'AWS_S3_BUCKET': 'test-bucket',
        'MLFLOW_TRACKING_URI': 'http://localhost:5000',
        'MLFLOW_EXPERIMENT_NAME': 'test-experiment'
    }
    
    with patch.dict(os.environ, env_vars):
        yield env_vars 