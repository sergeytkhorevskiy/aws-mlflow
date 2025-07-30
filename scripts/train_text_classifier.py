#!/usr/bin/env python3
"""
Train text classification model on AG News style dataset.
Supports both traditional ML and transformer-based models.
"""

import os
import boto3
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import mlflow.pytorch
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
import time
from dotenv import load_dotenv
import pickle
import json
import re

# Load environment variables
load_dotenv()

# Import monitoring
try:
    from scripts.monitoring import ModelMonitor, S3Monitor, log_function_call, timing_decorator
except ImportError:
    # Fallback for when scripts is not in Python path
    import sys
    import os
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from scripts.monitoring import ModelMonitor, S3Monitor, log_function_call, timing_decorator

# Initialize monitoring
model_monitor = ModelMonitor("text_classifier", "1.0.0")
s3_monitor = S3Monitor()

# Category mapping
CATEGORIES = {
    0: "World",
    1: "Sports", 
    2: "Business",
    3: "Sci/Tech"
}

def preprocess_text(text):
    """Preprocess text for better classification."""
    if not isinstance(text, str):
        return ""
    
    # Convert to lowercase
    text = text.lower()
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove special characters but keep basic punctuation
    text = re.sub(r'[^a-zA-Z0-9\s.,!?;:]', '', text)
    
    # Remove numbers (optional - uncomment if needed)
    # text = re.sub(r'\d+', '', text)
    
    return text.strip()

@log_function_call
@timing_decorator
def download_from_s3(bucket, key, dest):
    """Download file from S3 with monitoring and error handling."""
    try:
        s3 = boto3.client('s3')
        s3.download_file(bucket, key, dest)
        
        # Get file size for monitoring
        file_size = os.path.getsize(dest) if os.path.exists(dest) else 0
        s3_monitor.log_download(bucket, key, file_size)
        print(f"Downloaded {key} from s3://{bucket}")
        
    except Exception as e:
        s3_monitor.log_operation_error("download", str(e))
        raise

@log_function_call
@timing_decorator
def load_and_prepare_data(file_path):
    """Load and prepare text data for classification."""
    try:
        # Load data
        df = pd.read_csv(file_path)
        print(f"Loaded dataset with shape: {df.shape}")
        
        # Check if we have text data or need to use features
        if 'text' in df.columns:
            # Use text data for TF-IDF with preprocessing
            X = df['text'].apply(preprocess_text)
            y = df['category']
            use_text = True
            print("Using text data for TF-IDF vectorization with preprocessing")
        else:
            # Use feature columns
            feature_cols = [col for col in df.columns if col.startswith('feature_')]
            X = df[feature_cols]
            y = df['target']
            use_text = False
            print(f"Using {len(feature_cols)} feature columns")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")
        print(f"Class distribution: {y.value_counts().to_dict()}")
        
        return X_train, X_test, y_train, y_test, use_text
        
    except Exception as e:
        print(f"Error loading data: {e}")
        raise

@log_function_call
@timing_decorator
def train_text_classifier(X_train, y_train, X_test, y_test, use_text=True):
    """Train text classification model."""
    start_time = time.time()
    
    try:
        # Log training start
        model_monitor.log_training_start("TextClassifier", len(X_train))
        
        if use_text:
            # Create pipeline with TF-IDF and Random Forest
            pipeline = Pipeline([
                                 ('tfidf', TfidfVectorizer(
                     max_features=10000,  # Increase number of features
                     ngram_range=(1, 3),  # Add trigrams
                     stop_words='english',
                     min_df=2,            # Minimum word frequency
                     max_df=0.95,         # Maximum word frequency
                     lowercase=True,      # Convert to lowercase
                     strip_accents='unicode'  # Remove accents
                 )),
                 ('classifier', RandomForestClassifier(
                     n_estimators=200,    # Increase number of trees
                     random_state=42,
                     n_jobs=-1,
                     max_depth=20,        # Limit depth
                     min_samples_split=5  # Minimum samples for split
                 ))
            ])
        else:
            # Use Random Forest directly on features
            pipeline = RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                n_jobs=-1
            )
        
        # Train model
        print("Training text classifier...")
        pipeline.fit(X_train, y_train)
        
        # Make predictions
        y_pred = pipeline.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        # Calculate training time
        training_time = time.time() - start_time
        
        # Log training completion
        model_monitor.log_training_complete(training_time, accuracy)
        
        # Print detailed results
        print(f"\nModel Performance:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} seconds")
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=list(CATEGORIES.values())))
        
        # Confusion matrix
        print(f"\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return pipeline, accuracy, y_pred, y_test
        
    except Exception as e:
        training_time = time.time() - start_time
        model_monitor.log_prediction_error(str(e))
        raise

def save_model_and_artifacts(pipeline, accuracy, y_pred, y_test, model_path, use_text):
    """Save model and create artifacts for MLflow."""
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(pipeline, f)
    
    # Create prediction results
    results_df = pd.DataFrame({
        'true_label': y_test,
        'predicted_label': y_pred,
        'true_category': [CATEGORIES.get(y, 'Unknown') for y in y_test],
        'predicted_category': [CATEGORIES.get(y, 'Unknown') for y in y_pred],
        'correct': y_test == y_pred
    })
    
    results_path = "/tmp/prediction_results.csv"
    results_df.to_csv(results_path, index=False)
    
    # Create model info
    model_info = {
        "model_type": "TextClassifier",
        "use_text_features": use_text,
        "accuracy": accuracy,
        "n_classes": len(CATEGORIES),
        "categories": CATEGORIES,
        "model_path": model_path
    }
    
    info_path = "/tmp/model_info.json"
    with open(info_path, "w") as f:
        json.dump(model_info, f, indent=2)
    
    return results_path, info_path

def main():
    """Main training pipeline for text classification."""
    try:
        # Get environment variables
        bucket = os.environ.get("AWS_S3_BUCKET")
        dataset_key = os.environ.get("DATASET_KEY", "ag_news_dataset.csv")
        
        if not bucket:
            raise ValueError("AWS_S3_BUCKET environment variable is required")
        
        local_path = "/tmp/dataset.csv"
        
        # 1. Download dataset from S3
        download_from_s3(bucket, dataset_key, local_path)
        
        # 2. Load and prepare data
        X_train, X_test, y_train, y_test, use_text = load_and_prepare_data(local_path)
        
        # Load dataset for logging
        df = pd.read_csv(local_path)
        
        # 3. Train model
        pipeline, accuracy, y_pred, y_test = train_text_classifier(
            X_train, y_train, X_test, y_test, use_text
        )
        
        # 4. Save model and artifacts
        model_path = "/tmp/text_classifier.pkl"
        results_path, info_path = save_model_and_artifacts(
            pipeline, accuracy, y_pred, y_test, model_path, use_text
        )
        
        # 5. Log to MLflow
        # Configure MLflow to use local artifact store
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        artifact_root = os.environ.get("MLFLOW_ARTIFACT_ROOT", "./mlruns")
        
        # Set tracking URI
        mlflow.set_tracking_uri(tracking_uri)
        
        # Set experiment
        experiment_name = os.environ.get("MLFLOW_EXPERIMENT_NAME", "aws-mlflow-experiment")
        mlflow.set_experiment(experiment_name)
        
        # Use local artifact store with explicit path
        with mlflow.start_run(artifact_location=artifact_root):
            mlflow.log_param("model_type", "TextClassifier")
            mlflow.log_param("use_text_features", use_text)
            mlflow.log_param("n_estimators", 200 if use_text else 100)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("dataset_source", f"s3://{bucket}/{dataset_key}")
            mlflow.log_param("dataset_shape", f"{df.shape[0]}x{df.shape[1]}")
            mlflow.log_param("dataset_columns", list(df.columns))
            mlflow.log_metric("accuracy", accuracy)
            mlflow.log_metric("training_samples", len(X_train))
            mlflow.log_metric("test_samples", len(X_test))
            mlflow.log_metric("n_classes", len(CATEGORIES))
            
            # Log the model file as an artifact
            print(f"Logging model artifact: {model_path}")
            mlflow.log_artifact(model_path, "models")
            print(f"Model artifact logged successfully")
            
            # Log prediction results
            print(f"Logging prediction results: {results_path}")
            mlflow.log_artifact(results_path, "results")
            print(f"Prediction results logged successfully")
            
            # Log model info
            print(f"Logging model info: {info_path}")
            mlflow.log_artifact(info_path, "model_info")
            print(f"Model info logged successfully")
            
            # Log dataset sample and full dataset
            try:
                # Log sample dataset
                sample_df = pd.read_csv(local_path).head(100)
                sample_path = "/tmp/dataset_sample.csv"
                sample_df.to_csv(sample_path, index=False)
                
                # Verify file exists and has content
                if os.path.exists(sample_path) and os.path.getsize(sample_path) > 0:
                    mlflow.log_artifact(sample_path, "dataset")
                    print(f"Dataset sample logged: {sample_path}")
                else:
                    print(f"Warning: Dataset sample file not created properly")
                
                # Log full dataset
                full_dataset_path = "/tmp/full_dataset.csv"
                df.to_csv(full_dataset_path, index=False)
                if os.path.exists(full_dataset_path) and os.path.getsize(full_dataset_path) > 0:
                    mlflow.log_artifact(full_dataset_path, "full_dataset")
                    print(f"Full dataset logged: {full_dataset_path}")
                    
            except Exception as e:
                print(f"Warning: Could not log dataset: {e}")
            
            # Register model in Model Registry
            model_name = "text-classifier"
            
            # Log model to MLflow Model Registry
            mlflow.sklearn.log_model(
                pipeline,
                "model",
                registered_model_name=model_name
            )
            
            # Update model description after registration
            try:
                client = mlflow.tracking.MlflowClient()
                client.update_registered_model(
                    name=model_name,
                    description=f"Text classification model with accuracy {accuracy:.4f}"
                )
            except Exception as e:
                print(f"Warning: Could not update model description: {e}")
            
            print(f"Training completed successfully! Model accuracy: {accuracy:.4f}")
            print(f"Model saved to: {model_path}")
            print(f"Model and artifacts logged to MLflow")
            print(f"Model registered as: {model_name}")
        
        # 6. Log prediction for monitoring
        sample_prediction = pipeline.predict(X_test[:1])[0]
        model_monitor.log_prediction(len(X_test.columns) if not use_text else 1, sample_prediction)
        
    except Exception as e:
        print(f"Training failed: {e}")
        raise

if __name__ == "__main__":
    main() 