#!/usr/bin/env python3
"""
Manage registered models in MLflow Model Registry.
"""

import os
import mlflow
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def list_registered_models():
    """List all registered models."""
    try:
        client = mlflow.tracking.MlflowClient()
        models = client.list_registered_models()
        
        print("Registered Models:")
        print("=" * 50)
        
        for model in models:
            print(f"Name: {model.name}")
            print(f"Latest Version: {model.latest_versions}")
            print(f"Creation Timestamp: {model.creation_timestamp}")
            print(f"Last Updated: {model.last_updated_timestamp}")
            print(f"Description: {model.description}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error listing models: {e}")

def get_model_versions(model_name):
    """Get all versions of a specific model."""
    try:
        client = mlflow.tracking.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        
        print(f"Versions for model '{model_name}':")
        print("=" * 50)
        
        for version in versions:
            print(f"Version: {version.version}")
            print(f"Status: {version.status}")
            print(f"Run ID: {version.run_id}")
            print(f"Creation Timestamp: {version.creation_timestamp}")
            print(f"Last Updated: {version.last_updated_timestamp}")
            print("-" * 30)
            
    except Exception as e:
        print(f"Error getting model versions: {e}")

def transition_model_stage(model_name, version, stage):
    """Transition model to a specific stage."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Transition model to stage
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        
        print(f"Model '{model_name}' version {version} transitioned to '{stage}'")
        
    except Exception as e:
        print(f"Error transitioning model: {e}")

def load_model_for_inference(model_name, stage="Production"):
    """Load a model for inference."""
    try:
        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        print(f"Model '{model_name}' loaded from stage '{stage}'")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def delete_model_version(model_name, version):
    """Delete a specific model version."""
    try:
        client = mlflow.tracking.MlflowClient()
        
        # Delete model version
        client.delete_model_version(
            name=model_name,
            version=version
        )
        
        print(f"Model '{model_name}' version {version} deleted")
        
    except Exception as e:
        print(f"Error deleting model version: {e}")

def main():
    """Main function for model management."""
    # Set MLflow tracking URI
    tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
    mlflow.set_tracking_uri(tracking_uri)
    
    print("MLflow Model Registry Management")
    print("=" * 40)
    
    while True:
        print("\nAvailable commands:")
        print("1. List all registered models")
        print("2. Get model versions")
        print("3. Transition model stage")
        print("4. Load model for inference")
        print("5. Delete model version")
        print("6. Exit")
        
        choice = input("\nEnter your choice (1-6): ")
        
        if choice == "1":
            list_registered_models()
            
        elif choice == "2":
            model_name = input("Enter model name: ")
            get_model_versions(model_name)
            
        elif choice == "3":
            model_name = input("Enter model name: ")
            version = int(input("Enter version number: "))
            stage = input("Enter stage (Staging/Production/Archived): ")
            transition_model_stage(model_name, version, stage)
            
        elif choice == "4":
            model_name = input("Enter model name: ")
            stage = input("Enter stage (default: Production): ") or "Production"
            model = load_model_for_inference(model_name, stage)
            if model:
                print("Model loaded successfully!")
                
        elif choice == "5":
            model_name = input("Enter model name: ")
            version = int(input("Enter version number: "))
            confirm = input(f"Are you sure you want to delete {model_name} version {version}? (y/n): ")
            if confirm.lower() == 'y':
                delete_model_version(model_name, version)
                
        elif choice == "6":
            print("Goodbye!")
            break
            
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main() 