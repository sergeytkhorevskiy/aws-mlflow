import boto3
import pickle
import tempfile
import json
import os
import time
import sys
from typing import Dict, Any

# Add scripts directory to path for monitoring
sys.path.append('/opt/python/lib/python3.10/site-packages')
try:
    from scripts.monitoring import LambdaMonitor, log_function_call, timing_decorator
except ImportError:
    # Fallback for when monitoring is not available
    def log_function_call(func):
        return func
    
    def timing_decorator(func):
        return func
    
    class LambdaMonitor:
        def __init__(self, function_name):
            self.function_name = function_name
        
        def log_invocation_start(self, event_size):
            pass
        
        def log_invocation_complete(self, execution_time, memory_used):
            pass
        
        def log_invocation_error(self, error, execution_time):
            pass

# Initialize monitoring
lambda_monitor = LambdaMonitor("mlflow-inference")

# Try to import MLflow for model registry
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    print("MLflow not available, using S3 fallback")


@log_function_call
@timing_decorator
def download_model_from_s3(bucket: str, model_key: str) -> Any:
    """Download model from S3 with error handling."""
    try:
        s3 = boto3.client('s3')
        with tempfile.NamedTemporaryFile() as tmp:
            s3.download_fileobj(bucket, model_key, tmp)
            tmp.seek(0)
            model = pickle.load(tmp)
        return model
    except Exception as e:
        raise Exception(f"Failed to download model from S3: {str(e)}")

@log_function_call
@timing_decorator
def load_model_from_registry(model_name: str, stage: str = "Production") -> Any:
    """Load model from MLflow Model Registry."""
    try:
        if not MLFLOW_AVAILABLE:
            raise Exception("MLflow not available")
        
        # Set tracking URI
        tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
        mlflow.set_tracking_uri(tracking_uri)
        
        # Load model from registry
        model_uri = f"models:/{model_name}/{stage}"
        model = mlflow.sklearn.load_model(model_uri)
        
        return model
    except Exception as e:
        raise Exception(f"Failed to load model from registry: {str(e)}")

@log_function_call
@timing_decorator
def load_model(model_source: str = "s3") -> Any:
    """Load model from specified source (s3 or registry)."""
    try:
        if model_source == "registry":
            model_name = os.environ.get("MLFLOW_MODEL_NAME", "text-classifier")
            stage = os.environ.get("MLFLOW_MODEL_STAGE", "Production")
            return load_model_from_registry(model_name, stage)
        else:
            # Default to S3
            bucket = os.environ.get("AWS_S3_BUCKET")
            model_key = os.environ.get("MODEL_KEY", "models/text_classifier.pkl")
            return download_model_from_s3(bucket, model_key)
    except Exception as e:
        raise Exception(f"Failed to load model: {str(e)}")


@log_function_call
@timing_decorator
def make_prediction(model: Any, features: list) -> int:
    """Make prediction with error handling."""
    try:
        if not isinstance(features, list):
            raise ValueError("Features must be a list")
        
        if len(features) == 0:
            raise ValueError("Features list cannot be empty")
        
        result = model.predict([features])
        return int(result[0])
    except Exception as e:
        raise Exception(f"Prediction failed: {str(e)}")


def validate_input(event: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input event."""
    try:
        if 'body' not in event:
            raise ValueError("Missing 'body' in event")
        
        input_data = json.loads(event['body'])
        
        if 'features' not in input_data:
            raise ValueError("Missing 'features' in input data")
        
        if not isinstance(input_data['features'], list):
            raise ValueError("Features must be a list")
        
        return input_data
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in request body: {str(e)}")
    except Exception as e:
        raise ValueError(f"Input validation failed: {str(e)}")


def lambda_handler(event: Dict[str, Any], context: Any) -> Dict[str, Any]:
    """AWS Lambda handler with comprehensive monitoring and error handling."""
    start_time = time.time()
    
    try:
        # Log invocation start
        event_size = len(json.dumps(event)) if event else 0
        lambda_monitor.log_invocation_start(event_size)
        
        # Validate input
        input_data = validate_input(event)
        
        # Load model (try registry first, fallback to S3)
        model_source = os.environ.get("MODEL_SOURCE", "registry")
        model = load_model(model_source)
        
        # Make prediction
        features = input_data['features']
        prediction = make_prediction(model, features)
        
        # Calculate execution time
        execution_time = time.time() - start_time
        
        # Log successful invocation
        lambda_monitor.log_invocation_complete(execution_time, 0)  # Memory usage not available
        
        # Return response
        return {
            'statusCode': 200,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'prediction': prediction,
                'execution_time': execution_time,
                'model_source': model_source
            })
        }
        
    except Exception as e:
        execution_time = time.time() - start_time
        lambda_monitor.log_invocation_error(str(e), execution_time)
        
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'execution_time': execution_time
            })
        }