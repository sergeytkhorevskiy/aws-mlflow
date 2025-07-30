"""Monitoring and logging utilities for the MLflow project."""

import os
import logging
import structlog
from prometheus_client import Counter, Histogram, Gauge, start_http_server
from typing import Dict, Any, Optional
import time
import functools

# Prometheus metrics
MODEL_PREDICTIONS = Counter('model_predictions_total', 'Total number of model predictions', ['model_name', 'status'])
MODEL_TRAINING_TIME = Histogram('model_training_seconds', 'Time spent training models', ['model_type'])
MODEL_ACCURACY = Gauge('model_accuracy', 'Model accuracy score', ['model_name', 'version'])
S3_OPERATIONS = Counter('s3_operations_total', 'Total number of S3 operations', ['operation', 'status'])
LAMBDA_INVOCATIONS = Counter('lambda_invocations_total', 'Total number of Lambda invocations', ['function_name', 'status'])

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()


class MetricsCollector:
    """Collector for application metrics."""
    
    def __init__(self, enable_prometheus: bool = True, prometheus_port: int = 9090):
        """Initialize metrics collector."""
        self.enable_prometheus = enable_prometheus
        self.prometheus_port = prometheus_port
        
        if self.enable_prometheus:
            try:
                start_http_server(self.prometheus_port)
                logger.info("Prometheus metrics server started", port=self.prometheus_port)
            except Exception as e:
                logger.error("Failed to start Prometheus server", error=str(e))
    
    def record_prediction(self, model_name: str, status: str = "success"):
        """Record a model prediction."""
        MODEL_PREDICTIONS.labels(model_name=model_name, status=status).inc()
        logger.info("Model prediction recorded", model_name=model_name, status=status)
    
    def record_training_time(self, model_type: str, training_time: float):
        """Record model training time."""
        MODEL_TRAINING_TIME.labels(model_type=model_type).observe(training_time)
        logger.info("Training time recorded", model_type=model_type, time_seconds=training_time)
    
    def record_accuracy(self, model_name: str, version: str, accuracy: float):
        """Record model accuracy."""
        MODEL_ACCURACY.labels(model_name=model_name, version=version).set(accuracy)
        logger.info("Model accuracy recorded", model_name=model_name, version=version, accuracy=accuracy)
    
    def record_s3_operation(self, operation: str, status: str = "success"):
        """Record S3 operation."""
        S3_OPERATIONS.labels(operation=operation, status=status).inc()
        logger.info("S3 operation recorded", operation=operation, status=status)
    
    def record_lambda_invocation(self, function_name: str, status: str = "success"):
        """Record Lambda invocation."""
        LAMBDA_INVOCATIONS.labels(function_name=function_name, status=status).inc()
        logger.info("Lambda invocation recorded", function_name=function_name, status=status)


def timing_decorator(func):
    """Decorator to measure function execution time."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.time()
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info("Function executed successfully", 
                       function=func.__name__, 
                       execution_time=execution_time)
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error("Function execution failed", 
                        function=func.__name__, 
                        execution_time=execution_time,
                        error=str(e))
            raise
    return wrapper


def log_function_call(func):
    """Decorator to log function calls with parameters."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger.info("Function called", 
                   function=func.__name__,
                   args_count=len(args),
                   kwargs_keys=list(kwargs.keys()))
        try:
            result = func(*args, **kwargs)
            logger.info("Function completed successfully", function=func.__name__)
            return result
        except Exception as e:
            logger.error("Function failed", function=func.__name__, error=str(e))
            raise
    return wrapper


class ModelMonitor:
    """Monitor for ML model performance and health."""
    
    def __init__(self, model_name: str, version: str):
        """Initialize model monitor."""
        self.model_name = model_name
        self.version = version
        self.metrics = MetricsCollector()
        self.logger = structlog.get_logger().bind(model_name=model_name, version=version)
    
    def log_training_start(self, model_type: str, dataset_size: int):
        """Log training start."""
        self.logger.info("Training started", 
                        model_type=model_type, 
                        dataset_size=dataset_size)
    
    def log_training_complete(self, training_time: float, accuracy: float):
        """Log training completion."""
        self.metrics.record_training_time(self.model_name, training_time)
        self.metrics.record_accuracy(self.model_name, self.version, accuracy)
        self.logger.info("Training completed", 
                        training_time=training_time, 
                        accuracy=accuracy)
    
    def log_prediction(self, features_count: int, prediction: Any, confidence: Optional[float] = None):
        """Log model prediction."""
        self.metrics.record_prediction(self.model_name)
        self.logger.info("Prediction made", 
                        features_count=features_count,
                        prediction=prediction,
                        confidence=confidence)
    
    def log_prediction_error(self, error: str):
        """Log prediction error."""
        self.metrics.record_prediction(self.model_name, status="error")
        self.logger.error("Prediction failed", error=error)


class S3Monitor:
    """Monitor for S3 operations."""
    
    def __init__(self):
        """Initialize S3 monitor."""
        self.metrics = MetricsCollector()
        self.logger = structlog.get_logger().bind(component="s3")
    
    def log_upload(self, bucket: str, key: str, file_size: int):
        """Log S3 upload."""
        self.metrics.record_s3_operation("upload")
        self.logger.info("File uploaded to S3", 
                        bucket=bucket, 
                        key=key, 
                        file_size=file_size)
    
    def log_download(self, bucket: str, key: str, file_size: int):
        """Log S3 download."""
        self.metrics.record_s3_operation("download")
        self.logger.info("File downloaded from S3", 
                        bucket=bucket, 
                        key=key, 
                        file_size=file_size)
    
    def log_operation_error(self, operation: str, error: str):
        """Log S3 operation error."""
        self.metrics.record_s3_operation(operation, status="error")
        self.logger.error("S3 operation failed", operation=operation, error=error)


class LambdaMonitor:
    """Monitor for AWS Lambda functions."""
    
    def __init__(self, function_name: str):
        """Initialize Lambda monitor."""
        self.function_name = function_name
        self.metrics = MetricsCollector()
        self.logger = structlog.get_logger().bind(function_name=function_name)
    
    def log_invocation_start(self, event_size: int):
        """Log Lambda invocation start."""
        self.logger.info("Lambda invocation started", event_size=event_size)
    
    def log_invocation_complete(self, execution_time: float, memory_used: int):
        """Log Lambda invocation completion."""
        self.metrics.record_lambda_invocation(self.function_name)
        self.logger.info("Lambda invocation completed", 
                        execution_time=execution_time,
                        memory_used=memory_used)
    
    def log_invocation_error(self, error: str, execution_time: float):
        """Log Lambda invocation error."""
        self.metrics.record_lambda_invocation(self.function_name, status="error")
        self.logger.error("Lambda invocation failed", 
                         error=error,
                         execution_time=execution_time)


# Global instances
metrics_collector = MetricsCollector(
    enable_prometheus=os.getenv('ENABLE_METRICS', 'true').lower() == 'true',
    prometheus_port=int(os.getenv('PROMETHEUS_PORT', '9091'))  # Changed from 9090 to 9091
) 