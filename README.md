# AWS MLflow Project

[![Python](https://img.shields.io/badge/Python-3.10-blue.svg)](https://www.python.org/downloads/)
[![MLflow](https://img.shields.io/badge/MLflow-2.8.1-orange.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Complete machine learning system using MLflow, AWS Lambda and S3 for production model deployment.

## 🚀 Quick Start

### Option 1: Demo with AG News dataset

```bash
# 1. Clone and setup
git clone https://github.com/sergeytkhorevskiy/aws-mlflow.git
cd aws-mlflow

# 2. Quick start (Linux/Mac)
./setup.sh quick-start

# 2. Quick start (Windows)
bash setup.sh quick-start

# 3. Start services
docker-compose up -d
```

### Option 2: Step-by-step setup

```bash
# 1. Clone and setup
git clone https://github.com/sergeytkhorevskiy/aws-mlflow.git
cd aws-mlflow

# 2. Quick setup (Linux/Mac)
./setup.sh install

# 2. Quick setup (Windows)
bash setup.sh install

# 3. Generate AG News dataset
python scripts/generate_ag_news_dataset.py

# 4. Start full stack
docker-compose up -d
```

## 🎯 Features

- **MLflow Tracking**: Experiment and metric tracking
- **AWS Lambda**: Serverless model inference  
- **S3 Integration**: Data and model storage
- **Docker Support**: Containerization for deployment
- **Monitoring**: Prometheus + Grafana for monitoring
- **Testing**: Complete test coverage
- **Security**: Security checks and best practices
- **Text Classification**: AG News dataset and text models
- **TF-IDF Pipeline**: Advanced text processing

## 📋 Requirements

- Python 3.10+
- Docker & Docker Compose
- AWS CLI configured
- Git
- Bash (for setup script)

## 🛠 Installation

### Automatic installation

```bash
# Linux/Mac
./setup.sh install

# Windows
bash setup.sh install
```

### Manual installation

```bash
# 1. Clone
git clone https://github.com/sergeytkhorevskiy/aws-mlflow.git
cd aws-mlflow

# 2. Environment setup
cp env.example .env
# Edit .env file

# 3. Install dependencies
pip install -r requirements.txt

# 4. AWS setup
aws configure
```

## 📊 AG News Dataset

The project includes generation of AG News-style dataset for text classification:

### News categories:
- **World** (0): International news and politics
- **Sports** (1): Sports news and events
- **Business** (2): Business and economic news
- **Sci/Tech** (3): Science and technology

### Dataset features:
- 1000 balanced samples (250 per category)
- Realistic headlines and news content
- Automatic generation of text features
- Support for both text and numerical features

### Usage:
```bash
# Generate dataset
python scripts/generate_ag_news_dataset.py

# Train text classifier
python scripts/train_text_classifier.py
```

## 🏗 Architecture

```
aws-mlflow/
├── data/                    # Datasets and data
├── lambda/                  # AWS Lambda functions
│   └── inference_lambda.py  # Lambda for inference
├── mlflow/                  # MLflow configuration
├── monitoring/              # Prometheus and Grafana configuration
│   ├── prometheus.yml       # Prometheus configuration
│   └── grafana/             # Dashboards and data sources
├── scripts/                 # Training and upload scripts
│   ├── generate_ag_news_dataset.py    # AG News dataset generation
│   ├── train_text_classifier.py       # Text classifier training
│   ├── upload_ag_news_to_s3.py        # Data upload to S3
│   └── monitoring.py                   # Monitoring system
├── tests/                   # Unit and integration tests
│   ├── conftest.py          # pytest configuration
│   └── test_train_text_classifier.py  # Training tests
├── docker-compose.yml       # Docker configuration
├── Dockerfile               # Docker image
├── setup.sh                 # Setup and utility script
├── requirements.txt         # Python dependencies
├── env.example              # Environment variables example
└── README.md               # Documentation
```

## 🚀 Usage

### Setup Script (Recommended)

```bash
# Show all available commands
./setup.sh help

# Quick start
./setup.sh quick-start

# Install dependencies
./setup.sh install

# Run tests
./setup.sh test

# Format code
./setup.sh format

# Security checks
./setup.sh security
```

### Direct Commands

```bash
# Install dependencies
pip install -r requirements.txt                    # Production dependencies
pip install -r requirements.txt && pip install bandit safety black isort  # Development dependencies

# Run tests
pytest tests/ -v --cov=scripts --cov=lambda --cov-report=html

# Code quality check
flake8 scripts/ lambda/ tests/
black --check scripts/ lambda/ tests/
isort --check-only scripts/ lambda/ tests/

# Code formatting
black scripts/ lambda/ tests/
isort scripts/ lambda/ tests/

# Security check
bandit -r scripts/ lambda/
safety check

# Data operations
python scripts/generate_ag_news_dataset.py        # Generate AG News dataset
python scripts/upload_ag_news_to_s3.py bucket-name file.csv  # Upload data to S3

# Model training
python scripts/train_text_classifier.py           # Train text classifier

# Clean temporary files
find . -type f -name "*.pyc" -delete
find . -type d -name "__pycache__" -delete
find . -type d -name "*.egg-info" -exec rm -rf {} +
rm -rf .pytest_cache/ htmlcov/ .coverage

# Build Docker image
docker build -t aws-mlflow .

# Start MLflow
docker-compose up mlflow -d

# Model management
python scripts/manage_models.py                   # Manage MLflow models
```

### Start full stack

```bash
# Start all services
docker-compose up -d

# Available services:
# - MLflow UI: http://localhost:5000
# - Prometheus: http://localhost:9090
# - Grafana: http://localhost:3000 (admin/admin)
```

### Train model

```bash
# Locally
python scripts/train_text_classifier.py

# Via Docker
docker-compose up train
```

### Upload data to S3

```bash
# Via script
python scripts/upload_ag_news_to_s3.py your-bucket-name data/ag_news_dataset.csv
```

## 📊 Monitoring

### Prometheus metrics

- **Model**: `model_predictions_total`, `model_accuracy`, `model_training_seconds`
- **S3**: `s3_operations_total`
- **Lambda**: `lambda_invocations_total`

### Logging

Structured logging with `structlog`:

```python
import structlog
logger = structlog.get_logger()
logger.info("Training started", model_type="RandomForest", dataset_size=1000)
```

### Grafana dashboards

Pre-installed dashboards for:
- Model performance
- S3 operations
- Lambda metrics
- System resources

## 🧪 Testing

### Run tests

```bash
# All tests
pytest tests/ -v --cov=scripts --cov=lambda --cov-report=html

# Unit tests only
pytest tests/ -v

# With coverage
pytest --cov --cov-report=html

# Security tests
bandit -r scripts/ lambda/
safety check
```

### Test structure

```
tests/
├── conftest.py                    # pytest fixtures
├── test_train_text_classifier.py  # Unit tests for training
└── __init__.py                    # Test package
```

## 🔒 Security

### Automatic checks

```bash
# Dependency check
safety check

# Static analysis
bandit -r scripts/ lambda/
```

### Environment variables

All sensitive data in `.env`:
- AWS credentials
- API keys
- Encryption secrets

## 🔄 CI/CD Pipeline

### GitHub Actions

Automated pipeline includes:

1. **Testing**: Unit and integration tests
2. **Code quality**: Linting, formatting
3. **Security**: Dependency checks
4. **Build**: Docker images
5. **Deployment**: AWS Lambda and ECR

### Local check

```bash
# Full CI pipeline
flake8 scripts/ lambda/ tests/ && black --check scripts/ lambda/ tests/ && isort --check-only scripts/ lambda/ tests/ && pytest tests/ -v && bandit -r scripts/ lambda/ && safety check
```

## 📈 Deployment

### AWS Lambda

```bash
# Create deployment package
zip -r lambda-deployment.zip lambda/

# Deploy
aws lambda update-function-code \
  --function-name mlflow-inference \
  --zip-file fileb://lambda-deployment.zip
```

### Docker

```bash
# Build and run
docker build -t aws-mlflow .
docker run -p 8000:8000 aws-mlflow
```

### Production

```bash
# Build for production
docker build -t aws-mlflow:latest .

# Deploy to ECR
docker push your-registry/aws-mlflow:latest
```

## 🆘 Troubleshooting

### Common issues

1. **MLflow won't start**
   ```bash
   docker-compose logs mlflow
   ```

2. **AWS credentials errors**
   ```bash
   aws configure
   ```

3. **Test issues**
   ```bash
   find . -type f -name "*.pyc" -delete
   find . -type d -name "__pycache__" -delete
   pip install -r requirements.txt
   pytest tests/ -v
   ```

### Logs

```bash
# View logs
docker-compose logs -f

# Specific service logs
docker-compose logs -f train
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/amazing-feature`
3. Commit changes: `git commit -m 'Add amazing feature'`
4. Push to branch: `git push origin feature/amazing-feature`
5. Open Pull Request

## 📚 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/)
- [AWS Lambda Documentation](https://docs.aws.amazon.com/lambda/)
- [Docker Documentation](https://docs.docker.com/)
- [Prometheus Documentation](https://prometheus.io/docs/)
- [Grafana Documentation](https://grafana.com/docs/)

## 📝 License

MIT License - see [LICENSE](LICENSE) file for details.

---

**Version**: 1.0.0  
**Last updated**: 2024  
**Supported versions**: Python 3.10+, MLflow 2.8.1+