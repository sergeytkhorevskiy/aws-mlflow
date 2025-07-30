#!/bin/bash

# AWS MLflow Project Setup Script

set -e

echo "🚀 Setting up AWS MLflow Project..."

# Function to show help
show_help() {
    echo "Usage: $0 [COMMAND]"
    echo ""
    echo "Commands:"
    echo "  install     - Install dependencies"
    echo "  install-dev - Install development dependencies"
    echo "  test        - Run tests"
    echo "  lint        - Run linting"
    echo "  format      - Format code"
    echo "  security    - Run security checks"
    echo "  clean       - Clean temporary files"
    echo "  data        - Generate AG News dataset"
    echo "  train       - Train model"
    echo "  quick-start - Quick start setup"
    echo "  help        - Show this help"
}

# Function to install dependencies
install_deps() {
    echo "📦 Installing dependencies..."
    pip install -r requirements.txt
}

# Function to install development dependencies
install_dev_deps() {
    echo "📦 Installing development dependencies..."
    pip install -r requirements.txt
    pip install bandit safety black isort
}

# Function to run tests
run_tests() {
    echo "🧪 Running tests..."
    pytest tests/ -v --cov=scripts --cov=lambda --cov-report=html
}

# Function to run linting
run_lint() {
    echo "🔍 Running linting..."
    flake8 scripts/ lambda/ tests/
    black --check scripts/ lambda/ tests/
    isort --check-only scripts/ lambda/ tests/
}

# Function to format code
format_code() {
    echo "🎨 Formatting code..."
    black scripts/ lambda/ tests/
    isort scripts/ lambda/ tests/
}

# Function to run security checks
run_security() {
    echo "🔒 Running security checks..."
    bandit -r scripts/ lambda/
    safety check
}

# Function to clean temporary files
clean_files() {
    echo "🧹 Cleaning temporary files..."
    find . -type f -name "*.pyc" -delete
    find . -type d -name "__pycache__" -delete
    find . -type d -name "*.egg-info" -exec rm -rf {} +
    rm -rf .pytest_cache/ htmlcov/ .coverage
}

# Function to generate dataset
generate_data() {
    echo "📊 Generating AG News dataset..."
    python scripts/generate_ag_news_dataset.py
}

# Function to train model
train_model() {
    echo "🤖 Training model..."
    python scripts/train_text_classifier.py
}

# Function for quick start
quick_start() {
    echo "⚡ Quick start setup..."
    cp env.example .env
    install_deps
    generate_data
    echo "✅ Quick start complete! Run 'docker-compose up -d' to start services."
}

# Main script logic
case "${1:-help}" in
    install)
        install_deps
        ;;
    install-dev)
        install_dev_deps
        ;;
    test)
        run_tests
        ;;
    lint)
        run_lint
        ;;
    format)
        format_code
        ;;
    security)
        run_security
        ;;
    clean)
        clean_files
        ;;
    data)
        generate_data
        ;;
    train)
        train_model
        ;;
    quick-start)
        quick_start
        ;;
    help|*)
        show_help
        ;;
esac 