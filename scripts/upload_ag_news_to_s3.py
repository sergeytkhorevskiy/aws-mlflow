#!/usr/bin/env python3
"""
Script to generate AG News dataset and upload it to S3.
"""

import os
import sys
import subprocess
import boto3
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def run_command(cmd, description):
    """Run a command and return success status"""
    print(f"\n🔄 {description}")
    print(f"Command: {cmd}")
    
    try:
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"✅ {description} completed successfully")
            if result.stdout:
                print("Output:", result.stdout[-300:])  # Show last 300 chars
            return True
        else:
            print(f"❌ {description} failed")
            print("Error:", result.stderr)
            return False
    except Exception as e:
        print(f"❌ {description} failed with exception: {e}")
        return False

def upload_to_s3(file_path, bucket, key):
    """Upload file to S3"""
    try:
        # Get AWS credentials
        aws_access_key = os.environ.get("AWS_ACCESS_KEY_ID")
        aws_secret_key = os.environ.get("AWS_SECRET_ACCESS_KEY")
        aws_region = os.environ.get("AWS_DEFAULT_REGION", "us-east-1")
        
        if not all([aws_access_key, aws_secret_key, bucket]):
            print("❌ Missing AWS credentials or bucket name")
            return False
        
        # Create S3 client
        s3 = boto3.client(
            's3',
            aws_access_key_id=aws_access_key,
            aws_secret_access_key=aws_secret_key,
            region_name=aws_region
        )
        
        # Upload file
        print(f"📤 Uploading {file_path} to s3://{bucket}/{key}")
        s3.upload_file(file_path, bucket, key)
        
        # Get file size
        file_size = os.path.getsize(file_path)
        print(f"✅ Successfully uploaded {file_path} ({file_size} bytes) to s3://{bucket}/{key}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to upload to S3: {e}")
        return False

def main():
    """Main function to generate and upload AG News dataset"""
    print("=== AG News Dataset Generation and S3 Upload ===")
    
    # Get bucket name from environment
    bucket = os.environ.get("AWS_S3_BUCKET")
    if not bucket:
        print("❌ AWS_S3_BUCKET environment variable is not set")
        print("Please set it in your .env file or environment")
        return
    
    print(f"Using S3 bucket: {bucket}")
    
    # Step 1: Generate AG News dataset
    print("\n" + "="*50)
    print("STEP 1: Generating AG News Dataset")
    print("="*50)
    
    if not run_command("python scripts/generate_ag_news_dataset.py", "Generating AG News dataset"):
        print("❌ Failed to generate dataset. Exiting.")
        return
    
    # Step 2: Check if files were created
    ag_news_path = "data/ag_news_dataset.csv"
    simple_path = "data/dataset.csv"
    
    if not os.path.exists(ag_news_path):
        print(f"❌ AG News dataset not found at {ag_news_path}")
        return
    
    if not os.path.exists(simple_path):
        print(f"❌ Simple dataset not found at {simple_path}")
        return
    
    print(f"✅ Datasets created successfully")
    print(f"  - AG News dataset: {ag_news_path}")
    print(f"  - Simple dataset: {simple_path}")
    
    # Step 3: Upload AG News dataset to S3
    print("\n" + "="*50)
    print("STEP 2: Uploading AG News Dataset to S3")
    print("="*50)
    
    if not upload_to_s3(ag_news_path, bucket, "ag_news_dataset.csv"):
        print("❌ Failed to upload AG News dataset. Exiting.")
        return
    
    # Step 4: Upload simple dataset to S3 (for backward compatibility)
    print("\n" + "="*50)
    print("STEP 3: Uploading Simple Dataset to S3")
    print("="*50)
    
    if not upload_to_s3(simple_path, bucket, "dataset.csv"):
        print("❌ Failed to upload simple dataset. Exiting.")
        return
    
    # Step 5: Summary
    print("\n" + "="*50)
    print("SUMMARY")
    print("="*50)
    
    print("🎉 Successfully generated and uploaded datasets to S3!")
    print(f"\n📁 Files uploaded to s3://{bucket}/:")
    print(f"  - ag_news_dataset.csv (Full AG News dataset with text)")
    print(f"  - dataset.csv (Simplified dataset for basic ML)")
    
    print(f"\n🚀 Next steps:")
    print(f"1. Set DATASET_KEY=ag_news_dataset.csv in your environment")
    print(f"2. Run: make train-text")
    print(f"3. Or run: docker-compose run --rm train python scripts/train_text_classifier.py")
    
    print(f"\n📊 Dataset information:")
    print(f"  - 4 categories: World, Sports, Business, Sci/Tech")
    print(f"  - 1000 samples (250 per category)")
    print(f"  - Text features for TF-IDF vectorization")
    print(f"  - Numerical features for traditional ML")

if __name__ == "__main__":
    main() 