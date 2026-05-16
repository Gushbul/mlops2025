#!/usr/bin/env python3
"""Create S3 bucket in MinIO for DVC storage."""

import boto3
from botocore.exceptions import ClientError
import time
import sys


def wait_for_minio(max_attempts=30):
    """Wait for MinIO to be ready."""
    for attempt in range(max_attempts):
        try:
            client = boto3.client(
                "s3",
                endpoint_url="http://localhost:9000",
                aws_access_key_id="minioadmin",
                aws_secret_access_key="minioadmin123",
            )
            client.list_buckets()
            print("✅ MinIO is ready!")
            return client
        except Exception:
            if attempt < max_attempts - 1:
                print(f"⏳ Waiting for MinIO... (attempt {attempt + 1}/{max_attempts})")
                time.sleep(2)
            else:
                print("❌ MinIO is not responding after 60 seconds")
                sys.exit(1)


def create_bucket(client, bucket_name="dvc-storage"):
    """Create DVC storage bucket."""
    try:
        client.head_bucket(Bucket=bucket_name)
        print(f"✅ Bucket '{bucket_name}' already exists")
        return True
    except ClientError as e:
        if e.response["Error"]["Code"] == "404":
            # Bucket doesn't exist, create it
            try:
                client.create_bucket(Bucket=bucket_name)
                print(f"✅ Created bucket '{bucket_name}'")
                return True
            except ClientError as create_error:
                print(f"❌ Failed to create bucket: {create_error}")
                return False
        else:
            print(f"❌ Error checking bucket: {e}")
            return False


def main():
    print("🚀 Setting up S3 bucket for DVC...")
    client = wait_for_minio()

    if create_bucket(client):
        print("\n🎉 S3 setup complete!")
        print("You can now use:")
        print("  make setup-s3  # Configure DVC remote")
        print("  dvc push       # Push data to S3")
        print("  dvc pull       # Pull data from S3")
    else:
        print("❌ Failed to setup S3 bucket")
        sys.exit(1)


if __name__ == "__main__":
    main()
