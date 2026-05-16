# Bonus Step: DVC with S3 Storage

Advanced ML pipeline with Great Expectations validation and S3 remote storage using MinIO.

## Prerequisites

- Docker and Docker Compose installed
- Python 3.11+
- Free ports: 9000 (MinIO API), 9001 (MinIO Console UI)

## Quick Start

### 1. Install Dependencies
```bash
make install
```

### 2. Start S3 Storage
```bash
make s3-up
```
This will start MinIO S3-compatible storage with web UI.

### 3. Setup DVC with S3
```bash
make setup-s3
```
This will:
- Create DVC bucket in MinIO
- Configure DVC remote to use S3 storage
- Set up authentication

### 4. Run Pipeline
```bash
make repro
```

### 5. Push Data to S3
```bash
dvc push
```
Your data and models will be stored in S3!

### 6. Pull Data from S3
```bash
dvc pull
```

## S3 Web Interface

Access MinIO Console at: **http://localhost:9001**
- Username: `minioadmin`
- Password: `minioadmin123`

You can browse, upload, and manage files through the web interface.

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Local Files   │───▶│   DVC Pipeline   │───▶│   MinIO S3      │
│   data/         │    │   validation     │    │   s3://bucket   │
│   models/       │    │   training       │    │   Remote Storage │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Features

- **S3 Remote Storage**: Data and models stored in cloud-compatible S3
- **Web Interface**: Browse and manage files via MinIO Console
- **Data Validation**: Great Expectations quality gates
- **Version Control**: Track data and model versions with DVC
- **Collaborative**: Share data through S3 between team members
- **Local Development**: Run everything locally with Docker

## Management Commands

```bash
# S3 Management
make s3-up        # Start MinIO S3 storage
make s3-down      # Stop MinIO storage
make s3-status    # Check S3 status

# DVC Operations
make setup-s3     # Configure DVC with S3
dvc push         # Upload data/models to S3
dvc pull         # Download data/models from S3
dvc status       # Check DVC status

# Pipeline
make repro       # Run ML pipeline
make clean       # Clean local artifacts
```

## Use Cases

- **Team Collaboration**: Share datasets and models via S3
- **CI/CD Integration**: Automated pipelines with remote storage
- **Data Backup**: Persistent storage for important datasets
- **Scalability**: Handle large files without Git repository bloat

## Cleanup

Stop and remove all containers:
```bash
make s3-down
docker volume prune  # Remove MinIO data
```
