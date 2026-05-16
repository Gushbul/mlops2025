.PHONY: install clean repro s3-up s3-down s3-status setup-s3

install:
	pip install -r requirements.txt

# S3 Management
s3-up:
	docker-compose up -d
	@echo "Waiting for MinIO to start..."
	sleep 10
	@echo "✅ MinIO S3 is running:"
	@echo "   API: http://localhost:9000"
	@echo "   UI:  http://localhost:9001"
	@echo "   Login: minioadmin / minioadmin123"

s3-down:
	docker-compose down

s3-status:
	docker-compose ps
	@echo ""
	@echo "MinIO endpoints:"
	@echo "  API: http://localhost:9000"
	@echo "  UI:  http://localhost:9001"

setup-s3:
	@echo "🔧 Setting up DVC with S3 remote..."
	python create_bucket.py
	dvc init --no-scm || true
	dvc remote remove s3 || true
	dvc remote add -d s3 s3://dvc-storage
	dvc remote modify s3 endpointurl http://localhost:9000
	dvc remote modify s3 access_key_id minioadmin
	dvc remote modify s3 secret_access_key minioadmin123
	@echo "✅ DVC S3 remote configured"
	@echo "✅ Ready to use: dvc push, dvc pull"

repro:
	dvc repro

clean:
	rm -rf data/processed models metrics reports .dvc/cache
