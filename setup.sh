#!/bin/bash
set -e



echo "📂 Membuat struktur folder..."
mkdir -p app/api/routes
mkdir -p app/core
mkdir -p app/schemas
mkdir -p app/services
mkdir -p app/models
mkdir -p app/ml_models
mkdir -p app/static/outputs
mkdir -p app/utils
mkdir -p tests
mkdir -p scripts

echo "📄 Membuat file..."
# Root files
touch .env
touch .gitignore
touch requirements.txt
touch README.md

# app/
touch app/__init__.py
touch app/main.py
touch app/config.py

# app/api/
touch app/api/__init__.py
touch app/api/routes/__init__.py
touch app/api/routes/health.py
touch app/api/routes/predict.py

# app/core/
touch app/core/__init__.py
touch app/core/exceptions.py
touch app/core/logger.py

# app/schemas/
touch app/schemas/__init__.py
touch app/schemas/prediction.py

# app/services/
touch app/services/__init__.py
touch app/services/video_loader.py
touch app/services/model_loader.py
touch app/services/inference.py
touch app/services/grapher.py
touch app/services/frame_extractor.py

# app/models/
touch app/models/__init__.py
touch app/models/custom_layers.py

# app/utils/
touch app/utils/__init__.py
touch app/utils/image_utils.py
touch app/utils/file_utils.py

# tests/
touch tests/__init__.py
touch tests/test_health.py
touch tests/test_predict.py

# scripts/
touch scripts/download_dummy_model.py

# .gitkeep agar folder kosong tetap di-track Git
touch app/ml_models/.gitkeep
touch app/static/outputs/.gitkeep

echo ""
echo "✅ Struktur project berhasil dibuat!"
echo ""
echo "📋 Struktur:"
find . -type f -o -type d | sort | sed 's|[^/]*/|  |g'
