# 🎥 Shoplifting Detection API

REST API untuk deteksi **shoplifting (pencurian)** dari video menggunakan deep learning (EfficientNetV2B0 + BiLSTM + Temporal Attention).

## ✨ Fitur

- 🎯 **Klasifikasi video**: `Normal` atau `Shoplifting`
- 📊 **Confidence score** (0-1)
- 🧠 **Attention weights** per frame (32 frame) — untuk grafik
- 🖼️ **Top-3 frame** dengan bobot attention tertinggi (base64 PNG)
- 🎭 **Dummy mode** — bisa dites tanpa model asli
- 📚 **Swagger UI** otomatis di `/docs`
- 🐳 **Siap production** — logging, error handling, health check

---

## 📁 Struktur Project

```
fastapi-video-attention/
│
├── app/
│   ├── main.py                    # Entry point FastAPI
│   ├── config.py                  # Baca .env
│   │
│   ├── api/routes/
│   │   ├── health.py              # GET /health, /health/ready, /health/live
│   │   └── predict.py             # POST /predict
│   │
│   ├── core/
│   │   ├── logger.py              # Setup logging
│   │   └── exceptions.py          # Custom exception + handler
│   │
│   ├── schemas/
│   │   └── prediction.py          # Pydantic models (request/response)
│   │
│   ├── services/
│   │   ├── video_loader.py        # Load & sampling video
│   │   ├── model_loader.py        # Load model (dummy/real)
│   │   ├── inference.py           # Jalankan prediksi
│   │   └── frame_extractor.py     # Ambil top-K frame → base64
│   │
│   ├── models/
│   │   └── custom_layers.py       # SEBlock, TemporalAttention
│   │
│   ├── utils/
│   │   ├── file_utils.py          # Validasi & simpan file
│   │   └── image_utils.py         # Convert numpy → base64
│   │
│   ├── ml_models/                 # Taruh model .keras di sini
│   └── static/outputs/            # Output sementara (opsional)
│
├── tests/                         # Unit test
├── scripts/                       # Helper scripts
│
├── .env                           # Konfigurasi (JANGAN di-commit)
├── .env.example                   # Template .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## 🚀 Quick Start

### 1. Clone & Masuk Folder

```bash
git clone <repo-url>
cd fastapi-video-attention
```

### 2. Buat Virtual Environment

```bash
python -m venv venv

# Git Bash (Windows)
source venv/Scripts/activate

# Linux / macOS
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Setup `.env`

```bash
cp .env.example .env
```

Edit `.env` kalau perlu (default sudah oke untuk development).

### 5. Jalankan Server

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 6. Buka Dokumentasi

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc
- **Health**: http://localhost:8000/health

---

## 🎭 Mode: Dummy vs Production

### Dummy Mode (Default)

Di `.env`:
```env
USE_DUMMY_MODEL=True
```

- ✅ Tidak butuh TensorFlow
- ✅ Prediksi & attention **di-generate random** (tapi valid: sum = 1.0)
- ✅ Cocok untuk development & testing endpoint
- ⚠️ Hasil tidak nyata (cuma untuk demo)

### Production Mode

Saat model `.keras` sudah siap:

1. Uncomment TensorFlow di `requirements.txt`:
   ```txt
   tensorflow==2.17.0
   keras==3.5.0
   protobuf==4.25.3
   ```

2. Install ulang:
   ```bash
   pip install -r requirements.txt
   ```

3. Taruh model di `app/ml_models/best_model.keras`

4. Ubah `.env`:
   ```env
   USE_DUMMY_MODEL=False
   ```

5. Restart server

---

## 📡 API Endpoints

### 1. `GET /` — Info API

```bash
curl http://localhost:8000/
```

Response:
```json
{
  "app": "Shoplifting Detection API",
  "version": "0.1.0",
  "docs": "/docs",
  "health": "/health",
  "predict": "/predict"
}
```

---

### 2. `GET /health` — Status Aplikasi

```bash
curl http://localhost:8000/health
```

Response:
```json
{
  "status": "ok",
  "app_name": "Shoplifting Detection API",
  "app_version": "0.1.0",
  "environment": "development",
  "model_loaded": true,
  "model_mode": "dummy",
  "timestamp": "2025-01-15T10:23:45.123456"
}
```

---

### 3. `GET /health/ready` — Readiness Probe

```bash
curl http://localhost:8000/health/ready
```

- **200** kalau model sudah di-load
- **503** kalau belum

---

### 4. `GET /health/live` — Liveness Probe

```bash
curl http://localhost:8000/health/live
```

Selalu return 200 selama server hidup.

---

### 5. `POST /predict` — Prediksi Video ⭐

**Request:**
```bash
curl -X POST http://localhost:8000/predict \
  -F "video=@contoh.mp4"
```

**Batasan:**
- Format: `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`
- Maksimal: **50 MB**

**Response:**
```json
{
  "prediction": "Shoplifting",
  "confidence": 0.8734,
  "total_frames": 32,
  "attention_weights": [0.0125, 0.0421, ...],
  "top_frames": [
    {
      "rank": 1,
      "frame_index": 12,
      "weight": 0.1521,
      "timestamp_sec": 3.75,
      "image": "data:image/png;base64,..."
    }
  ],
  "video_metadata": {
    "filename": "contoh.mp4",
    "total_frames": 306,
    "fps": 30.0,
    "width": 1920,
    "height": 1080,
    "duration_sec": 10.2,
    "resolution": "1920x1080"
  },
  "prediction_metadata": {
    "processing_time_ms": 1842,
    "model_version": "efficientnetv2b0-bilstm-attention-v1",
    "mode": "production",
    "threshold": 0.5
  }
}
```

---

## 🎨 Integrasi Frontend (Chart.js)

`attention_weights` bisa langsung dirender jadi bar chart:

```javascript
const res = await fetch('/predict', {
  method: 'POST',
  body: formData  // FormData dengan field 'video'
});
const data = await res.json();

// Render chart
const dataScores = data.attention_weights.map(w => w * 100);  // ke persen
const labels = data.attention_weights.map((_, i) => `Frame ${i + 1}`);

// Tentukan warna: top-3 frame = hijau
const sorted = [...dataScores]
  .map((v, i) => ({ v, i }))
  .sort((a, b) => b.v - a.v);
const top3Index = new Set(sorted.slice(0, 3).map(x => x.i));

const backgroundColors = dataScores.map((_, i) =>
  top3Index.has(i) ? '#047857' : '#ef4444'
);

new Chart(ctx, {
  type: 'bar',
  data: {
    labels: labels,
    datasets: [{
      label: 'Bobot Attention (%)',
      data: dataScores,
      backgroundColor: backgroundColors,
    }]
  },
  options: {
    scales: {
      y: {
        beginAtZero: true,
        title: { display: true, text: 'Bobot Attention (%)' }
      }
    }
  }
});

// Render top frames
data.top_frames.forEach(f => {
  document.getElementById('frames').innerHTML += `
    <div>
      <img src="${f.image}" width="160">
      <p>Rank #${f.rank} | Frame ${f.frame_index + 1}</p>
      <p>Bobot: ${(f.weight * 100).toFixed(2)}%</p>
      <p>Timestamp: ${f.timestamp_sec}s</p>
    </div>
  `;
});
```

---

## ⚙️ Konfigurasi `.env`

| Variable | Default | Deskripsi |
|---|---|---|
| `APP_NAME` | Shoplifting Detection API | Nama aplikasi |
| `APP_VERSION` | 0.1.0 | Versi |
| `DEBUG` | True | Debug mode |
| `HOST` | 0.0.0.0 | Bind address |
| `PORT` | 8000 | Port server |
| `USE_DUMMY_MODEL` | True | Pakai dummy model? |
| `MODEL_PATH` | app/ml_models/best_model.keras | Path model |
| `FRAMES` | 32 | Jumlah frame sampling |
| `IMG_SIZE` | 160 | Ukuran resize frame |
| `CONFIDENCE_THRESHOLD` | 0.5 | Threshold klasifikasi |
| `MAX_UPLOAD_SIZE_MB` | 50 | Max ukuran upload |
| `TOP_K_FRAMES` | 3 | Jumlah top frame |
| `IMAGE_FORMAT` | PNG | Format gambar (PNG/JPEG) |
| `LOG_LEVEL` | INFO | Level logging |

---

## 🧪 Testing

### Test Manual via Swagger

1. Buka http://localhost:8000/docs
2. Cari **POST /predict**
3. Klik **Try it out**
4. Upload video
5. Klik **Execute**

### Test via curl

```bash
# Bikin video dummy
python -c "
import cv2, numpy as np
out = cv2.VideoWriter('test.mp4', cv2.VideoWriter_fourcc(*'mp4v'), 30, (320, 240))
for i in range(100):
    frame = np.full((240, 320, 3), i * 2 % 255, dtype=np.uint8)
    out.write(frame)
out.release()
"

# Upload
curl -X POST http://localhost:8000/predict -F "video=@test.mp4" | python -m json.tool
```

### Test Otomatis (pytest)

```bash
# Install dulu
pip install pytest httpx

# Jalankan
pytest tests/ -v
```

---

## 🐳 Docker (Opsional)

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# System deps untuk OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/
COPY .env .

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

### Build & Run

```bash
docker build -t shoplifting-api .
docker run -p 8000:8000 shoplifting-api
```

---

## 🔧 Troubleshooting

### 1. Error `ModuleNotFoundError: No module named 'app'`

**Solusi**: Jalankan dari root project, bukan dari dalam `app/`.

```bash
# ✅ Benar
cd fastapi-video-attention
uvicorn app.main:app --reload

# ❌ Salah
cd fastapi-video-attention/app
uvicorn main:app --reload
```

---

### 2. Error `Unknown layer: SEBlock`

**Solusi**: Pastikan `get_custom_layers()` dipanggil saat load model. Sudah otomatis di `model_loader.py`.

---

### 3. Video tidak bisa dibaca

**Penyebab**: Codec tidak didukung OpenCV.

**Solusi**: Convert dulu pakai FFmpeg:
```bash
ffmpeg -i input.mov -c:v libx264 -c:a aac output.mp4
```

---

### 4. Upload file besar gagal

**Solusi**: Naikkan `MAX_UPLOAD_SIZE_MB` di `.env`. Atau kompres video dulu.

---

### 5. Port 8000 sudah dipakai

**Solusi**: Ganti port:
```bash
uvicorn app.main:app --reload --port 8001
```

---

## 📊 Arsitektur Model

```
Input Video (32, 160, 160, 3)
    ↓
TimeDistributed(EfficientNetV2B0)       ← ekstrak fitur per frame
    ↓
TimeDistributed(SEBlock)                 ← channel attention
    ↓
TimeDistributed(GlobalAveragePooling2D)
    ↓
Bidirectional(LSTM(128))                 ← temporal modeling
    ↓
TemporalAttention                        ← frame attention
    ↓
[context_vector, attention_weights]
    ↓
Dropout(0.5) → Dense(128) → Dense(1) sigmoid
    ↓
Confidence score (0-1)
```

---

## 📝 License

MIT

---

## 👤 Author

Dibuat untuk keperluan deteksi shoplifting berbasis video.