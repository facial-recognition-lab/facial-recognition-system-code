# Facial Recognition System with Anti-Spoofing & Liveness Detection

[![Python](https://img.shields.io/badge/Python-3.10.11-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

> A comprehensive facial recognition system integrating MTCNN face detection, EfficientNet-B3 anti-spoofing, and ArcFace facial recognition for secure identity verification.

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [Requirements](#requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [API Documentation](#api-documentation)
- [Deployment](#deployment)
- [Performance Metrics](#performance-metrics)
- [Project Structure](#project-structure)
- [Team](#team)
- [License](#license)

---

## 🎯 Overview

This project implements a multi-layered facial recognition system with **AI-generated face detection** capabilities, designed for real-world security applications including access control, authentication platforms, and identity verification systems. The system not only recognizes faces but also determines if an image is of a **real person or AI-generated**.

**Live Demo**: 🌐 [https://ai-face-authenticity-9p0e.bolt.host/](https://ai-face-authenticity-9p0e.bolt.host/)

**GitHub Repository**: 📦 [https://github.com/facial-recognition-lab/facial-recognition-system-code](https://github.com/facial-recognition-lab/facial-recognition-system-code)

### Performance Achievements

- **99.98%** face detection accuracy using MTCNN
- **≥95%** anti-spoofing accuracy with EfficientNet-B3 (detects AI-generated faces)
- **≥95%** facial recognition accuracy using ArcFace
- **<2 seconds** processing time per face
- **<5%** False Acceptance Rate (FAR)

### Key Objectives

1. **Detect faces** in images and video streams with high precision
2. **Distinguish between real faces and AI-generated faces** using advanced anti-spoofing techniques
3. **Identify spoofing attempts** including photos, videos, and deepfakes
4. **Recognize and verify** individual identities accurately
5. **Provide real-time processing** capabilities for live video streams
6. **Ensure system robustness** against various attack vectors including AI-generated content

---

## ✨ Features

### Core Capabilities

- **🔍 Face Detection**: MTCNN-based detection with 5-point facial landmark extraction
- **🛡️ Anti-Spoofing & AI Detection**: EfficientNet-B3 model detecting:
  - Print attacks (photos)
  - Video replay attacks
  - **AI-generated faces** (StyleGAN, ProGAN, deepfakes)
  - Real vs. Fake face classification
- **👤 Face Recognition**: ArcFace with ResNet-50 backbone generating 512D embeddings
- **📹 Real-Time Processing**: Live camera feed processing at ≥5 FPS
- **📤 Image Upload**: Support for JPEG, PNG, BMP formats (max 10MB)
- **👥 Face Enrollment**: Multi-image enrollment with quality validation
- **📊 Access Logging**: Comprehensive audit trail with encryption
- **🔐 User Management**: Role-based access control (Admin, Operator, Viewer)
- **📈 Analytics & Reporting**: Usage statistics and PDF/CSV export

### Security Features

- AES-256 encryption for face embeddings
- HTTPS/TLS 1.3 for web communications
- Password hashing with bcrypt (cost factor ≥12)
- Rate limiting and session management
- Immutable audit logs with checksums
- GDPR compliance mechanisms

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       Web Interface                          │
│            (HTML5, CSS3, JavaScript/React)                   │
└─────────────────────┬───────────────────────────────────────┘
                      │ REST API (JSON)
┌─────────────────────▼───────────────────────────────────────┐
│                    Backend API                               │
│                 (Flask/FastAPI)                              │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────┐   ┌──────────────┐   ┌──────────────┐        │
│  │  MTCNN   │ → │ EfficientNet │ → │   ArcFace    │        │
│  │Face Det. │   │Anti-Spoofing │   │ Recognition  │        │
│  └──────────┘   └──────────────┘   └──────────────┘        │
├─────────────────────────────────────────────────────────────┤
│  Database Layer (PostgreSQL/SQLite) + FAISS Index           │
└─────────────────────────────────────────────────────────────┘
```

### Pipeline Flow

1. **Input** → Image/Video frame
2. **MTCNN** → Face detection + landmark extraction
3. **EfficientNet-B3** → Anti-spoofing verification
4. **ArcFace** → Feature embedding generation
5. **FAISS** → Similarity search in database
6. **Output** → Identity + confidence score + liveness status

---

## 💻 Requirements

### Hardware Requirements

**Minimum:**
- GPU: NVIDIA RTX 3060 (6GB VRAM) or equivalent
- CPU: Multi-core processor (4+ cores)
- RAM: 16GB
- Storage: 500GB SSD

**Recommended:**
- GPU: NVIDIA RTX 3070/3080 (8GB+ VRAM)
- CPU: 8+ core processor
- RAM: 32GB
- Storage: 1TB NVMe SSD

### Software Dependencies

**Core:**
- Python 3.10.11
- PyTorch 1.13+ (with CUDA support)
- OpenCV 4.5+
- Flask/FastAPI
- NumPy, Pandas

**Deep Learning:**
- torchvision
- facenet-pytorch
- efficientnet-pytorch

**Database & Search:**
- PostgreSQL 12+ / MySQL 8+ / SQLite
- FAISS (Facebook AI Similarity Search)

**Web & API:**
- Flask-CORS
- Flask-SQLAlchemy
- Gunicorn/Uvicorn
- Requests

**Utilities:**
- Pillow (PIL)
- scikit-learn
- matplotlib
- tqdm

---

## 🚀 Installation

### Step 1: Clone the Repository

```bash
git clone https://github.com/your-username/facial-recognition-system.git
cd facial-recognition-system
```

### Step 2: Create Virtual Environment

```bash
# Using venv
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Linux/Mac)
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install other requirements
pip install -r requirements.txt
```

### Step 4: Download Pre-trained Models

```bash
# Create models directory
mkdir -p models/weights

# Download MTCNN weights (automatically downloaded by facenet-pytorch)
# Download EfficientNet-B3 pretrained weights
python scripts/download_models.py --model efficientnet-b3

# Download ArcFace weights (ResNet-50 backbone)
python scripts/download_models.py --model arcface-resnet50
```

### Step 5: Download Datasets

The project uses three main datasets:

#### 1. 140k Real and Fake Faces Dataset
**Purpose**: Training MTCNN face detection and anti-spoofing model  
**Source**: [Kaggle - 140k Real and Fake Faces](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces?resource=download)  
**Size**: ~4GB  
**Contents**: 70,000 real faces + 70,000 AI-generated faces (StyleGAN, ProGAN)

```bash
# Download from Kaggle
kaggle datasets download -d xhlulu/140k-real-and-fake-faces

# Or use script
python scripts/download_datasets.py --dataset 140k-real-fake --output data/raw/
```

#### 2. Labeled Faces in the Wild (LFW) Dataset
**Purpose**: Training and evaluating ArcFace recognition model  
**Source**: [Kaggle - LFW Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)  
**Size**: ~118MB  
**Contents**: 13,233 images, 5,749 identities

```bash
# Download from Kaggle
kaggle datasets download -d jessicali9530/lfw-dataset

# Or use script
python scripts/download_datasets.py --dataset lfw --output data/raw/
```

#### 3. CelebFaces Attributes (CelebA) Dataset
**Purpose**: Training and evaluating ArcFace recognition model  
**Source**: [Kaggle - CelebA Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)  
**Size**: ~1GB  
**Contents**: 202,599 images, 10,177 identities

```bash
# Download from Kaggle
kaggle datasets download -d jessicali9530/celeba-dataset

# Or use script
python scripts/download_datasets.py --dataset celeba --output data/raw/
```

#### Download All Datasets at Once

```bash
# Setup Kaggle API credentials first
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json

# Download all datasets
python scripts/download_all_datasets.py
```

### Step 6: Database Setup

```bash
# Initialize database
python scripts/init_database.py

# Create tables
python scripts/create_tables.py

# (Optional) Load sample data
python scripts/load_sample_data.py
```

### Step 7: Configuration

```bash
# Copy example config
cp config/config.example.yaml config/config.yaml

# Edit configuration
nano config/config.yaml
```

---

## ⚙️ Configuration

### Configuration File (`config/config.yaml`)

```yaml
# Database Configuration
database:
  type: postgresql  # postgresql, mysql, sqlite
  host: localhost
  port: 5432
  name: facial_recognition_db
  username: admin
  password: your_secure_password

# Model Paths
models:
  mtcnn: models/weights/mtcnn/
  efficientnet: models/weights/efficientnet_b3_antispoofing.pth
  arcface: models/weights/arcface_resnet50.pth

# Detection Settings
detection:
  min_face_size: 40
  detection_threshold: 0.7
  iou_threshold: 0.5

# Anti-Spoofing Settings
antispoofing:
  threshold: 0.85
  model_input_size: [224, 224]

# Recognition Settings
recognition:
  embedding_size: 512
  similarity_threshold: 0.75
  max_database_size: 10000

# Performance Settings
performance:
  batch_size: 32
  num_workers: 4
  gpu_id: 0
  use_mixed_precision: true

# Security Settings
security:
  encryption_key: your_aes_256_key
  session_timeout: 1800  # 30 minutes
  max_login_attempts: 5
  bcrypt_rounds: 12

# Logging
logging:
  level: INFO
  file: logs/application.log
  max_size: 100MB
  backup_count: 10
```

### Environment Variables

Create a `.env` file:

```bash
# Flask Settings
FLASK_APP=app.py
FLASK_ENV=production
SECRET_KEY=your_secret_key_here

# Database
DATABASE_URL=postgresql://user:password@localhost:5432/facial_recognition_db

# GPU Settings
CUDA_VISIBLE_DEVICES=0

# API Keys (if using cloud services)
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
```

---

## 📖 Usage

### Running the Application Locally

#### 1. Start Backend Server

```bash
# Development mode
python app.py

# Production mode with Gunicorn
gunicorn -w 4 -b 0.0.0.0:5000 app:app
```

#### 2. Access Web Interface

Open browser and navigate to:
```
http://localhost:5000
```

#### 3. Default Login Credentials

```
Username: admin
Password: admin123
```

**⚠️ Change default credentials immediately after first login!**

### Command Line Interface

#### Enroll New Person

```bash
python cli.py enroll \
  --name "John Doe" \
  --national-id "1234567890" \
  --images person_photos/*.jpg
```

#### Recognize Face from Image

```bash
python cli.py recognize \
  --image test_image.jpg \
  --threshold 0.75
```

#### Process Video Stream

```bash
python cli.py video \
  --source 0  # Webcam
  --output results/video_output.avi
```

#### Batch Processing

```bash
python cli.py batch \
  --input-folder test_images/ \
  --output-csv results.csv
```

### Python API Usage

```python
from facial_recognition import FacialRecognitionSystem

# Initialize system
system = FacialRecognitionSystem(config_path='config/config.yaml')

# Enroll person
system.enroll_person(
    name="Jane Smith",
    national_id="9876543210",
    image_paths=["photo1.jpg", "photo2.jpg", "photo3.jpg"]
)

# Recognize face
result = system.recognize_face("unknown_person.jpg")
print(f"Identity: {result['name']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Is Real: {result['is_real']}")

# Process video
for frame_result in system.process_video(source=0):
    print(f"Detected: {frame_result['num_faces']} faces")
```

---

## 🔌 API Documentation

### Base URL

```
http://localhost:5000/api/v1
```

### Authentication

All API endpoints (except `/login`) require JWT token authentication:

```bash
# Login to get token
curl -X POST http://localhost:5000/api/v1/login \
  -H "Content-Type: application/json" \
  -d '{"username": "admin", "password": "admin123"}'

# Response
{
  "token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9...",
  "expires_in": 3600
}

# Use token in subsequent requests
curl -X GET http://localhost:5000/api/v1/persons \
  -H "Authorization: Bearer YOUR_TOKEN_HERE"
```

### Endpoints

#### 1. Face Detection

**POST** `/api/v1/detect`

Detect faces in an image.

```bash
curl -X POST http://localhost:5000/api/v1/detect \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@photo.jpg"
```

**Response:**
```json
{
  "success": true,
  "num_faces": 2,
  "faces": [
    {
      "bbox": [100, 150, 250, 300],
      "confidence": 0.99,
      "landmarks": {
        "left_eye": [120, 180],
        "right_eye": [230, 180],
        "nose": [175, 220],
        "mouth_left": [140, 270],
        "mouth_right": [210, 270]
      }
    }
  ]
}
```

#### 2. Anti-Spoofing Check

**POST** `/api/v1/antispoofing`

Check if face is real or spoofed.

```bash
curl -X POST http://localhost:5000/api/v1/antispoofing \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@face.jpg"
```

**Response:**
```json
{
  "success": true,
  "is_real": true,
  "confidence": 0.92,
  "spoof_type": "real",
  "processing_time": 0.45
}
```

#### 3. Face Recognition

**POST** `/api/v1/recognize`

Recognize person from face image.

```bash
curl -X POST http://localhost:5000/api/v1/recognize \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "image=@unknown.jpg" \
  -F "threshold=0.75"
```

**Response:**
```json
{
  "success": true,
  "recognized": true,
  "person": {
    "person_id": 42,
    "name": "John Doe",
    "national_id": "1234567890"
  },
  "confidence": 0.87,
  "is_real": true,
  "processing_time": 1.2,
  "top_matches": [
    {"person_id": 42, "name": "John Doe", "similarity": 0.87},
    {"person_id": 15, "name": "Jane Smith", "similarity": 0.62}
  ]
}
```

#### 4. Enroll Person

**POST** `/api/v1/enroll`

Add new person to database.

```bash
curl -X POST http://localhost:5000/api/v1/enroll \
  -H "Authorization: Bearer YOUR_TOKEN" \
  -F "name=Alice Johnson" \
  -F "national_id=5555555555" \
  -F "images=@photo1.jpg" \
  -F "images=@photo2.jpg" \
  -F "images=@photo3.jpg"
```

**Response:**
```json
{
  "success": true,
  "person_id": 123,
  "embeddings_count": 3,
  "average_quality": 0.89,
  "message": "Person enrolled successfully"
}
```

#### 5. Get Person Details

**GET** `/api/v1/persons/{person_id}`

Retrieve person information.

```bash
curl -X GET http://localhost:5000/api/v1/persons/123 \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "success": true,
  "person": {
    "person_id": 123,
    "name": "Alice Johnson",
    "national_id": "5555555555",
    "status": "active",
    "registered_at": "2025-11-30T10:30:00Z",
    "embeddings_count": 3,
    "created_by": "admin"
  }
}
```

#### 6. Authentication Logs

**GET** `/api/v1/logs`

Retrieve authentication logs.

```bash
curl -X GET "http://localhost:5000/api/v1/logs?start_date=2025-11-01&end_date=2025-11-30&status=success" \
  -H "Authorization: Bearer YOUR_TOKEN"
```

**Response:**
```json
{
  "success": true,
  "total": 156,
  "logs": [
    {
      "log_id": 1001,
      "person_id": 42,
      "person_name": "John Doe",
      "timestamp": "2025-11-30T08:15:30Z",
      "status": "success",
      "confidence": 0.87,
      "is_real": true,
      "device_info": "webcam",
      "response_time": 1.2
    }
  ]
}
```

---

## 🌐 Deployment

### Option 1: Local Deployment

See [Installation](#installation) and [Usage](#usage) sections above.

### Option 2: Docker Deployment

#### Build Docker Image

```bash
docker build -t facial-recognition-system .
```

#### Run Container

```bash
docker run -d \
  --name facial-recognition \
  --gpus all \
  -p 5000:5000 \
  -v $(pwd)/data:/app/data \
  -v $(pwd)/models:/app/models \
  -e DATABASE_URL=postgresql://user:pass@db:5432/facial_db \
  facial-recognition-system
```

#### Docker Compose

```yaml
version: '3.8'

services:
  web:
    build: .
    ports:
      - "5000:5000"
    volumes:
      - ./data:/app/data
      - ./models:/app/models
    environment:
      - DATABASE_URL=postgresql://user:pass@db:5432/facial_db
      - CUDA_VISIBLE_DEVICES=0
    depends_on:
      - db
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

  db:
    image: postgres:14
    environment:
      - POSTGRES_USER=user
      - POSTGRES_PASSWORD=pass
      - POSTGRES_DB=facial_db
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

Run with:
```bash
docker-compose up -d
```

### Option 3: Cloud Deployment (AWS/Azure/GCP)

Detailed cloud deployment guides available in `docs/deployment/`:
- AWS EC2 with GPU instances
- Azure VM with NVIDIA GPU
- Google Cloud Compute Engine
- Kubernetes deployment

### Accessing Deployed Web Application

**🌐 Live Application**: [https://ai-face-authenticity-9p0e.bolt.host/](https://ai-face-authenticity-9p0e.bolt.host/)

**📦 GitHub Repository**: [https://github.com/facial-recognition-lab/facial-recognition-system-code](https://github.com/facial-recognition-lab/facial-recognition-system-code)

The web interface includes:
- **Home Page**: System overview, statistics, and AI detection capabilities
- **Upload Page**: Image upload for face recognition and AI detection
- **Webcam Page**: Real-time video processing with live AI authenticity check
- **Results Page**: Detailed recognition results with:
  - Identity verification
  - Confidence scores
  - **Real vs. AI-generated classification**
  - Spoofing detection status

---

## 📊 Performance Metrics

### Model Performance (Test Set Results)

| Metric | MTCNN | EfficientNet-B3 | ArcFace |
|--------|-------|-----------------|---------|
| **Accuracy** | 99.98% | 95.3% | 96.2% |
| **Precision** | 99.95% | 94.8% | 95.7% |
| **Recall** | 99.92% | 95.1% | 96.0% |
| **F1-Score** | 99.93% | 94.9% | 95.8% |
| **FAR** | 0.05% | 4.2% | 1.8% |
| **FRR** | 0.08% | 4.9% | 2.3% |

### System Performance

| Metric | Target | Achieved |
|--------|--------|----------|
| **Face Detection Speed** | ≤0.5s | 0.12s |
| **Anti-Spoofing Speed** | ≤1.0s | 0.45s |
| **Recognition Speed** | ≤1.0s | 0.35s |
| **Total Processing Time** | ≤2.0s | 0.92s |
| **Video Processing FPS** | ≥5 FPS | 8.3 FPS |
| **Concurrent Users** | ≥10 | 15 |
| **System Uptime** | ≥99% | 99.7% |

### Dataset Statistics

#### 1. 140k Real and Fake Faces Dataset
**Used for**: MTCNN Face Detection & EfficientNet-B3 Anti-Spoofing Training  
**Total Images**: 140,000 (50% real, 50% AI-generated)
- **Real Faces**: 70,000 authentic human faces
- **Fake Faces**: 70,000 AI-generated faces (StyleGAN, ProGAN)
- **Resolution**: 256×256 pixels
- **Format**: JPEG

**Preprocessing Results**:
- Training Set: 99,989/100,000 (99.99% success rate)
- Validation Set: 19,998/20,000 (99.99% success rate)
- Test Set: 19,997/20,000 (99.99% success rate)

#### 2. Labeled Faces in the Wild (LFW)
**Used for**: ArcFace Face Recognition Training & Evaluation  
- **Total Images**: 13,233
- **Identities**: 5,749 individuals
- **Characteristics**: Unconstrained, real-world conditions
- **Purpose**: Recognition accuracy benchmarking

#### 3. CelebFaces Attributes (CelebA)
**Used for**: ArcFace Face Recognition Training & Evaluation  
- **Total Images**: 202,599
- **Identities**: 10,177 celebrities
- **Attributes**: 40 facial attributes per image
- **Purpose**: Diverse training data for robust recognition

---

## 📁 Project Structure

```
facial-recognition-system/
│
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── setup.py                     # Package setup
├── Dockerfile                   # Docker configuration
├── docker-compose.yml           # Docker Compose setup
├── .env.example                 # Environment variables template
├── .gitignore                   # Git ignore rules
│
├── app.py                       # Main Flask application
├── config/
│   ├── config.yaml              # Main configuration
│   └── config.example.yaml      # Configuration template
│
├── src/                         # Source code
│   ├── __init__.py
│   ├── detection/
│   │   ├── __init__.py
│   │   ├── mtcnn.py             # MTCNN face detection
│   │   └── preprocessing.py     # Image preprocessing
│   │
│   ├── antispoofing/
│   │   ├── __init__.py
│   │   ├── efficientnet.py      # EfficientNet-B3 model
│   │   └── evaluation.py        # Anti-spoofing evaluation
│   │
│   ├── recognition/
│   │   ├── __init__.py
│   │   ├── arcface.py           # ArcFace model
│   │   ├── embedding.py         # Embedding generation
│   │   └── matcher.py           # Face matching logic
│   │
│   ├── database/
│   │   ├── __init__.py
│   │   ├── models.py            # SQLAlchemy models
│   │   ├── operations.py        # CRUD operations
│   │   └── faiss_index.py       # FAISS indexing
│   │
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py            # API endpoints
│   │   ├── auth.py              # Authentication
│   │   └── validators.py        # Input validation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logger.py            # Logging utilities
│       ├── encryption.py        # Data encryption
│       └── metrics.py           # Performance metrics
│
├── models/                      # Model weights
│   ├── weights/
│   │   ├── mtcnn/
│   │   ├── efficientnet_b3_antispoofing.pth
│   │   └── arcface_resnet50.pth
│   └── configs/
│       └── model_configs.yaml
│
├── data/                        # Datasets
│   ├── raw/                     # Original datasets
│   ├── processed/               # Preprocessed data
│   └── embeddings/              # Generated embeddings
│
├── scripts/                     # Utility scripts
│   ├── download_models.py
│   ├── download_datasets.py
│   ├── init_database.py
│   ├── create_tables.py
│   ├── train_antispoofing.py
│   └── train_recognition.py
│
├── tests/                       # Unit tests
│   ├── test_detection.py
│   ├── test_antispoofing.py
│   ├── test_recognition.py
│   └── test_api.py
│
├── frontend/                    # Web interface (Bolt.new)
│   ├── index.html
│   ├── upload.html
│   ├── webcam.html
│   ├── results.html
│   ├── css/
│   └── js/
│
├── docs/                        # Documentation
│   ├── architecture.md
│   ├── api_reference.md
│   ├── user_manual.md
│   ├── deployment/
│   └── research/
│
└── logs/                        # Application logs
    └── application.log
```

---

## 👥 Team

**Track**: AI & Data Science - Microsoft Machine Learning Engineer  
**Project Duration**: 4 weeks (November 2025)

### Team Roles & Responsibilities

| Role | Team Member | Key Responsibilities |
|------|-------------|---------------------|
| **Project Leader & Documentation** | Hana Hany Fathy Ali | • ArcFace optimization & evaluation (Part 2)<br>• Complete technical documentation<br>• Model performance testing & reports<br>• Backend & Frontend development<br>• Project coordination |
| **MTCNN Specialist & Backend** | Jana Walid Sabry Mohamed | • MTCNN face detection implementation<br>• Face landmarks extraction (5 points)<br>• Face cropping & preprocessing pipeline<br>• Backend API development<br>• Frontend integration<br>• Final project presentation |
| **Anti-Spoofing Engineer (Part 1)** | Tasneem Osama Hassan Hassan | • Spoofing datasets preparation<br>• EfficientNet-B3 data preprocessing<br>• Model training (first phase)<br>• Texture analysis implementation<br>• Real vs. AI-generated classification |
| **Anti-Spoofing Engineer (Part 2)** | Youssef Mohamed Rasmy Ahmed | • EfficientNet-B3 training completion<br>• AI-generated faces integration<br>• Motion analysis implementation<br>• Hyperparameter tuning & optimization<br>• Model evaluation & pipeline integration |
| **Recognition Engineer (Part 1)** | Anas Mohamed Mostafa Mohamed | • ArcFace architecture implementation<br>• ResNet-50 backbone setup<br>• Angular margin loss implementation<br>• LFW & CelebA dataset preparation<br>• Initial model training<br>• 512D embeddings extraction |
| **Data Engineer** | Abdulrhman Osama Atwa Abu Jazar | • All datasets download & organization<br>• Data cleaning (remove blurry/duplicates)<br>• Image preprocessing & augmentation<br>• Dataset quality assurance<br>• Deliver processed data to team |

### Development Workflow

#### Phase 1: Data Preparation (Week 1)
**Led by**: Abdulrhman Osama
1. Download 140k Real/Fake dataset
2. Clean and organize data (remove unclear/duplicate images)
3. Preprocessing: face cropping, resizing, augmentation
4. Deliver organized datasets to team members

#### Phase 2: Model Development (Weeks 1-3)

**MTCNN Implementation** (Jana Walid)
- Setup and configure MTCNN for face detection
- Extract 5-point facial landmarks
- Crop faces from images and videos
- Ensure detection accuracy ≥99%
- Integrate into main pipeline

**EfficientNet-B3 Anti-Spoofing - Part 1** (Tasneem Osama)
- Gather and prepare spoofing datasets
- Preprocess data for anti-spoofing training
- Build and train EfficientNet-B3 (first half)
- Implement texture analysis features

**EfficientNet-B3 Anti-Spoofing - Part 2** (Youssef Mohamed)
- Complete EfficientNet-B3 training
- Collect and integrate AI-generated faces
- Implement motion analysis
- Fine-tune hyperparameters
- Evaluate model (Accuracy, Precision, Recall, FAR)
- Integrate into pipeline

**ArcFace Recognition - Part 1** (Anas Mohamed)
- Implement ArcFace architecture with ResNet-50
- Prepare LFW and CelebA datasets
- Implement angular margin loss
- Begin training on prepared datasets
- Extract 512-dimensional embeddings

**ArcFace Recognition - Part 2** (Hana Hany)
- Complete fine-tuning on LFW & CelebA
- Optimize inference speed
- Comprehensive evaluation:
  - Accuracy, Precision, Recall metrics
  - False Acceptance Rate (FAR) analysis
  - Confusion matrix generation
  - Comparison with FaceNet/CosFace
- Final pipeline integration

#### Phase 3: Integration & Deployment (Week 4)

**Backend & Frontend Development** (Jana & Hana)
- Build RESTful API with Flask/FastAPI
- Integrate MTCNN → EfficientNet → ArcFace pipeline
- Develop web interface with 4 pages
- Connect frontend to backend endpoints
- Real-time video processing implementation

**Documentation & Testing** (Hana)
- Write comprehensive technical documentation
- Create user manual and API documentation
- Perform system testing and optimization
- Prepare final presentation
- GitHub repository setup

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

### Research Papers & Algorithms

- **MTCNN**: Zhang, K., et al. (2016) - "Joint Face Detection and Alignment using Multi-task Cascaded Convolutional Networks"
- **EfficientNet**: Tan, M., & Le, Q. (2019) - "EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks"
- **ArcFace**: Deng, J., et al. (2019) - "ArcFace: Additive Angular Margin Loss for Deep Face Recognition"

### Datasets

1. **140k Real and Fake Faces**: [Kaggle Dataset](https://www.kaggle.com/datasets/xhlulu/140k-real-and-fake-faces)
   - Used for MTCNN face detection and anti-spoofing training
   
2. **Labeled Faces in the Wild (LFW)**: [Kaggle Dataset](https://www.kaggle.com/datasets/jessicali9530/lfw-dataset)
   - Used for ArcFace recognition training and evaluation
   
3. **CelebFaces Attributes (CelebA)**: [Kaggle Dataset](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
   - Used for ArcFace recognition training and evaluation

### Tools & Frameworks

- PyTorch for deep learning implementation
- OpenCV for image processing
- Flask/FastAPI for backend API
- FAISS for efficient similarity search
- Bolt.new for frontend deployment

---

## 🔗 Links

- **🌐 Live Demo**: [https://ai-face-authenticity-9p0e.bolt.host/](https://ai-face-authenticity-9p0e.bolt.host/)
- **📦 GitHub Repository**: [https://github.com/facial-recognition-lab/facial-recognition-system-code](https://github.com/facial-recognition-lab/facial-recognition-system-code)
- **📊 Project Documentation**: [Full Technical Docs](https://github.com/facial-recognition-lab/facial-recognition-system-code/tree/main/docs)

---

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues**: [Create an issue](https://github.com/facial-recognition-lab/facial-recognition-system-code/issues)
- **Pull Requests**: [Contribute to the project](https://github.com/facial-recognition-lab/facial-recognition-system-code/pulls)
- **Documentation**: [Full documentation](https://github.com/facial-recognition-lab/facial-recognition-system-code/wiki)

---

## 🔄 Version History

- **v1.0.0** (November 30, 2025): Initial Production Release
  - ✅ MTCNN face detection (99.98% accuracy achieved)
  - ✅ EfficientNet-B3 anti-spoofing (95.3% accuracy)
    - Real vs. AI-generated face detection
    - Print attack detection
    - Video replay detection
  - ✅ ArcFace recognition (96.2% accuracy)
    - ResNet-50 backbone
    - 512-dimensional embeddings
    - Trained on LFW & CelebA datasets
  - ✅ Web interface deployed at [ai-face-authenticity-9p0e.bolt.host](https://ai-face-authenticity-9p0e.bolt.host/)
  - ✅ RESTful API with JWT authentication
  - ✅ PostgreSQL/SQLite database support
  - ✅ Docker deployment configuration
  - ✅ Complete documentation and user manual
  - ✅ GitHub repository published

### Datasets Used
- 140k Real and Fake Faces (MTCNN & Anti-Spoofing)
- LFW - Labeled Faces in the Wild (ArcFace Training)
- CelebA - CelebFaces Attributes (ArcFace Training)

### Models Implemented
1. **MTCNN** - Face Detection & Landmark Extraction
2. **EfficientNet-B3** - Anti-Spoofing & AI Detection
3. **ArcFace with ResNet-50** - Face Recognition

---

**🎓 Developed by Microsoft Machine Learning Engineer Track Team**  
**Built with by the Facial Recognition Lab | November 2025**

---

## 📝 Citation

If you use this project in your research or work, please cite:

```bibtex
@software{facial_recognition_system_2025,
  title={Facial Recognition System with Anti-Spoofing and AI Detection},
  author={Facial Recognition Lab Team},
  year={2025},
  month={November},
  url={https://github.com/facial-recognition-lab/facial-recognition-system-code},
  note={AI & Data Science Track - Microsoft Machine Learning Engineer}
}
```