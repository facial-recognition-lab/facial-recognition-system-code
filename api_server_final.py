from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import numpy as np
from PIL import Image
import io
import base64
import torch
import torch.nn.functional as F
from torchvision import transforms, models
import torch.nn as nn
from facenet_pytorch import MTCNN
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# =============================================================================
# CONFIGURATION
# =============================================================================

class Config:
    ARCFACE_MODEL = r'E:\DEPE Project\ArcFace_Output\checkpoints\LFW_best.pth'
    ANTISPOOFING_MODEL = r'E:\DEPE Project\antispoofing_output\checkpoints\best_model.keras'
    DATABASE_FILE = 'api_face_database.npz'
    VERIFICATION_THRESHOLD = 0.6
    SPOOFING_THRESHOLD = 0.5
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = 224
    FACE_SIZE = 112

# =============================================================================
# INITIALIZE FASTAPI WITH FULL CORS
# =============================================================================
app = FastAPI(
    title="Face Recognition & Anti-Spoofing API",
    description="Complete face recognition and liveness detection system",
    version="1.0.0"
)

#IMPORTANT: CORS Configuration for Bolt.host
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://ai-face-authenticity-9p0e.bolt.host",  # Your Bolt.host URL
        "http://localhost:3000",  # For local testing
        "http://localhost:5173",  # Vite dev server
        "*"  # Allow all origins (for development)
    ],
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, PUT, DELETE, etc.)
    allow_headers=["*"],  # Allow all headers
)

# =============================================================================
# LOAD MODELS
# =============================================================================

print("Loading models:")

# MTCNN
mtcnn = MTCNN(
    image_size=Config.FACE_SIZE,
    margin=20,
    keep_all=False,
    device=Config.DEVICE,
    post_process=False
)
print(" MTCNN loaded")

# Anti-Spoofing
try:
    antispoofing_model = load_model(Config.ANTISPOOFING_MODEL)
    print("Anti-Spoofing model loaded")
except:
    antispoofing_model = None
    print("Anti-Spoofing model not found")

# ArcFace
class ArcFaceModel(nn.Module):
    def __init__(self, num_classes=5749, embedding_size=512):
        super().__init__()
        resnet = models.resnet50(weights=None)
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(2048, embedding_size)
        self.bn = nn.BatchNorm1d(embedding_size)
    
    def forward(self, x, labels=None):
        features = self.backbone(x)
        features = features.view(features.size(0), -1)
        embeddings = self.fc(features)
        embeddings = self.bn(embeddings)
        return embeddings

try:
    arcface_model = ArcFaceModel()
    checkpoint = torch.load(Config.ARCFACE_MODEL, map_location=Config.DEVICE)
    arcface_model.load_state_dict(checkpoint['model_state_dict'])
    arcface_model = arcface_model.to(Config.DEVICE)
    arcface_model.eval()
    print("ArcFace model loaded")
except:
    arcface_model = None
    print("ArcFace model not found")

# Transforms
antispoofing_transform = transforms.Compose([
    transforms.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x * 255)
])

arcface_transform = transforms.Compose([
    transforms.Resize((Config.FACE_SIZE, Config.FACE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# Face Database
class FaceDatabase:
    def __init__(self):
        self.embeddings = []
        self.names = []
        self.load()
    
    def add_face(self, name, embedding):
        self.embeddings.append(embedding)
        self.names.append(name)
        self.save()
    
    def find_match(self, embedding, threshold=Config.VERIFICATION_THRESHOLD):
        if len(self.embeddings) == 0:
            return None, 0.0
        
        similarities = []
        for db_emb, name in zip(self.embeddings, self.names):
            sim = np.dot(embedding.flatten(), db_emb.flatten()) / \
                  (np.linalg.norm(embedding) * np.linalg.norm(db_emb))
            similarities.append((name, float(sim)))
        
        best_name, best_sim = max(similarities, key=lambda x: x[1])
        
        if best_sim >= threshold:
            return best_name, best_sim
        return None, best_sim
    
    def save(self):
        if len(self.embeddings) > 0:
            np.savez(Config.DATABASE_FILE,
                     embeddings=np.array(self.embeddings),
                     names=np.array(self.names))
    
    def load(self):
        if os.path.exists(Config.DATABASE_FILE):
            data = np.load(Config.DATABASE_FILE, allow_pickle=True)
            self.embeddings = data['embeddings'].tolist()
            self.names = data['names'].tolist()
            print(f"✅ Database loaded: {len(self.names)} faces")

face_db = FaceDatabase()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def load_image_from_base64(base64_string: str) -> Image.Image:
    """Load PIL Image from base64 string"""
    try:
        # Remove data URL prefix if present
        if ',' in base64_string:
            base64_string = base64_string.split(',')[1]
        
        image_data = base64.b64decode(base64_string)
        image = Image.open(io.BytesIO(image_data)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid base64 image: {str(e)}")

async def load_image_from_upload(file: UploadFile) -> Image.Image:
    """Load PIL Image from uploaded file"""
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert('RGB')
        return image
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {str(e)}")

def detect_face(image: Image.Image):
    """Detect face and return cropped face"""
    try:
        boxes, probs, landmarks = mtcnn.detect(image, landmarks=True)
        
        if boxes is None or len(boxes) == 0:
            return None, None, None
        
        box = boxes[0]
        landmark = landmarks[0] if landmarks is not None else None
        
        x1, y1, x2, y2 = [int(b) for b in box]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(image.width, x2), min(image.height, y2)
        
        face_img = image.crop((x1, y1, x2, y2))
        
        return face_img, box.tolist(), landmark.tolist() if landmark is not None else None
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Face detection error: {str(e)}")

def check_spoofing(face_img: Image.Image):
    """Check if face is real or fake"""
    if antispoofing_model is None:
        return True, 1.0
    
    try:
        img_tensor = antispoofing_transform(face_img)
        img_tensor = img_tensor.unsqueeze(0).numpy()
        
        prediction = antispoofing_model.predict(img_tensor, verbose=0)[0][0]
        
        is_real = prediction >= Config.SPOOFING_THRESHOLD
        confidence = float(prediction if is_real else (1 - prediction))
        
        return is_real, confidence
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anti-spoofing error: {str(e)}")

def extract_embedding(face_img: Image.Image):
    """Extract face embedding"""
    if arcface_model is None:
        raise HTTPException(status_code=500, detail="ArcFace model not loaded")
    
    try:
        img_tensor = arcface_transform(face_img)
        img_tensor = img_tensor.unsqueeze(0).to(Config.DEVICE)
        
        with torch.no_grad():
            embedding = arcface_model(img_tensor)
            embedding = F.normalize(embedding, p=2, dim=1)
        
        return embedding.cpu().numpy()
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Embedding extraction error: {str(e)}")

# =============================================================================
# PYDANTIC MODELS
# =============================================================================

class Base64ImageRequest(BaseModel):
    image: str  # Base64 encoded image

class DetectionResponse(BaseModel):
    face_detected: bool
    box: Optional[List[float]] = None
    landmarks: Optional[List[List[float]]] = None

class SpoofingResponse(BaseModel):
    face_detected: bool
    is_real: bool
    confidence: float

class RecognitionResponse(BaseModel):
    face_detected: bool
    is_real: bool
    spoofing_confidence: float
    recognized: bool
    name: Optional[str] = None
    similarity: float

class VerificationResponse(BaseModel):
    is_match: bool
    similarity: float

class RegisterResponse(BaseModel):
    success: bool
    message: str

class DatabaseResponse(BaseModel):
    total_faces: int
    faces: List[dict]

# =============================================================================
# API ENDPOINTS
# =============================================================================

@app.get("/")
async def root():
    """API information"""
    return {
        "name": "Face Recognition & Anti-Spoofing API",
        "version": "1.0.0",
        "status": "running",
        "models": {
            "mtcnn": "loaded",
            "antispoofing": "loaded" if antispoofing_model else "not loaded",
            "arcface": "loaded" if arcface_model else "not loaded"
        },
        "cors_enabled": True,
        "supported_domains": ["https://ai-face-authenticity-9p0e.bolt.host"]
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "models_loaded": {
            "mtcnn": True,
            "antispoofing": antispoofing_model is not None,
            "arcface": arcface_model is not None
        }
    }

# ============================================================================
# Base64 Image Endpoints (for Web Integration)
# ============================================================================

@app.post("/api/analyze-base64", response_model=RecognitionResponse)
async def analyze_base64(request: Base64ImageRequest):
    """
    Complete analysis from base64 image (for web upload)
    """
    image = load_image_from_base64(request.image)
    
    # Detect face
    face_img, _, _ = detect_face(image)
    
    if face_img is None:
        return RecognitionResponse(
            face_detected=False,
            is_real=False,
            spoofing_confidence=0.0,
            recognized=False,
            similarity=0.0
        )
    
    # Check spoofing
    is_real, spoof_conf = check_spoofing(face_img)
    
    if not is_real:
        return RecognitionResponse(
            face_detected=True,
            is_real=False,
            spoofing_confidence=spoof_conf,
            recognized=False,
            similarity=0.0
        )
    
    # Recognize face
    embedding = extract_embedding(face_img)
    name, similarity = face_db.find_match(embedding)
    
    return RecognitionResponse(
        face_detected=True,
        is_real=True,
        spoofing_confidence=spoof_conf,
        recognized=name is not None,
        name=name,
        similarity=similarity
    )

@app.post("/api/register-base64", response_model=RegisterResponse)
async def register_base64(image: str = Form(...), name: str = Form(...)):
    """
    Register face from base64 image
    """
    if not name or name.strip() == "":
        raise HTTPException(status_code=400, detail="Name is required")
    
    img = load_image_from_base64(image)
    
    # Detect face
    face_img, _, _ = detect_face(img)
    
    if face_img is None:
        return RegisterResponse(
            success=False,
            message="No face detected in image"
        )
    
    # Check if real
    is_real, confidence = check_spoofing(face_img)
    
    if not is_real:
        return RegisterResponse(
            success=False,
            message=f"Fake face detected (confidence: {confidence*100:.1f}%)"
        )
    
    # Extract embedding and save
    embedding = extract_embedding(face_img)
    face_db.add_face(name.strip(), embedding)
    
    return RegisterResponse(
        success=True,
        message=f"Face registered successfully for {name}"
    )

# ============================================================================
# Original File Upload Endpoints (Keep for backward compatibility)
# ============================================================================

@app.post("/detect-face", response_model=DetectionResponse)
async def detect_face_endpoint(file: UploadFile = File(...)):
    """Detect face in uploaded image"""
    image = await load_image_from_upload(file)
    face_img, box, landmarks = detect_face(image)
    
    return DetectionResponse(
        face_detected=face_img is not None,
        box=box,
        landmarks=landmarks
    )

@app.post("/check-spoofing", response_model=SpoofingResponse)
async def check_spoofing_endpoint(file: UploadFile = File(...)):
    """Check if face is real or fake"""
    image = await load_image_from_upload(file)
    face_img, _, _ = detect_face(image)
    
    if face_img is None:
        return SpoofingResponse(
            face_detected=False,
            is_real=False,
            confidence=0.0
        )
    
    is_real, confidence = check_spoofing(face_img)
    
    return SpoofingResponse(
        face_detected=True,
        is_real=is_real,
        confidence=confidence
    )

@app.post("/recognize-face", response_model=RecognitionResponse)
async def recognize_face_endpoint(file: UploadFile = File(...)):
    """Complete pipeline: detect, check spoofing, recognize"""
    image = await load_image_from_upload(file)
    
    face_img, _, _ = detect_face(image)
    
    if face_img is None:
        return RecognitionResponse(
            face_detected=False,
            is_real=False,
            spoofing_confidence=0.0,
            recognized=False,
            similarity=0.0
        )
    
    is_real, spoof_conf = check_spoofing(face_img)
    
    if not is_real:
        return RecognitionResponse(
            face_detected=True,
            is_real=False,
            spoofing_confidence=spoof_conf,
            recognized=False,
            similarity=0.0
        )
    
    embedding = extract_embedding(face_img)
    name, similarity = face_db.find_match(embedding)
    
    return RecognitionResponse(
        face_detected=True,
        is_real=True,
        spoofing_confidence=spoof_conf,
        recognized=name is not None,
        name=name,
        similarity=similarity
    )

@app.post("/register-face", response_model=RegisterResponse)
async def register_face_endpoint(
    file: UploadFile = File(...),
    name: str = Form(...)
):
    """Register new face"""
    if not name or name.strip() == "":
        raise HTTPException(status_code=400, detail="Name is required")
    
    image = await load_image_from_upload(file)
    face_img, _, _ = detect_face(image)
    
    if face_img is None:
        return RegisterResponse(
            success=False,
            message="No face detected in image"
        )
    
    is_real, confidence = check_spoofing(face_img)
    
    if not is_real:
        return RegisterResponse(
            success=False,
            message=f"Fake face detected (confidence: {confidence*100:.1f}%)"
        )
    
    embedding = extract_embedding(face_img)
    face_db.add_face(name.strip(), embedding)
    
    return RegisterResponse(
        success=True,
        message=f"Face registered successfully for {name}"
    )

@app.get("/database", response_model=DatabaseResponse)
async def get_database():
    """Get all registered faces"""
    unique_names = list(set(face_db.names))
    faces_list = []
    
    for name in unique_names:
        count = face_db.names.count(name)
        faces_list.append({"name": name, "count": count})
    
    return DatabaseResponse(
        total_faces=len(face_db.names),
        faces=faces_list
    )

@app.delete("/database/{name}")
async def delete_face(name: str):
    """Delete face from database"""
    indices_to_remove = [i for i, n in enumerate(face_db.names) if n == name]
    
    if not indices_to_remove:
        raise HTTPException(status_code=404, detail=f"Name '{name}' not found")
    
    for i in sorted(indices_to_remove, reverse=True):
        del face_db.embeddings[i]
        del face_db.names[i]
    
    face_db.save()
    
    return {"message": f"Deleted {len(indices_to_remove)} face(s) for {name}"}

@app.delete("/database")
async def clear_database():
    """Clear database"""
    face_db.embeddings = []
    face_db.names = []
    face_db.save()
    
    return {"message": "Database cleared successfully"}

# =============================================================================
# RUN SERVER
# =============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("STARTING API SERVER WITH WEB INTEGRATION")
    print("="*80)
    print(f"\nServer running at: http://localhost:8000")
    print(f"API Documentation: http://localhost:8000/docs")
    print(f"\n CORS enabled for:")
    print(f"   - https://ai-face-authenticity-9p0e.bolt.host")
    print("\n" + "="*80 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=8000)