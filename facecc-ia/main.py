from app import app
from flask import request, jsonify
from deepface import DeepFace
from utils import check_best
import numpy as np
import cv2
import os
import secrets
import tempfile
import time
import joblib

# Preload ArcFace model at startup (warm up)
print("Loading ArcFace model...")
_arcface_model = DeepFace.build_model("ArcFace")
print("Model loaded successfully")

@app.route('/facecc/facecc-ia/predict', methods=['POST'])    
def identify_face():
    t0 = time.time()
    files = request.files['file']
    img_bytes = files.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    print('toy aqui')
    t1 = time.time()
    print(f"Image decode: {(t1-t0)*1000:.0f}ms")
    
    embedding = DeepFace.represent(img_path=img_cv2,
                                    model_name='ArcFace',
                                    enforce_detection=False)[0]["embedding"]
    print('sali del embdedding')
    
    t2 = time.time()
    print(f"Embedding generation: {(t2-t1)*1000:.0f}ms")
    
    best_match = check_best(embedding)
    t3 = time.time()
    print(f"Matching: {(t3-t2)*1000:.0f}ms")
    print(f"Total: {(t3-t0)*1000:.0f}ms")
    
    files.close()
    return best_match

@app.route('/facecc/facecc-ia/embed', methods=['POST'])
def generate_embedding():
    """Generate embedding for admin face registration"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    
    file = request.files['file']
    temp_path = None
    
    try:
        # Save temp file
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{secrets.token_hex(8)}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        file.save(temp_path)
        
        # Generate embedding with face detection enforced
        result = DeepFace.represent(
            img_path=temp_path,
            model_name="ArcFace",
            enforce_detection=True  # Require face detection for admin uploads
        )
        
        if not result:
            return jsonify({'error': 'No face detected'}), 400
        
        embedding = result[0]["embedding"]
        
        # Clean up
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        
        return jsonify({'embedding': embedding})
        
    except ValueError as e:
        # DeepFace raises ValueError when no face detected
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        return jsonify({'error': 'No face detected in image'}), 400
        
    except Exception as e:
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
        print(f"Error generating embedding: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == "__main__":
    app.run(port=7002, debug=True)