from app import app
from flask import request, jsonify
import torch
from utils import check_best, preprocess, get_embedding, SiameseArcFace
import numpy as np
import cv2
import os
import secrets
import tempfile
import time

model_face = 'best_siamese_arcface.pt'
model = SiameseArcFace()
model.load_state_dict(torch.load(model_face, map_location=torch.device('cpu')))
model.eval()


@app.route('/facecc/facecc-ia/predict', methods=['POST'])    
def identify_face():
    t0 = time.time()
    file = request.files['file']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print("toy aqui")
    t1 = time.time()
    # Preprocesar → tensor
    img_tensor = preprocess(img_cv2)

    # Embedding con tu modelo torch
    embedding = get_embedding(model, img_tensor)

    print("sali del embedding")

    
    t2 = time.time()
    print(f"Embedding generation: {(t2-t1)*1000:.0f}ms")
    
    best_match = check_best(model,embedding)
    t3 = time.time()
    print(f"Matching: {(t3-t2)*1000:.0f}ms")
    print(f"Total: {(t3-t0)*1000:.0f}ms")

    return best_match


@app.route('/facecc/facecc-ia/embed', methods=['POST'])
def generate_embedding():
    """Generate embedding for admin face registration"""
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400

    file = request.files['file']
    temp_path = None

    try:
        # Guardar archivo temporal
        temp_dir = tempfile.gettempdir()
        temp_filename = f"{secrets.token_hex(8)}.jpg"
        temp_path = os.path.join(temp_dir, temp_filename)
        file.save(temp_path)

        # Leer imagen como BGR
        img_bgr = cv2.imread(temp_path)
        if img_bgr is None:
            return jsonify({'error': 'Invalid image'}), 400

        # Convertir a tensor y obtener embedding
        img_tensor = preprocess(img_bgr)

        with torch.no_grad():
            model.eval()
            embedding = model.backbone(img_tensor)
            embedding = embedding.squeeze().cpu().numpy().tolist()  # serializable

        # Borrar imagen temporal
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