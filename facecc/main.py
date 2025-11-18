from app import app, IA_SERVER, IA_URL
# from flask import Flask, render_template, request, jsonify
from flask import render_template, request, jsonify
import base64
# import json
# import pickle,os, io
import os
# import numpy as np
import requests, secrets
print(app.template_folder)

import json
import logging
from datetime import datetime
from werkzeug.utils import secure_filename
from io import BytesIO
import numpy as np

from scipy.spatial.distance import cosine # TODO Needs scipy

SYSTEM_STATE_FILE = os.path.join(os.path.dirname(__file__), '..', 'system_state.json')
RECOGNITION_LOG_FILE = os.path.join(os.path.dirname(__file__), '..', 'logs', 'recognition.log')
IA_EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), '..', 'facecc-ia', 'TUI_embeddings')
IA_LABELS_FILE = os.path.join(IA_EMBEDDINGS_DIR, 'labels.json')

os.makedirs(os.path.dirname(SYSTEM_STATE_FILE), exist_ok=True)
os.makedirs(os.path.dirname(RECOGNITION_LOG_FILE), exist_ok=True)
os.makedirs(IA_EMBEDDINGS_DIR, exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

recognition_logger = logging.getLogger('recognition')
recognition_logger.setLevel(logging.INFO)
handler = logging.FileHandler(RECOGNITION_LOG_FILE)
handler.setFormatter(logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S'))
recognition_logger.addHandler(handler)

def load_system_state():
    if os.path.exists(SYSTEM_STATE_FILE):
        with open(SYSTEM_STATE_FILE, 'r') as f:
            return json.load(f)
    return {"recognition_enabled": True}

def save_system_state(state):
    with open(SYSTEM_STATE_FILE, 'w') as f:
        json.dump(state, f)

# ===

@app.route('/facecc/facecc')
def user_camera():
    print('Entre aqui')
    return render_template('index.html')

@app.route('/admin')
def admin_panel():
    return render_template('admin.html')

@app.route('/facecc/facecc/predict', methods=['POST'])
def process_image():
    
    state = load_system_state()
    if not state.get('recognition_enabled', True):
        return jsonify({'name': 'Sistema desactivado'}), 200
    
    data = request.get_json()
    image_data = data.get("image", "Sin nombre")
    try:
        format, imgstr = image_data.split(';base64,')
    except Exception as es:
        return jsonify({'error': 'Invalid image data format'}), 400

    reconocimiento = "Desconocido"
    image_data = base64.b64decode(imgstr)
    filename = secrets.token_hex(8) + ".jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Guardar el binario como archivo
    with open(filepath, "wb") as f:
        f.write(image_data)
    print("Guardando archivo:", filepath)
    # files = {'file': open(filepath, 'rb')}

    #Todo lo que viene aquí es para usar Deepface
    try:
        with open(filepath, 'rb') as fh:
            files = {'file': fh}
            apicall = requests.post(IA_SERVER + IA_URL, files=files, timeout=10)
        # apicall = requests.post(IA_SERVER+IA_URL, files=files)
        if apicall.status_code == 200:
            response = apicall.json()
            reconocimiento = response.get('name', 'Desconocido')
            # Log recognition event
            recognition_logger.info(f"Recognized: {reconocimiento}")
        else:
            return jsonify({'error': 'Error in IA server response'}), 502 # 500 -> 502
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 500
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
    
    return jsonify({'name': reconocimiento})


# Admin API endpoints
@app.route('/admin/system-status', methods=['GET'])
def get_system_status():
    state = load_system_state()
    return jsonify(state)

@app.route('/admin/system-toggle', methods=['POST'])
def toggle_system():
    state = load_system_state()
    state['recognition_enabled'] = not state.get('recognition_enabled', True)
    save_system_state(state)
    recognition_logger.info(f"System {'enabled' if state['recognition_enabled'] else 'disabled'}")
    return jsonify(state)

@app.route('/admin/faces', methods=['GET'])
def list_faces():
    try:
        with open(IA_LABELS_FILE, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        # Convert to list format for frontend
        faces = [{"id": k, "name": v} for k, v in labels.items()]
        return jsonify(faces)
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/faces', methods=['POST'])
def add_face():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    name = request.form.get('name', '').strip()
    
    if not name:
        return jsonify({'error': 'Name is required'}), 400
    
    if file.filename == '' or file.filename == 'Sin nombre':
        return jsonify({'error': 'No file selected'}), 400
    
    # Validate file type
    allowed_extensions = {'.jpg', '.jpeg', '.png'}
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in allowed_extensions:
        return jsonify({'error': 'Invalid file type. Use JPG or PNG'}), 400
    
    # Check file size (5MB limit)
    file.seek(0, os.SEEK_END)
    size = file.tell()
    if size > 5 * 1024 * 1024:
        return jsonify({'error': 'File too large (max 5MB)'}), 400
    file.seek(0)
    
    temp_path = None
    try:
        # Save temp file
        temp_filename = secrets.token_hex(8) + ext
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], temp_filename)
        file.save(temp_path)
        
        # Generate embedding using IA server
        with open(temp_path, 'rb') as fh:
            files = {'file': fh}
            # Call embeddings endpoint (we'll create this in facecc-ia)
            apicall = requests.post(IA_SERVER + '/facecc/facecc-ia/embed', files=files, timeout=15)
        
        if apicall.status_code != 200:
            error_msg = apicall.json().get('error', 'Unknown error')
            return jsonify({'error': f'Face detection failed: {error_msg}'}), 400
        
        new_embedding = apicall.json()['embedding']
        
        # Save embedding
        safe_name = ''.join(c if c.isalnum() or c == ' ' else '' for c in name.lower())
        safe_name = '_'.join(safe_name.split())  # replace spaces with underscores
        npy_filename = f"{safe_name}.npy"
        npy_path = os.path.join(IA_EMBEDDINGS_DIR, npy_filename)
        
        # Check if name exists
        if os.path.exists(IA_LABELS_FILE):
            with open(IA_LABELS_FILE, 'r', encoding='utf-8') as f:
                labels = json.load(f)
        else:
            labels = {}
        
        if npy_filename in labels:
            # os.remove(temp_path)
            return jsonify({'error': 'Name already exists'}), 400
        
        # region Duplicate check | Requires scipy
        # Check for duplicate face (compare embeddings)
        SIMILARITY_THRESHOLD = 0.9  # adjust based on testing (lower = stricter)
        
        for existing_file in os.listdir(IA_EMBEDDINGS_DIR):
            if existing_file.endswith('.npy'):
                existing_path = os.path.join(IA_EMBEDDINGS_DIR, existing_file)
                existing_embedding = np.load(existing_path)
                
                # Calculate cosine similarity (1 - cosine distance)
                similarity = 1 - cosine(new_embedding, existing_embedding)
                
                if similarity > SIMILARITY_THRESHOLD:
                    existing_name = labels.get(existing_file, 'Unknown')
                    return jsonify({
                        'error': f'Face already registered as "{existing_name}" (similarity: {similarity:.2f})'
                    }), 400
        # endregion
        
        # Save embedding and update labels
        np.save(npy_path, new_embedding)
        labels[npy_filename] = name
        
        with open(IA_LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=4)
        
        recognition_logger.info(f"New face added: {name}")
        
        return jsonify({'success': True, 'id': npy_filename, 'name': name})
    
    except Exception as e:
        print(f"Error adding face: {e}")
        return jsonify({'error': str(e)}), 500
    
    finally:
        # Always clean up temp file
        if temp_path and os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except Exception as cleanup_err:
                print(f"Warning: could not remove temp file: {cleanup_err}")

@app.route('/admin/faces/<face_id>', methods=['DELETE'])
def delete_face(face_id):
    try:
        # Remove .npy file
        npy_path = os.path.join(IA_EMBEDDINGS_DIR, face_id)
        if os.path.exists(npy_path):
            os.remove(npy_path)
        
        # Update labels
        with open(IA_LABELS_FILE, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        
        name = labels.pop(face_id, 'Unknown')
        
        with open(IA_LABELS_FILE, 'w', encoding='utf-8') as f:
            json.dump(labels, f, ensure_ascii=False, indent=4)
        
        recognition_logger.info(f"Face deleted: {name}")
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/admin/logs', methods=['GET'])
def get_logs():
    try:
        lines = int(request.args.get('lines', 50))
        if not os.path.exists(RECOGNITION_LOG_FILE):
            return jsonify([])
        
        with open(RECOGNITION_LOG_FILE, 'r') as f:
            all_lines = f.readlines()
        
        recent = all_lines[-lines:]
        return jsonify([line.strip() for line in recent])
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
   app.run(port=7001,debug=True)