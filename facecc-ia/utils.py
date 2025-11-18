import math, os, json, random
import numpy as np
import joblib

# Get absolute path to Embeddings directory
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "Embeddings")
LABELS_FILE = os.path.join(EMBEDDINGS_DIR, "labels.json")

# Load fine-tuned model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "finetuned_facerecognition_classifier.joblib")
svc_model = joblib.load(MODEL_PATH)

def check_best(embedding):
    best_match = [None, math.inf]
    
    if os.path.exists(LABELS_FILE):
        with open(LABELS_FILE, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    else:
        return json.dumps({'name': 'Desconocido'})
    
    for file in os.listdir(EMBEDDINGS_DIR):
        if file.endswith(".npy"):
            db_embedding = np.load(os.path.join(EMBEDDINGS_DIR, file))
            # Use fine-tuned model for distance calculation
            distance = 1 - svc_model.predict_proba([np.concatenate([db_embedding, embedding])])[0][1]
            
            if distance < best_match[1]:
                best_match = (file, distance)
    
    if best_match[1] < 1:  # umbral de distancia
        name = labels.get(best_match[0], "Desconocido")
    else:
        name = 'Desconocido'
    
    return json.dumps({'name': name})