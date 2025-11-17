import math,os,json
import numpy as np
from scipy.spatial.distance import cosine

# Get absolute path to Embeddings directory
EMBEDDINGS_DIR = os.path.join(os.path.dirname(__file__), "Embeddings")
LABELS_FILE = os.path.join(EMBEDDINGS_DIR, "labels.json")

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
            distance = 1 - np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
            # distance = cosine(embedding, db_embedding)
            # if best_match is None or distance < best_match[1]:
            if distance < best_match[1]:
                best_match = (file, distance)
                
    if best_match[1] < 0.6:  # umbral de distancia
        # with open(os.path.join(EMBEDDINGS_DIR, "labels.json"), "r", encoding="utf-8") as f:
        #     labels_dict = json.load(f)
        # name = labels_dict.get(best_match[0], "Desconocido")
        name = labels.get(best_match[0], "Desconocido")
    else:
        name = 'Desconocido'
        
    return json.dumps({'name': name})