import math,os,json,random
import numpy as np
import torch
from insightface.app import FaceAnalysis

def check_best(embedding):
    best_match = [None, math.inf]
    for file in os.listdir("Embeddings"):
        if file.endswith(".npy"):
            db_embedding = np.load(os.path.join("Embeddings", file))
            distance = 1 - np.dot(embedding, db_embedding) / (np.linalg.norm(embedding) * np.linalg.norm(db_embedding))
            if best_match is None or distance < best_match[1]:
                best_match = (file, distance)
    if best_match[1] < 0.6:  # umbral de distancia
        with open(os.path.join("Embeddings", "labels.json"), "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
        name = labels_dict.get(best_match[0], "Desconocido")
        return {'name': name}
    else:
        return {'name': 'Desconocido'}
    