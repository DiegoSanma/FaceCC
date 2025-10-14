import math,os,json
import numpy as np

def check_best(embedding):
    best_match = [None, math.inf]
    for file in os.listdir("Embeddings"):
        if file.endswith(".npy"):
            db_embedding = np.load(os.path.join("Embeddings", file))
            distance = np.linalg.norm(embedding - db_embedding)
            if best_match is None or distance < best_match[1]:
                best_match = (file, distance)
    if best_match[1] < 10:  # umbral de distancia
        with open(os.path.join("Embeddings", "labels.json"), "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
        name = labels_dict.get(best_match[0], "Desconocido")
        return {'name': name}
    else:
        return {'name': 'Desconocido'}