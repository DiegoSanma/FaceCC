import math,os,json,random
import numpy as np
import joblib

svc_model = joblib.load("finetuned_facerecognition_classifier.joblib")

def check_best(embedding):
    best_match = [None, math.inf]
    for file in os.listdir("Embeddings"):
        if file.endswith(".npy"):
            db_embedding = np.load(os.path.join("Embeddings", file))
            distance = 1 - svc_model.predict_proba([np.concatenate([db_embedding, embedding])])[0][1]
            if best_match is None or distance < best_match[1]:
                best_match = (file, distance)
    if best_match[1] < 1:  # umbral de distancia
        with open(os.path.join("Embeddings", "labels.json"), "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
        name = labels_dict.get(best_match[0], "Desconocido")
        return {'name': name}
    else:
        return {'name': 'Desconocido'}
    