import os
import numpy as np
import json
from deepface import DeepFace

img_folder = "CaraTui"
embedding_folder = "Embeddings"
os.makedirs(embedding_folder, exist_ok=True)

labels_dict = {}  # aquí guardaremos label bonito

facenet_model = DeepFace.build_model("Facenet")

for filename in os.listdir(img_folder):
    if filename.lower().endswith((".jpg", ".png", ".jpeg")):
        path = os.path.join(img_folder, filename)
        embedding = DeepFace.represent(
            img_path=path,
            model_name="Facenet",
            enforce_detection=False
        )[0]["embedding"]
        
        name, _ = os.path.splitext(filename)
        # Guardamos embedding
        np.save(os.path.join(embedding_folder, f"{name}.npy"), embedding)
        # Guardamos label bonito
        labels_dict[f"{name}.npy"] = name  # aquí puedes poner "Alice Smith" o cualquier formato

# Guardar diccionario en JSON
with open(os.path.join(embedding_folder, "labels.json"), "w", encoding="utf-8") as f:
    json.dump(labels_dict, f, ensure_ascii=False, indent=4)
print("Embeddings y etiquetas guardadas en", embedding_folder)