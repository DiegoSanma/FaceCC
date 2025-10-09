#Entrenamiento y evaluacion del modelo
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN
import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import random
import tarfile
from io import BytesIO
from PIL import Image
import webdataset as wds

# Cargar el dataset
dataset = list(wds.WebDataset("dataset.tar").decode("pil").to_tuple("jpg", "cls"))
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def load_embeddings(embedding_file):
    data = np.load(embedding_file)
    embeddings = []
    labels = []
    for key in data:
        embeddings.append(data[key])
        labels.append(key.split('/')[0])  # Asumiendo que la etiqueta es el nombre del directorio
    return np.array(embeddings), np.array(labels)

def compare_embeddings(emb1, emb2):
    return np.linalg.norm(emb1 - emb2)

def evaluate_model(val_set,model,mt,test_size=0.2, random_state=42):
    #Cargo embeddings sacadaos de u-cursos (con los nombres)
    embeddings, labels = load_embeddings("embeddings_.npz")

    #Obtengo los embeddings de validación
    val_embeddings = []
    val_labels = []
    for img, label in val_set:
        face = mt(img)
        if face is not None:
            with torch.no_grad():
                emb = model(face.unsqueeze(0).to(device))
            val_embeddings.append(emb.cpu().numpy().flatten())
            val_labels.append(label)
    val_embeddings = np.array(val_embeddings)
    val_labels = np.array(val_labels)

    # Ahora debo comparar los embeddings de validación con los del dataset
    y_true = []
    y_pred = []
    for i, val_emb in enumerate(val_embeddings):
        min_dist = float('inf')
        pred_label = None
        for j, emb in enumerate(embeddings):
            dist = compare_embeddings(val_emb, emb)
            if dist < min_dist:
                min_dist = dist
                pred_label = labels[j]
        y_true.append(val_labels[i])
        y_pred.append(pred_label)
    # Calculo métricas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")


