#La idea es que en este código se extraen los datos de entrenamiento y guardarlos como embeddings
import webdataset as wds
import tarfile
import random
from io import BytesIO
from PIL import Image
import numpy as np
import torch
from facenet_pytorch import InceptionResnetV1, MTCNN

#De aquí, extraigo de dataset.tar las caras con su etiqueta (jpg->imagen, cls->nombre de la persona)
dataset = list(wds.WebDataset("dataset_entry.tar").decode("pil").to_tuple("jpg", "cls"))

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Función para separar entre entrenamiento y validación
def split_dataset(dataset, train_ratio=0.8):
    random.shuffle(dataset)
    train_size = int(len(dataset) * train_ratio)
    train_set = dataset[:train_size]
    val_set = dataset[train_size:]
    return train_set, val_set


def save_embeddings_with_labels(tar_path_foto, mtcnn, facenet, device):
    """
    Extrae embeddings de todas las imágenes de un tar y devuelve un diccionario con embeddings y labels.
    
    Args:
        tar_path_foto: ruta al archivo .tar con imágenes
        mtcnn: detector de rostros (MTCNN)
        facenet: modelo preentrenado InceptionResnetV1
        device: 'cuda' o 'cpu'
    
    Returns:
        embeddings_dict: dict {imagen_nombre: embedding}
        labels_dict: dict {imagen_nombre: label}
    """
    embeddings_dict = {}
    labels_dict = {}

    with tarfile.open(tar_path_foto, "r") as tar:
        image_members = [m for m in tar.getmembers() if m.name.lower().endswith((".jpg", ".jpeg", ".png"))]

        for member in image_members:
            f = tar.extractfile(member)
            img = Image.open(BytesIO(f.read())).convert('RGB')

            # ---------- DETECTAR Y CORTAR ROSTRO ----------
            face = mtcnn(img)
            if face is not None:
                face_embedding = facenet(face.unsqueeze(0).to(device))
                emb = face_embedding.detach().cpu().numpy().flatten()
                embeddings_dict[member.name] = emb

                # Obtener label desde el nombre de archivo (por ejemplo carpeta)
                label = member.name.split("/")[0]  # "persona1/imagen1.jpg" -> "persona1"
                labels_dict[member.name] = label

                print(f"Procesada: {member.name} (Label: {label})")
            else:
                print(f"No se detectó rostro en {member.name}")

    # Guardar embeddings y labels juntos
    np.savez("embeddings_labels.npz", **embeddings_dict, **labels_dict)
    print("Embeddings y labels guardados en embeddings_labels.npz")

#Entrenamiento del modelo, finetuning
def train_model_embeddings(train_set, facenet, mtcnn, device, epochs=10, batch_size=4, learning_rate=0.001):
    facenet.train()
    optimizer = torch.optim.Adam(facenet.parameters(), lr=learning_rate)
    criterion = torch.nn.CrossEntropyLoss()

    for epoch in range(epochs):
        random.shuffle(train_set)
        for i in range(0, len(train_set), batch_size):
            batch = train_set[i:i+batch_size]
            images = []
            labels = []
            for img, label in batch:
                face = mtcnn(img)
                if face is not None:
                    images.append(face)
                    labels.append(label)
            if len(images) == 0:
                continue
            images = torch.stack(images).to(device)
            # Convertir etiquetas a índices
            unique_labels = list(set(labels))
            label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}
            labels_idx = torch.tensor([label_to_idx[label] for label in labels]).to(device)

            optimizer.zero_grad()
            outputs = facenet(images)
            loss = criterion(outputs, labels_idx)
            loss.backward()
            optimizer.step()

        print(f"Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}")

    return facenet