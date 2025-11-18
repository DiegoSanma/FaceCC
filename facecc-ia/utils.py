import math,os,json,random, cv2
import torchvision.transforms as T
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models

class SiameseArcFace(nn.Module):
    def __init__(self, embedding_size=512, backbone_pretrained=True):
        super().__init__()

        # ResNet backbone (pre-entrenado)
        backbone = models.resnet50(pretrained=backbone_pretrained)
        in_features = backbone.fc.in_features
        backbone.fc = nn.Identity()
        self.backbone = backbone

        # Layer para ajustar a embedding ArcFace
        self.embedding = nn.Linear(in_features, embedding_size)

        # Clasificador binario para similitud
        self.classifier = nn.Sequential(
            nn.Linear(embedding_size * 2, 512),
            nn.ReLU(),
            nn.Linear(512, 1)
        )

        # 👉 Congelar el backbone (finetuning)
        for param in self.backbone.parameters():
            param.requires_grad = True

    def encode(self, img):
        with torch.no_grad():                      # backbone congelado
            feat = self.backbone(img)
        emb = self.embedding(feat)                 # se entrena
        return F.normalize(emb, p=2, dim=1)

    def forward(self, img1, img2):
        emb1 = self.encode(img1)
        emb2 = self.encode(img2)

        pair = torch.cat([emb1, emb2], dim=1)
        logit = self.classifier(pair)              # 🔥 se entrena
        return logit.squeeze(), emb1, emb2

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])
def preprocess(img_cv2):
    # img_cv2 viene de cv2.imread en BGR
    img_rgb = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)  # convertir a PIL

    tensor = transform(img_pil)  # ahora sí funciona el transform
    tensor = tensor.unsqueeze(0)  # [1, 3, 112, 112]
    return tensor

def get_embedding(model,img_tensor):
    with torch.no_grad():
        # El embedding debe salir del backbone, NO de la cabeza siamese
        emb = model.backbone(img_tensor)   # shape [1, 512]
        emb = emb.cpu().numpy().flatten()  # [512]
    return emb


def check_best(model, embedding, threshold=0.67):
    """
    embedding: tensor de tamaño [D]
    """
    best_match = (None, 0)

    # asegurar tensor en CPU
    embedding = embedding.cpu() if isinstance(embedding, torch.Tensor) else torch.tensor(embedding)

    for file in os.listdir("TUI_embeddings"):
        if file.endswith(".npy"):
            db_embedding = np.load(os.path.join("TUI_embeddings", file))   # numpy -> tensor
            db_embedding = torch.tensor(db_embedding, dtype=torch.float32)

            # calcular similitud coseno como distancia
            distance = torch.nn.functional.cosine_similarity(
                embedding, db_embedding, dim=0
            ).item()

            print(f"Comparing with {file}, distance: {distance}")

            # guardar el mejor (menor distancia)
            if distance > best_match[1]:
                best_match = (file, distance)

    file_name, dist = best_match

    # si no hay match o sobre umbral → desconocido
    if file_name is None or dist < threshold:
        return {"name": "Desconocido"}

    # cargar nombres desde labels.json si existe
    labels_path = os.path.join("TUI_embeddings", "labels.json")
    if os.path.exists(labels_path):
        with open(labels_path, "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
        name = labels_dict.get(file_name, "Desconocido")
        print(f"Found label in JSON: {name}")
    else:
        name = file_name.replace(".npy", "")

    return {"name": name, "distance": float(dist)}