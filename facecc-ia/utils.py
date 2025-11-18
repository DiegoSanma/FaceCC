import math,os,json,random, cv2
import torchvision.transforms as T
import numpy as np
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
def preprocess(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    tensor = transform(img_rgb).unsqueeze(0)  # [1, 3, 112, 112]
    return tensor

def get_embedding(model,img_tensor):
    with torch.no_grad():
        # El embedding debe salir del backbone, NO de la cabeza siamese
        emb = model.backbone(img_tensor)   # shape [1, 512]
        emb = emb.cpu().numpy().flatten()  # [512]
    return emb

def check_best(model,embedding):
    best_match = [None, math.inf]
    for file in os.listdir("Embeddings"):
        if file.endswith(".npy"):
            db_embedding = np.load(os.path.join("Embeddings", file))
            distance = torch.nn.functional.cosine_similarity(embedding, db_embedding, dim=0).item()
            print(f"Comparing with {file}, distance: {distance}")
            if best_match is None or distance < best_match[1]:
                best_match = (file, distance)
    if best_match[1] < 0.5:  # umbral de distancia
        with open(os.path.join("Embeddings", "labels.json"), "r", encoding="utf-8") as f:
            labels_dict = json.load(f)
        name = labels_dict.get(best_match[0], "Desconocido")
        return {'name': name}
    else:
        return json.dumps({'name': 'Desconocido'})
    