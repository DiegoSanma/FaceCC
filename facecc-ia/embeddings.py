import os
import torch
import torchvision.transforms as T
import numpy as np
from torchvision import transforms
from utils import SiameseArcFace
from PIL import Image

# -------- CONFIG --------
CARPETA_IMAGENES = "./CaraTUI"
CARPETA_EMBEDDINGS = "./TUI_embeddings"
MODELO_PATH = "best_siamese_arcface.pt"
IMG_SIZE = (224, 224)   # cámbialo si tu modelo usa otro tamaño
# ------------------------

device = "cuda" if torch.cuda.is_available() else "cpu"

model = SiameseArcFace()

# cargar modelo
model.load_state_dict(torch.load(MODELO_PATH, map_location="cpu"))
model.eval()
model.to(device)
torch.set_grad_enabled(False)

# transformaciones
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

# crear carpeta de embeddings si no existe
os.makedirs(CARPETA_EMBEDDINGS, exist_ok=True)

# recorrer imágenes
for img_name in os.listdir(CARPETA_IMAGENES):
    ruta = os.path.join(CARPETA_IMAGENES, img_name)

    try:
        img = Image.open(ruta).convert("RGB")
    except:
        print(f"❌ No se pudo abrir: {img_name}")
        continue

    tensor = transform(img).unsqueeze(0).to(device)

    # generar embedding (ajústalo si tu modelo usa otra forma de forward)
    embedding = model.backbone(tensor)  
    embedding = embedding.squeeze().cpu().numpy()

    # nombre de guardado .npy
    nombre_archivo = os.path.splitext(img_name)[0] + ".npy"
    ruta_guardado = os.path.join(CARPETA_EMBEDDINGS, nombre_archivo)

    # Guardar en npy
    np.save(ruta_guardado, embedding)

    print(f"💾 Guardado: {nombre_archivo}")

print("\n✨ Embeddings generados y guardados en ./TUI_embeddings/")
