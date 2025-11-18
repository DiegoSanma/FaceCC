from app import app
from flask import request, jsonify
import torch
from utils import check_best, preprocess, get_embedding, SiameseArcFace
import numpy as np
import cv2, joblib
# from PIL import Image

model_face = '"best_siamese_arcface.pt"'
model = SiameseArcFace()
model.load_state_dict(torch.load(model_face, map_location=torch.device('cpu')))
model.eval()


@app.route('/facecc/facecc-ia/predict', methods=['POST'])    
def identify_face():
    file = request.files['file']
    img_bytes = file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    print("toy aqui")

    # Preprocesar → tensor
    img_tensor = preprocess(img_cv2)

    # Embedding con tu modelo torch
    embedding = get_embedding(model, img_tensor)

    print("sali del embedding")

    # Buscar mejor match
    best_match = check_best(model,embedding)

    return best_match


if __name__ == "__main__":
    app.run(port=7002)