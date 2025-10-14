from app import app, facenet_model
from flask import request, jsonify
import tempfile
from deepface import DeepFace
import os, io, base64
import numpy as np
import cv2
from PIL import Image

@app.route('/facecc/facecc-ia/predict', methods=['POST'])    
def identify_face():
    files = request.files['file']
    img_bytes = files.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    prediccion = DeepFace.find(img_path=img_cv2, db_path = "CaraTui/",
                               model_name = 'Facenet',
                           silent = False,
                           enforce_detection = False)
    if prediccion and not prediccion[0].empty:
        print(prediccion)
        df = prediccion[0]
        predicted_path = df['identity'].iloc[0]
        predicted_name = os.path.basename(predicted_path).split('.')[0]
        distance = df['distance'].iloc[0]
        if distance < 0.6:
            reconocimiento = predicted_name
        return {'name': reconocimiento}
    return {'name': 'Desconocido'}

if __name__ == "__main__":
    app.run(port=7002)