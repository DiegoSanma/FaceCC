from app import app
from flask import request, jsonify
from deepface import DeepFace
from utils import check_best
import numpy as np
import cv2
from PIL import Image

@app.route('/facecc/facecc-ia/predict', methods=['POST'])    
def identify_face():
    files = request.files['file']
    img_bytes = files.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img_cv2 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    embedding = DeepFace.represent(img_path=img_cv2,
                               model_name = 'ArcFace',
                           enforce_detection = False)[0]["embedding"]
    best_match = check_best(embedding)
    files.close()
    return(best_match)

if __name__ == "__main__":
    app.run(port=7002)