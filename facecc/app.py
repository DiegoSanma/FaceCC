from flask import Flask, render_template, request, jsonify
import base64 
import json
import pickle,os, io
import numpy as np
import cv2
import tempfile
from deepface import DeepFace
from PIL import Image


app = Flask(__name__)

@app.route('/')
def user_camera():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data.get("image", "Sin nombre")
    try:
        format, imgstr = image_data.split(';base64,') 
    except Exception as es:
        return jsonify({'error': 'Invalid image data format'}), 400
    image_file = base64.b64decode(imgstr)
    img_pil = Image.open(io.BytesIO(image_file))

    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=True) as tmp:
        img_pil.save(tmp.name)

        prediccion = DeepFace.find(img_path=tmp.name, db_path = "CaraTui/",
                                   model_name = 'Facenet',
                               silent = False,
                               enforce_detection = False)
        reconocimiento = "Desconocido"
        if prediccion and not prediccion[0].empty:
            print(prediccion)
            df = prediccion[0]
            predicted_path = df['identity'].iloc[0]
            predicted_name = os.path.basename(predicted_path).split('.')[0]
            distance = df['distance'].iloc[0]
            if distance < 0.6:
                reconocimiento = predicted_name
        # Aquí la idea sería realiazar el procesamiento de la imagen y ver si coincide con alguna cara
        #Luego la entrego al fetch del javascript para que que se vea el resultado
        return jsonify({'name': reconocimiento})
    

if __name__ == '__main__':
    app.run(debug=True)
        
