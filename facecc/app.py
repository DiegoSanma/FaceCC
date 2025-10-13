from flask import Flask, render_template, request, jsonify
import base64 
import json
import pickle,os
import numpy as np
import cv2
from deepface import DeepFace

app = Flask(__name__)

@app.route('/')
def user_camera():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data.get("image", "Sin nombre")
    format, imgstr = image_data.split(';base64,') 
    ext = format.split('/')[-1] 
    image_file = base64.b64decode(imgstr)
    nparr = np.frombuffer(image_file, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    prediccion = DeepFace.find(img_path=img, db_path = "CaraTui",
                               model_name = 'Facenet',
                               silent = True,
                               enforce_detection = False)
    reconocimiento = "Desconocido"
    if prediccion and not prediccion[0].empty:
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
        
