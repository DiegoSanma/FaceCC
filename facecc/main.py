from app import app, IA_SERVER, IA_URL
from flask import Flask, render_template, request, jsonify
import base64 
import json
import pickle,os, io
import numpy as np
import requests, secrets
from deepface import DeepFace
from PIL import Image


@app.route('/facecc/facecc')
def user_camera():
    return render_template('index.html')

@app.route('/facecc/facecc/predict', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data.get("image", "Sin nombre")
    try:
        format, imgstr = image_data.split(';base64,') 
    except Exception as es:
        return jsonify({'error': 'Invalid image data format'}), 400

    reconocimiento = "Desconocido"
    image_data = base64.b64decode(imgstr)
    filename = secrets.token_hex(8) + ".jpg"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    # Guardar el binario como archivo
    with open(filepath, "wb") as f:
        f.write(image_data)
    print("Guardando archivo:", filepath)
    files = {'file': open(filepath, 'rb')}
    
    #Todo lo que viene aquí es para usar Deepface
    try:
        apicall = requests.post(IA_SERVER+IA_URL, files=files)
        if apicall.status_code == 200:
            response = apicall.json()
            reconocimiento = response.get('name', 'Desconocido')
        else:
            return jsonify({'error': 'Error in IA server response'}), 500
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 500

    files['file'].close()
    #Elimino la imagen que me llego
    if os.path.exists(filepath):
        os.remove(filepath)
    return jsonify({'name': reconocimiento})
    

if __name__ == '__main__':
    app.run(port=7001)
        
