from main import app, IA_SERVER, IA_URL
from flask import Flask, render_template, request, jsonify
import base64 
import json
import pickle,os, io
import numpy as np
import requests
from deepface import DeepFace
from PIL import Image


@app.route('/facecc/facecc')
def user_camera():
    return render_template('index.html')

@app.route('facecc/facecc/predict', methods=['POST'])
def process_image():
    data = request.get_json()
    image_data = data.get("image", "Sin nombre")
    try:
        format, imgstr = image_data.split(';base64,') 
    except Exception as es:
        return jsonify({'error': 'Invalid image data format'}), 400
    image_file = base64.b64decode(imgstr)
    img_pil = Image.open(io.BytesIO(image_file))
    reconocimiento = "Desconocido"
    #Todo lo que viene aquí es para usar Deepface
    try:
        apicall = requests.post(IA_SERVER+IA_URL, img_pil=img_pil)
        if apicall.status_code == 200:
            response = apicall.json()
            reconocimiento = response.get('name', 'Desconocido')
        else:
            return jsonify({'error': 'Error in IA server response'}), 500
    except Exception as e:
        print(f"Error processing image: {e}")
        return jsonify({'error': 'Error processing image'}), 500
        # Aquí la idea sería realiazar el procesamiento de la imagen y ver si coincide con alguna cara
        #Luego la entrego al fetch del javascript para que que se vea el resultado
    return jsonify({'name': reconocimiento})
    

if __name__ == '__main__':
    app.run(port=7001)
        
