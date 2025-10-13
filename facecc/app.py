from flask import Flask, render_template, request, jsonify
import base64 
import json

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
    #image_file = ContentFile(base64.b64decode(imgstr), name='captured_image.' + ext)
    
    # Aquí la idea sería realiazar el procesamiento de la imagen y ver si coincide con alguna cara
    reconomiento = "Diego"
    #Luego la entrego al fetch del javascript para que que se vea el resultado
    return jsonify({'name': reconomiento})
    

if __name__ == '__main__':
    app.run(debug=True)
        
