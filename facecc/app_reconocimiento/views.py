from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required   
from django.core.files.base import ContentFile
import base64 
from django.http import JsonResponse
import json

def user_camera(request):
    return render(request, 'index.html')

def process_image(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode('utf-8'))
        image_data = data.get("image", "Sin nombre")
        format, imgstr = image_data.split(';base64,') 
        ext = format.split('/')[-1] 
        image_file = ContentFile(base64.b64decode(imgstr), name='captured_image.' + ext)
        
        # Aquí la idea sería realiazar el procesamiento de la imagen y ver si coincide con alguna cara
        reconomiento = "Diego"
        #Luego la entrego al fetch del javascript para que que se vea el resultado

        return JsonResponse({'status': 'success', 'name': reconomiento})
        
