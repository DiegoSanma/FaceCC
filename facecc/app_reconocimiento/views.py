from django.shortcuts import render
from django.shortcuts import render, redirect
from django.contrib.auth.decorators import login_required   
from django.core.files.base import ContentFile
import base64  

def user_camera(request):
    return render(request, 'index.html')
