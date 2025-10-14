# -*- coding: utf-8 -*- 
import json
from deepface import DeepFace
from flask import Flask

app = Flask(__name__)

# Ver otros modelos en https://docs.pytorch.org/vision/main/models.html
facenet_model = DeepFace.build_model("Facenet")
