from django.urls import path
from . import views

app_name = 'app_reconocimiento'

urlpatterns = [
    path('', views.user_camera, name='user_camera'),
    #path('upload_photo/', views.upload_photo, name='upload_photo'),
    #path('supervisor/', views.supervisor_view, name='supervisor'),
]
