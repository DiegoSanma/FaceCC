from django.urls import path
from . import views

app_name = 'app_reconocimiento'

urlpatterns = [
    path('', views.user_camera, name='user_camera'),
    path('process_image/', views.process_image, name='process_image'),
    
    path('admin/', views.admin_view, name='admin_view'),
    #path('upload_photo/', views.upload_photo, name='upload_photo'),
    #path('supervisor/', views.supervisor_view, name='supervisor'),
]
