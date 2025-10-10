from django.db import models
from django.db import models
from django.contrib.auth.models import User

class FaceEmbedding(models.Model):
    person_name = models.CharField(max_length=100)
    embedding = models.BinaryField()  # Guarda vector serializado
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return self.user.username

