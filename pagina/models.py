from django.db import models
from django.contrib.auth.models import AbstractUser
import json # Necesario si vas a usar JSONField

# Create your models here.

class Usuario(AbstractUser):
    nombre_completo = models.CharField(max_length=150)
    correo = models.EmailField(unique=True)
    carrera = models.CharField(max_length=100)
    carnet = models.CharField(max_length=20, unique=True)
    ciclo = models.CharField(max_length=20)
    avatar = models.ImageField(upload_to='avatars/', default='avatars/default.jpg')
    bio = models.TextField(blank=True, null=True)

    USERNAME_FIELD = 'username'
    REQUIRED_FIELDS = ['nombre_completo', 'correo', 'carrera', 'carnet', 'ciclo']

    def __str__(self):
        return self.username

class Historial(models.Model):
    usuario = models.ForeignKey(Usuario, on_delete=models.CASCADE)
    fecha = models.DateTimeField(auto_now_add=True)
    ecuacion = models.TextField()
    metodo = models.CharField(max_length=50)
    resultado = models.TextField()
    pasos = models.TextField(blank=True, null=True) # Habilitado para ser nulo y vacío
    iteraciones = models.PositiveIntegerField(null=True, blank=True) # Habilitado para ser nulo y vacío
    x0 = models.FloatField(null=True, blank=True) # Habilitado para ser nulo y vacío
    x1 = models.FloatField(null=True, blank=True) # Habilitado para ser nulo y vacío
    x2 = models.FloatField(null=True, blank=True) # Habilitado para ser nulo y vacío
    orden = models.IntegerField(null=True, blank=True)

    # Campos específicos para diferenciar y almacenar datos del método Simplex
    METODO_CHOICES = (
        ('Numerico', 'Método Numérico'),
        ('Simplex', 'Método Simplex'),
    )
    metodo_tipo = models.CharField(max_length=20, choices=METODO_CHOICES, default='Numerico')
    parametros_simplex = models.JSONField(null=True, blank=True) # Para guardar la función objetivo y restricciones

    def __str__(self):
        return f'{self.usuario.username} - {self.metodo} - {self.fecha.strftime("%Y-%m-%d %H:%M")}'