from django import forms
from django.contrib.auth.forms import UserCreationForm
from .models import Usuario



class RegistroForm(UserCreationForm):
    class Meta:
        model = Usuario
        fields = ['username', 'nombre_completo', 'correo', 'carrera', 'carnet', 'ciclo', 'avatar', 'password1', 'password2']

    def clean_username(self):
        username = self.cleaned_data.get('username')
        if Usuario.objects.filter(username=username).exists():
            raise forms.ValidationError("El nombre de usuario ya está en uso.")
        return username

    def clean_correo(self):
        correo = self.cleaned_data.get('correo')
        if Usuario.objects.filter(correo=correo).exists():
            raise forms.ValidationError("El correo electrónico ya está en uso.")
        return correo

    def clean_carnet(self):
        carnet = self.cleaned_data.get('carnet')
        if Usuario.objects.filter(carnet=carnet).exists():
            raise forms.ValidationError("El carnet ya está registrado.")
        return carnet

    

class PerfilForm(forms.ModelForm):
    class Meta:
        model = Usuario
        fields = ['nombre_completo', 'correo', 'carrera', 'carnet', 'ciclo', 'avatar']


class ConfirmacionContrasenaForm(forms.Form):
    password = forms.CharField(
        widget=forms.PasswordInput(attrs={'placeholder': 'Introduce tu contraseña', 'class': 'form-control'}),
        label='Contraseña',
        required=True
    )
    
    def __init__(self, *args, **kwargs):
        self.user = kwargs.pop('user', None)
        super().__init__(*args, **kwargs)

    def clean_password(self): 
        contrasena = self.cleaned_data.get('password')
        if self.user and not self.user.check_password(contrasena):
            raise forms.ValidationError("La contraseña ingresada no es correcta.")
        return contrasena