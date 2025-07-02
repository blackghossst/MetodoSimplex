from django.urls import path
from . import views

urlpatterns = [
    path('', views.inicio, name='inicio'),
    path('registro/', views.registro, name='registro'),
    path('login/', views.login_view, name='login'),
    path('logout/', views.logout_view, name='logout'),
    path('perfil/', views.perfil, name='perfil'),
    path('perfil/editar/', views.editar_perfil, name='editar_perfil'),
    path('perfil/eliminar/', views.mostrar_formulario_eliminar_perfil, name='mostrar_formulario_eliminar_perfil'),
    path('perfil/eliminar/confirmar/', views.eliminar_perfil_confirmar, name='eliminar_perfil_confirmar'),
    path('perfil/cambiar_foto/', views.cambiar_foto, name='cambiar_foto'),

    # URL para la validación AJAX
    path('validar-campos/', views.validar_campos, name='validar_campos'),

    # Vistas de los métodos numéricos
    path('metodo-simplex/', views.metodo_simplex_view, name='metodo_simplex'),

    # Historial de cálculos
    path('historial/', views.historial, name='historial'),
    path('historial/<int:historial_id>/', views.detalle_historial, name='detalle_historial'),
    path('historial/eliminar/<int:historial_id>/', views.eliminar_historial, name='eliminar_historial'),
    
    path('metodo-simplex/cargar-ejemplo/', views.load_example_simplex, name='load_example_simplex'),
]