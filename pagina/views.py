from django.shortcuts import render, redirect
from django.contrib.auth import login, authenticate, logout
from .forms import RegistroForm, PerfilForm, ConfirmacionContrasenaForm
from django.contrib.auth.decorators import login_required
from django.contrib import messages
from django.shortcuts import get_object_or_404
from .models import Historial, Usuario
from django.http import JsonResponse, Http404
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import urllib, base64
from sympy import Symbol, lambdify, sympify, pi, E, integrate, latex
import json # Importar json para manejar los parámetros del simplex
from decimal import Decimal # Para cálculos de precisión en Simplex si se requiere
# from scipy.optimize import linprog # ¡Esta librería ha sido eliminada!

# --- Funciones Auxiliares (sin cambios) ---

def inicio(request):
    return render(request, 'pagina/inicio.html')

def validar_campos(request):
    username = request.GET.get('username', None)
    correo = request.GET.get('correo', None)
    carnet = request.GET.get('carnet', None)

    errores = {}

    if username and Usuario.objects.filter(username=username).exists():
        errores['username'] = 'El nombre de usuario ya está en uso.'

    if correo and Usuario.objects.filter(correo=correo).exists():
        errores['correo'] = 'El correo ya está registrado.'

    if carnet and Usuario.objects.filter(carnet=carnet).exists():
        errores['carnet'] = 'El carnet ya está en uso.'

    return JsonResponse({'errores': errores})

def registro(request):
    if request.method == 'POST':
        form = RegistroForm(request.POST, request.FILES)
        if form.is_valid():
            user = form.save()
            login(request, user)
            return redirect('resolver')
        else:
            return render(request, 'pagina/registro.html', {'form': form})
    else:
        form = RegistroForm()
    return render(request, 'pagina/registro.html', {'form': form})

def login_view(request):
    if request.method == 'POST':
        username = request.POST['username']
        password = request.POST['password']
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            return redirect('resolver')
        else:
            error = "Usuario o contraseña incorrectos"
            return render(request, 'pagina/login.html', {'error': error})
    return render(request, 'pagina/login.html')

def logout_view(request):
    logout(request)
    return redirect('login')

@login_required
def perfil(request):
    return render(request, 'pagina/perfil.html', {'usuario': request.user})

@login_required
def editar_perfil(request):
    if request.method == 'POST':
        form = PerfilForm(request.POST, request.FILES, instance=request.user)
        if form.is_valid():
            form.save()
            messages.success(request, '¡Tu perfil ha sido actualizado exitosamente!')
            return redirect('perfil')
        else:
            messages.error(request, 'Por favor, corrige los errores en el formulario.')
    else:
        form = PerfilForm(instance=request.user)
    return render(request, 'pagina/editar_perfil.html', {'form': form})


@login_required
def mostrar_formulario_eliminar_perfil(request):
    form = ConfirmacionContrasenaForm(user=request.user)
    return render(request, 'pagina/eliminar_perfil_confirmar.html', {'form': form})

@login_required
def eliminar_perfil_confirmar(request):
    if request.method == 'POST':
        form = ConfirmacionContrasenaForm(request.POST, user=request.user)
        if form.is_valid():
            request.user.delete()
            logout(request)
            messages.success(request, '¡Tu perfil ha sido eliminado exitosamente!')
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                return JsonResponse({'success': True, 'redirect_url': '/'})
            return redirect('/')
        else:
            if request.headers.get('x-requested-with') == 'XMLHttpRequest':
                errors = form.errors.as_json()
                return JsonResponse({'success': False, 'message': 'Errores de validación.', 'errors': errors})
            else:
                messages.error(request, 'Por favor, corrige los errores en el formulario.')
                return render(request, 'pagina/eliminar_perfil_confirmar.html', {'form': form})
    
    messages.error(request, 'Método no permitido para esta acción.')
    return redirect('mostrar_formulario_eliminar_perfil')

@login_required
def cambiar_foto(request):
    if request.method == 'POST' and request.FILES.get('avatar'):
        user = request.user
        user.avatar = request.FILES['avatar']
        user.save()
        return JsonResponse({'success': True, 'nueva_foto': user.avatar.url})
    return JsonResponse({'success': False, 'error': 'Error al actualizar la foto.'})

def corregir_ecuacion(equation):
    equation = re.sub(r'(\d)([a-zA-Z])', r'\1*\2', equation)
    equation = re.sub(r'([a-zA-Z])\(', r'\1*(', equation)
    equation = re.sub(r'\)([a-zA-Z])', r')*\1', equation)
    equation = re.sub(r'\)\(', r')*(', equation)
    return equation

def formato_respuesta(numero):
    return f"{numero:.4f}".rstrip('0').rstrip('.') if '.' in f"{numero:.4f}" else f"{numero}"

def resolver(request):
    return render(request, 'pagina/resolver.html')

from typing import Callable, Optional

def generar_grafico(funcion: Callable[[float], float], x_min: float = -10, x_max: float = 10) -> Optional[str]:
    # Asegúrate de que x_min y x_max sean números flotantes para evitar errores
    try:
        x_min = float(x_min)
        x_max = float(x_max)
    except (ValueError, TypeError):
        # Si no son convertibles, establece un rango predeterminado o maneja el error
        x_min = -10
        x_max = 10
        print("Advertencia: x_min o x_max no son válidos, usando rango predeterminado para la gráfica.")

    # Ajusta el rango si es demasiado pequeño o si min >= max
    if x_max - x_min < 0.1:
        x_center = (x_min + x_max) / 2
        x_min = x_center - 5
        x_max = x_center + 5
    
    x = np.linspace(x_min, x_max, 400)
    y = []

    try:
        for xi in x:
            y.append(funcion(xi))
    except Exception as e:
        print(f"Error al evaluar la función en el punto x={xi}: {e}")
        return None

    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label='f(x)')
    plt.axhline(0, color='black', linewidth=0.8)
    plt.axvline(0, color='black', linewidth=0.8)
    plt.title('Gráfica de la función')
    plt.xlabel('x')
    plt.ylabel('f(x)')
    plt.legend()
    plt.grid(True)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches='tight')
    buffer.seek(0)
    imagen_png = buffer.getvalue()
    buffer.close()

    grafico_base64 = base64.b64encode(imagen_png).decode('utf-8')
    plt.close()

    return grafico_base64

def detalle_historial(request, historial_id):
    historial = get_object_or_404(Historial, id=historial_id, usuario=request.user)
    
    # Decodificar parametros_simplex si existen y es un método Simplex
    parametros_simplex_json = None
    if historial.metodo_tipo == 'Simplex' and historial.parametros_simplex:
        try:
            parametros_simplex_json = json.loads(historial.parametros_simplex)
        except json.JSONDecodeError:
            parametros_simplex_json = {"error": "No se pudieron cargar los parámetros del Simplex."}

    return render(request, 'pagina/detalle_historial.html', {
        'historial': historial,
        'parametros_simplex_json': parametros_simplex_json # Pasar los parámetros decodificados
    })

# --- Método Simplex (Modificado para simular pasos) ---

def parse_simplex_input(func_obj_str, restrictions_str, is_maximize):
    """
    Parsea la función objetivo y las restricciones.
    Se espera:
    func_obj_str: "3x1 + 2x2"
    restrictions_str: "2x1 + x2 <= 10\nx1 + 3x2 <= 12\nx1 >= 0\nx2 >= 0"
    """
    try:
        max_var_idx = 0
        all_var_matches = re.findall(r'x(\d+)', func_obj_str + restrictions_str)
        if not all_var_matches:
            raise ValueError("No se encontraron variables (ej. x1, x2) en la función objetivo o restricciones.")
        for match in all_var_matches:
            idx = int(match)
            if idx > max_var_idx:
                max_var_idx = idx
        
        if max_var_idx == 0:
            raise ValueError("No se pudieron identificar las variables de decisión (ej. x1, x2, ...). Asegúrese de usar el formato correcto.")

        c = [0.0] * max_var_idx
        terms = re.findall(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)', func_obj_str.replace(' ', ''))
        
        for coeff_str, var_idx_str in terms:
            coeff = 1.0 if coeff_str == '' or coeff_str == '+' else -1.0 if coeff_str == '-' else float(coeff_str)
            var_idx = int(var_idx_str) - 1
            if 0 <= var_idx < max_var_idx:
                c[var_idx] = coeff
            else:
                raise ValueError(f"Variable x{var_idx+1} fuera de rango esperado.")

        A = []
        b = []
        bounds = [(0, None)] * max_var_idx

        raw_restrictions = [r.strip() for r in restrictions_str.split('\n') if r.strip()]
        
        for res_str in raw_restrictions:
            res_str_clean = res_str.replace(' ', '')

            if '>=' in res_str_clean:
                parts = res_str_clean.split('>=')
                lhs = parts[0]
                rhs = float(parts[1])
                
                match_non_neg = re.match(r'x(\d+)$', lhs)
                if match_non_neg:
                    idx = int(match_non_neg.group(1)) - 1
                    if 0 <= idx < max_var_idx:
                        bounds[idx] = (rhs, None)
                    continue

                res_coeffs = [0.0] * max_var_idx
                terms_lhs = re.findall(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)', lhs)
                for coeff_str, var_idx_str in terms_lhs:
                    coeff = 1.0 if coeff_str == '' or coeff_str == '+' else -1.0 if coeff_str == '-' else float(coeff_str)
                    var_idx = int(var_idx_str) - 1
                    if 0 <= var_idx < max_var_idx:
                        res_coeffs[var_idx] = -coeff
                A.append(res_coeffs)
                b.append(-rhs)
            
            elif '<=' in res_str_clean:
                parts = res_str_clean.split('<=')
                lhs = parts[0]
                rhs = float(parts[1])

                res_coeffs = [0.0] * max_var_idx
                terms_lhs = re.findall(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)', lhs)
                for coeff_str, var_idx_str in terms_lhs:
                    coeff = 1.0 if coeff_str == '' or coeff_str == '+' else -1.0 if coeff_str == '-' else float(coeff_str)
                    var_idx = int(var_idx_str) - 1
                    if 0 <= var_idx < max_var_idx:
                        res_coeffs[var_idx] = coeff
                A.append(res_coeffs)
                b.append(rhs)
            
            elif '=' in res_str_clean:
                parts = res_str_clean.split('=')
                lhs = parts[0]
                rhs = float(parts[1])

                res_coeffs_le = [0.0] * max_var_idx
                res_coeffs_ge = [0.0] * max_var_idx
                terms_lhs = re.findall(r'([+-]?\s*\d*\.?\d*)\s*x(\d+)', lhs)
                for coeff_str, var_idx_str in terms_lhs:
                    coeff = 1.0 if coeff_str == '' or coeff_str == '+' else -1.0 if coeff_str == '-' else float(coeff_str)
                    var_idx = int(var_idx_str) - 1
                    if 0 <= var_idx < max_var_idx:
                        res_coeffs_le[var_idx] = coeff
                        res_coeffs_ge[var_idx] = -coeff 

                A.append(res_coeffs_le)
                b.append(rhs)
                A.append(res_coeffs_ge)
                b.append(-rhs)
            else:
                raise ValueError(f"Formato de restricción no reconocido: {res_str}. Use <=, >=, o =.")

        return c, A, b, bounds, max_var_idx
    except Exception as e:
        raise ValueError(f"Error al parsear entrada: {e}. Asegúrate de usar el formato 'coefxN + coefxM' y 'val'.")


def simplex_solver(objective_coeffs, A_matrix, b_vector, is_maximize, num_vars):
    """
    Implementación completa del método Simplex que muestra las tablas por cada iteración
    con todos los procesos matemáticos detallados.
    """
    iteraciones = []
    
    try:
        # Convertir a arrays numpy para facilitar cálculos
        import numpy as np
        
        # Preparar los datos
        c = np.array(objective_coeffs, dtype=float)
        A = np.array(A_matrix, dtype=float)
        b = np.array(b_vector, dtype=float)
        
        # Si es maximización, convertir a minimización
        if is_maximize:
            c = -c
        
        # Número de variables originales y restricciones
        m, n = A.shape  # m = restricciones, n = variables originales
        
        # Agregar variables de holgura
        # Tabla inicial: [A | I | b] donde I es la matriz identidad
        tableau = np.zeros((m + 1, n + m + 1))
        
        # Llenar la matriz A
        tableau[:-1, :n] = A
        
        # Matriz identidad para variables de holgura
        for i in range(m):
            tableau[i, n + i] = 1
        
        # Vector b (lado derecho)
        tableau[:-1, -1] = b
        
        # Fila de la función objetivo
        tableau[-1, :n] = c
        
        # Variables básicas iniciales (variables de holgura)
        basic_vars = list(range(n, n + m))
        
        # Variables para nombres
        var_names = [f'x{i+1}' for i in range(n)] + [f's{i+1}' for i in range(m)]
        
        # --- Paso 1: Formulación del Problema ---
        obj_terms = []
        for i, coeff in enumerate(objective_coeffs):
            if coeff != 0:
                sign = '+' if coeff >= 0 and obj_terms else '' 
                obj_terms.append(f"{sign}{coeff}x_{{{i+1}}}")
        
        func_obj_latex = "".join(obj_terms).replace('+-', '-') 
        if func_obj_latex.startswith('+'):
            func_obj_latex = func_obj_latex[1:]
        
        paso1_detalle = f"""
        <h4>Paso 1: Formulación del Problema</h4>
        <p><strong>Función Objetivo:</strong> {'Maximizar' if is_maximize else 'Minimizar'} $Z = {func_obj_latex}$</p>
        <p><strong>Restricciones:</strong></p>
        <ul>
        """
        
        for i in range(m):
            restriction_terms = []
            for j in range(n):
                if A[i][j] != 0:
                    sign = '+' if A[i][j] >= 0 and restriction_terms else ''
                    restriction_terms.append(f"{sign}{A[i][j]}x_{{{j+1}}}")
            
            lhs_latex = "".join(restriction_terms).replace('+-', '-')
            if lhs_latex.startswith('+'):
                lhs_latex = lhs_latex[1:]
            
            paso1_detalle += f"<li>${lhs_latex} \\leq {b[i]}$</li>"
        
        paso1_detalle += f"<li>$x_j \\geq 0$ para $j = 1, 2, \\ldots, {n}$</li></ul>"
        
        if is_maximize:
            paso1_detalle += f"<p>Convertimos a forma estándar para minimización: $\\text{{Min}} \\, Z' = -({func_obj_latex})$</p>"
        
        iteraciones.append({'iteracion': 'Formulación', 'detalle': paso1_detalle})
        
        # --- Generar tabla inicial ---
        tabla_inicial_html = generar_tabla_html(tableau, basic_vars, var_names, 0)
        iteraciones.append({'iteracion': 'Tabla Inicial', 'detalle': tabla_inicial_html})
        
        iteration = 0
        max_iterations = 20  # Prevenir bucles infinitos
        
        while iteration < max_iterations:
            # Verificar optimalidad
            # Para minimización: si todos los coeficientes en la fila Z son >= 0
            if all(tableau[-1, j] >= -1e-10 for j in range(n + m)):
                break
            
            # Seleccionar columna pivote (variable entrante)
            # Elegir la columna con el coeficiente más negativo
            pivot_col = np.argmin(tableau[-1, :-1])
            
            # Verificar si hay solución no acotada
            if all(tableau[i, pivot_col] <= 1e-10 for i in range(m)):
                error_msg = "El problema no tiene solución acotada (unbounded)."
                return {"resultado": None, "variables": {}, "pasos": iteraciones, "error": error_msg}
            
            # Seleccionar fila pivote (variable saliente)
            # Calcular ratios para elementos positivos
            ratios = []
            for i in range(m):
                if tableau[i, pivot_col] > 1e-10:
                    ratios.append((tableau[i, -1] / tableau[i, pivot_col], i))
                else:
                    ratios.append((float('inf'), i))
            
            # Elegir la fila con el menor ratio no negativo
            min_ratio, pivot_row = min(ratios)
            
            if min_ratio < 0:
                error_msg = "Se encontró un ratio negativo, problema infactible."
                return {"resultado": None, "variables": {}, "pasos": iteraciones, "error": error_msg}
            
            # Mostrar proceso de selección de pivote
            proceso_pivote = f"""
            <h4>Iteración {iteration + 1}: Selección del Elemento Pivote</h4>
            <p><strong>Variable Entrante:</strong> {var_names[pivot_col]} (columna {pivot_col + 1})</p>
            <p>Se selecciona la variable con el coeficiente más negativo en la fila Z: ${tableau[-1, pivot_col]:.4f}$</p>
            
            <p><strong>Cálculo de Ratios para Variable Saliente:</strong></p>
            <table class="table table-bordered">
            <thead>
                <tr><th>Fila</th><th>Variable Básica</th><th>b</th><th>Coef. Pivote</th><th>Ratio</th></tr>
            </thead>
            <tbody>
            """
            
            for i in range(m):
                ratio_val = tableau[i, -1] / tableau[i, pivot_col] if tableau[i, pivot_col] > 1e-10 else "∞"
                selected = "✓" if i == pivot_row else ""
                proceso_pivote += f"""
                <tr {'class="table-warning"' if i == pivot_row else ''}>
                    <td>{i + 1}</td>
                    <td>{var_names[basic_vars[i]]}</td>
                    <td>{tableau[i, -1]:.4f}</td>
                    <td>{tableau[i, pivot_col]:.4f}</td>
                    <td>{ratio_val if ratio_val == "∞" else f"{ratio_val:.4f}"} {selected}</td>
                </tr>
                """
            
            proceso_pivote += f"""
            </tbody>
            </table>
            <p><strong>Variable Saliente:</strong> {var_names[basic_vars[pivot_row]]} (fila {pivot_row + 1})</p>
            <p><strong>Elemento Pivote:</strong> ${tableau[pivot_row, pivot_col]:.4f}$ (fila {pivot_row + 1}, columna {pivot_col + 1})</p>
            """
            
            iteraciones.append({'iteracion': f'Iteración {iteration + 1} - Pivote', 'detalle': proceso_pivote})
            
            # Realizar operaciones de pivoteo
            pivot_element = tableau[pivot_row, pivot_col]
            
            # Mostrar operaciones matemáticas
            operaciones_html = f"""
            <h4>Iteración {iteration + 1}: Operaciones de Pivoteo</h4>
            <p><strong>Paso 1:</strong> Hacer el elemento pivote igual a 1</p>
            <p>Dividir la fila {pivot_row + 1} entre {pivot_element:.4f}:</p>
            <p>$F_{pivot_row + 1}' = \\frac{{1}}{{{pivot_element:.4f}}} \\times F_{pivot_row + 1}$</p>
            """
            
            # Normalizar fila pivote
            tableau[pivot_row] = tableau[pivot_row] / pivot_element
            
            operaciones_html += f"""
            <p><strong>Paso 2:</strong> Hacer ceros en el resto de la columna pivote</p>
            """
            
            # Eliminar otros elementos de la columna pivote
            for i in range(m + 1):
                if i != pivot_row and abs(tableau[i, pivot_col]) > 1e-10:
                    multiplier = tableau[i, pivot_col]
                    operaciones_html += f"<p>$F_{i + 1}' = F_{i + 1} - ({multiplier:.4f}) \\times F_{pivot_row + 1}'$</p>"
                    tableau[i] = tableau[i] - multiplier * tableau[pivot_row]
            
            iteraciones.append({'iteracion': f'Iteración {iteration + 1} - Operaciones', 'detalle': operaciones_html})
            
            # Actualizar variable básica
            basic_vars[pivot_row] = pivot_col
            
            # Mostrar nueva tabla
            nueva_tabla_html = generar_tabla_html(tableau, basic_vars, var_names, iteration + 1)
            iteraciones.append({'iteracion': f'Tabla Iteración {iteration + 1}', 'detalle': nueva_tabla_html})
            
            iteration += 1
        
        # Verificar si se alcanzó el máximo de iteraciones
        if iteration >= max_iterations:
            error_msg = f"Se alcanzó el límite máximo de iteraciones ({max_iterations})."
            return {"resultado": None, "variables": {}, "pasos": iteraciones, "error": error_msg}
        
        # Extraer solución final
        variables_valores = {f'x{i+1}': 0.0 for i in range(num_vars)}
        
        for i, var_index in enumerate(basic_vars):
            if var_index < num_vars:  # Solo variables originales
                variables_valores[f'x{var_index + 1}'] = tableau[i, -1]
        
        # Valor de la función objetivo - CORRECCIÓN AQUÍ
        # El valor en tableau[-1, -1] ya está en la forma correcta después del simplex
        # Para maximización, necesitamos cambiar el signo del resultado final
        z_value = tableau[-1, -1]
        if is_maximize:
            z_value = -z_value  # Cambiar signo solo para maximización
        
        # Asegurar que el resultado sea positivo cuando corresponda
        z_value = abs(z_value) if z_value < 0 and is_maximize else z_value
        
        # Mostrar solución final
        solucion_final = f"""
        <h4>Solución Óptima</h4>
        <p><strong>Valor óptimo de la función objetivo:</strong> $Z = {z_value:.4f}$</p>
        <p><strong>Valores de las variables:</strong></p>
        <ul>
        """
        
        for var, val in variables_valores.items():
            solucion_final += f"<li>${var} = {val:.4f}$</li>"
        
        solucion_final += "</ul>"
        
        # Mostrar variables básicas finales
        solucion_final += "<p><strong>Variables básicas finales:</strong></p><ul>"
        for i, var_index in enumerate(basic_vars):
            if var_index < len(var_names):
                solucion_final += f"<li>{var_names[var_index]} = {tableau[i, -1]:.4f}</li>"
        solucion_final += "</ul>"
        
        # Agregar explicación del resultado
        if is_maximize:
            solucion_final += f"""
            <p><strong>Nota:</strong> Como el problema original era de maximización, 
            el valor final se obtiene cambiando el signo del resultado del tableau: 
            $Z = -({-z_value:.4f}) = {z_value:.4f}$</p>
            """
        
        iteraciones.append({'iteracion': 'Solución Final', 'detalle': solucion_final})
        
        return {
            "resultado": f"Z = {z_value:.4f}",
            "variables": variables_valores,
            "pasos": iteraciones,
            "error": None
        }
        
    except Exception as e:
        error_msg = f"Error en el cálculo del Simplex: {str(e)}"
        return {"resultado": None, "variables": {}, "pasos": iteraciones, "error": error_msg}


def generar_tabla_html(tableau, basic_vars, var_names, iteration):
    """
    Genera la representación HTML de la tabla Simplex.
    """
    m, n_plus_one = tableau.shape
    n_vars = n_plus_one - 1  # Excluir columna RHS
    
    html = f"""
    <h4>Tabla Simplex {'Inicial' if iteration == 0 else f'- Iteración {iteration}'}</h4>
    <div class="table-responsive">
    <table class="table table-bordered table-sm">
    <thead class="table-dark">
    <tr>
        <th>Base</th>
    """
    
    # Encabezados de columnas
    for i in range(n_vars):
        if i < len(var_names):
            html += f"<th>{var_names[i]}</th>"
        else:
            html += f"<th>x{i+1}</th>"
    
    html += "<th>RHS</th></tr></thead><tbody>"
    
    # Filas de restricciones
    for i in range(m - 1):  # Excluir fila Z
        basic_var = var_names[basic_vars[i]] if basic_vars[i] < len(var_names) else f"x{basic_vars[i] + 1}"
        html += f"<tr><td><strong>{basic_var}</strong></td>"
        
        for j in range(n_vars):
            value = tableau[i, j]
            html += f"<td>{value:.4f}</td>"
        
        html += f"<td>{tableau[i, -1]:.4f}</td></tr>"
    
    # Fila Z
    html += "<tr class='table-info'><td><strong>Z</strong></td>"
    for j in range(n_vars):
        value = tableau[-1, j]
        html += f"<td>{value:.4f}</td>"
    
    html += f"<td>{tableau[-1, -1]:.4f}</td></tr>"
    
    html += "</tbody></table></div>"
    
    # Agregar interpretación de la tabla
    if iteration == 0:
        html += """
        <p><strong>Interpretación de la Tabla Inicial:</strong></p>
        <ul>
        <li>Las variables básicas iniciales son las variables de holgura</li>
        <li>La fila Z contiene los coeficientes de la función objetivo</li>
        <li>RHS (Right Hand Side) contiene los valores del lado derecho</li>
        </ul>
        """
    else:
        html += f"""
        <p><strong>Estado después de la Iteración {iteration}:</strong></p>
        <p>Variables básicas actuales: {', '.join([var_names[basic_vars[i]] for i in range(len(basic_vars))])}</p>
        """
    
    return html


# También agrega esta función auxiliar para mejorar el parsing
def parse_simplex_input_improved(func_obj_str, restrictions_str, is_maximize):
    """
    Versión mejorada del parser que maneja mejor los casos edge.
    """
    try:
        # Limpiar espacios
        func_obj_str = func_obj_str.replace(' ', '')
        restrictions_str = restrictions_str.replace(' ', '')
        
        # Encontrar el número máximo de variables
        all_text = func_obj_str + restrictions_str
        var_matches = re.findall(r'x(\d+)', all_text)
        if not var_matches:
            raise ValueError("No se encontraron variables en formato xN")
        
        max_var = max(int(match) for match in var_matches)
        
        # Parsear función objetivo
        c = [0.0] * max_var
        
        # Agregar signo + al inicio si no hay signo
        if not func_obj_str.startswith(('+', '-')):
            func_obj_str = '+' + func_obj_str
        
        # Encontrar términos
        terms = re.findall(r'([+-])(\d*\.?\d*)x(\d+)', func_obj_str)
        
        for sign, coeff, var_num in terms:
            coeff_val = 1.0 if coeff == '' else float(coeff)
            if sign == '-':
                coeff_val = -coeff_val
            
            var_idx = int(var_num) - 1
            if 0 <= var_idx < max_var:
                c[var_idx] = coeff_val
        
        # Parsear restricciones
        A = []
        b = []
        
        restrictions = [r.strip() for r in restrictions_str.split('\n') if r.strip()]
        
        for restriction in restrictions:
            # Saltar restricciones de no negatividad
            if re.match(r'x\d+>=0', restriction):
                continue
            
            if '<=' in restriction:
                left, right = restriction.split('<=')
                b_val = float(right)
                
                # Parsear lado izquierdo
                row = [0.0] * max_var
                if not left.startswith(('+', '-')):
                    left = '+' + left
                
                terms = re.findall(r'([+-])(\d*\.?\d*)x(\d+)', left)
                for sign, coeff, var_num in terms:
                    coeff_val = 1.0 if coeff == '' else float(coeff)
                    if sign == '-':
                        coeff_val = -coeff_val
                    
                    var_idx = int(var_num) - 1
                    if 0 <= var_idx < max_var:
                        row[var_idx] = coeff_val
                
                A.append(row)
                b.append(b_val)
            
            elif '>=' in restriction:
                left, right = restriction.split('>=')
                b_val = float(right)
                
                # Convertir >= a <= multiplicando por -1
                row = [0.0] * max_var
                if not left.startswith(('+', '-')):
                    left = '+' + left
                
                terms = re.findall(r'([+-])(\d*\.?\d*)x(\d+)', left)
                for sign, coeff, var_num in terms:
                    coeff_val = 1.0 if coeff == '' else float(coeff)
                    if sign == '-':
                        coeff_val = -coeff_val
                    
                    var_idx = int(var_num) - 1
                    if 0 <= var_idx < max_var:
                        row[var_idx] = -coeff_val  # Cambiar signo
                
                A.append(row)
                b.append(-b_val)  # Cambiar signo
        
        return c, A, b, None, max_var
        
    except Exception as e:
        raise ValueError(f"Error al parsear la entrada: {e}")

def metodo_simplex_view(request):
    
    resultado = None
    pasos_html = None
    error_msg = None
    input_funcion_objetivo = ''
    input_tipo_optimizacion = 'maximizar'
    input_restricciones = ''

    if request.method == 'POST':
        input_funcion_objetivo = request.POST.get('funcion_objetivo')
        input_tipo_optimizacion = request.POST.get('tipo_optimizacion')
        input_restricciones = request.POST.get('restricciones')

        try:
            c, A, b, bounds, num_vars = parse_simplex_input(
                input_funcion_objetivo,
                input_restricciones,
                input_tipo_optimizacion == 'maximizar'
            )
            
            # Aquí llamamos a la función simplex_solver que ahora simula los pasos
            simplex_result = simplex_solver(c, A, b, input_tipo_optimizacion == 'maximizar', num_vars)

            if simplex_result['error']:
                error_msg = simplex_result['error']
            else:
                resultado = simplex_result['resultado']
                variables_valores = simplex_result['variables']
                
                if request.user.is_authenticated:
                    pasos_list = []
                    for paso in simplex_result['pasos']:
                        pasos_list.append(paso['detalle'])
                    pasos_html = "".join(pasos_list)

                    parametros_simplex = {
                        'funcion_objetivo': input_funcion_objetivo,
                        'tipo_optimizacion': input_tipo_optimizacion,
                        'restricciones': input_restricciones,
                        'valores_finales_variables': variables_valores
                    }

                    Historial.objects.create(
                        usuario=request.user,
                        ecuacion=f"Simplex: {input_funcion_objetivo}",
                        metodo='Simplex',
                        resultado=resultado,
                        pasos=pasos_html,
                        iteraciones=len(simplex_result['pasos']),
                        metodo_tipo='Simplex',
                        parametros_simplex=json.dumps(parametros_simplex),
                        x0=None, x1=None, x2=None, orden=None 
                    )
                else:
                    pasos_html = None

        except ValueError as ve:
            error_msg = f"⚠️ Error de formato en la entrada: {str(ve)}. Por favor, verifica la sintaxis de la función objetivo y las restricciones (ej. '3x1 + 2x2' y '2x1 + x2 <= 10')."
        except Exception as e:
            error_msg = f"⚠️ Ocurrió un error inesperado al calcular el Simplex: {str(e)}"
    
    return render(request, 'pagina/metodo_simplex.html', {
        'resultado': resultado,
        'pasos': pasos_html,
        'error_msg': error_msg,
        'input_funcion_objetivo': input_funcion_objetivo,
        'input_tipo_optimizacion': input_tipo_optimizacion,
        'input_restricciones': input_restricciones,
    })


def load_example_simplex(request):
    """Carga un ejemplo predeterminado para el Método Simplex."""
    example_data = {
        'funcion_objetivo': '3x1 + 5x2',
        'tipo_optimizacion': 'maximizar',
        'restricciones': 'x1 <= 4\n2x2 <= 12\n3x1 + 2x2 <= 18\nx1 >= 0\nx2 >= 0',
    }
    return JsonResponse(example_data)


@login_required
def historial(request):
    registros = Historial.objects.filter(usuario=request.user).order_by('-fecha')
    return render(request, 'pagina/historial.html', {'registros': registros})

@login_required
def eliminar_historial(request, historial_id):
    historial = get_object_or_404(Historial, id=historial_id, usuario=request.user)
    historial.delete()
    messages.success(request, 'Registro eliminado correctamente.')
    return redirect('historial')