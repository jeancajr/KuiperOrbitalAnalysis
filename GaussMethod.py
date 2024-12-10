import pandas as pd
import numpy as np
from numpy.linalg import norm
from scipy.optimize import fsolve
from astropy.time import Time
from astropy.coordinates import get_sun, EarthLocation
from astropy.coordinates import AltAz
from math import sqrt
import math

# Leer el archivo Excel
excel_file = 'Orbital_Data.xlsx'  # Asegúrate de que el archivo esté en el mismo directorio

df = pd.read_excel(excel_file)
# Mostrar el DataFrame
print("Datos leídos del archivo Excel:")
print(df)
# Extraer las columnas de interés
dates = df['Date']
ra_degrees = df['R.A. (degrees)']
ra_radians = df['R.A. (radians)']
dec_degrees = df['Dec (degrees)']
dec_radians = df['Dec (radians)']

def calculate_julian_day_from_string(date_str):
    # Suponiendo que las fechas están en formato 'YYYY MMM DD'
    from datetime import datetime
    date = datetime.strptime(date_str, '%Y %b %d')
    jd = calculate_julian_day(date.year, date.month, date.day)
    return jd

def calculate_julian_day(year, month, day, hour=0, minute=0, second=0):
    if month <= 2:
        year -= 1
        month += 12
    A = np.floor(year / 100)
    B = 2 - A + np.floor(A / 4)
    JD = (np.floor(365.25 * (year + 4716))
          + np.floor(30.6001 * (month + 1))
          + day + B - 1524.5)
    JD += (hour + minute / 60 + second / 3600) / 24
    return JD

# Calcular los días julianos para cada fecha
jd_dates = []
for date_str in dates:
    jd = calculate_julian_day_from_string(date_str)
    jd_dates.append(jd)

# Agregar los días julianos al DataFrame
df['Julian Day'] = jd_dates

""" una vez se calculan los días julianos se puede determinar los valores de los cosenos directores """
l = np.cos(np.radians(ra_degrees)) * np.cos(np.radians(dec_degrees))
m = np.sin(np.radians(ra_degrees)) * np.cos(np.radians(dec_degrees))
n = np.sin(np.radians(dec_degrees))
df['l'] = l
df['m'] = m
df['n'] = n

""" Calculo de thao """
Thao1=jd_dates[2]-jd_dates[1]
Thao2=jd_dates[2]-jd_dates[0]
Thao3=jd_dates[1]-jd_dates[0]

def calculate_geocentric_coordinates_astropy(jd):
    """
    Calcula las coordenadas geocéntricas del Sol usando astropy.
    jd: Día juliano.
    """
    # Convertir día juliano a formato Time de astropy
    time = Time(jd, format='jd')
    
    # Obtener la posición del Sol en coordenadas geocéntricas
    sun = get_sun(time)
    
    # Retornar las coordenadas cartesianas x, y, z en AU
    return sun.cartesian.x.value, sun.cartesian.y.value, sun.cartesian.z.value

# Ejemplo de uso
jd1 = jd_dates[0] # Día juliano 
jd2 = jd_dates[1]  # Día juliano 
jd3 = jd_dates[2]  # Día juliano 
# Calcular coordenadas geocéntricas usando astropy
x01, y01, z01 = calculate_geocentric_coordinates_astropy(jd1)
x02, y02, z02 = calculate_geocentric_coordinates_astropy(jd2)
x03, y03, z03 = calculate_geocentric_coordinates_astropy(jd3)

b1=Thao1/Thao2
b3=Thao3/Thao2

# """"Para esta primera aproximación se tiene que b1=a1 y b3=a3"""
# print(f"x02 = {x02}")
# print(f"y02 = {y02}")
# print(f"z02 = {z02}")

R1=1
R2=1
R3=1
""" Calculo del delta 1, 2 y 3"""
def calculate_equation_terms1():
    """
    Calcula los términos para resolver el sistema de ecuaciones basándose en la imagen proporcionada.
    """
    # Coeficientes de la ecuación
    l1, l2, l3 = l[0],l[1],l[2] 
    a1=(R2/R1)*b1
    a3=(R2/R3)*b3
    x_o1, x_o2, x_o3 = x01, x02, x03
    # Términos individuales
    term_I = l1 * a1  # +
    term_II = -l2  # 
    term_III = l3 * a3  
    term_IV = a1 * x_o1  
    term_V = -x_o2  
    term_VI = a3 * x_o3  

    # Resultados totales
    rhs = term_IV + term_V + term_VI

    return term_I, term_II, term_III, rhs
terms1 = calculate_equation_terms1()

def calculate_equation_terms2():
    """
    Calcula los términos para resolver el sistema de ecuaciones basándose en la imagen proporcionada.
    """
    # Coeficientes de la ecuación
    m1, m2, m3 = m[0],m[1],m[2] 
    a1=(R2/R1)*b1
    a3=(R2/R3)*b3
    y_o1, y_o2, y_o3 = y01, y02, y03

    # Términos individuales
    term_I = m1 * a1  # 
    term_II = -m2  
    term_III = m3 * a3  
    term_IV = a1 * y_o1  
    term_V = -y_o2 
    term_VI = a3 * y_o3  
    # Resultados totales
    rhs = term_IV + term_V + term_VI
    return term_I, term_II, term_III, rhs
terms2 = calculate_equation_terms2()

def calculate_equation_terms3():
    """
    Calcula los términos para resolver el sistema de ecuaciones basándose en la imagen proporcionada.
    """
    # Coeficientes de la ecuación
    n1, n2, n3 = n[0],n[1],n[2] 
    a1=(R2/R1)*b1
    a3=(R2/R3)*b3
    z_o1, z_o2, z_o3 = z01, z02, z03

    # Términos individuales
    term_I = n1 * a1  # 
    term_II = -n2  
    term_III = n3 * a3  
    term_IV = a1 * z_o1  
    term_V = -z_o2 
    term_VI = a3 * z_o3  

    
    # Resultados totales
    rhs = term_IV + term_V + term_VI

    return term_I, term_II, term_III, rhs
terms3 = calculate_equation_terms3()

A=np.array([[terms1[0],terms1[1],terms1[2]],[terms2[0],terms2[1],terms2[2]],[terms3[0],terms3[1],terms3[2]]])
B=np.array([terms1[3],terms2[3],terms3[3]])
solution = np.linalg.solve(A, B)
# Mostrar la solución
#print("Solución del sistema:")
#print(f"x = {solution[0]:.2f}, y = {solution[1]:.2f}, z = {solution[2]:.2f}")

"""Distancias Heliocentricas """
def calculate_equation_Helio3():
    """
    Calcula los términos para resolver el sistema de ecuaciones basándose en la imagen proporcionada.
    """
    # Coeficientes de la ecuación
    n1, n2, n3 = n[0],n[1],n[2] 
    l1, l2, l3 = l[0],l[1],l[2]
    m1, m2, m3 = m[0],m[1],m[2]
    x_o1, x_o2, x_o3 = x01, x02, x03
    y_o1, y_o2, y_o3 = y01, y02, y03
    z_o1, z_o2, z_o3 = z01, z02, z03

    # Términos individuales
    term_I = (l1 *   solution[0])-x_o1 # 
    term_II = (m1 *   solution[0])-y_o1 
    term_III = (n1 *   solution[0])-z_o1  
    r1=sqrt((term_I* term_I)+(term_II* term_II)+(term_III* term_III))
    
    term_I_2 = (l2 *   solution[1])-x_o2 # 
    term_II_2 = (m2 *   solution[1])-y_o2 
    term_III_2 = (n2 *   solution[1])-z_o2  
    r2=sqrt((term_I_2* term_I_2)+(term_II_2* term_II_2)+(term_III_2* term_III_2))
    
    term_I_3 = (l3 *   solution[2])-x_o3 # 
    term_II_3 = (m3 *   solution[2])-y_o3 
    term_III_3 = (n3 *   solution[2])-z_o3  
    r3=sqrt((term_I_3* term_I_3)+(term_II_3* term_II_3)+(term_III_3* term_III_3))
    
    return r1, r2, r3
heliocentric_distances = calculate_equation_Helio3()

print("\nDistancias Heliocéntricas Primera iteración:")
print(f"r1 = {heliocentric_distances[0]:.6f} AU, r2 = {heliocentric_distances[1]:.6f} AU, r3 = {heliocentric_distances[2]:.6f} AU")

def calculate_r1_vector(l, m, n, solution, x01, y01, z01):

    r1_vector = [
        l[0] * solution[0] - x01,
        m[0] * solution[0] - y01,
        n[0] * solution[0] - z01
    ]
    return r1_vector
def calculate_r2_vector(l, m, n, solution, x02, y02, z02):

    r2_vector = [
        l[1] * solution[1] - x02,
        m[1] * solution[1] - y02,
        n[1] * solution[1] - z02
    ]
    return r2_vector
def calculate_r3_vector(l, m, n, solution, x03, y03, z03):

    r3_vector = [
        l[2] * solution[2] - x03,
        m[2] * solution[2] - y03,
        n[2] * solution[2] - z03
    ]
    return r3_vector
r1_vector=calculate_r1_vector(l, m, n, solution, x01, y01, z01)
r2_vector=calculate_r2_vector(l, m, n, solution, x02, y02, z02)
r3_vector=calculate_r3_vector(l, m, n, solution, x03, y03, z03)

def calculate_vector_angles(r1_vector, r2_vector, r3_vector):
    """
    Calcula productos punto, magnitudes, cosenos, ángulos dobles y simples (en radianes y grados) para los vectores r1, r2 y r3.
    
    :param r1_vector: Vector r1
    :param r2_vector: Vector r2
    :param r3_vector: Vector r3
    :return: Diccionario con productos punto, magnitudes, cosenos y ángulos en radianes y grados
    """
    # Productos punto
    dot_r1_r2 = np.dot(r1_vector, r2_vector)
    dot_r1_r3 = np.dot(r1_vector, r3_vector)
    dot_r2_r3 = np.dot(r2_vector, r3_vector)

    # Magnitudes
    mag_r1 = norm(r1_vector)
    mag_r2 = norm(r2_vector)
    mag_r3 = norm(r3_vector)

    # Calcular cosenos
    cos_2f1 = dot_r1_r3 / (mag_r1 * mag_r3)
    cos_2f2 = dot_r2_r3 / (mag_r2 * mag_r3)
    cos_2f3 = dot_r1_r2 / (mag_r1 * mag_r2)

    # Calcular ángulos dobles en radianes
    two_f1_radians = np.arccos(cos_2f1)
    two_f2_radians = np.arccos(cos_2f2)
    two_f3_radians = np.arccos(cos_2f3)

    # Calcular ángulos simples en radianes
    f1_radians = two_f1_radians / 2
    f2_radians = two_f2_radians / 2
    f3_radians = two_f3_radians / 2

    # Convertir ángulos a grados
    f1_degrees = np.degrees(f1_radians)
    f2_degrees = np.degrees(f2_radians)
    f3_degrees = np.degrees(f3_radians)

    # Retornar resultados en un diccionario
    return {
        "dot_products": {
            "dot_r1_r2": dot_r1_r2,
            "dot_r1_r3": dot_r1_r3,
            "dot_r2_r3": dot_r2_r3
        },
        "magnitudes": {
            "mag_r1": mag_r1,
            "mag_r2": mag_r2,
            "mag_r3": mag_r3
        },
        "cosines": {
            "cos_2f1": cos_2f1,
            "cos_2f2": cos_2f2,
            "cos_2f3": cos_2f3
        },
        "angles": {
            "two_f1_radians": two_f1_radians,
            "two_f2_radians": two_f2_radians,
            "two_f3_radians": two_f3_radians,
            "f1_radians": f1_radians,
            "f2_radians": f2_radians,
            "f3_radians": f3_radians,
            "f1_degrees": f1_degrees,
            "f2_degrees": f2_degrees,
            "f3_degrees": f3_degrees
        }
    }

results = calculate_vector_angles(r1_vector, r2_vector, r3_vector)
f1_radians=results["angles"]["f1_radians"]
f2_radians=results["angles"]["f2_radians"]
f3_radians=results["angles"]["f3_radians"]
f1_degrees=results["angles"]["f1_degrees"]
f2_degrees=results["angles"]["f2_degrees"]
f3_degrees=results["angles"]["f3_degrees"]

cos_2f1=results["cosines"]["cos_2f1"]
cos_2f2=results["cosines"]["cos_2f2"]
cos_2f3=results["cosines"]["cos_2f3"]

""" Re Calculo de thao """
k = 0.01720209895 # rad / dsm. 

Thao11 = (jd_dates[2] - jd_dates[1]) * k
Thao22 = (jd_dates[2] - jd_dates[0]) * k
Thao33 = (jd_dates[1] - jd_dates[0]) * k

# Valores conocidos
thao = [Thao11, Thao22, Thao33]  # Ejemplo: ajustar según valores reales
r = heliocentric_distances     # Valores de r1, r2, r3
cos_2f = [cos_2f1, cos_2f2, cos_2f3]  # Cos(2f1), Cos(2f2), Cos(2f3)

# Función para calcular M_i y N_i
def calculate_M_N(thao_i, rj, rk, cos_fi):
    sqrt_rj_rk = np.sqrt(rj * rk)
    common_factor = (sqrt_rj_rk * cos_fi)**(3 / 2)
    common_factor1 = (sqrt_rj_rk * cos_fi)
    Mi = thao_i / (2 * common_factor)
    Ni = (rj + rk) / (2 * common_factor1)
    return Mi, Ni

# Sistema para resolver R_i y g_i
def equations(vars, Mi, Ni):
    Ri, gi = vars
    eq1 = Ri**2 - (Mi**2 / (Ni - np.cos(gi)))  # Primera ecuación
    eq2 = Ri**3 - Ri**2 - (Mi**2 * (gi - np.sin(gi) * np.cos(gi)) / np.sin(gi)**3)  # Segunda ecuación
    return [eq1, eq2]

# Cálculo para cada i usando permutación cíclica
results = []
for i in range(3):
    # Índices cíclicos
    j, k = (i + 1) % 3, (i + 2) % 3
    cos_fi = np.sqrt((1 + cos_2f[i]) / 2)  # Relación trigonométrica
    fi = np.arccos(cos_fi)  # Calcular el valor de f_i a partir del coseno
    
    # Calcular M_i y N_i
    Mi, Ni = calculate_M_N(thao[i], r[j], r[k], cos_fi)
    
    # Resolver sistema para R_i y g_i
    initial_guess = [1.0, fi]  # g_i comienza con el valor de f_i
    Ri, gi = fsolve(equations, initial_guess, args=(Mi, Ni))
    gi_degrees = np.degrees(gi)  # Convertir g_i a grados
    
    # Guardar resultados
    results.append({
        "i": i + 1,
        "cos_fi": cos_fi,
        "Mi_squared": Mi**2,
        "Ni": Ni,
        "cos_gi": np.cos(gi),
        "Ri": Ri,
        "gi_rad": gi,
        "gi_deg": gi_degrees
    })

# Acceder a R3 (correspondiente al tercer elemento de la lista)
R3 = results[2]["Ri"]  # El índice 2 corresponde a i=3 (porque los índices son cero-indexados)
R1 = results[0]["Ri"]
R2 = results[1]["Ri"]

########### Calculos para la segunda iteración##############
#print("#################################################")
print("\nSolución del sistema:Segunda iteración ----------")

#funcion para el calculo de a y b
a1=(R2/R1)*b1
a3=(R2/R3)*b3
#print(a1,a3)
terms1 = calculate_equation_terms1()
terms2 = calculate_equation_terms2()
terms3 = calculate_equation_terms3()

A=np.array([[terms1[0],terms1[1],terms1[2]],[terms2[0],terms2[1],terms2[2]],[terms3[0],terms3[1],terms3[2]]])
B=np.array([terms1[3],terms2[3],terms3[3]])
solution = np.linalg.solve(A, B)

#print(f"x = {solution[0]:.2f}, y = {solution[1]:.2f}, z = {solution[2]:.2f}")
heliocentric_distances = calculate_equation_Helio3()

print("Distancias heliocéntricas: Segunda iteración")
print(f"r1 = {heliocentric_distances[0]:.6f} AU, r2 = {heliocentric_distances[1]:.6f} AU, r3 = {heliocentric_distances[2]:.6f} AU")
r1_vector=calculate_r1_vector(l, m, n, solution, x01, y01, z01)
r2_vector=calculate_r2_vector(l, m, n, solution, x02, y02, z02)
r3_vector=calculate_r3_vector(l, m, n, solution, x03, y03, z03)

results = calculate_vector_angles(r1_vector, r2_vector, r3_vector)
results = calculate_vector_angles(r1_vector, r2_vector, r3_vector)
f1_radians=results["angles"]["f1_radians"]
f2_radians=results["angles"]["f2_radians"]
f3_radians=results["angles"]["f3_radians"]
f1_degrees=results["angles"]["f1_degrees"]
f2_degrees=results["angles"]["f2_degrees"]
f3_degrees=results["angles"]["f3_degrees"]

cos_2f1=results["cosines"]["cos_2f1"]
cos_2f2=results["cosines"]["cos_2f2"]
cos_2f3=results["cosines"]["cos_2f3"]


""" Re Calculo de thao """
k = 0.01720209895 # rad / dsm. 

Thao11 = (jd_dates[2] - jd_dates[1]) * k
Thao22 = (jd_dates[2] - jd_dates[0]) * k
Thao33 = (jd_dates[1] - jd_dates[0]) * k

# Valores conocidos
thao = [Thao11, Thao22, Thao33]  # Ejemplo: ajustar según valores reales
r = heliocentric_distances     # Valores de r1, r2, r3
cos_2f = [cos_2f1, cos_2f2, cos_2f3]  # Cos(2f1), Cos(2f2), Cos(2f3)

# Cálculo para cada i usando permutación cíclica
results = []
for i in range(3):
    # Índices cíclicos
    j, k = (i + 1) % 3, (i + 2) % 3
    cos_fi = np.sqrt((1 + cos_2f[i]) / 2)  # Relación trigonométrica
    fi = np.arccos(cos_fi)  # Calcular el valor de f_i a partir del coseno
    
    # Calcular M_i y N_i
    Mi, Ni = calculate_M_N(thao[i], r[j], r[k], cos_fi)
    
    # Resolver sistema para R_i y g_i
    initial_guess = [1.0, fi]  # g_i comienza con el valor de f_i
    Ri, gi = fsolve(equations, initial_guess, args=(Mi, Ni))
    gi_degrees = np.degrees(gi)  # Convertir g_i a grados
    
    # Guardar resultados
    results.append({
        "i": i + 1,
        "cos_fi": cos_fi,
        "Mi_squared": Mi**2,
        "Ni": Ni,
        "cos_gi": np.cos(gi),
        "Ri": Ri,
        "gi_rad": gi,
        "gi_deg": gi_degrees
    })


# Acceder a R3 (correspondiente al tercer elemento de la lista)
R3 = results[2]["Ri"]  # El índice 2 corresponde a i=3 (porque los índices son cero-indexados)
R1 = results[0]["Ri"]
R2 = results[1]["Ri"]

########### Calculos para la Tercera iteración##############
#print("#################################################")
print("\nSolución del sistema:Tercera iteración ----------")
#print("#################################################")
#funcion para el calculo de a y b
#print(b1,b3,results[1]["Ri"])
a1=(R2/R1)*b1
a3=(R2/R3)*b3
#print(a1,a3)
terms1 = calculate_equation_terms1()
terms2 = calculate_equation_terms2()
terms3 = calculate_equation_terms3()

A=np.array([[terms1[0],terms1[1],terms1[2]],[terms2[0],terms2[1],terms2[2]],[terms3[0],terms3[1],terms3[2]]])
B=np.array([terms1[3],terms2[3],terms3[3]])
solution = np.linalg.solve(A, B)
# Mostrar la solución

#print(f"x = {solution[0]:.2f}, y = {solution[1]:.2f}, z = {solution[2]:.2f}")
heliocentric_distances = calculate_equation_Helio3()
print("Distancias heliocéntricas: Tercera iteración")
print(f"r1 = {heliocentric_distances[0]:.6f} AU, r2 = {heliocentric_distances[1]:.6f} AU, r3 = {heliocentric_distances[2]:.6f} AU")
r1_vector=calculate_r1_vector(l, m, n, solution, x01, y01, z01)
r2_vector=calculate_r2_vector(l, m, n, solution, x02, y02, z02)
r3_vector=calculate_r3_vector(l, m, n, solution, x03, y03, z03)

results = calculate_vector_angles(r1_vector, r2_vector, r3_vector)
results = calculate_vector_angles(r1_vector, r2_vector, r3_vector)
f1_radians=results["angles"]["f1_radians"]
f2_radians=results["angles"]["f2_radians"]
f3_radians=results["angles"]["f3_radians"]
f1_degrees=results["angles"]["f1_degrees"]
f2_degrees=results["angles"]["f2_degrees"]
f3_degrees=results["angles"]["f3_degrees"]

cos_2f1=results["cosines"]["cos_2f1"]
cos_2f2=results["cosines"]["cos_2f2"]
cos_2f3=results["cosines"]["cos_2f3"]


""" Re Calculo de thao """
k = 0.01720209895 # rad / dsm. 

Thao11 = (jd_dates[2] - jd_dates[1]) * k
Thao22 = (jd_dates[2] - jd_dates[0]) * k
Thao33 = (jd_dates[1] - jd_dates[0]) * k

# Valores conocidos
thao = [Thao11, Thao22, Thao33]  # Ejemplo: ajustar según valores reales
r = heliocentric_distances     # Valores de r1, r2, r3
cos_2f = [cos_2f1, cos_2f2, cos_2f3]  # Cos(2f1), Cos(2f2), Cos(2f3)

# Cálculo para cada i usando permutación cíclica
results = []
for i in range(3):
    # Índices cíclicos
    j, k = (i + 1) % 3, (i + 2) % 3
    cos_fi = np.sqrt((1 + cos_2f[i]) / 2)  # Relación trigonométrica
    fi = np.arccos(cos_fi)  # Calcular el valor de f_i a partir del coseno
    
    # Calcular M_i y N_i
    Mi, Ni = calculate_M_N(thao[i], r[j], r[k], cos_fi)
    
    # Resolver sistema para R_i y g_i
    initial_guess = [1.0, fi]  # g_i comienza con el valor de f_i
    Ri, gi = fsolve(equations, initial_guess, args=(Mi, Ni))
    gi_degrees = np.degrees(gi)  # Convertir g_i a grados
    
    # Guardar resultados
    results.append({
        "i": i + 1,
        "cos_fi": cos_fi,
        "Mi_squared": Mi**2,
        "Ni": Ni,
        "cos_gi": np.cos(gi),
        "Ri": Ri,
        "gi_rad": gi,
        "gi_deg": gi_degrees
    })

# Acceder a R3 (correspondiente al tercer elemento de la lista)
R3 = results[2]["Ri"]  # El índice 2 corresponde a i=3 (porque los índices son cero-indexados)
R1 = results[0]["Ri"]
R2 = results[1]["Ri"]

## Calculo de elementos orbitales
import math
r1 = heliocentric_distances[0]
r2 = heliocentric_distances[1]
r3 = heliocentric_distances[2]

# Asegúrate de que R3 y f3 sean números antes de usarlos
R3 = float(R3)
f3_radians = float(f3_radians)

# Formatear los valores con 6 cifras decimales
r1 = format(r1, ".7f")
r2 = format(r2, ".7f")
r3 = format(r3, ".7f")
Thao33 = format(Thao33, ".7f")
R3 = format(R3, ".7f")
f3_radians = format(f3_radians, ".7f")

# Convertir las cadenas formateadas de nuevo a números para cálculos
r1 = float(r1)
r2 = float(r2)
r3 = float(r3)
Thao33 = float(Thao33)
R3 = float(R3)
f3_radians = float(f3_radians)

# Calcular el lactus rectum
l = ((R3 * r1 * r2 * np.sin(2 * f3_radians)) / Thao33) ** 2

# Resolver el sistema de ecuaciones
numerador = (l / r1 - 1) * math.cos(2 * f2_radians) - (l / r3 - 1)
denominador = math.sin(2 * f2_radians)
e_sin_v1 = numerador / denominador
e_cos_v1 = l / r1 - 1

# Calcular e y v1
e = math.sqrt(e_sin_v1**2 + e_cos_v1**2)
v1 = math.atan2(e_sin_v1, e_cos_v1)

# Asegurarse de que los ángulos estén en el rango [0, 360) grados
def to_degrees_positive(angle_rad):
    angle_deg = math.degrees(angle_rad)
    if angle_deg < 0:
        angle_deg += 360
    return angle_deg

# Calcular las otras anomalías verdaderas
v2 = v1 + 2 * f3_radians
v3 = v1 + 2 * f2_radians

# Convertir los resultados a grados positivos
v1_grados = to_degrees_positive(v1)
v2_grados = to_degrees_positive(v2)
v3_grados = to_degrees_positive(v3)

print(v1_grados,v2_grados,v3_grados)

# Calcular el semi eje mayor (a)
a = l / (1 - e**2)

# Calcular el período en años siderales (P)
P_sideral_years = a**(3/2)

# Convertir el período a días solares medios (dsm)
P_dsm = P_sideral_years * 365.25

# Definir los días Julianos
df = pd.DataFrame({'Julian Day': [2452465.5, 2452470.5, 2452480.5]})

# Calcular la oblicuidad de la eclíptica (ε) para cada día Juliano
def calculate_obliquity(julian_day):
    T = (julian_day - 2451545.0) / 36525.0
    epsilon = 23.439292 - 0.013004167 * T - 0.0000001639 * T**2 + 0.0000005036 * T**3
    return epsilon

df['Obliquity'] = df['Julian Day'].apply(calculate_obliquity)

P_x = ((r1_vector[0]* r3 * math.sin(v3)) - (r3_vector[0] * r1 * math.sin(v1))) / (r1 * r3 * math.sin(2 * f2_radians))
#Q_x = (((r3_vector[0]*r1 * math.cos(v1)) - (r1_vector[0]*r3 * math.cos(v3)))) / (r1 * r3 * math.sin(2 * f2_radians))
Q_x = ((r3_vector[0]*r1 * math.cos(v1))-((r1_vector[0]*r3 * math.cos(v3))))/(r1 * r3 * math.sin(2 * f2_radians))
P_y = (r1_vector[1]* r3 * math.sin(v3) - r3_vector[1] * r1 * math.sin(v1)) / (r1 * r3 * math.sin(2 * f2_radians))
Q_y = (r3_vector[1]*r1 * math.cos(v1) - r1_vector[1]*r3 * math.cos(v3)) / (r1 * r3 * math.sin(2 * f2_radians))
P_z = (r1_vector[2]* r3 * math.sin(v3) - r3_vector[2] * r1 * math.sin(v1)) / (r1 * r3 * math.sin(2 * f2_radians))
Q_z = (r3_vector[2]*r1 * math.cos(v1) - r1_vector[2]*r3 * math.cos(v3)) / (r1 * r3 * math.sin(2 * f2_radians))

# Calcular el argumento del perihelio (ω)
epsilon_rad = math.radians(df['Obliquity'].iloc[0])  # Usar la oblicuidad del primer día Juliano como ejemplo
epsilon_rad = np.degrees(epsilon_rad)
epsilon_rad=2*np.pi*epsilon_rad/360
#epsilon_rad=0.40142572796
sin_w_sin_i = P_z * math.cos(epsilon_rad) - P_y * math.sin(epsilon_rad)
cos_w_sin_i = Q_z * math.cos(epsilon_rad) - Q_y * math.sin(epsilon_rad)

w_rad = math.atan2(sin_w_sin_i, cos_w_sin_i)
w_grados = to_degrees_positive(w_rad)

# Otra forma de calcular i

# Función para convertir a grados positivos
def to_degrees_positive(angle_rad):
    angle_deg = math.degrees(angle_rad)
    return angle_deg % 360  # Asegura que el ángulo esté entre 0 y 360 grados

# Cálculo de Ω (Omega)
sec_epsilon = 1 / math.cos(math.radians(epsilon_rad))  # sec(ϵ)

# sen(Ω) y cos(Ω)
sin_Omega = (P_y * math.cos(w_rad) - Q_y * math.sin(w_rad)) * sec_epsilon
cos_Omega = P_x * math.cos(w_rad) - Q_x * math.sin(w_rad)

# Calculamos Ω en radianes y grados
Omega_rad = math.atan2(sin_Omega, cos_Omega)
Omega_grados = to_degrees_positive(Omega_rad)

# Cálculo de i (inclinación)
csc_Omega = 1 / math.sin(Omega_rad)  # cosec(Ω)
cos_i = -(P_x * math.sin(w_rad) + Q_x * math.cos(w_rad)) * csc_Omega

# Asegurar que un valor esté en el rango [-1, 1]
def clamp(value, min_value=-1, max_value=1):
    return max(min(value, max_value), min_value)

# Secante y cosecante de Ω
sec_epsilon = 1 / math.cos(math.radians(epsilon_rad))  # sec(ϵ)
csc_Omega = 1 / math.sin(Omega_rad)  # cosec(Ω)

# Cálculo de sen(i) a partir de 13.15.8 y 13.15.9

sin_i_1 = (P_z * math.cos(epsilon_rad) - P_y * math.sin(epsilon_rad)) / math.sin(w_rad)
sin_i_2 = (Q_z * math.cos(epsilon_rad) - Q_y * math.sin(epsilon_rad)) / math.cos(w_rad)

# Ajustar valores al rango [-1, 1]
sin_i_1 = clamp(sin_i_1)
sin_i_2 = clamp(sin_i_2)

# Promedio de los resultados (si ambos son consistentes)
sin_i = (sin_i_1 + sin_i_2) / 2

# Validación de cos(i)
cos_i = -(P_x * math.sin(w_rad) + Q_x * math.cos(w_rad)) * csc_Omega
cos_i = clamp(cos_i)

# Cálculo de la inclinación (i)
i_rad = math.atan2(sin_i, cos_i)
i_grados = to_degrees_positive(i_rad)

# Ajustar si el ángulo está en un rango inesperado
if i_grados > 90:  # Consideramos que el rango esperado es menor a 90 grados
    i_grados = 180 - i_grados
elif i_grados < -90:
    i_grados = -(180 + i_grados)

# Imprimir los resultados correctamente cerrando la cadena f-string.
#print(f"Lactus Rectum (l3): {l}")
print(f"\nLos elementtos Orbitales para el cuerpo menor son:")
print(f"\nExcentricidad (e): {e}")
print(f"Semi eje mayor (a): {a}")
#print(f"Período en años siderales (P): {P_sideral_years}")
#print(f"Período en días solares medios (dsm): {P_dsm}")
print(f"Argumento del perihelio (ω): {w_grados} grados")
print(f"Longitud del nodo ascendente (Ω): {Omega_grados} grados")
print(f"Inclinación (i): {i_grados} grados")
