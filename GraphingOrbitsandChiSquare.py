#////////////////GRAFICA METODO DE GAUSS//////////////

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from mpl_toolkits.mplot3d import Axes3D


# Parámetros metodo de Gauss
a_Gauss = 2.809053148461936 * u.au  # Semi eje mayor en AU (Unidades astronómicas)
ecc_Gauss = 0.27131769339137674 * u.one  # Excentricidad, adimensional
inc_Gauss = 45.0 * u.deg  # Inclinación en grados
raan_Gauss = 156.8175283415527 * u.deg  # Longitud del nodo ascendente en grados
argp_Gauss = 337.5611303610332 * u.deg  # Argumento del perihelio en grados
nu_Gauss = 40.6 * u.deg  # Anomalía media en grados

orbit_Gauss= Orbit.from_classical(Sun, a_Gauss, ecc_Gauss, inc_Gauss, raan_Gauss, argp_Gauss, nu_Gauss)

# Calcular el período orbital
TGauss = orbit_Gauss.period.to(u.s).value  # Período en segundos

# Calcular la órbita completa
num_points3 = 500  # Cantidad de puntos en la órbita completa
true_anomalies3 = np.linspace(0, 2 * np.pi, num_points3)  # Anomalías verdaderas

# Listas para las coordenadas en x, y y z de cada órbita
x_1, y_1, z_1 = [], [], []

# Generar puntos para la órbita
for nu in true_anomalies3:
    time_of_flight_1 = (nu / (2 * np.pi)) * TGauss * u.s  # Suponiendo que tiene el mismo periodo que orbit1
    r_1, v_1 = orbit_Gauss.propagate(time_of_flight_1).rv()
    x_1.append(r_1[0].to(u.km).value)
    y_1.append(r_1[1].to(u.km).value)
    z_1.append(r_1[2].to(u.km).value)

# Crear la figura y el gráfico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar la órbita completa
ax.plot(x_1, y_1, z_1, label='Órbita Metodo de Gauss', color='b')

# Añadir el Sol en el origen
ax.scatter(0, 0, 0, color='orange', s=100, label='Sol')

# Configurar etiquetas y leyenda
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Órbita completa en 3D alrededor del Sol')
ax.legend()
# Cambiar la perspectiva
ax.view_init(elev=30, azim=30)  # Ajusta estos valores para explorar diferentes vistas

plt.show()

#//////////////////GRAFICA METODO DE VECTORES///////////////////////
import numpy as np
import pandas as pd
from astropy import units as u
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
import matplotlib.pyplot as plt

# Cargar el archivo CSV
data = pd.read_csv('PallasVectors.csv')

# Definir los datos de posición y velocidad con unidades
x = data['X'] * u.km
y = data['Y'] * u.km
z = data['Z'] * u.km
vx = data['VX'] * (u.km / u.s)
vy = data['VY'] * (u.km / u.s)
vz = data['VZ'] * (u.km / u.s)

# Crear los vectores de posición y velocidad iniciales como Quantity
r_initial = np.array([x.iloc[0], y.iloc[0], z.iloc[0]]) * u.km
v_initial = np.array([vx.iloc[0], vy.iloc[0], vz.iloc[0]]) * (u.km / u.s)

# Crear la órbita usando poliastro
orbit2 = Orbit.from_vectors(Sun, r_initial, v_initial)

true_anomaly_deg2 = orbit2.nu.to(u.deg).value
mean_anomaly = (true_anomaly_deg2 + orbit2.ecc.value * np.sin(np.deg2rad(true_anomaly_deg2))) * (180 / np.pi)

# Imprimir los elementos orbitales calculados
print(f"Semi-eje mayor (a_vectors): {orbit2.a.to(u.au):.6f}")
print(f"Eccentricidad (ecc_vectors): {orbit2.ecc:.6f}")
print(f"Inclinación (inc_vectors): {orbit2.inc.to(u.deg):.6f}")
print(f"Argumento del periastro (ω_vectors): {orbit2.argp.to(u.deg):.6f}")
print(f"Longitud del nodo ascendente (Ω_vectors): {orbit2.raan.to(u.deg):.6f}")
print(f"Anomalía media (ν_vectors): {orbit2.nu.to(u.deg):.6f}")

# Calcular el período orbital
T = orbit2.period.to(u.s).value  # Período en segundos

# Calcular la órbita completa
num_points2 = 500  # Cantidad de puntos en la órbita completa
true_anomalies2 = np.linspace(0, 2 * np.pi, num_points2)  # Anomalías verdaderas

# Inicializar arrays para las posiciones
x_full_orbit2 = []
y_full_orbit2 = []
z_full_orbit2 = []

# Generar puntos a lo largo de la órbita completa
for nu in true_anomalies2:
    # Calcular el tiempo correspondiente a la anomalía verdadera
    time_of_flight2 = (nu / (2 * np.pi)) * T * u.s  # Convertir a tiempo
    # Propagar la órbita al tiempo correspondiente
    r, v = orbit2.propagate(time_of_flight2).rv()  # r: vector de posición, v: vector de velocidad
    x_full_orbit2.append(r[0].to(u.km).value)
    y_full_orbit2.append(r[1].to(u.km).value)
    z_full_orbit2.append(r[2].to(u.km).value)

# Crear la figura y el gráfico 3D
fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Graficar la órbita completa
ax.plot(x_full_orbit2, y_full_orbit2, z_full_orbit2, label='Órbita completa Metodo Vectores', color='b')

# Añadir el Sol en el origen
ax.scatter(0, 0, 0, color='orange', s=100, label='Sol')

# Configurar etiquetas y leyenda
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Órbita completa en 3D alrededor del Sol')
ax.legend()

# Cambiar la perspectiva
ax.view_init(elev=30, azim=30)  # Ajusta estos valores para explorar diferentes vistas


plt.show()

#/////////////////////// GRAFICAS IMPLEMENTADAS, TRES METODOS ///////////////////////

import numpy as np
import matplotlib.pyplot as plt
from astropy import units as u
from poliastro.bodies import Sun
from poliastro.twobody import Orbit
from mpl_toolkits.mplot3d import Axes3D

# Parámetros dados Metodo de Gauss
a_Gauss = 2.7868423356598955 * u.au  # Semi eje mayor en AU (Unidades astronómicas)
ecc_Gauss = 0.24657895381743142 * u.one  # Excentricidad, adimensional
inc_Gauss = 36.82447887809102 * u.deg  # Inclinación en grados
raan_Gauss = 173.22927549348785 * u.deg  # Longitud del nodo ascendente en grados
argp_Gauss = 309.7233588239455 * u.deg  # Argumento del perihelio en grados
nu_Gauss = 40.6 * u.deg  # Anomalía media en grados

orbit_Gauss= Orbit.from_classical(Sun, a_Gauss, ecc_Gauss, inc_Gauss, raan_Gauss, argp_Gauss, nu_Gauss)
# Calcular el período orbital
T_gauss = orbit_Gauss.period.to(u.s).value  # Período en segundos
# Calcular el período orbital


# Elementos Orbitales Metodo de Vectores
a_vectors = 2.781031 * u.au  # Semi eje mayor en AU
ecc_vectors = 0.231413 * u.one  # Excentricidad, adimensional
inc_vectors = 34.871594 * u.deg  # Inclinación en grados
argp_vectors = 309.675390 * u.deg  # Argumento del periastro en grados
raan_vectors = 173.290362 * u.deg  # Longitud del nodo ascendente en grados
nu_vectors = 10.665016 * u.deg  # Anomalía media en grados

# # Crear la órbita teórica basada en los parámetros esperados
orbit_vectors = Orbit.from_classical(Sun, a_vectors, ecc_vectors, inc_vectors, raan_vectors, argp_vectors, nu_vectors)
T_vectors = orbit_vectors.period.to(u.s).value  # Período en segundos


# Parámetros directos o ideales
a_ideal = 2.77 * u.au  # Semi-major axis en AU (Unidades astronómicas)
ecc_ideal = 0.2302 * u.one  # Excentricidad, adimensional
inc_ideal = 34.93 * u.deg  # Inclinación en grados
raan_ideal = 172.92 * u.deg  # Longitud del nodo ascendente en grados
argp_ideal = 310.87 * u.deg  # Argumento del periapsis en grados
nu_ideal = 40.6 * u.deg  # Anomalía media en grados

# Crear la órbita con los parámetros ideales
orbit_ideal = Orbit.from_classical(Sun, a_ideal, ecc_ideal, inc_ideal, raan_ideal, argp_ideal, nu_ideal)
T_ideal = orbit_ideal.period.to(u.s).value  # Período en segundos

# # Parámetros de la órbita de la Tierra alrededor del Sol
a_tierra = (1  * u.au).to(u.km) # Distancia media Tierra-Sol
ecc_tierra = 0.0167 * u.one
inc_tierra = 0 * u.deg
raan_tierra = 0 * u.deg
argp_tierra = 102.9 * u.deg
nu_tierra = 0 * u.deg  # Posición inicial

# Crear la órbita de la Tierra
orbit_tierra = Orbit.from_classical(Sun, a_tierra, ecc_tierra, inc_tierra, raan_tierra, argp_tierra, nu_tierra)

#Calcular el período orbital

T_tierra = orbit_tierra.period.to(u.s).value

# Calcular la órbita completa
num_points3 = 500  # Cantidad de puntos en la órbita completa
true_anomalies3 = np.linspace(0, 2 * np.pi, num_points3)  # Anomalías verdaderas

# Listas para las coordenadas en x, y y z de cada órbita ideal
x_ideal, y_ideal, z_ideal = [], [], []

# Listas para las coordenadas en x, y y z de la órbita gauss
x_gauss, y_gauss, z_gauss = [], [], []

# Listas para las coordenadas en x, y y z de la órbita vectors
x_vectors, y_vectors, z_vectors = [], [], []

# Listas para las coordenadas en x, y y z de la órbita vectors
x_tierra, y_tierra, z_tierra = [], [], []

# Generar puntos para la órbita ideal
for nu in true_anomalies3:
    time_of_flight_ideal = (nu / (2 * np.pi)) * T_ideal * u.s  # Suponiendo que tiene el mismo periodo que orbit_ideal
    r_ideal, v_ideal = orbit_ideal.propagate(time_of_flight_ideal).rv()
    x_ideal.append(r_ideal[0].to(u.km).value)
    y_ideal.append(r_ideal[1].to(u.km).value)
    z_ideal.append(r_ideal[2].to(u.km).value)

# Generar puntos para la órbita Gauss
for nu in true_anomalies3:
    time_of_flight_gauss = (nu / (2 * np.pi)) * T_gauss * u.s  # Suponiendo que tiene el mismo periodo que orbit_gauss
    r_gauss, v_gauss = orbit_Gauss.propagate(time_of_flight_gauss).rv()
    x_gauss.append(r_gauss[0].to(u.km).value)
    y_gauss.append(r_gauss[1].to(u.km).value)
    z_gauss.append(r_gauss[2].to(u.km).value)

# Generar puntos para la órbita vectors
for nu in true_anomalies3:
    time_of_flight_vectors = (nu / (2 * np.pi)) * T_vectors * u.s  # Suponiendo que tiene el mismo periodo que orbit_vectors
    r_vectors, v_vectors = orbit_vectors.propagate(time_of_flight_vectors).rv()
    x_vectors.append(r_vectors[0].to(u.km).value)
    y_vectors.append(r_vectors[1].to(u.km).value)
    z_vectors.append(r_vectors[2].to(u.km).value)

# # Generar puntos para la órbita de la tierra
for nu in true_anomalies3:
    time_of_flight_3 = (nu / (2 * np.pi)) * T_tierra * u.s  # Suponiendo que tiene el mismo periodo que orbit1
    r_2, v_2 = orbit_tierra.propagate(time_of_flight_3).rv()
    x_tierra.append(r_2[0].to(u.km).value)
    y_tierra.append(r_2[1].to(u.km).value)
    z_tierra.append(r_2[2].to(u.km).value)

# Crear la figura y el gráfico 3D
# Crear la figura y el gráfico 3D
fig = plt.figure(figsize=(12, 8))
ax = fig.add_subplot(111, projection='3d')

# Graficar la órbita Gauss
ax.plot(x_gauss, y_gauss, z_gauss, label='Órbita Método de Gauss', color='b')

# Graficar la órbita vectors
ax.plot(x_vectors, y_vectors, z_vectors, label='Órbita Metodo Vectores', color='g')

# Graficar la órbita ideal
ax.plot(x_ideal, y_ideal, z_ideal, label="Órbita de Referencia", color="r", linestyle='--')
ax.plot(x_tierra, y_tierra, z_tierra, label="Órbita Tierra", color="brown", linestyle='-')

# Añadir el Sol en el origen
ax.scatter(0, 0, 0, color='orange', s=100, label='Sol')

# Configurar etiquetas y leyenda
ax.set_xlabel('X (km)')
ax.set_ylabel('Y (km)')
ax.set_zlabel('Z (km)')
ax.set_title('Órbita completa en 3D alrededor del Sol')
ax.legend()

# Cambiar la perspectiva
ax.view_init(elev=30, azim=30)  # Ajusta estos valores para explorar diferentes vistas


plt.show()


#//////////// PRUEBA DE CHI CUADRADO ///////////////////////

# Valores teóricos
valores_teoricos = np.array([a_ideal.to(u.km).value, ecc_ideal.value, inc_ideal.value, raan_ideal.value, argp_ideal.value])

# Valores calculados por el Método de Gauss
valores_gauss = np.array([a_Gauss.to(u.km).value, ecc_Gauss.value, inc_Gauss.value, raan_Gauss.value, argp_Gauss.value])

# Valores calculados por el Método de Vectores
valores_vectores = np.array([a_vectors.to(u.km).value, ecc_vectors.value, inc_vectors.value, raan_vectors.value, argp_vectors.value])

# Chi-cuadrado
chi_squared_gauss = np.sum((valores_gauss - valores_teoricos) ** 2 / valores_teoricos)
chi_squared_vectores = np.sum((valores_vectores - valores_teoricos) ** 2 / valores_teoricos)

print(f"Chi-cuadrado para el Método de Gauss: {chi_squared_gauss}")
print(f"Chi-cuadrado para el Método de Vectores: {chi_squared_vectores}")