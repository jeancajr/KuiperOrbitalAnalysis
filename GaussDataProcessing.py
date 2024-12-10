import pandas as pd
from datetime import datetime
import math

# Función para convertir R.A. y DEC en grados y radianes
def sexagesimal_to_decimal(ra_str, dec_str):
    # Convertir R.A.
    ra_parts = ra_str.split()
    ra_degrees = float(ra_parts[0]) + float(ra_parts[1])/60 + float(ra_parts[2])/3600
    ra_radians = math.radians(ra_degrees)

    # Convertir DEC
    dec_parts = dec_str.split()
    dec_degrees = float(dec_parts[0]) + float(dec_parts[1])/60 + float(dec_parts[2])/3600
    dec_radians = math.radians(dec_degrees)
    
    return ra_degrees, ra_radians, dec_degrees, dec_radians

# Abre el archivo y lee línea por línea
data = []
with open('horizons_resultsPallas.txt', 'r') as file:
    lines = file.readlines()
    for line in lines:
        # Eliminar espacios al principio y al final y omitir líneas vacías
        line = line.strip()
        if not line or line.startswith("$$SOE"):
            continue  # Omite las líneas vacías y las líneas que empiezan con "$$SOE"
        
        # Dividir la línea en partes
        parts = line.split()
        
        # Verificar si la primera parte parece una fecha válida
        try:
            if len(parts) >= 9:  # Asegurarse de que la línea tenga suficientes elementos
                # Fecha
                date_str = parts[0] + " " + parts[1]
                date = datetime.strptime(date_str, "%Y-%b-%d %H:%M").strftime("%Y %b %d")
                
                # R.A. y Dec
                ra_str = " ".join(parts[2:5])
                dec_str = " ".join(parts[5:8])
                
                # Convertir R.A. y Dec
                ra_degrees, ra_radians, dec_degrees, dec_radians = sexagesimal_to_decimal(ra_str, dec_str)
                
                # Guardar los datos en la lista
                data.append([date, ra_degrees, ra_radians, dec_degrees, dec_radians])
        except ValueError:
            # Si no se puede convertir la fecha, simplemente continuar con la siguiente línea
            continue

# Crear el DataFrame de pandas
df = pd.DataFrame(data, columns=["Date", "R.A. (degrees)", "R.A. (radians)", "Dec (degrees)", "Dec (radians)"])

# Guardar el DataFrame en un archivo .xlsx
df.to_excel('Orbital_data.xlsx', index=False)

print("El archivo Excel ha sido generado con éxito.")