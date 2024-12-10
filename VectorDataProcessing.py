import pandas as pd

def limpiar_datos(input_file, output_file):
    with open(input_file, "r") as file:
        lines = file.readlines()

    # Encuentra la línea donde comienzan los datos y extrae solo esas líneas
    start_index = next(i for i, line in enumerate(lines) if "$$SOE" in line) + 1
    end_index = next(i for i, line in enumerate(lines) if "$$EOE" in line)
    data_lines = lines[start_index:end_index]

    # Procesa los datos para eliminar filas no deseadas y dividir en columnas
    data = []
    for line in data_lines:
        row = line.strip().split(",")
        
        # Elimina el último elemento si está vacío
        if row[-1] == '':
            row = row[:-1]
        
        # Verifica si la longitud de la fila es la esperada
        if len(row) == 11:
            data.append(row)
        else:
            print(f"Línea con columnas incorrectas: {row}")

    # Convierte a DataFrame y guarda el archivo limpio en formato CSV
    df = pd.DataFrame(data, columns=["JDTDB", "Calendar Date", "X", "Y", "Z", "VX", "VY", "VZ", "LT", "RG", "RR"])
    df.to_csv(output_file, index=False)

    print(f"Archivo limpio guardado en {output_file}")

# Ejemplo de uso
limpiar_datos('PallasVectors_results.txt', 'PallasVectors.csv')