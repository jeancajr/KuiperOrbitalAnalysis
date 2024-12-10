# Proyecto de Cálculo de Elementos Orbitales 
Este proyecto se enfoca en el cálculo de elementos orbitales utilizando el método de Gauss, comparando los resultados con el método de vectores y con los datos directos descargados de JPL Horizons. El análisis se realiza en el objeto Pallas (A802 FA). 

## Archivos del Proyecto 
- **GaussDataProcessing.py**: Procesa los datos descargados de JPL Horizons. 
- **VectorDataProcessing.py**: Procesa los datos descargados de JPL Horizons. 
- **Orbital_Data.xsls**: Archivo con los datos orbitales procesados. 
- **PallasVectors.csv**: Archivo CSV con datos de vectores procesados. 
- **GaussMethod.py**: Contiene el método para calcular los elementos orbitales usando el método de Gauss, siguiendo las referencias de Tatum, J. B. (2012). 
- **GraphingOrbitsandChiSquare.py**: Muestra las gráficas de los métodos de Gauss, vectores y elementos orbitales directos del JPL Horizons, e incluye la prueba de chi cuadrado. 

## Descripción del Proyecto El objetivo principal de este proyecto es comparar los elementos orbitales calculados mediante dos métodos: 
1. **Método de Gauss**: Implementado en `GaussMethod.py`, siguiendo las referencias del libro de Tatum, J. B. (2012). 
2. **Método de Vectores**: Implementado en `VectorDataProcessing.py`. Los resultados obtenidos se comparan con los datos proporcionados por JPL Horizons. Además, se realiza un análisis de chi cuadrado para determinar la precisión de cada método en comparación con los datos de referencia. 

## Referencias - Tatum, J. B. (2012). *Cálculo de elementos orbitales*. En *Mecánica Celestial: En Astronomía y Cosmología*. - JPL Horizons. Recuperado de https://ssd.jpl.nasa.gov/horizons/app.html#/ 

## Uso 1. Ejecuta `GaussDataProcessing.py` y `VectorDataProcessing.py` para procesar los datos. 
2. Ejecuta `GaussMethod.py` para calcular los elementos orbitales mediante el método de Gauss. 
3. Ejecuta `GraphingOrbitsandChiSquare.py` para generar las gráficas y realizar el análisis de chi cuadrado. 

## Contacto 
Para más información, contacta al autor del proyecto.