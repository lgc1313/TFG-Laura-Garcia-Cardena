# TFG-Laura-Garcia-Cardena
Trabajo de Fin de Grado de Ingeniería Informática en la Universidad Nebrija. 

## Descripción breve:
Análisis de curvas del telescopio TESS con el fin de detectar eventos astrofísicos de interes mediante el uso de métodos  matemáticos, aprendizaje automático no supervisado y clustering.

## Fases principales:
- Descarga de archivos FITS
- Preprocesado y limpieza de las curvas
- Detección de periodicidad con Box Least Squares y Lomb-Scargle
- Detección de anomalías no periódicas con Isolation Forest
- Agregupación eventos  no periódicos mediante clustering
- Visualización de resultados

## Dependencias necesarias:
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Astropy
- SciPy
- Lightkurve

## Archivos:
- `tesscurl_sector_10_lc.sh`: archivo descargado de MAST utilizado para obtener las curvas de luz del sector 10 en formato FITS.
- `Análisis Curvas de Luz TESS.ipynb`: Notebook utilizado para desarrollar el código donde se muestran resultados. Las celdas de visualización de las gráficas de los clusters y del dataframe de periódicos no están ejecutados.
- `Análisis Curvas de Luz TESS.py`: Archivo ejecutable del código del Notebook.
