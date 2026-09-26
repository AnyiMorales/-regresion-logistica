# Exploración computacional de la regresión logística

Proyecto semanal del curso *Introducción al Machine Learning*
(Universidad El Bosque, Programa de Matemáticas y Estadística).

**Integrantes:** Karol Salcedo y Anyi Morales

## Objetivo

Estudiar computacionalmente propiedades del modelo logístico 
(superficie del riesgo empírico, separabilidad de datos, y consistencia del
estimador al aumentar el tamaño de muestra) mediante experimentos
reproducibles, usando únicamente `numpy`, `scipy` y `matplotlib`.

## Estructura del repositorio

- `src/` — funciones reutilizables:
  - `modelo.py`: función sigmoide, probabilidad `p_theta(x)` y riesgo empírico logístico.
  - `optimizacion.py`: ajuste del modelo por minimización numérica (`scipy.optimize.minimize`).
  - `simulacion.py`: generación de datos sintéticos bajo un modelo logístico conocido.
- `notebooks/experimentos.ipynb` — cuaderno que importa las funciones de `src/`
  y ejecuta los tres ejercicios del taller.
- `resultados/` — figuras generadas por el cuaderno (`.png`).

## Instalación

1. Clona el repositorio y ubícate en su carpeta.
2. Crea un entorno virtual e instálalo:
```bash
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   pip install -r requirements.txt
```

## Ejecución

1. Abre `notebooks/experimentos.ipynb`.
2. Selecciona el kernel del entorno `venv`.
3. Ejecuta todas las celdas. Las figuras se guardan en `resultados/`.