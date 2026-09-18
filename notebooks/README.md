# Notebooks

Tres cuadernos comentados que recorren el paquete paso a paso, con sus salidas ya ejecutadas para que se vea qué esperar antes de correr nada. Cada bloque de código dice qué módulo `.py` está usando, y al final de cada cuaderno hay una tabla que enlaza lo hecho con la función del paquete que lo hace.

Se leen en orden:

**`01_primeros_pasos.ipynb`** — para los tres. Cargar el TEP clásico, mirar las señales, ajustar el PCA, calcular T² y SPE, dibujar la carta de control, medir FAR, FDR y retardo sobre los 21 fallos, y ver con números por qué los umbrales se calibran con datos reservados.

**`02_arranque_diagnostico.ipynb`** — para Jonathan. Montar el conjunto etiquetado por tipo de fallo, estandarizar sin fuga de datos, entrenar kNN y Random Forest como línea base, F1 por clase, matriz de confusión, y el formato de resultados común. Termina donde empieza su bloque: el clustering difuso.

**`03_arranque_deeplearning.ipynb`** — para Cesar. Un autoencoder que detecta por error de reconstrucción, calibrado igual que el SPE del PCA, comparado contra él sobre los 21 fallos. La sección 5 muestra la trampa de calibración que infla buena parte de la literatura. Corre con scikit-learn para no exigir PyTorch; el flujo con PyTorch es idéntico.

Para ejecutarlos hace falta el paquete instalado (`pip install -e .` desde la raíz) y el TEP clásico en `data/tep_classic/`. Los cuadernos suben solos a la raíz del repositorio si se abren desde `notebooks/`.

Los cuadernos son ejemplos. **No son el sitio donde va el trabajo de cada uno**: eso va en el repositorio de cada cual, importando el paquete.
