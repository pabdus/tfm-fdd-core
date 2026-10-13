# Notebooks

Cuadernos comentados que recorren el paquete paso a paso, con sus salidas guardadas para que se vea qué esperar antes de correr nada. Cada bloque de código dice qué módulo `.py` está usando, y al final de cada cuaderno hay una tabla que enlaza lo hecho con la función del paquete que lo hace. En el texto no hay ninguna cifra de resultado escrita a mano: las lecturas remiten a la salida de cada celda (regla 10 de `docs/reglas_de_medida.md`).

Actualizados a la versión 0.3.0 del paquete (la de la etiqueta `v0.3`) el 12 de octubre de 2026, y corregidos ese mismo día tras la revisión independiente: leen el yaml con `tfmfdd.config.load_baseline()`, el B00 y el C00 citan las catorce reglas de `docs/reglas_de_medida.md` con su numeración, y las búsquedas de hiperparámetros llevan `GridSearchCV(..., error_score="raise")`.

## Puntos de partida

**`B00_punto_de_partida_diagnostico.ipynb`**: para Jonathan (bloque B, OE2). Las catorce reglas de medida, el conjunto etiquetado del TEP clásico, el escalado solo con la clase normal dentro de una `Pipeline` (decisión 16), kNN, SVM y Random Forest como línea base, F1 por clase y macro-F1 con `metrics.diagnosis_metrics`, por qué el "f1" del 02 anterior no era un F1, los fallos 3, 9 y 15 aparte, el coste con `cost.measure_cost`, el experimento del OE2 en miniatura sobre Rieth (partición por muestra frente a partición por simulación, y la trampa de las gemelas), el inicio efectivo con `data.effective_onset`, el formato común y un contraste con `stats.compare_methods(..., baseline=...)`. Termina con las tareas de Jonathan (FCM, KFCM, limpieza de atípicos, SVM ajustada sin mirar la prueba, escalado a `datos.runs_finales`).

**`C00_punto_de_partida_deeplearning.ipynb`**: para Cesar (bloque C, OE3). Las catorce reglas de medida, el autoencoder como primo no lineal del SPE, su entrenamiento con parada temprana y su umbral con normales reservados, la comparación con el PCA-SPE en los 21 fallos, las dos trampas de protocolo medidas (umbral en muestra y umbral elegido con la prueba), Rieth con varias simulaciones e intervalos, el diagnóstico por ventanas que no cruzan simulaciones, el coste, y lo que hay que replicar de Lyu et al. (2026) con la verificación de su código. Termina con las tareas de Cesar.

## De arranque

**`01_primeros_pasos.ipynb`**: para los tres. Cargar el TEP clásico, mirar las señales, ajustar el PCA, calcular T² y SPE, dibujar la carta de control, medir FAR, FDR y retardo sobre los 21 fallos, resumir con `metrics.summary`, y ver por qué los umbrales se calibran con datos reservados. Termina con un registro de lo que cambió el 22/09 y el 12/10, y por qué.

**`02_arranque_diagnostico.ipynb`**: introducción corta al diagnóstico (kNN y Random Forest sobre el clásico, macro-F1 con `metrics.diagnosis_metrics`, matriz de confusión y formato común). Lo sustituye el B00 como punto de partida.

**`03_arranque_deeplearning.ipynb`**: introducción corta al autoencoder de detección con scikit-learn, comparado con el PCA-SPE, y la trampa de calibrar el umbral con los datos del ajuste. Lo sustituye el C00 como punto de partida.

## Con qué se ejecutó cada uno

| Cuaderno | Salidas | Entorno |
|---|---|---|
| `01`, `02`, `03`, `B00` | Regeneradas el 12/10/2026 con la versión 0.3.0 del paquete (`jupyter nbconvert --execute`), de principio a fin y sin errores | Python 3.12.0, numpy 2.4.6, pandas 3.0.3, scikit-learn 1.9.0 (el Python global del equipo de Pablo), con `tfmfdd` 0.3.0 |
| `C00` | **No se ha ejecutado con la v0.3.** Conserva salidas del 27/09/2026 (PyTorch 2.14.0+cpu y el paquete de entonces) solo en las celdas cuyo cambio no altera lo que imprimen; las demás están en blanco. Un aviso al principio del cuaderno lo dice | PyTorch no carga en el equipo de Pablo. Cesar tiene que ejecutarlo en su entorno. Las partes que no dependen de PyTorch se probaron en una copia por el camino de scikit-learn, también después de las correcciones del 12/10 |

## Lo que escriben

Todos escriben en `results/raw/`, que git ignora: son pruebas de humo, no resultados del TFM (regla 11).

- `02`: `02_clasico_recall_por_simulacion.csv` y `02_clasico_por_clase.csv`.
- `B00`: `B00_clasico_por_clase.csv`, `B00_rieth_particion.csv`, `B00_rieth_por_simulacion.csv`, `B00_rieth_por_clase.csv` y `B00_coste_clasico.json`.
- `C00`: `C00_deteccion_clasico.csv`, `C00_trampas_clasico.csv`, `C00_deteccion_rieth.csv`, `C00_diagnostico_por_simulacion.csv`, `C00_diagnostico_por_clase.csv`, `C00_diagnostico_macro.csv`, `C00_configuracion.json` y `C00_coste_clasico.json`.

**`results/summary/B1_clasificadores_clasico.csv` se retiró el 12/10/2026.** Salía del 02 con un "f1" por fallo calculado solo con las muestras de cada clase real, que es 2r/(1+r) y no un F1 (defecto B3 de la revisión del 27/09; la docstring de `metrics.diagnosis_metrics` lo explica), y además era una prueba de humo, que va a `results/raw/`. El 02 escribe ahora la tabla correcta en `results/raw/`.

## Para ejecutarlos

Hace falta el paquete instalado (`pip install -e .` desde la raíz) y el TEP clásico en `data/tep_classic/`. El B00 y el C00 usan además los Parquet de Rieth: define la variable de entorno `TFM_RIETH_DIR` con su carpeta (si faltan, esas partes se saltan con un mensaje). Los cuadernos suben solos a la raíz del repositorio si se abren desde `notebooks/`. De Rieth solo leen simulaciones que el registro ya da por miradas (`data.RIETH_SEEN_RUNS`; en desarrollo, las 1 a 20 de cada fichero), y lo comprueban antes de leer.

La configuración sale del yaml del protocolo, leído con `tfmfdd.config.load_baseline()` (regla 9): `configs/baseline.yaml` del directorio de trabajo o, si no está, la copia que viaja con el paquete. Cada cuaderno lee del yaml las claves con valor que usa, también las nuevas de la v0.3: el α de los contrastes (`contrastes.alpha_contrastes`), `protocolo.min_fdr` y `protocolo.retardo_convencion` (que van a `evaluate_run` y a `summary`), y los métodos de los límites (`pca.spe_method`, `pca.t2_method` y `pca.quantile_method`) con `pca.calibration`. Ninguna se declara a mano.

Las claves en null son decisiones pendientes y no se usan como valores. Lo que depende de ellas lo declara la primera celda de código como parámetro, sin tomar la decisión:

- `INCLUIR_NORMAL` (02, B00 y C00): si la clase normal entra en la macro-F1 (`diagnostico.incluir_normal`). Con `None`, el cuaderno da las dos variantes; cuando la clave tenga valor en el yaml, manda el yaml.
- `VENTANAS_PRUEBA` (C00): si las ventanas de prueba del diagnóstico pueden empezar antes del inicio del fallo. Aún no tiene clave en el yaml. Con `None`, da las dos opciones.
- `COSTE_DEMO` (B00 y C00): lote, repeticiones y calentamientos de la medida de coste, de demostración mientras `coste.*` siga en null; cuando esas claves tengan valor, la celda de coste las lee del yaml.

Los cuadernos son ejemplos. **No son el sitio donde va el trabajo de cada uno**: eso va en el repositorio de cada cual, importando el paquete.
