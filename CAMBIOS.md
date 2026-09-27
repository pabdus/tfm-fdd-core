# Cambios del 22 de septiembre de 2026

Resultado de la revisión celda por celda del cuaderno 01 y del paquete. Todo lo que
sigue está explicado, con su motivo, en la última sección del cuaderno 01
(«Registro de cambios») y en `docs/decisiones.md`. Este fichero dice qué tocar y en
qué orden.

## Cómo aplicar

1. Descomprimir este zip **sobre** la raíz del repositorio, sobrescribiendo. No trae
   `data/`, `.venv/` ni `.git/`.
2. `pip install -e .` (hay una dependencia nueva: `pyyaml`, para que los scripts lean
   `configs/baseline.yaml`).
3. `pytest -q` → deben pasar 31 pruebas (20 anteriores + 11 nuevas en
   `tests/test_correcciones_2026_09.py`).
4. Abrir `notebooks/01_primeros_pasos.ipynb` y ejecutarlo completo (*Run All*). Las
   celdas modificadas o nuevas tienen las salidas borradas a propósito: no se
   fabricaron aquí porque los datos no viajan en el zip. Lo mismo con el `02`, cuyo
   conjunto de entrenamiento cambió. El `03` conserva sus salidas.
5. Leer el diff completo antes de hacer commit. Cada cambio hay que poder
   defenderlo con las propias palabras.
6. `python experiments/A0_calibracion_limites.py` y
   `python experiments/A1_pca_baseline.py` para regenerar los CSV con las columnas
   nuevas. La cifra del experimento de calibración que va a la cuartilla y a la
   memoria es la de `results/summary/A0_calibracion_limites.csv`, no ninguna anterior.
7. Commit y etiqueta nueva (`v0.3`): los compañeros instalan por etiqueta, y las
   funciones de métricas cambiaron de salida.

## Qué cambió y por qué

### Paquete `tfmfdd`

- `data.py`: `values(df, columns=None)` permite elegir un subconjunto de variables por
  nombre; `YIN_VARS` son las 33 de Yin et al. (2012), necesarias para que la
  verificación contra sus tablas sea comparable; `FAULT_ONSET_TRAIN_CLASSIC = 20`
  (Russell, Chiang y Braatz, 2000: el fallo entra tras 1 hora en los ficheros de
  entrenamiento; pendiente de contrastar con Chiang et al., 2001). Se corrigió la
  docstring de la transposición: el error no es silencioso, aparece en `score`.
- `pca.py`: la docstring de `fit` ya no cita cifras (12,5 % / 2,3 % con 17
  componentes, desactualizadas y sin configuración); remite al script A0 y aclara que
  el efecto solo existe en los límites del SPE calibrados con datos.
- `metrics.py`: `stratified_fdr` (FDR por mitades, partición fijada a priori),
  `detected_at_chance` en `evaluate_run` (racha con FDR < 0,10), y `aggregate` agrega
  las columnas nuevas. **Cambio de salida**: los CSV antiguos no las tienen.
- `stats.py`: `compare_methods` empareja por fallo y por simulación (antes, con
  varios fallos en la tabla, los índices se duplicaban y los pares se desalineaban
  sin avisar); `paired_test` usa `zero_method="zsplit"` en Wilcoxon porque las
  métricas acotadas producen muchas diferencias nulas.

### Experimentos

- `experiments/A0_calibracion_limites.py` (nuevo): el experimento de calibración
  **sin confundido**. Los dos modelos se ajustan con `X_fit`; solo cambia de dónde
  sale el límite. Escribe la configuración completa en el CSV. Admite `--yin-vars`.
- `experiments/A1_pca_baseline.py`: lee los valores por defecto de
  `configs/baseline.yaml` (antes usaba 0,90 de varianza mientras el yaml decía
  0,85). Un argumento de línea de comandos sigue sobrescribiendo para sensibilidad.

### Cuadernos

- `01_primeros_pasos.ipynb`: corrección de la sección 7 (confundido), figura de
  sedimentación con Kaiser y corte al 85 %, lectura de la tabla de 21 fallos en
  cuatro patrones, caso de estudio del fallo 5 (carta, tasas por bloque, FDR
  estratificada, contraste exploratorio con su advertencia), vocabulario de
  excedencia / alarma confirmada / fallo, nota sobre contribuciones (solo SPE,
  aislamiento ≠ diagnóstico), tabla de funciones actualizada y registro de cambios.
- `02_arranque_diagnostico.ipynb`: descarta las 20 primeras muestras de cada
  `dXX.dat` al etiquetar y explica por qué la validación es por simulación y no por
  muestra. Salidas borradas: hay que regenerarlas. El CSV que escribe es una prueba
  de humo de este repositorio, no un resultado del bloque B.
- `03_arranque_deeplearning.ipynb`: solo texto: la sección 5 ya tenía el diseño
  correcto, y se añade el precedente de Lyu et al. (2026) y las columnas nuevas.

### Documentos

- `docs/decisiones.md`: ocho filas nuevas con fecha 2026-09-22.
- `results/summary/README.md`: descripción de las columnas nuevas.
- `README.md` y cuadernos: español neutro, con ustedes (había formas de vosotros).
- `requirements.txt` y `pyproject.toml`: `pyyaml`.

## Pendiente que no se puede hacer sin los datos

- Ejecutar los cuadernos y los dos scripts y **anotar las cifras** con su
  configuración. La cifra de la cuartilla sale de A0.
- Transcribir la tabla de referencia de Yin et al. (2012) y correr la verificación en
  modo Yin (`--yin-vars`, 17 componentes, límite de Jackson, FAR sobre las muestras
  161–960 de `d00_te`): es el 5.1.1 de la primera entrega y hoy no existe.
- Validar la brecha de persistencia: debe salir grande en los fallos que la
  literatura describe como compensados por el control y cerca de cero en los
  escalones que persisten (1, 2, 6, 7). Si sale desordenada, la métrica no mide lo que
  dice y hay que decirlo.
- Acordar con el equipo la diferencia mínima relevante para los contrastes (propuesta:
  2 puntos porcentuales en FDR/FAR, 1 muestra en retardo) y dejarla en
  `docs/decisiones.md`.
