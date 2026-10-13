# tfm-fdd-core

Paquete común `tfmfdd` y bloque de detección del TFM *Detección y diagnóstico de fallos en el proceso Tennessee Eastman: comparativa de enfoques data-driven bajo un protocolo de evaluación común*.

Máster Universitario en Inteligencia Artificial, UNIR · Director: José Manuel Bernal de Lázaro
Autor de este repositorio: **Pablo Alberto Duque Marín**

Versión del paquete: **0.3.0** (etiqueta `v0.3`). Qué cambió respecto a la anterior y qué resultados quedan afectados: `CAMBIOS.md`.

---

## Qué hay aquí

Dos cosas separadas:

- **`tfmfdd/`**: el paquete común que usan los tres bloques del TFM. Carga de datos y particiones, escalado, límites de control, monitor PCA y DPCA, métricas de detección y de diagnóstico, contrastes estadísticos, mecanismos de las cuatro ablaciones, coste computacional, índice de robustez, configuración, procedencia de los resultados y estilo gráfico. Jonathan y Cesar lo instalan como dependencia; no trabajan dentro de él.
- **`experiments/`**: los scripts del bloque de detección (parte individual 1) y la verificación frente a Yin et al. (2012).

El paquete existe para que los tres bloques carguen los datos igual, los partan igual, los escalen igual y los midan igual. Sin eso los resultados no serían comparables, y la comparabilidad es la contribución del trabajo. Las reglas que lo concretan están en `docs/reglas_de_medida.md`.

## Instalación

Para Jonathan y Cesar, que solo necesitan usar el paquete:

```bash
pip install git+https://github.com/pabdus/tfm-fdd-core@v0.3
```

Esa línea vale cuando Pablo publique la etiqueta `v0.3` en GitHub (lo avisará en el grupo); antes de eso, falla. Para comprobar la instalación: `python -c "import tfmfdd; print(tfmfdd.__version__)"` tiene que dar `0.3.0`. Convertir los `.RData` de Rieth pide además el extra `[rieth]` (`pip install "tfmfdd[rieth] @ git+https://github.com/pabdus/tfm-fdd-core@v0.3"`), pero no hace falta: los Parquet se reparten ya convertidos. Fijen siempre una **etiqueta de versión**, no la rama principal: si apuntan a la rama, un cambio en el paquete altera sus resultados en silencio y las tablas de la memoria dejan de ser coherentes entre capítulos.

Para trabajar sobre el propio repositorio:

```bash
git clone https://github.com/pabdus/tfm-fdd-core.git
cd tfm-fdd-core
pip install -e .
```

En Windows, antes de crear un entorno, lean la sección siguiente. `pyproject.toml` declara versiones mínimas. Las versiones exactas con que pasaron las pruebas de la v0.3 están en `requirements-probado.txt`; para reproducir ese entorno, `pip install -r requirements-probado.txt` y después `pip install -e . --no-deps`.

## El entorno en Windows

En el PC de Pablo, el Control inteligente de aplicaciones de Windows está activado. Mientras el repositorio estuvo dentro de la carpeta OneDrive, bloqueó bibliotecas compiladas de pandas, scipy, pyarrow y torch en su `.venv`; el 13/10, ya fuera de esa carpeta, el mismo `.venv` las carga todas. Las pruebas de la v0.3 se ejecutaron con el Python global y con un entorno en `C:\Users\pabdu\venvs\tfm`, los dos fuera de OneDrive. Los hechos comprobados, las opciones y cómo saber si les pasa a ustedes están en `docs/entorno.md`.

Si les sale un error del tipo "DLL load failed" que nombra una directiva de Control de aplicaciones, no cambien la configuración de seguridad de Windows por su cuenta: avísenle a Pablo.

## Los datos

No están en el repositorio y no deben estarlo: pesan demasiado y no es su sitio. Lo que sí está es cómo obtenerlos.

**TEP clásico** (Chiang, Russell y Braatz, 2001; 44 ficheros `.dat`, 26 MB). Van en `data/tep_classic/`, la carpeta por defecto de `data.load_classic` (otra, con `root=`). Se usa solo en el desarrollo y en la verificación frente a Yin et al. (2012); las ablaciones y los resultados de los tres bloques salen de Rieth (memoria, párrafo 367).

Aviso: **`d00.dat` viene transpuesto**, como (52, 500) en lugar de (500, 52). Es el único. El cargador lo detecta, lo corrige y avisa.

**Conjunto de Rieth et al. (2017)** (Harvard Dataverse, doi:10.7910/DVN/6C3JR1): 500 simulaciones por condición y split, que es lo que permite contrastes pareados sobre muchas simulaciones. Pablo les pasa los 42 Parquet ya convertidos (`train_fault00.parquet` a `test_fault20.parquet`, 1,3 GB) con su manifiesto `SHA256SUMS.txt`, el script `verificar_rieth.py` y un LEEME. Pasos:

1. Guárdenlos fuera de OneDrive y de Google Drive (por ejemplo, en `C:\Users\<usuario>\datos_tfm\tep_rieth`).
2. Compruébenlos: `python verificar_rieth.py <carpeta>` tiene que terminar con "42 de 42 ficheros coinciden en tamaño y SHA-256".
3. Definan la variable de entorno `TFM_RIETH_DIR` con esa carpeta. `data.load_rieth` la lee sola; si no está definida, busca en `data/tep_rieth`.

Convertir los `.RData` uno mismo (`python -m tfmfdd.convert_rieth`) no entra en la memoria de un equipo de 16 GB (revisión del 27/09, DAT-2): usen los Parquet.

Las simulaciones de Rieth que ya se miraron en desarrollo no sirven para el resultado confirmatorio. Están registradas en `data.RIETH_SEEN_RUNS` (consulta: `data.seen_runs()` y `data.unseen_runs()`); en desarrollo, usen las simulaciones 1 a 20 (regla 3 de `docs/reglas_de_medida.md`).

Variables de entorno: `TFM_RIETH_DIR` la usan el paquete y las pruebas; `TFM_CLASSIC_DIR` (la carpeta de los `.dat`), solo las pruebas.

## Uso

Un ejemplo mínimo con el TEP clásico, que toma la configuración de `configs/baseline.yaml` y calibra los límites con datos normales reservados:

```python
from tfmfdd import config, data, metrics
from tfmfdd.pca import PCAMonitor

cfg = config.load_baseline()                                    # configs/baseline.yaml
normal = data.values(data.load_classic(0, "train"))             # d00.dat: 500 x 52
corte = int(len(normal) * (1 - cfg["particion"]["calibracion_limites"]))
X_fit, X_cal = normal[:corte], normal[corte:]                   # ajuste y normales reservadas

modelo = PCAMonitor(variance=cfg["pca"]["variance"], alpha=cfg["protocolo"]["alpha"],
                    spe_method=cfg["pca"]["spe_method"]).fit(X_fit, X_cal=X_cal)
print(modelo.n_components_, modelo.n_fit_, modelo.n_cal_, modelo.limit_source_)

prueba = data.values(data.load_classic(1, "test"))              # d01_te.dat: 960 x 52
t2, spe = modelo.score(prueba)
r = metrics.evaluate_run(spe, modelo.spe_limit_, onset=data.FAULT_ONSET_TEST,
                         k=cfg["protocolo"]["k_alarmas"])
print({c: r[c] for c in ("far", "fdr", "delay_samples", "detected", "detected_at_chance",
                         "n_far", "n_fdr")})
```

Salida de una ejecución real (12/10/2026, `tfmfdd` 0.3.0, Python global con las versiones de `requirements-probado.txt`, desde la raíz del repositorio), después del aviso de que `d00.dat` viene transpuesto:

```
27 350 150 {'t2': 'parametric_fit', 'spe': 'holdout'}
{'far': 0.025, 'fdr': 0.9975, 'delay_samples': 2, 'detected': True, 'detected_at_chance': False, 'n_far': 160, 'n_fdr': 800}
```

Configuración de esa salida: las 52 variables; ajuste con las 350 primeras filas de `d00.dat` y calibración con las 150 últimas (`particion.calibracion_limites` = 0,3); varianza acumulada 0,85, que da 27 componentes; nivel de confianza 0,99; T² con la F de Fase II (paramétrico, con cantidades del ajuste) y SPE por Box calibrado con las 150 filas reservadas; estadístico SPE; fallo 1 (`d01_te.dat`), con el fallo tras la muestra 160; alarma confirmada con k = 3. La FAR se mide en las muestras 1 a 160 (`n_far`) y la FDR en las 161 a 960 (`n_fdr`). Es desarrollo sobre el TEP clásico: sirve para comprobar que todo funciona, no es un resultado del TFM.

Para los otros dos bloques, lo que interesa sobre todo es `data` (para cargar igual) y `metrics` (para medir igual):

```python
from tfmfdd import data

df = data.load_rieth(1, "test", runs=[1, 2])    # carpeta: TFM_RIETH_DIR; ids 1 y 2, ya mirados
X = data.values(df)                              # solo las 52 variables, en float64
y = df["faultNumber"].to_numpy()                 # la etiqueta, aparte
print(df.shape, df["simulationRun"].dtype, X.shape, X.dtype)
# (1920, 55) int64 (1920, 52) float64            <- salida real, 12/10/2026
```

El escalador del diagnóstico va dentro de la `Pipeline` y se ajusta solo con la clase normal (`preprocessing.Scaler()`, regla 2); la métrica es `metrics.diagnosis_metrics(y_true, y_pred, include_normal=...)` (regla 5). Los cuadernos `notebooks/B00_punto_de_partida_diagnostico.ipynb` y `notebooks/C00_punto_de_partida_deeplearning.ipynb` son los puntos de partida de los bloques B y C; el resto, en `notebooks/README.md`.

## Pruebas

Desde la raíz del repositorio, con el paquete instalado (`pip install -e .`):

```bash
python -m pytest -q
```

Las pruebas que leen datos reales se activan con `TFM_CLASSIC_DIR` y `TFM_RIETH_DIR`; sin ellas se saltan y dicen qué variable falta. El 12/10/2026, con el Python global de Pablo (versiones de `requirements-probado.txt`), `PYTHONPATH` en la raíz y `python -m pytest -q -p no:cacheprovider`:

- con las dos variables: `643 passed`;
- sin ellas: `633 passed, 10 skipped`.

## La verificación frente a Yin et al. (2012)

La memoria promete verificar **la implementación**, no el protocolo: el código, ejecutado con la configuración de Yin et al. (2012), que calibra los límites en muestra, tiene que reproducir la FDR por fallo y la FAR en operación normal de sus Tablas 4 y 5, "sin ajustes para que coincidan y sin contar los fallos 3, 9 y 15", dentro de una tolerancia "fijada antes de ejecutar" (párrafos 355 y 369). Parecerse a sus cifras con otra configuración no verifica nada: A1, con 52 variables y límites calibrados con datos reservados, no es la verificación.

La hace `experiments/V1_verificacion_yin.py` con `configs/yin2012.yaml` y la transcripción de `configs/referencias/`. El orden:

1. Pablo verifica la transcripción contra el PDF (`configs/referencias/LEEME_yin2012_transcripcion.md`) y escribe la fecha en el yaml.
2. Se rellenan las claves pendientes del yaml (la ventana de la FAR, la tolerancia, la configuración secundaria) y se revisan los supuestos.
3. `python experiments/V1_verificacion_yin.py --huella` imprime la huella del yaml y de las referencias, que se registra en una fila fechada y visible de `docs/decisiones.md`.
4. Commit del yaml, del registro, de las referencias y del código que se va a ejecutar (el script y `tfmfdd/`), en el mismo repositorio.
5. `python experiments/V1_verificacion_yin.py`, desde la raíz y con el paquete de ese repositorio (instalado en modo editable o con `PYTHONPATH`): con un `tfmfdd` instalado desde una rueda, V1 se niega.

Hasta que se cumple todo eso, V1 no abre ningún fichero de prueba. `--solo-limites` calcula los límites solo con `d00.dat`, el fichero de ajuste, y los compara con los que dibujan Yin et al.; no necesita nada de lo anterior. El detalle, con los supuestos y su evidencia, está en `docs/verificacion_yin.md`.

## El registro de decisiones

`docs/decisiones.md` recoge con fecha lo que se decide, quién y por qué; no se borra nada, y lo superado se anota. Las decisiones pendientes que necesita el código están en los yaml como claves con valor null: ningún script las sustituye por un valor por defecto, y el que necesita una se niega a ejecutar (`tfmfdd.config.require`). Antes de los experimentos confirmatorios, lo que fija el planteamiento de la comparativa se escribe ahí y queda fijado con una etiqueta de versión (memoria, párrafo 369). La última sección de `docs/decisiones.md` dice cómo se registra la fila de la verificación y cómo listar las pendientes.

## Reproducir los resultados de desarrollo

```bash
python experiments/A1_pca_baseline.py                  # línea base PCA, configuración del yaml
python experiments/A0_calibracion_limites.py           # límites calibrados en muestra frente a reservados
python experiments/V1_verificacion_yin.py --solo-limites
```

Escriben en `results/summary/` un CSV con la configuración en cada fila y, al lado, su `.provenance.json` (configuración efectiva, ficheros con su SHA-256, ids, commit, máquina y versiones). Todo lo que hay en `results/summary/` es desarrollo sobre el TEP clásico, no resultados del TFM; las cifras se leen en esos CSV con su configuración, no en este README.

## Una advertencia sobre los límites de control

En el protocolo, los límites se calibran con datos normales **reservados**, que el modelo no vio (regla 3). El motivo es el mecanismo, no una cifra: el modelo se ajusta para describir sus propios datos, así que su estadístico en esos datos tiende a salir distinto del que tendrá en datos nuevos, y un límite calibrado con él describe los datos del ajuste, no los nuevos. Cuánto cambia la tasa de falsas alarmas, y en qué sentido, depende del límite, del tamaño del ajuste y de los datos: el signo del efecto no se presupone, se mide y se reporta (memoria, párrafos 279 y 284). Es la ablación del factor 1, sobre Rieth; `experiments/A0_calibracion_limites.py` es su desarrollo sobre el TEP clásico.

Calibrar con los datos del ajuste es un **sesgo**, no una fuga: no usa la prueba. Calibrar con la prueba sí sería una fuga (memoria, párrafos 283 y 307). En el paquete, `PCAMonitor.limit_source_` dice de dónde sale cada límite, `fit(X)` sin `X_cal` avisa cuando calibra algún límite con los datos del ajuste, y la calibración en muestra a propósito (la de Yin) se pide con `fit(X, calibration="in_sample")`.

## Estructura

```
tfmfdd/
  data.py           carga del TEP clásico y de Rieth, particiones por simulación, registro de simulaciones miradas
  preprocessing.py  escalado ajustado solo con datos normales de entrenamiento, matriz de retardos por trayectoria
  limits.py         límites de control: F y Beta del T², Box y Jackson-Mudholkar del SPE, KDE, empírico
  pca.py            monitor PCA y DPCA con T², SPE y la regla T² o SPE
  metrics.py        FAR, FDR, retardo, falsas alarmas por hora, PR-AUC, agregados con IC, macro-F1
  stats.py          Wilcoxon pareado exacto, Holm, Hodges-Lehmann, no superioridad y veredicto de la hipótesis,
                    efecto frente a un margen, ventaja publicada, mejor compromiso
  ablation.py       mecanismos de las cuatro ablaciones (particiones, brazo S2, una frente a muchas, ventanas)
  cost.py           coste: tiempo de ajuste, tiempo de inferencia por muestra, pico de memoria, máquina
  robustness.py     índice de robustez de Bernal-de Lázaro et al. (2018) y ruido por niveles
  config.py         lectura de configs/baseline.yaml (y de la copia que viaja con el paquete)
  provenance.py     el .provenance.json de cada resultado
  plots.py          estilo gráfico único para toda la memoria
  convert_rieth.py  conversión de los .RData de Rieth a Parquet
experiments/        A0 y A1 (desarrollo del bloque de detección) y V1 (verificación frente a Yin)
notebooks/          cuadernos comentados; B00 y C00 son los puntos de partida de los bloques B y C
configs/            baseline.yaml (protocolo común), yin2012.yaml y referencias/ (verificación)
tests/              643 pruebas; python -m pytest -q
docs/               reglas de medida, registro de decisiones, entorno, verificación, revisiones fechadas
results/summary/    resúmenes versionados, cada uno con su .provenance.json
```

## Contrato de datos

Todos los cargadores devuelven lo mismo, y de eso depende que los tres bloques sean comparables:

| Columna | Contenido |
|---|---|
| `faultNumber` | 0 = operación normal; 1 a 21 = tipo de fallo en el clásico, 1 a 20 en Rieth |
| `simulationRun` | identificador de la simulación (1 en el clásico) |
| `sample` | índice de muestra, empieza en 1 |
| `xmeas_1` … `xmeas_41` | variables medidas |
| `xmv_1` … `xmv_11` | variables manipuladas |

Los tres identificadores llegan en `int64` en las dos fuentes. Las variables, en `float64` en el clásico y en `float32` en Rieth, que es la precisión con que se convirtieron; `data.values` devuelve siempre `float64`. Las columnas de identificación (`data.ID_COLUMNS`) no son variables y ningún modelo las recibe: `sample` delata el inicio fijo del fallo y `simulationRun`, la trayectoria. La matriz de entrada se construye con `data.values`, que se niega a devolverlas.

En los ficheros de prueba de las dos fuentes, el fallo entra después de la muestra 160: las muestras 1 a 160 sirven para medir falsas alarmas y las 161 a 960 para medir detección y retardo. En los de entrenamiento, el fallo está activo desde la primera fila en el clásico (`data.fault_onset("train", "classic")` devuelve `None`) y entra tras la muestra 20 en Rieth. En Rieth, la simulación r de cada fichero de fallo es gemela de la simulación normal r: por eso se parte por `simulationRun` en todas las clases a la vez (regla 1).

## Si encuentran un fallo en el paquete

Abran un *issue* con el caso que falla, o escríbanle a Pablo. **No hagan commits aquí**: cada repositorio tiene un único autor (`docs/decisiones.md`, 16/09). Pablo lo corrige, publica una versión nueva, la anota en `CAMBIOS.md` con los resultados que quedan afectados y avisa en el grupo para que los tres recalculen lo que haga falta.
