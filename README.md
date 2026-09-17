# tfm-fdd-core

Infraestructura común y bloque de detección del TFM *Implementación de enfoques data-driven para la detección de fallos en procesos industriales*.

Máster Universitario en Inteligencia Artificial, UNIR · Director: José Manuel Bernal de Lázaro
Autor de este repositorio: **Pablo Alberto Duque Marín**

---

## Qué hay aquí

Dos cosas separadas:

- **`tfmfdd/`** — el paquete común que usan los tres bloques del TFM. Carga de datos, particiones, escalado, límites de control, métricas de detección, contrastes estadísticos y estilo gráfico. Jonathan y Cesar lo instalan como dependencia; no trabajan dentro de él.
- **`experiments/`** — los experimentos del bloque de detección (parte individual 1), que son míos.

El paquete existe para que los tres bloques carguen los datos igual, los partan igual, los escalen igual y los midan igual. Sin eso los resultados no serían comparables, y la comparabilidad es la contribución del trabajo.

## Instalación

Para Jonathan y Cesar, que solo necesitan usar el paquete:

```bash
pip install git+https://github.com/pabdus/tfm-fdd-core@v0.1
```

Fijad siempre una **etiqueta de versión** (`@v0.1`), no la rama principal. Si apuntáis a la rama, un cambio mío altera vuestros resultados en silencio y las tablas de la memoria dejan de ser coherentes entre capítulos.

Para trabajar sobre el propio repositorio:

```bash
git clone https://github.com/pabdus/tfm-fdd-core.git
cd tfm-fdd-core
python -m venv .venv && source .venv/bin/activate   # en Windows: .venv\Scripts\activate
pip install -e .
pytest -q
```

## Los datos

No están en el repositorio y no deben estarlo: pesan demasiado y no es su sitio. Lo que sí está es cómo obtenerlos.

**TEP clásico** (26 MB, 44 ficheros `.dat`). Se descomprime en `data/tep_classic/`. Es el que se usa para prototipar y para verificar el protocolo contra las cifras publicadas.

**TEP-Rieth 2017** (~1,3 GB, cuatro ficheros `.RData`). Se descargan de Harvard Dataverse (DOI 10.7910/DVN/6C3JR1) a `data/tep_rieth/` y se convierten a Parquet. Son la fuente de los resultados definitivos, porque sus 500 simulaciones por condición permiten dar intervalos de confianza.

Aviso: **`d00.dat` viene transpuesto**, como (52, 500) en lugar de (500, 52). Es el único. Cargarlo sin corregir no lanza ningún error, simplemente ajusta el modelo sobre 52 observaciones y 500 variables y produce resultados sin sentido. El cargador lo detecta, lo corrige y avisa.

## Uso

```python
from tfmfdd import data, metrics
from tfmfdd.pca import PCAMonitor

# Datos de operación normal, separando ajuste y calibración
normal = data.values(data.load_classic(0, "train"))
X_fit, X_cal = normal[:350], normal[350:]

modelo = PCAMonitor(variance=0.85, alpha=0.99).fit(X_fit, X_cal=X_cal)

# Evaluación sobre un fallo
prueba = data.values(data.load_classic(1, "test"))
t2, spe = modelo.score(prueba)

print(metrics.evaluate_run(t2, modelo.t2_limit_, onset=data.FAULT_ONSET_TEST))
# {'far': 0.0, 'fdr': 0.99, 'delay_samples': 3, 'delay_minutes': 9.0, 'detected': True}
```

Para los otros dos bloques, lo que interesa sobre todo es `data` (para cargar igual que yo) y `metrics` (para medir igual que yo):

```python
from tfmfdd import data, metrics

df = data.load_classic(5, "test")
X = data.values(df)
y = df["faultNumber"].to_numpy()
# ... aquí va vuestro clasificador o vuestra red ...
```

## Reproducir el baseline

```bash
python experiments/A1_pca_baseline.py --variance 0.85 --alpha 0.99
```

Resultados sobre el TEP clásico, 27 componentes al 85 % de varianza, límites al 99 %:

| Estadístico | FAR | FDR medio | FDR sin fallos 3, 9 y 15 | FDR en 3, 9 y 15 |
|---|---|---|---|---|
| T² | 0,015 | 0,601 | 0,693 | 0,052 |
| SPE | 0,028 | 0,666 | 0,767 | 0,060 |

La tasa de falsas alarmas queda cerca del 1 % teórico y los tres fallos difíciles se detectan en torno al 5 % de las muestras, que es el orden de magnitud que reporta la literatura. Esa coincidencia es la verificación de que el protocolo está bien implementado.

## Una advertencia sobre los límites de control

Los límites se calibran con datos normales **reservados**, no con los mismos datos del ajuste. Parece un detalle y no lo es.

El modelo se ajusta minimizando el residuo sobre sus datos de entrenamiento, de modo que el SPE que observa ahí es artificialmente pequeño. Si el umbral se calibra con ese SPE optimista, queda demasiado estrecho, y en cuanto llegan datos que el modelo no ha visto el estadístico lo supera continuamente.

Medido sobre el TEP clásico con 17 componentes: calibrando con los datos del ajuste salen **12,5 %** de falsas alarmas; reservando datos normales para calibrar, **2,3 %**. Cinco veces menos, por un cambio de tres líneas.

Es una forma de fuga de datos poco discutida en la literatura y merece un párrafo en el capítulo 4.

## Estructura

```
tfmfdd/
  data.py           carga del TEP clásico y de Rieth, particiones por simulación
  preprocessing.py  escalado ajustado solo con entrenamiento, matriz de retardos
  limits.py         límites de control: F, Box, Jackson-Mudholkar, KDE, empírico
  metrics.py        FAR, FDR, retardo de detección, agregación con IC
  pca.py            monitor PCA y DPCA con T² y SPE
  stats.py          Wilcoxon pareado, corrección de Holm, tamaño de efecto
  plots.py          estilo gráfico único para toda la memoria
experiments/        experimentos del bloque de detección
configs/            parámetros del protocolo común
tests/              20 pruebas; ejecutar con pytest -q
docs/               registro de decisiones y actas de reunión
results/summary/    resúmenes versionados: la procedencia de cada tabla
```

## Contrato de datos

Todos los cargadores devuelven lo mismo, y de eso depende que los tres bloques sean comparables:

| Columna | Contenido |
|---|---|
| `faultNumber` | 0 = operación normal, 1 a 21 = tipo de fallo |
| `simulationRun` | identificador de la simulación (1 en el clásico) |
| `sample` | índice de muestra, empieza en 1 |
| `xmeas_1` … `xmeas_41` | variables medidas |
| `xmv_1` … `xmv_11` | variables manipuladas |

En los ficheros de prueba el fallo entra después de la muestra 160: las muestras 1 a 160 sirven para medir falsas alarmas y las 161 a 960 para medir detección y retardo.

## Si encontráis un fallo en el paquete

Abrid un *issue* con el caso que falla, o escribidme. **No hagáis commits aquí**: las instrucciones de UNIR exigen que cada repositorio tenga un único autor y en el mío no puede aparecer ningún commit vuestro. Yo lo corrijo, publico una versión nueva y aviso en el grupo para que los tres recalculemos lo que quede afectado.
