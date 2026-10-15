# Registro de cambios

Cada versión dice qué cambió y **qué resultados quedan afectados**. Es la norma del paquete común: un cambio se anuncia y se dice qué hay que recalcular. Una cifra calculada con una versión anterior no se cita junto a las de esta sin recalcularla, o sin decir con qué versión salió.

## Sin publicar (14 de octubre de 2026)

Cierre de la configuración de la verificación frente a Yin et al. (2012). No cambia el código de `tfmfdd/` ni el de `experiments/`.

- `configs/yin2012.yaml`: Pablo fijó las decisiones pendientes (tolerancia A con comparación por conteo, DPCA descriptivo, ventana de la FAR [160, null] y configuración secundaria con DPCA l = 40) y la fecha en que verificó la transcripción. Fila del 2026-10-14 en `docs/decisiones.md`.
- `configs/referencias/`: solo cambian las líneas de comentario que decían "pendiente de la verificacion de Pablo". Ningún valor cambia, pero la huella sí, porque resume el texto entero de los dos CSV: se registra después de este cambio.
- `tests/test_v03_experimentos.py`: las tres pruebas que exigían decisiones pendientes en el yaml real usan ahora una copia con esas claves en null, y la del yaml real comprueba que ya no le queda nada pendiente.
- `docs/verificacion_yin.md` y `docs/reglas_de_medida.md`: la regla de las variantes (no rescatan el veredicto), la primera ejecución, el DPCA descriptivo y el estado de S9, S12 y la configuración secundaria.

**Qué resultados quedan afectados:** ninguno. Jonathan y Cesar siguen con la etiqueta v0.3 y no tienen que hacer nada.

## v0.3 (12 de octubre de 2026)

Sale de la revisión del código del 11 y el 12 de octubre sobre la rama `revision-2026-09` (commit `58a07e2`). Este repositorio se volvió a crear el 13/10 con la v0.3 como primer commit: la historia anterior, `58a07e2` incluido, solo está en la copia de Pablo. La versión del paquete pasa de 0.1.0, la que decían `pyproject.toml` y `tfmfdd/__init__.py`, a 0.3.0. Los identificadores entre paréntesis (B2, M7, DAT-5, H8…) remiten a la revisión del 27/09 (`docs/revision_2026-09-27.md` y `docs/revision_2026-09-27_agentes.md`), a la del 03/10 (informe 07, código) y a los informes de la revisión del 12/10.

### Cómo actualizar

- **Jonathan y Cesar**, cuando Pablo publique la etiqueta: `pip install --upgrade git+https://github.com/pabdus/tfm-fdd-core@v0.3` (y `@v0.3` en su `requirements.txt`). Después, recalcular lo de la lista siguiente que les afecte.
- **Pablo**, antes de publicar la etiqueta:
  1. Ejecutar la suite (`README.md`, "Pruebas").
  2. Después del commit, regenerar `results/summary/` desde la raíz (`python experiments/A1_pca_baseline.py`, `python experiments/A0_calibracion_limites.py` con y sin `--yin-vars`, y `python experiments/V1_verificacion_yin.py --solo-limites`), para que cada `.provenance.json` registre un commit limpio. Hecho el 13/10 con el entorno de la opción 2 de `docs/entorno.md`, sobre el primer commit de este repositorio: los diez CSV salieron idénticos a los confirmados y cada `.provenance.json` registra ese commit sin cambios sin confirmar.
  3. Los cuadernos 01, 02, 03 y B00 ya se volvieron a ejecutar en el repositorio el 12/10, sin errores, con el Python global; no hace falta repetirlos salvo que cambie algo. El C00 sigue sin volver a ejecutarse: desde el 13/10 PyTorch carga en el equipo de Pablo (`docs/entorno.md`), así que se puede ejecutar allí o pedírselo a Cesar.
  4. Publicar la etiqueta `v0.3` y mandar la línea de instalación.

### Qué resultados quedan afectados (resumen)

1. **Retardos agregados.** `metrics.aggregate` y `metrics.summary` ya no promedian los retardos de las rachas al azar (`detected_at_chance`). Cambia cualquier retardo medio calculado con ellos; en un fallo cuya única racha es al azar, el retardo medio queda vacío.
2. **DPCA.** `PCAMonitor.score` devuelve una fila por fila de entrada y deja sin evaluar (NaN) las `lags` primeras de cada trayectoria. El inicio del fallo es el mismo índice que en el PCA. Cambian el retardo y la FDR de cualquier DPCA evaluado con la v0.2 sin la vuelta `onset − lags`, y su FAR, que ya no cuenta esas muestras.
3. **Contrastes.** El p de Wilcoxon es exacto con empates y ceros (antes, scipy daba la aproximación normal con empates o ceros y más de 13 pares, y con más de 50 siempre). `effect_size` es ahora la correlación biserial de rangos, con signo. Holm deja los p NaN fuera de la familia. Cambian los p y los tamaños de efecto de cualquier contraste con empates (los de B00 y C00 entre ellos).
4. **Escalado del diagnóstico.** `Scaler(normal_label=0)` por defecto: con etiquetas (dentro de una `Pipeline`), se ajusta solo con la clase normal (memoria, párrafo 367; decisión del 04/10). Cambian las cifras de los clasificadores sensibles a la escala que se escalaban con todas las clases: en el B00, el kNN y la SVM; el bosque aleatorio cambió muy poco (ejecución del 12/10).
5. **`PCAMonitor()` sin `variance`.** El valor por defecto pasa de 0,90 a 0,85, el de `configs/baseline.yaml` (REP-4). Cambia el número de componentes de quien no lo pasaba.
6. **Claves de `summary`.** `far_medio` y `fallos_nunca_detectados` se retiran: pedirlas da un `KeyError` que dice cuál usar.
7. **Inicio del fallo en el entrenamiento del clásico.** `data.FAULT_ONSET_TRAIN_CLASSIC` (20) era falsa y se retira (B2). Lo que descartaba las 20 primeras filas de los `dXX.dat` estaba mal: el cuaderno 02 y su CSV `results/summary/B1_clasificadores_clasico.csv`, que además usaba un "f1" que era 2r/(1 + r) (D2). El CSV se retira.
8. **Ids de Rieth.** `load_rieth` los devuelve en `int64` (antes `int16`, que desborda cuentas como `simulationRun * 1000 + sample`) y falla con ids inexistentes (antes los ignoraba y devolvía menos simulaciones sin avisar). Cambia cualquier cálculo que dependiera de eso.
9. **Muestras no evaluadas.** Un NaN del estadístico ya no cuenta como "sin alarma" en la FAR ni en la FDR, y un límite NaN o infinito es un error. Cambian las tasas de cualquier serie con NaN.
10. **`X_cal` vacío.** Es un error (M11). Con la v0.2, `A1 --cal-fraction 0` daba un límite NaN y escribía FAR 0 en el CSV: esos CSV no valen.
11. **Límites KDE.** `limits.limit_kde` da ahora el cuantil exacto de la KDE (ronda de correcciones, I5). La v0.2 usaba la misma rejilla que se corrigió: cambia cualquier límite KDE calculado antes, un poco con estadísticos positivos, y del todo con estadísticos negativos, donde el de antes no tenía sentido.

### Ronda de correcciones del 12/10 (antes de publicar)

Dos revisiones independientes del código unido, una del código (hallazgos V-01 a V-17) y otra de la estadística (I1 a I6 y M1 a M11), encontraron fallos que no dan un resultado equivocado por la vía que usará el TFM con Rieth, pero que se corrigieron antes de publicar. Aquí va, por tema, qué cambió y a qué resultados afecta. "Antes" es la v0.3 sin corregir, con la que se ejecutaron el 12/10 los cuadernos 01, 02, 03 y B00; donde el cambio afecta también a lo calculado con la v0.2, se dice.

#### Límites, monitor y preprocesado

- **`limits.limit_kde` da el cuantil exacto de la KDE** (I5): la raíz de su función de distribución, que es una mezcla de normales, en lugar de una rejilla que llegaba hasta 1,5 veces el máximo. Afecta a todo límite KDE (`limit_kde`, `t2_method="kde"`, `spe_method="kde"`), también a los de la v0.2, que usaba la misma rejilla: con estadísticos positivos cambia un poco, sobre todo cerca de 0 (un chi-cuadrado con pocos grados de libertad, una lognormal), y con estadísticos negativos deja de dar un límite absurdo. Ningún resultado versionado usa la KDE; el umbral KDE del autoencoder del C00 (celda `c00-19`, cuya salida guardada es del 27/09) cambiará al volver a ejecutarlo.
- **`limits.spe_limit_jackson` exige los autovalores en orden decreciente** (M6): si no, `ValueError` con el remedio. Antes daba otro límite sin avisar. Afecta solo a quien lo llamara con los autovalores en orden creciente (el de `np.linalg.eigh`); `PCAMonitor` ya los ordenaba, y con él nada cambia.
- **El escalador externo se copia** (V-01): `PCAMonitor.fit(..., scaler=...)` guarda una copia, y el atributo nuevo `scaler_n_samples_fit_` dice con cuántas filas se ajustó. Antes se guardaba el mismo objeto: reajustarlo después (por ejemplo, para el brazo S1) cambiaba en silencio el modelo limpio. Afecta a las comparaciones de brazos que reutilizaran el objeto; las que creaban un escalador nuevo para cada brazo, como V1, dan lo mismo.
- **Aviso de trayectoria demasiado larga** (V-02): `preprocessing.TEP_MAX_RUN_LENGTH = 960`, y `trajectory_starts` avisa cuando una trayectoria lo supera, ya venga de `lag_by_run`, de `PCAMonitor` con retardos o de las funciones de `ablation` que cortan por trayectoria. Delata dos ficheros seguidos con el mismo `simulationRun` (en el clásico vale 1 en todos), que se fundían en una sola trayectoria. Afecta a lo que se calculara juntando ficheros con `runs` de una sola columna (DPCA, ventanas, brazo S2), que estaba mal; A1 y V1 pasan fichero a fichero y no avisan.
- **`samples=`** (V-10): `preprocessing.lag_by_run(..., samples=)`, `PCAMonitor.fit(..., samples=, samples_cal=)` y `PCAMonitor.score(..., samples=)` reciben la columna `sample` y, con retardos, comprueban que las muestras de cada trayectoria son consecutivas (si no, `ValueError`). Es opcional: sin `samples`, todo sigue como antes. Detiene el DPCA construido sobre un reparto por muestra, que antes apilaba muestras no consecutivas sin aviso.
- **`Scaler(normal_label=...)`** (V-09): el docstring dice ahora que la etiqueta es la que llega en `y` (si se recodifican las clases, hay que pasar la nueva) y que las búsquedas van con `error_score="raise"`, porque un pliegue sin la clase normal no debe convertirse en una puntuación NaN. Afecta: nada (solo docstring).

#### Datos

- **Un entero de numpy en `load_rieth(runs=...)` es un `TypeError`** (V-04), con un mensaje que dice qué pasar: `[n]` para esa simulación, `int(n)` para las n primeras. Antes cargaba las n primeras con solo un aviso, así que un id de `split_runs` (que es un entero de numpy) leía cientos de simulaciones, quizá no miradas: si alguien lo hizo, conviene revisar qué leyó. Los cuadernos pasan listas de enteros y no les afecta.
- **`effective_onset` cuenta por el valor de `sample`** (V-07): en cada (fallo, simulación) exige las muestras posteriores al inicio desde `onset + 1` y seguidas hasta la primera distinta de la gemela; si no, `ValueError`. Antes, con filas recortadas o con ventanas, daba un número falso. Con las filas completas el resultado no cambia (el contraste con el informe 11 vuelve a dar 800 de 800), y los cuadernos se las pasan completas.
- **En `split_runs`, una parte con fracción 0 queda vacía** (V-08). Antes, el redondeo podía darle simulaciones, también en la v0.2. Con las tres fracciones positivas el reparto no cambia (comprobado en 2816 combinaciones de tamaño, fracciones y semilla, y con `exclude`). `ablation.split_by_run` llama ahora a `data.split_runs` (el mismo reparto, redondeo incluido) y exige ids enteros desde 1.
- Docstrings: el de `data.py` habla de verificar la implementación, no el protocolo; el de `convert_rieth` y su mensaje final ya no remiten a un script de descarga que no existe ni dicen que la conversión entre en 16 GB (DAT-2, DAT-3). Afecta: nada.

#### Experimentos, configuración y procedencia

- **La regla de orden de V1 cubre el código** (V-05, V-11, V-12). Comprueba ruta por ruta (`provenance.committed_status`) que el yaml, el registro, las referencias, el propio script y la carpeta del paquete `tfmfdd` que se importa están confirmados en un mismo repositorio: versionados, sin cambios, fuera de `.git` y sin marcas skip-worktree ni assume-unchanged. Por eso V1 solo se ejecuta con el paquete del repositorio (instalación editable o `PYTHONPATH`), no con una rueda instalada. La fila del registro tiene que verse (fuera de comentarios `<!-- -->` y de bloques de código) y su fecha tiene que existir. Con un yaml no válido (una variante con un supuesto en null, una tolerancia no finita) se niega sin leer nada, y `--solo-limites` se niega con un supuesto del modelo en null. Afecta: nada calculado (V1 no se ha ejecutado sobre ningún fichero de prueba, y `V1_limites_d00.csv` sale igual); cambia el procedimiento para fijar la verificación (`docs/decisiones.md`).
- **Últimos arreglos de la regla de orden de V1** (W-01 a W-03 de la segunda revisión del código): los bloques de código de `docs/decisiones.md` siguen las reglas de las vallas de CommonMark, así que una fila dentro de un ejemplo no cuenta aunque el ejemplo use una valla de cuatro comillas o tenga dentro una línea ```` ```python ````; la fila tiene que estar en `docs/decisiones.md`, salvo que se pida otro fichero con `--permitir-otro-registro`, y la tabla y el `.provenance.json` dicen qué fichero fue (`decision_file`, `extra.registro.fichero`); y `provenance.committed_status` se niega con bytecode que Python ejecutaría en lugar del fuente confirmado (un `.pyc` "unchecked-hash" en `__pycache__` o un `.pyc` sin su `.py`). Afecta: nada calculado; cambia el procedimiento para fijar la verificación (`docs/decisiones.md`).
- **`provenance`** (V-13, V-14): nuevos `committed_status`, `GitCheckError` y `run_context`, que calcula git y la máquina una vez por ejecución; `write_provenance` y `build_provenance` aceptan `git=` y `machine=` como dict. Las órdenes de git no heredan variables `GIT_*` y toman las rutas tal cual (`--literal-pathspecs`). La carpeta personal se oculta también con otra capitalización o en la forma de Git Bash. A0, A1 y V1 calculan el contexto una vez. Afecta al contenido de los `.provenance.json`, que en una misma ejecución describen ahora todos el mismo estado del repositorio. Los CSV de A0, A1 y V1 `--solo-limites` salen idénticos.
- **`config.load_baseline` avisa** (V-15) si el `configs/baseline.yaml` del directorio de trabajo no es el que viaja con el paquete, salvo con una ruta explícita o `TFMFDD_CONFIG`. Afecta: nada; antes una copia vieja ganaba sin avisar.
- `configs/yin2012.yaml` y el docstring de V1 remiten a `docs/verificacion_yin.md` (solo cambian comentarios: la huella no cambia), y `results/summary/README.md` ya no cita el CSV B1.

#### Coste y robustez

- **`measure_cost(..., infer_memory="same_process" | "subprocess")`** (V-06), con la columna nueva `infer_memory_where`. La memoria de la inferencia se mide justo después del ajuste: en el mismo proceso solo es exacta si supera la del ajuste, aunque el proceso sea nuevo. Con `"subprocess"` se mide en un proceso aparte, con el modelo serializado. Los avisos dicen ahora qué lo arregla y qué no. Afecta: el valor por defecto conserva el comportamiento, la fila tiene una columna más y cambian los textos de los avisos (la salida guardada de la celda 19 del B00 muestra el antiguo hasta que se vuelva a ejecutar).
- Sin psutil, `measure_cost` falla antes de ajustar y su mensaje ya no ofrece un parámetro que no tiene (V-17). En `robustness`, sin la rama muerta (M9) y sin resultados de una ejecución en los docstrings (M10). Afecta: nada.

#### Estadística

- **p exacto según las diferencias no nulas** (I1): la rama exacta se elige por el número de diferencias no nulas (hasta 500) y por el tamaño del cálculo, no por el número de pares. Antes, con más de 500 pares y pocas diferencias no nulas, el p salía de una aproximación normal que no valía y podía cambiar el desenlace. Afecta solo a contrastes con más de 500 pares; con 500 o menos, el p era y sigue siendo exacto.
- **`unit=` obligatorio con `replicate_col`** (I4), en `compare_methods` y en `compare_arms`: `"replicate"` (la media de las simulaciones dentro de cada réplica) o `"simulation"` (la media de las réplicas dentro de cada simulación de prueba); sin él, error. Con `unit="replicate"` sale lo mismo que antes. Cuál se usa es una decisión pendiente (regla 8 de `docs/reglas_de_medida.md`).
- **`hypothesis_verdict`**:
  - "apoya" exige la familia declarada (los fallos 3, 9 y 15, un método frente a dos baselines: seis comparaciones) y fallos de control; si no, el veredicto es "inconclusa" y la explicación dice por qué (M3; clave nueva `declared_family`). "Refuta" sigue siendo posible, con un aviso;
  - una comparación sin ventaja (`advantage=None`) es "no contrastable" y entra en Holm con p = 1; el desenlace de los controles va en la columna nueva `control_outcome`, y `outcome` queda vacío en ellos (M4);
  - el campo `published_condition` vale siempre "no evaluada": el veredicto no evalúa la condición de la ventaja publicada (párrafo 349); con el parámetro opcional `published=` muestra los resultados de `published_advantage_test` sin combinarlos con la regla (I2).
  - Afecta: un "apoya" con otra familia o sin controles pasa a "inconclusa".
- **Holm**: `significant` sale del procedimiento por pasos (M5); antes, `p_adjusted <= alpha` podía fallar por coma flotante con un p justo en un umbral α/k (con los p exactos de Wilcoxon no se da). Con `family` declarada, las comparaciones no contrastables cuentan con p = 1 y `family_m` es el m de Holm (M1): sus p ajustados suben. Y avisa de las simulaciones sin pareja (M2).
- Los infinitos son un error claro (M7) y las escalas diminutas ya no dan NaN (M8). Los docstrings dicen dónde se decide que el Wilcoxon supone una ventaja simétrica alrededor de su pseudomediana y qué análisis de sensibilidad se pueden declarar (I3). `non_superiority_test` y `published_advantage_test` comparten el núcleo, con la misma API. Afecta: nada que antes no fallara.

#### Ablaciones

- **`block_bootstrap_ci` usa bloques circulares** (I6): el centro del bootstrap es exactamente la media de la serie, y con bloques móviles se desplazaba en las series con un retardo. `interval_coverage` admite una referencia por intervalo, y las docstrings dicen qué mezcla su falta de cobertura: el error del propio intervalo, la no estacionariedad y lo que varía entre simulaciones. Afecta a los intervalos de `block_bootstrap_ci` (con L = 1, a nada); ningún resultado versionado los usa.

#### Cuadernos

- Leen el yaml con `tfmfdd.config`, usan sus claves nuevas con valor, citan las reglas con la numeración del documento y sus búsquedas de guía llevan `error_score="raise"` (V-09); el detalle, en la sección "Cuadernos" de abajo. Afecta: nada; las cifras del 01, el 02, el 03 y el B00 salen iguales que en su ejecución anterior (comparadas celda a celda en el informe de los cuadernos). Se volvieron a ejecutar en el repositorio el 12/10, con el paquete final y sin errores.

#### Después de la revisión independiente (12/10, por la tarde)

- **`cost` en Linux:** el pico de memoria sale de `VmHWM` (`/proc/self/status`) y no de `ru_maxrss`, que se hereda a través de exec: un proceso lanzado desde otro arrancaba con el pico del padre, y con `infer_memory="subprocess"` la inferencia nunca salía exacta (la prueba `test_en_un_proceso_aparte_la_inferencia_sale_exacta` fallaba en Linux). Comprobado en WSL2: un hijo de un padre con 161 MiB de pico da `ru_maxrss` 162 MiB y `VmHWM` 10 MiB. Afecta: solo a Linux (WSL2, Colab); en Windows no cambia nada. macOS sigue sin comprobar.
- **`pyreadr` pasa al extra `[rieth]`:** deja de ser dependencia obligatoria; solo lo necesita `convert_rieth`, cuyo error dice ahora cómo instalarlo. Afecta: nada calculado; una dependencia menos para quien usa los Parquet ya convertidos.
- Erratas de este registro: el desglose de las pruebas no contaba las 32 de `test_v03_correcciones_v1_final.py`, y el recuento de "Pruebas" era solo de Windows.

### Por módulo

#### `tfmfdd.data`

- `FAULT_ONSET_TRAIN_CLASSIC` retirada (B2, D1): pedirla da `AttributeError` (y un `FutureWarning`, para que el mensaje no se pierda en un `from ... import`) con el motivo y qué usar: `data.fault_onset("train", "classic")`, que devuelve `None`, y las 480 filas enteras. `fault_onset` valida sus argumentos (`"Rieth"` con mayúscula daba `None` sin avisar).
- `load_rieth` (DAT-4 a DAT-7, M7): ids en `int64`; variables en `float32`, como están en los Parquet; `runs` admite un `int` de Python (las n primeras) o cualquier iterable de ids, y un entero de numpy suelto es un error que dice qué pasar; ids inexistentes, `runs` fuera de rango o vacío y `split` mal escrito son errores; ids repetidos avisan. Lee columna a columna: lo transitorio es una columna, no el fichero. Carpeta: `root`, luego la variable de entorno `TFM_RIETH_DIR`, luego `data/tep_rieth`. Mensajes que dicen qué hacer si falta la carpeta o un fichero, si pyarrow no puede leer Parquet (el bloqueo de Windows, `docs/entorno.md`) o si un fichero está truncado.
- `load_classic`: ids en `int64` de forma explícita.
- `split_runs(..., exclude=)`: saca ids del reparto. Sin `exclude` y con las tres fracciones positivas, el reparto es idéntico al de la v0.2; una parte con fracción 0 queda vacía.
- Nuevo: `ID_COLUMNS`; `values` se niega a devolver columnas de identificación. Registro de simulaciones de Rieth ya miradas (`SeenRuns`, `RIETH_SEEN_RUNS`, `RIETH_SEEN_PENDING`, `seen_runs`, `unseen_runs`). `effective_onset`: hasta qué muestra una simulación con fallo sigue siendo copia de su gemela normal (exige las muestras posteriores al inicio desde `onset + 1` y seguidas).
- Resultados afectados: los puntos 7 y 8 del resumen. Con ids que existen, `load_rieth` devuelve las mismas filas y valores que antes, salvo el tipo de los ids.

#### `tfmfdd.preprocessing`

- `Scaler` hereda de `TransformerMixin` y `BaseEstimator` de scikit-learn: entra en una `Pipeline` y en `GridSearchCV` (M6). `normal_label=0` por defecto: con etiquetas, la media y la desviación salen solo de la clase normal; `normal_label=None` usa todas las clases y hay que pedirlo. Sin etiquetas, `Scaler().fit(X)` hace lo de siempre. Con menos de 2 filas o valores no finitos, error (antes, NaN).
- `lag_by_run` corta por trayectoria contigua y conserva el orden de entrada (MET-2); `runs` puede tener dos columnas (fallo y simulación); `return_index=True` da la posición de cada fila.
- `trajectory_starts` (nueva, pública): la definición de trayectoria que comparten `lag_by_run`, el DPCA y `tfmfdd.ablation`. `_trajectory_starts` queda como alias. Avisa si una trayectoria pasa de `TEP_MAX_RUN_LENGTH` (960) filas.
- `lag_by_run(..., samples=)`: con la columna `sample`, comprueba que cada fila junta muestras consecutivas.
- `import tfmfdd` carga ahora `sklearn.base`.
- Resultados afectados: el punto 4 del resumen; y cualquier matriz con retardos construida con `lag_by_run` sobre varias clases con ids compartidos (antes reordenaba y mezclaba trayectorias).

#### `tfmfdd.limits`

- Nuevos: `t2_limit_beta` (límite de Fase I del T²) y `limit_empirical(..., method="conformal")` (el estadístico de orden de Diallo et al., 2025). `method="linear"`, el de siempre, es el valor por defecto.
- Todas las funciones: `alpha` fuera de (0, 1) es error; los NaN del estadístico no entran (muestra no evaluada); un infinito, un conjunto vacío o un límite no finito es error. `spe_limit_jackson` con h0 ≤ 0 es error (MET-7); `limit_kde` pide al menos 20 valores.
- Resultados afectados: ninguno con los valores por defecto (los CSV de A0 y A1 salían iguales, byte a byte, antes y después de este cambio). Lo que antes daba un límite NaN ahora falla.

#### `tfmfdd.pca`

- `PCAMonitor(variance=0.85, ..., t2_method="f", quantile_method="linear")`: `t2_method` admite `"f"` (Fase II, el de siempre), `"beta"`, `"empirical"` y `"kde"` (I4, DIS-5).
- `fit(X, runs=None, X_cal=None, runs_cal=None, *, calibration=None, scaler=None, samples=None, samples_cal=None)`:
  - sin `X_cal`, si algún límite se calibra con datos, calibra en muestra **y avisa**; `calibration="in_sample"` lo hace a propósito y sin aviso (la verificación frente a Yin); `calibration="holdout"` exige `X_cal`;
  - un `X_cal` "reservado" que comparte filas con `X` es un error, y uno "en muestra" con filas que no están en `X`, también; `X_cal` con 0 filas es un error (M11);
  - `scaler=`: un escalador ya ajustado sustituye al del ajuste (el brazo S1 del factor 2; antes el factor era nulo por construcción, DIS-7). Se guarda una copia, y `scaler_n_samples_fit_` dice con cuántas filas se ajustó;
  - `samples=` y `samples_cal=`: la columna `sample`, para comprobar que los retardos juntan muestras consecutivas.
- Atributos nuevos: `limit_source_` (`"parametric_fit"`, `"holdout"` o `"in_sample"` por estadístico), `n_fit_`, `n_cal_`, `scaler_`, `scaler_source_`. `calibrated_on_holdout_` significa "algún límite calibrado con `X_cal`": con F y Jackson ya no dice `True` (M2).
- `score(X, runs=None, *, scaler=None, samples=None)`: una fila por fila de entrada, con NaN en las `lags` primeras de cada trayectoria (M4, MET-1); sin `runs` y con `lags > 0`, `X` es una sola trayectoria y avisa; `scaler=` puntúa con otro escalador sin tocar el modelo (el brazo S2).
- `combined_index(t2, spe)`: la regla "T² o SPE" de Yin como un índice con límite 1.
- Docstrings: calibrar en muestra es un **sesgo**, no una fuga (memoria, párrafos 283 y 307), y el signo de su efecto no se presupone.
- Resultados afectados: los puntos 2, 5 y 10 del resumen. Con PCA y los valores por defecto, nada (A0 y A1 dieron los mismos CSV).

#### `tfmfdd.metrics`

- Muestras no evaluadas (NaN): no cuentan en la FAR ni en la FDR; si no queda ninguna, la tasa es NaN. `alarms` con un límite NaN o infinito es error (MET-11).
- `evaluate_run`: seis claves nuevas (`n_far`, `n_fdr`, `n_fdr_h1`, `n_fdr_h2`, `false_alarm_samples_per_hour`, `false_alarm_episodes_per_hour`), las mismas en el mismo orden con y sin `onset`; `delay_convention` (`"appearance"`, la de siempre, o `"confirmation"`); `far_window` declara la ventana de la FAR en una simulación normal. Valida `onset` y `k`.
- `aggregate` (I1, MET-4): el retardo, solo con detecciones válidas; `valid_detection_ratio`; `<métrica>_n`; medianas del retardo, condicionada y censurada; intervalos recortados al rango de cada métrica; un NaN en la clave de agrupación es error.
- `summary`: `far_normal` y `far_prefallo_medio` sustituyen a `far_medio`; `fallos_sin_racha` y `fallos_bajo_umbral`, a `fallos_nunca_detectados`; `retardo_medio_muestras` sin rachas al azar y `retardo_mediano_muestras`, con su n; falla con más de una fila por fallo.
- Nuevas: `false_alarm_samples_per_hour`, `false_alarm_episodes_per_hour`, `pr_auc`, `diagnosis_metrics` (F1 por clase sobre todo el conjunto y macro-F1, con `include_normal` obligatorio; B3), `diagnosis_recall_by_run` y `macro_f1_by_replicate`. `detects_above_far` es la misma función que `stats.detects_above_far`.
- El docstring de `evaluate_run` ya no cita una cifra escrita a mano que además estaba mal (H8).
- Resultados afectados: los puntos 1, 6 y 9 del resumen; los intervalos que se salían de [0, 1]; y cualquier F1 por clase calculada solo con las muestras de esa clase.

#### `tfmfdd.stats`

- `paired_test`: p exacto hasta 500 pares (programación dinámica) y aproximación normal con corrección de continuidad por encima; las diferencias se redondean a 10 cifras significativas antes de asignar signos y rangos; `effect_size` es la biserial de rangos, con signo y entre −1 y 1 (M8, MET-9); con todas las diferencias nulas, p = 1 y efecto 0 sin la tolerancia escondida de `np.allclose`. Claves nuevas: `hl_estimate`, `ci_low`, `ci_high`, `ci_level`, `n_nonzero`, `p_method`, `alternative`.
- `holm_correction`: columna `p_adjusted`; los NaN, fuera de la familia (MET-8); `rank` pasa a `Int64`.
- `compare_methods(..., baseline=, alpha=, alternative=, replicate_col=, family=, unit=)`: con `baseline`, k − 1 comparaciones por fallo (I2); con `replicate_col`, las celdas de las réplicas de ajuste, que exigen declarar la unidad (`unit`); con `family`, Holm sobre la familia declarada.
- Nuevas: `hodges_lehmann`, `min_runs_for_significance`, `paired_advantage`, `non_superiority_test` y `hypothesis_verdict` (la regla del párrafo 368), `detects_above_far` (el suelo de detección, con el mismo Wilcoxon exacto), `effect_test` y `compare_arms` (efecto frente a un margen en las ablaciones), `published_advantage_test` (la ventaja publicada), `pareto_front` y `best_compromise` (el mecanismo del mejor compromiso, sin criterio por defecto), y el aviso `LowPowerWarning`.
- Un `alpha` de los contrastes de 0,5 o más es un error que nombra la confusión con el 0,99 del límite. Ninguna función tiene δ por defecto.
- Resultados afectados: el punto 3 del resumen, en todas las funciones que contrastan. Las llamadas de los cuadernos B00 y C00 siguen funcionando y encuentran las columnas que leen.

#### `tfmfdd.ablation` (nuevo)

Los mecanismos de las cuatro ablaciones, iguales para las tres familias: `RowSplit`, `split_by_run` (brazo limpio del factor 3), `split_by_sample` y `block_ids` (brazo sucio, por muestra o por bloques), `standardize_by_run` y `score_standardized_by_run` (brazo S2 del factor 2), `block_bootstrap_ci` (bootstrap de bloques circulares), `interval_coverage`, `contrary_conclusion_rate` y `subsample_runs` (factor 4), y `window_ends`, `stack_windows` y `align_to_samples` (ventanas por trayectoria). No ejecuta ninguna ablación ni fija ids: fracciones, bloques, semillas y estimandas llegan como parámetros sin valor por defecto. Resultados afectados: ninguno (es nuevo).

#### `tfmfdd.cost` y `tfmfdd.robustness` (nuevos)

- `cost`: `measure_cost` (el procedimiento común: tiempo de ajuste, tiempo de inferencia por muestra y pico de memoria del proceso), `measure`, `time_fit`, `time_inference` y `machine_info` (la máquina y las versiones, sin importar torch). La memoria de la inferencia, en el mismo proceso o en uno aparte (`infer_memory`). Necesita psutil.
- `robustness`: el índice de Bernal-de Lázaro et al. (2018) (`robustness_curve`, `robustness_index`), el ruido del lema 1 (`add_noise`, `noisy_versions`), `severity_level` y `error_rate`, con la métrica de la adaptación obligatoria. Reproduce las Tablas 1 a 3 del artículo (prueba).
- Resultados afectados: ninguno. Los cronómetros a mano de B00 y C00 pasan a `cost.measure`.

#### `tfmfdd.config` y `tfmfdd.provenance` (nuevos)

- `config`: dónde está `baseline.yaml` y cómo se lee (`load_baseline`: ruta explícita, `TFMFDD_CONFIG`, `configs/baseline.yaml` del directorio de trabajo y la copia empaquetada `tfmfdd/configs/baseline.yaml`), `get` sin valor por defecto, `require` (null = decisión pendiente) y `pending` (REP-4). Avisa si el yaml del directorio de trabajo no es el del paquete.
- `provenance`: el `.provenance.json` de cada resultado (configuración, ficheros con su SHA-256, ids, commit de git leído sin escribir en el repositorio, máquina y versiones); `run_context` (git y máquina una vez por ejecución) y `committed_status` (si unas rutas están confirmadas en git, que usa la regla de orden de V1).
- Resultados afectados: ninguno.

#### `tfmfdd.plots` y `tfmfdd/__init__.py`

- `plots.fdr_by_fault`: el docstring ya no dice que 3, 9 y 15 sean "la única diferencia real" (presuponía lo contrario de lo que predice la hipótesis; M12).
- `__init__.py`: versión 0.3.0; exporta `ablation`, `config`, `cost`, `provenance` y `robustness` (así, `import tfmfdd` carga también PyYAML, por `config`); el ejemplo calibra con `X_cal` y lee los valores del yaml (antes calibraba en muestra sin decirlo).

### Experimentos y resultados versionados

- `experiments/A1_pca_baseline.py`: ya no pide `fallos_nunca_detectados`; el DPCA pasa `runs`; columna `method` y la configuración en cada fila; `.provenance.json` de cada CSV; opciones `--t2-method`, `--spe-method jackson`, `--quantile-method`, `--calibration holdout|in_sample` y `--config`; los valores por defecto, del yaml (`tfmfdd.config`). El docstring ya no dice que A1 verifique el protocolo (M7): la verificación frente a Yin es V1.
- `experiments/A0_calibracion_limites.py`: el brazo en muestra, con `calibration="in_sample"`; brazos nuevos del SPE (`en_muestra_mismo_ncal`, `jackson`) y del T² (`A0_calibracion_limites_t2*.csv`: `f`, `beta`, empírico reservado y en muestra); columnas `n_cal_used`, `limit_source`, `far_window_*`. Ya no se presenta como una fila de la tabla de ablaciones (M6): es desarrollo sobre el clásico.
- `experiments/V1_verificacion_yin.py` (nuevo): la verificación de la implementación frente a Yin et al. (2012), con la regla de orden (`docs/verificacion_yin.md`). `--solo-limites` y `--huella` no abren ningún fichero de prueba.
- `results/summary/`: A0 y A1 regenerados con su configuración de siempre. Las columnas antiguas salen iguales salvo dos cosas: `delay_samples_mean` y `delay_minutes_mean` de `A1_pca_baseline_T2.csv` y `A1_pca_baseline_SPE.csv` quedan vacías en los fallos cuya única racha es al azar (punto 1 del resumen), y el `spe_limit` de `en_muestra` de `A0_calibracion_limites_yin33.csv` cambia en la última cifra escrita (el entorno numérico; el paquete sin cambios da lo mismo en el entorno nuevo). Nuevo: `V1_limites_d00.csv`. Se retira `B1_clasificadores_clasico.csv` (punto 7).

### Configuración

- `configs/baseline.yaml`: las 17 claves de la v0.2 conservan su valor. 69 claves nuevas: 63 en null, que son decisiones pendientes (los scripts que las necesitan se niegan a ejecutar), y 6 con valor, cada una con su fuente en el comentario: `contrastes.alpha_contrastes` (0,05), `pca.t2_method` (f), `pca.quantile_method` (linear), `pca.calibration` (holdout), `protocolo.min_fdr` (0,10) y `protocolo.retardo_convencion` (appearance). Quedan por confirmar (`docs/decisiones.md`, última sección).
- `tfmfdd/configs/baseline.yaml`: la copia que viaja con el paquete, idéntica (una prueba lo exige).
- `configs/yin2012.yaml` y `configs/referencias/` (nuevos): la configuración de Yin, con los supuestos S1 a S12 y 11 claves pendientes, y la transcripción de sus Tablas 4, 5, 7 y 8 y de los umbrales que dibujan, pendiente de que Pablo la verifique contra el PDF.

### Cuadernos

Resumen del informe de la revisión de los cuadernos (12/10):

- 01, 02, 03 y B00 actualizados a la API de la v0.3 y ejecutados el 12/10; el C00, actualizado sin ejecutar (PyTorch no carga en el PC de Pablo), con un aviso al principio.
- El 02 ya no usa `FAULT_ONSET_TRAIN_CLASSIC` y ya no escribe el CSV B1; su diagnóstico usa `diagnosis_metrics`.
- El escalado del diagnóstico, solo con la clase normal (B00, C00, 02). La macro-F1 sale en las dos variantes (`include_normal` verdadero y falso) mientras no se decida.
- Retardos y resúmenes con `aggregate` y `summary`; contrastes con `baseline=` y Hodges-Lehmann; cronómetros con `cost.measure` y una celda de coste con valores de demostración.
- Las reglas de `docs/reglas_de_medida.md` en el B00 y el C00, con su numeración y sus títulos, de la 1 a la 14 (la 12, no reportar `best_score_`).
- `notebooks/README.md` reescrito.
- En la ronda de correcciones del 12/10, los cinco cuadernos leen el yaml con `tfmfdd.config.load_baseline()` y `config.require`, que se niega con una clave en null. Usan las claves nuevas con valor: el α de los contrastes (antes declarado a mano), y `min_fdr`, la convención del retardo y los métodos de los límites, que pasan a cada llamada con los diccionarios `DETECCION` y `METODOS_PCA` de la primera celda.
- Lo que depende de una clave en null (`INCLUIR_NORMAL`, `COSTE_DEMO`) va en la primera celda como parámetro, y manda el yaml en cuanto la clave tenga valor. `VENTANAS_PRUEBA` (C00) sigue declarado: no tiene clave en el yaml.
- Las búsquedas de guía del B00 y del C00 llevan `error_score="raise"`, con el aviso de `normal_label` si se recodifican las clases.
- 01, 02, 03 y B00 se volvieron a ejecutar tras esas correcciones; el C00 solo se editó. Hay 41 pruebas nuevas en `tests/test_v03_correcciones_cuadernos.py`: leen los cuadernos como JSON, sin ejecutarlos, y comprueban, entre otras cosas, que las reglas que citan existen y tienen los títulos de `docs/reglas_de_medida.md`.
- Resultados afectados: las cifras de kNN y SVM del B00 cambian por el escalado (punto 4); el 03 toma la semilla del yaml (42), así que sus cifras cambian respecto a las del 18/09; los p de B00 y C00 son exactos (punto 3).

### Documentos

- `README.md`: título aprobado, instalación por etiqueta, entorno de Windows, datos, ejemplo con su salida real, pruebas y flujo de la verificación. Sin la cifra anulada de la calibración (`docs/decisiones.md`, 16/09) ni la "verificación" con los fallos 3, 9 y 15.
- `docs/reglas_de_medida.md`: al día con la v0.3; reglas nuevas 12 (selección sin mirar la prueba), 13 (verificación frente a Yin y prerregistro) y 14 (coste y robustez).
- `docs/decisiones.md`: anotadas como superadas tres filas, y anotadas las decisiones 15 y 16 del 04/10; cómo se registra la fila de la huella de V1 y dónde están las decisiones pendientes.
- `docs/entorno.md` y `docs/verificacion_yin.md` (nuevos).

### Empaquetado

- `pyproject.toml`: versión 0.3.0; `license = "MIT"` con `license-files = ["LICENSE"]` (la tabla `{text = "MIT"}` está en desuso y setuptools dejará de construirla a partir del 18/02/2027), con `setuptools>=77`; así un `LICENSE.zip` en la raíz ya no entra en la rueda. `psutil>=5.9` en las dependencias (lo usa `cost`). `package-data` para `tfmfdd/configs/*.yaml`.
- `scikit-learn>=1.6` (antes 1.3): una prueba y la receta del OE2 de `ablation.block_ids` usan `GroupKFold(shuffle=True)`, que existe desde la 1.6. Los demás mínimos se mantienen: nada de lo que usa el código pide más.
- `requirements-probado.txt` (nuevo): las versiones exactas con que pasaron las pruebas, en el Python global de Pablo. El `.venv` del repositorio tiene otras (`docs/entorno.md`).

### Pruebas

Las 31 de la v0.2 y 612 nuevas, en `tests/test_v03_*.py`: datos (70), monitor (38), métricas (52), contrastes (41), coste y robustez (38), experimentos (61), ablaciones (40) y estadística (56), y las de la ronda de correcciones, en `tests/test_v03_correcciones_*.py`: monitor (19), datos (28), experimentos (43), coste y robustez (10), estadística (35), ablaciones (8), cuadernos (41) y los últimos arreglos de V1 (32). De las antiguas se cambiaron cuatro, porque probaban lo que cambió: dos declaran ahora la calibración en muestra que hacían sin decirlo (`.fit(X, calibration="in_sample")`), una añade la columna `detected_at_chance` a su tabla sintética, y la que fijaba `FAULT_ONSET_TRAIN_CLASSIC = 20` (tautológica) comprueba ahora que la constante falla y que `fault_onset("train", "classic")` es `None`. En la ronda de correcciones se ajustaron diez de las `test_v03_*` por la misma razón: una de datos (el entero de numpy ahora es un error), dos de experimentos (el aviso del yaml y los argumentos de git), una de contrastes y cuatro de estadística (el veredicto sin controles y `unit=`) y dos de ablaciones (la definición circular del bootstrap y los mensajes de `split_runs`).

Ejecutadas el 12/10/2026 con el Python global (`requirements-probado.txt`), `python -m pytest -q -p no:cacheprovider` desde la raíz, sobre una copia con todo unido (también la ronda de correcciones de los cuadernos):

- con `TFM_CLASSIC_DIR` y `TFM_RIETH_DIR` definidas: `643 passed`;
- sin ellas: `633 passed, 10 skipped` (las 10 que leen datos reales).

Esas cifras son de Windows. **Segundo entorno: Linux**, comprobado el 12/10 por una revisión independiente (la sesión de chat de Pablo, no reproducida aquí), con Python 3.13.16, numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, pyarrow 25.0.1, pytest 9.1.1 y psutil 7.2.2, sobre los ficheros del árbol de trabajo: `637 passed, 1 failed, 5 skipped` (4 sin los Parquet de Rieth y 1 porque su copia no era un repositorio git). El fallo era el pico de memoria de `cost` en Linux (ver "Después de la revisión independiente"); con el arreglo, `638 passed`. Esa revisión comprobó además que sus CSV de A0, A1 y V1 `--solo-limites` difieren de los versionados como mucho 5e-14 y 1,6e-12, que los límites de Yin calculados desde cero con numpy y scipy, sin `tfmfdd`, coinciden con `V1_limites_d00.csv`, que el DPCA pone el salto de un escalón sintético en el índice 160 con 0, 1 y 2 retardos, que V1 no abre ningún `_te.dat` sin argumentos, con `--huella` ni con `--solo-limites` (vigilado con un `sys.addaudithook` sobre "open"), y que el Wilcoxon exacto, Hodges-Lehmann y Holm coinciden con cálculos propios.

## 22 de septiembre de 2026 (antes de la v0.3)

Resultado de la revisión celda por celda del cuaderno 01 y del paquete. Todo lo que
sigue está explicado, con su motivo, en la última sección del cuaderno 01
(«Registro de cambios») y en `docs/decisiones.md`. Este fichero dice qué tocar y en
qué orden.

### Cómo aplicar

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
   *(Superado: el A0 del TEP clásico es desarrollo, no resultado; las ablaciones van
   sobre Rieth, memoria, párrafo 367.)*
7. Commit y etiqueta nueva (`v0.3`): los compañeros instalan por etiqueta, y las
   funciones de métricas cambiaron de salida.

### Qué cambió y por qué

#### Paquete `tfmfdd`

- `data.py`: `values(df, columns=None)` permite elegir un subconjunto de variables por
  nombre; `YIN_VARS` son las 33 de Yin et al. (2012), necesarias para que la
  verificación contra sus tablas sea comparable; `FAULT_ONSET_TRAIN_CLASSIC = 20`
  (Russell, Chiang y Braatz, 2000: el fallo entra tras 1 hora en los ficheros de
  entrenamiento; pendiente de contrastar con Chiang et al., 2001). Se corrigió la
  docstring de la transposición: el error no es silencioso, aparece en `score`.
  *(Superado en la v0.3: `FAULT_ONSET_TRAIN_CLASSIC = 20` era falso y se retiró; ver
  arriba.)*
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

#### Experimentos

- `experiments/A0_calibracion_limites.py` (nuevo): el experimento de calibración
  **sin confundido**. Los dos modelos se ajustan con `X_fit`; solo cambia de dónde
  sale el límite. Escribe la configuración completa en el CSV. Admite `--yin-vars`.
- `experiments/A1_pca_baseline.py`: lee los valores por defecto de
  `configs/baseline.yaml` (antes usaba 0,90 de varianza mientras el yaml decía
  0,85). Un argumento de línea de comandos sigue sobrescribiendo para sensibilidad.

#### Cuadernos

- `01_primeros_pasos.ipynb`: corrección de la sección 7 (confundido), figura de
  sedimentación con Kaiser y corte al 85 %, lectura de la tabla de 21 fallos en
  cuatro patrones, caso de estudio del fallo 5 (carta, tasas por bloque, FDR
  estratificada, contraste exploratorio con su advertencia), vocabulario de
  excedencia / alarma confirmada / fallo, nota sobre contribuciones (solo SPE,
  aislamiento ≠ diagnóstico), tabla de funciones actualizada y registro de cambios.
- `02_arranque_diagnostico.ipynb`: descarta las 20 primeras muestras de cada
  `dXX.dat` al etiquetar *(superado en la v0.3: no hay que descartarlas; el fallo
  está activo desde la primera fila)* y explica por qué la validación es por simulación y no por
  muestra. Salidas borradas: hay que regenerarlas. El CSV que escribe es una prueba
  de humo de este repositorio, no un resultado del bloque B.
- `03_arranque_deeplearning.ipynb`: solo texto: la sección 5 ya tenía el diseño
  correcto, y se añade el precedente de Lyu et al. (2026) y las columnas nuevas.

#### Documentos

- `docs/decisiones.md`: ocho filas nuevas con fecha 2026-09-22.
- `results/summary/README.md`: descripción de las columnas nuevas.
- `README.md` y cuadernos: español neutro, con ustedes (había formas de vosotros).
- `requirements.txt` y `pyproject.toml`: `pyyaml`.

### Pendiente que no se puede hacer sin los datos

- Ejecutar los cuadernos y los dos scripts y **anotar las cifras** con su
  configuración. La cifra de la cuartilla sale de A0. *(Superado: ver la nota del
  paso 6 de "Cómo aplicar".)*
- Transcribir la tabla de referencia de Yin et al. (2012) y correr la verificación en
  modo Yin (`--yin-vars`, 17 componentes, límite de Jackson, FAR sobre las muestras
  161–960 de `d00_te`): es el 5.1.1 de la primera entrega y hoy no existe.
  *(Superado en la v0.3: la configuración de Yin es la de `configs/yin2012.yaml`, con
  33 variables, 9 componentes en el PCA y 17 en el DPCA y la regla T² o SPE, y la
  verificación es `experiments/V1_verificacion_yin.py`; ver `docs/verificacion_yin.md`.)*
- Validar la brecha de persistencia: debe salir grande en los fallos que la
  literatura describe como compensados por el control y cerca de cero en los
  escalones que persisten (1, 2, 6, 7). Si sale desordenada, la métrica no mide lo que
  dice y hay que decirlo.
- Acordar con el equipo la diferencia mínima relevante para los contrastes (propuesta:
  2 puntos porcentuales en FDR/FAR, 1 muestra en retardo) y dejarla en
  `docs/decisiones.md`.
