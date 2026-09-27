# Reglas de medida del protocolo común

Versión del 27 de septiembre de 2026, sobre la rama `revision-2026-09` (paquete `tfmfdd` sin etiqueta; la próxima será `v0.3`). Salen de la sección 8 del traspaso v3, de `configs/baseline.yaml`, de `docs/decisiones.md`, de la revisión del 27/09 (`docs/revision_2026-09-27.md`) y de las dos verificaciones que se hicieron ese mismo día sobre esa revisión. Los identificadores entre paréntesis (B2, I1, DAT-1…) remiten a esos informes.

Las tres partes del TFM se van a comparar entre sí, y eso solo vale si las cifras se miden igual. Cada regla dice por qué existe, qué función del paquete la implementa y qué defecto conocido evita. Cuando el paquete todavía hace algo mal, se dice cómo esquivarlo hasta la `v0.3`. Los cuadernos `notebooks/B00_punto_de_partida_diagnostico.ipynb` y `notebooks/C00_punto_de_partida_deeplearning.ipynb` aplican estas reglas y comprueban con datos las que se pueden comprobar.

Una advertencia sobre los nombres: en este documento, «nivel de confianza» es `protocolo.alpha` del yaml (0,99, el del límite de control) y «α de los contrastes» es el 0,05 de las pruebas estadísticas. Son cosas distintas y en la memoria se nombran distinto.

## 1. Partición por simulación, nunca por muestra

**Por qué.** Las muestras consecutivas de una simulación se parecen mucho: el proceso tiene inercia y se muestrea cada 3 minutos. Si una muestra va a prueba y su vecina temporal a entrenamiento, el modelo acierta porque reconoce la simulación, no el fallo. En Rieth hay un segundo mecanismo: la simulación r de cada fichero de fallo comparte semilla con la simulación normal r del mismo split, así que son la misma trayectoria hasta que entra el fallo y, en los fallos 3, 9 y 15, siguen casi pegadas después (DAT-1, DIS-2). Con una partición por muestra, las dos cosas actúan a la vez y con signos contrarios, de modo que el error de partir por muestra no tiene un signo fijo: el B00 (secciones 7.1 y 7.2) lo mide con los dos diseños.

**Cómo.** `data.split_runs` reparte identificadores de simulación, una sola vez y con los mismos ids para todas las clases. Si usan validación cruzada, `GroupKFold` con `groups = simulationRun`, nunca con el par (fallo, simulación), que separaría a las gemelas. En el TEP clásico la regla se cumple sola, porque `dXX.dat` (entrenamiento) y `dXX_te.dat` (prueba) son simulaciones distintas. En Rieth, los ficheros `test_*` no comparten semillas con los `train_*`. Las ventanas temporales se cortan por trayectoria (el par fallo y simulación) y nunca cruzan de una a otra.

**Cuidados de Rieth.** Los ids llegan en `int16` y una cuenta como `simulationRun * 1000 + sample` se desborda: conviértanlos a `int64` al cargar (DAT-5). `load_rieth` ignora sin avisar los ids que no existen y falla con un `np.int64` suelto: pásenle listas de `int` de Python y comprueben `nunique()` (DAT-6). Como la simulación r de cada clase es gemela de la normal r, los pares (fallo, simulación) de fallos distintos con el mismo id no son independientes: el contraste por fallo no se ve afectado, pero cualquier agregado entre fallos sí.

**Evita.** El inflado por autocorrelación y el efecto de las gemelas.

## 2. Escalado ajustado solo con los datos de entrenamiento

**Por qué.** Si la media y la desviación incluyen la prueba, el modelo usa información del examen para prepararse. Además, un desplazamiento de media que es justo el fallo quedaría absorbido en parte por el escalado.

**Cómo.** `preprocessing.Scaler().fit(X_entrenamiento)` y después `transform` para todo lo demás. `PCAMonitor` lo hace por dentro con `X_fit`. En una búsqueda de hiperparámetros, el escalador va dentro de cada pliegue (una `Pipeline`), no antes.

**Matiz.** En el PCA del paquete, un escalado previo con fuga no cambia nada, porque `PCAMonitor` vuelve a escalar con `X_fit` y deshace cualquier transformación afín por variable (DIS-7). En kNN, SVM, Random Forest o redes sí cambia la cifra, y puede moverla en cualquier sentido. Queda por decidir, y anotar en `decisiones.md`, si en diagnóstico se escala con todas las clases o solo con la normal: las dos opciones están libres de fuga.

**Evita.** La fuga de datos por estandarización con la prueba.

## 3. Umbrales calibrados con datos normales reservados

**Por qué.** Un modelo minimiza su error sobre los datos con que se ajusta, así que ahí el estadístico sale artificialmente pequeño. Un umbral calibrado con esos datos queda bajo y dispara falsas alarmas en cuanto llegan datos nuevos. Pasa en el PCA (lo mide `experiments/A0_calibracion_limites.py`) y en el autoencoder (sección 5 del C00). Elegir el umbral maximizando el F1 sobre la prueba es fuga de datos en estado puro.

**Cómo.** `PCAMonitor.fit(X_fit, X_cal=X_cal)` para el PCA; `limits.limit_empirical` o `limits.limit_kde` sobre el estadístico de `X_cal` para cualquier otro detector. La fracción reservada es `particion.calibracion_limites` del yaml. En el clásico se corta en el tiempo (las últimas muestras de `d00.dat`); en Rieth se reservan simulaciones normales distintas. Si el modelo tiene parada temprana, esa decisión usa otro bloque de normales, distinto del de calibración y, por supuesto, de la prueba.

**Matices de la verificación del 27/09.**
- `PCAMonitor.fit` sin `X_cal` calibra en muestra sin avisar, y su `variance` por defecto no es la del yaml (REP-4). Pasen siempre `X_cal` y los valores del yaml, y comprueben `calibrated_on_holdout_`.
- El límite F del T² es de Fase II y no usa `X_cal`: con él, el efecto de calibrar en muestra es cero por construcción. El contraste que importa (percentil empírico en muestra o límite de Fase I) está pendiente (I4, DIS-5).
- El límite del SPE por Jackson se calcula con los autovalores de `X_fit`, así que en la práctica es una calibración en muestra (DIS-6). No lo presenten como «reservado».
- El percentil empírico con pocas muestras reservadas da, en promedio, algo más de FAR que la nominal (MET-3). Reporten siempre el tamaño de `X_cal` y la FAR nominal al lado.
- En el clásico, reservado no significa bien calibrado: con una sola simulación normal no se separa el sobreajuste de la deriva temporal (I5). Rieth, con simulaciones distintas, sí lo permite.

**Nunca mirando la prueba.** La regla también cubre lo que se decide alrededor del umbral. Los cortes y umbrales de las métricas (el 0,10 de `detected_at_chance`, las mitades de `stratified_fdr`, cualquier tramo) se declaran antes de ver los resultados (I8); un corte elegido a posteriori cambia la conclusión (I3). Y las simulaciones que ya se han mirado en desarrollo no sirven para el resultado confirmatorio (DIS-17). Hasta el 27/09 se han mirado estas de Rieth (análisis de datos, verificaciones y cuadernos de arranque):
- En los 42 ficheros (`train_fault00` a `train_fault20` y `test_fault00` a `test_fault20`), las simulaciones 1 a 20.
- En `train_fault00`, además, de la 21 a la 40.
- En los ficheros de entrenamiento, el diseño de ids distintos del B00 (sección 7.1): en `train_faultKK`, de la 20·KK + 1 a la 20·(KK + 1), es decir, hasta la 420 en `train_fault20`.
- Las que usa el C00, guardadas en `results/raw/C00_configuracion.json`: en los 21 ficheros `test_*`, además de las anteriores, la 23, 24, 32, 35, 36, 37, 42, 43, 44, 50, 58, 64 y 66; en los `train_*` con fallo, la 21, 25 y 26.
- Los prototipos del C00 tomaron prefijos de las mismas listas de `split_runs(500)` con la semilla del yaml; el tamaño exacto dependía de sus argumentos, que no quedaron registrados.

Los ids confirmatorios se fijan con un procedimiento escrito en `decisiones.md` antes de ejecutar (por ejemplo, `split_runs` sobre los ids no mirados), y son los mismos en los tres bloques.

**Evita.** La calibración en muestra, el umbral elegido con la prueba y los grados de libertad del investigador.

## 4. Detección con `metrics.evaluate_run`

**Por qué.** FAR, FDR y retardo son las tres cifras de la detección, y solo son comparables si se cortan igual.

**Cómo.** La FAR se calcula sobre `stat[:onset]` y la FDR sobre `stat[onset:]`, con `onset = data.FAULT_ONSET_TEST` (160) en los ficheros de prueba. El retardo es la primera alarma de la primera racha de `k` alarmas seguidas (`protocolo.k_alarmas`). Si `detected_at_chance` es `True` (FDR < 0,10), ese retardo no se interpreta: en 800 muestras autocorrelacionadas, una racha de `k` puede salir por azar. No usen `metrics.summary` para el retardo medio ni para `fallos_nunca_detectados` (I1): filtren a mano por `detected` y `~detected_at_chance`, como hacen los dos cuadernos.

**Matices.**
- La FDR se lee siempre al lado de la FAR. `detected_at_chance` solo mira si la FDR es baja, así que un detector que alarma casi siempre pasa el filtro con FDR alta y retardo cero (B00, sección 8). Una FDR del orden de la FAR no es detección (I6).
- En Rieth, la FAR se mide en las simulaciones normales de prueba (`test_fault00`). Las 160 primeras muestras de cada fichero con fallo son copia de su gemela normal: la columna `far` de esos ficheros repite la misma medida, y promediarla cuenta el mismo tramo normal una vez por fichero de fallo.
- `metrics.aggregate` calcula el retardo medio sobre las simulaciones con racha, incluidas las rachas al azar, sin decir cuántas son, y sus intervalos de métricas acotadas pueden salirse de [0, 1] (MET-4). Calculen el retardo a mano, como los cuadernos, y reporten el número de simulaciones que entra en cada media.
- En Rieth, el inicio efectivo del fallo puede llegar varias muestras después del nominal (regla 6): parte del retardo es física del proceso, no defecto del detector.
- Con DPCA (`pca.lags` > 0), `lag_matrix` desplaza el inicio `lags` muestras y favorece al DPCA; hasta que la `v0.3` lo corrija, usen `onset − lags` y `lag_by_run` con los ids de simulación (MET-1, MET-2).
- Un NaN en el estadístico cuenta como «sin alarma» (MET-11): comprueben que no hay NaN antes de medir.

**Evita.** Retardos por azar tomados como detección, la FAR contada varias veces y el sesgo a favor del DPCA.

## 5. Diagnóstico: F1 por clase sobre todo el conjunto de prueba, y macro-F1

**Por qué.** La precisión de una clase necesita los falsos positivos que llegan desde las demás clases. Si el F1 se calcula solo con las muestras cuya clase real es esa, no puede haber falsos positivos, la precisión vale 1 y lo que sale es 2r/(1+r), una función del recall que siempre es mayor o igual que el F1 de verdad (B3). El inflado se concentra en las clases con menos precisión, que son la normal y los fallos 3, 9 y 15.

**Cómo.** Hoy con scikit-learn, porque el paquete aún no tiene función propia: `precision_recall_fscore_support(y_true, y_pred, labels=clases)` sobre toda la prueba y `f1_score(..., average="macro")`. En Rieth, el recall está bien definido por (fallo, simulación), pero la precisión y el F1 por clase se calculan sobre el conjunto (*pool*) de simulaciones de prueba de todas las clases. El F1 de las clases 0, 3, 9 y 15 se lee al lado de lo que daría repartir al azar entre ellas (B00, sección 8).

**Evita.** El «f1» del cuaderno 02 anterior y del CSV `B1_clasificadores_clasico.csv`.

## 6. Dónde entra el fallo

**Por qué.** Etiquetar como fallo muestras sanas, o al revés, contamina las clases.

**Cómo.**
- TEP clásico, entrenamiento: los `dXX.dat` tienen 480 filas con el fallo activo desde la fila 0, y se usan enteros; `data.fault_onset("train", "classic")` devuelve `None`. No usen `data.FAULT_ONSET_TRAIN_CLASSIC`: vale 20 y es un dato falso (B2). `d00.dat` tiene 500 filas y viene transpuesto; el paquete lo corrige y avisa.
- TEP clásico, prueba: 960 muestras, con el fallo tras la 160.
- Rieth: el fallo entra tras la muestra 20 en entrenamiento y tras la 160 en prueba (`data.fault_onset(split, "rieth")`), comprobado con los datos. En los ficheros de fallo, las filas anteriores son copia exacta de la gemela normal y se descartan.

**Matiz (DAT-1).** El inicio nominal no siempre es el efectivo. En algunos fallos de Rieth la trayectoria sigue idéntica a su gemela normal varias muestras después del inicio, de modo que al descartar solo `sample <= onset` quedan unas pocas filas con etiqueta de fallo que son copia exacta de la clase 0. Es ruido de etiqueta menor. Los cuadernos lo miden (B00, secciones 7.2 y 8; C00, sección 6), y queda por decidir en `decisiones.md` si ese tramo se declara o se descarta, igual en los tres bloques.

**Evita.** Descartar muestras con fallo en el clásico y meter muestras normales con etiqueta de fallo en Rieth.

## 7. Los fallos 3, 9 y 15, siempre aparte

**Por qué.** Su huella en las 52 variables apenas se separa del ruido y, en Rieth, sus simulaciones siguen pegadas a su gemela normal. Promediarlos con el resto esconde el único sitio donde los métodos podrían diferenciarse, que es además donde la fuga más infla. Un F1 alto en 3, 9 y 15 es, antes que nada, sospecha de fuga.

**Cómo.** `metrics.HARD_FAULTS`, que coincide con `fallos_dificiles` del yaml. Cada tabla lleva dos bloques: el resto de fallos, y 3, 9 y 15.

**Evita.** Una media global que oculta dónde está la dificultad.

## 8. Contrastes pareados con Holm por fallo

**Por qué.** Con muchas simulaciones, casi cualquier diferencia sale significativa. Un p sin la diferencia al lado no dice si la ventaja importa.

**Cómo.** `stats.compare_methods`: Wilcoxon pareado por simulación con `zero_method="zsplit"` y corrección de Holm dentro de cada fallo. El α de los contrastes (0,05) se declara antes de ver resultados; hoy no está en el yaml (I2), así que los cuadernos lo declaran en la primera celda y comprueban que coincide con el de `stats.holm_correction`. Se reporta la diferencia (hoy, `median_difference`) antes que el p. Con una sola simulación (TEP clásico) no hay contraste posible, y los números son orientativos.

**Limitaciones de hoy.** `compare_methods` hace todos contra todos; el análisis principal contra el baseline, con el intervalo de la diferencia, llega en la `v0.3` (I2). `effect_size` pierde el signo (I2), `median_difference` no es un estimador de Hodges–Lehmann ni trae intervalo (DIS-10), y con todas las diferencias nulas el p vale 1 (MET-9). La diferencia mínima relevante está sin acordar (propuesta: 2 puntos de FDR o FAR y 1 muestra de retardo). Para emparejar métodos de bloques distintos, los tres tienen que usar los mismos ids de prueba.

**Evita.** Declarar «significativo» lo que no es relevante, y p que no son los del análisis declarado.

## 9. Semillas y configuración, desde `configs/baseline.yaml`

**Por qué.** Si un valor del protocolo está en dos sitios, acaba teniendo dos valores: el A1 usó 0,90 de varianza mientras el yaml decía 0,85 (`decisiones.md`, 22/09), y los cuadernos anteriores fijaban a mano semillas y fracciones que no coincidían (M3, M4).

**Cómo.** Se leen del yaml `protocolo.semilla`, `protocolo.alpha` (nivel de confianza), `protocolo.k_alarmas`, `protocolo.sample_minutes`, `particion.*`, `datos.runs_desarrollo`, `datos.runs_finales`, `pca.variance`, `pca.spe_method`, `pca.lags` y `fallos_dificiles`. Lo que no está en el yaml (el α de los contrastes o los hiperparámetros de cada modelo) se declara en la primera celda, antes de ver resultados. Con PyTorch en CPU, el resultado cambia con el número de hilos: regístrenlo junto a las versiones.

**Cuidado (REP-4).** El yaml no viaja con el paquete: quien instale `tfmfdd` por `pip` en su propio repositorio necesita una copia de `configs/baseline.yaml` de la misma versión. Los scripts A0 y A1 ignoran el yaml sin avisar si no lo encuentran, y `PCAMonitor` tiene sus propios valores por defecto.

**Evita.** Semillas y constantes distintas en cada cuaderno.

## 10. Ningún número de resultado escrito a mano en el texto

**Por qué.** Un número copiado a mano se desactualiza en cuanto cambia algo, y en este proyecto ya pasó: cifras de dos configuraciones distintas acabaron citadas como la misma (`decisiones.md`, 22/09), y una cifra del traspaso dependía de un corte que nadie había declarado (I3).

**Cómo.** Las interpretaciones remiten a la salida de la celda («fíjate en la salida anterior…»). Sí se escriben los hechos de los datos (52 variables, 160, 480, 500, 960), los valores de configuración y las citas de la literatura con su página. Toda tabla o figura de la memoria sale de código, con su configuración al lado.

**Evita.** Cifras desfasadas o sin procedencia.

## 11. Formato común de resultados

**Por qué.** Es lo que permite que `metrics.aggregate` y `stats.compare_methods` funcionen igual en los tres bloques.

**Cómo.** Una fila por (`method`, `faultNumber`, `simulationRun`). En detección, las columnas de `evaluate_run`. En diagnóstico, el recall por (fallo, simulación), y en un fichero aparte la precisión y el F1 por clase calculados sobre el conjunto de prueba. Cada fichero lleva, en columnas o en un JSON al lado, la configuración, los ids usados y las versiones de las librerías (I7). Mientras sea una prueba de humo, va a `results/raw/`, que git ignora; los resultados que se citan van a `results/summary/`.

**Evita.** Métricas que no se pueden agregar ni contrastar igual en los tres bloques, y cifras sin procedencia.

## Qué cambia con la `v0.3`

Varias de las vueltas de arriba desaparecen cuando la `v0.3` corrija el paquete: quitar `FAULT_ONSET_TRAIN_CLASSIC` y probarlo con datos (B2), una función de diagnóstico con F1 por clase (B3), `summary` sin los retardos al azar (I1), `compare_methods` con `baseline=`, intervalo de la diferencia, `r` con signo y el α de los contrastes en el yaml (I2), el inicio corregido del DPCA (MET-1, MET-2), `load_rieth` con ids en `int64`, aviso de ids inexistentes y lectura filtrada (DAT-4 a DAT-6) y un yaml que no se pierda por el camino (REP-4). Hasta entonces, las reglas se cumplen como hacen los cuadernos B00 y C00.
