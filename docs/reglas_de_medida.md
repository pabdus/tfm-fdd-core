# Reglas de medida del protocolo común

Versión del 12 de octubre de 2026, para el paquete `tfmfdd` 0.3.0 (etiqueta `v0.3`). Sustituye a la del 27 de septiembre, e incluye la ronda de correcciones del 12/10, posterior a dos revisiones independientes del código. Salen de la memoria E2 (capítulo 3 y apartado 2.2.4, citados por párrafo), de `configs/baseline.yaml`, de `docs/decisiones.md`, de la revisión del 27/09 (`docs/revision_2026-09-27.md` y `docs/revision_2026-09-27_agentes.md`) y de la revisión del código del 12/10. Los identificadores entre paréntesis (B2, I1, DAT-1…) remiten a esas revisiones; lo que cambió en cada módulo está en `CAMBIOS.md`.

Las tres partes del TFM se van a comparar entre sí, y eso solo vale si las cifras se miden igual. Cada regla dice por qué existe, qué función del paquete la implementa, qué defecto evita y, si queda algo por decidir, qué clave del yaml lo recoge. Las claves con valor null son decisiones pendientes de Pablo: ningún script las sustituye por un valor por defecto, y el que necesita una se niega a ejecutar (`tfmfdd.config.require`). Los cuadernos `notebooks/B00_punto_de_partida_diagnostico.ipynb` y `notebooks/C00_punto_de_partida_deeplearning.ipynb` aplican estas reglas.

Una advertencia sobre los nombres: en este documento, "nivel de confianza" es `protocolo.alpha` del yaml (0,99, el del límite de control) y "α de los contrastes" es `contrastes.alpha_contrastes` (0,05, el de las pruebas estadísticas). Son cosas distintas, la memoria las nombra distinto, y pasar el 0,99 a una función de `tfmfdd.stats` es un error.

Dos palabras con el sentido de la memoria (2.2.4, párrafo 283): una **fuga** es una decisión en la que influye el conjunto de prueba (estandarizar con la prueba, partir al azar por muestra una serie autocorrelacionada, elegir algo mirando la prueba); un **sesgo** es una decisión que mueve la métrica sin usar la prueba (calibrar los límites con los datos del ajuste), y su signo no se presupone.

## 1. Partición por simulación, nunca por muestra

**Por qué.** Las muestras consecutivas de una simulación se parecen mucho: el proceso tiene inercia y se muestrea cada 3 minutos. Si una muestra va a prueba y su vecina temporal a entrenamiento, el modelo acierta porque reconoce la simulación, no el fallo (la fuga L3.2, memoria, párrafo 282). En Rieth hay un segundo mecanismo: la simulación r de cada fichero de fallo comparte semilla con la simulación normal r del mismo split, así que son la misma trayectoria hasta que entra el fallo y, en los fallos 3, 9 y 15, siguen casi pegadas después (DAT-1, DIS-2). Con una partición por muestra, las dos cosas actúan a la vez y en sentidos contrarios, de modo que el error de partir por muestra no tiene un signo fijo: el B00 (secciones 7.1 y 7.2) lo mide con los dos diseños.

**Cómo.**
- `data.split_runs(n_runs, fractions, seed, exclude=...)` reparte identificadores de simulación, una sola vez y con los mismos ids para todas las clases; `exclude` saca los que no se pueden usar (regla 3), y una parte con fracción 0 queda vacía. Con validación cruzada, `GroupKFold` con `groups = simulationRun` sobre todas las clases a la vez, nunca con el par (fallo, simulación), que separaría a las gemelas.
- En el TEP clásico la regla se cumple sola, porque `dXX.dat` (entrenamiento) y `dXX_te.dat` (prueba) son simulaciones distintas. En Rieth, los ficheros `test_*` no comparten semillas con los `train_*`.
- **Trayectoria y grupo no son la misma clave.** El grupo de una partición es `simulationRun` solo, común a todas las clases. Una trayectoria son las filas seguidas de una misma simulación de una misma clase: en Rieth, el par (`faultNumber`, `simulationRun`). Los retardos, las ventanas, los bloques y la estandarización por simulación se cortan por trayectoria (`preprocessing.trajectory_starts`, la misma definición en todo el paquete); pásenles las dos columnas, `df[["faultNumber", "simulationRun"]].to_numpy()`. Con una sola columna, dos ficheros o clases seguidos con el mismo id se funden en una trayectoria; en el clásico, `simulationRun` vale 1 en todos los ficheros. El paquete avisa cuando una trayectoria pasa de 960 filas (`preprocessing.TEP_MAX_RUN_LENGTH`), lo más que dura una simulación del TEP.
- **Con DPCA o con ventanas, se reparten filas ya apiladas, nunca muestras sueltas.** Si se reparte por muestra y después se apilan retardos o ventanas, cada fila junta muestras que no son consecutivas. Brazo sucio del factor 3 con DPCA:
  1. Se apila primero, con `preprocessing.lag_by_run(X, runs, lags, return_index=True, samples=df["sample"].to_numpy())`: con `samples`, comprueba que cada fila junta muestras seguidas.
  2. Se reparten las filas apiladas con `ablation.split_by_sample(runs[pos], limpio.sizes, seed, block_length=L)`.
  3. Se ajusta `PCAMonitor(lags=0)` sobre ellas, que es el mismo modelo que `PCAMonitor(lags=lags)` sin el gancho del escalador.
  4. La prueba se apila igual y se alinea con `ablation.align_to_samples`.

  Si alguien reparte por muestra y después pasa las filas a `PCAMonitor(lags=...)`, con la columna `sample` (`fit(..., samples=, samples_cal=)`, `score(..., samples=)`) el error lo detiene; sin ella, nada lo detecta. Pásenla siempre: en el brazo limpio no cambia nada. Con ventanas, `ablation.window_ends` no comprueba el tiempo: se reparten las ventanas, no las muestras.
- El brazo limpio del factor 3 es `ablation.split_by_run`, que llama a `data.split_runs` (el mismo reparto, redondeo incluido). El sucio, `ablation.split_by_sample` (por muestra o por bloques), y `ablation.block_ids` como grupos de una validación cruzada, existen solo para medir la ablación: nunca dan un resultado.

**Cuidados de Rieth.** Como la simulación r de cada clase es gemela de la normal r, los pares (fallo, simulación) de fallos distintos con el mismo id no son independientes: el contraste por fallo no se ve afectado, pero un agregado entre fallos se calcula primero dentro de cada simulación. `data.load_rieth` ya da los ids en `int64` y falla con ids que no existen: no hacen falta las conversiones ni las comprobaciones a mano de la v0.2.

**Evita.** La fuga por partición de series autocorrelacionadas y el efecto de las gemelas.

## 2. Escalado ajustado solo con los datos normales del ajuste

**Por qué.** Si la media y la desviación incluyen la prueba, el modelo usa información del examen para prepararse (la fuga L1.2, memoria, párrafo 282). Además, un desplazamiento de media que es justo el fallo quedaría absorbido en parte por el escalado. La memoria fija la regla también para el diagnóstico: "la estandarización se ajustará solo con las simulaciones normales del ajuste, también en el diagnóstico" (párrafo 367; `docs/decisiones.md`, 04/10).

**Cómo.**
- `preprocessing.Scaler()`: con etiquetas (por ejemplo, dentro de una `Pipeline` de diagnóstico, que se las pasa), la media y la desviación salen solo de la clase normal (`normal_label=0`, el valor por defecto), y después se transforma todo. `normal_label=None` ajusta con todas las clases: se aparta del protocolo y hay que pedirlo. Sin etiquetas, `Scaler().fit(X)` usa todas las filas de `X`, así que fuera de una `Pipeline` pasen las etiquetas (`Scaler().fit(X_tr, y_tr)`). `normal_label` es la etiqueta tal como llega en `y`: si recodifican las clases (por ejemplo, con `LabelEncoder` para una red que pide 0..K−1), pasen la etiqueta nueva.
- En una búsqueda de hiperparámetros, el escalador va dentro de cada pliegue (una `Pipeline`), no antes, y la búsqueda va con `error_score="raise"` (regla 12). El `Scaler` entra en `Pipeline` y en `GridSearchCV`.
- `PCAMonitor` escala por dentro con `X_fit`, que ya es solo operación normal.

**Factor 2 (estandarización con datos de prueba).** El brazo S1 (ajuste y prueba juntos) se mide con el gancho del monitor: `PCAMonitor.fit(..., scaler=Scaler().fit(np.vstack([X_fit, X_prueba])))`, y el limpio con `scaler=Scaler().fit(X_fit)`. El monitor guarda una copia del escalador (reajustar el objeto después no cambia el modelo), y `scaler_n_samples_fit_` dice con cuántas filas se ajustó. El S2 (cada simulación de prueba con sus propios estadísticos, como Deng et al., 2021) con `ablation.score_standardized_by_run` en PCA y DPCA, o `ablation.standardize_by_run` en las demás familias. En el DPCA, los brazos usan el gancho (escalar antes de apilar retardos), mientras que el DPCA sin gancho escala la matriz con retardos: cuál es el DPCA del protocolo está pendiente (`limites_protocolo.orden_escalado_dpca`), y qué brazo sucio se mide también (`ablaciones.brazo_sucio_factor2`).

**Evita.** La fuga por estandarización con la prueba.

## 3. Umbrales calibrados con datos normales reservados

**Por qué.** Un modelo se ajusta para describir sus propios datos, así que su estadístico en esos datos tiende a salir distinto del que tendrá en datos nuevos, y un umbral calibrado con él describe el ajuste, no los datos nuevos. Calibrar con los datos del ajuste es un **sesgo**: cuánto mueve la tasa de falsas alarmas, y en qué sentido, depende del límite, del tamaño del ajuste y de los datos, y la memoria se niega a presuponerlo (párrafos 279 y 284). Se mide: es el factor 1 de las ablaciones. Elegir el umbral mirando la prueba (por ejemplo, el que maximiza la F1 sobre ella) ya no es un sesgo sino una fuga (regla 12). La memoria fija la regla: "todos los umbrales se calibrarán a la misma FAR nominal con simulaciones normales reservadas" (párrafo 367).

**Cómo.**
- PCA y DPCA: `PCAMonitor.fit(X_fit, runs, X_cal=X_cal, runs_cal=runs_cal)`. Cualquier otro detector: `limits.limit_empirical` o `limits.limit_kde` sobre su puntuación en las normales reservadas.
- En Rieth se reservan simulaciones normales distintas de las del ajuste; cuántas van a cada parte y cuántas réplicas de ajuste hay está pendiente (`particion.rieth.*`). En el clásico se corta en el tiempo (las últimas filas de `d00.dat`, `particion.calibracion_limites`), y eso solo vale para el desarrollo. Si el modelo tiene parada temprana, esa decisión usa otro bloque de normales, distinto del de calibración y, por supuesto, de la prueba.
- `PCAMonitor.limit_source_` dice de dónde sale cada límite: `"holdout"` (calibrado con `X_cal`), `"in_sample"` (con el estadístico del ajuste) o `"parametric_fit"` (una fórmula con cantidades del ajuste: el T² por la F o la Beta y el SPE por Jackson y Mudholkar, que son formas paramétricas de calibrar en muestra, DIS-6). Mírenlo en lugar de `calibrated_on_holdout_`.
- Si algún límite se calibra con datos, `fit` sin `X_cal` lo calibra en muestra y avisa. La calibración en muestra a propósito se pide: `fit(X, calibration="in_sample")` (la verificación frente a Yin; regla 13) o, para el brazo en muestra del factor 1 con el mismo n_cal que el reservado, `fit(X_fit, X_cal=X_fit[-n_cal:], calibration="in_sample")`. Un `X_cal` vacío es un error, y uno "reservado" que comparte filas con `X_fit` también.
- **El T² y el factor 1.** El límite F del T² (`t2_method="f"`, Fase II, el de siempre) no usa `X_cal`: con él, calibrar en muestra no cambia nada por construcción. Para medir el factor 1 en el T² están `t2_method="beta"` (Fase I), `"empirical"` y `"kde"` (I4, DIS-5); A0 los compara en el clásico.
- **El percentil empírico.** `quantile_method="linear"` es el `np.percentile` de siempre; con datos intercambiables da, en promedio, algo más de FAR que la nominal, y se nota con pocas muestras reservadas (MET-3). `"conformal"` usa el estadístico de orden de Diallo et al. (2025). Reporten siempre el tamaño de `X_cal` (`n_cal_`) y la FAR nominal al lado. El límite KDE (`limits.limit_kde`, `"kde"`) es el cuantil exacto de la KDE ajustada. Quien llame a `limits.spe_limit_jackson` directamente tiene que pasar los autovalores en orden decreciente (si no, es un error; `PCAMonitor` ya los ordena).
- **La regla T² o SPE.** `PCAMonitor.combined_index(t2, spe)` da un índice con límite 1 que supera 1 exactamente cuando lo supera alguno de los dos. Con los dos límites al mismo nivel, la FAR nominal de la disyunción no baja de 1 − α y en general la supera: el nivel de cada estadístico está pendiente (`limites_protocolo.nivel_por_estadistico`).
- Qué límite y qué estimador usa el protocolo en Rieth está pendiente (`limites_protocolo.t2_method`, `spe_method`, `quantile_method`). Los valores de `pca.*` del yaml son los del desarrollo (A0 y A1).
- En el clásico, reservado no significa bien calibrado: con una sola simulación normal no se separa el sobreajuste de la deriva temporal (I5). Rieth, con simulaciones distintas, sí lo permite.

**Nunca mirando la prueba.** La regla también cubre lo que se decide alrededor del umbral. Los cortes y umbrales de las métricas (el 0,10 de `detected_at_chance`, las mitades de `stratified_fdr`, cualquier tramo) se declaran antes de ver los resultados (I8); un corte elegido a posteriori cambia la conclusión (I3). Y las simulaciones que ya se han mirado en desarrollo no sirven para el resultado confirmatorio (DIS-17). Su registro está en el código, `data.RIETH_SEEN_RUNS`, con la fuente de cada entrada: las simulaciones 1 a 20 de los 42 ficheros; de la 21 a la 40 de `train_fault00`; el diseño de ids distintos del B00 en entrenamiento (en `train_faultKK`, de la 20·KK + 1 a la 20·(KK + 1)); las que usó el C00 del 27/09; y la 250 y la 500 de los ficheros 00, 01, 03, 06, 09 y 15. Queda una entrada sin cerrar, los prototipos del C00 (`data.RIETH_SEEN_PENDING`), y mientras siga ahí las consultas avisan. Para consultarlo: `data.seen_runs(split, faults)` y `data.unseen_runs(split, faults)`. En desarrollo, usen las simulaciones 1 a 20, y si miran otra, anótenla en el registro.

Los ids confirmatorios se fijan con un procedimiento escrito en `docs/decisiones.md` antes de ejecutar (`datos.ids_confirmatorios`; por ejemplo, `split_runs(500, seed=..., exclude=data.seen_runs(split, faults))` o `data.unseen_runs(...)`), y son los mismos en los tres bloques. Ojo al decidir qué ficheros cuentan como "no mirados": por el diseño de ids distintos del B00, en entrenamiento quedan muchos menos ids libres que en prueba (`data.unseen_runs` lo calcula).

**Evita.** La calibración en muestra sin decirlo, el umbral elegido con la prueba y los grados de libertad del investigador.

## 4. Detección con `metrics.evaluate_run`

**Por qué.** FAR, FDR y retardo son las tres cifras de la detección, y solo son comparables si se cortan igual. No se resumen en un solo número porque se compensan entre sí (memoria, párrafo 276).

**Cómo.**
- En los ficheros de prueba, `evaluate_run(stat, limite, onset=data.FAULT_ONSET_TEST, k=...)`: la FAR sobre las muestras 1 a 160 y la FDR sobre las 161 a 960. En las simulaciones normales (`onset=None`), la FAR sobre la serie entera o sobre la ventana declarada con `far_window`; cuál usa el protocolo está pendiente (`deteccion.ventana_far_normal`), y A0 y A1 usan la serie entera y lo dicen en sus columnas.
- El retardo es la primera alarma de la primera racha de k alarmas seguidas (`protocolo.k_alarmas`): la convención "appearance" (`protocolo.retardo_convencion`), que da 0 si la racha empieza en la muestra 161 (memoria, párrafos 276 y 289). La otra, "confirmation", es la aparición más k − 1.
- Si `detected_at_chance` es `True` (racha de k con FDR por debajo de `protocolo.min_fdr`), ese retardo no se interpreta: en 800 muestras autocorrelacionadas, una racha de k puede salir por azar. `metrics.aggregate` y `metrics.summary` ya excluyen esos retardos y dan el n de cada media (`<métrica>_n`), `valid_detection_ratio` y dos medianas del retardo: la de las detecciones válidas (`delay_samples_median`) y la censurada (`delay_samples_median_censored`). Cuál se reporta está pendiente (`deteccion.mediana_retardo`). Los intervalos salen recortados al rango de cada métrica.
- **Muestras no evaluadas.** Un NaN en el estadístico es una muestra que no se evaluó (las primeras de cada trayectoria en el DPCA o con ventanas): no cuenta en la FAR ni en la FDR, y `n_far` y `n_fdr` dicen cuántas se evaluaron. Un límite NaN o infinito es un error.
- **DPCA.** `PCAMonitor.score` devuelve una fila por fila de entrada, en el mismo orden, con NaN en las `lags` primeras de cada trayectoria: el inicio del fallo es el mismo índice que en el PCA y no hay que restar nada (la vuelta `onset − lags` de la v0.2 aplicada a la salida nueva sería un error). Con varias clases que comparten ids, pasen como `runs` las dos columnas de la trayectoria.
- **Ventanas (OE3).** Con `ablation.window_ends` (paso 1 para evaluar la detección) y `ablation.align_to_samples`, la puntuación de cada ventana queda en su última muestra y las L − 1 primeras de cada trayectoria quedan en NaN: el mismo `onset` vale.

**Matices.**
- La FDR se lee siempre al lado de la FAR. `detected_at_chance` solo mira si la FDR es baja, así que un detector que alarma casi siempre pasa el filtro con FDR alta y retardo cero (B00, sección 8). Una FDR del orden de la FAR no es detección (I6). Para la hipótesis, el suelo es `stats.detects_above_far` (el mismo Wilcoxon de la regla 8), con su margen y la FAR que se empareja pendientes (`deteccion.suelo.*`).
- En Rieth, la FAR se mide en las simulaciones normales de prueba (`test_fault00`). Las 160 primeras muestras de cada fichero con fallo son copia de su gemela normal: `summary` da `far_normal` (la de la simulación normal) y, aparte, `far_prefallo_medio`, que cuenta el mismo tramo una vez por fichero de fallo y no es independiente.
- En Rieth, el inicio efectivo del fallo puede llegar varias muestras después del nominal (regla 6): parte del retardo es física del proceso, no defecto del detector.
- Falsas alarmas por hora (OE3): `false_alarm_samples_per_hour` (muestras en alarma) y `false_alarm_episodes_per_hour` (rachas de al menos k); la unidad está pendiente (`deteccion.falsas_alarmas_por_hora`). PR-AUC: `metrics.pr_auc`, que se reporta siempre con su prevalencia (su nivel de azar no es 0,5); qué muestras forman el conjunto está pendiente (`deteccion.pr_auc_conjunto`).

**Evita.** Retardos por azar tomados como detección, la FAR contada varias veces y el inicio desplazado del DPCA.

## 5. Diagnóstico: F1 por clase sobre todo el conjunto de prueba, y macro-F1

**Por qué.** La métrica del diagnóstico es la macro-F1, que promedia la F1 de cada fallo y pesa lo mismo cada tipo de fallo (memoria, párrafo 278; `docs/decisiones.md`, 04/10). La precisión de una clase necesita los falsos positivos que llegan desde las demás clases. Si la F1 se calcula solo con las muestras cuya clase real es esa, no puede haber falsos positivos, la precisión vale 1 y lo que sale es 2r/(1 + r), una función del recall que siempre es mayor o igual que la F1 de verdad, y más cuanto más baja es la precisión de la clase (B3).

**Cómo.**
- `metrics.diagnosis_metrics(y_true, y_pred, include_normal=...)`: tabla por clase (soporte, predicciones, precisión, recall, F1) sobre toda la prueba, `macro_f1` y, aparte, `macro_f1_rest` y `macro_f1_hard` (los fallos 3, 9 y 15, como pide el OE2, párrafo 358). `include_normal` no tiene valor por defecto: si la clase normal entra en el promedio está pendiente (`diagnostico.incluir_normal`) y tiene que ser el mismo en los tres bloques. Una clase que no se predice nunca tiene precisión NaN y F1 0.
- En Rieth, el recall está bien definido por (fallo, simulación): `metrics.diagnosis_recall_by_run`. La precisión y la F1 por clase se calculan sobre el conjunto (*pool*) de simulaciones de prueba de todas las clases. La F1 de las clases 0, 3, 9 y 15 se lee al lado de lo que daría repartir al azar entre ellas (B00, sección 8).
- Para el contraste del OE2, una macro-F1 por réplica: `metrics.macro_f1_by_replicate`, con la unidad de réplica declarada (particiones externas o grupos de `simulationRun`), que está pendiente.
- El escalado, solo con la clase normal (regla 2).
- Pendiente también: qué etiqueta llevan las muestras previas al inicio del fallo (`diagnostico.etiqueta_prefallo`) y, en los modelos con ventanas, si una ventana de prueba puede empezar antes del inicio (el C00 deja las dos opciones como parámetro).

**Evita.** El "f1" del cuaderno 02 anterior y del CSV `B1_clasificadores_clasico.csv`, retirado en la v0.3.

## 6. Dónde entra el fallo

**Por qué.** Etiquetar como fallo muestras sanas, o al revés, contamina las clases.

**Cómo.**
- TEP clásico, entrenamiento: los `dXX.dat` tienen 480 filas con el fallo activo desde la fila 0, y se usan enteros; `data.fault_onset("train", "classic")` devuelve `None`. La constante `FAULT_ONSET_TRAIN_CLASSIC`, que valía 20 y era falsa (B2), se retiró en la v0.3: pedirla da un error que dice qué usar. `d00.dat` tiene 500 filas y viene transpuesto; el paquete lo corrige y avisa.
- TEP clásico, prueba: 960 muestras, con el fallo tras la 160.
- Rieth: el fallo entra tras la muestra 20 en entrenamiento y tras la 160 en prueba (`data.fault_onset(split, "rieth")`), comprobado con los datos. En los ficheros de fallo, las filas anteriores son copia exacta de la gemela normal y se descartan.

**Matiz (DAT-1).** El inicio nominal no siempre es el efectivo. En algunos fallos de Rieth la trayectoria sigue idéntica a su gemela normal después del inicio, en algunos casos durante decenas de muestras (informe 11 de la revisión del 03/10, B.5), de modo que al descartar solo `sample <= onset` quedan filas con etiqueta de fallo que son copia exacta de la clase 0. `data.effective_onset(fault_df, normal_df, onset)` lo mide por (fallo, simulación) y su docstring trae la receta para descartar ese tramo. Necesita las muestras posteriores al inicio desde `onset + 1` y seguidas hasta la primera distinta de la gemela: no le pasen filas recortadas ni ventanas (si faltan, da un error). Si se declara, se mide o se descarta está pendiente (`datos.tramo_copiado`), igual en los tres bloques.

**Evita.** Descartar muestras con fallo en el clásico y meter muestras normales con etiqueta de fallo en Rieth.

## 7. Los fallos 3, 9 y 15, siempre aparte

**Por qué.** Su huella en las 52 variables apenas se separa del ruido y, en Rieth, sus simulaciones siguen pegadas a su gemela normal. Promediados con el resto, lo que pasa en ellos queda escondido detrás de los fallos fáciles, y es en ellos donde la literatura publica las cifras más discutidas. Separarlos no presupone ningún resultado: la hipótesis predice que en ellos la ventaja de los métodos no lineales y profundos no será significativa (memoria, párrafo 348), y eso se contrasta. Una F1 alta en 3, 9 y 15 obliga a revisar antes que nada el protocolo (partición, escalado, selección).

**Cómo.** `metrics.HARD_FAULTS`, que coincide con `fallos_dificiles` del yaml. Cada tabla lleva dos bloques: el resto de fallos, y 3, 9 y 15 (`summary` da `fdr_medio_sin_dificiles` y `fdr_medio_dificiles`; `diagnosis_metrics`, `macro_f1_rest` y `macro_f1_hard`).

**Evita.** Una media global que oculta dónde está la dificultad.

## 8. Contrastes pareados

**Por qué.** Con muchas simulaciones, casi cualquier diferencia sale significativa. Un p sin la diferencia al lado no dice si la ventaja importa. La regla que decide si una diferencia es real es el Wilcoxon pareado con corrección de Holm, fijada antes de ver los resultados (memoria, párrafos 344 y 358).

**Cómo.**
- El α de los contrastes se lee del yaml (`contrastes.alpha_contrastes`) y se pasa como `alpha=` a las funciones de `tfmfdd.stats`.
- Análisis principal: `stats.compare_methods(results, metric, baseline=..., alpha=...)`. Wilcoxon pareado por simulación sobre método − baseline, con p exacto (también con empates y ceros; hasta 500 pares siempre, y por encima mientras las diferencias no nulas sean pocas) y Holm dentro de cada fallo sobre las k − 1 comparaciones. Sin `baseline`, todos contra todos (secundario).
- Se reporta antes que el p: `hl_estimate` (Hodges-Lehmann, la pseudomediana de la diferencia) con su intervalo (`ci_low`, `ci_high`), `effect_size` (correlación biserial de rangos, con signo, de −1 a 1) y `p_adjusted`. `LowPowerWarning` avisa cuando el número de diferencias no nulas no permite llegar a la significación (`stats.min_runs_for_significance`); con menos de 5 pares no hay contraste. Con una sola simulación (TEP clásico) no hay contraste posible, y los números son orientativos.
- `family` declara la familia de Holm; con ella, las comparaciones no contrastables cuentan con p = 1, porque la familia es la declarada.
- **Réplicas de ajuste (factor 1).** Con `replicate_col`, cada celda (réplica, simulación de prueba) da una diferencia, y las celdas están cruzadas: las R réplicas comparten las mismas N simulaciones de prueba. `compare_methods` y `compare_arms` exigen entonces declarar la unidad: `unit="replicate"` (la media de las simulaciones dentro de cada réplica) o `unit="simulation"` (la media de las réplicas dentro de cada simulación de prueba, la unidad de análisis de la memoria, párrafo 367). Ninguna de las dos vale en general, porque cada una es condicional a la otra: con `"replicate"`, el contraste vale para esas N simulaciones de prueba y no generaliza a otras; con `"simulation"`, vale para esas R réplicas. Lo muestra un Monte Carlo sintético de la revisión del 12/10 (guion `despues_I4_unidades.py` del informe de imp_estadistica). Configuración: diferencias d_rj = u_r + v_j + e_rj sin efecto medio, con desviaciones sd(u) de la réplica, sd(v) de la simulación de prueba y sd(e) de la celda; R = 10 réplicas y N = 50 simulaciones; 300 repeticiones por escenario con la semilla 20261012; `compare_methods(..., baseline=..., replicate_col=..., unit=...)` y rechazo con p ≤ 0,05. Su salida, que reproduje con el código de la v0.3 y da lo mismo:

  ```
  sd(u)=0.002 sd(v)=0.01 sd(e)=0.002: rechazo sin efecto con unit='replicate' 0.437 | unit='simulation' 0.067
  sd(u)=0.01 sd(v)=0.002 sd(e)=0.002: rechazo sin efecto con unit='replicate' 0.063 | unit='simulation' 0.843
  sd(u)=0.002 sd(v)=0.0 sd(e)=0.002: rechazo sin efecto con unit='replicate' 0.063 | unit='simulation' 0.807
  ```

  Cuando la diferencia varía de una simulación de prueba a otra, la unidad réplica rechaza sin efecto mucho más del 5 %; cuando varía de una réplica a otra, le pasa a la unidad simulación. Generalizar a las dos poblaciones a la vez exigiría un contraste para diseños cruzados, que el paquete no tiene. La unidad, y con ella la estimanda del factor 1, es una decisión pendiente (`ablaciones.unidad_replica`).
- Para emparejar métodos de bloques distintos, los tres tienen que usar los mismos ids de prueba.

**La hipótesis (párrafo 368).** `stats.paired_advantage` construye la ventaja por simulación con el sentido de "mejor" explícito (`higher_is_better`, obligatorio); `stats.non_superiority_test(ventaja, delta)` da los tres desenlaces de cada comparación ("apoya", "ventaja relevante", "inconclusa"); y `stats.hypothesis_verdict` aplica la regla a la familia confirmatoria (las seis comparaciones del autoencoder frente a PCA y DPCA en los fallos 3, 9 y 15): refuta si alguna muestra una ventaja relevante tras Holm, apoya solo si las seis apoyan, una comparación en la que ninguno de los dos detecta por encima de su FAR no cuenta como apoyo (`detects_above_far`), y los fallos de control entran aparte. Solo dice "apoya" con la familia declarada de seis comparaciones y con fallos de control; si no, "inconclusa", y la explicación dice por qué. Una comparación sin ventaja (`advantage=None`) es "no contrastable", y el desenlace de los controles va aparte, en `control_outcome`. El campo `published_condition` recuerda que el veredicto no evalúa la condición de la ventaja publicada del párrafo 349: con `published=` muestra los resultados de `published_advantage_test`, sin combinarlos. Pendiente: δ (`contrastes.delta.*`, sobre la pseudomediana y en las unidades de la métrica; la propuesta de 2 puntos de FDR o FAR y 1 muestra de retardo sigue sin acordar, DIS-11), la lateralidad, las comparaciones no contrastables, la familia entera (`hipotesis.*`: métrica y sentido de la ventaja, qué son "PCA" y "DPCA", fallos de control) y cómo entra en el veredicto la condición de la ventaja publicada. Ojo: con la FDR cruda como métrica de la ventaja, una comparación en el suelo puede refutar (el suelo solo impide apoyar); el exceso de la FDR sobre la FAR no tiene ese problema.

**Las ablaciones (párrafo 355).** `stats.compare_arms(results, metric, arm_col, clean, margin=..., family=...)` decide cada efecto frente a un margen por métrica: "efecto relevante", "equivalente a cero" o "inconcluso", con Holm sobre la familia declarada (`stats.effect_test` para una comparación). Con margen 0 es el Wilcoxon bilateral con Holm de siempre. Con `replicate_col`, la unidad se declara como en `compare_methods`. Márgenes, familias, métrica principal y unidad de cada factor están pendientes (`ablaciones.*`). Para el factor 4, `ablation.block_bootstrap_ci` (bootstrap de bloques circulares) da el intervalo que construiría quien solo tiene una simulación, e `ablation.interval_coverage` mide con qué frecuencia contiene su referencia: la media de la misma serie en las N simulaciones, mejor sin la propia. Su falta de cobertura mezcla el error del propio intervalo (autocorrelación fuerte, alarmas raras, series no estacionarias como la de un fallo con retardo) con lo que varía entre simulaciones, y aparece aunque todas salgan del mismo proceso: no se lee como pseudorreplicación sin una cobertura de calibración (lo explican sus docstrings). La inferencia del protocolo es siempre entre simulaciones.

**La ventaja publicada y el mejor compromiso.** `stats.published_advantage_test` compara la ventaja medida con la publicada en la misma métrica (párrafos 349 y 350), a partir de una fila de la tabla de ventajas publicadas, que se fija antes de los experimentos confirmatorios (`hipotesis.tabla_ventajas_publicadas`). Con un contraste unilateral, "igual que la publicada" no se puede probar: si debe refutar, hace falta una tolerancia, y eso está pendiente. `stats.pareto_front` y `stats.best_compromise` aplican el criterio de mejor compromiso, que no tiene valor por defecto y es uno para la detección y otro para el diagnóstico (párrafo 369; `compromiso.*`).

**Limitaciones.** El nivel exacto de Wilcoxon supone que la diferencia es simétrica alrededor de su pseudomediana. En fallos casi en el suelo puede ser muy asimétrica (casi todas las simulaciones cerca de 0 y unas pocas lejos), y entonces el error de uno de los dos unilaterales pasa de α; en la intersección-unión de la hipótesis, el apoyo en falso queda acotado por el peor de los niveles. El docstring de `stats.non_superiority_test` ("Supuesto de simetria") lo explica y da los análisis de sensibilidad que se pueden declarar de antemano: la prueba de signos sobre la mediana, la t pareada o el bootstrap de la media. Qué parámetro contrasta la hipótesis y qué sensibilidad se declara es una decisión pendiente.

**Evita.** Declarar "significativo" lo que no es relevante, p que no son los del análisis declarado y la pseudorreplicación.

## 9. Semillas y configuración, desde `configs/baseline.yaml`

**Por qué.** Si un valor del protocolo está en dos sitios, acaba teniendo dos valores: el A1 usó 0,90 de varianza mientras el yaml decía 0,85 (`decisiones.md`, 22/09), y los cuadernos anteriores fijaban a mano semillas y fracciones que no coincidían (M3, M4).

**Cómo.**
- `tfmfdd.config.load_baseline()` busca el yaml en este orden: la ruta que se pase, la variable de entorno `TFMFDD_CONFIG`, `configs/baseline.yaml` del directorio de trabajo y, por último, la copia que viaja con el paquete (`tfmfdd/configs/baseline.yaml`, idéntica). Quien instala `tfmfdd` con pip en su repositorio recibe esa copia; con `with_info=True` sabe cuál se usó y su huella (REP-4, resuelto).
- `config.get(cfg, "pca.variance")` no tiene valor por defecto (una clave que falta es un error), y `config.require` se niega con una clave en null: es una decisión pendiente. `config.pending(cfg)` las lista.
- Se leen del yaml `protocolo.*` (nivel de confianza, k, minutos por muestra, semilla, `min_fdr`, convención del retardo), `contrastes.alpha_contrastes`, `particion.*`, `datos.*`, `pca.*` (los valores de desarrollo) y `fallos_dificiles`. Lo que no está en el yaml (los hiperparámetros de cada modelo) se declara en la primera celda, antes de ver resultados. Lo que depende de una clave en null también va ahí, como parámetro y sin tomar la decisión, y en cuanto la clave tenga valor manda el yaml: así lo hacen los cuadernos con `INCLUIR_NORMAL` (`diagnostico.incluir_normal`) y con `COSTE_DEMO` (`coste.*`).
- Un cambio en el yaml es un evento de coordinación: se anuncia y los tres recalculan lo afectado (cabecera del yaml).
- Con PyTorch en CPU, el resultado cambia con el número de hilos: regístrenlo junto a las versiones (`cost.machine_info()` los da).

**Evita.** Semillas y constantes distintas en cada cuaderno, y valores por defecto que esconden una decisión.

## 10. Ningún número de resultado escrito a mano en el texto

**Por qué.** Un número copiado a mano se desactualiza en cuanto cambia algo, y en este proyecto ya pasó: cifras de dos configuraciones distintas acabaron citadas como la misma (`decisiones.md`, 22/09), y una cifra dependía de un corte que nadie había declarado (I3).

**Cómo.** Las interpretaciones remiten a la salida de la celda o al CSV ("fíjate en la salida anterior…"). Sí se escriben los hechos de los datos (52 variables, 160, 480, 500, 960), los valores de configuración y las citas de la literatura con su página. Toda tabla o figura de la memoria sale de código, con su configuración al lado y su `.provenance.json` (regla 11).

**Evita.** Cifras desfasadas o sin procedencia.

## 11. Formato común de resultados

**Por qué.** Es lo que permite que `metrics.aggregate` y `stats.compare_methods` funcionen igual en los tres bloques.

**Cómo.**
- Una fila por (`method`, `faultNumber`, `simulationRun`).
- En detección, las 15 columnas de `evaluate_run`: `far`, `fdr`, `delay_samples`, `delay_minutes`, `detected`, `detected_at_chance`, `fdr_h1`, `fdr_h2`, `persistence_gap`, `n_far`, `n_fdr`, `n_fdr_h1`, `n_fdr_h2`, `false_alarm_samples_per_hour` y `false_alarm_episodes_per_hour`.
- En diagnóstico, el recall por (fallo, simulación) de `diagnosis_recall_by_run` (`faultNumber`, `simulationRun`, `recall`, `n_samples`, `n_other_label`) y, en un fichero aparte, la tabla por clase de `diagnosis_metrics` (precisión y F1 sobre el conjunto de prueba).
- La configuración va en columnas o en el JSON de procedencia: `tfmfdd.provenance.write_provenance(csv, config=..., ids=..., inputs=...)` escribe al lado del CSV un `.provenance.json` con la configuración efectiva, los ficheros de datos con su SHA-256, los ids, el commit de git (leído sin escribir en el repositorio), la máquina y las versiones (I7). Un script que escribe varios CSV calcula git y la máquina una vez, con `ctx = provenance.run_context()`, y lo pasa a cada uno (`write_provenance(..., git=ctx["git"], machine=ctx["machine"])`): así todos los JSON de una ejecución describen el mismo estado del repositorio.
- Mientras sea una prueba de humo, va a `results/raw/`, que git ignora; los resultados que se citan van a `results/summary/`.

**Evita.** Métricas que no se pueden agregar ni contrastar igual en los tres bloques, y cifras sin procedencia.

## 12. Selección de modelos y umbrales solo con entrenamiento y validación

**Por qué.** "Ninguna elección de hiperparámetros, umbral o parada temprana (early stopping) usará las simulaciones de prueba" (memoria, párrafo 367). Elegir algo mirando la prueba es una fuga por selección (párrafo 282): la cifra que se reporta ya no es una estimación del desempeño en datos nuevos. Y la mejor puntuación de una búsqueda es optimista por construcción: el máximo de varias estimaciones con ruido sobreestima, en promedio, la del modelo elegido.

**Cómo.**
- Los hiperparámetros, el umbral y la parada temprana se eligen solo con simulaciones de entrenamiento y validación: `GroupKFold` con `groups = simulationRun`, agrupando todas las clases a la vez (regla 1), y con el escalador dentro de la `Pipeline` (regla 2). En los detectores, esas simulaciones son normales; en los clasificadores del OE2 y del OE3, llevan también fallos.
- La cifra que se reporta sale de simulaciones de prueba que no participaron en ninguna elección.
- **Nunca se reporta `best_score_`**, ni ninguna otra puntuación interna de la búsqueda (tampoco el mejor valor de una optimización como Optuna).
- Las búsquedas van con `error_score="raise"`. Si un pliegue falla (por ejemplo, el escalador de la regla 2 en un pliegue sin la clase normal), la búsqueda se detiene; sin esa opción, `GridSearchCV` puntúa el pliegue con NaN y elige un candidato sin decirlo.
- En Rieth no hace falta validación anidada, porque los ficheros `test_*` son una prueba independiente con muchas simulaciones; solo hace falta si se ajusta algo sin una prueba independiente, como en el TEP clásico.
- **Las columnas de identificación no son variables.** `faultNumber`, `simulationRun` y `sample` no entran en ningún modelo: `sample` anticipa el fallo, que empieza siempre en la misma muestra (la variable ilegítima L2, párrafo 283), y `simulationRun` delata la trayectoria. Construyan la matriz con `data.values(df)`, que se niega a devolverlas (`data.ID_COLUMNS`), y no con `df.drop(columns=["faultNumber"])`, que deja dentro las otras dos.
- **Las simulaciones confirmatorias** son las mismas en los tres bloques, se eligen con el procedimiento escrito del registro y no se han mirado en desarrollo (regla 3).

**Evita.** La fuga por selección y las variables ilegítimas.

## 13. Verificación frente a Yin et al. (2012) y prerregistro

**Por qué.** La memoria promete dos cosas con fecha. Primera: verificar la implementación del código frente a las cifras de Yin et al. (2012), "dentro de una tolerancia fijada antes de ejecutar" (párrafo 355), "sin ajustes para que coincidan y sin contar los fallos 3, 9 y 15" (párrafo 369). Segunda: escribir con fecha en el registro de decisiones todo lo que concrete el planteamiento de la comparativa y fijarlo con una etiqueta de versión antes de los experimentos confirmatorios (párrafo 369).

**Cómo (verificación).**
- Se verifica la **implementación**, no las reglas del protocolo: el código, ejecutado con la configuración de Yin (33 variables, 9 componentes en el PCA y 17 en el DPCA, límites en muestra con la F de Fase II y Jackson y Mudholkar, regla T² o SPE), tiene que reproducir su FDR por fallo y su FAR en operación normal (Tablas 4 y 5). Los fallos 3, 9 y 15 se calculan y se reportan, pero no cuentan.
- La hace `experiments/V1_verificacion_yin.py` con `configs/yin2012.yaml` y la transcripción de `configs/referencias/`. Las cifras de Yin se leen del CSV, no se teclean.
- **Regla de orden.** V1 no abre ningún fichero de prueba (`dXX_te.dat`) hasta que no queda ninguna clave en null en el yaml, hay una fila fechada en `docs/decisiones.md` con la huella que imprime `--huella`, y el yaml, el registro, las referencias y el código que se ejecuta (el propio script y la carpeta `tfmfdd` que se importa) están confirmados en git, en un mismo repositorio. Por eso V1 se ejecuta con el paquete del repositorio (instalación editable o `PYTHONPATH`), no con una rueda instalada. La fila tiene que verse (fuera de comentarios `<!-- -->` y de bloques de código) y su fecha tiene que existir. `--solo-limites` usa solo `d00.dat` y no necesita nada de eso. La última sección de `docs/decisiones.md` da los pasos.
- A1 no es la verificación, y `--cal-fraction 0` no es una forma de calibrar en muestra (ahora es un error).
- Si no cuadra: primero la comparación de los límites con los que dibujan Yin, después solo las variantes declaradas de antemano; si sigue sin cuadrar, se reporta "no verificado", con la diferencia por fallo. El detalle, con los supuestos y su evidencia, está en `docs/verificacion_yin.md`.

**Cómo (prerregistro).** Lo que el registro tiene que fijar antes de los experimentos confirmatorios (párrafos 368 y 369, y la Figura 2) está hoy en el yaml como claves en null: las simulaciones de prueba (`datos.ids_confirmatorios`), la tabla de ventajas publicadas, la diferencia mínima relevante y el número de simulaciones, la familia confirmatoria, el suelo y los controles, las familias y márgenes de las ablaciones, la adaptación del índice de robustez con su rejilla de ruido, el coste y el criterio de mejor compromiso. Cada una, cuando se decide: una fila con fecha en `docs/decisiones.md` y el valor en el yaml, en el mismo commit; y todo antes de la etiqueta de versión que fija el prerregistro.

**Evita.** Una tolerancia o una regla elegida después de ver los resultados.

## 14. Coste computacional y robustez ante el ruido

**Por qué.** El objetivo general pide el mejor compromiso entre detección, falsas alarmas, retardo, diagnóstico y coste (párrafo 342), y la memoria fija cómo se miden el coste ("tiempo de ajuste, tiempo de inferencia por muestra y pico de memoria en una misma máquina de referencia") y la robustez (el índice de Bernal-de Lázaro et al., 2018, "sobre los modelos finales de cada bloque", con su adaptación "declarada antes de ejecutar"; párrafo 369).

**Cómo (coste).**
- `cost.measure_cost(fit, predict, X_infer, method=..., batch_size=..., n_repeats_fit=..., n_warmup_fit=..., n_repeats_infer=..., n_warmup_infer=...)`: el procedimiento común. Ningún parámetro del protocolo tiene valor por defecto; salen del yaml (`coste.*`, pendiente), igual en los tres bloques.
- El pico de memoria de una medida solo es exacto si supera el pico anterior del proceso, y la fila lo dice (`*_exact`). Para el ajuste, eso se consigue midiendo cada método en un proceso nuevo. Para la inferencia, no: su memoria se mide justo después del ajuste, así que en el mismo proceso solo sale exacta si supera la del ajuste, aunque el proceso sea nuevo. `measure_cost(..., infer_memory="subprocess")` la mide en un proceso aparte, con el modelo serializado, y la columna `infer_memory_where` lo registra. En qué modo se mide para la tabla de coste es parte de la decisión pendiente sobre la memoria que se reporta (`coste.memoria_reportada`).
- Los mismos hilos de BLAS, OpenMP y PyTorch para todos los métodos, registrados: `cost.machine_info()` en un JSON al lado de la tabla.
- El reloj principal es `perf_counter`. El tiempo de CPU avanza a saltos (`machine_info()["process_time_tick_s"]` mide el salto) y solo sirve por pasadas largas. La memoria de la GPU no se mide.

**Cómo (robustez).**
- `robustness.noisy_versions` genera el ruido del lema 1 del artículo, por niveles y repeticiones, con una semilla declarada; `robustness.error_rate` calcula el error con la métrica de la adaptación (obligatoria); `robustness.robustness_index` da el índice y sus tres componentes por separado.
- La adaptación está pendiente entera (`robustez.*`): la rejilla de niveles (el término de pendientes depende del paso, así que el índice solo se compara con la misma rejilla), el tipo de ruido, las variables que lo reciben y de qué datos sale su desviación, las repeticiones, si se usan números aleatorios comunes, la métrica para detectores y para el diagnóstico, si el límite es fijo o se recalibra en cada nivel y el nivel de confianza.
- En detección, el índice se da por curva (falsas alarmas y detecciones perdidas), no con la ecuación 3 del artículo, que las mezcla en un solo número (párrafo 276). Con el límite fijo, el ruido añade alarmas y la curva de detecciones perdidas puede bajar con el nivel: solo mide detección perdida si el límite se recalibra (lo explica el docstring de `robustness.error_rate`).
- Las lecturas del artículo marcadas [VERIFICAR] en los docstrings conviene confirmarlas con el director, que es su autor.
- Los fallos 3, 9 y 15, aparte (regla 7).

**Evita.** Tiempos y memorias que no se pueden comparar entre bloques, y un índice cuyo resultado dependa de una adaptación decidida después.

## Qué cambió en la v0.3

Las vueltas que hacía falta dar con la v0.2 ya no hacen falta (el detalle, con los resultados afectados, en `CAMBIOS.md`):

- Regla 1: `load_rieth` da los ids en `int64`, falla con ids inexistentes y lee filtrando (DAT-4 a DAT-6); `split_runs` admite `exclude`; las particiones del factor 3 y las ventanas por trayectoria están en `tfmfdd.ablation`.
- Regla 2: el `Scaler` entra en una `Pipeline` y se ajusta por defecto solo con la clase normal (decidido el 04/10); el factor 2 se puede medir en PCA y DPCA con el gancho del escalador (S1) y `score(..., scaler=...)` (S2), y ya no es nulo por construcción (DIS-7).
- Regla 3: `limit_source_`, `calibration="in_sample"`, el aviso de `fit` sin `X_cal` y el error con `X_cal` vacío; el T² calibrable (`t2_method`, I4, DIS-5), el percentil conforme y la regla T² o SPE (`combined_index`); el registro de simulaciones miradas en el código (`data.RIETH_SEEN_RUNS`).
- Regla 4: el DPCA alineado con NaN en lugar de la vuelta `onset − lags` (MET-1, MET-2); NaN = muestra no evaluada (MET-11); `aggregate` y `summary` sin los retardos al azar, con n e intervalos recortados (I1, MET-4); `far_window`; falsas alarmas por hora y PR-AUC; el suelo de la hipótesis.
- Regla 5: `diagnosis_metrics`, `diagnosis_recall_by_run` y `macro_f1_by_replicate` (B3); el CSV B1, retirado.
- Regla 6: `FAULT_ONSET_TRAIN_CLASSIC` retirada con un error que dice qué usar (B2); `effective_onset` (DAT-1).
- Regla 8: `compare_methods(..., baseline=...)` con Hodges-Lehmann, intervalo, efecto con signo, p exacto y p ajustado (I2, DIS-10, MET-8, MET-9); el α de los contrastes en el yaml; la no superioridad y el veredicto de la hipótesis; `compare_arms` para las ablaciones; la ventaja publicada y el mejor compromiso.
- Regla 9: el yaml viaja con el paquete y se lee con `tfmfdd.config` (REP-4); los valores por defecto de `PCAMonitor` coinciden con el yaml.
- Regla 11: `tfmfdd.provenance` escribe el JSON de procedencia de cada CSV.
- Reglas nuevas: la 12 (selección sin mirar la prueba), la 13 (verificación frente a Yin y prerregistro) y la 14 (coste y robustez).

Ronda de correcciones del 12/10, antes de publicar (el detalle, en `CAMBIOS.md`):

- Regla 1: aviso de las trayectorias de más de 960 filas; `samples=` para comprobar que los retardos juntan muestras seguidas; `ablation.split_by_run` llama a `data.split_runs`, y una parte con fracción 0 queda vacía.
- Regla 2: el monitor copia el escalador externo; `normal_label` es la etiqueta tal como llega; `error_score="raise"` en las búsquedas.
- Regla 3: el límite KDE es el cuantil exacto de la KDE; Jackson exige los autovalores en orden decreciente.
- Regla 6: `effective_onset` exige las muestras posteriores al inicio seguidas.
- Regla 8: `unit=` con réplicas de ajuste; el veredicto exige la familia declarada y los controles, y dice que no evalúa la ventaja publicada; Holm por pasos; el p exacto según las diferencias no nulas; el supuesto de simetría, escrito; el bootstrap de bloques circulares.
- Regla 9: lo que depende de una clave en null va en la primera celda como parámetro, y manda el yaml en cuanto la clave tenga valor.
- Regla 11: `provenance.run_context`.
- Regla 12: `error_score="raise"` en las búsquedas.
- Regla 13: la regla de orden de V1 cubre el código.
- Regla 14: la memoria de la inferencia, en un proceso aparte si se quiere exacta.

Lo que sigue abierto son decisiones, no vueltas: las claves en null de `configs/baseline.yaml` y `configs/yin2012.yaml` (`docs/decisiones.md`, última sección).
