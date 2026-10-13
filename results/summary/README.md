Resumenes de resultados. Estos SI se versionan: son la procedencia de cada tabla de la memoria.

Cada CSV lleva al lado un `.provenance.json` (`tfmfdd.provenance`) con la configuración efectiva, los ficheros de configuración leídos y su huella, los ids usados, los ficheros de datos con su SHA-256, el commit de git (leído sin escribir en el repositorio), la máquina y las versiones de las bibliotecas, y la fecha. Una cifra de estos CSV solo se cita con su configuración (`docs/decisiones.md`, 22/09).

**Todo lo que hay hoy aquí es de desarrollo sobre el TEP clásico.** El clásico solo se usa en el desarrollo y en la verificación frente a Yin et al. (2012) (memoria, párrafo 367); las cuatro ablaciones y los resultados del TFM salen de Rieth, con muchas simulaciones y contraste pareado. Ninguna cifra de esta carpeta es una fila de la tabla de ablaciones.

## Qué hay

| Ficheros | Script | Qué son |
|---|---|---|
| `A0_calibracion_limites.csv` y `..._yin33.csv` | `experiments/A0_calibracion_limites.py` (con `--yin-vars`, las 33 variables de Yin) | Brazos del SPE con el mismo modelo: límite calibrado con todas las filas del ajuste (`en_muestra`), con datos reservados (`reservada`), en muestra con el mismo n_cal que el reservado (`en_muestra_mismo_ncal`) y Jackson y Mudholkar (`jackson`, calibración en muestra implícita). FAR antes del fallo (media de `d01_te` a `d21_te`) y en `d00_te`. |
| `A0_calibracion_limites_t2.csv` y `..._t2_yin33.csv` | el mismo | Brazos del T² con el mismo modelo: F de Fase II (`f`), Beta de Fase I (`beta`), percentil empírico reservado y en muestra con el mismo n_cal; `far_teorica_faseII` es la FAR que la F de Fase II predice para cada límite paramétrico en datos nuevos. |
| `A0_calibracion_limites_por_fichero.csv` y `..._yin33.csv` | el mismo | FAR de cada brazo en cada fichero de prueba. |
| `A1_pca_baseline_runs.csv` | `experiments/A1_pca_baseline.py` | Línea base con la configuración del protocolo (52 variables, varianza de `configs/baseline.yaml`, límites con datos reservados): una fila por fallo y estadístico, en el formato común (`method`, `faultNumber`, `simulationRun`), con las columnas de `metrics.evaluate_run` y la configuración. |
| `A1_pca_baseline_T2.csv` y `A1_pca_baseline_SPE.csv` | el mismo | `metrics.aggregate` por fallo, con la configuración. |
| `V1_limites_d00.csv` | `experiments/V1_verificacion_yin.py --solo-limites` | Límites de la configuración de Yin et al. (2012) y de sus variantes declaradas, calculados solo con `d00.dat` (el ajuste), frente a los umbrales que dibujan en sus Fig. 2 y 5 (`configs/referencias/yin2012_umbrales_dibujados.csv`), con el residuo de la lectura. No usa ningún fichero de prueba. |

La verificación frente a Yin (`V1_verificacion_yin.csv` y `V1_verificacion_yin_resumen.csv`) todavía no está aquí a propósito: el script se niega a abrir ningún fichero de prueba hasta que la tolerancia, la ventana de la FAR y los supuestos de `configs/yin2012.yaml` estén fijados en una fila fechada de `docs/decisiones.md` con su huella, y hasta que el yaml, el registro, las referencias y el código que se ejecuta (el script y el paquete) estén confirmados en git (párrafo 355: tolerancia "fijada antes de ejecutar"; párrafo 369: "sin ajustes para que coincidan"). Los pasos están en `docs/decisiones.md` y la especificación, en `docs/verificacion_yin.md`.

## Cambios del 12-10-2026 (v0.3)

- A0 y A1 se regeneraron con su configuración de siempre. Las columnas antiguas salen iguales, salvo dos cosas que no vienen de los scripts:
  - `A1_pca_baseline_T2.csv` y `A1_pca_baseline_SPE.csv`: `delay_samples_mean` y `delay_minutes_mean` quedan vacías en los fallos cuya única racha es al azar (`detected_at_chance`), porque `metrics.aggregate` ya no promedia esos retardos (v0.3).
  - `A0_calibracion_limites_yin33.csv`: el `spe_limit` de `en_muestra` difiere en la última cifra escrita. Es el entorno numérico (versiones de numpy y BLAS, en el `.provenance.json`): el paquete sin cambios da lo mismo en el entorno nuevo.
- Columnas nuevas de A1: `method`; las de `evaluate_run` de la v0.3 (`n_far`, `n_fdr`, `n_fdr_h1`, `n_fdr_h2`, `false_alarm_samples_per_hour`, `false_alarm_episodes_per_hour`); en los agregados, `valid_detection_ratio`, `<métrica>_n` y las medianas del retardo; y la configuración en cada fila (`n_components`, `variance`, `alpha`, `lags`, `t2_method`, `spe_method`, `quantile_method`, `calibration`, límites y su origen `t2_limit_source` y `spe_limit_source`, `n_vars`, `n_fit`, `n_cal`, `n_cal_rows`, `k`, `min_fdr`, `delay_convention`, `sample_minutes`, `far_window_normal`).
- Columnas y filas nuevas de A0: los brazos `en_muestra_mismo_ncal` y `jackson` del SPE y el CSV de los brazos del T²; `statistic`, `limit_source`, `n_cal_used`, `quantile_method`, `far_window_d00_te` y `far_window_prefallo`. La columna `n_cal` es, como siempre, el tamaño del bloque reservado (el mismo en todas las filas); cuántos valores calibraron cada límite lo dice `n_cal_used` (vacía en los límites paramétricos).
- La FAR de `d00_te` en A0 y A1 sigue siendo la de la serie entera, ahora declarada en sus columnas `far_window_*`. Qué ventana usa el protocolo es una decisión pendiente (`deteccion.ventana_far_normal` en `configs/baseline.yaml`).

## Columnas añadidas el 22-09-2026

`evaluate_run` devuelve ahora, además de `far`, `fdr`, `delay_samples`, `delay_minutes` y `detected`:

- `detected_at_chance`: `True` cuando hay racha de k alarmas pero la FDR queda por debajo de 0,10. Marca una confirmación que no debe leerse como detección. El 0,10 es una convención del protocolo (`protocolo.min_fdr`).
- `fdr_h1`, `fdr_h2`, `persistence_gap`: FDR sobre la primera y la segunda mitad del periodo con fallo, y su diferencia. Partición fijada a priori, idéntica para todos los fallos, métodos y bloques. Expone los fallos que se detectan y luego se pierden.

`aggregate` añade `at_chance_ratio` y las medias e intervalos de las tres nuevas columnas.
