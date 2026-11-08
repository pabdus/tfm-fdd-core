# LEEME de las tablas de ventajas publicadas

Dos ficheros, fijados antes de los experimentos confirmatorios:

- **`ventajas_publicadas.csv`:** la tabla de ventajas publicadas que pide el párrafo 368 de la memoria. Es la que espera la clave `hipotesis.tabla_ventajas_publicadas` de `configs/baseline.yaml`.
- **`ventajas_citadas_no_contrastables.csv`:** las ventajas que el 4.3 cita pero no contrasta (hoy, las tres filas de Lomov et al., 2021). Están para que la cita sea trazable, con un nombre de métrica que impide contrastarlas.

## Estado (26/10/2026)

- **Los dos CSV los genera `construir_tabla.py`** en `Contexto/04_Investigacion/Revisiones_E2/Tabla_Ventajas_Publicadas_2026-10-15/` y no se editan a mano. Si se corrige alguna transcripción en `fuentes/`, se vuelve a ejecutar y se copian de nuevo.
- **Verificación (15/10/2026):** Pablo comprobó en lectura ciega (`Revisiones_E2/DuquePablo_TFM_Hoja_Verificacion_VentajasPublicadas_15oct.md`) las 54 celdas de 3, 9 y 15 de Diallo et al. (2025), Golyadkin et al. (2023) y Lomov et al. (2021), con las de la FAR que usan: coinciden las 84 cifras. Además dictó completas las Tablas 4, 7, 8 y 9 de Diallo et al. (2025), y las 75 celdas transcritas de ese artículo coinciden con su lectura.
- **Decidido después:** la fila comparadora (18/10, con `hipotesis.baselines`), los nombres de `metric`, la referencia lineal clásica y la tolerancia (26/10; `Revisiones_E2/DuquePablo_TFM_Acta_Decisiones_26oct.md`, secciones 1 a 3). Están abajo.
- **En el repositorio** van este LEEME y los dos CSV, en `configs/referencias/`, con la ruta de `ventajas_publicadas.csv` en `hipotesis.tabla_ventajas_publicadas` y su fila en `docs/decisiones.md`, antes de la etiqueta del prerregistro.

## Qué filas entran en `ventajas_publicadas.csv` (decisión de Pablo del 15/10/2026, 12:39)

Solo las comparaciones dentro del mismo artículo de un método no lineal o profundo frente a un baseline lineal, en los fallos 3, 9 y 15. Con las fuentes leídas hasta hoy son 72 filas:

- **Diallo et al. (2025), 36 filas.** El autoencoder frente al PCA de 12 componentes, con SPE y T², con umbral tradicional y conformal marginal, sobre el conjunto ampliado (Rieth). Son métricas de un detector (FDR y FAR).
- **Golyadkin et al. (2023), 36 filas.** Es un preprint: SensorSCAN y un autoencoder convolucional frente a un PCA con k-means, sobre Rieth y TEPRicker. Van marcadas `otra_metrica`, porque son la TPR y la FPR de un diagnóstico no supervisado, no la FDR y la FAR de un detector.

**Lo que no entra** está en `ventajas_contexto.csv` (543 filas, en la carpeta de origen), con su `tipo_comparacion` y su `motivo_contexto`, y no entra en la condición del párrafo 349:
- los métodos estadísticos de Yin et al. (2012), Russell et al. (2000), Rato y Reis (2013) y Rato et al. (2016);
- las comparaciones entre artículos con Ren et al. (2024) y con Lomov et al. (2021);
- las 12 filas globales de Golyadkin et al. (2023), porque la tabla es por fallo.

## Columnas

- **Las de la tabla:** `articulo`, `metodo`, `baseline`, `conjunto`, `fallo`, `metrica` (`fdr_cruda`, `far` o `exceso_fdr_far`), `valor_metodo`, `valor_baseline`, `ventaja` (método menos baseline, en fracción: 0,01 es un punto), `far_metodo`, `far_baseline`, `tabla_y_pagina`, `verificado`, `tipo_comparacion`, `familia_metodo`, `fuente_baseline`, `configuracion`, `metrica_original` y `nota`.
- **`marca_metrica`:** `misma_metrica` si todas las cifras de la fila son de un detector (FDR, FAR o MDR); `otra_metrica` si no.
- **`origen_metrica`:** `publicada` si la cifra está impresa en el artículo; `derivada` si la calcula el script (el exceso de la FDR sobre la FAR).
- **`papel_349`:** `comparadora` en las tres filas con que se evalúa la condición del párrafo 349 (ver abajo); `sensibilidad` en las de las otras tres comparaciones de Diallo en exceso; vacía en el resto.
- **Las que lee `tfmfdd.stats.published_advantage_test`:** `article`, `metric`, `method`, `baseline`, `value`, `page`, `comparison` ("dentro del articulo" o "entre articulos"), `fault`, `dataset`, `baseline_article` y `verified`.
  - `metric` sigue la regla del 26/10/2026: en las filas de la misma métrica, el nombre de la columna de resultados del paquete (`fdr`, `far`, `exceso_far_gemela`); en las de otra métrica, un nombre propio que ninguna columna de resultados tiene (Golyadkin: `tpr_diagnostico`, `fpr_diagnostico`, `exceso_tpr_fpr_diagnostico`), para que la función se niegue a contrastarlas. La columna `metrica` conserva los nombres de la tabla (`fdr_cruda`, `far`, `exceso_fdr_far`).
  - `value` lleva el signo de esa función: positivo si el método es mejor.
  - En la FAR, `value` es baseline menos método, al revés que `ventaja`.
  - En `ventajas_citadas_no_contrastables.csv`, `metric` es `tpr_clase_clasificador` (ver abajo).

## Las dos referencias de la ventaja publicada en 3, 9 y 15

Pablo decidió el 15/10/2026 (12:39) tomar las dos y declarar cada una por lo que es. A las 17:18 aceptó una corrección: la condición del párrafo 349 solo puede evaluarse en la misma métrica, así que se evalúa frente a Diallo y no frente a Lomov.

- **Diallo et al. (2025): la referencia contrastable.** Está dentro del mismo artículo, en la misma métrica y sobre el mismo conjunto, con los dos métodos en el suelo. Está en `ventajas_publicadas.csv`.
- **Lomov et al. (2021): la referencia citada, no contrastada.** Es la ventaja que la literatura cita como grande, pero no tiene con qué compararse en la misma métrica. Difiere en cuatro cosas:
  - **la tarea:** un clasificador supervisado de 21 clases, entrenado con ejemplos de cada fallo, frente a un detector entrenado solo con operación normal;
  - **la métrica:** la TPR por clase, frente al exceso de la FDR sobre la FAR;
  - **la FAR del baseline:** el PCA [53] no la publica (guion en la Tabla 7), así que no hay exceso;
  - **el conjunto:** la GRU está medida en Rieth y el PCA [53], en el TEP clásico, por otro trabajo.
- **Qué filas de Lomov:** las tres que publica el propio artículo para su GRU de tipo 2 con todos los pasos frente al PCA [53] (Tabla 7, p. 12), que son las de la frase del 4.3. Están en `ventajas_citadas_no_contrastables.csv`, con `metric = tpr_clase_clasificador`. Ninguna columna de resultados del paquete se llama así, de modo que `published_advantage_test` se niega a contrastarlas (comprobado con `exceso_far_gemela`, `fdr` y `exceso_fdr_far`). No hay que renombrarla. Las demás filas de Lomov (60 pasos, TCNN1D2D-GAN y las construidas con el PCA de Diallo) quedan en `ventajas_contexto.csv`.
- **Cómo se usa:** la condición "menor que la publicada" se evalúa frente a Diallo con `published_advantage_test` y se reporta como información. El veredicto sigue la regla del párrafo 368, que es lo que hace `hypothesis_verdict`: no evalúa esa condición.

## Qué fila de Diallo es la comparadora (regla aceptada por Pablo el 15/10/2026, 17:18; la fila, fijada el 18/10/2026)

- **La regla:** la comparadora es la comparación de Diallo que corresponde al baseline PCA de la familia, con umbral tradicional (KDE en el autoencoder) y el mismo estadístico que fije `hipotesis.baselines`. Las otras tres comparaciones se reportan como sensibilidad.
- **La fila:** con `hipotesis.baselines` = SPE (acta del 18/10, sección 11), es el autoencoder con KDE y umbral tradicional frente al PCA de 12 componentes con SPE y umbral tradicional (`D25-1` en `fuentes/comparaciones.csv`), en exceso: `papel_349 = comparadora`, tres filas (fallos 3, 9 y 15), con `metric = exceso_far_gemela`. Las del T² con umbral tradicional y las dos con el umbral conformal marginal llevan `papel_349 = sensibilidad`.
- **La métrica:** la de la hipótesis, que es el exceso. La fila es la derivada, con `VERIFICAR`; la FDR cruda queda como cifra publicada de contexto.
- **Por qué fijarla antes de la etiqueta:** escoger después entre cuatro comparaciones sería la misma selección a posteriori que el trabajo critica en la literatura.
- **Dos límites:** si el baseline es la regla "T² o SPE", Diallo no tiene equivalente y se dice así. Y como Diallo no tiene DPCA, la condición del párrafo 349 solo existe para las tres comparaciones con PCA de la familia.

## Cómo se pasa una fila a `published_advantage_test` (decisiones del 26/10/2026)

- **La métrica se llama igual.** La fila se pasa tal cual (`published=fila`) y la función exige que `metric` coincida con la métrica de la ventaja medida. Con la regla de `metric`, la fila comparadora se contrasta con la ventaja medida en `exceso_far_gemela` (`hipotesis.metrica_ventaja`).
- **Las dos FAR del exceso.** El exceso de Diallo resta la FAR de sus simulaciones normales (alfa del 1 % inferido, no impreso); el de este trabajo, la FAR de la gemela en `deteccion.ventana_far_normal`. Se tratan como la misma cantidad estimada de dos formas: las dos son la FDR por encima de la FAR en operación normal. Lo dice la `nota` de esas filas, y lo dirá el 4.3.
- **Golyadkin y Lomov no se contrastan.** La función no mira `marca_metrica`, así que lo que impide contrastarlas es su nombre: Golyadkin lleva `tpr_diagnostico`, `fpr_diagnostico` y `exceso_tpr_fpr_diagnostico`, y Lomov, `tpr_clase_clasificador`. No renombrarlas.
- **Sin tolerancia.** Se usa la función tal cual: una ventaja medida igual a la publicada queda "inconclusa". La cifra principal que se reporta es la diferencia entre la publicada y la medida, `published_minus_measured`, con su intervalo (`gap_ci_low`, `gap_ci_high`); la etiqueta (`outcome`) la acompaña. Con ±δ, toda ventaja medida entre unos -1,9 y +2,2 puntos contaría como "igual"; una tolerancia más pequeña no tendría de dónde salir, porque Diallo no publica el error de su ventaja.

## El exceso de Diallo et al. (2025) (decisión de Pablo del 15/10/2026, 12:39, con la evidencia de las 17:18)

- **Por qué no es una cifra impresa:** las Tablas 7 y 9 no declaran el alfa, y el exceso usa la FAR a α = 1 % (Tablas 4 y 8).
- **Qué dice la evidencia interna** (`verificacion/evidencia_alfa_diallo2025.py`, sobre las celdas que Pablo dictó):
  - en los fallos 3, 9 y 15, la FDR de cada uno de los cinco umbrales queda, en media, de 0,04 a 0,37 puntos de la FAR al 1 % y de 3,57 a 3,87 puntos de la FAR al 5 %;
  - la FDR queda por debajo de la FAR al 5 % en 45 de 45 celdas: con α = 5 %, el detector alarmaría menos con fallo que sin él;
  - el orden de los cinco umbrales por FDR es el de su FAR al 1 % (Spearman 1,00 en los 9 casos; con la FAR al 5 %, de 0,80 a 0,90);
  - el mismo grupo, con el alfa impreso (Diallo et al., 2026, Tabla 4, el mismo PCA sobre Rieth), da de 1,25 a 1,62 % con α = 1 % y de 5,88 a 6,25 % con α = 5 %.
- **Su límite:** el argumento supone que 3, 9 y 15 no son detectables para estos métodos, que es en parte lo que pone a prueba la hipótesis. Por eso es una inferencia y no una lectura.
- **Cómo queda:** como `VERIFICAR`, con la nota "alpha 1 % inferido y no impreso". La cifra publicada es la FDR cruda (`origen_metrica = publicada`) y el exceso es derivado.
- **Los autores:** no se les escribe antes de la etiqueta. Son los únicos que pueden confirmar el alfa.

## Texto para el 4.3 (corregido el 15/10/2026, 17:18)

> La ventaja publicada en los fallos 3, 9 y 15 se toma de dos referencias, que se declaran cada una por lo que es. Diallo et al. (2025) dan la única comparación de un autoencoder con un PCA dentro de un mismo artículo, en la misma métrica y sobre el mismo conjunto ampliado del TEP: los dos métodos están en el suelo, y la ventaja del autoencoder en la tasa de detección no llega a medio punto (como mucho, 0,48 puntos). Lomov et al. (2021) dan la ventaja que la literatura cita como grande, de 93,90 a 99,90 puntos, pero entre artículos, en otra tarea y en otra métrica: la tasa de aciertos por clase de un clasificador supervisado, entrenado con ejemplos de cada fallo, frente a la cifra de un PCA de otro trabajo, medida en el TEP clásico y publicada sin tasa de falsas alarmas. Como la comparación solo se hace en la misma métrica, la condición de que la ventaja medida sea menor que la publicada se evalúa frente a Diallo et al. (2025) y se reporta como información; Lomov et al. (2021) se cita para situar la cifra que circula en la literatura, pero no se contrasta. El veredicto sigue la regla de no superioridad. Frente a Diallo et al. (2025), la ventaja publicada ya es menor que la relevante, así que esa condición no discrimina: decide sobre décimas de punto que la propia diferencia mínima relevante declara sin importancia.

**Añadido el 26/10/2026 (propuesta para el mismo párrafo, por las decisiones de esa fecha):**

> El exceso que publican Diallo et al. (2025) resta la tasa de falsas alarmas de sus simulaciones normales, y el de este trabajo, la de la simulación normal gemela de cada simulación con fallo: las dos son la tasa de detección por encima de la de falsas alarmas en operación normal, estimada de dos formas. Para esa condición se reporta la diferencia entre la ventaja publicada y la medida, con su intervalo, y no solo si es menor.

Las cifras salen de `verificacion/cifras_frase_4_3.txt`, que escribe `construir_tabla.py`.

**Qué se corrigió.** La versión del 15/10 a las 12:39 decía que la condición "se evalúa frente a cada una". Frente a Lomov no puede evaluarse sin romper la regla de la misma métrica, y la función, con razón, se niega. La versión corregida evalúa la condición solo frente a Diallo, cita a Lomov sin contrastarlo y añade lo que lo separa de la hipótesis: otra tarea y ninguna tasa de falsas alarmas del baseline.

**Sobre la última frase.** La decisión decía que con Diallo "no se puede cumplir 'menor'". Pero `published_advantage_test` no usa δ: compara la ventaja medida con la cifra publicada. Con una ventaja medida casi nula y poco variable, "menor que la publicada" sí sale. Se comprobó con una ventaja simulada de media 0 y desviación 0,003 en 100 simulaciones: frente a los 0,08 puntos de Diallo en exceso da "menor que la publicada" en 831 de 1.000 repeticiones, y "inconclusa" en las demás. Por eso el texto no dice que la condición no se pueda cumplir: dice que no discrimina. Se mantiene, tal cual, que la ventaja publicada dentro del mismo artículo ya es menor que la relevante.
