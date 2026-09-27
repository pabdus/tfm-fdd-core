# Estado de los cuadernos de arranque B00 y C00 — 27 de septiembre de 2026

Pablo: este es el cierre de los dos cuadernos punto de partida que se escribieron hoy, el `B00` para Jonathan y el `C00` para Cesar. Los ejecuté de principio a fin en limpio, los revisé contra las once reglas de medida (ahora en `docs/reglas_de_medida.md`) y contra los veredictos de las dos verificaciones de la revisión, corregí lo que fallaba y los volví a ejecutar. Todas las cifras de este documento salen de las salidas guardadas en los cuadernos o de mis ejecuciones, y digo de dónde en cada caso.

## 1. En pocas palabras

- **Los dos corren sin errores** con `.venv/Scripts/python.exe -m nbconvert --execute`, desde la raíz del repositorio. Entre ejecuciones, todas las salidas de texto coinciden; solo cambian los tiempos.
- **Cumplen las once reglas.** No encontré ninguna fuga ni ninguna métrica mal calculada. Lo que corregí son matices que la verificación del 27/09 añadió a las reglas 1, 3, 4 y 6, una lista de fallos escrita a mano en los dos cuadernos (regla 10) y tres afirmaciones inexactas sobre la instalación y el paquete (sección 5).
- **Tiempo:** en mis tres ejecuciones, el B00 tardó entre 2,6 y 10,5 minutos y el C00, entre 0,3 y 1,7, según lo cargado que estuviera el equipo (sección 3). La versión guardada tardó 6,4 y 1,7 minutos.
- **Lo que bloquea que Jonathan y Cesar los usen:** los dos cuadernos, y todo lo que corrigen, solo existen en tu rama local `revision-2026-09`, sin commit. En GitHub no están. Hasta que publiques la `v0.3` (apartado 5 de la revisión), no los pueden recorrer como está previsto.

## 2. Qué contiene cada cuaderno

**`notebooks/B00_punto_de_partida_diagnostico.ipynb` (Jonathan, OE2).** Sustituye al `02` como punto de partida. Recorre el conjunto etiquetado del TEP clásico, con la comprobación en `d06.dat` de que el fallo está activo desde la fila 0 (B2), el escalado sin fuga y kNN, SVM y Random Forest con los valores por defecto escritos. Después mide bien: F1 por clase, macro-F1, matriz de confusión y la demostración de que el «f1» del `02` era 2r/(1+r) (B3). Los fallos 3, 9 y 15 van aparte. Luego viene el OE2 en miniatura sobre Rieth, partición por muestra frente a partición por simulación, con dos diseños: ids distintos por clase (7.1) e ids compartidos (7.2, la trampa de las gemelas). También mide de dónde sale el vecino más cercano, el inicio efectivo del fallo (nuevo) y el formato común en `results/raw/B00_*.csv`. Evalúa con los ficheros de prueba de Rieth, usa el clasificador como detector, hace un contraste kNN frente a RF y termina con el «Te toca»: FCM, KFCM, limpieza de atípicos, SVM ajustada por validación por simulación y escalado a `runs_finales`.

**`notebooks/C00_punto_de_partida_deeplearning.ipynb` (Cesar, OE3).** Sustituye al `03`. Parte del autoencoder como primo no lineal del SPE (con la comprobación numérica) y de la partición del clásico en entrenar, parar y calibrar. Entrena un autoencoder de detección solo con normales, con parada temprana, y calibra su umbral con normales reservados. Lo evalúa en los 21 fallos junto al PCA-SPE, con el retardo filtrado a mano. Mide las dos trampas del umbral (en muestra y F1 máximo sobre la prueba) contra un detector trivial. En Rieth usa 20 simulaciones: FAR en `test_fault00`, intervalos, Wilcoxon con Holm, la trampa (a) con incertidumbre y el inicio efectivo del fallo (nuevo). Hace un diagnóstico con ventanas que no cruzan trayectorias (LSTM de PyTorch) y lo guarda en el formato común (`results/raw/C00_*`). Sigue con lo que hay que replicar de Lyu et al. (2026), con páginas del PDF y [VERIFICAR], y termina con el «Te toca»: autoencoder profundo, red recurrente, réplica de Lyu y tabla de atribución.

## 3. Ejecución y tiempos

Todas las ejecuciones son `.venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800`, desde la raíz del repositorio y una detrás de otra. El tiempo de nbconvert lo medí con `date`; el de la celda final lo imprime el propio cuaderno.

| Ejecución | Versión | B00: nbconvert / celda final | C00: nbconvert / celda final | RAM libre |
|---|---|---|---|---|
| 1 (12:06) | la de los autores | 163 s / 2,6 min | 25 s / 0,3 min | 5,1 GB al empezar |
| 2 (12:15) | corregida | 633 s / 10,5 min | 100 s / 1,6 min | 1,3 GB a mitad del B00 |
| 3 (12:28) | corregida, definitiva (la guardada) | 388 s / 6,4 min | 110 s / 1,7 min | 3,7 y 3,4 GB al empezar; 1,7 GB a mitad del B00 |

Después de la ejecución 3 solo cambié dos frases de markdown en celdas nuevas (sin tocar código), así que las salidas guardadas son las de esa ejecución. Todas sus salidas de texto, y los diez ficheros de `results/raw/`, coinciden con los de la versión de los autores; solo cambian los tiempos (la columna `segundos` de `B00_rieth_particion.csv` y la fecha de `C00_configuracion.json`).

La diferencia de tiempos la explica la carga del equipo, no el cuaderno. En las ejecuciones 2 y 3 había otros programas abiertos (Chrome ocupaba varios GB) y la predicción del kNN sobre la prueba de Rieth pasó de 15,9 s a 116,4 s y 115,7 s (salida de la celda de la sección 8 del B00). Es lo mismo que le pasó al autor del B00: 721 s con entre 0,3 y 0,8 GB libres. Con el equipo descargado, el B00 queda muy por debajo del objetivo de 10 minutos; con el navegador abierto, se acerca o lo pasa.

## 4. Checklist contra las reglas de medida

«Cumple» quiere decir que lo comprobé leyendo el código de las celdas y sus salidas; «corregido», que fallaba en algún matiz y ya está arreglado.

| Regla | B00 | C00 |
|---|---|---|
| 1. Partición por simulación | Cumple. Clásico: `dXX.dat` frente a `dXX_te.dat`. Rieth: `split_runs` por posición, los mismos ids en todas las clases; la partición por muestra solo aparece como brazo (a) del experimento. Corregido: la regla ahora menciona las gemelas y `groups = simulationRun` | Cumple. Clásico: corte temporal de `d00.dat`; Rieth: `split_runs(500)` una vez, los mismos ids en todas las clases; ventanas por (fallo, simulación) |
| 2. Escalado solo con entrenamiento | Cumple (`Scaler().fit` con el entrenamiento de cada partición). La fuga solo está en un «Verifícalo tú» comentado | Cumple (`X_fit`, `XR_fit`, `diag_tr`) |
| 3. Umbrales con normales reservados | No aplica a los clasificadores, que no tienen umbral. El detector usa una alarma 0/1 con límite 0,5. Corregido: avisos de REP-4 y DIS-6 y simulaciones ya miradas | Cumple: umbral en `Z_cal`, parada temprana con `Z_es`, PCA con `X_cal`; las trampas van rotuladas como trampas. Corregido: matices REP-4, I4, DIS-6 y MET-3, y la nota sobre Jackson en la sección 1 |
| 4. Detección con `evaluate_run` | Cumple: onset 160, `k` del yaml, retardo filtrado a mano, sin `summary`, FAR solo de las normales. Corregido: la FDR se lee frente a la FAR | Cumple: igual, más `retardo_interpretable` y FAR en `test_fault00`. Corregido: la misma nota |
| 5. F1 por clase sobre toda la prueba | Cumple (`precision_recall_fscore_support` sobre toda la prueba y sobre el pool de Rieth). El F1 restringido solo aparece como demostración de B3 | Cumple. El F1 restringido solo está en un «Verifícalo tú» comentado, marcado «¡NO HAGAS ESTO!» |
| 6. Inicio del fallo | Cumple: `fault_onset("train", "classic")` es `None` y se usan las 480 filas; Rieth, 20/160. Corregido: el inicio efectivo (DAT-1) se mide en dos celdas nuevas | Cumple (con un `assert` en el camino clásico). Corregido: celda nueva sobre el inicio efectivo |
| 7. 3, 9 y 15 aparte | Cumple en todas las tablas | Cumple en todas las tablas |
| 8. Contrastes | Cumple: `compare_methods`, α declarado y comprobado con `inspect`, diferencia antes que p, sin `effect_size`, advertencia de desarrollo | Cumple: `compare_methods` y `paired_test`, `median_difference`, α declarado |
| 9. Semillas y configuración del yaml | Cumple: semilla, `k`, fracciones y `runs_*` del yaml; `ALPHA_CONTRASTES` declarado porque no está en el yaml (I2) | Cumple: semilla, nivel, `k`, `calibracion_limites`, fracciones, `variance` y `spe_method` del yaml; se registran los hilos de PyTorch |
| 10. Sin números de resultado a mano | Corregido: la lista «(8, 10, 13, 17, 18 y 20)» de la sección 8 venía del informe de datos y se aplicaba a los ficheros de prueba, donde no es exacta. Ahora la mide una celda. El resto de números en markdown son hechos de los datos, secciones, fechas, parámetros del protocolo o el caso de juguete de la 5.3 | Corregido: la misma lista, llamada «fallos aleatorios» (el 13 es una deriva lenta y el 17, el 18 y el 20 son de tipo desconocido). Las cifras de Lyu de la sección 9 son citas con página; comprobé las principales en el texto extraído del PDF |
| 11. Formato común | Cumple: `B00_rieth_por_simulacion.csv` con una fila por (method, faultNumber, simulationRun) y el F1 por clase en fichero aparte; todo en `results/raw/` | Cumple: los mismos formatos, un `assert` contra filas repetidas y `C00_configuracion.json` con versiones e ids |
| Idioma | Español neutro con tuteo; no encontré voseo ni «vosotros» (búsqueda con expresiones regulares en todas las celdas) | Igual |
| `FAULT_ONSET_TRAIN_CLASSIC`, f1 restringido, `summary` | No se usan en el código (búsqueda en las celdas de código) | No se usan |
| Ficheros fuera de lo autorizado | `git status --porcelain`: solo los dos cuadernos y los dos documentos nuevos, además de lo que ya estaba. Ningún fichero ya existente cambió después de la revisión de las 09:20 | Igual |

## 5. Qué corregí

En los dos cuadernos:
1. **Regla 6 (DAT-1).** Nuevo matiz: el inicio nominal de Rieth no siempre es el efectivo. En el B00 hay dos celdas nuevas, una tras la 7.2 (ficheros de entrenamiento) y otra en la sección 8 (ficheros de prueba), y en el C00, una en la sección 6. Todas usan la misma función, `separacion_de_la_gemela`, que cuenta cuántas muestras sigue cada simulación idéntica a su gemela normal después del inicio. Sustituyen a la lista escrita a mano. Crucé el resultado con el análisis del agente de datos (`v2_resultado.json`, ficheros de entrenamiento, simulaciones 1 a 20): el retraso mínimo y el máximo coinciden en los 20 fallos (por ejemplo, en el fallo 18, de 3 a 71 muestras).
2. **Regla 3.** Matices de la verificación. `PCAMonitor` sin `X_cal` calibra en muestra sin avisar (REP-4). El límite F del T² no usa `X_cal` (I4). Jackson es una calibración en muestra implícita (DIS-6), y en el C00 se corrigió además la frase de la sección 1 que lo presentaba como límite paramétrico sin más. El percentil empírico con pocas muestras sube la FAR (MET-3), también en el «Cómo leerlo» de la sección 4 del C00. Por último, las simulaciones de Rieth ya miradas no sirven para lo confirmatorio (DIS-17), con un aviso de «todo esto es desarrollo» en la sección 7 del B00 y en la 6 del C00.
3. **Regla 4.** La FDR se lee frente a la FAR (`detected_at_chance` solo mira la FDR), y en Rieth la FAR se mide solo en las normales.
4. **Regla 1 (B00).** Gemelas y `groups = simulationRun`.

Solo en el C00:
5. **Instalación (B1, matizado por la verificación del repositorio).** Decía «no se puede instalar por etiqueta». Lo exacto es que `v0.2` no existe en GitHub y que `v0.1` sí se instala, pero entrega el paquete del 17/09, sin `convert_rieth` y con el `compare_methods` antiguo. Lo cambié en la portada y en la sección 12, y añadí que `configs/baseline.yaml` no viaja con el paquete (REP-4).

Solo en el B00:
6. **Tabla «Qué cambia con v0.3».** La fila M6 decía que faltaba un parámetro `root`, y `load_classic` y `load_rieth` ya lo tienen (matiz de la verificación). Añadí filas para REP-4, el inicio efectivo y la memoria de `load_rieth` (DAT-4).
7. **«Te toca».** Añadí dos decisiones a la lista para `decisiones.md`: qué hacer con el tramo del inicio efectivo, y qué ids quedan para lo confirmatorio, que deben ser los mismos en los tres bloques.

Los dos llevan en la portada una línea que remite a `docs/reglas_de_medida.md` y a este documento. No cambió ninguna salida de las celdas que ya existían: las comparé una a una, sin los tiempos, entre la versión de los autores y la corregida.

## 6. Cómo los recorren Jonathan y Cesar

**Primero tienes que publicar el código.** Hoy no hay nada que puedan instalar ni clonar con estos cuadernos. GitHub tiene `v0.1` (el paquete del 17/09) y `main` (la subida web del 22/09), y ninguno de los dos trae B00, C00 ni las correcciones. El orden está en el apartado 5 de la revisión: commit en `revision-2026-09`, fusión con `origin/main` sin forzar, `git rm --cached` de `__pycache__`, sacar los dos zip de la raíz y fijar `license-files = ["LICENSE"]` en `pyproject.toml`. La verificación del repositorio comprobó que, si no, setuptools mete `LICENSE.zip` (unos 104 MiB) en la rueda como fichero de licencia. Después, PR a `main`, la etiqueta `v0.3` y el push de la etiqueta.

**Después, cada uno en su equipo:**
1. Clonar el repositorio en la etiqueta (`git clone https://github.com/pabdus/tfm-fdd-core.git` y `git checkout v0.3`). Para recorrer estos cuadernos hace falta el repositorio entero: leen `configs/baseline.yaml` y `data/` desde la raíz, y el yaml no viaja con el paquete.
2. Crear un entorno con Python 3.12 e instalar en modo editable: `python -m venv .venv`, activarlo (en Git Bash, `source .venv/Scripts/activate`; en PowerShell, `.venv\Scripts\Activate.ps1`) y `python -m pip install -e ".[dev]"`. Cesar, además, PyTorch para CPU (`python -m pip install torch --index-url https://download.pytorch.org/whl/cpu`); sin él, el C00 usa scikit-learn y corre igual, con otras cifras. Para Jonathan: `scikit-fuzzy` no está en las dependencias, así que el FCM va en numpy o hay que añadirla.
3. Datos. El TEP clásico (44 `.dat`) en `data/tep_classic/`: no hay script de descarga, así que se lo pasas tú. Rieth, mejor los 42 Parquet ya convertidos que su conversión en un portátil de 16 GB: pásaselos con el manifiesto SHA-256 (el agente de datos dejó uno en su carpeta de trabajo, `inventario_sha256.txt`, que conviene llevar al repositorio o a la carpeta compartida). Que los guarden fuera de OneDrive y Google Drive, en `~/datos_tfm/tep_rieth`, o que definan `TFM_RIETH_DIR` antes de abrir Jupyter. Sin Rieth, los dos cuadernos se saltan esas partes con un mensaje y el resto corre.
4. Orden: `01_primeros_pasos.ipynb` primero (los dos) y después el suyo, `B00` o `C00`. Primero *Run All* para comprobar que corre, y después celda por celda. Para ejecutarlo entero desde la terminal: `python -m nbconvert --to notebook --execute ...` con el Python del entorno, nunca un `jupyter` que resuelva a otro Python (I7). Mientras corre el B00, conviene cerrar el navegador (sección 3).
5. Su trabajo va en su propio repositorio, con `tfmfdd @ git+https://github.com/pabdus/tfm-fdd-core@v0.3` en `requirements.txt` (nunca `@main`) y una copia de `configs/baseline.yaml` de la misma versión.

## 7. Qué les queda («Te toca»)

**Jonathan (B00, sección 9).** Las celdas tienen un interruptor `HACER_... = False` para que el cuaderno corra entero mientras implementa:
- 9.1, un FCM supervisado (centroides por clase y pertenencia de Bezdek, con m fijado antes de mirar).
- 9.2, el KFCM (el ancho del núcleo, con las simulaciones de validación, nunca con la prueba).
- 9.3, la limpieza de atípicos con Noise Clustering o DOFCM (solo en el entrenamiento de cada partición, contando cuántas muestras quita por clase y simulación).
- 9.4, la SVM ajustada con `GroupKFold` por `simulationRun` y el escalador dentro de la `Pipeline`.
- 9.5, el paso a `runs_finales`, con la cuenta de memoria ya hecha en la celda.

Referencias sin verificar, todas marcadas [VERIFICAR]: Prieto-Moreno 2013, Bezdek 1981, Zhang y Chen 2003, Rodríguez-Ramos 2017, Davé 1991, Kaur y Gosain 2010, Karadeniz 2026 y Rieth 2017/2018.

**Cesar (C00, sección 10).** Primero la réplica de Lyu (TODO 3: clonar `KitchinHUB/tep-manuscript`, sacar los ids de sus cuadernos 01 y reproducir su Tabla 6). Después, su autoencoder profundo (TODO 1) y la red recurrente de verdad (TODO 2), y al final la tabla de atribución por componentes (TODO 4), con el orden de los brazos declarado antes de medir. Quedan cinco [VERIFICAR] de Lyu en la sección 9 que se resuelven en su código. Dos de ellos son si la validación de sus autoencoders tenía fallos y si su prueba binaria comparte ids entre normales y fallos.

## 8. Decisiones que tienes que llevar a `decisiones.md` antes de lo confirmatorio

1. **Brazo (a) del OE2.** Con ids distintos por clase, que aíslan la autocorrelación, con ids compartidos, que reproducen la práctica habitual y mezclan el efecto de las gemelas, o con los dos. Los dos diseños dan signos distintos para el kNN (tabla de diferencias de la 7.1 y de la 7.2 del B00).
2. **Ids confirmatorios de Rieth**, fijados por un procedimiento escrito antes de ejecutar y los mismos en los tres bloques. La lista de ids ya mirados está en `docs/reglas_de_medida.md` (regla 3). Ojo: hoy el B00 prueba con las simulaciones 1 a 10 de los ficheros `test_*` y el C00, con las 20 primeras de `split_runs(500)`, así que sus CSV no se pueden emparejar por (fallo, simulación).
3. **Unidad de evaluación del diagnóstico:** el recall por (fallo, simulación) y el F1 por clase sobre el pool (lo que ya hacen los dos cuadernos).
4. **Escalado en diagnóstico:** con todas las clases (lo que hacen hoy) o solo con la normal.
5. **Tramo del inicio efectivo:** declararlo como ruido de etiqueta o descartarlo, igual en los tres bloques.
6. **Diferencia mínima relevante** (propuesta: 2 puntos de FDR o FAR y 1 muestra de retardo), la sensibilidad de `min_fdr` a 0,05, 0,10 y 0,15 (I8) y una marca de «detección al azar» que tenga en cuenta la FAR (hallazgo del B00: un clasificador usado como detector pasa `detected_at_chance` con una FAR altísima).
7. **Hilos de PyTorch:** fijarlos en la configuración, porque cambian el resultado del C00 (el autor lo midió en la LSTM de diagnóstico).

## 9. Pendientes que dependen de la `v0.3`

- **Publicación (B1, REP-5, I9):** commit, fusión, etiqueta y push; `license-files`; sacar los zip; `__pycache__` fuera del índice; versión `0.3.0` en `pyproject.toml` y en `tfmfdd/__init__.py` (M1); `requirements-lock.txt` (I7, M2).
- **Paquete:**
  - Quitar `FAULT_ONSET_TRAIN_CLASSIC` con una prueba que lea los datos (B2) y una función de diagnóstico con F1 por clase (B3).
  - `summary` sin los retardos al azar (I1).
  - `compare_methods` con `baseline=`, intervalo y `r` con signo, y `alpha_contrastes` en el yaml (I2).
  - `t2_method` para el brazo de Fase I (I4).
  - El inicio del DPCA (MET-1, MET-2).
  - `load_rieth` con `int64`, aviso de ids inexistentes y lectura filtrada (DAT-4 a DAT-6).
  - Que el protocolo viaje con el paquete o que los scripts fallen si no encuentran el yaml (REP-4).
- **Cuadernos:** cuando todo eso exista, las vueltas de la tabla «Qué cambia con v0.3» del B00 y de la sección 12 del C00 se pueden quitar. Los dos llevan el aviso «Escrito sobre la rama `revision-2026-09`», que habrá que actualizar a la etiqueta.
- **Documentación que no toqué** (no estaba autorizado):
  - `notebooks/README.md` sigue presentando el `02` y el `03` como puntos de partida.
  - `README.md` sigue diciendo `@v0.2`.
  - `docs/decisiones.md:18` sigue con `FAULT_ONSET_TRAIN_CLASSIC = 20`.
  - El traspaso tiene las cifras de Lyu que el C00 corrige (87,57 % es de TransKal, no de un autoencoder; corridas por partición según el PDF, pp. 8-9).

## 10. Cifras que conviene que mires

Son salidas de los cuadernos ejecutados hoy (orientativas, de desarrollo, no resultados de la memoria):
- **B3 con datos** (B00, sección 5.3): el «f1» viejo coincide exactamente con 2r/(1+r) e infla la media del Random Forest de 0,6311 a 0,7256.
- **La partición por muestra no tiene signo fijo** (B00, 7.1 y 7.2). En macro-F1, (a) menos (b): con ids distintos, +0,069 en el RF y +0,047 en el kNN; con ids compartidos, +0,020 en el RF y −0,056 en el kNN. Frente a la prueba independiente (sección 8), las estimaciones (b) quedan cerca y las (a) se mueven con el diseño.
- **Un clasificador no es un detector** (B00, sección 8): usado como detector, la FAR media sobre las simulaciones normales de prueba es 0,889 en el RF y 0,324 en el kNN, y aun así `detected_at_chance` no marca nada.
- **Trampa (a) en el clásico** (C00, sección 5): FAR previa al fallo del PCA-SPE, 0,274 en muestra frente a 0,036 con reserva (coincide con A0); del autoencoder, 0,438 frente a 0,053. **Trampa (b):** el F1 del umbral óptimo sobre la prueba solo supera al del detector trivial en +0,008 (AE) y +0,005 (PCA).
- **Inicio efectivo** (celdas nuevas). En los fallos 1 a 7, 14, 15 y 19, todas las simulaciones se separan de su gemela en la primera muestra con fallo. En los demás, el retraso llega a 71 muestras (fallo 18, entrenamiento, B00 7.2) y a 84 (fallo 18, prueba, C00 sección 6). La fracción de filas con etiqueta de fallo idénticas a la clase 0 es como mucho 0,067 en entrenamiento (fallo 18, B00) y 0,044 en prueba (fallo 18, C00). Es ruido de etiqueta pequeño, pero no nulo, y concentrado en los fallos 10, 13, 17, 18 y 20.
