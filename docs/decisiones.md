# Registro de decisiones

Una linea por decision: fecha, que se decidio, quien y por que. Se incluyen las
que tome el director. Este fichero evita discutir dos veces lo mismo y sirve de
respaldo si algo se cuestiona en la defensa.

| Fecha | Decision | Quien | Motivo |
|---|---|---|---|
| 2026-09-16 | Tres repositorios, uno por estudiante, con `tfmfdd` instalado como dependencia | Equipo | Las instrucciones de UNIR exigen que cada repositorio tenga un unico autor |
| 2026-09-16 | TEP clasico para prototipar y verificar; TEP-Rieth para los resultados definitivos | Pablo | El clasico pesa 26 MB y permite contrastar con Yin (2012); Rieth da intervalos de confianza |
| 2026-09-16 | Los limites de control se calibran con datos normales RESERVADOS, no con los del ajuste | Pablo | Calibrar con los datos del ajuste dio 12,5 % de falsas alarmas frente al 2,3 % con reserva. **[2026-10-12: motivo anulado.]** Las cifras 12,5 y 2,3 salieron de un experimento confundido y de configuraciones distintas (fila del 2026-09-22 sobre el experimento de calibracion; traspaso v12, 9.3) y no se citan. La decision sigue en pie porque la fija la memoria (parrafos 348 y 367), no por esas cifras. |
| 2026-09-22 | El experimento de calibracion se corrige: mismo ajuste (X_fit) para los dos modelos, solo cambia el limite; la cifra publicable sale de `experiments/A0_calibracion_limites.py` con su configuracion en el CSV | Pablo | La version anterior cambiaba a la vez tamano de ajuste y calibracion (confundido); las cifras 12,5/2,3 (17 comp.) y 19,9/3,6 (27 comp.) son de configuraciones distintas y quedan superadas |
| 2026-09-22 | Ningun numero se reporta sin su configuracion (variables, componentes, alpha, metodo del SPE, n_fit, n_cal); los docstrings apuntan al CSV, no citan cifras | Pablo | Dos corridas con distinta configuracion acabaron citadas como la misma |
| 2026-09-22 | `configs/baseline.yaml` es la unica fuente de los valores por defecto; los scripts lo leen | Pablo | El script A1 usaba 0,90 de varianza mientras el yaml decia 0,85 |
| 2026-09-22 | Metricas nuevas del protocolo: `detected_at_chance` (marca rachas con FDR < 0,10) y FDR estratificada por mitades (`fdr_h1`, `fdr_h2`, `persistence_gap`), calculadas igual en los tres bloques | Pablo, propuesta a validar con el equipo | Fallo 3: racha por azar en la muestra 537 con FDR 3,75 %; fallo 5: FDR 0,29 que promedia 0,95 y 0,07. **[2026-10-12: motivo superado.]** Esas cifras no se citan: el 0,95 y el 0,07 del fallo 5 salen de un corte elegido a posteriori (traspaso v12, 10.4), y 537 es el retardo del SPE en el fallo 3, no la muestra donde empieza la racha (informe de la revision del codigo del 12/10, imp_metricas, H8). Las metricas siguen; el 0,10 esta ahora en configs/baseline.yaml (protocolo.min_fdr), pendiente de confirmar (ver la ultima seccion). |
| 2026-09-22 | Contrastes: alpha = 0,05 declarado; correccion de Holm por fallo entre los k-1 pares contra el baseline (analisis principal) y todos contra todos como secundario; Wilcoxon con `zero_method="zsplit"`; se reporta la diferencia con intervalo, no solo el p | Pablo, pendiente de acordar la diferencia minima relevante | Con 100 simulaciones casi todo sale significativo; sin diferencia minima declarada se cae en el mismo error que se denuncia |
| 2026-09-22 | El nivel de confianza del limite de control (0,99) y el alpha de los contrastes (0,05) se nombran distinto en la memoria | Pablo | Mismo simbolo para dos cosas distintas |
| 2026-09-22 | `data.values(columns=YIN_VARS)` y `FAULT_ONSET_TRAIN_CLASSIC = 20` (Russell et al., 2000). **[2026-10-12: superada la segunda parte.]** `FAULT_ONSET_TRAIN_CLASSIC = 20` era falso: en los dXX.dat del TEP clasico el fallo esta activo desde la fila 0 (revision del 27/09, B2; informe 11 de la revision del 03/10, A.2). La v0.3 la retira: pedirla da un error que remite a `data.fault_onset("train", "classic")`, que devuelve None (las 480 filas se usan enteras). `YIN_VARS` sigue. | Pablo | La verificacion contra Yin exige sus 33 variables; los dXX.dat traen una hora sana al principio |
| 2026-09-22 | Documentos y cuadernos en espanol neutro, con ustedes | Pablo | Dos companeros de Costa Rica y Colombia |
| 2026-10-04 | Diagnostico: la metrica es la macro-F1, que promedia la F1 de cada fallo (memoria, 2.2.4) | Jonathan (11:49), con el visto bueno de Pablo (11:51) | Pesa lo mismo cada tipo de fallo y no cada muestra (memoria, parrafos 278, 344 y 358). Es la decision 15 del traspaso v12 (5.1, linea 328; 8.1, linea 1139); se anota en este registro el 2026-10-12. Si la clase normal entra en el promedio sigue pendiente (`diagnostico.incluir_normal` en configs/baseline.yaml) |
| 2026-10-04 | Diagnostico: la estandarizacion se ajusta solo con los datos normales del ajuste, como en la deteccion | Jonathan (11:49), con el visto bueno de Pablo (11:51) | La memoria lo fija: "la estandarizacion se ajustara solo con las simulaciones normales del ajuste, tambien en el diagnostico" (parrafo 367). Es la decision 16 del traspaso v12 (5.1, linea 329; 8.1, linea 1139); se anota en este registro el 2026-10-12. En el paquete, `preprocessing.Scaler(normal_label=0)` por defecto desde la v0.3 |
| 2026-10-14 | Verificacion frente a Yin (configs/yin2012.yaml): tolerancia A, replica estricta (abs(dFDR) <= 0,5 pp, 4 muestras de 800, en cada uno de los 18 fallos de los bloques 1 y 2, y abs(dFAR) <= 0,5 pp), comparacion por conteo; DPCA descriptivo (S11 abierto); ventana de la FAR [160, null] (muestras 161 a 960 de d00_te.dat), con 1 a 960 y 1 a 800 como variantes informativas; configuracion secundaria (Tablas 7 y 8) informativa, con DPCA l = 40; transcripcion de las Tablas 4, 5, 7 y 8 (nueve columnas) y umbrales dibujados comprobados por Pablo contra el PDF el 2026-10-14. Solo la configuracion principal decide el veredicto: si no cuadra, se reporta "no verificado" y las variantes van como informacion, sin calificar ninguna y sin rescatar el veredicto. Vale la primera ejecucion con esta huella; otra ejecucion se reporta como tal, con su motivo. En esta verificacion el DPCA no pasa a tener veredicto, y S11 solo se cierra con evidencia que no salga de los ficheros de prueba. Etiqueta git: verificacion-yin-2026-10-14 [V1-yin huella=509751e8faac] | Pablo | Parrafos 355 y 369: tolerancia fijada antes de ejecutar y sin ajustes para que coincidan. Una variante que rescatara el veredicto, o una ejecucion elegida despues de ver otra, seria seleccion a posteriori: el mecanismo que documenta el capitulo 2 |
| 2026-10-15 | Familia confirmatoria (parrafo 368): la ventaja de cada comparacion se mide como el exceso de la FDR (muestras 161 a 960) sobre la FAR de la simulacion normal gemela (el mismo simulationRun de test_fault00, en la ventana de deteccion.ventana_far_normal), y mayor es mejor (hipotesis.metrica_ventaja = exceso_far_gemela, hipotesis.mayor_es_mejor = true). La FDR cruda y la FDR a FAR igualada quedan como analisis descriptivos, no confirmatorios | Pablo | Con la FDR cruda, una comparacion en la que ninguno detecta puede refutar solo por la FAR (imp_contrastes, H13, problema 2: con FAR del 3 % y del 1 % el veredicto sale "refuta"); en Yin (2012), TPLS saca 11,37 puntos al PCA en FDR cruda en el fallo 3 con una FAR del 19,62 %, y en exceso queda por debajo. La gemela comparte ventana, rejilla (1/800) y ruido (DIS-2: del 99,6 % al 100 % de las alarmas del SPE coinciden en 3, 9 y 15); el tramo previo solo tiene 160 muestras y toca el tramo copiado (datos.tramo_copiado, pendiente) |
| 2026-10-15 | Diferencia minima relevante de la FDR: delta = 0,02 (2 puntos porcentuales) sobre la pseudomediana de Hodges-Lehmann de la ventaja por simulacion, en la metrica de la fila anterior (contrastes.delta.fdr). Se fija antes del piloto de varianza y no se revisa con su resultado. El 15/10 por la manana corrio un piloto provisional (salida de las 11:51) con las simulaciones 1 a 20 y claves provisionales (Revisiones_E2/Piloto_Varianza_2026-10-15/); delta venia de la hoja del 15/10 (10:10), anterior a ese piloto, y no se cambio | Pablo | delta mide relevancia, no ruido: fijarla despues de ver sigma seria la seleccion a posteriori que documenta el capitulo 2. Anclas: el doble del nivel nominal de falsas alarmas (1 %), unas diez veces el ruido entre detectores lineales (sigma del exceso de 0,17 a 0,23 puntos en el piloto del 27/09) y menos de la mitad de la mejor ventaja de Yin sobre el PCA en exceso, en la configuracion principal (Tablas 4 y 5), en el fallo en que esa mejor ventaja es menor (MICA, fallo 9: 5,00 puntos). Potencia con dos unilaterales al 5 % (hoja del 15/10, Hoja_Delta_15oct/potencia_delta.py): sin ventaja real, "apoya" con 0,97 (n = 100, sigma = 0,05); con ventaja 2 delta, "ventaja relevante" con 0,99. Con 1 punto no se llega a 0,60 ni con 400 simulaciones si sigma = 0,10; con 3 o 5 puntos un tribunal podria leer un umbral a la medida. 0,02 es exactamente 16/800: las simulaciones con ventaja igual a delta dan ceros que zsplit reparte (imp_contrastes, nota 8), ya incluido en la potencia |
| 2026-10-15 | Diferencia minima relevante de la FAR: 0,005 (0,5 puntos) sobre la pseudomediana de la diferencia por simulacion (contrastes.delta.far), para las ablaciones y comparaciones que contrastan la FAR (por ejemplo, el A0 en Rieth). Condicion: la calibracion de los limites reserva bastantes muestras normales para que la FAR realizada tenga una desviacion claramente menor que 0,5 puntos (particion.rieth.normales_calibracion, pendiente) | Pablo | Los 2 puntos propuestos en septiembre (DIS-11) son laxos: con el 1 % nominal, un 3 % contaria como equivalente a un 1 %, y el propio A0 trato un 3,6 % frente a un 1 % como relevante. Informe 06, A2 y A3: con 500 muestras de calibracion la FAR realizada tiene desviacion de 0,44 puntos; hacen falta de 1 000 a 1 450 muestras iid para que el 90 a 95 % quede entre 0,5 % y 1,5 %, y mas con autocorrelacion |
| 2026-10-15 | Los delta del retardo (contrastes.delta.retardo_muestras) y de la macro-F1 (contrastes.delta.macro_f1) no se fijan en esta etiqueta: van en una adenda con fecha y etiqueta antes de usarlos. El retardo se reporta descriptivo y con censura; la macro-F1 va con la estimanda del OE2 (Jonathan) antes de sus confirmatorios | Pablo | Una muestra es la resolucion del retardo (DIS-11) y descartar los retardos sin deteccion sesga la comparacion (DIS-12); un delta del retardo tiene que salir de un criterio operativo en minutos y hablarse con el equipo. El prerregistro por etapas (adendas fechadas antes de la ejecucion que las use) es compatible con el parrafo 369 |
| 2026-10-15 | Lateralidad del contraste de no superioridad: dos contrastes unilaterales de Wilcoxon, cada uno al alfa de los contrastes (0,05), sobre la ventaja menos delta, con el intervalo de Hodges-Lehmann al 90 % (contrastes.lateralidad = dos_unilaterales). En la familia, la refutacion lleva Holm sobre los seis p de "ventaja relevante" y el apoyo exige que las seis comparaciones apoyen, cada una al 5 % sin correccion (interseccion-union). Se escribe asi en el capitulo 4 | Pablo | Es lo implementado en stats.non_superiority_test e hypothesis_verdict y lo que calculo el informe 06 (anexo A4); es la convencion de los dos unilaterales en equivalencia y es coherente con el alfa = 0,05 de la memoria (parrafo 358). Cada uno al 2,5 % (intervalo al 95 %) pediria un 27 % mas de simulaciones para una potencia del 80 % (un 19 % mas para una de 0,97). La memoria no lo dice hoy (imp_contrastes, H13, problema 3) |
| 2026-10-15 | Ventana de la FAR en las simulaciones normales de Rieth: muestras 161 a 960 de test_fault00 (deteccion.ventana_far_normal = [160, null], base cero, fin abierto), la misma de la verificacion frente a Yin. Es la ventana de la gemela en la metrica de exceso y la de la FAR de cualquier resultado en Rieth; A0 y A1 (TEP clasico) siguen con la serie entera y son de desarrollo | Pablo | DIS-2 e informe 06 (2.2): la FAR se mide en el mismo tramo en que se mide la FDR (800 muestras, rejilla 1/800) para que el exceso reste el ruido comun; en Rieth la normal gemela es copia de la de fallo hasta el inicio nominal, asi que la serie entera mezclaria dos regimenes |
| 2026-10-15 | Numero de simulaciones de prueba confirmatorias por fallo (datos.regla_n): regla prerregistrada en vez de un numero fijo. Despues de esta etiqueta y antes de cualquier ejecucion confirmatoria, experiments/P1_piloto_varianza.py estima, solo con las simulaciones de desarrollo 1 a 20 y con las decisiones ya fijadas (incluidas limites_protocolo.* y las normales de parada del autoencoder, que se fijan antes de la etiqueta), la desviacion de la ventaja del autoencoder frente a PCA y DPCA en los fallos 3, 9 y 15; sigma_ref es el maximo, entre las seis celdas, de la cota superior unilateral al 95 % de sigma (columna sigma_upper). n = 100 si sigma_ref <= 0,05; n = 200 si 0,05 < sigma_ref <= 0,07; n = 400 si sigma_ref > 0,07; si sigma_ref > 0,10, el capitulo 4 declara la potencia que corresponde a esa sigma; y en todo caso n no supera el numero de ids libres del procedimiento de la fila siguiente. El CSV del piloto se cita con su procedencia y la n resultante se anota en este registro antes de los confirmatorios. datos.runs_finales = 100 queda como primer peldano. Esta regla se escribio despues de un piloto provisional (15/10, salida de las 11:51, simulaciones 1 a 20, claves provisionales; sigma_upper de 0,00251 a 0,00413 en las seis celdas). La hoja del 15/10 (10:10), anterior a ese piloto, proponia dos peldanos (100 si sigma_ref <= 0,05; si no, 400); el de 200 se anadio despues, y los umbrales salen de la rejilla de sigma de esa hoja. Con cualquier umbral por encima de 0,00413, el resultado provisional da n = 100. La ejecucion que cuenta es la de despues de la etiqueta | Pablo | Con delta = 0,02 (potencia_delta.csv): sin ventaja real, "apoya" por comparacion sale con 0,97 si sigma <= 0,05 y n = 100, pero con 0,63 si sigma = 0,10 y n = 100. La potencia depende de sqrt(n) / sigma, asi que cada peldano mantiene sqrt(n) / sigma_ref >= 200 hasta sigma_ref = 0,10: en el borde de cada uno, "apoya" sale con 0,98 a 0,99 por comparacion (Revisiones_E2/Revision_Cambios_15oct/potencia_regla_n.csv, 1 000 replicas; con 0,10 como umbral de n = 200 bajaba a 0,85). En la familia de seis seria del orden de 0,9 si las comparaciones fueran independientes, y no lo son. La sigma del autoencoder en Rieth no se conoce (entre lineales fue de 0,002), asi que un n fijo seria insuficiente o arbitrario. n sale de simulaciones ya miradas y no entra en el contraste: fijarlo con el piloto no es seleccion a posteriori (informe 06, 1a y 2.3: n condicionado a un piloto); fijar delta con el piloto si lo seria, por eso delta va en una fila anterior. Las 100 simulaciones de la cuartilla se conservan como primer peldano y el capitulo 4 declara la regla. Con 20 simulaciones la cota superior es 1,37 veces la sigma estimada: elegirla es conservador (empuja hacia n mayor) |
| 2026-10-15 | Procedimiento de los ids confirmatorios (datos.ids_confirmatorios): un id de simulacion es confirmatorio si no se ha mirado en NINGUN fichero de prueba (test_fault00 a test_fault20), es decir, si esta en data.unseen_runs("test"); los ids confirmatorios son los n ids mas altos de esa lista (n de datos.regla_n), los mismos en los tres bloques y en todas las clases (regla de las gemelas); las simulaciones normales de ajuste y de calibracion salen de data.unseen_runs("train", 0) con el reparto de particion.rieth (pendiente). Los ficheros de entrenamiento con fallo (train_fault01 a train_fault20) solo se usan para ajustar el diagnostico y no cuentan para "no mirado". Antes de la etiqueta se cierra RIETH_SEEN_PENDING: Cesar declara que prefijos de split_runs(500) miro el C00 y se registran como mirados (o se registra como mirado un prefijo de longitud K declarada); hasta entonces data.unseen_runs avisa. Al 2026-10-15: 465 ids libres en prueba (entre el 21 y el 499) y 79 si contaran todos los ficheros (421 a 499) | Pablo | Regla 3: las simulaciones miradas en desarrollo no sirven para el resultado confirmatorio. Contar tambien los ficheros de entrenamiento dejaria 79 ids (el diseno de ids distintos del B00 miro hasta el 420 en entrenamiento) y haria imposible n = 100. Los ids mas altos son deterministas, sin semilla ni eleccion, y cualquier solape con lo que miro el C00 lo resuelve el registro antes de la etiqueta, no una excepcion |

## Como se fija la verificacion frente a Yin (la fila de la huella)

`experiments/V1_verificacion_yin.py` no abre ningun fichero de prueba del TEP
clasico hasta que la tolerancia y la configuracion de la verificacion esten
escritas aqui con fecha y confirmadas en git, junto con el codigo que la
ejecuta: la memoria promete una tolerancia "fijada antes de ejecutar"
(parrafo 355) y una verificacion "sin ajustes para que coincidan" (parrafo 369).
El detalle esta en `docs/verificacion_yin.md`. Los pasos:

1. Rellenar las claves en null de `configs/yin2012.yaml` (la fecha en que se
   verifico la transcripcion, la ventana de la FAR, los criterios de la
   tolerancia y la configuracion secundaria) y revisar los supuestos S1 a S12 y
   las variantes.
2. Ejecutar `python experiments/V1_verificacion_yin.py --huella`. Imprime la
   huella (12 cifras hexadecimales del SHA-256 de los valores del yaml y del
   contenido de los dos CSV de `configs/referencias/`) y la etiqueta exacta. Si
   quedan claves en null, las lista y termina con codigo 1: esa huella no se
   registra, porque cambiara al rellenarlas.
3. Anadir al final de la tabla una fila cuya primera celda sea la fecha
   (AAAA-MM-DD, una fecha que exista y no posterior al dia en que se ejecute
   V1) y que contenga la etiqueta tal como la imprime `--huella`. La fila tiene
   que verse y estar en este fichero: V1 no cuenta una fila escrita dentro de un
   comentario `<!-- -->` ni dentro de un bloque de codigo (con las reglas de las
   vallas de CommonMark: un bloque solo lo cierra una valla del mismo caracter,
   al menos tan larga como la de apertura y sin texto detras), y tiene que
   empezar por `|` en la primera columna. Una fila en otro fichero
   (`--decisiones`) solo vale si se pide con `--permitir-otro-registro`, y la
   procedencia guarda que fichero fue. Por ejemplo (este, dentro de un bloque
   de codigo, no cuenta), con
   los huecos por rellenar:

   ```
   | AAAA-MM-DD | Verificacion frente a Yin: tolerancia ..., ventana de la FAR ..., supuestos y variantes de configs/yin2012.yaml [V1-yin huella=<la de --huella>] | Pablo | Parrafo 355: tolerancia fijada antes de ejecutar |
   ```

4. Confirmar en git (commit) `configs/yin2012.yaml`, este fichero,
   `configs/referencias/` y el codigo que se va a ejecutar:
   `experiments/V1_verificacion_yin.py` y la carpeta `tfmfdd/`. V1 comprueba,
   ruta por ruta y sin escribir en el repositorio, que todo esta en el mismo
   repositorio, versionado, sin cambios pendientes, fuera de `.git` y sin
   marcas skip-worktree ni assume-unchanged (`provenance.committed_status`).
   Dentro de `tfmfdd/` tampoco puede haber ningun `.py` sin versionar, aunque
   git lo ignore, ni bytecode que Python ejecutaria en lugar del fuente: un
   `.pyc` "unchecked-hash" en `__pycache__` o un `.pyc` sin su `.py` (si V1 lo
   encuentra, se niega y dice cual borrar).
5. Ejecutar `python experiments/V1_verificacion_yin.py` desde la raiz del
   repositorio, con el paquete de ese mismo repositorio (instalado en modo
   editable, `pip install -e .`, o con `PYTHONPATH` en la raiz). Con un
   `tfmfdd` instalado desde una rueda, V1 se niega: ese codigo no esta
   confirmado en el repositorio del registro.

Si despues cambia cualquier valor del yaml o de los CSV de referencia, la huella
cambia y V1 vuelve a negarse: hace falta una fila nueva con la huella nueva. La
fila vieja no se borra (esto es un registro); la nueva dice que la sustituye y
por que. Cambiar un comentario del yaml no cambia la huella. Cambiar el codigo
no cambia la huella, pero V1 se niega mientras el cambio no este confirmado, y
el commit con que se ejecuto queda en la procedencia de sus resultados
(`package_commit`).

## Donde estan las decisiones pendientes

Las decisiones que el codigo necesita y que todavia no se han tomado estan en los
yaml como claves con valor null. Ningun script las sustituye por un valor por
defecto: el que necesita una se niega a ejecutar mientras siga en null
(`tfmfdd.config.require`). Para listarlas:

```
python -c "from tfmfdd import config; print(config.pending(config.load_baseline()))"
python experiments/V1_verificacion_yin.py --huella
```

- `configs/baseline.yaml` (el protocolo comun): 63 claves en null el
  2026-10-12, en las secciones `contrastes` (diferencia minima relevante por
  metrica, lateralidad, comparaciones no contrastables), `hipotesis` (la familia
  confirmatoria), `deteccion`, `diagnostico`, `particion.rieth`, `datos` (entre
  ellas el procedimiento de los ids confirmatorios), `limites_protocolo`,
  `ablaciones`, `compromiso`, `coste` y `robustez`. El comentario de cada clave
  dice que opciones ofrece el codigo y de donde sale la pregunta. El
  2026-10-15 se fijaron siete (contrastes.delta.fdr y far, lateralidad,
  hipotesis.metrica_ventaja y mayor_es_mejor, deteccion.ventana_far_normal y
  datos.ids_confirmatorios) y se anadio datos.regla_n (filas de esa fecha):
  quedan 56 en null, y los delta del retardo y de la macro-F1 van a una adenda.
- `configs/yin2012.yaml` (la verificacion): 11 claves en null el 2026-10-12;
  todas fijadas el 2026-10-14 (fila de esa fecha).
- Seis claves de `configs/baseline.yaml` llevan valor desde la v0.3 y no son
  todavia decisiones de este registro, porque las propuso la revision del
  codigo del 12/10: `contrastes.alpha_contrastes` (0,05), `pca.t2_method` (f),
  `pca.quantile_method` (linear), `pca.calibration` (holdout),
  `protocolo.min_fdr` (0,10) y `protocolo.retardo_convencion` (appearance). El
  comentario de cada una dice de donde sale el valor. Quedan para confirmar o
  cambiar.

Cuando se tome una, se anota aqui en una fila con fecha, quien y por que, y el
valor se escribe en el yaml en el mismo commit. Las que fijan el planteamiento
de la comparativa (simulaciones de prueba, tabla de ventajas publicadas,
tolerancia de la verificacion, diferencia minima, numero de simulaciones,
criterio de mejor compromiso) van antes de los experimentos confirmatorios y
quedan fijadas con una etiqueta de version en git (memoria, parrafo 369).
