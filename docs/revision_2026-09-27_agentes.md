# Revisión multiagente de `tfm-fdd-core` — 27 de septiembre de 2026

Complementa a `docs/revision_2026-09-27.md`. Hubo cuatro revisores independientes (métricas y estadística, datos y Rieth, diseño experimental, reproducibilidad) y dos verificadores adversariales, que intentaron refutar cada hallazgo, incluidos los del informe original. Este documento se genera por script (`generar_consolidado.py`) a partir de lo que devolvieron los agentes. Los hallazgos, veredictos y anexos se copian tal cual, y solo la introducción y los encabezados están escritos a mano. Las cifras que aparecen dentro de cada hallazgo son **verificaciones ad hoc** de los agentes (sus scripts están en la carpeta temporal de la sesión). Sirven como evidencia, **no como resultados para la memoria**.

**Balance:** 52 hallazgos nuevos (2 bloqueantes, 29 importantes, 21 menores). 75 veredictos en total (incluidos los 23 hallazgos del informe original): 62 CONFIRMADO, 13 MATIZADO.


## 1. Veredictos sobre el informe original

| ID | Veredicto | Matiz del verificador |
|---|---|---|
| B1 | MATIZADO | Lo esencial se sostiene: v0.2 no está en GitHub, las dos historias no tienen ancestro común, hay .pyc versionados y main no tiene commit ni etiqueta que lo identifique. Hay dos afirmaciones falsas. (1) El README que se ve hoy en GitHub dice @v0.1, y esa instalación funciona. Lo que falla es el README local, que aún no está publicado. El riesgo real es el contrario: quien siga GitHub instala el paquete del 17/09, sin convert_rieth y con el compare_methods antiguo. (2) Que main «no coincide con ningún estado tuyo» es falso: su código de paquete es el de tu working tree actual. Lo que falta es un commit y una etiqueta. La corrección propuesta (commit, merge --allow-unrelated-histories, git rm --cached de __pycache__, PR y tag v0.3) es coherente. |
| B2 | CONFIRMADO | La cita de Russell et al. (2000) sigue sin comprobarse en el PDF, pero los datos bastan. El número de muestras descartadas por error es 21 × 20 = 420, como dice el informe. |
| B3 | CONFIRMADO | Nada que objetar. El inflado es mayor cuanto menor es la precisión real, y por eso se concentra en 3, 9 y 15. |
| I1 | CONFIRMADO | far_medio incluye la fila del fallo 0 (d00_te completo) junto con los tramos pre-fallo, como dice el informe. |
| I2 | CONFIRMADO | compare_methods no pasa alternative, así que hoy el error de r con unilaterales solo aparece si alguien llama a paired_test directamente. |
| I3 | MATIZADO | La frase «ninguna partición razonable reproduce el 0,0577» es falsa: lo reproduce el corte en la muestra 461. Al Fisher no le falta reproducibilidad. Le falta declarar un corte elegido a posteriori, que es un grado de libertad del investigador y refuerza I8. La conclusión «no concluyente» se mantiene con los dos cortes. El resto de I3 queda confirmado. |
| I4 | CONFIRMADO | Ninguno. El límite de Fase I es más estrecho que el de Fase II, como dice el informe. |
| I5 | CONFIRMADO | La prueba propuesta de «52 variables sin los analizadores» equivale, en variables, a --yin-vars (XMEAS 23-41 son los 19 analizadores), que ya existe, aunque con 15 componentes. Además, --spe-method empirical con n_cal = 150 trae un sesgo propio del estimador (E[FAR] ≈ 1,6 %; ver MET-3), así que no sirve como prueba limpia. |
| I6 | CONFIRMADO | La referencia exacta es A1:88 (la 87 es el score). No cambia nada. |
| I7 | MATIZADO | Es cierto que el .venv no tenía las dependencias antes de hoy y que no hay registro de las versiones de las bibliotecas. Es falso que el Python global tenga tfmfdd en modo editable: solo importa el código fuente cuando el directorio actual es la raíz del repo. Por eso 'desinstalar tfmfdd del Python global' no tiene ningún efecto. Las salidas de los cuadernos de v0.2 no salen del global (3.12.0), sino de un entorno POSIX con Python 3.12.3. Con el global tal como está hoy, A1 ni siquiera arranca, así que tampoco se puede atribuirle los CSV de v0.2: su procedencia es indeterminable. El resto de la corrección sí vale: el lock, 'python -m nbconvert' con el .venv y ENTORNO.txt. |
| I8 | CONFIRMADO | Ninguno. |
| I9 | MATIZADO | Los tres riesgos existen, con dos precisiones. (a) Un sync en UNIR no arrastraría «todo»: el repo interno entraría como gitlink, sin su contenido. Lo que sí entraría es el zip de 1,59 GB, que bloquearía el push, y los documentos de Contexto. (b) El informe no recoge un riesgo más inmediato: setuptools trata LICENSE.zip como fichero de licencia (patrón LICEN[CS]E*). El .venv ya contiene tfmfdd-0.1.0.dist-info/licenses/LICENSE.zip, de 108.894.512 bytes, y su METADATA dice 'License-File: LICENSE.zip'. Lo comprobé: con un LICENSE.zip ficticio, 'pip wheel' del working tree lo mete en la rueda. Añadir *.zip al .gitignore no lo evita; hace falta license-files = ["LICENSE"] y sacar los zip de la raíz. |
| M1 | CONFIRMADO | En v0.1 la versión sí corresponde. El desajuste está en v0.2 (que solo existe en local) y en main. Además, la versión está duplicada en dos ficheros sin una fuente única: conviene hacerla dinámica o leerla con importlib.metadata para que no diverjan. |
| M2 | CONFIRMADO | El rango 12-22 está desplazado una línea: la lista de dependencias ocupa 13-22, y la 12 es 'authors'. Lo de pyreadr es una redundancia, porque el extra rieth deja de tener efecto, no un error. |
| M3 | CONFIRMADO | Ninguno. |
| M4 | CONFIRMADO | Ninguno. |
| M5 | CONFIRMADO | Ninguno. |
| M6 | MATIZADO | load_classic y load_rieth YA aceptan un parámetro root (data.py:102 y :145). Lo que falta es que los cuadernos y los scripts lo tomen de una variable de entorno o de una configuración, no un parámetro nuevo en el paquete. |
| M7 | CONFIRMADO | Ninguno. |
| M8 | CONFIRMADO | Sin matiz. La misma línea contiene el «fallo desde la muestra 20», que corresponde a B2. |
| M9 | CONFIRMADO | Los tamaños de calibración difieren (350 frente a 150), así que «sin confundido» no es literal. Aun así, al igualar n_cal = 150 dentro de la muestra, la FAR queda en 0,143 (las últimas 150 de Z_fit) o en 0,214 (media de 50 subconjuntos aleatorios), frente a 0,028: el efecto se mantiene. |
| M10 | MATIZADO | El riesgo de conflictos de sincronización es real. Los 109 s, en cambio, no son un efecto reproducible de OneDrive: lo más probable es una primera ejecución en frío. Hay que quitar la cifra o presentarla como una medición única. |
| M11 | MATIZADO | Era cierto cuando se escribió, pero hoy está desfasado: Rieth ya está convertido, fuera del repo, y hay que decir dónde y que hay que pasar root=. La salvedad de «no revisé convert_rieth ni plots» sigue siendo cierta y deja fuera algo relevante: el parámetro runs no limita la memoria de pico. |

## 2. Hallazgos nuevos, por gravedad


### 2.1 Bloqueantes (2)

| ID | Veredicto | Título |
|---|---|---|
| DIS-1 | CONFIRMADO | «No significativa en los fallos 3, 9 y 15» no es contrastable tal como está redactada |
| DIS-2 | CONFIRMADO | En Rieth, la simulación r de todos los ficheros de fallo comparte la realización del ruido: la FAR por fichero se pseudorreplica y los grupos de partición deben ser por índice de simulación |

#### DIS-1 · «No significativa en los fallos 3, 9 y 15» no es contrastable tal como está redactada

- **Veredicto:** CONFIRMADO. *Matiz:* Las referencias de TOST y de unión-intersección siguen marcadas [VERIFICAR], con razón.
- **Dónde:** Hipótesis aprobada: traspaso v3, §4, línea 42 (texto de la cuartilla). Capítulo 3 de la E1 (objetivos y metodología).
- **Qué pasa:** La cláusula predice que un contraste no rechace. No rechazar no es evidencia de ausencia de efecto, y el resultado depende del tamaño de muestra, no del fenómeno. Con pocas simulaciones la hipótesis se confirma sola. Con 100 simulaciones cualquier diferencia irrelevante sale significativa. En el piloto ad hoc de Rieth, con solo 20 simulaciones, el T² del PCA supera al azar en el fallo 15 por 0,15 puntos (exceso medio 0,00150, sd 0,00201, t ≈ 3,3): una diferencia irrelevante que ya sería «significativa». La otra mitad de la hipótesis («menor que la publicada») tampoco tiene un referente fijado (ver DIS-16).
- **Por qué importa:** Es la columna vertebral del capítulo 3 de la E1 (5/10). Un tribunal con formación estadística lo detecta enseguida: es el error de «ausencia de evidencia = evidencia de ausencia», justo el tipo de error de protocolo que el TFM denuncia. decisiones.md:16 ya reconoce que con 100 simulaciones «casi todo sale significativo».
- **Corrección propuesta:** No hace falta cambiar el texto aprobado por el director. Hace falta operacionalizarlo en la metodología:
(1) Contraste de no superioridad por (método, fallo): H0: Δ ≥ δ frente a H1: Δ < δ, con Δ = diferencia pareada por simulación en la métrica de exceso sobre la FAR, a FAR igualada, y δ = diferencia mínima relevante. Se declara si la cota superior del IC unilateral al 95 % queda por debajo de δ.
(2) TOST (añadir también Δ > −δ) como análisis secundario.
(3) Unión-intersección para la afirmación global «en 3, 9 y 15 para todos los métodos»: cada contraste al 5 %, sin Holm [VERIFICAR: Berger, 1982; no está en 04_Investigacion]. Referencia del TOST: Schuirmann (1987) [VERIFICAR].
El detalle está en extra_markdown §3.
- **Evidencia:** Texto de la hipótesis: traspaso v3:42. Piloto: wf/diseno/piloto_potencia.csv y techo_detectabilidad.out (scratch). El t = 0,00150/(0,00201/√20) está calculado a partir de la media y la sd impresas. Potencias: wf/diseno/potencia_tost.out.

#### DIS-2 · En Rieth, la simulación r de todos los ficheros de fallo comparte la realización del ruido: la FAR por fichero se pseudorreplica y los grupos de partición deben ser por índice de simulación

- **Veredicto:** CONFIRMADO. *Matiz:* La reducción de la sd es de 2,6 a 5,2 veces, no de 3 a 5. La coincidencia depende del modelo: con el mío, 99,1-99,9 %. Es el mismo hallazgo que DAT-1 y conviene fusionarlos.
- **Dónde:** Datos de Rieth (C:/Users/pabdu/datos_tfm/tep_rieth). tfmfdd/metrics.py:aggregate y summary (far_medio). tfmfdd/data.py:split_runs. Futuros scripts de Rieth de los tres bloques. Nota del cuaderno 02 sobre GroupKFold.
- **Qué pasa:** Lo comprobé ad hoc.
- Antes del inicio, test_fault01/02/03 run 1 son idénticos a test_fault00 run 1 (máx. |dif| = 0), y train_fault06[:20] es idéntico a train_fault00[:20].
- Después del inicio, la diferencia es solo el efecto del fallo. En 3, 9 y 15, las alarmas del SPE coinciden con las de la simulación normal del mismo índice en el 99,6–100 % de las 800 muestras (runs 1–3).
- Entrenamiento y prueba con el mismo índice NO comparten ruido (máx. |dif| ≈ 5,2 sd).

Consecuencias:
(i) Las FAR pre-fallo de los 20 ficheros son los mismos datos. Cualquier media con IC, conteo de signo («21 de 21») o contraste sobre unidades (fallo, simulación) multiplica n por 20. La media no cambia, pero el error estándar sí.
(ii) En los clasificadores, agrupar por (fallo, simulación) en vez de por índice de simulación pone gemelos casi idénticos con etiquetas distintas a los dos lados de la partición. Eso distorsiona sobre todo 3, 9 y 15.
(iii) Es también una oportunidad: la métrica pareada FDR_f(r) − FAR_0(r) quita el ruido común y reduce la sd entre 3 y 5 veces (0,0011–0,0023 frente a 0,0051–0,0060 sin parear).
- **Por qué importa:** Es justo antes de que Jonathan escale a Rieth (la revisión B3 ya pide fijar la unidad de evaluación). Sin esta regla, los IC de FAR y los contrastes de Rieth salen pseudorreplicados, y la ablación de partición del OE2 mezcla dos fugas.
- **Corrección propuesta:** Añadir una fila a decisiones.md antes de ejecutar Rieth:
(a) la FAR se calcula solo con test_fault00, una por simulación, sobre las muestras 161–960 (misma ventana que la FDR);
(b) el grupo de partición es simulationRun, compartido entre fallos: split_runs se aplica una sola vez y los mismos identificadores se reutilizan en todos los ficheros;
(c) la métrica principal en 3, 9 y 15 es el exceso pareado sobre la FAR de la misma simulación;
(d) nunca se agregan (fallo, simulación) como unidades independientes.
Queda marcar [VERIFICAR] en la documentación de Rieth si las semillas comunes son deliberadas (en 04_Investigacion solo consta que las semillas de entrenamiento y de prueba no se solapan).
- **Evidencia:** wf/diseno/semillas_comunes.out y piloto_rieth.out (líneas «test pre-fallo … max|dif| = 0»). techo_detectabilidad.out (sd del exceso frente a la sd sin parear). Verificación en línea de entrenamiento frente a prueba (máx. 5,245 / 5,386 / 5,272 sd).

### 2.2 Importantes (29)

| ID | Veredicto | Título |
|---|---|---|
| MET-1 | CONFIRMADO | DPCA: score devuelve n − lags filas por simulación y nadie desplaza el inicio del fallo |
| MET-2 | CONFIRMADO | lag_by_run agrupa por valor del identificador y no por bloque contiguo, y reordena la salida por id |
| MET-3 | CONFIRMADO | El percentil empírico con n_cal = 150 tiene una FAR esperada de ≈1,6 %, no del 1 % |
| MET-4 | CONFIRMADO | aggregate: el retardo y su IC salen de un n distinto de n_runs que no se reporta, los IC t se salen de [0, 1] y el retardo por fallo incluye rachas al azar |
| DAT-1 | CONFIRMADO | En Rieth, la simulación normal r y las 20 simulaciones de fallo r del mismo split comparten semilla: son la misma trayectoria hasta el inicio del fallo |
| DAT-2 | CONFIRMADO | El pico de convert_rieth está dentro de pyreadr.read_r (unas 4 veces la tabla float64), no en rename ni en df[COLUMNS]; en un equipo de 16 GB no entra |
| DAT-3 | CONFIRMADO | Ubicación y distribución de Rieth: el destino por defecto queda dentro del repositorio y de OneDrive, el script aconseja borrar los .RData remitiendo a un script de descarga que no existe y no hay manifiesto para verificar los datos |
| DIS-3 | CONFIRMADO | La unidad de réplica de A0 es 1: los 21 segmentos solo miden la variabilidad de evaluación, no la del ajuste |
| DIS-4 | MATIZADO | Los brazos de A0 difieren también en el tamaño de calibración, y el percentil empírico tiene sesgo de muestra finita |
| DIS-5 | CONFIRMADO | T² de Fase I (Beta) frente a Fase II (F): el efecto existe, está predicho por la teoría y A0 no lo mide |
| DIS-6 | CONFIRMADO | El límite de Jackson–Mudholkar es una calibración en muestra implícita, no un límite inmune |
| DIS-7 | CONFIRMADO | El factor (c), estandarizar con datos de prueba, no se puede medir en el PCA con el código actual, y en el Random Forest es cero por construcción |
| DIS-8 | CONFIRMADO | El factor (b), partición por muestra frente a por simulación, no tiene script, y en el clásico no se puede medir sin confundido |
| DIS-9 | CONFIRMADO | El factor (d), una simulación frente a muchas, no sesga en esperanza: hay que medirlo como dispersión y como tasa de conclusiones erróneas |
| DIS-10 | CONFIRMADO | Contrastes: la familia de Holm, la estimanda, el intervalo y el baseline no están definidos |
| DIS-11 | MATIZADO | La diferencia mínima relevante propuesta es laxa en la FAR, vacía en el retardo y ambigua en la FDR |
| DIS-12 | CONFIRMADO | Los retardos NaN se descartan del par y eso introduce un sesgo de supervivencia |
| DIS-13 | CONFIRMADO | Comparar métodos sin igualar la FAR ni la memoria confunde «no lineal» con «punto de operación» y con «contexto temporal» |
| DIS-14 | CONFIRMADO | Con una sola simulación, todo contraste es pseudorreplicación: el clásico solo admite descripción |
| DIS-15 | MATIZADO | La parte «3, 9 y 15» corre el riesgo de ser trivialmente cierta: no hay casi nada que detectar |
| DIS-16 | CONFIRMADO | «Menor que la publicada» necesita una tabla prerregistrada de ventajas publicadas, y hay que separar el efecto del protocolo del efecto del conjunto de datos |
| DIS-17 | MATIZADO | Prerregistro: fijar la operacionalización antes de ejecutar Rieth y apartar las simulaciones que ya se han visto |
| REP-1 | CONFIRMADO | El Python global no tiene tfmfdd instalado: lo importa desde el directorio de trabajo (contradice I7) |
| REP-2 | CONFIRMADO | Las salidas de los cuadernos de v0.2, y las del 03 que irían en v0.3, salen de Linux con Python 3.12.3, no de tu PC |
| REP-3 | CONFIRMADO | GitHub manda hoy instalar @v0.1, que sí funciona; esa etiqueta es de tu historia local, y main trae el mismo código que tu working tree |
| REP-4 | CONFIRMADO | El protocolo común (configs/baseline.yaml) no viaja con el paquete, y lo que sí viaja trae otros valores por defecto |
| REP-5 | CONFIRMADO | La construcción de v0.3 dejará de estar soportada el 18/02/2027, en plena ventana de defensa |
| REP-6 | CONFIRMADO | requires-python >=3.10 promete compatibilidades que el lock no puede cumplir: las versiones fijadas exigen 3.12 |
| REP-7 | CONFIRMADO | Los datos no tienen procedencia versionada: ni script de descarga, ni sumas de comprobación, y su documentación está ignorada |

#### MET-1 · DPCA: score devuelve n − lags filas por simulación y nadie desplaza el inicio del fallo

- **Veredicto:** CONFIRMADO. *Matiz:* No reejecuté t13 sobre el TEP. El mecanismo es exacto y siempre favorece a DPCA. El piloto de diseño (diseno/piloto_rieth.py:127-129) ya usa onset 158 para DPCA, así que el problema no le afecta.
- **Dónde:** tfmfdd/pca.py:172-177 (score); tfmfdd/metrics.py:76, 108, 152-153 (onset como índice del vector recibido); tfmfdd/plots.py:65-67; experiments/A1_pca_baseline.py:73-76 y :92; configs/baseline.yaml:27
- **Qué pasa:** Con lags = l, la fila j del estadístico corresponde a la muestra j + l. evaluate_run(stat, lim, onset=160) toma la fila 160 (muestra 160 + l) como la primera con fallo. Eso tiene tres efectos: (1) el retardo sale l muestras más corto; (2) las muestras con fallo 160 … 160+l−1 caen en el tramo de la FAR; (3) la FDR pierde l muestras. control_chart dibuja la línea de inicio l muestras tarde, y todo el eje de tiempo queda desplazado. Además, A1 ofrece --lags (y el yaml dice «1 o 2 = DPCA»), pero llama a fit sin runs: `A1_pca_baseline.py --lags 1` falla con ValueError en pca.py:74. Por eso ninguna cifra actual está contaminada, pero el primer resultado DPCA lo estará si esto no se corrige antes.
- **Por qué importa:** El sesgo mide exactamente l muestras, justo la diferencia mínima relevante declarada para el retardo (1 muestra), y siempre favorece a DPCA. En la ablación PCA/DPCA del OE1, DPCA saldría «más rápido» por construcción, y en los fallos rápidos (6, 7) sumaría falsas alarmas que en realidad son detecciones.
- **Corrección propuesta:** Que score devuelva también el identificador de simulación y el índice de muestra original de cada fila (por ejemplo, `return_index=True`), y que la evaluación use onset − l. Otra opción es un parámetro `offset=lags` en evaluate_run y control_chart. Hay que corregir A1 (pasar runs=np.ones(...) a fit y score, y usar onset − lags) antes de publicar la opción --lags. Prueba mínima: con un fallo visible en el índice 170, el retardo debe dar 10 con lags 0, 1 y 2.
- **Evidencia:** scratch/wf/metricas/t01_dpca_onset.py (sintético, fallo visible en el índice 170): delay_samples = 10 / 9 / 8 con lags 0 / 1 / 2 (verdad: 10). Con el fallo desde 160 y lags=2, las filas 158-159 están en alarma, la FAR pre-fallo sale 0,0125 y la FDR se calcula sobre 798 filas en lugar de 800. t13_dpca_tep.py (verificación ad hoc, TEP clásico, 52 variables, 85 %, Box, 350/150): comparado con onset − lags, el retardo queda infraestimado en media 0,95 muestras (T²) y 0,74 (SPE) con lags=1, y en 1,79 y 1,50 con lags=2. Con lags=2, la FAR de los fallos 6 y 7 da 0,0125 frente a 0,0. `.venv python experiments/A1_pca_baseline.py --lags 1 --out <scratch>` → ValueError: «Con lags > 0 hay que pasar el vector de simulaciones».

#### MET-2 · lag_by_run agrupa por valor del identificador y no por bloque contiguo, y reordena la salida por id

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/preprocessing.py:81-85 (lag_by_run); tfmfdd/pca.py:176-177 (score descarta los ids)
- **Qué pasa:** `for r in np.unique(runs): m = runs == r` junta todas las filas con el mismo identificador aunque vengan de ficheros distintos, y devuelve las simulaciones ordenadas por id, no en el orden de entrada. En Rieth, cada fichero de fallo numera sus simulaciones de 1 a 500. Si se concatenan varios fallos y se pasa df.simulationRun, la simulación 1 del fallo 1 se une con la 1 del fallo 2, y la fila frontera apila una muestra de un fallo con la «anterior» de otro: justo lo que la docstring de lag_matrix dice evitar. Como score descarta el vector de ids, quien recibe el estadístico no tiene forma de saber que las filas cambiaron de orden ni dónde empieza cada simulación.
- **Por qué importa:** No da ningún error ni aviso. Si alguien trocea el vector devuelto por posición (960 − l por simulación) para aplicar evaluate_run, calcula las métricas sobre trozos de simulaciones y de fallos distintos.
- **Corrección propuesta:** Agrupar por bloques contiguos (cortes en np.flatnonzero(np.diff(runs) != 0) + 1) manteniendo el orden de entrada, o lanzar ValueError si un mismo id aparece en bloques no contiguos. Que score devuelva los ids y el índice de muestra (enlaza con MET-1). Prueba: dos bloques con el mismo id y lags=1 deben dar 8 filas (o un error), y el orden de salida debe ser igual al de entrada.
- **Evidencia:** scratch/wf/metricas/t02_lag_by_run.py: dos bloques de 5 filas con run=1 y lags=1 dan 9 filas en lugar de 8, y la fila [200,0; 104,0] mezcla los dos bloques. Con la entrada «run 3 y luego run 1», la salida trae los ids [1, 1, 1, 3, 3, 3]. score con runs=[2]*50+[1]*50 devuelve primero la run 1 (T² 169,2; 164,4; 161,8), sin ids. Rieth convertido (leídas solo las columnas de id): train_fault00, train_fault01 y test_fault01 tienen simulationRun de 1 a 500, ordenado, con 500 o 960 muestras por simulación.

#### MET-3 · El percentil empírico con n_cal = 150 tiene una FAR esperada de ≈1,6 %, no del 1 %

- **Veredicto:** CONFIRMADO. *Matiz:* Aplica a datos intercambiables, así que es cota del brazo reservado. En el autoencoder, el sesgo de n no explica el efecto (ver M9).
- **Dónde:** tfmfdd/limits.py:124-131 (limit_empirical); tfmfdd/pca.py:151-152 (spe_method='empirical'); notebooks/03_arranque_deeplearning.ipynb celdas [6] y [13]; A0 con --spe-method empirical
- **Qué pasa:** np.percentile(stat, 99) con interpolación lineal toma la posición 0,99·(n−1)+1 = 148,51 (en base 1) cuando n = 150. Con datos iid continuos, la probabilidad de que una observación nueva supere el estadístico de orden j es (n+1−j)/(n+1). La FAR esperada del límite es, por tanto, ≈1,65 % sea cual sea la distribución. El sesgo baja despacio con n: 1,27 % con 350, 1,19 % con 500 y 1,09 % con 1000.
- **Por qué importa:** El protocolo compara la FAR con el 1 % nominal. Con este estimador, un exceso del 58 % sale del estimador y no del modelo. Consecuencias: (1) en A0 con --spe-method empirical (lo que propone I5), los brazos no son simétricos: en muestra se calibra con n = 350 (1,27 %) y con reserva con n = 150 (1,58 %), así que el estimador añade 0,3 puntos al brazo reservado y diluye el efecto medido; (2) lo mismo afecta al brazo T² empírico que propone I4; (3) para el 2,8 % del autoencoder del cuaderno 03 (M9, calibrado con 150 muestras), la referencia a priori no es el 1 % sino ≈1,6 %. A0 y A1 no están afectados hoy porque usan Box.
- **Corrección propuesta:** Usar el estadístico de orden k = ⌈α(n+1)⌉ (cuantil conforme). Con n = 150 y α = 0,99, k = 150, es decir, el máximo, con FAR esperada ≤ 1/151 = 0,66 %. Otra opción es mantener el percentil pero reportar a su lado la FAR esperada efectiva (n+1−pos)/(n+1). En A0 con el método empírico, calibrar ambos brazos con el mismo n o declarar la asimetría. Prueba: un Monte Carlo con n = 150 debe dar una FAR media dentro de ±0,2 puntos del nivel declarado.
- **Evidencia:** scratch/wf/metricas/t04_empirical.py (chi2(8), 20 000 réplicas por n; FAR verdadera con chi2.sf): FAR esperada del empírico 0,0158 con n = 150 (teoría 0,0165), 0,0127 con 350, 0,0119 con 500 y 0,0109 con 1000. KDE (2000 réplicas): 0,0107 con n = 150. El cuaderno 03, celda [3], usa un corte de int(500·0,7) = 350, con lo que Z_cal tiene 150 filas, y la celda [6] llama a limit_empirical(error(Z_cal), 0.99).

#### MET-4 · aggregate: el retardo y su IC salen de un n distinto de n_runs que no se reporta, los IC t se salen de [0, 1] y el retardo por fallo incluye rachas al azar

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno. El punto (c) amplía I1 correctamente.
- **Dónde:** tfmfdd/metrics.py:176-194 (aggregate: dropna en 183, IC t en 187-191, groupby en 173)
- **Qué pasa:** (a) `v = g[m].dropna()`: el retardo solo existe en las simulaciones que confirman la detección, así que delay_samples_mean y su IC se calculan sobre esas, mientras la fila muestra n_runs con todas. No hay ninguna columna con el n usado por métrica. (b) El IC t simétrico sobre proporciones se sale del rango: con 5 simulaciones y FAR [0; 0; 0; 0; 0,0125] da IC [−0,0044; 0,0094], y con FDR [1; 1; 1; 1; 0,95] da [0,962; 1,018]. (c) Evidencia nueva sobre I1: el delay_samples_mean por fallo también promedia los retardos marcados como detected_at_chance, así que corregir solo summary no basta. Con el CSV actual de A1, el SPE del fallo 3 da delay_samples_mean = 537 con at_chance_ratio = 1,0. (d) groupby descarta sin aviso las filas cuya clave es NaN.
- **Por qué importa:** En la comparación de retardos entre métodos, uno que solo confirma en las simulaciones fáciles parece más rápido, y el IC se presenta junto a un n_runs que no es el que lo generó. El retardo es una de las métricas centrales del protocolo, y hay una diferencia mínima relevante de 1 muestra.
- **Corrección propuesta:** Añadir `{m}_n` por métrica. Para el retardo, reportar también una versión censurada (retardo = longitud del tramo si no detecta, o una mediana de Kaplan–Meier) o, como mínimo, obligar a mostrar detection_ratio junto a él. Excluir o separar detected_at_chance también en aggregate. Para FAR y FDR, usar un IC acotado: bootstrap por simulación (percentil) o transformación logit, o recortar a [0, 1] y declararlo. Usar groupby(dropna=False) o lanzar un error si hay NaN en la clave.
- **Evidencia:** scratch/wf/metricas/t05_aggregate.py: con n_runs = 5 y detection_ratio = 0,40, delay_samples_mean = 21,5 con IC [−213,6; 256,6], calculado con n = 2; 10 filas de entrada (5 con faultNumber NaN) dan n_runs total = 5. t12_delay_seleccion.py: B es 1 muestra más lento en todas las simulaciones pero solo confirma en 5 de 20; aggregate da un retardo de 15,78 (IC [11,68; 19,88]) para B frente a 19,61 para A, con n_runs = 20 en ambas filas. Comprobación sobre results/summary/A1_pca_baseline_runs.csv (SPE): fallo 3 → delay_samples_mean 537,0 y at_chance_ratio 1,0.

#### DAT-1 · En Rieth, la simulación normal r y las 20 simulaciones de fallo r del mismo split comparten semilla: son la misma trayectoria hasta el inicio del fallo

- **Veredicto:** CONFIRMADO. *Matiz:* La primera diferencia real puede llegar varias muestras después del inicio nominal. Descartar solo sample ≤ onset deja algunas filas idénticas a la clase 0: es un ruido de etiqueta menor.
- **Dónde:** Propiedad de los datos. Donde documentarla: tfmfdd/data.py:183-209 (split_runs), tfmfdd/data.py:141-162 (docstring de load_rieth), docs/decisiones.md; afecta a los futuros B00/C00 y al porte de A0/A1 a Rieth.
- **Qué pasa:** En cada split, la simulación con simulationRun = r de *_fault00 y la simulación r de cada *_faultKK son idénticas bit a bit antes del inicio del fallo. En entrenamiento coinciden las filas 0-19 y en prueba las 0-159, en los 20 fallos x 20 simulaciones comprobados. Después del inicio, en los fallos sutiles siguen casi pegadas a su gemela normal: la distancia media |Δz| tras el fallo es 0,013 en IDV 3, 0,063 en IDV 9 y 0,093 en IDV 15, frente a 1,14 entre simulaciones no gemelas. Entre entrenamiento y prueba no se comparten semillas: 0 de 400 pares de filas iniciales iguales. El TEP clásico no tiene esta propiedad: 0 de 21 segmentos 1-160 de dKK_te coinciden con d00_te.
- **Por qué importa:** (1) Partición: la unidad independiente no es el par (fallo, simulación), sino el id de simulación dentro de cada split, compartido por las 21 clases. Si se reparten simulaciones por clase con semillas distintas, o se agrupa por faultNumber*1000+simulationRun, la gemela normal de una simulación de validación acaba en entrenamiento. En 3, 9 y 15 es casi la misma serie con otra etiqueta, justo los fallos de la hipótesis. Que un método de vecinos la explote es previsible, pero no está medido. (2) Etiquetado: las primeras 20 filas (entrenamiento) o 160 (prueba) de cada fichero de fallo son copias exactas de la clase 0. Etiquetarlas con el fallo es ruido de etiqueta exacto; etiquetarlas como 0 las duplica 21 veces. (3) FAR: los 20 segmentos pre-fallo de prueba de la simulación r son el mismo segmento. Si se porta a Rieth el promedio de segmentos pre-fallo de A0, cada simulación cuenta 20 veces. Y si se calibra con simulaciones de test_fault00, esas filas reaparecen en el pre-fallo de todos los ficheros de prueba con ese id.
- **Corrección propuesta:** Antes de B00/C00, documentarlo en split_runs, en load_rieth y en decisiones.md. (a) Llamar a split_runs(500, seed=...) una sola vez y usar los mismos ids en todas las clases; en validación cruzada, GroupKFold con groups=df['simulationRun'] sobre el conjunto de todas las clases, nunca con una clave (fallo, simulación). (b) En los ficheros de fallo, descartar sample <= data.fault_onset(split, 'rieth'), porque ya están en *_fault00. (c) Medir la FAR por simulación en test_fault00, no promediar como independientes los pre-fallo de los 20 ficheros. Los ids usados para calibrar quedan fuera de la evaluación de todas las clases. Opcional: un ayudante load_rieth_pool(faults, split, runs, drop_prefault=True) que aplique (a) y (b).
- **Evidencia:** Scripts en <scratch> = C:/Users/pabdu/AppData/Local/Temp/claude/C--Users-pabdu-OneDrive-Documentos-Universidades-UNIR-TFM/b1423c25-c9de-48ed-b4f5-76701e58807e/scratchpad/wf/datos-rieth. <scratch>/v2_onset_semillas.py, salida en v2_resultado.json: pre_onset_identico_20_runs = true en los 20 fallos y en los dos splits; la primera diferencia llega en el índice 20 (entrenamiento) o 160 (prueba) o después, nunca antes. <scratch>/v3_gemelos.py, salida en v3_resultado.json: distancias gemela/no gemela de 0,0128/1,142 (IDV 3), 0,063/1,146 (IDV 9), 0,0933/1,146 (IDV 15), 1,866/2,509 (IDV 1) y 0,516/1,399 (IDV 5); normal frente a normal, 1,1415; en el clásico, 0 de 21. Semillas entre entrenamiento y prueba: v2_resultado.json, train_vs_test_normal_cualquier_par_fila0_igual = 0.

#### DAT-2 · El pico de convert_rieth está dentro de pyreadr.read_r (unas 4 veces la tabla float64), no en rename ni en df[COLUMNS]; en un equipo de 16 GB no entra

- **Veredicto:** CONFIRMADO. *Matiz:* No abrí los .RData (regla dura), así que no verifiqué la disposición int32 real ni la propuesta alternativa. La extrapolación a faulty_testing (9,6 M filas, 3,93 GiB en float64) × ≈ 4 es coherente.
- **Dónde:** tfmfdd/convert_rieth.py:127 (pyreadr.read_r); docstring en :5-9 y :24-27; :140-141 (_normalizar_columnas, _aligerar).
- **Qué pasa:** pyreadr guarda las 55 columnas como arrays y, sin soltarlas, llama a pd.DataFrame.from_dict. En los .RData de Rieth, 'sample' (y en tres de los cuatro ficheros también 'faultNumber') es INTEGER de R, es decir, int32, intercalado entre columnas double. Con pandas 3.0.6 (el del .venv), esa disposición hace que from_dict añada entre 2,92 y 2,98 veces la tabla; sube solo 1,0 si todas son float64 o si la int32 va al final. Medido: read_r sobre el .RData real más pequeño llega a 3,99 veces la tabla float64 (418 MiB para 104,9 MiB); la conversión actual completa sobre un sintético con la disposición real, a 4,11 veces. Coste propio de cada paso: rename + df[COLUMNS], 0,0 veces (copy-on-write y columnas ya en orden); _aligerar, +0,47 veces transitorio; filtrado por fallo, +0,16 veces. Extrapolado a faulty_testing (9,6 M filas; tabla float64 de 3,93 GiB): entre 15,3 y 16,2 GiB. Encaja con los ~15,5 GB de memoria comprometible que midió el revisor principal. La docstring afirma que 'si el equipo tiene 16 GB debería entrar', y --solo no sirve de nada porque faulty_testing es un único fichero.
- **Por qué importa:** Jonathan y Cesar convertirían en portátiles de ~16 GB, y faulty_testing agota la memoria comprometible o provoca swapping severo (el revisor llegó a 0,01 GB de RAM disponible). Además, atribuir el pico a rename + df[COLUMNS] llevaría a un arreglo inútil: la variante con API pública que elimina esas copias mide el mismo pico (4,11 veces).
- **Corrección propuesta:** (1) Lo preferible: que no conviertan. Se distribuyen los 42 Parquet ya convertidos con su manifiesto SHA-256 (DAT-3). (2) Si hay que convertir: una subclase de PyreadrParser que baje cada columna double a float32 en handle_column y escriba cada fallo desde los arrays, sin construir el DataFrame completo. El código está en <scratch>/propuesta_convert_rieth_v2.py. Pico medido: entre 0,54 veces (parseo del .RData real pequeño) y 0,80 veces (conversión completa del sintético con la disposición real). Estimado para faulty_testing: 2,1-3,1 GiB, entre 5 y 7 veces menos; para faulty_training, 1,1-1,6 GiB frente a los 8,0-8,4 GiB de hoy. La salida es idéntica en valores y tipos en 4 sintéticos (20 de 20 ficheros cada uno) y, en memoria, frente al train_fault00.parquet existente. Tarda lo mismo (8,0 s frente a 8,4 s con 480 000 filas). Usa la API interna de pyreadr 0.5.6, así que hay que fijar esa versión en el lock, añadir una prueba con un .RData pequeño escrito con pyreadr.write_rdata (con sample int32) y dejar un respaldo con read_r y aviso. (3) Corregir la docstring: con la vía actual hacen falta ~16-17 GB de memoria comprometible libres. Con pandas 2.x no está medido; ahí rename sí copiaría.
- **Evidencia:** <scratch>/m1_medir_conversion.py y m2_lanzar.py. m2_resultado_sintreal_R25.json: con una tabla de 201,4 MiB, la versión actual llega a 827 MiB, la API pública sin rename/df[COLUMNS] a 827 MiB y la propuesta a 161 MiB. m2_resultado_sint_R25.json y m2_resultado_sint_R50.json (todo float64): 2,40 y 2,37 veces frente a 0,80 y 0,76, lo que confirma que escala linealmente. m3_pasos_aislados.py real_pequeno (3,98-3,99 veces) y m4_traza_parser.py (parseo 1,02 veces; 3,99 veces tras el DataFrame). m5_from_dict.py, m6_variantes.py y m9_layout_faulty.py: from_dict +1,0 con todo float64; +2,97 con int32 en la posición 2; +1,0 con int32 al final; +2,92 con int32 en las posiciones 0 y 2. m8_tipos_sin_cargar.py sobre los cuatro .RData reales, en solo lectura y sin retener columnas (pico de 147 MiB o menos): int32 en las posiciones 0 y 2 en FaultFree_Testing, Faulty_Training y Faulty_Testing, y en la 2 en FaultFree_Training. m10_pasos_layout_real.py y m11_muestreo_pasos.py: rename + df[COLUMNS] 0,0; aligerar +0,47; bucle +0,16. m7_v2_real_parse.py: 0,54 veces.

#### DAT-3 · Ubicación y distribución de Rieth: el destino por defecto queda dentro del repositorio y de OneDrive, el script aconseja borrar los .RData remitiendo a un script de descarga que no existe y no hay manifiesto para verificar los datos

- **Veredicto:** CONFIRMADO. *Matiz:* git ls-files lista 28 ficheros versionados, no 23. No cambia la conclusión.
- **Dónde:** tfmfdd/convert_rieth.py:19 y :194-197 (--origen/--destino por defecto 'data/tep_rieth'), :238-241 (mensaje de borrado); tfmfdd/data.py:145 (root por defecto); .gitignore:1; README.md:45; docs/revision_2026-09-27.md sección 5 (pide que cada uno convierta).
- **Qué pasa:** La conversión escribe por defecto en data/tep_rieth, relativo al directorio de trabajo. En el equipo de Pablo eso es dentro del repositorio y de OneDrive: 1.235,3 MiB de Parquet más los 1.338 MiB de .RData que ya están ahí (1.402.950.911 bytes). Git está protegido (.gitignore excluye data/, *.parquet y *.RData); OneDrive, no. Al terminar, el script dice que se pueden borrar los .RData porque 'el script de descarga los vuelve a traer', pero ese script no existe en el repositorio (git ls-files). Además, los Parquet son float32, así que el original float64 no se recupera a partir de ellos. No hay manifiesto ni función que compruebe que los tres autores tienen exactamente los mismos ficheros.
- **Por qué importa:** El protocolo común exige los mismos datos en los tres bloques. Si cada uno convierte por su cuenta, con otras versiones de pandas o pyarrow y arriesgando la memoria (DAT-2), no hay forma de demostrarlo. Sincronizar ~2,6 GB en OneDrive y arriesgar la única copia float64 son riesgos evitables.
- **Corrección propuesta:** Distribuir los 42 Parquet de C:/Users/pabdu/datos_tfm/tep_rieth (1.295.333.849 bytes) con el manifiesto SHA-256 del extra_markdown. Versionar ese manifiesto en el repositorio y añadir data.verificar_rieth(root), que compare los hashes. Que root y --destino tomen su valor por defecto de una variable de entorno (por ejemplo TFM_RIETH), o que sean obligatorios, y que apunten fuera de OneDrive. Quitar el consejo de borrar los .RData (o condicionarlo a tener una copia fuera) y la mención al script de descarga, también en .gitignore:1.
- **Evidencia:** <scratch>/inventario_sha256.txt y v1_resultado.json (tamaños y SHA-256 de los 42 ficheros). git ls-files: 23 ficheros versionados, ninguno de descarga. ls -la data/tep_rieth: 4 .RData, con mtimes del 16 y 17/09 sin cambios.

#### DIS-3 · La unidad de réplica de A0 es 1: los 21 segmentos solo miden la variabilidad de evaluación, no la del ajuste

- **Veredicto:** CONFIRMADO. *Matiz:* El brazo «otra simulación (150)» depende de qué simulación se use: con las primeras 150 muestras de la r+10, obtuve 2,27 %, no 1,21 %.
- **Dónde:** experiments/A0_calibracion_limites.py:80-99. docs/revision_2026-09-27.md §4 («21 de 21 segmentos»).
- **Qué pasa:** Los 21 segmentos pre-fallo del clásico son simulaciones distintas (comprobé que sus datos difieren), pero todos se evalúan con el mismo modelo y los mismos límites, sacados de una sola realización de d00.dat. El efecto de calibración depende de la muestra de ajuste (Fase I), y de esa fuente hay n = 1. «21 de 21» describe esa realización, no la generalidad del efecto.

El piloto en Rieth usa 10 réplicas independientes: 350 muestras de una simulación normal distinta por réplica, y FAR media sobre 20 simulaciones de test_fault00. Resultados:
- SPE (Box) calibrado en muestra: FAR 25,1 % (sd 7,2; rango 15,1–38,4).
- Reserva temporal (150): 1,25 % (sd 0,93).
- Otra simulación (150): 1,21 %.
- Diferencia en muestra − reserva: 23,8 puntos (sd 7,2), positiva en 10 de 10.

En Rieth, la calibración reservada queda en la nominal, mientras que en el clásico queda en 3,6 %. Eso apoya la hipótesis de I5 (el exceso sería propio del clásico), aunque no la demuestra.
- **Por qué importa:** Es la cifra central del OE1. Sin réplicas de ajuste no hay intervalo del tamaño de efecto, y la pregunta «¿y con otra simulación normal?» no tiene respuesta. Jensen et al. (2006) y Does et al. (2020), ambos en 04_Investigacion, dicen justamente que la FAR condicional varía con la muestra de Fase I.
- **Corrección propuesta:** A0 en Rieth con R = 20 réplicas de entrenamiento independientes (train_fault00, simulaciones disjuntas, sin superar 20–40 simulaciones cargadas a la vez). La unidad es la réplica, y el intervalo sale de la diferencia pareada entre réplicas. Añadir dosis-respuesta en n_fit ∈ {100, 200, 350, 1000, 5000}. El clásico queda como ilustración y como puente con la literatura, no como estimación.
- **Evidencia:** wf/diseno/piloto_rieth.py, piloto_rieth.out, piloto_factor_a.csv (scratch). Clásico: t2_fases_clasico.out (última línea: los tramos pre-fallo difieren entre ficheros).

#### DIS-4 · Los brazos de A0 difieren también en el tamaño de calibración, y el percentil empírico tiene sesgo de muestra finita

- **Veredicto:** MATIZADO. *Matiz:* Es cierto para el T² empírico, pero no se puede extender al umbral del autoencoder (M9): al igualar n_cal = 150 dentro de la muestra, la FAR queda en 0,143 o 0,214 frente a 0,028 (<vc>/v05_m9_ae.py). El tamaño de calibración no explica «parte del efecto» del autoencoder.
- **Dónde:** A0:81-89 (X_fit 350 frente a X_cal 150). tfmfdd/limits.py:limit_empirical (np.percentile con interpolación lineal). Cuaderno 03, celda [13] (umbral del autoencoder: 350 en muestra frente a 150 reservado).
- **Qué pasa:** Para el SPE por Box, el tamaño de calibración no importa: 14,8203 con 350 frente a 14,8194 con las últimas 150 de X_fit. Para límites empíricos sí importa. El percentil 99 del T² en muestra vale 43,80 con 350 y 47,66 con 150 de la misma X_fit, con FAR pre-fallo de 6,40 % y 3,30 %: cambiar n_cal mueve la FAR tanto como cambiar la procedencia.

Además, np.percentile(99) tiene una E[FAR] mayor que la nominal incluso con datos iid: 1,65 % con n = 150, 1,28 % con 350 y 1,20 % con 500. El estadístico de orden k = ⌈0,99(n+1)⌉ da 0,66 %, 0,85 % y 1,00 %. En el piloto de Rieth, cualquier límite empírico de T² con 150 muestras da una FAR de 2,3–4,0 %, sea cual sea la procedencia.
- **Por qué importa:** En el T² y en el umbral del autoencoder (M9, «17,7 frente a 2,8») parte del efecto atribuido a «en muestra frente a reservada» es efecto del tamaño de calibración. Es un confundido del mismo tipo que el que se corrigió el 22/09.
- **Corrección propuesta:** Igualar n_cal entre brazos. El brazo en muestra se calibra con un subconjunto de X_fit del mismo tamaño que X_cal, repetido sobre varios subconjuntos, o en Rieth con 350 reservadas de otra simulación. Cuantil por estadístico de orden (np.quantile con method='higher' o k explícito) [VERIFICAR la referencia conformal, p. ej. Lei et al., 2018; no está en 04_Investigacion]. Añadir la dosis-respuesta en n_cal. Avisar a Cesar para el umbral del autoencoder.
- **Evidencia:** wf/diseno/t2_fases_clasico.out (líneas «T2 empirico…», «SPE Box en muestra con 150»). piloto_rieth.out. E[FAR] del percentil: cálculo en línea con el .venv, E[U(k)] = k/(n+1); no quedó guardado en fichero.

#### DIS-5 · T² de Fase I (Beta) frente a Fase II (F): el efecto existe, está predicho por la teoría y A0 no lo mide

- **Veredicto:** CONFIRMADO. *Matiz:* La atribución de la fórmula Beta de Fase I sigue marcada [VERIFICAR], con razón.
- **Dónde:** tfmfdd/limits.py:29-45 (t2_limit). tfmfdd/pca.py:143. A0:13-15. Cuaderno 01, celdas [33]–[35]. Revisión I4.
- **Qué pasa:** Fórmulas, con a componentes, m muestras de ajuste y nivel 0,99:
- Fase II (observación nueva, independiente de las m): UCL = a(m+1)(m−1)/[m(m−a)]·F₀,₉₉(a, m−a). Es lo que implementa t2_limit (Tracy, Young y Mason, 1992, según la nota técnica de Yin).
- Fase I (las mismas observaciones que estimaron media y covarianza): UCL = [(m−1)²/m]·B₀,₉₉(a/2, (m−a−1)/2) [VERIFICAR la atribución a Tracy et al., 1992: 04_Investigacion solo los cita para la F].
- Asintótico: χ²₀,₉₉(a).

Con a = 27 y m = 350, mismo montaje que A0:
- F = 52,6266 (coincide con el código).
- Beta = 45,6140.
- χ² = 46,9629.
- Percentil empírico 99 del T² en muestra sobre X_fit = 43,7977.
- Media en muestra = 26,9229 = a(m−1)/m exacta; máximo 49,10.

La teoría (normal e iid) predice que el límite Beta aplicado a datos nuevos da una FAR de 4,05 % frente al 1 %.

FAR observada (media de 21 segmentos pre-fallo | d00_te completo):
- F: 1,40 % | 1,46 %.
- Beta: 4,64 % | 6,35 %.
- Empírico en muestra: 6,40 % | 8,54 %.
- Empírico reservado: 2,23 % | 2,08 %.
- Signo empírico en muestra − reservado: > 0 en 20 segmentos, = 0 en 1.

Piloto en Rieth (10 réplicas): F 0,85 %, Beta 2,52 %, empírico en muestra 2,99 % frente a reservado 2,28 %, positivo solo en 6 de 10.
- **Por qué importa:** Es el puente con Champ et al. (2005) y Jensen et al. (2006), que el traspaso llama la contribución más original. Poner «T²: sin efecto» sería falso. Lo robusto en T² es Fase I frente a Fase II. El efecto de la procedencia con un percentil empírico es pequeño y ruidoso, y se mezcla con el de n_cal (DIS-4).
- **Corrección propuesta:** Añadir a A0 los brazos de T²:
- T-F (Fase II);
- T-Beta (Fase I aplicado en Fase II, la versión paramétrica de «en muestra»);
- T-empírico en muestra y T-empírico reservado, con igual n_cal;
- opcionalmente, T-KDE.
Añadir la columna far_teorica_faseII (0,01 para F y 1 − F_cdf(UCL_Beta/c) para Beta). Todo en Rieth con réplicas. Detalle en extra_markdown §1–2.
- **Evidencia:** wf/diseno/t2_fases_clasico.py y .out. Teoría para a = 22–27: cálculo en línea (a = 22: 3,12 %; a = 24: 3,46 %; a = 26: 3,84 %; a = 27: 4,05 %). piloto_rieth.out.

#### DIS-6 · El límite de Jackson–Mudholkar es una calibración en muestra implícita, no un límite inmune

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** Cuaderno 01, celdas [33] y [35] («los paramétricos no lo están»). tfmfdd/pca.py:104-107 (docstring: Jackson «no usa X_cal»). A0:13-15.
- **Qué pasa:** θ₁ es la suma de los autovalores muestrales descartados de X_fit, y el SPE medio en muestra vale exactamente θ₁(m−1)/m (7,5557 frente a 7,5341). Fuera de muestra, el SPE medio sobre X_cal es 11,545, un 53 % mayor.

Por eso Jackson hereda el optimismo: su límite es 15,79, frente a 14,82 de Box en muestra y 23,46 de Box reservado. Su FAR pre-fallo es 22,1 % en el clásico (d00_te: 25,1 %) y 22,6 % en el piloto de Rieth. «No usa X_cal» no quiere decir «no tiene sesgo».
- **Por qué importa:** (1) El material docente del equipo afirma algo falso.
(2) La verificación contra Yin usa Jackson (nota técnica, Tabla 4), así que reproduce un límite del SPE calibrado implícitamente en muestra. Parte del 6,13 % de FAR del PCA de Yin podría venir de ahí. Es una hipótesis: falta correr A1 con --yin-vars --spe-method jackson.
(3) Es una segunda vía del puente Fase I/Fase II: un límite paramétrico con parámetros estimados.
- **Corrección propuesta:** Reformular: «Jackson no usa X_cal, pero estima θ con los autovalores de X_fit y hereda el sesgo en muestra». Incluir Jackson como brazo en A0 y, en la memoria, tratarlo como «calibración en muestra implícita». La teoría del sesgo de los autovalores muestrales pequeños queda [VERIFICAR referencia]. La evidencia empírica sí está (7,53 frente a 11,55).
- **Evidencia:** wf/diseno/t2_fases_clasico.out (bloque SPE: theta1, medias, Jackson 15,7893, FAR 0,2211). piloto_rieth.out (spe_jackson 0,2259).

#### DIS-7 · El factor (c), estandarizar con datos de prueba, no se puede medir en el PCA con el código actual, y en el Random Forest es cero por construcción

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/pca.py:112-113 (el Scaler se ajusta dentro de fit). tfmfdd/preprocessing.py:Scaler. Cuaderno 02, celdas [5] y [7] (kNN y Random Forest).
- **Qué pasa:** Preescalar con las estadísticas del conjunto combinado (X_fit más los 22 ficheros de prueba) produce el mismo modelo: a 27/27, T² 52,6266/52,6266, SPE 23,4563/23,4563 y una diferencia máxima de SPE en el fallo 1 de 6,6e-12. El Scaler interno vuelve a estandarizar con X_fit y anula la fuga. Inyectar el escalador después de fit no recalcula ni las cargas ni los límites.

En los clasificadores, el Random Forest es invariante a transformaciones afines por variable, así que el efecto es cero por diseño. kNN, SVM y FCM sí se ven afectados. En detección, estandarizar con datos que incluyen fallos infla las desviaciones y puede bajar la FAR y la FDR a la vez: el signo no está garantizado (hipótesis no verificada).
- **Por qué importa:** Si se ejecuta tal cual, el factor (c) daría «efecto nulo en PCA», y sería un artefacto del código, como el T² de A0. El eje de la tesis dice «inflan», pero aquí el efecto puede ir en la otra dirección (Vanhatalo y Kulahci, 2016, en 04_Investigacion, piden reportar el signo).
- **Corrección propuesta:** Añadir un gancho: PCAMonitor.fit(..., scaler=None), que acepte un Scaler ya ajustado, o una subclase en el script del experimento. Brazos: limpio (X_fit), combinado (X_fit ∪ prueba) y por simulación (cada simulación de prueba con sus propias estadísticas). Fijar n_components en todos los brazos. En el Random Forest, declarar «cero por construcción» y no medirlo.
- **Evidencia:** Ejecución en línea con el .venv. Salida: «a 27 27 T2 52.6266 52.6266 SPE 23.4563 23.4563» y «max|dif SPE| fallo 1: 6.59e-12».

#### DIS-8 · El factor (b), partición por muestra frente a por simulación, no tiene script, y en el clásico no se puede medir sin confundido

- **Veredicto:** CONFIRMADO. *Matiz:* Es una propuesta de diseño, sin cifra que verificar.
- **Dónde:** No hay script. data.split_runs (data.py:183-209) solo parte por simulación. Cuaderno 02, celda [3] (clásico: dXX.dat frente a dXX_te.dat).
- **Qué pasa:** El clásico tiene una simulación por clase. Juntar dXX.dat con dXX_te.dat y barajar cambia a la vez la partición y la mezcla de etapas del fallo: entrenamiento con 24 h de fallo activo desde la fila 0, prueba con 40 h tras el inicio. En el PCA, el factor aparece como la forma de partir los datos normales entre ajuste y calibración (temporal, barajada, por bloques u otra simulación), y ahí la regla del 85 % elige a sobre X_fit, así que a cambia entre brazos si no se fija.
- **Por qué importa:** Es el OE2 entero (Jonathan) y la fila 3 de la tabla de ablaciones. Sin fijar la estimanda (el sesgo del estimador interno), el experimento compara dos números que no significan lo mismo.
- **Corrección propuesta:** Hacerlo en Rieth.
- Clasificadores: los dos brazos (KFold por muestra y GroupKFold por índice de simulación) con el mismo modelo, los mismos hiperparámetros, el mismo tamaño y el mismo equilibrio de clases. Estimanda: optimismo = estimación interna − desempeño en simulaciones de prueba externas, las mismas para los dos brazos. Dosis-respuesta por tamaño de bloque, como en arXiv 2607.16493, que está en 04_Investigacion. Réplica: partición externa con simulaciones disjuntas (R = 5–10, ≤ 20 simulaciones por clase).
- PCA: brazos temporal, barajado y otra simulación, con n_components fijo.
Tabla completa en extra_markdown §2.
- **Evidencia:** Lectura de data.py y del cuaderno 02, celda [3]. B2 de la revisión (480 filas por fichero de fallo del clásico).

#### DIS-9 · El factor (d), una simulación frente a muchas, no sesga en esperanza: hay que medirlo como dispersión y como tasa de conclusiones erróneas

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** Traspaso v3, §4 («cuánto inflan … una simulación frente a muchas»). configs/baseline.yaml (runs_finales: 100).
- **Qué pasa:** Con el mismo modelo, la métrica de una simulación estima sin sesgo la media sobre simulaciones. La nota técnica de Yin (§5.3) lo dice para la autocorrelación: aumenta la varianza, no la esperanza. El daño está en la varianza, en la selección (quedarse con la mejor semilla) y en la inferencia pseudorreplicada.

Con el mismo modelo, la FAR del SPE reservado va de 0,6 % a 12,5 % entre los 21 segmentos del clásico (sd 2,85 puntos). En Rieth, la sd por simulación de la FDR en 3, 9 y 15 es de unos 0,5 puntos.
- **Por qué importa:** Si la tesis dice «infla», el tribunal puede contestar que la esperanza no cambia. Hay que medir lo que sí cambia.
- **Corrección propuesta:** Estimandas:
(d1) Lado de evaluación: dispersión de la métrica de una sola simulación, y probabilidad de que la comparación con una sola simulación (con un contraste por muestra, como Fisher sobre 800 muestras) contradiga la conclusión de 100 simulaciones (signo o significación).
(d2) Lado de ajuste: 1 simulación frente a K con n fijo ({1×500, 2×250, 5×100, 10×50}).
(d3) Opcional: inflado por «mejor de k semillas».
En el eje de la tesis, cambiar «inflan» por «desplazan o vuelven inestables».
- **Evidencia:** docs/revision_2026-09-27.md §4 (tabla de dispersión de los 21 segmentos). piloto_rieth.out (± de la FDR por fallo).

#### DIS-10 · Contrastes: la familia de Holm, la estimanda, el intervalo y el baseline no están definidos

- **Veredicto:** CONFIRMADO. *Matiz:* Los rangos de líneas están desplazados entre 1 y 2 líneas (paired_test 20-49, holm 52-79, compare 82-122). Se solapa con I2 y MET-8.
- **Dónde:** tfmfdd/stats.py:20-47 (paired_test), 51-80 (holm_correction), 83-121 (compare_methods). decisiones.md:16. Complementa la revisión I2.
- **Qué pasa:** (i) «Holm por fallo» controla el error de familia dentro de cada fallo, no entre los 20 fallos. La afirmación confirmatoria es sobre 3, 9 y 15, y esa familia no está declarada.
(ii) Wilcoxon contrasta la pseudomediana. median_difference es la mediana de las diferencias, que no es el estimador de Hodges–Lehmann. La diferencia mínima relevante está en puntos, sin decir si de media o de mediana. El test, el estimador y el intervalo no se refieren al mismo parámetro.
(iii) No hay intervalo de la diferencia.
(iv) holm_correction devuelve solo la marca, no los p ajustados.
(v) «Baselines lineales» es ambiguo: T², SPE, su disyunción, DPCA.
- **Por qué importa:** Sin estimanda única, el «intervalo de la diferencia» que promete decisiones.md:16 no se puede comparar con la diferencia mínima relevante, y el análisis principal no es el que se declaró.
- **Corrección propuesta:** - Estimanda principal: diferencia media pareada por simulación, en puntos. IC t o bootstrap pareado BCa sobre simulaciones; contraste t pareado o de permutación por cambio de signo. Wilcoxon queda como sensibilidad; si se mantiene como principal, usar Hodges–Lehmann con su IC invertido [VERIFICAR la referencia].
- Familia confirmatoria: {3, 9, 15} × métodos frente al baseline. El resto es exploratorio, con Holm por fallo y rotulado como tal.
- Reportar los p ajustados por Holm y, si se quiere cobertura simultánea, IC de Bonferroni (1 − α/m) [VERIFICAR IC compatibles con Holm].
- Baseline fijado a priori: PCA con disyunción T² ∨ SPE a nivel Šidák por estadístico, para que la nominal global sea 1 % (nota de Yin, §6.2), y DPCA como baseline lineal con memoria.
- **Evidencia:** Lectura de stats.py (líneas citadas) y decisiones.md:16.

#### DIS-11 · La diferencia mínima relevante propuesta es laxa en la FAR, vacía en el retardo y ambigua en la FDR

- **Veredicto:** MATIZADO. *Matiz:* «Vacía en el retardo» es exagerado: la diferencia MEDIA pareada entre simulaciones no es entera, así que un umbral de 1 muestra sí discrimina. Solo carece de sentido por simulación. La comparación con el «techo del 3,2 %» del fallo 3 depende de DIS-15, que no es robusto.
- **Dónde:** Traspaso v3, §8 «Contrastes». CAMBIOS.md:92-94. decisiones.md:16.
- **Qué pasa:** - FAR (2 puntos): con una nominal del 1 %, un 3 % sería «equivalente». Eso contradice el propio mensaje de A0 (3,6 % frente a 1 % es relevante).
- Retardo (1 muestra): es la resolución de la métrica, que es entera. Cualquier diferencia distinta de cero cuenta como relevante, así que el criterio no discrimina.
- FDR (2 puntos): no se dice la escala (cruda o exceso sobre la FAR, media o mediana). En 3, 9 y 15, 2 puntos equivalen a todo el techo estático del fallo 3 (3,2 %), pero son poco frente a las ventajas publicadas (nota de Yin, Tabla 3: mejor − PCA = 11,37, 15,12 y 15,75 puntos).
- **Por qué importa:** La diferencia mínima relevante decide si la hipótesis es trivial o exigente. Tiene que quedar fijada y justificada antes de ver Rieth.
- **Corrección propuesta:** - FAR: escala de cociente (p. ej., ≤ 1,5×) o absoluta de 0,5 puntos; o, mejor, sacar la FAR de las comparaciones igualándola (DIS-13).
- Retardo: diferencia mínima en minutos con un criterio operativo, a decidir con el equipo y el director (no copies mi número); o tratar el retardo de forma descriptiva y con censura.
- FDR: δ sobre el exceso pareado sobre la FAR, justificada frente a las ventajas publicadas prerregistradas.
- Dejarlo en decisiones.md con fecha.
- **Evidencia:** Cifras de Yin: nota técnica, Tabla 3 (restas hechas a mano sobre los valores del PDF). Techo: techo_detectabilidad.out. A0: revisión §4.

#### DIS-12 · Los retardos NaN se descartan del par y eso introduce un sesgo de supervivencia

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/stats.py:26-27 (máscara de NaN) junto con metrics.detection_delay (NaN cuando no se confirma). compare_methods.
- **Qué pasa:** Si alguno de los dos métodos no confirma la detección en una simulación, el par desaparece del contraste. La comparación queda condicionada a que ambos detecten y favorece al método que falla justo en las simulaciones difíciles. Además, las detecciones marcadas detected_at_chance siguen aportando su retardo.
- **Por qué importa:** En los fallos difíciles y transitorios (3, 9, 15, 5), que es donde se juega la hipótesis, el retardo comparado puede cambiar de signo.
- **Corrección propuesta:** Censurar por la derecha en 800 muestras y comparar con un método que admita censura, o reportar la proporción de detección junto con un retardo medio restringido. Tratar los retardos detected_at_chance como censurados. Declararlo en decisiones.md.
- **Evidencia:** Lectura de stats.py:26-27 y metrics.py:79-86.

#### DIS-13 · Comparar métodos sin igualar la FAR ni la memoria confunde «no lineal» con «punto de operación» y con «contexto temporal»

- **Veredicto:** CONFIRMADO. *Matiz:* No reejecuté el valor a = 56 del DPCA: lo leí de piloto_rieth.out.
- **Dónde:** Nota técnica de Yin, §6.1 y §6.6. Cuaderno 03, celda [10] (umbral del autoencoder frente a límites del PCA). Hipótesis aprobada.
- **Qué pasa:** Cada método fija su umbral de una forma distinta y opera a una FAR distinta, y la ventaja en FDR se puede comprar con FAR. La nota de Yin lo documenta: el DPCA tiene 10,13 % de FAR frente a 6,13 % del PCA. Además, los modelos recurrentes y por ventanas usan contexto temporal y el PCA estático no. Una ventaja «no lineal» puede ser en realidad una ventaja por tener memoria, y el techo estático de DIS-15 no acota a los métodos con memoria.
- **Por qué importa:** La hipótesis habla de «no lineales y profundos frente a lineales». Si la comparación no iguala el punto de operación y la memoria, el resultado no se puede atribuir a la no linealidad.
- **Corrección propuesta:** Comparación principal a FAR igualada: el umbral de cada método se calibra al 1 % con simulaciones normales reservadas de entrenamiento, nunca con las de prueba, y se reporta la FAR realizada en prueba. Incluir el DPCA (y, opcionalmente, un lineal con memoria tipo EWMA sobre el PCA [VERIFICAR]) como baseline lineal con memoria. ROC o AUC parcial como análisis secundario.
- **Evidencia:** Nota técnica de Yin, §6.1 y §6.6 (en 04_Investigacion). Piloto: el DPCA con lags = 2 retiene a = 56 (piloto_rieth.out).

#### DIS-14 · Con una sola simulación, todo contraste es pseudorreplicación: el clásico solo admite descripción

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** Cuaderno 01, celda [28] (Fisher del fallo 5). CSV de A1 (simulationRun = 1). README:97. Revisión I3 e I6.
- **Qué pasa:** Dentro de una simulación, las muestras son las únicas «réplicas». Un contraste por muestra responde sobre esa realización, no sobre el método, y además supone una independencia que la autocorrelación rompe. La corrección por n efectivo de la celda [28] solo se aplica a un grupo (revisión I3). En Rieth pasa lo mismo si se agregan (fallo, simulación) como unidades (DIS-2).
- **Por qué importa:** Cualquier «X es mejor que Y» que salga del clásico es atacable. La verificación contra Yin se presenta hoy como prueba de corrección (README:97).
- **Corrección propuesta:** El clásico es solo descriptivo: verificación contra Yin con una tolerancia declarada de antemano, en los fallos detectables. Toda inferencia se hace en Rieth, con el índice de simulación como unidad. Retirar de la memoria el Fisher del fallo 5 o dejarlo como ilustración de por qué no se puede contrastar con una sola simulación.
- **Evidencia:** Cuaderno 01, celda [28], salida ejecutada (p = 0,2329, ρ₁ = 0,158, n_ef ≈ 291). Código de la celda.

#### DIS-15 · La parte «3, 9 y 15» corre el riesgo de ser trivialmente cierta: no hay casi nada que detectar

- **Veredicto:** MATIZADO. *Matiz:* Lo cualitativo se mantiene (el PCA no ve casi nada en 3, 9 y 15), pero el número del techo depende de un corte numérico y la afirmación «la cota es generosa» no se sostiene. No sirve para decir que «no hay casi nada que detectar»: queda señal en las direcciones de los lazos de control. Los controles positivos (ii) siguen siendo una buena idea.
- **Dónde:** Hipótesis, parte de 3, 9 y 15. metrics.HARD_FAULTS. Nota de Yin, §2.5 (explicación física).
- **Qué pasa:** En Rieth, las alarmas del SPE en 3, 9 y 15 coinciden con las de la simulación normal en el 99,6–100 % de las muestras. Calculé un techo orientativo por muestra con un oráculo gaussiano que conoce la dirección del efecto: D(t)² = d'Σ⁻¹d, con d = X_f(r) − X_0(r). Con FAR del 1 %, da FDR de 3,16 % (fallo 3), 5,14 % (9) y 18,45 % (15), con medianas de D de 0,45, 0,63 y 1,24. En contraste, da 0,96 en el 10 y 1,00 en el 1 y el 5. Σ tiene número de condición 1,9e10, así que la cota es generosa. Los métodos con memoria pueden superar este techo estático si el efecto persiste (el fallo 3 es un escalón).
- **Por qué importa:** Si ningún detector estático puede superar unos pocos puntos, «no hay ventaja en 3, 9 y 15» queda garantizado por la física, y el tribunal puede decir que esa parte de la hipótesis no informa de nada.
- **Corrección propuesta:** (i) Presentar el techo como contexto y reencuadrar: una ventaja publicada por encima del techo señala un artefacto del protocolo.
(ii) Controles positivos elegidos a priori con una regla, por ejemplo fallos con FDR del PCA-SPE entre 0,2 y 0,8 en A1 clásico, sin el 21 (no existe en Rieth) ni el 5 (compensado): 10, 11, 16 y 20. Así se demuestra que el diseño es capaz de detectar ventajas.
(iii) Un detector trivial como control (Kim et al., 2022, y Hartung et al., en 04_Investigacion).
(iv) Un techo por ventanas para los métodos con memoria.
- **Evidencia:** wf/diseno/techo_detectabilidad.py y .out, y semillas_comunes.out (scratch). Es exploratorio: supone ruido gaussiano y dirección conocida. FDR del A1 clásico: revisión §3.3.

#### DIS-16 · «Menor que la publicada» necesita una tabla prerregistrada de ventajas publicadas, y hay que separar el efecto del protocolo del efecto del conjunto de datos

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** Hipótesis («menor que la publicada»). Estado del arte (Lyu excluye 3, 9 y 15; Hartung usa el mejor F1 con ajuste sobre la prueba).
- **Qué pasa:** Las ventajas publicadas vienen de datos distintos (el clásico con una simulación, las particiones de Lyu, el Rieth de Hartung con ajuste sobre la prueba) y de métricas distintas (exactitud, F1, FDR). Comparar nuestra Δ (Rieth, FDR a FAR igualada) con la publicada mezcla protocolo, conjunto de datos y métrica. Lyu excluye 3, 9 y 15 (04_Investigacion), así que para esos fallos puede no existir una ventaja profunda publicada con la que comparar.
- **Por qué importa:** Sin un referente fijado antes de ejecutar, la comparación se puede elegir a posteriori. Es el mismo grado de libertad que el TFM critica.
- **Corrección propuesta:** Prerregistrar una tabla con fuente, conjunto de datos, métrica, fallo y Δ publicada, marcada [VERIFICAR] hasta abrir cada PDF. Calcular nuestra Δ también en el clásico (los mismos datos que las publicaciones) y en Rieth, para descomponer protocolo y conjunto de datos. Contraste unilateral H0: Δ_nuestra ≥ Δ_publicada.
- **Evidencia:** Estado_del_arte_efecto_del_protocolo_TEP.md (Lyu: «Fallos: 17 de 20 (excluye 3, 9 y 15)»; Hartung: «best F1-score»). Nota de Yin, Tabla 3.

#### DIS-17 · Prerregistro: fijar la operacionalización antes de ejecutar Rieth y apartar las simulaciones que ya se han visto

- **Veredicto:** MATIZADO. *Matiz:* La lista está incompleta: ya se han visto las simulaciones 1-20 de los 42 ficheros y las 21-40 de train_fault00. Para lo confirmatorio hay que usar las simulaciones 21 o posteriores en todos los ficheros (41 o posteriores en train_fault00), no solo en los que cita.
- **Dónde:** Revisión I8. Datos de Rieth usados en mi piloto.
- **Qué pasa:** La diferencia mínima relevante, la familia, la métrica principal, n, la regla de parada y las listas de fallos no están fijadas. Además, mi piloto ad hoc ya miró datos de Rieth:
- test_fault00/01/03/09/15, simulaciones 1–20;
- test_fault05/10/20, 1–5;
- test_fault02/06, 1–2;
- train_fault00, 1–40;
- train_fault06, 1–2.
- **Por qué importa:** Si esas simulaciones entran en el análisis confirmatorio, el prerregistro pierde valor, porque ya condicionaron decisiones (entre otras, esta revisión).
- **Corrección propuesta:** Commit fechado con etiqueta (p. ej. prereg-rieth) que contenga la operacionalización de extra_markdown §3 antes de ejecutar. Análisis confirmatorio con test ≥ 21 y train_fault00 ≥ 41, o declarar el solape. Análisis de sensibilidad de min_fdr, como pide I8.
- **Evidencia:** Los scripts piloto_rieth.py, semillas_comunes.py y techo_detectabilidad.py (scratch) declaran qué simulaciones cargan.

#### REP-1 · El Python global no tiene tfmfdd instalado: lo importa desde el directorio de trabajo (contradice I7)

- **Veredicto:** CONFIRMADO. *Matiz:* El egg-info no está «obsoleto»: se generó hoy a las 09:01:32 con el pip install -e del .venv, refleja el pyproject actual (incluye pyyaml) y lista LICENSE.zip como License-File. Borrarlo no tiene consecuencias, porque se regenera. La fecha de site-packages solo descarta una desinstalación posterior al 26/09 a las 13:01, no una anterior.
- **Dónde:** Entorno: C:/Users/pabdu/AppData/Local/Programs/Python/Python312 (Lib/site-packages y Scripts/pytest.exe); tfmfdd.egg-info en la raíz del repo; docs/revision_2026-09-27.md:88-90 (I7)
- **Qué pasa:** I7 afirma que el Python global tiene tfmfdd instalado en modo editable y propone desinstalarlo, pero no hay nada instalado: ni dist-info, ni .pth, ni egg-link, ni nada en el site de usuario. El global solo importa tfmfdd cuando el directorio actual es la raíz del repo, que es justo lo que ocurre en los cuadernos tras os.chdir(".."). Además, importlib.metadata informa 0.1.0 porque encuentra el tfmfdd.egg-info obsoleto que hay en la raíz. Resultado: 'python -m pytest' con el global da 31 passed usando las bibliotecas del global (numpy 2.4.6, pandas 3.0.3, scipy 1.18.0, scikit-learn 1.9.0, matplotlib 3.11.0), y el 'pytest' a secas, que Git Bash resuelve a Python312/Scripts/pytest.exe, falla al recoger las pruebas.
- **Por qué importa:** La corrección de I7 (desinstalar) no tiene ningún efecto, y '31 passed' no dice con qué entorno se ejecutó. Cualquier intérprete que arranque en la raíz, o un cuaderno tras el chdir, usa el código fuente con las bibliotecas que tenga ese intérprete, y no avisa.
- **Corrección propuesta:** No hay nada que desinstalar. Ejecuta siempre con '.venv/Scripts/python.exe -m ...' (pytest, nbconvert, pip) y los scripts con ese mismo intérprete. Añade un tests/conftest.py con un hook pytest_report_header que imprima sys.executable, tfmfdd.__file__ y las versiones de numpy, pandas, scipy y scikit-learn. Pon en cada cuaderno una primera celda que imprima lo mismo. El tfmfdd.egg-info de la raíz está ignorado y es obsoleto: puedes borrarlo. Corrige I7 en el informe.
- **Evidencia:** Python global, 'python -m pip show tfmfdd' desde la raíz del repo y desde scratch: 'WARNING: Package(s) not found: tfmfdd'. 'import tfmfdd' desde scratch y desde notebooks/: ModuleNotFoundError. Tras os.chdir('..') importa ...\tfm-fdd-core\tfmfdd\__init__.py. Desde la raíz, importlib.metadata.distribution('tfmfdd')._path = 'tfmfdd.egg-info', versión 0.1.0. Los únicos .pth de Lib/site-packages son a1_coverage.pth, distutils-precedence.pth y pywin32.pth. El directorio site-packages se modificó por última vez el 2026-09-26 a las 13:01:20 (instalación de lxml), así que no hay rastro de una desinstalación hecha hoy. En el clon de scratch, Python312/Scripts/pytest.exe da 'ModuleNotFoundError: No module named tfmfdd' en los 2 módulos de prueba, y 'Python312/python.exe -m pytest' da '31 passed in 2.37s'. Las versiones del global salen de 'pip list'.

#### REP-2 · Las salidas de los cuadernos de v0.2, y las del 03 que irían en v0.3, salen de Linux con Python 3.12.3, no de tu PC

- **Veredicto:** CONFIRMADO. *Matiz:* Lo de «Linux» es una inferencia a partir de la ruta POSIX. Lo seguro es que 3.12.3 no está en este PC, ni en Windows ni en WSL. El rastro que se conserva es solo la versión de Python y la plataforma, no las de numpy ni sklearn, así que en eso I7 tenía razón. Decir que el entorno no se puede reconstruir es exagerado: Python 3.12.3 se puede instalar; lo que no se puede reconstruir son las versiones de las bibliotecas.
- **Dónde:** v0.2:notebooks/*.ipynb (metadata.language_info.version) y v0.2:notebooks/01_primeros_pasos.ipynb celda [1]; notebooks/03_arranque_deeplearning.ipynb del working tree; docs/revision_2026-09-27.md:89
- **Qué pasa:** I7 dice que no hay registro de con qué versiones se generaron las salidas de v0.2, y atribuye los resultados anteriores al Python global. Sí hay rastro. Los tres cuadernos de v0.2 registran Python 3.12.3, y la celda [1] del 01 imprime un directorio de trabajo /tmp/repo/tfm-fdd-core. En tu PC solo existe Python 3.12.0, tanto el global como el del .venv. Hoy los cuadernos 01 y 02 ya registran 3.12.0 porque el revisor los volvió a ejecutar, pero el 03 conserva las salidas de 3.12.3: se ejecutó una copia y el original no se tocó.
- **Por qué importa:** Si etiquetas v0.3 así, el 03 (que contiene la cifra 17,7/2,8 del autoencoder, M9) llevará salidas de un entorno que ni tú ni tus compañeros pueden reconstruir. Es la pregunta de la defensa («¿con qué versión se generó esto?») y seguiría sin respuesta.
- **Corrección propuesta:** Antes de etiquetar, ejecuta el 03 en su sitio con '.venv/Scripts/python.exe -m nbconvert --to notebook --execute --inplace' (fase F de la secuencia del anexo) y guarda results/summary/ENTORNO.txt. Corrige también I7: las salidas de v0.2 no son del Python global, sino de un entorno Linux externo.
- **Evidencia:** Leí el JSON de cada cuaderno. En v0.2, los cuadernos 01, 02 y 03 tienen language_info.version '3.12.3'. En v0.2, la celda [1] del 01 tiene como stdout 'Directorio de trabajo: /tmp/repo/tfm-fdd-core'. En el working tree: 01 y 02 tienen 3.12.0 y 03 tiene 3.12.3. 'py -0p' solo lista '-V:3.12 * ...\Python312\python.exe', y .venv/pyvenv.cfg dice version = 3.12.0. Esto encaja, como inferencia y no como prueba, con que los commits locales estén firmados por 'pablo@local' con zona horaria +0000.

#### REP-3 · GitHub manda hoy instalar @v0.1, que sí funciona; esa etiqueta es de tu historia local, y main trae el mismo código que tu working tree

- **Veredicto:** CONFIRMADO. *Matiz:* Lo de «idénticos byte a byte» vale para los blobs de git. Instalado con pip en Windows (con el autocrlf=true del sistema), los .py llegan con CRLF (36-245 CRLF por fichero), así que solo son iguales tras normalizar los fines de línea. No cambia la conclusión.
- **Dónde:** README.md:24 y :27 en origin/main; etiqueta remota refs/tags/v0.1; docs/revision_2026-09-27.md:23-27 (B1)
- **Qué pasa:** Tres precisiones a B1. (1) El README que se ve en GitHub (el de origin/main) dice 'pip install git+https://github.com/pabdus/tfm-fdd-core@v0.1', con 'Fijad' en forma de vosotros. Esa instalación funciona y entrega el paquete del 17/09: sin convert_rieth.py, con values(df) sin el parámetro columns, sin stratified_fdr y con el compare_methods antiguo, que según CAMBIOS.md:45-47 desalineaba los pares cuando había varios fallos. (2) La v0.1 remota apunta a bf6cb4f, que es la raíz de tu historia local, no la de main. Las dos historias «sin relación» son copias: 19a1603 y bf6cb4f tienen el mismo árbol, igual que 0f7a4be y b3b530d; solo cambian la fecha y la zona horaria. (3) Instalar @main hoy da un paquete cuyos 9 módulos son idénticos byte a byte a los de tu working tree. Eso incluye FAULT_ONSET_TRAIN_CLASSIC = 20 (B2). La rueda no lleva .pyc y la versión es 0.1.0, así que no se distingue de v0.1 por __version__. Por tanto, el paquete de main sí coincide con un estado tuyo; lo que falta es un commit o una etiqueta que lo contenga.
- **Por qué importa:** Jonathan o Cesar pueden estar trabajando ya con v0.1 o con main sin saberlo, y tfmfdd.__version__ les dará 0.1.0 en los tres casos. Con v0.1 no pueden convertir Rieth, y compare_methods les da pares desalineados.
- **Corrección propuesta:** Pídeles hoy la salida de 'python -m pip freeze | grep -i tfmfdd', que muestra la URL y el commit de lo que tienen instalado. Si tienen v0.1, que rehagan lo que hayan calculado con ella. Tras la unificación, v0.1 queda como ancestro de main (comprobado en el simulacro). El README que va dentro de v0.3 debe decir @v0.3 antes de etiquetar: las líneas 24 y 27 del working tree todavía dicen @v0.2.
- **Evidencia:** 'git ls-remote origin' devuelve a3e0eac HEAD, a3e0eac refs/heads/main y bf6cb4f refs/tags/v0.1. 'git diff master origin/main -- README.md' muestra '+pip install git+https://github.com/pabdus/tfm-fdd-core@v0.1'. Con 'git rev-parse <c>^{tree}': bf6cb4f y 19a1603 dan 5180a488b2; b3b530d y 0f7a4be dan db5b2fb8d9. Las fechas son 2026-09-17 13:59:59 +0000 frente a 09:51:51 -0500. 'git ls-tree -r --name-only v0.1' no incluye convert_rieth.py, y 'git grep' sobre v0.1 da 'def values(df: pd.DataFrame)'. En el clon, los 9 tfmfdd/*.py de origin/main tienen el mismo blob que los de la rama. La rueda construida desde origin/main es tfmfdd-0.1.0-py3-none-any.whl, con 'pyc: []', 9 módulos y un Requires-Dist sin pyyaml. En el simulacro, 'git merge-base --is-ancestor v0.1 HEAD' devuelve sí.

#### REP-4 · El protocolo común (configs/baseline.yaml) no viaja con el paquete, y lo que sí viaja trae otros valores por defecto

- **Veredicto:** CONFIRMADO. *Matiz:* Sin matiz relevante. Hoy no altera ninguna cifra, porque los valores de reserva coinciden con el yaml.
- **Dónde:** pyproject.toml:28-29; configs/baseline.yaml:1-3; tfmfdd/pca.py:52 y :133; tfmfdd/__init__.py:14; experiments/A1_pca_baseline.py:30-42; experiments/A0_calibracion_limites.py:38-49
- **Qué pasa:** El paquete solo contiene tfmfdd/*.py. Quien instala por pip, como Jonathan y Cesar, no recibe baseline.yaml, aunque el propio fichero dice que los tres bloques parten de él. Lo que sí reciben es un PCAMonitor con variance=0.90 por defecto (el yaml dice 0.85) que, si no le pasan X_cal, calibra en muestra: la práctica que A0 mide como inflada (FAR del SPE 0,2738 frente a 0,0357). El ejemplo del docstring del paquete, que es lo que muestra help(tfmfdd), usa precisamente 0.90 y no pasa X_cal. Además, A0 y A1 hacen 'try: import yaml' y devuelven {} si falta pyyaml o si el yaml no está en la ruta relativa al directorio actual; en ese caso siguen con los valores escritos en el código, sin avisar.
- **Por qué importa:** decisiones.md:14 fija el yaml como única fuente de los valores por defecto. Hoy los valores de reserva del código coinciden con el yaml y las cifras no cambian, pero cualquier cambio futuro del yaml se ignoraría en silencio al ejecutar desde otra carpeta, y los otros dos bloques no tienen forma de leerlo.
- **Corrección propuesta:** Mueve el yaml a tfmfdd/configs/baseline.yaml como dato del paquete ([tool.setuptools.package-data] tfmfdd = ["configs/*.yaml"]) y léelo con importlib.resources desde una función tfmfdd.config.cargar(). En A0 y A1, quita el try/except (pyyaml ya es dependencia obligatoria) y haz que fallen si no encuentran el fichero. Escribe en cada CSV la ruta y un hash del yaml usado. Haz que PCAMonitor tome variance del yaml o la exija como argumento, y que avise si X_cal es None. Corrige el ejemplo de tfmfdd/__init__.py.
- **Evidencia:** La instalación del paso d ($SIM/inst) solo contiene los 9 tfmfdd/*.py y el dist-info, e importlib.resources.files('tfmfdd') da 'configs empaquetados: False'. grep: pca.py:52 'variance: float = 0.90'; pca.py:133 'if X_cal is not None:'; __init__.py:14 'PCAMonitor(variance=0.90, alpha=0.99).fit(data.values(normal))'; baseline.yaml 'variance: 0.85'. Ejecuté A1 con el venv del lock desde $SIM/neutral, que no tiene configs/: exit=0 y ninguna línea menciona yaml, config ni protocolo. Imprime 27 componentes al 85 % porque el valor de reserva del código también es 0.85. Las FAR 0.273810 y 0.035714 salen de A0 ejecutado en el venv del lock.

#### REP-5 · La construcción de v0.3 dejará de estar soportada el 18/02/2027, en plena ventana de defensa

- **Veredicto:** CONFIRMADO. *Matiz:* La fecha anuncia el fin del soporte, no una rotura garantizada ese día. El riesgo es que una setuptools posterior elimine la forma de tabla y pip la use por la cota abierta. La corrección propuesta resuelve además la inclusión de LICENSE.zip, que el informe no recoge.
- **Dónde:** pyproject.toml:2 (requires = ["setuptools>=61"]) y pyproject.toml:11 (license = {text = "MIT"})
- **Qué pasa:** pip construye el paquete desde el código fuente con la setuptools más reciente, porque la cota solo es inferior. Hoy esa versión es setuptools 84.0.0, que avisa de que la licencia en forma de tabla TOML está obsoleta y fija el 18/02/2027 como fecha a partir de la cual esas construcciones dejarán de estar soportadas.
- **Por qué importa:** El depósito es el 10/02/2027 y la defensa, entre 30 y 90 días después (traspaso v3, sección 2). Una etiqueta no se cambia: si v0.3 lleva la licencia en forma de tabla, 'pip install ...@v0.3' puede fallar justo cuando haya que reproducir una tabla ante el tribunal.
- **Corrección propuesta:** Antes de etiquetar, cambia a license = "MIT", license-files = ["LICENSE"] y requires = ["setuptools>=77,<90"] (pyproject propuesto en la sección 5 del anexo). Lo probé: construye sin ningún aviso de obsolescencia y publica License-Expression MIT.
- **Evidencia:** '.venv/Scripts/python.exe -m pip wheel --no-deps -v' sobre el clon ($SIM/pip_wheel_v.log) muestra 'Using cached setuptools-84.0.0-py3-none-any.whl', tres veces 'SetuptoolsDeprecationWarning: `project.license` as a TOML table is deprecated', y 'By 2027-Feb-18, you need to update your project and remove deprecated calls'. Con el pyproject propuesto ($SIM/pip_wheel_fix.log) hay 0 líneas con 'deprecat', y una vez instalado da '0.3.0 0.3.0 MIT >=3.12'.

#### REP-6 · requires-python >=3.10 promete compatibilidades que el lock no puede cumplir: las versiones fijadas exigen 3.12

- **Veredicto:** CONFIRMADO. *Matiz:* requirements-lock.txt no existe en el repo: es una propuesta de la revisión. Lo que pasaría con 3.10 o 3.11 se deduce de Requires-Python, pero no se ha probado.
- **Dónde:** pyproject.toml:10 (requires-python = ">=3.10")
- **Qué pasa:** numpy 2.5.3, scipy 1.18.1 y contourpy 1.4.0 exigen Python >=3.12, y pandas 3.0.6 exige >=3.11. Con Python 3.10 o 3.11, 'pip install tfmfdd' resuelve en silencio versiones más antiguas de numpy, pandas y scipy, y 'pip install -r requirements-lock.txt' falla con 'No matching distribution'.
- **Por qué importa:** No sabes qué Python tienen Jonathan y Cesar. Otra combinación de numpy o scikit-learn puede mover cifras (no lo he comprobado entre versiones, precisamente porque hoy no hay nada fijado) y el protocolo común deja de ser común.
- **Corrección propuesta:** Pon requires-python = ">=3.12" en pyproject.toml y exige Python 3.12.x en las instrucciones (sección 4 del anexo). Lo probé: el pyproject propuesto publica Requires-Python >=3.12.
- **Evidencia:** Campo Requires-Python en los metadatos del .venv: numpy 2.5.3 >=3.12, scipy 1.18.1 >=3.12, contourpy 1.4.0 >=3.12, pandas 3.0.6 >=3.11, scikit-learn 1.9.1 >=3.11, matplotlib 3.11.2 >=3.11. El lock se instaló y probó con CPython 3.12.0 en Windows y se resolvió con dry-run para CPython 3.12 en Linux. No lo probé en 3.10 ni en 3.11.

#### REP-7 · Los datos no tienen procedencia versionada: ni script de descarga, ni sumas de comprobación, y su documentación está ignorada

- **Veredicto:** CONFIRMADO. *Matiz:* Sin matiz.
- **Dónde:** .gitignore:1-2; README.md:43-45; data/tep_classic/LEEME_TEP_clasico.md
- **Qué pasa:** 'data/' se ignora entero, así que LEEME_TEP_clasico.md, README.md y LICENSE de data/tep_classic, que B2 cita como documentación del equipo, no están en git: no hay ningún fichero versionado bajo data/. El comentario de .gitignore:1 promete «el script que los descarga», pero ese script no existe. README.md:43 no dice de dónde sale el TEP clásico. No hay sumas de comprobación, así que nada garantiza que los tres bloques trabajen con los mismos bytes.
- **Por qué importa:** El protocolo común empieza por los datos. Si un compañero descarga otra copia del TEP clásico, o convierte Rieth con otra versión de pyreadr o de pandas, sus cifras pueden diferir sin que nadie lo detecte.
- **Corrección propuesta:** Mueve los documentos de datos a docs/datos/ (o añade excepciones en el .gitignore). Versiona docs/datos/CHECKSUMS.sha256; ya está calculado en $SIM/CHECKSUMS_datos.sha256 (48 líneas, rutas relativas a la raíz), y cada uno lo comprueba con 'sha256sum -c'. Indica en el README la URL de origen del TEP clásico. Para Rieth, reparte los Parquet ya convertidos (C:/Users/pabdu/datos_tfm/tep_rieth: 42 ficheros, 1,3 GB) con sus sumas, en vez de que cada uno haga su propia conversión.
- **Evidencia:** 'git check-ignore -v data/tep_classic/LEEME_TEP_clasico.md' devuelve '.gitignore:2:data/', y 'git ls-files data | wc -l' devuelve 0. Busqué 'dataverse|descarg|download|urllib|sha256|md5|checksum' en *.py, *.md y *.yaml: solo hay menciones en texto (README.md:45, convert_rieth.py:119 y :240, data.py:170), ningún script. 'sha256sum data/tep_classic/*.dat data/tep_rieth/*.RData' da 48 sumas en 4,9 s; por ejemplo, d00.dat empieza por 4c3c0b11eefa. Los ficheros son locales: sus atributos son solo 'Archive', no están únicamente en la nube.

### 2.3 Menors (21)

| ID | Veredicto | Título |
|---|---|---|
| MET-5 | CONFIRMADO | fdr_by_fault dibuja el IC de la FDR aunque cambies value, y fija el eje y la etiqueta |
| MET-6 | CONFIRMADO | plots.save sobrescribe en silencio si el nombre lleva un punto |
| MET-7 | CONFIRMADO | Jackson–Mudholkar con h0 < 0 devuelve un límite por debajo de la media del SPE (latente en el TEP) |
| MET-8 | CONFIRMADO | holm_correction cuenta los p NaN en la familia y devuelve umbrales en vez de p ajustados |
| MET-9 | CONFIRMADO | paired_test: effect_size NaN cuando el efecto es nulo, statistic 0,0 en la rama de iguales y método de Wilcoxon implícito |
| MET-10 | CONFIRMADO | limit_kde corta la cola superior cuando el estadístico es negativo |
| MET-11 | CONFIRMADO | Sin validar alpha ni variance, y los NaN pasan como «sin alarma»: métricas limpias a partir de entradas rotas |
| MET-12 | CONFIRMADO | El núcleo numérico que más falla no tiene pruebas |
| DAT-4 | CONFIRMADO | load_rieth lee el fichero entero antes de filtrar: runs no reduce el pico transitorio |
| DAT-5 | CONFIRMADO | Las columnas del contrato coinciden, pero los tipos no: Rieth devuelve int16/float32 y la aritmética con los ids desborda sin aviso |
| DAT-6 | CONFIRMADO | load_rieth no valida runs ni split |
| DAT-7 | CONFIRMADO | La comprobación posterior a la conversión es mínima y no hay pruebas de Rieth |
| DIS-18 | CONFIRMADO | El factor (a) tiene dos estimandas legítimas y conviene nombrarlas por separado |
| DIS-19 | CONFIRMADO | La cifra de la demostración sintética del corte barajado no tiene script |
| DIS-20 | MATIZADO | load_rieth lee el fichero entero antes de filtrar las simulaciones |
| REP-8 | CONFIRMADO | pip freeze atribuye el .venv a un commit que no está en GitHub y oculta que el árbol tiene cambios sin commit |
| REP-9 | CONFIRMADO | La configuración de git que originó la división sigue activa |
| REP-10 | CONFIRMADO | Las instrucciones de desarrollo del README no funcionan tal cual en Windows y arrastran restos |
| REP-11 | CONFIRMADO | OneDrive no explica los 109 s de las pruebas: hoy tardan 1,5 s en la misma carpeta |
| REP-12 | CONFIRMADO | Fines de línea mixtos y sin .gitattributes |
| REP-13 | MATIZADO | No hay integración continua, y un control de etiqueta habría detectado M1 |

#### MET-5 · fdr_by_fault dibuja el IC de la FDR aunque cambies value, y fija el eje y la etiqueta

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/plots.py:77-78 (valores por defecto), 93-97 (yerr con clip), 103-104 (ylabel e ylim fijos)
- **Qué pasa:** err_low y err_high no dependen de value. Si llamas con value="fdr_h2_mean", las barras son de fdr_h2 pero el intervalo es el de fdr, y el `.clip(lower=0)` oculta la longitud negativa en lugar de avisar. Además, ylim (0; 1,02) y la etiqueta «Tasa de deteccion» son fijos: con delay_samples_mean, una barra de 673 queda cortada en 1,02 bajo esa etiqueta.
- **Por qué importa:** Genera una figura de la memoria con los intervalos de otra métrica, sin ningún error. Hoy el cuaderno 01, celda [32], usa los valores por defecto (correcto), pero FDR por mitades, FAR y retardo son justo las siguientes figuras previstas.
- **Corrección propuesta:** Derivar los nombres de las columnas del IC a partir de value (value.replace('_mean', '_ci_low'/'_ci_high')) y lanzar un error si no existen. Pasar ylabel e ylim como parámetros (ylim solo para tasas). Quitar el clip o comprobar err_low ≤ value ≤ err_high.
- **Evidencia:** scratch/wf/metricas/t06_plots.py: con value="fdr_h2_mean", el fallo 1 tiene barra 0,000 con IC dibujado [0,000; 0,500], cuando su IC real de fdr_h2 es [0,000; 0,000] (el [0,500; 0,500] es el de fdr). En los fallos 2 y 3, el IC dibujado coincide exactamente con el de fdr. Con value="delay_samples_mean", alturas [0,0; 0,3; 673,0], ylim (0,0; 1,02) y ylabel «Tasa de deteccion».

#### MET-6 · plots.save sobrescribe en silencio si el nombre lleva un punto

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/plots.py:127 (save)
- **Qué pasa:** `path.with_suffix(f".{fmt}")` sustituye lo que haya tras el último punto del nombre. «A0_var0.85» y «A0_var0.90» terminan los dos en «A0_var0.pdf» y «A0_var0.png», y la segunda figura sobrescribe a la primera.
- **Por qué importa:** Es habitual poner la configuración en el nombre de la figura (varianza, alpha). Si pasa, la memoria cita una figura que corresponde a otra configuración, justo el tipo de error que decisiones.md (22-09) quiere evitar.
- **Corrección propuesta:** Usar path.with_name(path.name + f".{fmt}") cuando el sufijo no sea un formato conocido, o rechazar nombres con puntos.
- **Evidencia:** scratch/wf/metricas/t06_plots.py: dos llamadas a save con A0_var0.85 y A0_var0.90 (formato png) dejan un único fichero, ['A0_var0.png'].

#### MET-7 · Jackson–Mudholkar con h0 < 0 devuelve un límite por debajo de la media del SPE (latente en el TEP)

- **Veredicto:** CONFIRMADO. *Matiz:* Hoy es latente, como el propio hallazgo declara.
- **Dónde:** tfmfdd/limits.py:90-103 (en particular la línea 96: sqrt(2·θ2·h0²) = |h0|·sqrt(2θ2))
- **Qué pasa:** El término c_α·sqrt(2θ2h0²)/θ1 pierde el signo de h0. Con h0 < 0, la transformación (Q/θ1)^h0 es decreciente y el resultado corresponde a un cuantil inferior. La guarda solo cubre |h0| < 1e-10. Cambiar solo el signo tampoco basta, porque la aproximación es mala con h0 < 0.
- **Por qué importa:** No afecta a ninguna cifra actual: en todas las configuraciones probadas del TEP, h0 es positivo. Pero con otros conjuntos de variables o con pocas componentes retenidas, el límite saldría absurdo (por debajo de E[SPE] = θ1) sin ningún aviso.
- **Corrección propuesta:** Si h0 ≤ 0 (o muy pequeño), lanzar ValueError o volver a Box con un aviso. Añadir pruebas: autovalores iguales (h0 = 1/3) comparados con Monte Carlo, y un caso con h0 < 0 que debe fallar o avisar.
- **Evidencia:** scratch/wf/metricas/t03_jackson.py (Monte Carlo con 400 000 muestras de Σλ·chi2(1)): con autovalores descartados [1,0] + [0,1]×50, h0 = −0,867 y θ1 = 6,0; tfmfdd da 3,368 frente a un q99 de Monte Carlo de 11,960 (la versión con signo da 13,815). Con [2, 1] + [0,02]×200, h0 = −0,628; tfmfdd da 2,817 frente a 18,784. Con autovalores iguales (h0 = 1/3), 18,796 frente a 18,761 (correcto). TEP clásico, d00[:350]: h0 = 0,2770 (52 variables, 27 componentes), 0,2465 (33, 15) y 0,2478 (33, 17). DPCA con lags 1-2 y 85/90/95 %: entre 0,196 y 0,245. Barrido a = 0..51 sobre 52 variables: el mínimo es −0,0499, solo en a = 0.

#### MET-8 · holm_correction cuenta los p NaN en la familia y devuelve umbrales en vez de p ajustados

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/stats.py:59-79 (holm_correction); stats.py:115-121 (compare_methods le pasa los p NaN)
- **Qué pasa:** paired_test devuelve p = NaN cuando n < 5, por ejemplo con retardos donde ambos métodos confirman en menos de 5 simulaciones. np.argsort manda el NaN al final, pero n lo cuenta: con p = [0,02; 0,03; NaN] nada sale significativo, y sin el NaN los dos lo son. Además, la tabla trae threshold pero no el p ajustado: con p = [0,001; 0,03; 0,04], la fila de 0,04 tiene threshold 0,05 y significant = False. Es correcto por el paso descendente, pero leído en un CSV parece un error y alguien lo «corregirá». Los empates se tratan bien.
- **Por qué importa:** Los pares no contrastables inflan la familia en silencio y vuelven Holm más conservador que el análisis declarado, que se suma a lo que ya dice I2. La columna threshold invita a una lectura errónea de la significación.
- **Corrección propuesta:** Excluir de la familia los p NaN y marcarlos como «no contrastable», o declararlo de forma explícita. Devolver el p ajustado de Holm, p_adj(i) = max_{j≤i} min(1, (n−j+1)·p(j)), además del umbral o en su lugar.
- **Evidencia:** scratch/wf/metricas/t08_holm.py: [0,02; 0,03; NaN] da umbrales 0,0167 / 0,025 / 0,05 y ningún significativo; [0,02; 0,03] da 0,025 / 0,05 y los dos significativos. Con [0,001; 0,03; 0,04], la fila de 0,04 tiene threshold 0,05 y significant = False. Con [0,02]×3, ninguno es significativo (correcto: p ajustado 0,06). Lista vacía: DataFrame vacío, sin error. En compare_methods con retardos, los pares ae–dpca y ae–pca tienen n = 3, p = NaN, y aun así cuentan en la familia.

#### MET-9 · paired_test: effect_size NaN cuando el efecto es nulo, statistic 0,0 en la rama de iguales y método de Wilcoxon implícito

- **Veredicto:** CONFIRMADO. *Matiz:* El p = 7,74e-06 del informe depende de sus datos: con los míos, el exacto da 1,9e-06. La conclusión no cambia.
- **Dónde:** tfmfdd/stats.py:33-35 (rama allclose), 37 (wilcoxon sin method), 40-41 (z a partir de p)
- **Qué pasa:** (a) `z = norm.ppf(1 - p/2) if 0 < p < 1 else nan`: con diferencias simétricas (efecto exactamente nulo), p = 1,0 y r = NaN, mientras que la rama allclose devuelve r = 0,0. (b) La rama allclose devuelve statistic = 0,0, que en la convención de scipy (min(W+, W−)) es el valor de máxima evidencia: con 20 simulaciones idénticas, scipy con zsplit da 105. En la tabla, ese 0,0 coincide con el de «20 de 20 empeoran» (p = 7,7e-6). (c) No se fija method. scipy 1.18 usa el método exacto si no hay empates, permutación si hay empates y n ≤ 13, y el asintótico si hay empates y n > 13. Las FDR son múltiplos de 1/800, así que los empates son habituales. La cota scipy>=1.10 admite versiones con otra regla, y la salida no dice qué método se usó.
- **Por qué importa:** Al promediar o tabular tamaños de efecto se pierden justo los casos nulos, y la columna statistic se contradice. El p puede variar entre los ordenadores del equipo para los mismos datos (el efecto medido es pequeño; se suma a M2).
- **Corrección propuesta:** Poner r = 0 cuando p == 1. En la rama de iguales, devolver el estadístico real de scipy (o NaN). Fijar method (por ejemplo, 'asymptotic' con zsplit, o PermutationMethod) y guardarlo en la salida. El signo de r ya está en I2.
- **Evidencia:** scratch/wf/metricas/t07_paired.py: con diferencias ±0,01/±0,02/±0,03 sobre 6 simulaciones, p = 1,0 y effect_size = NaN; con idénticos, effect_size = 0,0. Comprobación en línea: tfmfdd con idénticos (n = 20) da statistic 0,0; scipy (zsplit) da statistic 105,0 y p 1,0; tfmfdd con 20/20 peores da statistic 0,0 y p 7,74e-06. Mismos datos con empates (5 empates en |d| y 1 cero): p de tfmfdd 0,02753 = asintótico; exacto 0,02664. Código revisado: .venv/Lib/site-packages/scipy/stats/_wilcoxon.py:230-244.

#### MET-10 · limit_kde corta la cola superior cuando el estadístico es negativo

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/limits.py:118 (limit_kde, rejilla linspace(min, max*1.5))
- **Qué pasa:** Si max < 0, entonces max·1,5 < max y la rejilla termina antes del dato mayor: la cola superior desaparece y el cuantil sale desplazado hacia abajo. Con estadísticos no negativos, la rejilla es precisa. Con un atípico, la renormalización de la masa por debajo del mínimo sube el límite frente a la KDE sin truncar. Eso es una decisión de frontera no documentada, no un error.
- **Por qué importa:** Si el bloque B o el C calibran con KDE una puntuación que puede ser negativa (log-verosimilitud, score_samples de Isolation Forest, puntuaciones centradas), el límite sale mal sin ningún aviso. SPE, T² y el error de reconstrucción no están afectados.
- **Corrección propuesta:** Rejilla [min − 3·bw, max + 3·bw] con bw = kde.factor·std, o cuantil exacto con integrate_box_1d y brentq. Documentar la truncación (en 0 o en el mínimo) para estadísticos no negativos.
- **Evidencia:** scratch/wf/metricas/t09_kde.py (n = 150; «exacto» = brentq sobre integrate_box_1d de la misma KDE): con −chi2(8), tfmfdd da −1,603 frente a −0,320. Con chi2(8), 22,987 frente a 22,932 (0,2 %); con chi2(27), 48,035 frente a 47,987; con lognormal, 9,977 frente a 9,719. Con chi2(8) más un atípico de 320: 40,805 frente a 35,701 sin truncar (41,110 truncando en el mínimo); el 27 % de la masa de la KDE queda fuera de la rejilla.

#### MET-11 · Sin validar alpha ni variance, y los NaN pasan como «sin alarma»: métricas limpias a partir de entradas rotas

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/metrics.py:26-28, 37-38, 43-44; tfmfdd/limits.py:43, 55-61, 131; tfmfdd/pca.py:49-61 y 123-125; tfmfdd/preprocessing.py:29-33
- **Qué pasa:** PCAMonitor(alpha=99) da límites NaN; alarms(stat, NaN) es todo False, y evaluate_run devuelve FAR 0,0, FDR 0,0 y detected False con aspecto de resultado. variance=85 da n_components_ = 11 con 10 variables (loadings con 10 columnas, pero t2_limit usa 11). Un NaN en el estadístico cuenta como «sin alarma» y entra en el denominador: con 50 NaN de 100 muestras sanas, la FAR sale 0,0, y con todo el tramo de fallo en NaN, la FDR sale 0,0 en lugar de NaN. Un NaN en la calibración hace que limit_empirical y spe_limit_box devuelvan NaN (la guarda `mu <= 0 or v <= 0` no detecta NaN). Scaler con una sola fila lo devuelve todo en NaN.
- **Por qué importa:** I2 ya documenta la confusión de nombres entre el alpha 0,99 y el 0,05. Con este código, un alpha mal pasado no falla. En el TEP no hay NaN, así que hoy es latente, pero cualquiera de estos fallos sería silencioso.
- **Corrección propuesta:** Validar 0 < alpha < 1, 0 < variance ≤ 1 y n_components ≤ n_vars. En alarms, lanzar ValueError si el límite no es finito. En far y fdr, calcular sobre ~isnan y avisar (o fallar) si hay NaN. Usar np.isfinite en las guardas de Box y Scaler.
- **Evidencia:** scratch/wf/metricas/t10_validacion.py: alpha=99 da límites (nan, nan) y T² {far 0,0, fdr 0,0, detected False}; variance=85 da n_components_ = 11 con 10 variables y loadings_ de 10 columnas; FAR con 50 NaN de 100 = 0,0; FDR con el tramo de fallo todo NaN = 0,0; limit_empirical y spe_limit_box con un NaN dan nan; alarms(1e9, NaN).sum() = 0. t11_pca_checks.py: Scaler con una fila da [[nan nan nan]].

#### MET-12 · El núcleo numérico que más falla no tiene pruebas

- **Veredicto:** CONFIRMADO. *Matiz:* holm_correction y paired_test se ejecutan indirectamente en las dos pruebas de compare_methods, pero ninguna comprueba p, umbrales ni tamaño de efecto. La falta de cobertura efectiva se mantiene.
- **Dónde:** tests/test_tfmfdd.py, tests/test_correcciones_2026_09.py (en particular test_tfmfdd.py:91-99)
- **Qué pasa:** Ninguna prueba cubre spe_limit_jackson, lag_by_run, PCAMonitor con lags > 0, holm_correction, paired_test, contributions, summary ni plots (0 apariciones con grep; los dos «lags=» son de lag_matrix). test_limites_spe_coherentes_entre_si acepta ratios de 0,7 a 1,4 con n = 2000, así que no detectaría el sesgo del empírico con n = 150 (MET-3) ni Jackson con h0 < 0 (MET-7). Es distinto de M7, que se refiere a pruebas con datos reales.
- **Por qué importa:** MET-1, MET-2, MET-7, MET-8 y MET-9 pasan hoy la batería en verde. La etiqueta v0.3 distribuiría ese núcleo a Jonathan y Cesar sin ninguna red.
- **Corrección propuesta:** Convertir en pruebas deterministas, con semilla, los scripts del scratch: retardo DPCA = verdad con lags 0/1/2 (t01); lag_by_run con ids repetidos y desordenados (t02); Jackson frente a Monte Carlo con h0 = 1/3 y error con h0 < 0 (t03); FAR esperada del empírico (t04); Holm con NaN y p ajustados de referencia (t08); paired_test con p = 1 → r = 0 (t07); fdr_by_fault con value ≠ fdr_mean (t06). Añadir también la comprobación ya hecha de T² de Fase II (FAR de Monte Carlo 0,0100).
- **Evidencia:** grep -c sobre tests/*.py: spe_limit_jackson 0, lag_by_run 0, holm_correction 0, paired_test 0, plots 0, contributions 0, summary 0; limit_empirical y limit_kde, 1 cada una (solo en test_limites_spe_coherentes_entre_si).

#### DAT-4 · load_rieth lee el fichero entero antes de filtrar: runs no reduce el pico transitorio

- **Veredicto:** CONFIRMADO. *Matiz:* Medí working set, no memoria privada: los valores absolutos son menores que los del informe, pero la conclusión es la misma.
- **Dónde:** tfmfdd/data.py:173-178; tfmfdd/convert_rieth.py:151 (to_parquet sin row_group_size).
- **Qué pasa:** pd.read_parquet(path) carga las 500 simulaciones y después filtra. Medido como mediana de 5 procesos nuevos: load_rieth(1, 'train', runs=20) sube el pico de memoria privada 199 MiB para devolver 2,0 MiB, y load_rieth(1, 'test', runs=20), 318 MiB para devolver 3,9 MiB. Es lo mismo que con runs=None (189 y 308 MiB). Además, los 42 Parquet tienen un solo row group, así que no hay bloques que saltarse.
- **Por qué importa:** En un bucle de 21 clases el pico es transitorio y no se acumula: la memoria privada queda entre 0,34 y 0,58 GB y se mantiene estable en 3 repeticiones. En 16 GB no rompe nada hoy. Pero el parámetro runs sugiere un ahorro que no existe, y con varios procesos en paralelo (joblib) el pico se multiplica.
- **Corrección propuesta:** Leer primero solo simulationRun y pasar filters=[('simulationRun', 'in', ids)]. Con un único row group esto ya baja el pico a 166/255 MiB (medido). En la conversión, escribir row groups de 25 simulaciones (row_group_size = 25 * n_muestras; ya está en la propuesta de DAT-2) para que el filtro salte bloques. Ese segundo ahorro no está medido con datos reales.
- **Evidencia:** <scratch>/v4_tiempos_carga.py, salida en v4_resultado.json; v1_resultado.json (row_groups = 1 en los 42); v5_bordes_y_pool.py, salida en v5_resultado.json (memoria privada en 3 repeticiones: 420/411/338 MiB en entrenamiento y 575/580/550 MiB en prueba).

#### DAT-5 · Las columnas del contrato coinciden, pero los tipos no: Rieth devuelve int16/float32 y la aritmética con los ids desborda sin aviso

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/data.py:11-19 (docstring: 'el MISMO contrato') y :173-180; tfmfdd/convert_rieth.py:97-101.
- **Qué pasa:** load_classic devuelve ids int64 y variables float64; load_rieth devuelve ids int16 y variables float32. Con int16, simulationRun*1000 + sample produce solo 65.536 claves distintas para 480.000 filas (medido en test_fault01) en lugar de 480.000. data.values() devuelve float64 en ambos casos.
- **Por qué importa:** B00 construirá claves de grupo (GroupKFold) y C00 índices de ventana por simulación. Un desbordamiento silencioso mezcla simulaciones distintas, y eso es fuga o error de agrupación.
- **Corrección propuesta:** Hacer astype('int64') de los tres ids en load_rieth (cuesta 18 B más por fila; unos 7 MB para 21 clases x 20 simulaciones de prueba) y documentar que las variables son float32. Si no se cambia, avisarlo en B00 y C00.
- **Evidencia:** <scratch>/v5_bordes_y_pool.py, salida en v5_resultado.json: claves_distintas_int16 = 65536, claves_distintas_int64 = 480000, filas = 480000. En v4_resultado.json, dtypes ['float32', 'int16'].

#### DAT-6 · load_rieth no valida runs ni split

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/data.py:141-180 (load_rieth).
- **Qué pasa:** runs=np.int64(5) falla con TypeError ('only list-like objects are allowed to be passed to isin()'), porque isinstance(runs, int) es falso para los enteros de numpy, que es lo habitual si el valor sale de split_runs. Con una lista, los ids que no existen se ignoran sin aviso: runs=[1, 2, 999] devuelve 2 simulaciones. split tampoco se valida: en Windows 'Train' carga porque el sistema de ficheros no distingue mayúsculas; en Linux daría FileNotFoundError con un mensaje que manda a convertir. Por último, runs=int toma los ids 1..n, que no es una muestra aleatoria.
- **Por qué importa:** Son errores silenciosos o engañosos justo en la frontera que usarán B00 y C00.
- **Corrección propuesta:** Usar isinstance(runs, (int, np.integer)); validar que split esté en ('train', 'test'); lanzar un error si falta algún id pedido; y decir en la docstring que, para los resultados, los ids salen de split_runs.
- **Evidencia:** <scratch>/v5_bordes_y_pool.py, salida en v5_resultado.json: runs_np_int64 = TypeError, runs_lista_con_999 = [1, 2], split_Train = 'carga sin error (1000 filas)'.

#### DAT-7 · La comprobación posterior a la conversión es mínima y no hay pruebas de Rieth

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** tfmfdd/convert_rieth.py:166-188 (comprobar); tests/ (ninguna prueba menciona Rieth).
- **Qué pasa:** comprobar() solo mira el máximo de sample por simulación. No valida columnas, faultNumber, número de simulaciones, duplicados, orden, NaN ni inf. Ninguna prueba cubre load_rieth ni convert_rieth.
- **Por qué importa:** Una conversión truncada o hecha con otra versión pasaría la comprobación, y la propuesta de DAT-2 depende de la API interna de pyreadr, así que necesita una prueba que avise si esa API cambia.
- **Corrección propuesta:** Convertir las comprobaciones de v1 en data.validar_rieth(root); con SHA-256 y recorrido por lotes, tarda 21 s para los 42 ficheros. Añadir pruebas con skipif sobre los datos reales (xmeas_1 de IDV 6 cae en los índices 20 y 160, y el pre-fallo es idéntico a la gemela) y una prueba de conversión con un .RData pequeño escrito con pyreadr.write_rdata, con sample int32 para reproducir la disposición real.
- **Evidencia:** <scratch>/v1_inventario_ids.py (21,4 s, 42 ficheros); grep 'rieth' en tests/: sin resultados.

#### DIS-18 · El factor (a) tiene dos estimandas legítimas y conviene nombrarlas por separado

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** A0:6-15. decisiones.md:12.
- **Qué pasa:** A0 mide el efecto controlado: mismo X_fit de 350 y solo cambia la procedencia del límite. La práctica de la literatura es ajustar con 500 y calibrar sobre esas mismas 500, así que el «contraste de práctica» (500/500 en muestra frente a 350/150 limpio) es otra estimanda, confundida por n_fit, pero es la que responde a «cuánto cambia lo publicado».
- **Por qué importa:** Así se evita que el tribunal confunda las dos cosas, y que se repita la confusión de las cifras 12,5/2,3 y 19,9/3,6.
- **Corrección propuesta:** Cifra principal: el efecto controlado. El contraste de práctica, como secundario, con nombre propio y su configuración en el CSV.
- **Evidencia:** Lectura de A0 y de decisiones.md:12.

#### DIS-19 · La cifra de la demostración sintética del corte barajado no tiene script

- **Veredicto:** CONFIRMADO. *Matiz:* Ninguno.
- **Dónde:** Cuaderno 01, celda [35] («43 % … frente a un 3 %», demostración sintética).
- **Qué pasa:** La celda de markdown cita un 43 % frente a un 3 % de una «demostración sintética con el propio PCAMonitor». No encontré ningún script ni celda en el repositorio que la genere (búsqueda de «43 %», «sintétic», «barajad» y «shuffle»).
- **Por qué importa:** Motiva el factor (b) en el PCA, y la regla del equipo es «ningún número se copia a mano».
- **Corrección propuesta:** Convertirla en una celda o un script reproducible, o quitar la cifra.
- **Evidencia:** Grep en *.py, *.ipynb y *.md del repositorio: solo aparece en el markdown del cuaderno 01.

#### DIS-20 · load_rieth lee el fichero entero antes de filtrar las simulaciones

- **Veredicto:** MATIZADO. *Matiz:* «Pasaría de 2 GB» solo vale si los ficheros se cargan en paralelo. Si se cargan uno tras otro, como en una lista por comprensión, el pico es transitorio y no se acumula (DAT-4). Conviene fusionarlo con DAT-4.
- **Dónde:** tfmfdd/data.py:load_rieth (pd.read_parquet(path) y después el filtro por runs).
- **Qué pasa:** Cada fichero de prueba tiene 480 000 filas × 55 columnas (variables float32 y simulationRun int16, según los metadatos de pyarrow): del orden de 100 MB en memoria antes de filtrar. Es una estimación, no una medición. Uno a uno no es problema. Cargar los 21 ficheros a la vez, aunque sea con runs = 20, pasaría de 2 GB en el pico.
- **Por qué importa:** El equipo de Pablo tiene 15,6 GB y unos 5 GB ocupados. Las ablaciones con réplicas recorren muchos ficheros.
- **Corrección propuesta:** Usar pd.read_parquet(path, filters=[('simulationRun', 'in', ids)]) o pyarrow.dataset, y liberar tras cada fichero.
- **Evidencia:** Metadatos de pyarrow (ejecución en línea): train 250 000 filas, test 480 000, xmeas float, simulationRun int16. Tamaños en disco con ls.

#### REP-8 · pip freeze atribuye el .venv a un commit que no está en GitHub y oculta que el árbol tiene cambios sin commit

- **Veredicto:** CONFIRMADO. *Matiz:* Sin matiz.
- **Dónde:** Salida de '.venv/Scripts/python.exe -m pip freeze'
- **Qué pasa:** La línea de tfmfdd en el freeze es '-e git+https://github.com/pabdus/tfm-fdd-core.git@7541a5502e61f14cc24b64c796d8fa228e6c0194#egg=tfmfdd'. El commit 7541a55 (v0.2) no es alcanzable desde ninguna referencia remota, y los CSV actuales se generaron con 18 ficheros modificados y 10 sin seguimiento respecto a ese commit.
- **Por qué importa:** Si se aplica I7 tal cual (guardar las versiones en ENTORNO.txt) con un 'pip freeze' sin filtrar, el registro dirá que los resultados son de 7541a55, y eso es falso. Además, un lock copiado del freeze tendría una línea imposible de instalar.
- **Corrección propuesta:** Usa 'pip freeze --exclude-editable' para ENTORNO.txt y registra aparte 'git describe --always --dirty' en cada ejecución. El lock propuesto ya excluye esa línea.
- **Evidencia:** Línea de tfmfdd en $SIM/freeze_venv.txt. 'git ls-remote origin' solo muestra a3e0eac (main) y bf6cb4f (v0.1), y en el grafo del simulacro 7541a55 queda fuera de la historia de origin/main. 'git status --porcelain' del repo real da 18 entradas ' M' y 10 '??'. 'pip freeze --exclude-editable' en el .venv da 116 líneas, ninguna de tfmfdd.

#### REP-9 · La configuración de git que originó la división sigue activa

- **Veredicto:** CONFIRMADO. *Matiz:* Que init.defaultbranch=master explique que tu repo tenga master es plausible, pero no está demostrado. Los commits raíz de origin/main (19a1603 y 0f7a4be) también son de pablo@local, pero con zona horaria -0500; los de master tienen +0000.
- **Dónde:** .git/config (user.email); C:/Program Files/Git/etc/gitconfig (init.defaultbranch); refs/tags/v0.1 y v0.2; rama local main
- **Qué pasa:** El .git/config del repo fija user.email = pablo@local, una dirección que GitHub no puede asociar a tu cuenta. La llevan 6 de los 8 commits del historial, y también la llevarían la fusión y v0.3. La configuración del sistema de Git for Windows tiene init.defaultbranch=master, y por eso tu repo local tiene master mientras GitHub tiene main. v0.1 y v0.2 son etiquetas ligeras, sin autor ni fecha. La rama local main (0f7a4be) está desfasada y no sigue a ninguna rama remota.
- **Por qué importa:** La regla de UNIR de autor único se defiende mejor si todos los commits aparecen vinculados a tu cuenta. Una etiqueta anotada deja registradas la fecha y el autor de la publicación, que es lo que se cita.
- **Corrección propuesta:** Ejecuta git config user.email "112780608+pabdus@users.noreply.github.com" (la dirección noreply que ya aparece en tus commits web) y git config --global init.defaultBranch main. Usa git tag -a para v0.3. Después de publicar, actualiza la rama local main con --ff-only y asóciala a origin/main (fases B, G y J de la secuencia).
- **Evidencia:** 'git config --local --get user.email' devuelve pablo@local (lo comparé sin imprimir otras direcciones). 'git log --all --format=%ae | sort | uniq -c' da 6 con pablo@local y 2 con la dirección noreply de GitHub. 'git config --show-origin' muestra 'file:C:/Program Files/Git/etc/gitconfig init.defaultbranch=master'. 'git for-each-ref refs/tags' da v0.1 y v0.2 con objecttype commit, es decir, ligeras. 'git branch -vv' muestra 'main 0f7a4be' sin [origin/main].

#### REP-10 · Las instrucciones de desarrollo del README no funcionan tal cual en Windows y arrastran restos

- **Veredicto:** CONFIRMADO. *Matiz:* README:34 ya incluye el comentario «en Windows: .venv\Scripts\activate». Lo que falla en Git Bash es la ruta (hay que usar .venv/Scripts/activate), y en PowerShell 5.1 el '&&'. Lo demás se sostiene.
- **Dónde:** README.md:24, :27, :34-36, :123, :142; CAMBIOS.md:12
- **Qué pasa:** 'pip install -e .' no instala pytest, que está en el extra dev, así que el 'pytest -q' de la línea siguiente resuelve al pytest global y falla (REP-1). 'source .venv/bin/activate' no existe en Windows; en Git Bash la ruta es .venv/Scripts/activate. README.md:123 dice «20 pruebas» y son 31. README.md:142 conserva «Si encontráis», en forma de vosotros. README.md:24 y :27 dicen @v0.2.
- **Por qué importa:** Es lo primero que seguirá cualquiera que clone el repo, y el README que vaya dentro de v0.3 quedará fijado con la etiqueta.
- **Corrección propuesta:** Cambia las instrucciones por 'python -m pip install -r requirements-lock.txt', 'python -m pip install --no-deps -e .' y 'python -m pytest -q'. Para activar el entorno, indica '.venv\Scripts\activate' (PowerShell o cmd) y 'source .venv/Scripts/activate' (Git Bash). Corrige «31 pruebas» y «Si encuentran un fallo». En las líneas 24 y 27, pon @v0.3 con la forma 'tfmfdd @ git+https://github.com/pabdus/tfm-fdd-core@v0.3'.
- **Evidencia:** Leído con 'cat -n README.md' en el working tree. Prueba en el clon: Python312/Scripts/pytest.exe da 2 errores de recogida por ModuleNotFoundError, y 'python -m pytest' da 31 passed.

#### REP-11 · OneDrive no explica los 109 s de las pruebas: hoy tardan 1,5 s en la misma carpeta

- **Veredicto:** CONFIRMADO. *Matiz:* Sin matiz. Que los 109 s fueran una ejecución en frío sigue siendo una hipótesis sin verificar.
- **Dónde:** docs/revision_2026-09-27.md:113 (M10) y :240
- **Qué pasa:** M10 atribuye a OneDrive que las pruebas tarden 109 s. En la misma carpeta y con el .venv, hoy tardan 1,46 s.
- **Por qué importa:** El argumento para sacar el repo de OneDrive es el riesgo de conflictos de sincronización, que sigue en pie, no la velocidad. Mi hipótesis, sin verificar, es que los 109 s fueron una primera ejecución en frío (compilación de .pyc e importación de scipy y scikit-learn).
- **Corrección propuesta:** Reformula M10 sin presentar los 109 s como efecto de OneDrive, o mide varias veces antes de citar la cifra.
- **Evidencia:** En el repo real, 'PYTHONDONTWRITEBYTECODE=1 .venv/Scripts/python.exe -m pytest -q -p no:cacheprovider' da '31 passed in 1.46s' (2,12 s de tiempo real). El md5 de 'git status --porcelain' es el mismo antes y después. En scratch tarda 1,50 s, y en el venv limpio recién creado la primera ejecución tarda 8,01 s.

#### REP-12 · Fines de línea mixtos y sin .gitattributes

- **Veredicto:** CONFIRMADO. *Matiz:* Con autocrlf=true, lo versionado queda en LF en las dos plataformas. La diferencia afecta a los bytes del working tree y a lo que pip instala en Windows (CRLF), no al contenido del repositorio. Impacto bajo.
- **Dónde:** Falta .gitattributes; core.autocrlf=true solo viene de C:/Program Files/Git/etc/gitconfig
- **Qué pasa:** En el working tree, los cuadernos 01 y 02 y los 4 CSV de A1 y B1 están en CRLF, porque se regeneraron en Windows y pandas escribe os.linesep. El resto está en LF. Git los normaliza a LF al hacer commit solo porque tu Git for Windows trae autocrlf=true en la configuración del sistema.
- **Por qué importa:** En otra máquina o en la CI (Linux, LF), la misma regeneración produce bytes distintos; para comparar los CSV del simulacro tuve que quitar los \r. No afecta a los números.
- **Corrección propuesta:** Añade un .gitattributes con '* text=auto eol=lf', '*.csv text eol=lf' y '*.ipynb text eol=lf', o escribe los CSV con lineterminator='\n'.
- **Evidencia:** 'git ls-files --eol' da w/crlf en notebooks/01, notebooks/02, results/summary/A1_pca_baseline_{SPE,T2,runs}.csv y B1_clasificadores_clasico.csv; el resto es w/lf, y todo el índice es i/lf. 'git config --show-origin --get-all core.autocrlf' da 'file:C:/Program Files/Git/etc/gitconfig true'. En el clon, tras el commit, 36 de 36 ficheros quedan en i/lf.

#### REP-13 · No hay integración continua, y un control de etiqueta habría detectado M1

- **Veredicto:** MATIZADO. *Matiz:* «v0.1 y v0.2 se publicaron con 0.1.0 y nadie lo notó» es inexacto. v0.1 sí corresponde a 0.1.0, así que un control de etiqueta no habría detectado nada. v0.2 nunca se publicó, así que ninguna CI la habría visto. La CI sigue siendo útil (pytest con el lock y una construcción sin avisos), pero no habría evitado M1. No se ha ejecutado en GitHub.
- **Dónde:** .github/ (no existe)
- **Qué pasa:** No existe .github/workflows. Nada comprueba que el paquete se construya, que las pruebas pasen con las versiones fijadas ni que la etiqueta coincida con la versión del paquete.
- **Por qué importa:** v0.1 y v0.2 se publicaron con version = 0.1.0 (M1) y nadie lo notó. Con el paso a Rieth habrá más publicaciones.
- **Corrección propuesta:** Añade el workflow de la sección 3 del anexo: pytest en Python 3.12 con el lock, pip check y control de etiqueta.
- **Evidencia:** 'ls .github' indica que no existe. Ejecuté en local el paso de control de etiqueta con GITHUB_REF_NAME=v0.3: pasa con el paquete 0.3.0 ('etiqueta 0.3 version 0.3.0') y falla con el actual ("AssertionError: ('0.1.0', '0.3')"). Validé el YAML con PyYAML; no se ha ejecutado en GitHub.

## 3. Resúmenes de los verificadores


### Verificador de codigo

Casi todo el informe resiste la refutación: 44 hallazgos confirmados, 10 matizados, ninguno refutado. Todo lo he ejecutado con el Python del .venv. Los scripts están en C:/Users/pabdu/AppData/Local/Temp/claude/C--Users-pabdu-OneDrive-Documentos-Universidades-UNIR-TFM/b1423c25-c9de-48ed-b4f5-76701e58807e/scratchpad/wf/verificar-codigo (<vc>). No he modificado el repositorio y no he abierto los .RData.

LOS TRES HALLAZGOS MÁS IMPORTANTES QUE SOBREVIVEN
1. DAT-1 / DIS-2: en Rieth, la simulación r de cada fichero de fallo es la misma trayectoria que la r normal hasta el inicio del fallo. Lo reproduje en los 20 fallos, los dos splits y las simulaciones 1-20. En 3, 9 y 15, la distancia media entre gemelas es de 0,012 a 0,092, frente a 1,13 entre simulaciones no gemelas. Esto obliga a:
   - agrupar por simulationRun en todas las clases;
   - calcular la FAR una sola vez, en test_fault00;
   - no tratar los pares (fallo, simulación) como unidades independientes.
2. B2 y B3 bloquean la v0.3:
   - FAULT_ONSET_TRAIN_CLASSIC = 20 es falso: en d06.dat, xmeas_1 vale ≈ 0 desde la fila 0.
   - El «f1» de B1 es exactamente 2r/(1+r) (0,728 frente a un macro-F1 real de 0,634).
3. I4, DIS-5 y DIS-6 tocan la contribución más original. A0 no mide el T² de Fase I (límite Beta 45,61 frente a F 52,63; FAR 4,64 % frente a 1,40 %). Además, Jackson es una calibración en muestra implícita: θ1 sale de X_fit y su FAR es del 22,1 %, aunque el 01 y pca.py digan que no le afecta. Todas las cifras se reproducen exactamente.
   - A continuación va MET-1: DPCA desplaza el inicio del fallo l muestras y favorece a DPCA. Hoy no contamina ninguna cifra.

LO QUE EL CONJUNTO DE REVISORES SE HA DEJADO O HA DICHO MAL
- I3: el Fisher del traspaso SÍ se reproduce, con las cuatro cifras exactas (p = 0,0577, ρ = 0,274, n_ef = 285 y p ≈ 0,082), usando el tramo 461-960 (las últimas 500 muestras). El problema no es la reproducibilidad, sino un corte elegido a posteriori y no declarado, un ejemplo más de lo que señala I8.
- DIS-15: el «techo» de 3,2/5,1/18,5 % depende de pinv(rcond=1e-10), que descarta dos direcciones casi singulares de los lazos de nivel (xmeas_15/xmv_8 y xmeas_12/xmv_7). Con la inversa completa sube a ≈ 22/38/50 %. Por tanto, no respalda que en 3, 9 y 15 «no haya nada que detectar», y tampoco la comparación de DIS-11 con ese 3,2 %.
- DIS-4: el confundido por el tamaño de calibración vale para el T² empírico, pero no para el autoencoder de M9. Igualando n_cal = 150 dentro de la muestra, la FAR queda en 0,143-0,214 frente a 0,028.
- DIS-17: la lista de simulaciones de Rieth ya vistas está incompleta. Los revisores de datos y de Jonathan, y también esta verificación, han cargado las simulaciones 1-20 de los 42 ficheros y las 1-40 de train_fault00. Para lo confirmatorio hay que usar las simulaciones 21 o posteriores en todos los ficheros (41 o posteriores en train_fault00).
- DAT-1: la primera diferencia real puede llegar hasta 6 muestras después del inicio nominal (la 26 en entrenamiento y la 162 en prueba). Descartar solo sample ≤ onset deja algunas filas idénticas a la clase 0.
- Detalles menores:
  - M6: load_classic y load_rieth ya aceptan root.
  - DAT-3: hay 28 ficheros versionados, no 23.
  - DIS-2: la reducción de la sd es de 2,6 a 5,2 veces.
  - DIS-20: el pico de más de 2 GB solo se da si se carga en paralelo.
  - MET-12: holm_correction y paired_test se ejecutan de forma indirecta, pero ninguna prueba comprueba sus valores.
- Hay duplicados que conviene fusionar: DAT-1 con DIS-2, DAT-4 con DIS-20, e I1 con MET-4(c).


### Verificador de repo

Los tres hallazgos más importantes que sobreviven son estos:
(1) B1+REP-3, estado de publicación. v0.2 no existe en GitHub (reproduje el error 'pathspec v0.2' con pip) y las dos historias no tienen ancestro común. El README de GitHub manda instalar @v0.1: funciona, pero entrega el paquete del 17/09, sin convert_rieth y con el compare_methods que desalinea pares. main tiene el mismo código de paquete que tu working tree, pero sin commit ni etiqueta. Hace falta commit, unificación y etiqueta v0.3 antes de que Jonathan y Cesar instalen.
(2) REP-4, el protocolo común no viaja con el paquete. baseline.yaml no va en la rueda, y PCAMonitor usa variance=0.90 y calibra en muestra si no se pasa X_cal (la práctica que A0 mide: 0,2738 frente a 0,0357). A0 y A1 ignoran el yaml en silencio si no lo encuentran: lo comprobé ejecutando A1 sin configs/, y termina con exit 0 y las mismas cifras.
(3) I7+REP-1+REP-2, procedencia del entorno. El Python global no tiene tfmfdd instalado: desinstalarlo no hace nada. Las salidas de los cuadernos de v0.2, y las del 03 actual, vienen de un entorno POSIX con Python 3.12.3 que no está en este PC. Hay que ejecutar todo con '.venv/Scripts/python.exe -m ...' y registrar el entorno. REP-5 (licencia en tabla TOML, fin de soporte el 18/02/2027) también se confirma.

Lo que se dejaron los revisores:
(a) setuptools empaqueta LICENSE.zip como licencia (patrón LICEN[CS]E*). El .venv ya tiene tfmfdd-0.1.0.dist-info/licenses/LICENSE.zip de 108.894.512 bytes, y su METADATA dice 'License-File: LICENSE.zip'. Con un LICENSE.zip ficticio, 'pip wheel' del working tree lo mete en la rueda. Los dos zip contienen copias completas de .git y .venv (313 MB descomprimidos). Se corrige con license-files = ["LICENSE"] (probado) y sacando los zip de la raíz; añadir *.zip al .gitignore no basta.
(b) load_rieth lee el Parquet entero antes de filtrar por runs. Con runs=20, el DataFrame ocupa 4,1 MB, pero el proceso pasa de 163 MB a 416 MB de pico (data.py:173-178). Conviene usar filters= en read_parquet, dada la RAM disponible. Además, su root por defecto (data/tep_rieth) no es donde están los Parquet (C:/Users/pabdu/datos_tfm/tep_rieth), así que M11 está desfasado.
(c) Un 'git add -A' en UNIR no arrastraría el contenido del repo interno, que entra como gitlink embebido. Sí añadiría tfm-fdd-core.zip (1,59 GB, que rompería el push) y los documentos de TFM/Contexto.
(d) La versión está duplicada en pyproject.toml y __init__.py, sin una fuente única.
(e) Con el Python global, A1 falla con ModuleNotFoundError, así que los CSV de v0.2 tampoco salieron de él.
(f) En Windows, pip instala los .py con CRLF por el autocrlf=true del sistema: son idénticos a los del repo solo tras normalizar los fines de línea.

No modifiqué nada en el repo. Todo lo ejecuté en C:/Users/pabdu/AppData/Local/Temp/claude/C--Users-pabdu-OneDrive-Documentos-Universidades-UNIR-TFM/b1423c25-c9de-48ed-b4f5-76701e58807e/scratchpad/wf/verificar-repo/ (logs pip_wheel_wt.log, pip_wheel_fix.log y pip_gh_v0.2.log; neutral/a1.log). Las únicas excepciones son pytest en el repo, sin escribir cachés y con 'git status' idéntico antes y después, y un 'git add --dry-run' en el repo UNIR, que no modifica el índice.


## Anexos (entregables de cada revisor, sin editar)


### Anexo · Métricas: qué se revisó y quedó correcto

## Revisor adversarial del núcleo numérico: lo revisado que quedó correcto

He revisado a fondo metrics.py, stats.py, limits.py, pca.py, preprocessing.py, plots.py y tests/, y encontré doce hallazgos nuevos (MET-1 a MET-12). Cuatro son importantes, y ninguno bloquea las cifras actuales. Todo se hizo en solo lectura: el `git status` del repositorio sigue con las mismas 28 entradas que al empezar.

Los scripts de prueba están en `C:/Users/pabdu/AppData/Local/Temp/claude/C--Users-pabdu-OneDrive-Documentos-Universidades-UNIR-TFM/b1423c25-c9de-48ed-b4f5-76701e58807e/scratchpad/wf/metricas/` (t01 a t13). Se ejecutaron con el Python del `.venv`, cuyas versiones son numpy 2.5.3, scipy 1.18.1, pandas 3.0.6 y matplotlib 3.11.2.

- La prueba de `A1 --lags 1` escribía en el scratch y falló antes de escribir nada.
- Rieth se leyó solo en las columnas de id (`faultNumber`, `simulationRun`, `sample`) de tres ficheros.
- No ejecuté `pytest`, para no escribir `.pytest_cache` en el repositorio; el informe ya da 31/31.

**Cómo afecta a las cifras actuales:**
- **MET-1 y MET-2:** ningún resultado usa lags > 0. Además, A1 con `--lags` falla antes de producir nada.
- **MET-3:** solo afecta al cuaderno 03 (autoencoder, límite empírico con 150 muestras). A0 y A1 usan Box.
- **MET-7:** en el TEP, h0 sale positivo en todas las configuraciones.
- **MET-4(c):** es la cifra de retardo por fallo que ya señala I1.

### metrics.py
- `alarms`: correcta con un límite finito (el `>` es estricto). Con NaN falla, ver MET-11.
- `far` / `fdr`: correctas. Con un array vacío devuelven NaN.
- `detection_delay`: correcta. El retardo es el inicio de la primera ventana de k alarmas.
  - Casos límite probados: onset = 0; onset > longitud → no detectado; k mayor que el tramo → no detectado, NaN.
  - Con onset = None usa toda la serie, pero a ese caso solo se llega llamándola directamente.
- `stratified_fdr`: correcta. Parte con `a.size // 2` y la segunda mitad se lleva la muestra impar. Con onset = None directo usa toda la serie; evaluate_run lo evita.
- `evaluate_run`: correcta en PCA estático.
  - FAR sobre `stat[:onset]`, FDR sobre `stat[onset:]`, y `detected_at_chance` bien calculado.
  - Con onset = 0 la FAR sale NaN; con onset = longitud, la FDR sale NaN y no hay detección.
  - Con DPCA falla, ver MET-1.
- `aggregate`: correctos `std(ddof=1)`, `stats.sem` (ddof = 1) y t con n − 1 grados de libertad. Las claves en lista funcionan con pandas 3.0.6 (probado sobre el CSV de A1). Lo que falla está en MET-4.
- `summary`: nada nuevo aparte de lo que ya dice I1.

### stats.py
- `paired_test`: correctos el emparejamiento, la máscara de NaN, `zsplit` y `median_difference`. scipy 1.18 trata `zsplit` con ceros pasando a permutación si n ≤ 13 o a asintótico si n > 13, sin error. Lo demás en MET-9.
- `holm_correction`: el paso descendente es correcto, `rank` y `threshold` quedan alineados con el orden original, y los empates se tratan bien (tres p = 0,02 → ninguno significativo). Con una lista vacía no falla.
- `compare_methods`: empareja bien por fallo y por simulación, y la intersección de índices es correcta.
  - El signo de `median_difference` depende del orden alfabético de los nombres de método. Está documentado en `method_a`/`method_b`, pero conviene tenerlo en cuenta al implementar el `baseline=` de I2.

### limits.py
- `t2_limit`: la fórmula F de Fase II es correcta. Monte Carlo (12 variables, todas las componentes, n = 350, 400 réplicas × 2000 muestras): FAR 0,0100.
- `spe_limit_box`: g = v/(2μ) y h = 2μ²/v son correctos. Sobre 2,5·chi2(6) con 200 000 muestras da 42,012, frente a 42,030 teórico.
- `spe_limit_jackson`: correcto con h0 > 0 (autovalores iguales: 18,796 frente a 18,761 de Monte Carlo). Con h0 < 0 falla, ver MET-7.
- `limit_kde`: la rejilla es precisa para estadísticos no negativos, con un error del 0,1 al 0,3 % frente al cuantil exacto. Con estadísticos negativos falla, ver MET-10.
- `limit_empirical`: el percentil está bien implementado. El problema es su FAR esperada con n pequeño, ver MET-3.

### pca.py
- `PCAMonitor.fit`:
  - El escalado se ajusta solo con X, y `np.cov` (ddof = 1) es coherente con el escalador. El orden de los autovalores es correcto.
  - La selección de componentes da el mínimo a con varianza acumulada ≥ objetivo (comprobado: 0,9087 con a y 0,7869 con a − 1).
  - En DPCA, el T² usa `len(Z)` después de los retardos: 348 con una simulación y lags = 2, y 346 con dos simulaciones.
  - Con lags > 0, pasar `X_cal` sin `runs_cal` da un error explícito.
- `_statistics`: comprobado que ‖z‖² = Σt² + SPE.
- `score`: correcto con lags = 0. Con lags > 0 falla, ver MET-1 y MET-2.
- `contributions`: la suma es igual al SPE (comprobado).
  - Con varias filas devuelve un vector aplanado de n·m; la docstring dice «una observación».
  - Con DPCA falla con un error de broadcast. No es silencioso, pero no admite DPCA.

### preprocessing.py
- `Scaler`: usa ddof = 1, pone desviación 1 a las variables constantes y solo se ajusta con `fit`. Con una sola fila, ver MET-11.
- `lag_matrix`: el orden [x_t, x_{t−1}, …] y la forma (n − l, m(l + 1)) son correctos (hay prueba).
- `lag_by_run`: ver MET-2.

### plots.py
- `control_chart`:
  - Trata onset como índice, igual que evaluate_run. La línea queda en 8,0 h, sobre la primera muestra con fallo.
  - El eje pone la muestra 1 en t = 0, así que está desplazado 3 min; no tiene importancia.
  - `ncols` es compatible con `matplotlib>=3.7`.
  - Con DPCA, la línea y el eje quedan desplazados l muestras, ver MET-1.
- `fdr_by_fault`: con los valores por defecto es correcta, que es como la usa el cuaderno 01, celda [32]. Con otras métricas falla, ver MET-5.
- `contribution_plot`: el orden de las barras y el de las etiquetas coinciden.
- `save`: ver MET-6.

### tests/
- Las pruebas existentes son coherentes con el código, salvo la tautológica de `FAULT_ONSET_TRAIN_CLASSIC` (ya en B2).
- `test_agregacion_calcula_intervalos` usa `np.random` sin semilla. Siempre pasa, pero no es determinista.
- Lo que falta está en MET-12.


### Anexo · Datos: inventario y validación de Rieth

## Datos de Rieth: inventario, validación del contrato y forma de carga (para los autores de B00 y C00)

**Resumen para quien escriba B00 o C00.** Los 42 Parquet de `C:/Users/pabdu/datos_tfm/tep_rieth` están completos y cumplen el contrato: se pueden usar tal cual. Hay tres cuidados que conviene aplicar desde el primer cuaderno:

1. Partir por `simulationRun`, con los mismos ids para todas las clases (DAT-1).
2. Descartar las filas anteriores al inicio del fallo en los ficheros de fallo (DAT-1).
3. Convertir los ids a int64 antes de operar con ellos (DAT-5).

Además, **no conviertan los `.RData`**: pídanle a Pablo los Parquet y verifiquen el SHA-256 (DAT-2 y DAT-3).

Qué hice y qué no:
- No convertí datos reales ni modifiqué nada del repositorio ni de `datos_tfm`: `git status` sigue con las mismas 28 entradas y los mtimes no han cambiado.
- Leí en solo lectura los `.RData` originales:
  - el más pequeño (24,7 MB), cargado en memoria, para conocer su estructura;
  - los otros tres, recorridos sin retener columnas (pico de 147 MiB o menos), solo para leer los tipos de columna.
- Los `.RData` sintéticos de las mediciones se generaron en el scratch y los borré después, porque ocupaban 3,2 GB. Se regeneran con `m0_generar_rdata.py` y `m0b_generar_rdata_real.py`, y los JSON de resultados se conservan.

Scripts y resultados en `<scratch>` = `C:/Users/pabdu/AppData/Local/Temp/claude/C--Users-pabdu-OneDrive-Documentos-Universidades-UNIR-TFM/b1423c25-c9de-48ed-b4f5-76701e58807e/scratchpad/wf/datos-rieth/`.

### 1. Inventario de `C:/Users/pabdu/datos_tfm/tep_rieth`

- 42 ficheros `{split}_fault{NN}.parquet`, con NN de 00 a 20 en `train` y en `test`. No hay fallo 21.
- Total: 1.295.333.849 bytes (1.235,3 MiB).
  - `train`: 433,6 MiB (entre 15,4 y 23,8 MiB por fichero), 5.250.000 filas.
  - `test`: 801,7 MiB (entre 27,7 y 43,5 MiB), 10.080.000 filas.
- Formato común a los 42: compresión snappy, un row group por fichero, escritos con `parquet-cpp-arrow 25.0.1`. Ids en int16 y 52 variables en float32.

| Fallo | bytes `train` | bytes `test` | | Fallo | bytes `train` | bytes `test` |
|---|---|---|---|---|---|---|
| 00 | 20.570.080 | 38.783.047 | | 11 | 20.785.327 | 39.153.794 |
| 01 | 22.886.750 | 42.184.499 | | 12 | 24.979.052 | 45.192.753 |
| 02 | 22.077.843 | 40.922.890 | | 13 | 24.747.115 | 45.347.163 |
| 03 | 20.559.195 | 38.777.040 | | 14 | 20.911.284 | 39.334.546 |
| 04 | 20.593.018 | 38.838.282 | | 15 | 20.616.515 | 38.868.583 |
| 05 | 22.302.216 | 41.231.184 | | 16 | 20.924.964 | 39.310.193 |
| 06 | 16.157.494 | 29.020.712 | | 17 | 21.126.137 | 39.695.559 |
| 07 | 23.716.312 | 43.250.651 | | 18 | 23.454.615 | 37.926.534 |
| 08 | 24.821.493 | 45.635.132 | | 19 | 20.747.732 | 39.082.804 |
| 09 | 20.577.394 | 38.812.549 | | 20 | 20.984.084 | 39.527.983 |
| 10 | 21.172.900 | 39.726.431 | | | | |

Manifiesto SHA-256 en formato `sha256sum -c`: se ejecuta dentro de la carpeta de los Parquet. Sale de `v1_inventario_ids.py`.

```
3a23c5fa17dfafde95310bd787489c3b622937f393551c61eec8732667b20449  test_fault00.parquet
64ea480271eb853aec5f97ee581af0f9681c8b7d0d763e792287ae66ca3dc0c6  test_fault01.parquet
516c307123004192caacd7c3a663d326be36039e0def1466f761ba847386a95f  test_fault02.parquet
6188fb97aa55b8074e5fd9397d5b9fc97507d3926da07f26c8bd53a9de250cf5  test_fault03.parquet
57c0a822cdf1931be7ec35faff8ebf09cdbddefc4191c18a476e365eb481ff8b  test_fault04.parquet
494cab789b6c1f582198a715949c2ec468142d4c609963c34fa13cbf584665b5  test_fault05.parquet
5d7cadeb9cb9e181059a24552073956600a022eeeb7f3660e9ff690e8d49c638  test_fault06.parquet
7833b7393b3a56565357db6703830ff1923f253cfe0236e7d411a1d6b7e79ae0  test_fault07.parquet
0648ab9c1eff642ff5c06eee5fbc0329de8bef243d47e9320a5705be166787e9  test_fault08.parquet
cce79afc40a79f7f1c84eb669fe82591304b537b5396252a2601381050f6f07a  test_fault09.parquet
98fcc4e22c6c6e4d34d9068d2fa4f047059267bfe9e4b28a8936983b0851cf68  test_fault10.parquet
9e3267f5538852ebad5af361a004aaaa7cfefe45bdf04dd70064e206d21bf946  test_fault11.parquet
731bb82eec112a57057f83032435f43a75fe8510f7bdb81a6476e373150a3cd2  test_fault12.parquet
cff14fff95e4ee93d0a6c024f923c4ce76184335522259bf5de618ee6438dcd2  test_fault13.parquet
b062957b003fc3cd80073933cf53b335707d672aaaad7f27901a39a73f740e11  test_fault14.parquet
51dba8a1c112f53ed0185fdf1373f13a1ee14145cbb5caaf1a7b4aae08463846  test_fault15.parquet
d158c41886640b9f9a030e96b08dde9233d975d2378d7c763f9b1f30c5d1fb98  test_fault16.parquet
b5589384e14568f6637ba8b518c94d576f3f944d81c6f9c92ae5a4b6b23df68a  test_fault17.parquet
a4fb06740fb177f990087d86f1f0abbe38483efe403bdc50710c792393f7c7b3  test_fault18.parquet
5fd9163b4582061dea656dde82a62d5abcac6fc9079fa4a528340e750a34303d  test_fault19.parquet
2c2219e76a7c14175cd92f37e121e4361c0a628da1871f30a4580959df2ad7e6  test_fault20.parquet
b3b14d3069b757f4aba552b80600869b49e03b88cd68401049895137bbe40efa  train_fault00.parquet
d616437645203075b916b259d683f7ec78bd11985b51c239fe7105d604526701  train_fault01.parquet
5058fd788d3b132ef2714d2b101a72318c6acbb49884536684f8420671f64ac0  train_fault02.parquet
63c90694298a52a5e22d856b425bfa5598a882c9af22938833ad73bb73556aea  train_fault03.parquet
692890c18b7a5d3347474042ad0e505679e76570b5b7f08cdae80a04f8597a87  train_fault04.parquet
b4d4940d814291bb2e7ec7c2e9611a0a8b1052de56d9f242de77890aefd46206  train_fault05.parquet
9253148448121698315b30606db03d6301b0d804c448adc36bab0c053b35f9c0  train_fault06.parquet
e424ea543787ee1ec5d1b77003d5e91d7001f03ad3bf626df1606d5ba90035d2  train_fault07.parquet
21831c646cd6fd2582e32f8e159771e2838a7f99cb7caec2144e8de51d459c1a  train_fault08.parquet
24d671aed6d82a7e458b0e00c76327195660a52b400ee132627bdfd2e7c6e70e  train_fault09.parquet
d39246135126741c0d14f520f2ea1c83f3fbe3738b730bff6022712c6f5e6cec  train_fault10.parquet
e323cc27501c1aefd85eedb4680add7941e6eb2fda797f3b855be7340a4bdf32  train_fault11.parquet
259fc2119d5962631042040bc516ed8f8b9fd75105379321910ffe64932ce908  train_fault12.parquet
79efd0f58eef53e394d0a97062864313b89a507a4e20757a34437ba84212a2b1  train_fault13.parquet
93b996b78761fc4edca3409b1103fac067a202957173d713022aacb3151446f2  train_fault14.parquet
161fa3622bd3ed737926ec959d510f866d9fe85362e4987823bcb0f847e1c73b  train_fault15.parquet
cf2590764f1a14a7481f8a6f5add5adbe073bfbf66f65f742dc491e559be67d2  train_fault16.parquet
d6fc10cb0a0f2b332433dce45ebd2ef3f78175dcde7bb2d0791c0cc4a07d9894  train_fault17.parquet
e2c7c0df42412d24309ec91fa10f0444ede8abb86e04e8ee23ddb38cc097eb5c  train_fault18.parquet
f652041f4e28be7e21a6d1ed4a6dc9c1fc101d984eb6b957a46ef947467bc7d8  train_fault19.parquet
0009464e7aff8cf3ac3bcc9d134eb6b7c2fe033d03116f1eb7b77b70a4461413  train_fault20.parquet
```

Los `.RData` originales (`data/tep_rieth/`, sin tocar) suman 1.402.950.911 bytes: FaultFree_Training 24.678.017, FaultFree_Testing 47.327.663, Faulty_Training 494.063.194 y Faulty_Testing 836.882.037. Los cuatro son gzip.

### 2. Validación del contrato (ejecutada)

| Comprobación | Esperado | Observado | Evidencia |
|---|---|---|---|
| Columnas | `data.COLUMNS` (55, en orden) | Iguales en 42 de 42 ficheros | `v1_inventario_ids.py` (esquema Parquet) |
| Tipos | Clásico: int64 y float64 | Rieth: ids int16, variables float32 (ver DAT-5) | v1, v4 |
| `faultNumber` | 0..20, sin 21 | Un único valor por fichero, igual a su NN; no hay fichero 21 | v1 |
| Simulaciones por fallo y split | 500 | 500 en los 42 ficheros; ids de 1 a 500 contiguos | v1 |
| Muestras por simulación | train 500, test 960 | 500 y 960 en todas; `sample` de 1 a N, sin duplicados (run, sample), ordenado | v1 |
| NaN / inf | 0 | 0 en los 42, con recorrido completo por lotes de 65.536 filas | v1 |
| Inicio del fallo en entrenamiento | `FAULT_ONSET_TRAIN_RIETH = 20` | **Coincide.** IDV 6: `xmeas_1` < 0,05 por primera vez en el índice 20 (muestra 21) en 20 de 20 simulaciones. Simulación 1, índices 17-20: 0,2381 / 0,2518 / 0,2530 / **-0,0018**. Media del tramo previo 0,2506; media normal 0,2503 | `v2_onset_semillas.py` |
| Inicio del fallo en prueba | `FAULT_ONSET_TEST = 160` | **Coincide.** IDV 6: primer índice con `xmeas_1` < 0,05 = 160 en 20 de 20. Simulación 1, índices 157-160: 0,2173 / 0,3065 / 0,3061 / **-0,0008** | v2 |
| Inicio del fallo en los 20 fallos | Filas previas = normal | Con 20 simulaciones por fallo, las filas previas al inicio son idénticas bit a bit a la simulación normal con el mismo id (ver cuidado 1). La primera diferencia aparece en el índice 20 o 160 en los fallos 1-7, 14, 15 y 19, y más tarde en los aleatorios: hasta el índice 91 en entrenamiento (IDV 18) y 231 en prueba (IDV 18) | v2 |
| ¿Se solapan los ids normales y de fallo? | — | Sí: los 500 ids coinciden en cada fallo y split, **y además comparten semilla** (DAT-1) | v1, v2, v3 |
| Semillas entre entrenamiento y prueba | Distintas | 0 de 400 pares (20x20) con la fila inicial igual | v2 |
| Tamaño en disco | — | 1.235,3 MiB | v1 |
| `load_rieth(1, 'train', runs=20)` | — | 10.000 filas, 2,0 MiB. **0,065 s** (mediana de 5 procesos; entre 0,053 y 0,075). Pico transitorio de memoria privada ~0,2 GB | `v4_tiempos_carga.py` |
| `load_rieth(1, 'test', runs=20)` | — | 19.200 filas, 3,9 MiB. **0,087 s** (entre 0,077 y 0,091). Pico ~0,32 GB | v4 |
| 21 clases x 20 simulaciones, concatenadas | — | train: 210.000 filas, 0,8 s, 42,9 MiB. test: 403.200 filas, 1,5-3,0 s, 82,3 MiB. Memoria del proceso entre 0,34 y 0,58 GB, estable | `v5_bordes_y_pool.py` |
| `fault_onset('train'/'test', 'rieth')` | 20 / 160 | 20 / 160 | v5 |

Los tiempos son en caliente: los ficheros ya estaban en la caché del sistema, y no medí lecturas en frío.

Evidencia nueva sobre B2: la fila de `LEEME_TEP_clasico.md` que sitúa el «fallo en entrenamiento en la muestra 21 (1 h)» es correcta **para Rieth**. Queda confirmado con datos que el 20 pertenece a Rieth y no al clásico.

### 3. Cómo cargar

```python
from pathlib import Path
import pandas as pd
from tfmfdd import data

RIETH = Path(r"C:/Users/pabdu/datos_tfm/tep_rieth")   # en tu equipo: donde copies los 42 Parquet, fuera de OneDrive

# Desarrollo: una clase y las simulaciones 1..20
df = data.load_rieth(1, "test", runs=20, root=RIETH)
X = data.values(df)                                     # (19200, 52), float64

# Conjunto multiclase sin fuga
tr, va, te = data.split_runs(500, seed=42)              # UNA vez: los mismos ids para TODAS las clases
ids = [int(i) for i in tr[:20]]                         # enteros de Python (DAT-6)
onset = data.fault_onset("train", "rieth")              # 20 (en prueba, 160)
partes = []
for k in range(21):
    d = data.load_rieth(k, "train", runs=ids, root=RIETH)
    if k > 0:
        d = d[d["sample"] > onset]                      # lo anterior es copia exacta de la simulación normal con el mismo id
    partes.append(d)
pool = pd.concat(partes, ignore_index=True)
ID = ["faultNumber", "simulationRun", "sample"]
pool[ID] = pool[ID].astype("int64")                     # int16 desborda (DAT-5)
assert pool.groupby("faultNumber")["simulationRun"].nunique().eq(len(ids)).all()
groups = pool["simulationRun"]                          # GroupKFold: por id, compartido entre clases
```

### 4. Cuidados (léanlos antes de B00 y C00)

1. **Semillas compartidas (DAT-1).** La simulación normal r y la de fallo r del mismo split son la misma trayectoria hasta el inicio del fallo. En los fallos 3, 9 y 15 siguen casi idénticas después: distancia media |Δz| de 0,013, 0,063 y 0,093, frente a 1,14 entre simulaciones distintas. Esto implica:
   - Partir por id, con los mismos ids en todas las clases.
   - No usar nunca una clave (fallo, simulación) como grupo.
   - Descartar las filas previas al inicio en los ficheros de fallo.
   - Medir la FAR en `test_fault00` por simulación, sin contar los 20 pre-fallo de prueba como segmentos independientes.
   - Sacar de la evaluación de todas las clases los ids que se usen para calibrar.
   - Entre los ficheros `train` y `test` no hay semillas compartidas: entrenar con `train` y probar con `test` no produce fuga.
2. **Retardo en fallos aleatorios.** En IDV 8, 10, 13, 17, 18 y 20 la trayectoria se separa de su gemela varias muestras después del inicio: hasta 71 muestras en entrenamiento (IDV 17: índice 91 frente al inicio en 20) y 71 en prueba (IDV 18: índice 231 frente a 160). Un retardo mayor que cero en esos fallos es en parte física del proceso, no un defecto del detector.
3. **Tipos (DAT-5).** Los ids son int16 y las variables float32. Hay que pasar los ids a int64 antes de hacer aritmética con ellos: `simulationRun*1000+sample` en int16 da 65.536 claves distintas para 480.000 filas. `data.values()` ya devuelve float64.
4. **`runs` (DAT-6).**
   - Pasen listas de `int` de Python.
   - Si piden ids que no existen, se ignoran sin aviso: comprueben `nunique()`.
   - `runs=20` son los ids 1..20, útiles para desarrollo; en los resultados, los ids salen de `split_runs`.
5. **Memoria (DAT-4).**
   - Cada llamada a `load_rieth` tiene un pico transitorio de ~0,2 GB en `train` y ~0,32 GB en `test`, sea cual sea `runs`.
   - Lo que queda residente es pequeño: 21 x 20 ocupa 43 MiB en `train` y 82 MiB en `test`; con `values` en float64, 83 y 160 MiB.
   - Por cálculo, sin medir: 21 clases x 100 simulaciones de prueba son 2.016.000 filas, unos 411 MiB en float32 y 800 MiB en float64. Con 500 simulaciones serían ~2,0 GiB en float32 y ~3,9 GiB en float64: con la app de Claude abierta en 16 GB, conviene trabajar por clases o en float32.
   - No lancen varias cargas en paralelo sin tenerlo en cuenta.
6. **No conviertan (DAT-2 y DAT-3).** La conversión actual de faulty_testing necesita entre 15,3 y 16,2 GiB (unas 4 veces la tabla float64). Usen los Parquet de Pablo y verifiquen el hash con `sha256sum -c`.

### 5. Memoria de `convert_rieth`: qué explica el pico y cuánto baja la propuesta

La tabla float64 se calcula como filas x 55 x 8 B. Para faulty_testing son 3,93 GiB. Las cifras de la tabla son el pico de memoria privada comprometida del proceso (`PeakPagefileUsage`), medido en procesos nuevos con pandas 3.0.6 y pyreadr 0.5.6.

| Paso de la versión actual | Coste propio (veces la tabla) | Evidencia |
|---|---|---|
| `pyreadr.read_r`: columnas acumuladas por el parser | 1,0 | `m4_traza_parser.py` (1,02) |
| `pyreadr.read_r`: `DataFrame.from_dict` con int32 intercalada (así son los `.RData` de Rieth) | +2,9 a +3,0 (+1,0 si todas son float64 o si la int32 va al final) | `m5_from_dict.py`, `m6_variantes.py`, `m9_layout_faulty.py`, `m8_tipos_sin_cargar.py` |
| `rename` + `df[COLUMNS]` | 0,0 (copy-on-write; columnas ya en orden) | `m3`, `m10`, `m11` |
| `_aligerar` (astype columna a columna) | +0,47 transitorio | `m11` |
| Filtrado por fallo + `to_parquet` | +0,16 | `m11` |
| **Total medido** | **3,99** (`.RData` real pequeño) / **4,11** (sintético con la disposición real) | `m3` real_pequeno, `m2_resultado_sintreal_R25.json` |

| Fichero | Filas | Tabla float64 | Actual (3,9-4,1 veces) | Propuesta (0,54-0,80 veces) |
|---|---|---|---|---|
| FaultFree_Training | 250.000 | 0,10 GiB | 0,40-0,42 GiB | 0,06-0,08 GiB |
| FaultFree_Testing | 480.000 | 0,20 GiB | 0,77-0,81 GiB | 0,11-0,16 GiB |
| Faulty_Training | 5.000.000 | 2,05 GiB | 8,0-8,4 GiB | 1,1-1,6 GiB |
| Faulty_Testing | 9.600.000 | 3,93 GiB | 15,3-16,2 GiB | 2,1-3,1 GiB |

- **Por qué es lineal.** Con todo float64, el pico da 2,40 veces con 480.000 filas y 2,37 con 960.000; la propuesta, 0,80 y 0,76. Los cocientes de `from_dict` tampoco cambian entre 250.000 y 1.000.000 de filas.
- **Contraste con lo observado.** La extrapolación de la versión actual (15,3-16,2 GiB) encaja con la caída de ~15,5 GB de memoria comprometible que midió el revisor.
- **Propuesta.** Está en `<scratch>/propuesta_convert_rieth_v2.py`: baja cada columna a float32 al vuelo y no construye el DataFrame completo.
  - Salida idéntica en 4 sintéticos (20 de 20 ficheros cada uno).
  - Idéntica, en memoria, al `train_fault00.parquet` existente.
  - Mismo tiempo de ejecución.
  - Depende de la API interna de pyreadr: hay que fijar `pyreadr==0.5.6` y añadir una prueba.
- **Lo que no ayuda.** La alternativa con API pública, que elimina `rename`/`df[COLUMNS]`, no baja el pico: 827 MiB, igual que la versión actual.
- **No medido.** El comportamiento con pandas 2.x.


### Anexo · Diseño experimental: tabla por factor de ablación e hipótesis

# Revisión metodológica del diseño (etiqueta `diseno`), 27/09/2026

Solo he leído el repositorio: la rama sigue en `revision-2026-09` y `git status` está igual que al empezar. Todo lo ad hoc está en `<scratch>` = `C:/Users/pabdu/AppData/Local/Temp/claude/C--Users-pabdu-OneDrive-Documentos-Universidades-UNIR-TFM/b1423c25-c9de-48ed-b4f5-76701e58807e/scratchpad/wf/diseno/`, con estos ficheros:

- `t2_fases_clasico.py/.out`
- `piloto_rieth.py/.out`
- `piloto_factor_a.csv` y `piloto_potencia.csv`
- `semillas_comunes.py/.out`
- `techo_detectabilidad.py/.out`
- `potencia_tost.py/.out`

Todo se ejecutó con el Python del `.venv`, desde la raíz del repositorio. **Son evidencia de revisión, no resultados para la memoria.**

## 1. T² de Fase I frente a Fase II (montaje de A0: 52 variables, 85 % → a = 27, m = n_fit = 350, n_cal = 150, nivel 0,99)

| Límite | Valor | FAR pre-fallo (media de 21 segmentos) | FAR d00_te (960) | FAR teórica en Fase II |
|---|---|---|---|---|
| F, Fase II: a(m+1)(m−1)/[m(m−a)]·F(a, m−a) | **52,6266** | 1,40 % | 1,46 % | 1 % |
| Beta, Fase I: [(m−1)²/m]·B(a/2, (m−a−1)/2) | **45,6140** | 4,64 % | 6,35 % | **4,05 %** |
| χ²(a), asintótico | 46,9629 | 3,78 % | 4,48 % | 3,13 % |
| Percentil 99 empírico **en muestra** (X_fit, 350) | **43,7977** | 6,40 % | 8,54 % | — |
| Percentil 99 empírico en muestra (150 de X_fit) | 47,6640 | 3,30 % | 3,85 % | — |
| Percentil 99 empírico **reservado** (X_cal, 150) | 50,0439 | 2,23 % | 2,08 % | — |
| KDE en muestra / reservado | 45,69 / 51,05 | 4,61 % / 1,85 % | 6,15 % / 1,98 % | — |

- **Comprobaciones de cordura:** la media del T² en muestra es 26,9229, que coincide exactamente con a(m−1)/m. En X_fit superan el límite el 0 % (F), el 0,86 % (Beta) y el 1,14 % (percentil empírico).
- **Piloto en Rieth** (10 réplicas; a = 22–26): F 0,85 %, Beta 2,52 %, empírico en muestra 2,99 % frente a reservado 2,28 %. La diferencia es positiva solo en 6 de 10 réplicas.
- **Lectura:** lo robusto en T² es Fase I frente a Fase II, y la teoría lo predice. El efecto de la procedencia con percentil empírico es pequeño y se confunde con n_cal.

**Cómo encajarlo en A0:**

1. Parámetro `t2_method ∈ {f, beta, chi2, empirical, kde}` en `PCAMonitor` (propuesta de I4 más Beta y χ²).
2. Una fila por (estadístico, método, procedencia, n_cal), con la columna `far_teorica_faseII`.
3. Contrastes:
   - T-Beta frente a T-F: el puente con Champ (2005) y Jensen (2006), con predicción teórica.
   - T-empírico en muestra frente a T-empírico reservado, con **igual n_cal**: el efecto puro de la procedencia.
   - Dosis-respuesta en n_cal y en n_fit.
4. Todo en Rieth con réplicas (DIS-3).

## 2. Tabla de diseño por factor de ablación

Reglas comunes a todos los factores:

- Solo cambia un factor por brazo. Todo lo demás queda fijado en `configs/baseline.yaml` y escrito en el CSV.
- `n_components` queda fijo dentro de cada réplica.
- En Rieth, la FAR se calcula solo con `test_fault00`, una por simulación, sobre las muestras 161–960 (DIS-2).
- Los grupos son el índice de simulación, compartido entre fallos.
- Se reportan el signo y la nominal.
- Las simulaciones del piloto quedan fuera (DIS-17).

| Factor | ¿Medible hoy? | Brazos | Qué se mantiene fijo | Métrica principal | Unidad de réplica | Datos | Script (propuesto) |
|---|---|---|---|---|---|---|---|
| **(a) Calibración, SPE** | Sí en el clásico (A0, una realización); no en Rieth (A0 solo carga el clásico) | En muestra (Box, empírico); reservada temporal; reservada con otra simulación; Jackson como «en muestra implícita»; todos con igual n_cal | X_fit, escalador, cargas, a, α = 0,99, método del límite dentro de cada contraste, conjunto de evaluación | FAR sobre `test_fault00` (y FDR como secundaria); diferencia pareada en muestra − reservada | Réplica de ajuste (simulación normal de entrenamiento independiente), R = 20 | Rieth (clásico como ilustración) | `A0_calibracion_limites.py` con `--datos rieth --replicas 20 --n-cal` |
| **(a) Calibración, T²** | No (sin brazo T²) | F (Fase II); Beta (Fase I aplicado en Fase II); empírico en muestra; empírico reservado; opcional KDE | Ídem | FAR frente a `far_teorica_faseII`; diferencia Beta − F; diferencia empírico en muestra − reservado | Ídem | Rieth y clásico | Ídem, con `--t2-method` |
| **(b) Partición, PCA** (normales ajuste/calibración) | No (sin script; `n_components` sí existe) | Corte temporal; barajado dentro de la simulación; bloques intercalados; otra simulación | n_fit, n_cal, **a fijo**, método del límite | FAR sobre `test_fault00` | Réplica de ajuste, R = 20 | Rieth | `A2_particion_normal.py` |
| **(b) Partición, clasificadores (OE2)** | No (no hay script; `split_runs` solo parte por simulación) | KFold por muestra; GroupKFold por índice de simulación; dosis-respuesta por tamaño de bloque (1, 10, 50, 500) | Modelo, hiperparámetros, tamaño y equilibrio de entrenamiento, simulaciones de prueba externas | **Optimismo** = estimación interna − desempeño en simulaciones de prueba externas (F1 por clase sobre un conjunto común de simulaciones, B3) | Partición externa con simulaciones disjuntas, R = 5–10 (≤ 20 simulaciones por clase) | Rieth (train sin las 20 primeras muestras en los fallos) | `B2_ablacion_particion.py` (repositorio de Jonathan) |
| **(b) Ventanas, profundos (OE3)** | No | Ventanas solapadas partidas al azar frente a partición por simulación; dosis-respuesta en el paso | Arquitectura, semillas, presupuesto de ajuste en validación | Ídem, más la FAR del umbral | Réplica de entrenamiento (≥ 5 semillas y simulaciones disjuntas) | Rieth | `C2_ventanas.py` (Cesar) |
| **(c) Estandarización, PCA** | **No**: el Scaler interno anula la fuga (límites idénticos, dif. 6,6e-12) | Limpio (X_fit); combinado (X_fit ∪ prueba); por simulación (cada una con sus estadísticas) | X_fit, a fijo, límites calibrados igual | FAR y FDR, **con signo** | Réplica de ajuste | Rieth; clásico solo para ver el mecanismo | `A3_estandarizacion.py` (necesita el gancho `scaler=`) |
| **(c) Estandarización, clasificadores** | Sí (sklearn), en el repositorio de B | Scaler con entrenamiento frente a entrenamiento ∪ prueba | Todo lo demás | F1 por clase en prueba | Partición externa | Rieth | `B3_estandarizacion.py`; **Random Forest = 0 por construcción** |
| **(d1) Una simulación frente a muchas, evaluación** | Parcial (`load_rieth` y `evaluate_run`; falta el script) | Métrica de 1 simulación (cada una de las 100) frente a la media de 100 | Modelo entrenado | Dispersión; P(conclusión con 1 simulación ≠ conclusión con 100) en signo y significación, con un contraste por muestra frente al pareado por simulación | Simulación de prueba | Rieth | `A4_una_vs_muchas.py` |
| **(d2) Una simulación frente a muchas, ajuste** | Parcial | {1×500, 2×250, 5×100, 10×50} con n = 500 fijo | n total, a, método del límite | FAR y su dispersión entre réplicas | Réplica de ajuste | Rieth | Ídem |

## 3. Propuesta concreta para la hipótesis

**Se mantiene el texto aprobado.** Lo que se añade en la metodología (3.x) es su operacionalización, prerregistrada con una etiqueta antes de ejecutar Rieth:

- **H-A (factores del protocolo).** Para cada factor k, Δ_k = métrica(sucio) − métrica(limpio), en el mismo método y con los mismos datos, sobre R réplicas de Rieth, con IC al 95 %. Se declara un efecto relevante si el IC excluye ±δ_k, y se reporta el **signo**. No se da por supuesto que el efecto infle.
- **H-B (ventaja menor que la publicada).** Una tabla prerregistrada de Δ_publicada con fuente, datos, métrica y fallo, marcada [VERIFICAR] hasta abrir cada PDF. Contraste unilateral H0: Δ_nuestra ≥ Δ_publicada. Nuestra Δ se calcula en el clásico y en Rieth para separar el efecto del protocolo del de los datos.
- **H-C (sin ventaja relevante en 3, 9 y 15).** Para cada método M no lineal o profundo y cada fallo f ∈ {3, 9, 15}:
  - Estimanda:
    Δ_{M,f} = media_r[(FDR_{M,f}(r) − FAR_{M,0}(r)) − (FDR_{B,f}(r) − FAR_{B,0}(r))].
    Se calcula a FAR igualada al 1 % en simulaciones normales reservadas de entrenamiento. B es el baseline lineal fijado a priori (PCA con T² ∨ SPE a nivel Šidák y DPCA como baseline con memoria). r recorre las simulaciones de prueba, emparejadas por índice (ruido común, DIS-2).
  - **Principal: no superioridad.** H0: Δ ≥ δ frente a H1: Δ < δ. t pareado o bootstrap al α = 0,05 unilateral. **Secundario: TOST** (±δ).
  - δ = 2 puntos de exceso sobre la FAR, justificada frente a H-B (las ventajas de la Tabla 3 de Yin rondan 11–16 puntos).
  - Afirmación global por unión-intersección: todos los contrastes al 5 %, sin Holm [VERIFICAR: Berger, 1982].
  - **Controles:** positivos en los fallos 10, 11, 16 y 20 (regla: FDR del PCA-SPE entre 0,2 y 0,8 en A1 clásico, sin el 5 ni el 21), y un detector trivial. El techo del oráculo estático (3,2 %, 5,1 % y 18,5 %) se da como contexto.
- **Potencia (Monte Carlo, t pareado, δ = 0,02, α = 0,05):**

| σ_d | TOST n = 20, Δ = 0 | TOST n = 100, Δ = 0 | TOST n = 100, Δ = 0,01 | No superioridad n = 100, Δ = 0 |
|---|---|---|---|---|
| 0,007 | 1,000 | 1,000 | 1,000 | 1,000 |
| 0,03 | 0,782 | 1,000 | 0,950 | 1,000 |
| 0,05 | 0,118 | 0,980 | 0,637 | 0,991 |
| 0,07 | 0,006 | 0,770 | 0,407 | 0,883 |
| 0,10 | 0,000 | 0,267 | 0,166 | 0,627 |

- **Umbrales con n = 100 y Δ = 0:** el TOST alcanza el 80 % de potencia si σ_d ≤ 0,068 y el 90 % si σ_d ≤ 0,061. La no superioridad, si σ_d ≤ 0,080 y 0,068.
- **n para el TOST al 90 %:** con σ_d = 0,05 hacen falta unas 68 simulaciones (71 con Wilcoxon); con 0,07, 133 (139); con 0,10, 271 (284).
- **Piloto:** la σ_d entre detectores lineales es de 0,0049–0,0074 cruda y de 0,0017–0,0023 en la métrica de exceso, con 20 simulaciones. Entre lineales, la potencia es ≈ 1 incluso con n = 20. **El n lo decide la σ_d de los métodos profundos**, que incluye la variabilidad entre semillas. Hay que estimarla con un piloto de 20 simulaciones (fuera del conjunto confirmatorio) antes de fijar n.
- **Lo que no se escribe:** «no significativa» como prueba de equivalencia.

## 4. Riesgos ante el tribunal y cómo blindarse

| Riesgo | Blindaje |
|---|---|
| «Ya se sabía que hay que recalibrar» (Rato 2016; Vanhatalo 2017) | La aportación es medir el tamaño de efecto por factor y por familia, y el puente Fase I/Fase II con predicción teórica (4,05 % frente a 1 %) |
| «Hipótesis infalsable o trivial» | DIS-1 y DIS-15: no superioridad/TOST, controles positivos, detector trivial, techo de detectabilidad |
| «El efecto sale de una sola simulación» | DIS-3: réplicas de ajuste en Rieth, con IC |
| «Pseudorreplicación» | DIS-2 y DIS-14: el índice de simulación como unidad; FAR solo con `test_fault00` |
| «Comparan a FAR distinta» o «la ventaja viene de la memoria» | DIS-13: FAR igualada y DPCA como baseline |
| «Eligieron métricas y umbrales mirando la prueba» | DIS-17 e I8: prerregistro con etiqueta y simulaciones del piloto excluidas |
| «Los factores no siempre inflan» | Reportar el signo; (c) puede deflactar y (d) no sesga en esperanza (DIS-7 y DIS-9) |
| «Jackson y Yin también calibran en muestra» | Convertirlo en hallazgo (DIS-6), no esconderlo |



### Anexo · Reproducibilidad: secuencia para v0.3, lock, CI e instalación

# Reproducibilidad: empaquetado, entorno, historia git e integración continua

En adelante, `$SIM` es la carpeta de scratch `C:/Users/pabdu/AppData/Local/Temp/claude/C--Users-pabdu-OneDrive-Documentos-Universidades-UNIR-TFM/b1423c25-c9de-48ed-b4f5-76701e58807e/scratchpad/wf/reproducibilidad`. El repo real no se tocó: el md5 de `git status --porcelain` es el mismo antes y después (`50068ff6…`) y no hay etiqueta `v0.3`. En las dos copias de scratch dejé las URL de push apuntando a `NO_PUSH_SIMULACRO`, así que desde ellas no se puede publicar nada. No se hizo ningún push.

## 0. Resultado del simulacro

| Paso | Qué se hizo | Resultado |
|---|---|---|
| a | `git clone --no-hardlinks -o local -b master <repo> $SIM/sim` | Hay una desviación: clono con `-o local` porque un clon local ya trae un remoto `origin` (la ruta del repo), y el `git remote add origin …` del paso c fallaría con «remote origin already exists». |
| b | `git switch -c revision-2026-09 master` y copia de lo que lista `git status --porcelain` | Salen 28 entradas; excluyo 2 (`LICENSE.zip` y `notebooks.zip`) y copio 26. Commit `380fe70`, árbol `f9db8a2`. Los 26 blobs coinciden con `git hash-object` del working tree real. El índice queda entero en LF. |
| c | `remote add origin`, `fetch origin` y `merge origin/main --allow-unrelated-histories` | `merge-base` vacío. Salen **13 conflictos add/add** y 8 `.pyc` entran sin conflicto. Los resuelvo con `--ours`. El índice frente a la rama solo difiere en los 8 `.pyc`. Aplico `git rm -r --cached tfmfdd/__pycache__` y hago el commit `ddfc949`. El **árbol final es idéntico al de la rama** (`f9db8a2`). `origin/main`, `v0.1` y `v0.2` quedan como ancestros, así que el push a `main` será fast-forward. Un `merge -s ours` da el mismo árbol. |
| d | `.venv/Scripts/python.exe -m pip install --no-deps --target $SIM/inst $SIM/sim` | Instala `tfmfdd-0.1.0` (rueda de 27 231 bytes, 9 módulos) e importa desde un directorio neutro. Las 31 pruebas pasan contra la copia instalada en 1,50 s. Hay aviso de setuptools 84.0.0 (REP-5) y `configs/` no viaja (REP-4). |
| e | Sin push | Ver el primer párrafo. |

Conflictos exactos, con el número de bloques `<<<<<<<` de cada uno:

| Fichero | Bloques |
|---|---|
| notebooks/01_primeros_pasos.ipynb | 46 |
| notebooks/02_arranque_diagnostico.ipynb | 25 |
| README.md | 4 |
| notebooks/03_arranque_deeplearning.ipynb | 4 |
| experiments/A1_pca_baseline.py | 3 |
| docs/decisiones.md, notebooks/README.md, pyproject.toml, requirements.txt | 1 cada uno |
| results/summary/A1_pca_baseline_{SPE,T2,runs}.csv, B1_clasificadores_clasico.csv | 1 cada uno |

En `pyproject.toml` y `requirements.txt`, el lado de la rama añade `pyreadr` y `pyyaml` (o solo `pyyaml`) y el lado de `origin/main` no añade nada.

**Comprobaciones añadidas:**
- **Venv limpio** creado con el Python global 3.12.0 e instalado desde el lock: `pip check` sin errores, `pip freeze` igual al lock (27 paquetes), 31 passed. La instalación tardó 1 min 35 s.
- **Mismos CSV:** con ese venv y el código del clon, A0, A0 `--yin-vars` y A1 regeneran los 7 CSV de `results/summary/` idénticos byte a byte salvo el fin de línea, con una diferencia numérica máxima de 0.0. Los datos se leyeron del repo real en solo lectura.
- **Lock en Linux:** con `pip install --dry-run --only-binary=:all: --platform manylinux… --python-version 3.12` se obtienen 27 ruedas binarias para Linux CPython 3.12, el mismo conjunto que el lock.
- **Simulacro 2** (`$SIM/simulacro2.sh`): ejecuta las fases A a G de la sección 1 en otro clon, con los `__pycache__` ignorados reales y dos `.zip` ficticios. Resultados:
  - salen los mismos 13 conflictos;
  - el árbol final es el de la rama;
  - los `.pyc` ignorados se sobrescriben sin abortar la fusión;
  - los `.zip` quedan ignorados;
  - `v0.3` instala como 0.3.0 y el control de etiqueta pasa;
  - 31 passed.
- **Lo que no se simuló:**
  - tus correcciones de B2, B3 e I1;
  - `pip install -e` en tu `.venv` y la ejecución de los cuadernos en la fase F;
  - las fases H, I y J (publicación).
- Los SHA de tu repo real serán otros, porque cambian el autor y la fecha.

## 1. Secuencia exacta para tu repositorio real (Git Bash)

Requisito previo, no incluido: corregir B2, B3, I1 y el README (líneas 24, 27, 97, 105, 123 y 142). Mientras ejecutas las fases C a H, pausa la sincronización de OneDrive.

```bash
# ---- Preparación ----
cd "/c/Users/pabdu/OneDrive/Documentos/Universidades/UNIR/TFM/tfm-fdd-core/tfm-fdd-core"
export MSYS_NO_PATHCONV=1 PYTHONIOENCODING=utf-8   # sin esto Git Bash reescribe "rama:ruta"
PY=.venv/Scripts/python.exe
PYABS="$PWD/.venv/Scripts/python.exe"
SIM="/c/Users/pabdu/AppData/Local/Temp/claude/C--Users-pabdu-OneDrive-Documentos-Universidades-UNIR-TFM/b1423c25-c9de-48ed-b4f5-76701e58807e/scratchpad/wf/reproducibilidad"

# ---- A. Comprobaciones (no cambian nada) ----
git branch --show-current            # revision-2026-09
git fetch origin
git rev-parse origin/main            # a3e0eacb8dc72469b97e679e4f1244db8830891f; si es otro, PARA y vuelve a simular
git ls-remote origin                 # HEAD y refs/heads/main -> a3e0eac..., refs/tags/v0.1 -> bf6cb4f...

# ---- B. Identidad de autor, solo para este repositorio (hoy es pablo@local) ----
git config user.email "112780608+pabdus@users.noreply.github.com"

# ---- C. Commit del estado actual con rutas explícitas (nunca git add -A: hay dos .zip de 109 MB sin ignorar) ----
git add -- README.md CAMBIOS.md docs/decisiones.md docs/revision_2026-09-27.md \
  experiments/A0_calibracion_limites.py experiments/A1_pca_baseline.py \
  notebooks/01_primeros_pasos.ipynb notebooks/02_arranque_diagnostico.ipynb \
  notebooks/03_arranque_deeplearning.ipynb notebooks/README.md \
  pyproject.toml requirements.txt results/summary tests/test_correcciones_2026_09.py \
  tfmfdd/data.py tfmfdd/metrics.py tfmfdd/pca.py tfmfdd/stats.py
git status --porcelain | grep -v '^[AM]  '   # solo "?? LICENSE.zip" y "?? notebooks.zip"
git commit -m "Aplica el zip del 22/09 y regenera los CSV de A0 y A1"

# ---- D. Unificar con GitHub sin reescribir nada ----
git merge origin/main --allow-unrelated-histories --no-edit
#   esperado: "Automatic merge failed" con los 13 conflictos add/add de la tabla
CONF=$(git diff --name-only --diff-filter=U); echo "$CONF" | wc -l   # 13
git checkout --ours -- $CONF
git add -- $CONF
git rm -r -q --cached tfmfdd/__pycache__
git diff --cached --stat HEAD        # debe salir VACÍO: el árbol es exactamente el de tu rama
git commit --no-edit
git merge-base --is-ancestor origin/main HEAD && echo "main avanzará sin forzar"

# ---- E. Preparar v0.3 ----
printf '\n# Copias comprimidas: nunca al repositorio (GitHub rechaza ficheros de mas de 100 MB)\n*.zip\n' >> .gitignore
git check-ignore -q LICENSE.zip && git check-ignore -q notebooks.zip && echo "zips ignorados"
cp "$SIM/sim_fix/pyproject.toml" pyproject.toml          # texto en la sección 5
sed -i 's/^__version__ = "0.1.0"$/__version__ = "0.3.0"/' tfmfdd/__init__.py
grep -n '__version__ = ' tfmfdd/__init__.py              # 23:__version__ = "0.3.0"
cp "$SIM/requirements-lock.txt" requirements-lock.txt    # texto en la sección 2
mkdir -p .github/workflows && cp "$SIM/pruebas.yml" .github/workflows/pruebas.yml   # texto en la sección 3

# ---- F. Reinstalar, probar y regenerar SIEMPRE con el .venv ----
$PY -m pip install -r requirements-lock.txt              # en tu .venv no cambia nada: el lock sale de él
$PY -m pip install --no-deps -e .                        # refresca los metadatos a 0.3.0
$PY -c "import tfmfdd, importlib.metadata as m; print(tfmfdd.__version__, m.version('tfmfdd'))"   # 0.3.0 0.3.0
$PY -m pytest -q
$PY experiments/A0_calibracion_limites.py
$PY experiments/A0_calibracion_limites.py --yin-vars
$PY experiments/A1_pca_baseline.py
for nb in 01_primeros_pasos 02_arranque_diagnostico 03_arranque_deeplearning; do
  $PY -m nbconvert --to notebook --execute --inplace notebooks/$nb.ipynb
done
{ $PY --version; $PY -m pip freeze --exclude-editable; } > results/summary/ENTORNO.txt

# ---- G. Commit y etiqueta anotada ----
git add -- .gitignore pyproject.toml tfmfdd requirements-lock.txt .github README.md CAMBIOS.md docs experiments notebooks results/summary tests
git status --porcelain | grep -v '^[AM]  '   # vacío
git commit -m "tfmfdd 0.3.0: correcciones de la revision del 27/09, lock de dependencias y CI"
git tag -a v0.3 -m "tfmfdd 0.3.0"
git merge-base --is-ancestor origin/main v0.3 && echo "push de main: fast-forward"

# ---- H. Publicar sin --force ----
git push origin revision-2026-09
git push origin revision-2026-09:main   # fast-forward; si GitHub lo rechaza, NO fuerces: vuelve a A
git push origin v0.3
git push origin v0.2                    # opcional: solo para que exista la referencia histórica
git ls-remote origin                    # refs/heads/main y refs/tags/v0.3^{} deben ser el mismo commit

# ---- I. Verificar como lo harán Jonathan y Cesar ----
TMPV=$(cygpath -m "$(mktemp -d)")
$PY -m pip install --no-deps --target "$TMPV/v03" "tfmfdd @ git+https://github.com/pabdus/tfm-fdd-core@v0.3"
(cd "$TMPV" && PYTHONPATH="$TMPV/v03" "$PYABS" -c "import tfmfdd; print(tfmfdd.__version__, tfmfdd.__file__)")   # 0.3.0 y una ruta dentro de TMPV

# ---- J. Higiene de ramas (opcional) ----
git fetch origin
git switch main && git merge --ff-only origin/main && git branch -u origin/main && git switch revision-2026-09
git config --global init.defaultBranch main
```

Notas sobre la secuencia:
- **Alternativa con PR.** En lugar de `git push origin revision-2026-09:main`, puedes abrir un PR de `revision-2026-09` a `main` y fusionarlo **solo** con «Create a merge commit». «Squash» o «Rebase» sacarían de `main` tu historia local, y `v0.2` quedaría fuera.
- **Forma de instalación.** Comprobé con el analizador de pip 23.2.1 (packaging 21.3) que `tfmfdd @ git+https://github.com/pabdus/tfm-fdd-core@v0.3` es un requisito válido. La forma `git+file:///C:/…` no lo es con esa versión de pip, así que en scratch probé la etiqueta con `git+file:///…@v0.3` sin el prefijo `tfmfdd @`.
- **Los `.pyc` de la fusión.** La fusión sobrescribe los `.pyc` ignorados de `tfmfdd/__pycache__` con los de `origin/main`. No hace daño, porque Python los recompila al no coincidir con la fecha de los fuentes, pero puedes borrarlos después.

## 2. requirements-lock.txt propuesto

Sale de `pip freeze` del `.venv` (117 líneas). Se quedan 27 paquetes: el cierre de dependencias de pyproject más pytest, calculado desde los metadatos del `.venv` con `$SIM/cierre_deps.py`. Se filtran dos cosas: la línea `-e git+…@7541a55` (REP-8) y 89 paquetes del entorno Jupyter (jupyter, jupyterlab, notebook, nbconvert, ipykernel, ipython, pywinpty, etc.), que entran como dependencias directas o por extras. `colorama` y `tzdata` llevan marcador porque pytest y pandas solo los piden en Windows.

```text
# requirements-lock.txt: versiones exactas del entorno con el que se generaron los
# resultados de tfm-fdd-core v0.3 (congelado del .venv de Pablo el 2026-09-27).
# Solo CPython 3.12 (numpy 2.5.3, scipy 1.18.1 y contourpy 1.4.0 exigen >= 3.12).
# Contiene las dependencias de ejecucion de tfmfdd, pytest y el cierre transitivo
# de ambas. NO contiene tfmfdd (se instala aparte, por etiqueta) ni Jupyter.
#
# Uso:  python -m pip install -r requirements-lock.txt
#       python -m pip install --no-deps "tfmfdd @ git+https://github.com/pabdus/tfm-fdd-core@v0.3"

# --- Dependencias directas de tfmfdd (pyproject.toml) ---
numpy==2.5.3
pandas==3.0.6
scipy==1.18.1
scikit-learn==1.9.1
matplotlib==3.11.2
pyarrow==25.0.1
pyreadr==0.5.6
PyYAML==6.0.3

# --- Pruebas (extra dev) ---
pytest==9.1.1

# --- Transitivas ---
cloudpickle==3.1.2
contourpy==1.4.0
cycler==0.12.1
fonttools==4.65.0
iniconfig==2.3.0
joblib==1.6.0
kiwisolver==1.5.1
narwhals==2.26.0
packaging==26.3
pillow==12.3.0
pluggy==1.6.0
Pygments==2.21.0
pyparsing==3.3.2
python-dateutil==2.9.0.post0
six==1.17.0
threadpoolctl==3.7.0

# --- Solo Windows ---
colorama==0.4.6 ; sys_platform == "win32"
tzdata==2026.4 ; sys_platform == "win32"
```

Para ejecutar cuadernos, aparte y sin que afecte a las cifras: `python -m pip install -c requirements-lock.txt nbconvert==7.17.1 ipykernel==7.3.0` (son las versiones del `.venv`).

## 3. Workflow de GitHub Actions propuesto (`.github/workflows/pruebas.yml`)

No está escrito en el repo; el texto está en `$SIM/pruebas.yml`.

```yaml
name: pruebas

on:
  push:
    branches: [main, "revision-*"]
    tags: ["v*"]
  pull_request:
    branches: [main]
  workflow_dispatch:

permissions:
  contents: read

jobs:
  pytest:
    runs-on: ubuntu-24.04
    timeout-minutes: 15
    steps:
      - uses: actions/checkout@v6

      - uses: actions/setup-python@v6
        with:
          python-version: "3.12"
          cache: pip
          cache-dependency-path: requirements-lock.txt

      - name: Instalar dependencias fijadas y el paquete
        run: |
          python -m pip install -r requirements-lock.txt
          python -m pip install --no-deps .
          python -m pip check

      - name: Registrar el entorno
        run: |
          python --version
          python -m pip freeze

      - name: La version del paquete coincide con la etiqueta
        if: startsWith(github.ref, 'refs/tags/v')
        run: |
          python - <<'EOF'
          import os, importlib.metadata as m, tfmfdd
          tag = os.environ["GITHUB_REF_NAME"].removeprefix("v")
          v = m.version("tfmfdd")
          assert tfmfdd.__version__ == v, (tfmfdd.__version__, v)
          assert v == tag or v.startswith(tag + "."), (v, tag)
          print("etiqueta", tag, "version", v)
          EOF

      - name: Pruebas
        run: python -m pytest -q
```

Qué está comprobado y qué no:
- El YAML se validó con PyYAML.
- El control de etiqueta, ejecutado en local, pasa con 0.3.0 y falla con el 0.1.0 actual.
- En Windows se reprodujeron en local los pasos de instalación y pruebas (lock, `--no-deps .` y 31 passed). En Linux solo se comprobó que el lock resuelve con ruedas binarias.
- Las pruebas no necesitan los datos (M7), así que la CI puede correrlas.
- `git ls-remote` muestra que ya existe `v7` de `actions/checkout` y de `actions/setup-python`. Uso `v6` porque no he verificado los parámetros de `v7`.
- El workflow no se ha ejecutado en GitHub.

## 4. Instalación para Jonathan y Cesar (versión fija v0.3)

Pueden enviarlo tal cual cuando v0.3 esté publicada; antes de eso, la etiqueta y los enlaces no existen.

**Requisitos:** Python 3.12 (cualquier 3.12.x; ni 3.11 ni 3.13) y Git instalado, porque pip lo usa para descargar la etiqueta.

1. Comprueben la versión de Python: `py -3.12 --version` en Windows, `python3.12 --version` en macOS o Linux.
2. En la raíz de su propio repositorio, creen un entorno virtual, mejor fuera de OneDrive, Dropbox o iCloud: `py -3.12 -m venv .venv` en Windows, `python3.12 -m venv .venv` en macOS o Linux.
3. Actívenlo:
   - PowerShell o cmd: `.venv\Scripts\activate`
   - Git Bash: `source .venv/Scripts/activate`
   - macOS o Linux: `source .venv/bin/activate`
4. Instalen las versiones exactas y después el paquete por etiqueta:
   ```bash
   python -m pip install -r https://raw.githubusercontent.com/pabdus/tfm-fdd-core/v0.3/requirements-lock.txt
   python -m pip install --no-deps "tfmfdd @ git+https://github.com/pabdus/tfm-fdd-core@v0.3"
   python -m pip check
   ```
5. Comprueben la instalación:
   ```bash
   python -c "import tfmfdd, numpy, pandas, scipy, sklearn; print(tfmfdd.__version__, numpy.__version__, pandas.__version__, scipy.__version__, sklearn.__version__)"
   # esperado: 0.3.0 2.5.3 3.0.6 1.18.1 1.9.1
   python -m pip freeze | grep -i tfmfdd      # en PowerShell: | findstr tfmfdd
   # esperado: tfmfdd @ git+https://github.com/pabdus/tfm-fdd-core@<commit de v0.3>
   ```
6. **Si ya tenían tfmfdd instalado** (por ejemplo @v0.1 siguiendo el README actual de GitHub), desinstálenlo antes con `python -m pip uninstall -y tfmfdd` y repitan cualquier resultado que hayan calculado con esa versión.
7. **Para añadir sus propias bibliotecas** (PyTorch en el caso de Cesar, un paquete de c-medias difusas en el de Jonathan), usen siempre el lock como restricción, para que pip no cambie numpy, pandas ni scikit-learn:
   ```bash
   python -m pip install -c https://raw.githubusercontent.com/pabdus/tfm-fdd-core/v0.3/requirements-lock.txt <paquete>
   ```
   Si pip dice que no puede cumplir la restricción, no la quiten: avisen en el grupo.
8. **En el `requirements.txt` de su repositorio**, pongan estas dos líneas, nunca `@main` ni una instalación sin etiqueta:
   ```text
   -r https://raw.githubusercontent.com/pabdus/tfm-fdd-core/v0.3/requirements-lock.txt
   tfmfdd @ git+https://github.com/pabdus/tfm-fdd-core@v0.3
   ```
9. **Ejecución:** usen siempre `python -m pytest`, `python -m pip` y `python -m nbconvert` con el entorno activado. No usen los comandos sueltos `pytest` o `jupyter`, porque pueden resolverse a otro Python. Para cuadernos, instalen además `nbconvert==7.17.1 ipykernel==7.3.0` con la misma restricción `-c`.
10. **Datos:** no viajan con el paquete. `data.load_classic` busca por defecto `data/tep_classic` relativo a la carpeta desde la que ejecutan; si los tienen en otro sitio, pasen `root=`. Si Pablo publica `CHECKSUMS.sha256`, comprueben su copia con `sha256sum -c CHECKSUMS.sha256` desde la carpeta que corresponda.

## 5. pyproject.toml propuesto (probado en `$SIM/sim_fix`)

Construye sin avisos de obsolescencia. Una vez instalado da `0.3.0 0.3.0 MIT >=3.12`. La versión tiene una sola fuente, `tfmfdd/__init__.py`, con lo que M1 no puede repetirse. `pyreadr` queda solo en el extra `rieth` (convert_rieth lo importa dentro de una función), y el lock lo sigue incluyendo.

```toml
[build-system]
requires = ["setuptools>=77,<90"]
build-backend = "setuptools.build_meta"

[project]
name = "tfmfdd"
dynamic = ["version"]
description = "Infraestructura comun para la comparativa de metodos de deteccion y diagnostico de fallos sobre el Tennessee Eastman Process (TFM UNIR)"
readme = "README.md"
requires-python = ">=3.12"
license = "MIT"
license-files = ["LICENSE"]
authors = [{name = "Pablo Alberto Duque Marin"}]
dependencies = [
    "numpy>=1.24",
    "pandas>=2.0",
    "scipy>=1.10",
    "scikit-learn>=1.3",
    "matplotlib>=3.7",
    "pyarrow>=12.0",
    "pyyaml>=6.0",
]

[project.optional-dependencies]
rieth = ["pyreadr>=0.5"]
dev = ["pytest>=7.0"]

[tool.setuptools.dynamic]
version = {attr = "tfmfdd.__version__"}

[tool.setuptools.packages.find]
include = ["tfmfdd*"]

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-ra"
```

## 6. Ficheros en `$SIM`

- **Propuestas:** `requirements-lock.txt`, `pruebas.yml`, `sim_fix/pyproject.toml`, `CHECKSUMS_datos.sha256` (48 sumas SHA-256 de `data/tep_classic/*.dat` y `data/tep_rieth/*.RData`).
- **Scripts:** `simulacro2.sh`, `cierre_deps.py`, `tagcheck.py`, `ver_wheel.py`.
- **Registros:**
  - freeze: `freeze_venv.txt`, `freeze_venv_lock.txt`;
  - fusión: `merge_output.txt`, `conflictos_bloques.txt`, `sim2_merge.txt`;
  - construcción e instalación: `pip_wheel_v.log`, `pip_wheel_fix.log`, `lock_install.log`;
  - resolución para Linux: `dry_linux.log`, `report_linux.json`.
- **Copias y entornos:**
  - clones: `sim/` y `sim2/` (push anulado);
  - venv del lock: `venv_lock/`;
  - CSV regenerados: `out_lock/`;
  - instalaciones: `inst/`, `inst_v03/`, `inst_fix/`, `inst_sim2/`.

