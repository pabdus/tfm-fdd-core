# Especificación de la verificación frente a Yin et al. (2012)

Es "la especificación" a la que remiten `configs/yin2012.yaml` y `experiments/V1_verificacion_yin.py`: qué se verifica, la configuración de Yin et al. (2012), lo que el artículo no dice (los supuestos S1 a S12, con su evidencia), qué produce el script y las opciones de tolerancia. Sale de la especificación que escribió el agente ref_yin el 12/10/2026 a partir del PDF del artículo y del informe 15 de la revisión del 03/10, adaptada al repositorio: las rutas son las del repositorio, las funciones las de `tfmfdd` 0.3.0, y las cifras de los límites se recalcularon el 12/10 con el paquete (apartado 10). Las páginas son las de la paginación impresa (1567 a 1581).

Referencia: Yin, S., Ding, S. X., Haghani, A., Hao, H. y Zhang, P. (2012). A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process. *Journal of Process Control*, 22, 1567-1581. https://doi.org/10.1016/j.jprocont.2012.06.009

Cada elemento dice si es **explícito** en el artículo o un **supuesto** (S1, S2…), con su evidencia.

## 0. Qué se verifica y qué no

- Se verifica la **implementación** del código ejecutada con la configuración de Yin, que calibra los límites en muestra (memoria, párrafos 355 y 369). No se verifica el protocolo del TFM, que calibra con datos reservados.
- Se comparan la **FDR por fallo** y la **FAR en operación normal** de PCA y DPCA con las Tablas 4 y 5 de Yin (p. 1574). Los fallos 3, 9 y 15 se calculan y se reportan, pero **no cuentan** para el veredicto (párrafo 369).
- **La tolerancia no se decide aquí**: la fija Pablo antes de ejecutar (apartado 7). V1 no abre ningún fichero de prueba (`dXX_te.dat`) hasta que la tolerancia, la configuración y las variantes están escritas en `configs/yin2012.yaml`, fijadas con su huella en una fila fechada de `docs/decisiones.md`, el registro de decisiones del repositorio (una fila que se vea, con una fecha que exista; otro fichero solo con `--permitir-otro-registro`), y confirmadas en git junto con el código que se ejecuta: el script y la carpeta `tfmfdd` que importa, en el mismo repositorio que el yaml y el registro, sin bytecode que Python ejecute en lugar del fuente (un `.pyc` "unchecked-hash"). Así tampoco se puede cambiar el código después de ver un resultado y volver a ejecutar sin que quede rastro ("sin ajustes para que coincidan", párrafo 369). Los pasos están en la última sección de `docs/decisiones.md`. Si se ejecutara antes, la tolerancia dejaría de estar "fijada antes de ejecutar" (párrafo 355). Hasta el 12/10/2026, V1 solo se ha ejecutado sobre datos reales en los modos `--solo-limites` y `--huella`, que no abren ningún fichero de prueba.

## 1. Referencia

- `configs/referencias/yin2012_tablas_transcripcion.csv`: la transcripción de las Tablas 4, 5, 7 y 8, pendiente de que Pablo la verifique contra el PDF (`configs/referencias/LEEME_yin2012_transcripcion.md`; la fecha va en `referencia.transcripcion_verificada`). V1 **lee** las cifras de Yin de ese CSV (`pandas.read_csv(ruta, comment="#")`); no están escritas en ningún sitio del código.
- Principal: `tabla == 4` (FDR) y `tabla == 5` (FAR), `metodo` PCA y DPCA. Secundaria: Tablas 7 y 8 (otra configuración, apartado 2.3).
- Columnas útiles: `fallo`, `bloque` (1, 2 o 3), `valor_num`, `fallo_excluido_parrafo369` y `dudosa` (ninguna celda de PCA ni de DPCA está marcada; V1 se niega a comparar con una celda dudosa).
- Yin dan **una cifra por método**, la de la regla conjunta T² o SPE (apartado 2.5). No hay cifras de T² ni de SPE por separado con las que comparar.
- `configs/referencias/yin2012_umbrales_dibujados.csv`: los umbrales que Yin dibujan en las Fig. 2 (p. 1573) y 5 (p. 1576), leídos de los vectores del PDF, con el residuo de esa lectura (`residuo_max_marcas`). Lo usan `--solo-limites`, la prueba de regresión de los límites (apartado 8) y dos variantes del supuesto S11.

## 2. Configuración de Yin (Tablas 3 a 5)

### 2.1 Datos y variables

| Elemento | Valor | Dónde | Estado |
|---|---|---|---|
| Fuente | Ficheros de Chiang, Russell y Braatz (2001) [6]: 22 de entrenamiento de 24 h y 22 de prueba de 48 h, con el fallo a las 8 h y muestreo de 3 min | p. 1572 | Explícito |
| Ficheros | `data/tep_classic` (`d00.dat` … `d21_te.dat`); V1 acepta `--root` | | Hecho de los datos |
| Variables | 33: XMEAS(1–22) y XMV(1–11), "included for FDA, PCA and ICA related methods" (`data.YIN_VARS`) | p. 1573 | Explícito para el PCA; para el DPCA, S3 |
| Ajuste (y calibración) | `d00.dat` entero, N = 500 | La p. 1568 solo dice "Collect N samples" | S1 |
| Prueba, FDR | `d01_te.dat` … `d21_te.dat` (960 muestras; fallo tras la 160) | p. 1572 | Explícito |
| Prueba, FAR | `d00_te.dat` ("Fault Free", IDV(0), Tabla 5) | p. 1574 | S2 (el fichero no se nombra) |
| Una simulación por fallo | Sí, un fichero por fallo | | Hecho de los datos |

### 2.2 Preprocesado

| Elemento | Valor | Dónde | Estado |
|---|---|---|---|
| Escalado | Media cero y varianza uno con las N muestras del ajuste; cada muestra nueva se normaliza con esos parámetros | p. 1568, paso 1 fuera de línea y paso 1 en línea | Explícito; que la muestra nueva use la media y la desviación del ajuste se sobrentiende |
| Longitud de `d00.dat` | 500 filas, aunque la p. 1572 hable de conjuntos de entrenamiento de 24 h (480 muestras de 3 min) | Hecho de los datos frente a la p. 1572 | S1 |
| Estimador de la desviación | ddof = 1 | | S4 |
| Covarianza | (1/(N − 1)) ZᵀZ = PΛPᵀ | ec. (1), p. 1568 | Explícito |
| Otro preprocesado | Ninguno mencionado | | Explícito por omisión |
| DPCA: matriz aumentada | z_i(h)ᵀ = [z_iᵀ … z_{i−h}ᵀ], (h + 1)·m columnas con los vectores ya normalizados | p. 1570 | Explícito (la Tabla 1, p. 1571, dice "hm × hm", incoherente con la definición; manda la definición) |
| DPCA: orden | Yin escriben la matriz con los z normalizados: escalar y después retardar. En V1, con el gancho `PCAMonitor.fit(..., scaler=Scaler().fit(X))`; sin el gancho, `PCAMonitor` retarda y después escala | p. 1570 | S5 |

### 2.3 Componentes

| Configuración | PCA | DPCA | Criterio | Dónde |
|---|---|---|---|---|
| Principal (Tablas 4 y 5) | l = 9 | l = 17, h sin declarar (S6: h = 2) | PRESS por validación cruzada | Tabla 3, p. 1572; texto, p. 1573 |
| Secundaria (Tablas 7 y 8) | l = 17 | h = 2; l = 40 en la Tabla 6 y 42 en el texto | Porcentaje de varianza, en torno al 90 % | Tabla 6 y texto, p. 1575 |

- Se fija l; no se reproduce el PRESS, porque el artículo no dice qué variante usó.
- Con `d00.dat` (33 variables, ddof = 1), el 90 % de la varianza da 17 componentes en el PCA y 40 en el DPCA con h = 2, en los dos órdenes de escalado (apartado 10). Apoya que la Tabla 6 (40) es la buena y que la errata está en el texto (42). Lo decide Pablo, si incluye la configuración secundaria (`secundaria.n_components.DPCA`).

### 2.4 Nivel y límites

| Elemento | Valor | Dónde | Estado |
|---|---|---|---|
| Nivel | α = 0,01 (confianza 0,99) | La p. 1568 solo dice "a given significant level α" | S7 |
| Límite del T² | J_th,T² = l(N² − 1) / [N(N − l)] · F_α(l, N − l), Fase II (Tracy et al. [47]) | ec. (3), p. 1568 | Explícito. En el paquete: `limits.t2_limit(l, N, 0.99)`; en `PCAMonitor`, `t2_method="f"` |
| Límite del SPE | Jackson y Mudholkar [25], con θ_i = Σ_{j=l+1}^{m} λ_j^i | ec. (2), p. 1568 | Explícito, con errata: el PDF imprime el exponente 2 para i = 1, 2, 3 (S8). En el paquete: `limits.spe_limit_jackson`; `spe_method="jackson"` |
| ¿Jackson o g·χ²_h? | El artículo menciona los dos y no dice cuál usó en las Tablas 4 y 5 | p. 1568 | S8 (Jackson) |
| Calibración | En muestra: la F usa el N del ajuste y Jackson, sus autovalores. Los dos son paramétricos: un `X_cal` no cambiaría nada. En V1, `calibration="in_sample"`, y `limit_source_` vale `"parametric_fit"` en los dos | ec. (2) y (3) | Explícito |
| DPCA | "The remaining procedures for off-line design and on-line computation are identical with standard PCA": las mismas ec. (2) y (3) sobre la matriz aumentada | p. 1570 | Explícito; pero el T² dibujado no se reproduce (S11) |

### 2.5 Estadísticos y regla de alarma

- SPE = zᵀ P_res P_resᵀ z (ec. 4) y T² = zᵀ P_pc Λ_pc⁻¹ P_pcᵀ z (ec. 5), p. 1568. Es lo que calcula `PCAMonitor`.
- **Regla: disyunción.** "SPE ≤ Jth,SPE and T2 ≤ Jth,T2 ⇒ fault free, otherwise faulty" (p. 1568, paso 3); "if one of the test statistics exceeds threshold, a successful fault detection is achieved" (p. 1572-1573). Desigualdad estricta: igual al límite no es alarma, como en `metrics.alarms`.
- En el paquete: `PCAMonitor.combined_index(t2, spe)`, max(T²/J_T², SPE/J_SPE) con límite 1, da exactamente las mismas alarmas que la disyunción.

### 2.6 Ventanas de FDR y FAR, y retardo

| Elemento | Valor | Dónde | Estado |
|---|---|---|---|
| FDR | n.º de muestras con J > J_th y f ≠ 0 / n.º de muestras con f ≠ 0 × 100 | p. 1572 | Explícito |
| Ventana de la FDR | Muestras 161 a 960 de cada `dXX_te.dat` (800) | p. 1572 (fallo a las 8 h = 160 muestras de 3 min) | S10 |
| FAR | n.º de muestras con J > J_th y f = 0 / n.º de muestras con f = 0 × 100 | p. 1572 | Explícito |
| Ventana de la FAR | 800 muestras de `d00_te.dat`, previsiblemente las 161 a 960 | No lo dice | S9, **pendiente** |
| Retardo | Yin no lo reportan | | Explícito por omisión |

- En V1: la FDR de un fichero con fallo es `metrics.evaluate_run(indice, 1.0, onset=160)`; la FAR, `metrics.evaluate_run(indice, 1.0, onset=None, far_window=ventana)` con la ventana de `supuestos.S9_far_ventana`. Las dos cuentan además las alarmas y las muestras evaluadas.
- El retardo (k = 3) se calcula y se reporta como información, pero **no entra en el veredicto**: no hay cifra de Yin con la que compararlo.

## 3. Supuestos (lo que el artículo no dice)

Cada supuesto va con su evidencia y con la alternativa que conviene declarar de antemano como variante (ninguna se elige después de ver resultados). Están en `configs/yin2012.yaml`, sección `supuestos`, y las variantes en `variantes`.

Las cifras de esta tabla son límites calculados solo con `d00.dat`, el fichero de ajuste: ningún fichero de prueba. Configuración común: 33 variables, ddof = 1, nivel de confianza 0,99, calibración en muestra, T² por la ec. (3) y SPE por Jackson con el exponente i, PCA con l = 9 y N = 500 filas, DPCA con l = 17 y h = 2, salvo donde se dice otra cosa. Procedencia: `results/summary/V1_limites_d00.csv` (`python experiments/V1_verificacion_yin.py --solo-limites`), con el umbral dibujado y el cociente entre la diferencia y el residuo de la lectura en `abs_diff_over_residual`; y, para las que V1 no calcula, las llamadas del apartado 10. Los umbrales dibujados están en `configs/referencias/yin2012_umbrales_dibujados.csv`.

| Id | Supuesto | Evidencia | Alternativa a declarar |
|---|---|---|---|
| S1 | Ajuste con `d00.dat` entero, N = 500 | El SPE del PCA con 500 filas (23,4063) queda dentro de una vez el residuo de la lectura en las dos figuras; con las 480 primeras (23,5083), a unas cinco veces (`abs_diff_over_residual`). El T² discrimina menos: 22,3948 con 500 filas y 22,4261 con 480 | N = 480 (24 h × 20 muestras/h, lo que dice la p. 1572 de los ficheros de entrenamiento): variante `S1_n480` |
| S2 | La FAR se mide en `d00_te.dat` | La Tabla 5 la da en la fila "Fault Free, IDV(0)" | La media de las 160 primeras muestras de los 21 ficheros con fallo. Es incompatible con la aritmética: con 3360 muestras, solo 3 de las 18 FAR de las Tablas 5 y 8 son de la forma k/3360 (apartado 10) |
| S3 | El DPCA usa las mismas 33 variables | "PCA ... related methods" (p. 1573) | Ninguna razonable |
| S4 | ddof = 1 en el escalado | El SPE del PCA es 23,4063 con ddof = 1 y 23,4532 con ddof = 0; el segundo queda a más de dos veces el residuo de la lectura en las dos figuras | ddof = 0: variante `S4_ddof0` |
| S5 | DPCA: escalar y después retardar (la notación de la p. 1570) | SPE del DPCA: 57,8496 escalando primero (con el gancho del paquete, que recentra la matriz con retardos; el cálculo a mano de ref_yin, sin recentrar, difiere en la cuarta cifra decimal) y 57,9052 retardando primero. El primero queda dentro de una vez el residuo de la lectura en las dos figuras; el segundo, no | Retardar y después escalar, lo que hace `PCAMonitor` sin gancho: variante `S5_retardar_y_escalar` |
| S6 | h = 2 también en la configuración principal | Solo h = 2 da un SPE de Jackson cercano al dibujado. Con l = 17, escalando primero y retardando primero: h = 1, 32,0867 y 32,1001; h = 2, 57,8496 y 57,9052; h = 3, 84,3352 y 84,4212 | Ninguna: si no se acepta, el DPCA de la Tabla 4 no se puede verificar |
| S7 | α = 0,01 | El T² del PCA con la ec. (3), l = 9, N = 500 y confianza 0,99 (22,3948) queda dentro de la lectura del dibujado; con 0,95 sería 17,4037 | Ninguna |
| S8 | SPE por Jackson con el exponente i | Jackson da 23,4063; Box en muestra, 21,8084; Jackson con la errata impresa (exponente 2), 21,8765. Solo el primero queda cerca del dibujado | Ninguna |
| S9 | FAR sobre las muestras 161 a 960 de `d00_te.dat` | Las 18 FAR de las Tablas 5 y 8 son de la forma k/800, y solo 2 de 18 son de la forma k/960 (apartado 10) | Las 960 muestras (`S9_far_1_960`); las muestras 1 a 800 (`S9_far_1_800`), también compatibles con k/800. **Pendiente**: la ventana la fija Pablo (`supuestos.S9_far_ventana`; propuesta [160, null]) |
| S10 | FDR sobre las muestras 161 a 960 | Las 84 FDR de PCA y DPCA de las Tablas 4 y 7 son de la forma k/800 (apartado 10) | Ninguna |
| S11 | Límite del T² del DPCA | **Abierto.** La ec. (3) con l = 17 y N = 498 (las filas del ajuste con retardos) da 35,2549 y queda a más de 40 veces el residuo de la lectura del dibujado. Ninguna alternativa natural lo reproduce: Fase I (Beta), 32,8571; χ²(17), 33,4087; ec. (3) con l = 16, 33,6829; percentil 99 empírico en muestra, 32,3640 retardando primero (32,4512 escalando primero); KDE en muestra, 33,6058 (33,6271). La ec. (3) con l = 17 da 34,0041 con N = 3 × 498 y 34,0017 con N = 1500, dentro de la lectura: es compatible con que Yin usaran en el T² del DPCA un N igual a las filas por h + 1, pero es una conjetura numérica, no algo que diga el artículo | Principal: la ec. (3) con N = 498, lo que dice el artículo (`f_n_ajuste`). Variantes: N = (h + 1) por las filas (`S11_t2_n_por_h1`) y el umbral dibujado de cada figura (`S11_t2_dibujado_fig2`, `S11_t2_dibujado_fig5`) |
| S12 | Las h primeras muestras de cada fichero de prueba del DPCA no se evalúan (NaN) | Es lo que hace `PCAMonitor.score`. No afecta a la FDR (161 a 960) ni a la FAR en [160, null]; solo a la variante de la FAR sobre las 960 (quedan 958) | Ninguna |

## 4. Lo que el artículo no dice y no hay que inventar

- El retardo de detección: no se reporta y no se compara.
- Intervalos o repeticiones: una sola simulación por fallo en los datos de [6]; ninguna cifra de Yin trae incertidumbre.
- El redondeo: Yin dan dos decimales; como las tasas son k/800, la diferencia se puede expresar exactamente en muestras (apartado 6). Yin no redondean igual todos los casos que acaban en 5 (en la Tabla 5, por ejemplo, la FAR de MICA es 13/800 = 1,625 y sale como 1,63, y la de TPLS es 157/800 = 19,625 y sale como 19,62): con `tolerancia.comparacion: conteo`, V1 compara el recuento, no el redondeo.
- El tratamiento de las primeras muestras del DPCA en los ficheros de prueba (S12).

## 5. Algoritmo (PCA; el DPCA igual con `lags = 2`)

1. Cargar `d00.dat` con `data.load_classic(0, "train", root)` (corrige la transposición y avisa) y quedarse con `data.YIN_VARS`.
2. Ajustar el monitor con las filas que diga S1, l fijo (PCA: 9 en la principal y 17 en la secundaria; DPCA: 17 y la que se decida), confianza 0,99, T² por la F con el N del ajuste, SPE por Jackson y `calibration="in_sample"`. **No usar `--cal-fraction 0` de A1** para "quitar" la reserva: desde la v0.3 es un error.
3. Comprobar los límites con los umbrales dibujados (`--solo-limites`, apartado 8) antes de seguir.
4. Para cada fallo de 1 a 21: cargar `dXX_te.dat`, puntuar, formar el índice combinado y calcular la FDR en las muestras 161 a 960, más la FDR de T² y SPE por separado (diagnóstico) y el retardo (información).
5. Cargar `d00_te.dat`, puntuar y calcular la FAR en la ventana de S9 y en las variantes declaradas.
6. Leer las cifras de Yin del CSV, calcular las diferencias y aplicar la tolerancia fijada.

En V1, esos pasos son `ajustar`, `cargar_prueba` (que exige el permiso de la regla de orden), `medir_fdr`, `medir_far`, `comparar`, `evaluar_tolerancia` y `resumir`. Para el DPCA, el ajuste y la puntuación reciben `runs` (cada fichero es una trayectoria) y la salida queda alineada con las muestras originales, así que la ventana 161 a 960 es la misma que en el PCA.

## 6. Qué produce el script

- `results/summary/V1_verificacion_yin.csv`: una fila por configuración, variante, método y fichero, con `ours` (nuestra tasa en %), `yin_value`, `diff_pp` (en puntos), `n_alarms`, `n_evaluated`, `yin_count` (el recuento que implica Yin sobre 800 muestras), `diff_samples`, `diff_pp_count`, `abs_diff_verdict`, `within_tolerance` (vacío en 3, 9 y 15), `fdr_t2`, `fdr_spe`, `delay_samples`, `block_yin`, `excluded_369` y la configuración en cada fila: variables, componentes, retardos, filas del ajuste, ddof, orden de escalado, nivel, métodos y fuentes de los límites, límites, ventanas, SHA-256 de los ficheros, huella, fichero y fecha de la fila del registro (`decision_file`, `decision_date`) y commit.
- `results/summary/V1_verificacion_yin_resumen.csv`: una fila por configuración, variante y método, con los criterios de la tolerancia, la correlación de Spearman (solo como complemento: no detecta un desplazamiento uniforme) y el veredicto. Solo la configuración principal sin variantes tiene veredicto ("verificado" o "no verificado"; "descriptivo" en el DPCA si así se decide). Las variantes y la secundaria salen como "informativo", con los valores pero sin aprobado ni suspenso.
- Cada CSV lleva su `.provenance.json`, que incluye el fichero, la fecha y la fila del registro y el estado de git.
- `--solo-limites` escribe `results/summary/V1_limites_d00.csv` (los límites de la configuración principal y de las variantes que cambian el modelo, frente a los dibujados).

## 7. Tolerancia: opciones y lo que implica cada una

No se decide aquí: la fija Pablo antes de ejecutar, en la sección `tolerancia` de `configs/yin2012.yaml`. Las tres opciones del informe 15 (§5) usan la misma configuración principal, cuentan los 18 fallos de los bloques 1 y 2 y dejan las Tablas 7 y 8 como comprobación secundaria. En el yaml se escriben como criterios (un criterio que no se usa se escribe `no_aplica`; null significa "sin decidir"), y la FAR es obligatoria (párrafo 369).

| Opción | Regla | Qué implica |
|---|---|---|
| A. Réplica estricta | \|ΔFDR\| ≤ 0,5 puntos en cada uno de los 18 fallos y \|ΔFAR\| ≤ 0,5 | 0,5 puntos son 4 muestras de 800. Detecta casi cualquier error de implementación. Un detalle no publicado (S1, S4, S5, S9) puede hacerla fallar sin error nuestro; por eso las variantes tienen que estar declaradas antes. En el DPCA no es aplicable mientras S11 siga abierto |
| B. Tolerancia igual a la diferencia mínima relevante | \|ΔFDR\| ≤ 2 puntos en al menos 16 de los 18, ninguno por encima de 5, y \|ΔFAR\| ≤ 2 | 2 puntos son 16 muestras: un error que mueva 10 pasa. Ata la tolerancia a δ, que **todavía no está fijado**; si δ cambia, la tolerancia cambia con él. Admite dos fallos fuera sin decir cuáles: si los dos están en el bloque 2, el más sensible a l y al límite, la verificación pasa con dos de sus seis fallos sin cuadrar |
| C. Global | Media de \|ΔFDR\| ≤ 1 punto en los 18, máximo ≤ 5, y \|ΔFAR\| ≤ 2 | Una sola cifra. Una media puede esconder un fallo concreto mal implementado; los fallos en los que Yin dan una FDR del 100 % en el PCA rebajan la media sin informar de casi nada |

Además, en el yaml: `tolerancia.comparacion` (`conteo`, la diferencia exacta en muestras, o `impreso`, frente a la cifra redondeada; solo difieren en el borde del umbral) y `tolerancia.dpca` (`criterio`, la misma regla que el PCA, o `descriptivo`, la diferencia por fallo sin aprobado ni suspenso mientras S11 siga abierto).

**Qué hacer si no cuadra** (informe 15, §5.3): primero la comparación de los límites (apartado 8); después, solo las variantes declaradas; si sigue fuera, se reporta "no verificado", con la diferencia por fallo en puntos y en muestras, se informa al director y va a las limitaciones de la memoria. Las ablaciones en Rieth no quedan invalidadas, pero ya no se puede escribir que la implementación está verificada.

Sea cual sea la opción, el bloque 3 (fallos 3, 9 y 15) se reporta y no decide.

## 8. Prueba de regresión de los límites (sin ficheros de prueba)

Con `d00.dat`, 33 variables y confianza 0,99, el paquete tiene que dar los umbrales que Yin dibujan. `tests/test_v03_experimentos.py` lo comprueba (se salta si no está `d00.dat`): lee los umbrales del CSV de referencia, no los escribe a mano, y usa como tolerancia de lectura una vez `residuo_max_marcas` de cada fila. Comprueba seis pares: T² y SPE del PCA y SPE del DPCA, en las Fig. 2 y 5. El T² del DPCA no puede entrar mientras S11 siga abierto.

## 9. Trampas conocidas

- `d00.dat` viene transpuesto (52 × 500); `load_classic` lo corrige y avisa.
- `A1_pca_baseline.py` no es la verificación: usa 52 variables, componentes por varianza, límites con datos reservados, T² y SPE por separado y la FAR sobre las 960 muestras. Por eso no ofrece las 33 variables ni un número fijo de componentes: la configuración de Yin sobre los ficheros de prueba solo pasa por V1.
- `--cal-fraction 0` en A1 es un error desde la v0.3 (antes dejaba el límite del SPE en NaN sin avisar).
- La configuración de Yin no va en `configs/baseline.yaml`, que es el protocolo común: va en `configs/yin2012.yaml`.
- V1 se niega con un `tfmfdd` instalado desde una rueda y con cualquier cambio sin confirmar en el script o en `tfmfdd/` (también un `.py` sin versionar, aunque git lo ignore): el código forma parte de lo que se fija. Se ejecuta desde la raíz del repositorio, con el paquete instalado en modo editable o con `PYTHONPATH`.
- Una fila del registro escrita dentro de un comentario `<!-- -->` o de un bloque de código no cuenta, ni una cuya fecha no exista. Un yaml no válido (por ejemplo, una variante con un supuesto en null o una tolerancia no finita) hace que V1 se niegue sin leer nada.
- En el clásico, `simulationRun` vale 1 en todos los ficheros: juntar varios y pasarlos al DPCA con `runs` de una sola columna fundiría sus trayectorias. V1 puntúa cada fichero por separado, como una trayectoria, y el paquete avisa si una trayectoria pasa de 960 filas.
- Las erratas de Yin (el exponente de θ_i en la ec. 2; 42 frente a 40 componentes; s = 15 frente a 16; las celdas de MPLS de las Tablas 4 y 7) se documentan en la memoria, para que nadie "corrija" nuestras cifras hacia una errata.

## 10. Cómo se reproducen las cifras de este documento

Todas salen de `d00.dat` o del CSV de la transcripción; ninguna de un fichero de prueba. Recalculadas el 12/10/2026 con `tfmfdd` 0.3.0, ya con la ronda de correcciones, y el Python global (versiones de `requirements-probado.txt`). La corrección del límite KDE (ahora el cuantil exacto de la KDE) movió las dos cifras KDE de S11 en la cuarta cifra decimal; las demás no cambiaron.

- Límites de la configuración principal y de las variantes que cambian el modelo, frente a los dibujados: `python experiments/V1_verificacion_yin.py --solo-limites` (escribe `results/summary/V1_limites_d00.csv`).
- Las demás cifras de los apartados 2.3 y 3:

```python
import numpy as np
from scipy import stats
from tfmfdd import data, limits
from tfmfdd.pca import PCAMonitor
from tfmfdd.preprocessing import Scaler

X = data.values(data.load_classic(0, "train"), columns=data.YIN_VARS)     # d00.dat, 500 x 33
uno = np.ones(len(X))                                                     # una sola trayectoria

def dpca(h, escalar_primero, **kw):                                       # l = 17, en muestra
    m = PCAMonitor(n_components=17, lags=h, alpha=0.99, spe_method="jackson", **kw)
    extra = {"scaler": Scaler().fit(X)} if escalar_primero else {}
    return m.fit(X, uno, calibration="in_sample", **extra)

limits.t2_limit(9, 500, 0.95)                                             # S7
PCAMonitor(n_components=9, alpha=0.99, spe_method="box").fit(X, calibration="in_sample").spe_limit_  # S8, Box
[dpca(h, e).spe_limit_ for h in (1, 2, 3) for e in (True, False)]        # S5 y S6
n = dpca(2, True).n_fit_                                                  # 498
limits.t2_limit(17, n, 0.99), limits.t2_limit_beta(17, n, 0.99), stats.chi2.ppf(0.99, 17)  # S11
limits.t2_limit(16, n, 0.99), limits.t2_limit(17, 3 * n, 0.99), limits.t2_limit(17, 1500, 0.99)
[dpca(2, e, t2_method=t).t2_limit_ for t in ("empirical", "kde") for e in (False, True)]
PCAMonitor(variance=0.90, alpha=0.99).fit(X, calibration="in_sample").n_components_  # 2.3
```

- El SPE de Jackson con la errata impresa (S8) se calcula a mano con los autovalores residuales del PCA de 9 componentes (`eigenvalues_[9:]`), poniendo θ_1 = θ_2 = θ_3 = Σ λ_j² en la ec. (2).
- Las rejillas (S2, S9 y S10): una cifra impresa v es de la forma k/n si existe un entero k con |100 k / n − v| ≤ 0,005, es decir, si 100 k / n redondeado al centésimo más cercano da v, con los empates en cualquier sentido. Se aplica a las 18 FAR de las Tablas 5 y 8 (todos los métodos) y a las 84 FDR de PCA y DPCA de las Tablas 4 y 7 de `configs/referencias/yin2012_tablas_transcripcion.csv`.
