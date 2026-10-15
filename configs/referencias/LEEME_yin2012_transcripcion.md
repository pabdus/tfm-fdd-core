# Referencias de Yin et al. (2012) para la verificación de la implementación

**Transcripción de Claude (agente ref_yin, 11 y 12/10/2026), verificada por Pablo contra el PDF el 14/10/2026.** Pablo tecleó, leyendo la tabla impresa, las nueve columnas de las Tablas 4, 5, 7 y 8, y comprobó los umbrales dibujados de las Fig. 2 y 5: ninguna discrepancia. El registro de esa comprobación está fuera del repositorio (`Contexto/04_Investigacion/Revisiones_E2/DuquePablo_TFM_Verificacion_Transcripcion_Yin_14oct.md`), y la fecha, en `configs/yin2012.yaml` (clave `referencia.transcripcion_verificada`). Sin esa fecha, `experiments/V1_verificacion_yin.py` se niega a comparar la transcripción con nada.

## Qué hay en esta carpeta

| Fichero | Qué es | Quién lo lee |
|---|---|---|
| `yin2012_tablas_transcripcion.csv` | Las Tablas 4, 5, 7 y 8 de Yin et al. (2012): FDR y FAR (%) de los nueve métodos sobre los datos de Chiang et al. (2001), los del TEP clásico | `experiments/V1_verificacion_yin.py` (las cifras de Yin se leen de aquí; no se teclean en ningún sitio) |
| `yin2012_umbrales_dibujados.csv` | Los límites de control que Yin et al. dibujan como línea roja discontinua en las Fig. 2 (p. 1573) y 5 (p. 1576), leídos de los vectores del PDF | La prueba de regresión de los límites (`tests/test_v03_experimentos.py`), el modo `--solo-limites` de V1 y una variante del T² del DPCA (supuesto S11) |

Los dos se leen con `pandas.read_csv(ruta, comment="#")`: las primeras líneas, que empiezan por `#`, dicen la fuente y el estado. Excel las muestra como filas.

## Fuente

Yin, S., Ding, S. X., Haghani, A., Hao, H. y Zhang, P. (2012). A comparison study of basic data-driven fault diagnosis and process monitoring methods on the benchmark Tennessee Eastman process. *Journal of Process Control*, 22, 1567-1581. https://doi.org/10.1016/j.jprocont.2012.06.009

PDF usado: `Contexto\04_Investigacion\TFM_Bibliografia\00_Comunes\Yin_2012_JProcessControl_ComparativaTEP.pdf` (15 páginas; las etiquetas de página del PDF, 1567 a 1581, coinciden con la paginación impresa). El PDF no está en el repositorio.

## Qué tablas son

| Tabla | Título | Página | Configuración |
|---|---|---|---|
| 4 | "FDRs (%) based on TE data sets given in [6]." | 1574 | Tabla 3 (p. 1572): PCA con 9 componentes, DPCA con 17 |
| 5 | "FARs (%) based on TE data sets given in [6]." | 1574 | Tabla 3 |
| 7 | "FDRs (%) based on TE data sets given in [6] with different design parameters." | 1576 | Tabla 6 (p. 1575): PCA con 17, DPCA con 40 (el texto dice 42) |
| 8 | "FARs (%) based on TE data sets given in [6] with different design parameters." | 1577 | Tabla 6 |

[6] es Chiang, Russell y Braatz (2001). Las Tablas 9 a 12 usan simulaciones propias con otro simulador y no se pueden reproducir con los datos del TFM: no se transcriben.

Yin et al. dan **una sola cifra por método y fallo**, la de la regla conjunta: hay alarma si cualquiera de los estadísticos del método supera su límite (p. 1568, paso 3 del PCA; p. 1572-1573). No hay cifras de T² ni de SPE por separado. Por eso la columna `estadistico` vale siempre `combinado` y `estadisticos_combinados` dice qué combina cada método.

## Columnas de `yin2012_tablas_transcripcion.csv`

- `tabla`, `pagina_impresa`, `metrica` (FDR o FAR), `configuracion` (la tabla de parámetros que le corresponde).
- `metodo`, `estadistico`, `estadisticos_combinados`.
- `fallo` (0 = operación normal, la fila IDV(0) de las tablas de FAR), `etiqueta_impresa`, `orden_fila`, `bloque` (1, 2 o 3: los tres bloques que separan las líneas horizontales de las Tablas 4 y 7; 0 en la fila IDV(0) de las Tablas 5 y 8).
- `valor_impreso`: el texto tal como está impreso (cuatro celdas del original llevan coma decimal: "24,25", "18,75", "21,88", "18,75"). `valor_num`: el mismo valor como número.
- `resaltado_rojo`: 1 si la celda está en rojo y negrita (la FDR más alta del fallo; en las tablas de FAR, la más baja).
- `fragmentos_pdf`: cómo viene partido el número en la capa de texto del PDF.
- `compatible_k800`: 1 si el valor es k/800 × 100 redondeado a dos decimales (las tasas se calculan sobre 800 muestras).
- `fallo_excluido_parrafo369`: 1 para los fallos 3, 9 y 15, que el párrafo 369 de la memoria deja fuera del veredicto.
- `dudosa`: 1 si la celda queda pendiente (ver `nota`). Ninguna celda de PCA ni de DPCA está marcada; V1 se niega a comparar con una celda dudosa.
- `nota`: errata o incoherencia detectada, con su evidencia.

## Columnas de `yin2012_umbrales_dibujados.csv`

`pagina_impresa`, `figura`, `fallo` (el fallo de la figura), `panel`, `metodo`, `estadistico`, `y_pdf` (coordenada de la línea en el PDF), `umbral_leido` (en unidades del eje), `unidades_por_punto`, `residuo_max_marcas` (el mayor residuo del ajuste lineal de las marcas del eje: mide la precisión de la lectura) y `n_marcas`. La prueba de regresión usa `residuo_max_marcas` para fijar su tolerancia de lectura.

## Cómo se hizo

1. Texto de cada tabla con coordenadas, color y fuente (PyMuPDF), filas por la coordenada y, columnas por la distancia al centro de la cabecera.
2. Lectura visual de las cuatro tablas renderizadas a 4x: coinciden celda a celda con la reconstrucción.
3. Controles internos: todas las celdas están en la rejilla k/800 salvo MPLS-IDV(16) de la Tabla 4 (marcada como dudosa); las celdas rojas son el máximo de su fila de FDR; FDA y MPLS coinciden entre las Tablas 4 y 7, como dice la p. 1575, salvo dos celdas de MPLS (marcadas).
4. Comprobación cruzada, después de cerrar la transcripción, con la del informe 15 de la revisión integral del 03/10 (88 celdas de PCA y DPCA) y con las cifras de la nota técnica de Pablo (24 celdas y 11 cifras de la prosa): ninguna discrepancia de valor.
5. Umbrales dibujados: líneas rojas discontinuas de los vectores del PDF, convertidas a unidades con un ajuste lineal de las marcas del eje de cada panel.

Los scripts que lo hicieron y sus salidas están fuera del repositorio (carpeta de trabajo del agente ref_yin). El detalle está en su informe y en la especificación de la verificación que usó `experiments/V1_verificacion_yin.py`.

## Cómo la verifica Pablo

Para el OE1 basta con las columnas PCA y DPCA de las Tablas 4 y 5: 42 FDR y 2 FAR (`tabla in (4, 5)` y `metodo in ("PCA", "DPCA")`). Como V1 también lee las Tablas 7 y 8 (la configuración secundaria entró el 14/10/2026), la verificación de esa fecha cubrió las cuatro tablas, con sus nueve columnas.

1. Abre el PDF en la p. 1574 (Tablas 4 y 5).
2. Filtra el CSV: `df = pd.read_csv(ruta, comment="#"); df[df.tabla.isin([4, 5]) & df.metodo.isin(["PCA", "DPCA"])][["tabla", "metodo", "etiqueta_impresa", "valor_impreso"]]`.
3. Compara cada `valor_impreso` con la celda del PDF (fila `etiqueta_impresa`, columna `metodo`). Comprueba también el `bloque` de cada fallo (las líneas horizontales de la Tabla 4).
4. Para los umbrales: en la Fig. 2 (p. 1573, IDV 16) y la Fig. 5 (p. 1576, IDV 19), la línea roja discontinua de los paneles PCA y DPCA debe quedar donde dice `umbral_leido`, con la precisión de `residuo_max_marcas`.
5. Si todo cuadra, escribe la fecha en `configs/yin2012.yaml` (`referencia.transcripcion_verificada: "AAAA-MM-DD"`). Si algo no cuadra, corrige el CSV, anótalo en la columna `nota` y deja constancia en `docs/decisiones.md`.

Una vez fijada la verificación (fila de `docs/decisiones.md` con la huella que imprime `python experiments/V1_verificacion_yin.py --huella`), cualquier cambio en estos dos CSV cambia la huella y V1 vuelve a negarse a ejecutar: la huella resume el yaml de la verificación y el contenido de las dos referencias.
