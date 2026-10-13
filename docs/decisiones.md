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
  dice que opciones ofrece el codigo y de donde sale la pregunta.
- `configs/yin2012.yaml` (la verificacion): 11 claves en null el 2026-10-12.
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
