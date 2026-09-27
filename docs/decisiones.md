# Registro de decisiones

Una linea por decision: fecha, que se decidio, quien y por que. Se incluyen las
que tome el director. Este fichero evita discutir dos veces lo mismo y sirve de
respaldo si algo se cuestiona en la defensa.

| Fecha | Decision | Quien | Motivo |
|---|---|---|---|
| 2026-09-16 | Tres repositorios, uno por estudiante, con `tfmfdd` instalado como dependencia | Equipo | Las instrucciones de UNIR exigen que cada repositorio tenga un unico autor |
| 2026-09-16 | TEP clasico para prototipar y verificar; TEP-Rieth para los resultados definitivos | Pablo | El clasico pesa 26 MB y permite contrastar con Yin (2012); Rieth da intervalos de confianza |
| 2026-09-16 | Los limites de control se calibran con datos normales RESERVADOS, no con los del ajuste | Pablo | Calibrar con los datos del ajuste dio 12,5 % de falsas alarmas frente al 2,3 % con reserva |
| 2026-09-22 | El experimento de calibracion se corrige: mismo ajuste (X_fit) para los dos modelos, solo cambia el limite; la cifra publicable sale de `experiments/A0_calibracion_limites.py` con su configuracion en el CSV | Pablo | La version anterior cambiaba a la vez tamano de ajuste y calibracion (confundido); las cifras 12,5/2,3 (17 comp.) y 19,9/3,6 (27 comp.) son de configuraciones distintas y quedan superadas |
| 2026-09-22 | Ningun numero se reporta sin su configuracion (variables, componentes, alpha, metodo del SPE, n_fit, n_cal); los docstrings apuntan al CSV, no citan cifras | Pablo | Dos corridas con distinta configuracion acabaron citadas como la misma |
| 2026-09-22 | `configs/baseline.yaml` es la unica fuente de los valores por defecto; los scripts lo leen | Pablo | El script A1 usaba 0,90 de varianza mientras el yaml decia 0,85 |
| 2026-09-22 | Metricas nuevas del protocolo: `detected_at_chance` (marca rachas con FDR < 0,10) y FDR estratificada por mitades (`fdr_h1`, `fdr_h2`, `persistence_gap`), calculadas igual en los tres bloques | Pablo, propuesta a validar con el equipo | Fallo 3: racha por azar en la muestra 537 con FDR 3,75 %; fallo 5: FDR 0,29 que promedia 0,95 y 0,07 |
| 2026-09-22 | Contrastes: alpha = 0,05 declarado; correccion de Holm por fallo entre los k-1 pares contra el baseline (analisis principal) y todos contra todos como secundario; Wilcoxon con `zero_method="zsplit"`; se reporta la diferencia con intervalo, no solo el p | Pablo, pendiente de acordar la diferencia minima relevante | Con 100 simulaciones casi todo sale significativo; sin diferencia minima declarada se cae en el mismo error que se denuncia |
| 2026-09-22 | El nivel de confianza del limite de control (0,99) y el alpha de los contrastes (0,05) se nombran distinto en la memoria | Pablo | Mismo simbolo para dos cosas distintas |
| 2026-09-22 | `data.values(columns=YIN_VARS)` y `FAULT_ONSET_TRAIN_CLASSIC = 20` (Russell et al., 2000) | Pablo | La verificacion contra Yin exige sus 33 variables; los dXX.dat traen una hora sana al principio |
| 2026-09-22 | Documentos y cuadernos en espanol neutro, con ustedes | Pablo | Dos companeros de Costa Rica y Colombia |
