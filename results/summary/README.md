Resumenes de resultados. Estos SI se versionan: son la procedencia de cada tabla de la memoria.

## Columnas añadidas el 22-09-2026

`evaluate_run` devuelve ahora, además de `far`, `fdr`, `delay_samples`, `delay_minutes` y `detected`:

- `detected_at_chance`: `True` cuando hay racha de k alarmas pero la FDR queda por debajo de 0,10. Marca una confirmación que no debe leerse como detección (fallo 3 en el TEP clásico). El 0,10 es una convención del protocolo.
- `fdr_h1`, `fdr_h2`, `persistence_gap`: FDR sobre la primera y la segunda mitad del periodo con fallo, y su diferencia. Partición fijada a priori, idéntica para todos los fallos, métodos y bloques. Expone los fallos que se detectan y luego se pierden (fallo 5).

`aggregate` añade `at_chance_ratio` y las medias e intervalos de las tres nuevas columnas.

Los CSV `A0_calibracion_limites*.csv` salen de `experiments/A0_calibracion_limites.py` y llevan la configuración completa junto a cada cifra; son la fuente de la primera fila de la tabla de ablaciones del protocolo. Los ficheros generados antes de esta fecha no tienen las columnas nuevas y hay que regenerarlos.
