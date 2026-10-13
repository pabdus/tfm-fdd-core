# El entorno de Python en Windows

Por qué existe este documento: en el PC de Pablo, el entorno virtual del repositorio (`.venv`) no podía cargar varias bibliotecas mientras estuvo dentro de la carpeta OneDrive, y las pruebas de la v0.3 se ejecutaron con el Python global. Aquí están los hechos comprobados, cómo saber si a Jonathan o a Cesar les pasa lo mismo y las opciones que hay. No recomienda cambiar la seguridad de Windows: eso lo decide Pablo.

## Hechos comprobados

Comprobados por la sesión principal de la revisión del código el 11/10/2026:

- El **Control inteligente de aplicaciones** de Windows (Smart App Control) está activado en modo de aplicación: en el registro, `HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy`, valor `VerifiedAndReputablePolicyState = 1`.
- El registro de eventos `Microsoft-Windows-CodeIntegrity/Operational` muestra bloqueos (eventos 3033 y 3077) de:
  - en el `.venv` del repositorio, que está dentro de OneDrive: `pandas\_libs\json...pyd`, `scipy\stats\_sobol...pyd`, `pyarrow\parquet.dll` y `torch\lib\torch_cpu.dll`;
  - en el Python global: `torch\lib\c10.dll`.
- El **Python global** (3.12, con numpy 2.4.6, pandas 3.0.3, pyarrow 25.0.1, scipy 1.18.0, scikit-learn 1.9.0, psutil 7.2.2 y pytest 8.4.2) carga todo menos torch.
- **WSL2 con Ubuntu 26.04** está instalado, con Python 3.14.4 y `venv` disponible, pero sin pip ni numpy en el sistema.
- Las sesiones de la revisión no cambian la configuración de Windows: eso lo decide Pablo.

Comprobado el 12/10/2026 durante la preparación de la v0.3:

- Con el Python global, la suite de pruebas de la v0.3 pasa entera (`README.md`, "Pruebas"). Las versiones exactas de ese entorno están en `requirements-probado.txt`. pyreadr no está instalado en él: solo hace falta para convertir los `.RData` de Rieth (`tfmfdd.convert_rieth`).
- El `.venv` del repositorio (creado con ese mismo Python 3.12.0) tiene **otras versiones**: numpy 2.5.3, pandas 3.0.6, scipy 1.18.1, scikit-learn 1.9.1, matplotlib 3.11.2, pytest 9.1.1 y torch 2.14.0+cpu, entre otras (leído en los nombres de sus carpetas `.dist-info`, sin ejecutar nada).
- `wsl -l -v` lista la distribución `Ubuntu-26.04` con la versión 2 de WSL.

Comprobado el 13/10/2026, después de sacar Documentos de la carpeta OneDrive:

- OneDrive no está instalado ni tiene cuenta: `C:\Users\pabdu\OneDrive` era una carpeta local, resto de una copia de seguridad antigua, y Windows guardaba ahí Documentos, Escritorio e Imágenes. Pablo devolvió las tres a `C:\Users\pabdu\` y el repositorio está ahora en `C:\Users\pabdu\Documents\...`.
- El mismo `.venv` del repositorio, ya fuera de esa carpeta, carga pandas, scikit-learn, el Parquet de pyarrow y torch 2.14.0+cpu. Confirma que el bloqueo de esas bibliotecas dependía de la ubicación.
- En el Python global, torch 2.11.0+cu128 también carga. El 11/10 su `c10.dll` estaba bloqueada, así que el veredicto de Windows sobre un mismo fichero puede cambiar con el tiempo: conviene comprobarlo de nuevo antes de fiarse de un entorno.
- Se volvió a crear el entorno de la opción 2 en `C:\Users\pabdu\venvs\tfm` (Python 3.12.0, `requirements-probado.txt` y el paquete en modo editable, como indica ese fichero), y la suite completa da `643 passed`.

## Qué significa para el proyecto

- **Dos entornos con versiones distintas.** La memoria promete que los tres bloques usarán "la misma versión fija del paquete común" (párrafo 367). Para que eso fije también las bibliotecas, la referencia es `requirements-probado.txt`: las versiones con que pasaron las pruebas. Un resultado calculado en el `.venv` no se ejecutó con esas versiones.
- **torch no cargaba en ningún Python de este PC hasta el 12/10**, y por eso el cuaderno C00 de la v0.3 se actualizó sin ejecutarse. Desde el 13/10 carga (arriba), así que el C00 ya se puede volver a ejecutar aquí.
- **Si pyarrow no puede leer Parquet**, `data.load_rieth` lanza un error que lo explica y remite a la sección "pyarrow y Windows" del LEEME de los datos de Rieth.

## Cómo saber si les pasa a ustedes

1. Con el Python que vayan a usar:

   ```
   python -c "import numpy, pandas, scipy.stats, sklearn, pyarrow.parquet, matplotlib, yaml, psutil; print('ok')"
   ```

   Cesar, además: `python -c "import torch; print(torch.__version__)"`.
2. Si sale un error del tipo `DLL load failed while importing ...` que nombra una directiva de Control de aplicaciones, es Windows bloqueando una biblioteca compilada: no es un fallo de los datos ni de `tfmfdd`. **No cambien la configuración de seguridad por su cuenta**: avísenle a Pablo con el mensaje completo.
3. Para ver si el Control inteligente de aplicaciones está activado, sin cambiar nada, en una consola de Windows:

   ```
   reg query HKLM\SYSTEM\CurrentControlSet\Control\CI\Policy /v VerifiedAndReputablePolicyState
   ```

   En el PC de Pablo da 1, con el control en modo de aplicación.

## Opciones

En este PC se han probado de principio a fin la primera y la segunda. Las demás dicen qué hay que comprobar antes de fiarse de ellas.

1. **El Python global, fuera de OneDrive.** Es con el que pasaron las pruebas de la v0.3. Carga todo; su torch (2.11.0+cu128) estaba bloqueado el 11/10 y carga el 13/10. Inconvenientes: es un entorno compartido por todo lo que se instale en el equipo, y sus versiones pueden moverse con cualquier `pip install`. Para dejarlo con las versiones probadas: `python -m pip install -r requirements-probado.txt` y `python -m pip install -e . --no-deps` desde la raíz del repositorio.
2. **Un entorno virtual fuera de OneDrive** (`python -m venv C:\Users\<usuario>\venvs\tfm`, y después lo mismo que en la opción 1 con ese Python). **Comprobado el 12/10/2026 en el PC de Pablo**, con su autorización: un entorno nuevo en `C:\Users\pabdu\venvs\tfm` (Python 3.12.0), con `requirements-probado.txt` instalado con pip y la rueda de la v0.3, carga numpy, pandas, scipy, scikit-learn y el Parquet de pyarrow (la línea de importación de "Cómo saber si les pasa a ustedes" da "ok"); `load_rieth(0, "test", runs=[1, 2])` da `(1920, 55)`; `pip check` no encuentra problemas; y la suite completa, ejecutada desde el repositorio con ese Python, da `643 passed`. Antes, en solo lectura, se comparó el SHA-256 de `pyarrow\parquet.dll`, `pyarrow\arrow.dll` y `pyarrow\_parquet...pyd` del `.venv` del repositorio y del Python global: son idénticos, y ninguno lleva el flujo `Zone.Identifier`. Los mismos bytes se bloquean dentro de OneDrive y cargan fuera: el bloqueo depende de la ubicación, no de la versión. torch no se probó (no es dependencia del paquete, y el 12/10 su `c10.dll` se bloqueaba también en el Python global). El 13/10 se volvió a crear, esta vez con el paquete en modo editable, y volvió a dar `643 passed`. **Es la opción recomendada** para trabajar en Windows.
3. **WSL2 (Ubuntu 26.04, ya instalado).** Es Linux: el Control inteligente de aplicaciones es una directiva de Windows, y lo esperable es que no actúe sobre las bibliotecas de Linux que se ejecutan dentro de WSL2. **No está comprobado aquí.** Lo que hay que mirar antes:
   - Ubuntu 26.04 trae Python 3.14.4. El paquete admite Python 3.10 o posterior, pero las pruebas solo se han ejecutado con 3.12, y hay que comprobar que existen versiones de las dependencias (y de torch, para Cesar) para 3.14.
   - El sistema no tiene pip ni numpy: hay que crear un entorno virtual (`python3 -m venv ~/venvs/tfm`), comprobar que trae pip e instalar ahí `requirements-probado.txt`, o las versiones que se fijen para ese entorno.
   - Conviene clonar el repositorio y copiar los datos dentro del sistema de ficheros de Linux (por ejemplo, en `~/`), no trabajar sobre `/mnt/c/...` ni sobre OneDrive: el acceso a los discos de Windows desde WSL suele ser bastante más lento, y el entorno y los datos no deben estar en OneDrive ("Qué no hacer"). `TFM_RIETH_DIR` se define igual, con la ruta de Linux.
   - Sería un tercer entorno, con sus propias versiones: los resultados que se citen tienen que decir con cuál salieron (el `.provenance.json` lo registra).
   - En Linux, `tfmfdd.cost` mide el pico de memoria con `VmHWM` y no con `ru_maxrss`, que se hereda a través de exec (arreglado el 12/10; antes, la inferencia medida en un proceso aparte nunca salía exacta en Linux). Fuera de este PC, una revisión independiente ejecutó la suite en Linux (Python 3.13) el 12/10: con ese arreglo, `638 passed` (`CAMBIOS.md`, "Pruebas"). macOS queda sin comprobar.
4. **Desactivar el Control inteligente de aplicaciones.** Es una protección de seguridad de Windows y la decisión es solo de Pablo; este documento no la recomienda. Algo que habría que comprobar antes, en la documentación de Microsoft para su versión de Windows: en algunas versiones, una vez desactivado no se puede volver a activar sin reinstalar o restablecer el sistema. Aquí no se afirma ni se descarta: es una pregunta a resolver antes de decidir.

## Qué no hacer

- No cambiar la configuración de seguridad de Windows por cuenta propia ni añadir excepciones para que una biblioteca cargue.
- No guardar el entorno virtual ni los datos dentro de OneDrive o de Google Drive.
- No mezclar resultados de entornos distintos sin decir de cuál sale cada uno.

## Comprobación mínima de un entorno nuevo

1. La línea de importación del apartado "Cómo saber si les pasa a ustedes".
2. `python -c "import tfmfdd; print(tfmfdd.__version__)"`, que debe dar `0.3.0`.
3. `python -m pytest -q` desde la raíz del repositorio, con `TFM_CLASSIC_DIR` y `TFM_RIETH_DIR` definidas, y comparar el resultado con el del `README.md`.
4. Guardar `tfmfdd.cost.machine_info()` (sistema, CPU, versiones e hilos) junto a los primeros resultados.
