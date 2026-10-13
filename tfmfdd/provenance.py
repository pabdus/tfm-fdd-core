"""Procedencia de los resultados: un JSON al lado de cada CSV.

Por que existe. Una cifra sin su configuracion no se puede citar: en este
proyecto ya acabaron citadas como la misma dos corridas con distinto numero de
componentes (`docs/decisiones.md`, 22/09), y la regla 11 de
`docs/reglas_de_medida.md` pide que cada fichero de resultados lleve "la
configuracion, los ids usados y las versiones de las librerias". Hasta la v0.2
cada script lo hacia a su manera (A0 escribia la configuracion en columnas, A1
no escribia nada). Este modulo es la ayuda comun: despues de escribir un CSV,
`write_provenance` deja al lado un JSON con

- la configuracion efectiva del calculo (la que se pase, tal cual);
- los ficheros de configuracion leidos, con su origen y su huella SHA-256;
- los ids usados (ficheros, fallos, simulaciones: lo que diga quien llama);
- los ficheros de datos de entrada, con su huella SHA-256;
- el CSV que acompana, con su huella;
- el commit de git y si habia cambios sin confirmar, leidos SIN escribir nada en
  el repositorio (`git --no-optional-locks` y GIT_OPTIONAL_LOCKS=0: un `git
  status` normal puede reescribir el indice);
- la maquina y las versiones de las bibliotecas (`tfmfdd.cost.machine_info`);
- la fecha y hora (UTC y local), el script y sus argumentos.

El JSON se llama como el CSV con la extension `.provenance.json`
(`A1_pca_baseline_runs.csv` -> `A1_pca_baseline_runs.provenance.json`).

Privacidad: como `machine_info`, no registra el nombre del equipo ni del usuario.
Las rutas de los ficheros de configuracion y los argumentos que son rutas se
escriben relativas al directorio de trabajo (la raiz del repositorio, si los
scripts se ejecutan desde ella); si quedan fuera, con "~" en lugar de la carpeta
personal, la escriba como la escriba (C:\\Users\\x, c:/users/x o /c/Users/x). De
los ficheros de datos se guarda el nombre y la huella, no la ruta.

Un script que escribe varios CSV calcula una vez lo que no cambia entre ellos (el
estado de git y la maquina) con `run_context`, antes de escribir ningun
resultado, y lo pasa a cada `write_provenance`: cada JSON cuesta asi una fraccion
de segundo menos, y todos describen el mismo estado del repositorio.

`committed_status` es la otra mitad: comprueba, tambien sin escribir en el
repositorio, que unas rutas (el registro, la configuracion, el codigo) estan
confirmadas en git tal cual. La usa la regla de orden de
`experiments/V1_verificacion_yin.py`.

Uso tipico:

    from tfmfdd import config, provenance

    cfg, info = config.load_baseline(with_info=True)
    contexto = provenance.run_context(__file__)          # git y maquina, una vez
    ...
    res.to_csv(ruta_csv, index=False)
    provenance.write_provenance(ruta_csv, config={...}, config_files=[info],
                                ids={...}, inputs=[ruta_d00, ...], script=__file__,
                                **contexto)
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

import numpy as np

#: Version del formato del JSON. Si cambian las claves, sube.
SCHEMA = "tfmfdd.provenance/1"
#: Sufijo del JSON de procedencia.
SUFFIX = ".provenance.json"
#: Bibliotecas cuya version se anade a la de machine_info (que no las lista).
_OTRAS_BIBLIOTECAS = (("pyyaml", "PyYAML"), ("pyarrow", "pyarrow"), ("matplotlib", "matplotlib"))
#: Lineas de `git status` que se guardan como mucho.
_MAX_ESTADO = 200


def provenance_path(csv_path) -> Path:
    """Ruta del JSON de procedencia de un CSV: mismo nombre, extension `.provenance.json`."""
    p = Path(csv_path)
    return p.with_name(p.stem + SUFFIX)


def file_sha256(path, *, text: bool = False) -> str:
    """SHA-256 de un fichero.

    Con text=False (por defecto), de los bytes tal cual: es la huella de los
    ficheros de datos (la misma que da `sha256sum`). Con text=True, del texto con
    los finales de linea normalizados a LF, para que un CSV o un yaml de Windows
    (CRLF) y su copia de Linux (LF) den la misma huella.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        if text:
            h.update(f.read().replace(b"\r\n", b"\n"))
        else:
            for bloque in iter(lambda: f.read(1 << 20), b""):
                h.update(bloque)
    return h.hexdigest()


def _sin_casa(texto: str) -> str:
    """Sustituye la carpeta personal por "~" (no dejar el nombre del usuario).

    La reconoce con cualquiera de los dos separadores y, en Windows, sin distinguir
    mayusculas y tambien en la forma de Git Bash: una misma ruta llega como
    C:\\Users\\x, c:/users/x o /c/Users/x segun quien la escriba (V-13 de la
    revision del 12/10). Solo cuando la carpeta acaba ahi (C:\\Users\\x2 no es la de x).
    """
    casa = Path.home().as_posix()
    if len(casa) < 3:
        return texto
    if casa[1:2] == ":":                    # Windows: C:/Users/x, c:\users\x o /c/Users/x
        unidad, cuerpo = casa[0], casa[2:]
        prefijo = rf"(?:{re.escape(unidad)}:|/{re.escape(unidad)})"
    else:
        prefijo, cuerpo = "", casa
    patron = prefijo + re.escape(cuerpo).replace("/", r"[\\/]") + r"(?![^\\/])"
    return re.sub(patron, "~", texto, flags=re.IGNORECASE if os.name == "nt" else 0)


def _ruta_legible(ruta) -> str:
    """Ruta relativa al directorio de trabajo (la raiz del repositorio, si los scripts se
    ejecutan desde ella) con "/"; si queda fuera, la absoluta con "~" en vez de la
    carpeta personal. Asi el JSON dice "configs/baseline.yaml" y no la ruta del equipo."""
    p = Path(ruta)
    try:
        return p.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (ValueError, OSError):
        return _sin_casa(p.as_posix())


def _argumento_legible(argumento: str) -> str:
    """Un argumento de la linea de ordenes; si es una ruta que existe, como `_ruta_legible`."""
    try:
        existe = Path(argumento).exists()
    except (OSError, ValueError):
        existe = False
    return _ruta_legible(argumento) if existe else argumento


def _serializable(obj):
    """Convierte numpy, Path, tuplas, conjuntos y NaN en tipos de JSON (NaN -> None)."""
    if isinstance(obj, dict):
        return {str(k): _serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set, frozenset)):
        return [_serializable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_serializable(v) for v in obj.tolist()]
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        valor = float(obj)
        return valor if math.isfinite(valor) else None
    if isinstance(obj, Path):
        return _sin_casa(str(obj))
    if isinstance(obj, str):
        return _sin_casa(obj)
    if obj is None or isinstance(obj, (bool, int)):
        return obj
    return _sin_casa(str(obj))


def _entorno_git() -> dict:
    """Entorno para git sin variables GIT_* heredadas y sin bloqueos opcionales.

    Una GIT_DIR, GIT_WORK_TREE o GIT_INDEX_FILE heredada haria que git mirara otro
    repositorio u otro indice que el de la ruta (V-05). GIT_OPTIONAL_LOCKS=0 evita,
    como --no-optional-locks, que `git status` reescriba el indice.
    """
    entorno = {k: v for k, v in os.environ.items() if not k.upper().startswith("GIT_")}
    entorno["GIT_OPTIONAL_LOCKS"] = "0"
    return entorno


def _git(cwd, *args) -> subprocess.CompletedProcess:
    """Una orden de git de solo lectura en `cwd`.

    Siempre con --no-optional-locks (no escribe el indice) y --literal-pathspecs (una
    ruta con corchetes o asteriscos se toma tal cual, no como patron). Puede lanzar
    OSError si git no esta instalado.
    """
    return subprocess.run(["git", "--no-optional-locks", "--literal-pathspecs", *args], cwd=cwd,
                          capture_output=True, text=True, encoding="utf-8", errors="replace",
                          timeout=30, env=_entorno_git())


def git_info(path=None) -> dict:
    """Commit y estado del repositorio que contiene `path`, sin escribir nada en el.

    Usa solo ordenes de lectura (`rev-parse`, `status --porcelain`, `tag
    --points-at`) con `--no-optional-locks`, GIT_OPTIONAL_LOCKS=0 y sin variables
    GIT_* heredadas (`_git`), para que `git status` no reescriba el indice. Si git no
    esta instalado o `path` no esta en un repositorio, lo dice en el resultado y no
    falla: la procedencia nunca debe romper un calculo.

    Devuelve
    --------
    dict con `available` (git instalado), `repository` (path esta en un
    repositorio), `commit`, `tags` (etiquetas que apuntan al commit), `dirty`
    (hay cambios sin confirmar en ficheros versionados), `untracked` (numero de
    ficheros sin versionar) y `status` (las lineas de `git status --porcelain`,
    como mucho 200).
    """
    base = Path(path) if path is not None else Path.cwd()
    base = base.resolve()
    if base.is_file():
        base = base.parent
    resultado = {"available": False, "repository": False, "commit": None, "tags": [],
                 "dirty": None, "untracked": None, "status": []}

    def git(*args):
        r = _git(base, *args)
        return r.stdout if r.returncode == 0 else None

    try:
        raiz = git("rev-parse", "--show-toplevel")
    except (OSError, subprocess.SubprocessError):
        return resultado
    resultado["available"] = True
    if raiz is None:
        return resultado
    resultado["repository"] = True
    try:
        commit = git("rev-parse", "HEAD")
        resultado["commit"] = commit.strip() if commit else None
        etiquetas = git("tag", "--points-at", "HEAD") if commit else None
        resultado["tags"] = etiquetas.split() if etiquetas else []
        estado = git("status", "--porcelain")
    except (OSError, subprocess.SubprocessError):
        return resultado
    if estado is not None:
        lineas = [l for l in estado.splitlines() if l.strip()]
        resultado["dirty"] = any(not l.startswith("??") for l in lineas)
        resultado["untracked"] = sum(l.startswith("??") for l in lineas)
        resultado["status"] = lineas[:_MAX_ESTADO]
    return resultado


class GitCheckError(RuntimeError):
    """No se puede comprobar en git que una ruta esta confirmada: git no esta, la ruta no
    existe, esta fuera de un repositorio o dentro de una carpeta .git."""


#: Opciones de la cabecera de un .pyc (PEP 552) que dicen "hash, sin comprobar el fuente".
_PYC_SIN_COMPROBAR = 0b01


def _pyc_sin_comprobar(pyc: Path) -> bool:
    """Si un .pyc es "unchecked-hash" (PEP 552): Python lo usa sin comparar con el fuente.

    La cabecera son 16 bytes: la marca magica (4) y las opciones (4, little endian):
    0 = por marca de tiempo; 3 = por hash, comprobando el fuente; 1 = por hash, SIN
    comprobarlo. Solo el ultimo deja ejecutar un codigo distinto del confirmado con el
    fuente intacto en el disco.
    """
    try:
        with open(pyc, "rb") as f:
            cabecera = f.read(8)
    except OSError:
        return False
    return len(cabecera) == 8 and int.from_bytes(cabecera[4:8], "little") == _PYC_SIN_COMPROBAR


def _problemas_bytecode(raiz: Path, pycs) -> list[str]:
    """El bytecode que haria ejecutar otra cosa que el fuente confirmado (W-03 de la
    segunda revision): un .pyc "unchecked-hash" en __pycache__, o un .pyc suelto sin su
    .py al lado (Python lo importa en lugar de un modulo que no existe en el fuente).

    Limite conocido: un .pyc por marca de tiempo con la fecha y el tamano del fuente
    falsificados tampoco se compara con el fuente. Para cerrar eso habria que ejecutar
    sin cache de bytecode (por ejemplo, PYTHONPYCACHEPREFIX a una carpeta vacia).
    """
    problemas = []
    for pyc in sorted(pycs):
        relpyc = pyc.resolve().relative_to(raiz).as_posix()
        if pyc.parent.name == "__pycache__":
            if _pyc_sin_comprobar(pyc):
                problemas.append(f"{relpyc}: bytecode 'unchecked-hash' (PEP 552) que Python usa sin "
                                 "comparar con el fuente confirmado; borralo")
        elif not pyc.with_suffix(".py").exists():
            problemas.append(f"{relpyc}: bytecode sin su .py al lado, que Python importaria como "
                             "modulo sin fuente confirmado; borralo")
    return problemas


def _problemas_fichero(raiz: Path, rel: str) -> list[str]:
    """Lo que impide decir que un fichero esta confirmado tal cual (vacio si lo esta).

    Si es un .py, tambien su bytecode en __pycache__ (`_problemas_bytecode`).
    """
    problemas = []
    marca = _git(raiz, "ls-files", "-v", "--error-unmatch", "--", rel)
    lineas = [l for l in marca.stdout.splitlines() if l.strip()]
    if marca.returncode != 0 or not lineas:
        problemas.append(f"{rel}: sin versionar")
    elif not lineas[0].startswith("H "):
        # S = skip-worktree; minuscula = assume-unchanged: git no veria sus cambios.
        problemas.append(f"{rel}: marcado {lineas[0][:1]!r} en el indice (skip-worktree o "
                         "assume-unchanged): git no veria sus cambios")
    estado = _git(raiz, "status", "--porcelain", "--untracked-files=all", "--ignored=matching", "--", rel)
    if estado.returncode != 0:
        problemas.append(f"{rel}: git status fallo ({estado.stderr.strip()})")
    problemas += [l for l in estado.stdout.splitlines() if l.strip()]
    fichero = raiz / rel
    if fichero.suffix == ".py":
        problemas += _problemas_bytecode(raiz, (fichero.parent / "__pycache__").glob(f"{fichero.stem}.*.pyc"))
    return problemas


def _problemas_carpeta(raiz: Path, carpeta: Path, rel: str) -> list[str]:
    """Lo mismo para una carpeta de codigo: todo versionado, sin marcas ni cambios, y
    ningun .py en el disco que git no versione (aunque este ignorado). Lo que git ignora
    y no es un .py (por ejemplo __pycache__) no cuenta, salvo el bytecode que haria
    ejecutar otra cosa que el fuente (`_problemas_bytecode`)."""
    problemas = []
    seguidos = _git(raiz, "ls-files", "-v", "--", rel)
    lineas = [l for l in seguidos.stdout.splitlines() if l.strip()]
    if seguidos.returncode != 0 or not lineas:
        return [f"{rel}/: ningun fichero versionado"]
    problemas += [f"{l[2:]}: marcado {l[:1]!r} en el indice (skip-worktree o assume-unchanged)"
                  for l in lineas if not l.startswith("H ")]
    estado = _git(raiz, "status", "--porcelain", "--untracked-files=all", "--", rel)
    if estado.returncode != 0:
        problemas.append(f"{rel}/: git status fallo ({estado.stderr.strip()})")
    lineas_estado = [l for l in estado.stdout.splitlines() if l.strip()]
    problemas += lineas_estado
    versionados = {l[2:] for l in lineas}
    for py in sorted(carpeta.rglob("*.py")):
        if "__pycache__" in py.parts:
            continue
        relpy = py.resolve().relative_to(raiz).as_posix()
        if relpy not in versionados and not any(relpy in l for l in lineas_estado):
            problemas.append(f"{relpy}: modulo en el disco que git no versiona (ignorado)")
    problemas += _problemas_bytecode(raiz, carpeta.rglob("*.pyc"))
    return problemas


def committed_status(paths) -> dict:
    """Comprueba, sin escribir en el repositorio, que unas rutas estan confirmadas en git tal cual.

    Por que existe. La regla de orden de V1 (y la de cualquier script confirmatorio)
    exige que el registro, la configuracion y el codigo que se ejecuta esten
    confirmados antes de abrir los datos de prueba. Un unico `git status -- <rutas>`
    no basta: no dice nada de un fichero dentro de .git, de uno de un repositorio
    anidado ni de uno marcado con skip-worktree o assume-unchanged (V-05 de la
    revision del 12/10). Aqui cada ruta se comprueba por separado:

    - un fichero: no esta dentro de una carpeta .git, esta en un repositorio, esta
      versionado sin marcas que oculten sus cambios y `git status` no dice nada de el
      (ni que esta modificado, ni que esta ignorado);
    - una carpeta (el codigo de un paquete): todo lo que contiene esta versionado,
      sin marcas y sin cambios, sin contar lo que git ignora (por ejemplo
      __pycache__); pero cualquier .py que haya en el disco tiene que estar
      versionado aunque git lo ignore, porque un modulo importable que git no ve
      seria codigo sin confirmar;
    - en los dos casos, ningun bytecode que Python ejecutaria en lugar del fuente
      confirmado: un .pyc "unchecked-hash" en __pycache__ o un .pyc sin su .py
      (`_problemas_bytecode`; W-03 de la segunda revision).

    Solo usa ordenes de lectura (`_git`). Devuelve un dict con `roots` (las raices de
    los repositorios en que estan, ordenadas), `commits` ({raiz: commit}) y
    `problems` (una linea por problema; vacia si todo esta confirmado). Quien llama
    decide si exige un solo repositorio. Lanza GitCheckError si git no esta o una
    ruta no se puede comprobar.
    """
    raices: dict = {}
    problemas: list[str] = []
    for ruta in paths:
        r = Path(ruta).resolve()
        legible = _sin_casa(str(r))
        if any(parte.lower() == ".git" for parte in r.parts):
            raise GitCheckError(f"{legible} esta dentro de una carpeta .git: no es un fichero versionado")
        if not r.exists():
            raise GitCheckError(f"{legible} no existe")
        carpeta = r if r.is_dir() else r.parent
        try:
            top = _git(carpeta, "rev-parse", "--show-toplevel")
        except (OSError, subprocess.SubprocessError) as exc:
            raise GitCheckError(f"No se puede usar git ({exc})") from None
        if top.returncode != 0:
            raise GitCheckError(f"{legible} no esta en un repositorio git")
        raiz = Path(top.stdout.strip()).resolve()
        try:
            rel = r.relative_to(raiz).as_posix()
        except ValueError:
            raise GitCheckError(f"{legible} no esta dentro de la raiz de su repositorio") from None
        if raiz not in raices:
            head = _git(raiz, "rev-parse", "HEAD")
            raices[raiz] = head.stdout.strip() if head.returncode == 0 else None
        if r.is_dir():
            problemas += _problemas_carpeta(raiz, r, rel)
        else:
            problemas += _problemas_fichero(raiz, rel)
    return {"roots": sorted(str(x) for x in raices), "commits": {str(k): v for k, v in raices.items()},
            "problems": problemas}


def _versiones_extra() -> dict:
    versiones = {}
    for clave, distribucion in _OTRAS_BIBLIOTECAS:
        try:
            versiones[f"{clave}_version"] = metadata.version(distribucion)
        except metadata.PackageNotFoundError:
            versiones[f"{clave}_version"] = None
    return versiones


def _maquina() -> dict:
    """machine_info del paquete, sin dejar que un fallo al leer la maquina rompa nada."""
    try:
        from .cost import machine_info

        return machine_info()
    except Exception as exc:  # noqa: BLE001 - la procedencia nunca rompe el calculo
        return {"error": f"{type(exc).__name__}: {exc}"}


def run_context(script=None, *, machine: bool = True, git_path=None) -> dict:
    """El estado de git y la maquina, calculados una vez para todos los JSON de una ejecucion.

    Por que existe. Cada `write_provenance` lanzaba cuatro ordenes de git y volvia a
    leer la maquina: medio segundo por JSON, y A0 y A1 escriben tres (V-14 de la
    revision del 12/10). Llamala una vez, antes de escribir ningun resultado (asi el
    estado de git es el del codigo, no el de los CSV que se acaban de escribir), y
    pasa lo que devuelve a cada `write_provenance` con `**contexto`.

    Devuelve {"git": git_info(...), "machine": machine_info() o False}.
    """
    donde = git_path if git_path is not None else (Path(script).parent if script else None)
    return {"git": git_info(donde), "machine": _maquina() if machine else False}


def build_provenance(*, config: dict, config_files=None, ids=None, inputs=None,
                     script=None, argv=None, extra=None, machine=True,
                     git_path=None, git: dict | None = None) -> dict:
    """Diccionario de procedencia (lo que `write_provenance` escribe, sin el CSV).

    Parametros
    ----------
    config : dict
        Configuracion efectiva del calculo (variables, componentes, nivel, metodo
        del limite, n_fit, n_cal, k...). Obligatoria: es lo que hace citable la
        cifra.
    config_files : lista de dict o None
        Ficheros de configuracion leidos, por ejemplo el `info` de
        `config.load_baseline(with_info=True)` (claves path, source, sha256).
    ids : dict o None
        Ids usados: ficheros, fallos, simulaciones, filas de ajuste y calibracion.
    inputs : lista de rutas o None
        Ficheros de datos de entrada; se guarda su nombre, su tamano y su SHA-256
        (de los bytes). En Rieth, para no leer 1,3 GB, pasa las huellas de
        SHA256SUMS.txt en `extra` en lugar de las rutas.
    script : ruta o None
        El script que produce el resultado (normalmente `__file__`). El commit se
        busca desde su carpeta; sin script, desde el directorio de trabajo.
    argv : lista o None
        Argumentos de la linea de ordenes (por defecto, sys.argv[1:]).
    extra : dict o None
        Cualquier otra cosa que deba viajar con el resultado.
    machine : bool o dict
        True: incluir `cost.machine_info()` (tarda un cuarto de segundo en medir el
        tic del reloj de CPU); False: no incluirla; un dict: la ya calculada (por
        ejemplo, la de `run_context`).
    git_path : ruta o None
        Donde buscar el repositorio, si no es la carpeta del script.
    git : dict o None
        El `git_info` ya calculado (por ejemplo, el de `run_context`); None lo
        calcula ahora.
    """
    if not isinstance(config, dict):
        raise TypeError("config debe ser un dict con la configuracion efectiva del calculo")
    ahora = datetime.now(timezone.utc)
    entradas = []
    for ruta in inputs or []:
        p = Path(ruta)
        entradas.append({"name": p.name, "bytes": p.stat().st_size, "sha256": file_sha256(p)})

    donde_git = git_path if git_path is not None else (Path(script).parent if script else None)
    if isinstance(machine, dict):
        maquina = machine
    else:
        maquina = _maquina() if machine else None
    ficheros_cfg = []
    for f in config_files or []:
        f = dict(f)
        if f.get("path") is not None:
            f["path"] = _ruta_legible(f["path"])
        ficheros_cfg.append(f)
    datos = {
        "schema": SCHEMA,
        "created_utc": ahora.isoformat(timespec="seconds"),
        "created_local": ahora.astimezone().isoformat(timespec="seconds"),
        "script": Path(script).name if script else None,
        "argv": [_argumento_legible(str(a)) for a in (sys.argv[1:] if argv is None else argv)],
        "config": config,
        "config_files": ficheros_cfg,
        "ids": ids or {},
        "inputs": entradas,
        "git": git if git is not None else git_info(donde_git),
        "versions": _versiones_extra(),
        "machine": maquina,
        "extra": extra or {},
    }
    return _serializable(datos)


def write_provenance(csv_path, *, config: dict, config_files=None, ids=None, inputs=None,
                     script=None, argv=None, extra=None, machine=True,
                     git_path=None, git: dict | None = None) -> Path:
    """Escribe el JSON de procedencia al lado de un CSV ya escrito y devuelve su ruta.

    El CSV tiene que existir: su nombre y su huella (SHA-256 del texto con
    finales de linea LF) entran en el JSON, de modo que el JSON identifica el
    fichero exacto que describe. El resto de parametros, como en
    `build_provenance` (`machine` y `git` pueden venir de `run_context`).
    """
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"No existe {csv_path}: escribe primero el CSV y despues su procedencia")
    datos = build_provenance(config=config, config_files=config_files, ids=ids, inputs=inputs,
                             script=script, argv=argv, extra=extra, machine=machine,
                             git_path=git_path, git=git)
    datos = {"csv": csv_path.name, "csv_sha256": file_sha256(csv_path, text=True), **datos}
    destino = provenance_path(csv_path)
    texto = json.dumps(datos, indent=2, ensure_ascii=False, allow_nan=False)
    destino.write_text(texto + "\n", encoding="utf-8", newline="\n")
    return destino


def read_provenance(csv_path) -> dict:
    """Lee el JSON de procedencia de un CSV (o el JSON directamente)."""
    p = Path(csv_path)
    destino = p if p.name.endswith(SUFFIX) else provenance_path(p)
    return json.loads(destino.read_text(encoding="utf-8"))
