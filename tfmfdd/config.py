"""Configuracion del protocolo comun: donde esta baseline.yaml y como se lee.

Por que existe. `docs/decisiones.md` (22/09) fija `configs/baseline.yaml` como la
unica fuente de los valores por defecto de los tres bloques. Hasta la v0.2 ese
fichero solo vivia en el repositorio del paquete: quien instalaba `tfmfdd` con
pip en su propio repositorio no lo recibia, y los scripts lo buscaban con una
ruta relativa al directorio de trabajo y, si no lo encontraban, seguian con los
valores escritos en el codigo sin avisar (REP-4 de la revision del 27/09). Este
modulo da un orden de busqueda explicito y una copia del yaml que viaja dentro
del paquete.

Orden de busqueda (`find_baseline`), del primero al ultimo:

1. La ruta que se pase (`path`). Si no existe, es un error: nunca se cae a la
   siguiente en silencio.
2. La variable de entorno TFMFDD_CONFIG, si esta definida. Si apunta a un
   fichero que no existe, tambien es un error.
3. `configs/baseline.yaml` en el directorio de trabajo: lo que encuentra quien
   ejecuta los scripts desde la raiz del repositorio del paquete. Si no es igual
   que la copia del paquete instalado, `load_baseline` avisa (una copia vieja en
   otro repositorio ganaria sin que nadie lo viera).
4. La copia empaquetada, `tfmfdd/configs/baseline.yaml`: la de la version
   instalada, que es la que reciben Jonathan y Cesar en sus repositorios.

La copia empaquetada es identica a `configs/baseline.yaml` (una prueba lo
exige): si cambias una, cambia la otra. Para dejar constancia de cual se uso,
`load_baseline(with_info=True)` devuelve tambien la ruta, el origen y la huella
SHA-256 del texto (con los finales de linea normalizados a LF, para que la
huella no cambie entre una copia de Windows y una de Linux), y
`tfmfdd.provenance` la escribe junto a cada resultado.

Las claves con valor null del yaml son decisiones pendientes de Pablo. Este
modulo no las sustituye por nada: el script que las necesite debe negarse a
ejecutar mientras sigan en null (ver `require`).

Uso tipico:

    from tfmfdd import config

    cfg = config.load_baseline()                      # el orden de arriba
    cfg, info = config.load_baseline(with_info=True)  # info: path, source, sha256
    alpha_c = config.require(cfg, "contrastes.alpha_contrastes")
"""

from __future__ import annotations

import hashlib
import os
import warnings
from importlib import resources
from pathlib import Path
from typing import NamedTuple

import yaml

#: Variable de entorno con la ruta de un baseline.yaml concreto.
CONFIG_ENV = "TFMFDD_CONFIG"
#: Ruta del yaml relativa al directorio de trabajo (la raiz del repositorio).
BASELINE_RELATIVE = Path("configs") / "baseline.yaml"
#: Origenes posibles, en el orden en que se buscan.
SOURCES = ("explicit", "env", "cwd", "package")


class BaselineLocation(NamedTuple):
    """Donde se encontro el yaml: ruta (texto) y origen (uno de SOURCES)."""

    path: str
    source: str


def packaged_baseline():
    """Recurso de la copia empaquetada (`tfmfdd/configs/baseline.yaml`).

    Devuelve un objeto de `importlib.resources` (con `is_file` y `read_text`),
    que funciona igual con el paquete instalado en una carpeta o dentro de un
    zip.
    """
    return resources.files("tfmfdd").joinpath("configs").joinpath("baseline.yaml")


def _locate(path=None):
    """(recurso con read_text, origen) del yaml que se va a usar."""
    if path is not None:
        p = Path(path)
        if not p.is_file():
            raise FileNotFoundError(
                f"No existe el fichero de configuracion {p}. Se pidio de forma "
                "explicita, asi que no se busca otro."
            )
        return p, "explicit"

    entorno = os.environ.get(CONFIG_ENV)
    if entorno:
        p = Path(entorno)
        if not p.is_file():
            raise FileNotFoundError(
                f"La variable de entorno {CONFIG_ENV} apunta a {p}, que no existe. "
                "Corrigela o borrala para usar configs/baseline.yaml o la copia del paquete."
            )
        return p, "env"

    p = Path.cwd() / BASELINE_RELATIVE
    if p.is_file():
        return p, "cwd"

    empaquetado = packaged_baseline()
    if empaquetado.is_file():
        return empaquetado, "package"

    raise FileNotFoundError(
        "No encuentro baseline.yaml: ni la ruta pedida, ni la variable de entorno "
        f"{CONFIG_ENV}, ni {BASELINE_RELATIVE} en el directorio de trabajo "
        f"({Path.cwd()}), ni la copia del paquete (tfmfdd/configs/baseline.yaml). "
        "Si instalaste tfmfdd con pip y falta la copia, la instalacion no trae los "
        "datos del paquete: reinstala la version etiquetada."
    )


def find_baseline(path=None) -> BaselineLocation:
    """Ruta y origen del baseline.yaml que usara `load_baseline` (orden del modulo)."""
    recurso, origen = _locate(path)
    return BaselineLocation(str(recurso), origen)


def _sha256_texto(texto: str) -> str:
    """SHA-256 del texto con los finales de linea normalizados a LF."""
    return hashlib.sha256(texto.replace("\r\n", "\n").encode("utf-8")).hexdigest()


def _avisar_si_distinto_del_paquete(texto_cwd: str) -> None:
    """Avisa si el configs/baseline.yaml del directorio de trabajo no es el del paquete instalado.

    El orden de busqueda pone el del directorio de trabajo antes que la copia del
    paquete, y es lo correcto en el repositorio del paquete, donde los dos son el
    mismo fichero (una prueba lo exige). Pero en el repositorio de Jonathan o de
    Cesar puede quedar una copia vieja que ya no es la de la version instalada, y
    ganaria sin avisar (V-15 de la revision del 12/10). La procedencia guarda la
    huella del yaml usado, asi que se puede descubrir despues; el aviso lo dice
    antes de calcular.
    """
    empaquetado = packaged_baseline()
    if not empaquetado.is_file():
        return
    propia = _sha256_texto(texto_cwd)
    del_paquete = _sha256_texto(empaquetado.read_text(encoding="utf-8"))
    if propia != del_paquete:
        warnings.warn(
            f"{BASELINE_RELATIVE.as_posix()} del directorio de trabajo no es el de la version "
            f"instalada de tfmfdd (huella {propia[:12]} frente a {del_paquete[:12]} de "
            "tfmfdd/configs/baseline.yaml). Si es una copia vieja, actualizala; si es a "
            f"proposito, pasa su ruta (path= o la variable {CONFIG_ENV}) y no se avisa.",
            UserWarning,
            stacklevel=3,
        )


def load_yaml(path) -> dict:
    """Lee un yaml cualquiera (por ejemplo `configs/yin2012.yaml`) y exige un diccionario.

    No busca en ningun otro sitio: si la ruta no existe, FileNotFoundError.
    """
    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No existe el fichero de configuracion {p}")
    datos = yaml.safe_load(p.read_text(encoding="utf-8"))
    if not isinstance(datos, dict):
        raise ValueError(f"{p} no contiene un diccionario de claves (llego {type(datos).__name__})")
    return datos


def load_baseline(path=None, *, with_info: bool = False):
    """Lee baseline.yaml siguiendo el orden de busqueda del modulo.

    Parametros
    ----------
    path : ruta o None
        Ruta explicita. None aplica el orden: TFMFDD_CONFIG, configs/baseline.yaml
        del directorio de trabajo y la copia del paquete.
    with_info : bool
        Si es True devuelve tambien un dict con `path`, `source` (uno de SOURCES)
        y `sha256` (del texto con finales de linea LF), para la procedencia.

    Devuelve
    --------
    dict con la configuracion, o (dict, info) con with_info=True.
    """
    recurso, origen = _locate(path)
    texto = recurso.read_text(encoding="utf-8")
    cfg = yaml.safe_load(texto)
    if not isinstance(cfg, dict):
        raise ValueError(f"{recurso} no contiene un diccionario de claves (llego {type(cfg).__name__})")
    if origen == "cwd":
        _avisar_si_distinto_del_paquete(texto)
    if with_info:
        return cfg, {"path": str(recurso), "source": origen, "sha256": _sha256_texto(texto)}
    return cfg


def get(cfg: dict, key: str):
    """Valor de una clave con puntos ("pca.variance"). KeyError si falta algun tramo.

    No hay valor por defecto a proposito: una clave que falta es un yaml de otra
    version, y seguir con un valor escrito en el codigo es justo lo que REP-4
    queria evitar.
    """
    valor = cfg
    recorrido = []
    for tramo in key.split("."):
        recorrido.append(tramo)
        if not isinstance(valor, dict) or tramo not in valor:
            raise KeyError(
                f"Falta la clave '{'.'.join(recorrido)}' en la configuracion: el yaml "
                "es de otra version o le falta una seccion."
            )
        valor = valor[tramo]
    return valor


def require(cfg: dict, key: str):
    """Como `get`, pero una clave con valor null es un error que dice que falta decidir.

    Las claves null del yaml son decisiones pendientes de Pablo (registro de
    decisiones). Un script que las necesita no puede sustituirlas por nada.
    """
    valor = get(cfg, key)
    if valor is None:
        raise ValueError(
            f"La clave '{key}' esta en null: es una decision pendiente (registro de "
            "decisiones, docs/decisiones.md). Este calculo no se ejecuta hasta que se fije."
        )
    return valor


def pending(cfg: dict, prefix: str = "") -> list[str]:
    """Claves con valor null (decisiones pendientes), con su ruta con puntos.

    Recorre los diccionarios anidados; una lista o un escalar no null cuentan
    como decididos.
    """
    claves: list[str] = []
    for nombre, valor in cfg.items():
        ruta = f"{prefix}.{nombre}" if prefix else str(nombre)
        if valor is None:
            claves.append(ruta)
        elif isinstance(valor, dict):
            claves.extend(pending(valor, ruta))
    return claves
