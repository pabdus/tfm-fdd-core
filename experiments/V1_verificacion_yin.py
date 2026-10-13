"""V1 - Verificacion de la implementacion frente a Yin et al. (2012).

Por que existe. La memoria (parrafos 355 y 369) promete comprobar sobre el TEP
clasico que el codigo, ejecutado con la configuracion de Yin et al. (2012),
reproduce la FDR por fallo y la FAR en operacion normal que ellos publican
(Tablas 4 y 5, p. 1574), "dentro de una tolerancia fijada antes de ejecutar",
"sin ajustes para que coincidan y sin contar los fallos 3, 9 y 15". Lo que se
verifica es la IMPLEMENTACION (escalado, PCA y DPCA, limites, regla de alarma,
metricas), no las reglas del protocolo: Yin calibran los limites en muestra, y
el protocolo del TFM, con datos reservados. Si la implementacion reproduce sus
cifras, las ablaciones sobre Rieth se construyen sobre un codigo verificado.

Que hace, con la configuracion de configs/yin2012.yaml (la especificacion esta
en docs/verificacion_yin.md: la configuracion de Yin, los supuestos S1 a S12 con
su evidencia, la salida y las opciones de tolerancia):

1. Ajusta PCA (9 componentes) y DPCA (17 componentes, h = 2) con las 33
   variables de Yin y las 500 filas de d00.dat, en muestra a proposito
   (`calibration="in_sample"`), con el T2 por la F de Fase II (ec. 3) y el SPE
   por Jackson y Mudholkar (ec. 2).
2. Forma la regla de alarma de Yin, "T2 o SPE", con el indice combinado del
   monitor (`PCAMonitor.combined_index`, limite 1).
3. Mide la FDR de cada dXX_te.dat sobre las muestras 161 a 960 y la FAR de
   d00_te.dat en la ventana declarada en el yaml (`far_window` de
   `metrics.evaluate_run`), con las alarmas y las muestras evaluadas al lado.
4. Lee las cifras de Yin de configs/referencias/yin2012_tablas_transcripcion.csv
   (no estan escritas en ningun sitio del codigo), calcula la diferencia por
   fallo en puntos y en muestras, y aplica la tolerancia del yaml a los 18
   fallos de los bloques 1 y 2: el veredicto no cuenta 3, 9 y 15, que salen en
   la tabla igual.
5. Repite lo mismo con las variantes declaradas (filas aparte, sin veredicto) y,
   si se decide, con la configuracion secundaria (Tablas 7 y 8).
6. Escribe results/summary/V1_verificacion_yin.csv (una fila por configuracion,
   metodo, variante y fallo, con la configuracion en cada fila) y
   V1_verificacion_yin_resumen.csv (los criterios y el veredicto), cada uno con
   su JSON de procedencia (tfmfdd.provenance: configuracion, huella del
   registro, ficheros con su SHA-256, commit, versiones).

Regla de orden: este script NO abre ningun fichero de prueba (dXX_te.dat)
mientras la tolerancia no este fijada. Lo comprueba asi, antes de leer nada:
- ninguna clave del yaml en null (las decisiones pendientes);
- una fila fechada de docs/decisiones.md, el registro de decisiones del
  repositorio (una fecha que exista y no sea posterior a hoy), con la etiqueta
  [V1-yin huella=<12 hex>], donde la huella resume los valores del yaml y el
  contenido de las dos referencias: si cambia cualquiera despues de fijarla, la
  huella deja de coincidir. Cuenta solo una fila visible de la tabla: no una dentro
  de un comentario <!-- --> ni de un bloque de codigo (vallas de CommonMark). Una
  fila en otro fichero (--decisiones) solo vale si se pide con
  --permitir-otro-registro, y la procedencia dice que fichero fue;
- el yaml, decisiones.md, las referencias y el CODIGO que se ejecuta (este script
  y la carpeta del paquete tfmfdd que se importa), confirmados en git en un mismo
  repositorio: versionados, sin cambios, sin marcas skip-worktree ni
  assume-unchanged, fuera de .git y sin bytecode que Python ejecute en lugar del
  fuente (un .pyc "unchecked-hash"; `provenance.committed_status`, que solo lee).
  Asi tampoco se puede cambiar el codigo despues de ver un resultado y repetir
  (parrafo 369: "sin ajustes para que coincidan").
Solo con esas tres cosas se crea el permiso que exige la funcion que carga los
ficheros de prueba. `--solo-limites` usa solo d00.dat (el ajuste) y no lo
necesita; `--huella` imprime la huella para la fila de decisiones.md (los pasos
estan en docs/decisiones.md).

Uso:
    python experiments/V1_verificacion_yin.py --solo-limites   # limites con d00.dat
    python experiments/V1_verificacion_yin.py --huella         # huella para decisiones.md
    python experiments/V1_verificacion_yin.py                  # la verificacion
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import re
import sys
import warnings
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sps

import tfmfdd
from tfmfdd import config as tconfig
from tfmfdd import data, limits, metrics, provenance
from tfmfdd.pca import PCAMonitor
from tfmfdd.preprocessing import Scaler

METODOS = ("PCA", "DPCA")
#: Etiqueta de la fila de docs/decisiones.md que fija la verificacion.
ETIQUETA = "V1-yin"
PATRON_FILA = re.compile(r"\[V1-yin huella=([0-9a-f]{12})\]")
PATRON_FECHA = re.compile(r"^\d{4}-\d{2}-\d{2}$")
#: Vallas de los bloques de codigo de markdown (CommonMark): apertura y cierre.
PATRON_VALLA = re.compile(r"^ {0,3}(`{3,}|~{3,})")
PATRON_CIERRE = re.compile(r"^ {0,3}(`{3,}|~{3,})[ \t]*$")
#: El registro de decisiones del repositorio del paquete comun (memoria, parrafo 369),
#: relativo a la raiz del repositorio.
REGISTRO = "docs/decisiones.md"
#: Muestras sobre las que Yin calculan sus tasas (rejilla k/800; docs/verificacion_yin.md, S9 y S10).
N_REJILLA_YIN = 800
#: Nombres de los ficheros de salida.
SALIDA = "V1_verificacion_yin.csv"
SALIDA_RESUMEN = "V1_verificacion_yin_resumen.csv"
SALIDA_LIMITES = "V1_limites_d00.csv"

#: Valores admitidos en las claves con opciones.
OPCIONES = {
    "explicito.variables": ("yin33",),
    "explicito.t2_method": ("f",),
    "explicito.calibration": ("in_sample",),
    "explicito.regla_alarma": ("t2_o_spe",),
    "supuestos.S2_fichero_far": ("d00_te.dat",),
    "supuestos.S3_variables_dpca": ("yin33",),
    "supuestos.S4_ddof": (0, 1),
    "supuestos.S5_orden_dpca": ("escalar_y_retardar", "retardar_y_escalar"),
    "supuestos.S8_spe_method": ("jackson",),
    "supuestos.S11_t2_limite_dpca": ("f_n_ajuste", "f_n_por_h1", "dibujado_fig2", "dibujado_fig5"),
    "supuestos.S12_dpca_primeras_muestras": ("no_evaluadas",),
    "tolerancia.comparacion": ("conteo", "impreso"),
    "tolerancia.dpca": ("criterio", "descriptivo"),
}
#: Criterios de la tolerancia: numero (en puntos porcentuales o fallos) o "no_aplica".
CRITERIOS = ("fdr_umbral_por_fallo_pp", "fdr_min_fallos_dentro", "fdr_max_pp",
             "fdr_media_max_pp", "far_max_pp")
NO_APLICA = "no_aplica"


class RegistroNoFijado(RuntimeError):
    """La verificacion no esta fijada en el registro: V1 no toca los ficheros de prueba."""


class _PermisoPrueba:
    """Prueba de que la regla de orden se cumplio. Solo la crea `comprobar_registro`."""

    __slots__ = ("huella", "fecha", "fila", "git", "registro")

    def __init__(self, huella: str, fecha: str, fila: str, git: dict, registro: str = REGISTRO) -> None:
        self.huella, self.fecha, self.fila, self.git = huella, fecha, fila, git
        self.registro = registro       # fichero del registro, relativo a la raiz del repositorio


# --------------------------------------------------------------- configuracion

def _opcion(cfg: dict, clave: str):
    valor = tconfig.get(cfg, clave)
    if valor is not None and clave in OPCIONES and valor not in OPCIONES[clave]:
        raise ValueError(f"{clave} = {valor!r} no es una opcion admitida: {OPCIONES[clave]}")
    return valor


def _ventana(valor, nombre: str):
    """[inicio, fin] del yaml -> tupla para far_window (None = el final)."""
    if valor is None:
        return None
    if not isinstance(valor, (list, tuple)) or len(valor) != 2:
        raise ValueError(f"{nombre} debe ser [inicio, fin] (fin null = el final); llego {valor!r}")
    inicio, fin = valor
    if not isinstance(inicio, int) or (fin is not None and not isinstance(fin, int)):
        raise ValueError(f"{nombre}: inicio y fin tienen que ser enteros (o fin null); llego {valor!r}")
    return (inicio, fin)


def _validar_supuestos(cfg: dict) -> None:
    """Valores de los supuestos S1 a S12 (de la configuracion o de una variante)."""
    for clave in OPCIONES:
        if clave.startswith("supuestos."):
            _opcion(cfg, clave)
    n = tconfig.get(cfg, "supuestos.S1_n_ajuste")
    if isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 500:
        raise ValueError(f"supuestos.S1_n_ajuste debe ser un entero entre 2 y 500 (d00.dat); llego {n!r}")
    h = tconfig.get(cfg, "supuestos.S6_lags_dpca")
    if isinstance(h, bool) or not isinstance(h, int) or h < 1:
        raise ValueError(f"supuestos.S6_lags_dpca debe ser un entero >= 1; llego {h!r}")
    nivel = tconfig.get(cfg, "supuestos.S7_nivel")
    if not isinstance(nivel, float) or not 0 < nivel < 1:
        raise ValueError(f"supuestos.S7_nivel es el nivel de confianza (por ejemplo 0.99); llego {nivel!r}")
    ventana = _ventana(tconfig.get(cfg, "supuestos.S9_far_ventana"), "supuestos.S9_far_ventana")
    if ventana is not None:
        inicio, fin = ventana
        if not 0 <= inicio < (960 if fin is None else fin) <= 960:
            raise ValueError(f"supuestos.S9_far_ventana = {list(ventana)} no cabe en las 960 muestras de d00_te.dat")
    onset = tconfig.get(cfg, "supuestos.S10_fdr_desde")
    if onset != data.fault_onset("test", "classic"):
        raise ValueError(
            f"supuestos.S10_fdr_desde = {onset!r}, pero en los dXX_te.dat el fallo entra tras "
            f"la muestra {data.fault_onset('test', 'classic')}: la ventana de la FDR empieza ahi"
        )


def validar_config(cfg: dict) -> None:
    """Comprueba la estructura del yaml y los valores que no estan en null.

    No exige que no haya nulls (eso es `pendientes`): sirve tambien para
    --solo-limites y --huella, que se pueden ejecutar antes de fijar nada.
    """
    for clave in OPCIONES:
        _opcion(cfg, clave)
    for metodo in METODOS:
        l = tconfig.get(cfg, f"explicito.n_components.{metodo}")
        if isinstance(l, bool) or not isinstance(l, int) or l < 1:
            raise ValueError(f"explicito.n_components.{metodo} debe ser un entero positivo; llego {l!r}")
    _validar_supuestos(cfg)
    if sorted(tconfig.get(cfg, "explicito.excluidos_veredicto")) != sorted(metrics.HARD_FAULTS):
        raise ValueError("explicito.excluidos_veredicto debe ser [3, 9, 15] (memoria, parrafo 369)")
    for criterio in CRITERIOS:
        valor = tconfig.get(cfg, f"tolerancia.{criterio}")
        if valor is None or valor == NO_APLICA:
            continue
        if (isinstance(valor, bool) or not isinstance(valor, (int, float)) or not math.isfinite(valor)
                or valor < 0):
            raise ValueError(f"tolerancia.{criterio} debe ser un numero finito >= 0 o 'no_aplica'; "
                             f"llego {valor!r}")
    for variante in tconfig.get(cfg, "variantes") or []:
        if not isinstance(variante, dict) or set(variante) != {"nombre", "cambios"}:
            raise ValueError(f"Cada variante necesita 'nombre' y 'cambios'; llego {variante!r}")
        for clave, valor in variante["cambios"].items():
            if clave not in cfg["supuestos"]:
                raise ValueError(f"La variante {variante['nombre']} cambia {clave}, que no es un supuesto")
            if valor is None:
                raise ValueError(
                    f"La variante {variante['nombre']} deja {clave} en null: una variante se declara "
                    "con su valor (o se quita de la lista), no queda pendiente"
                )
        _aplicar(cfg, variante["cambios"])  # valida los valores de la variante


def _aplicar(cfg: dict, cambios: dict) -> dict:
    """Copia del yaml con los supuestos de una variante cambiados (y validados)."""
    nuevo = copy.deepcopy(cfg)
    nuevo["supuestos"].update(cambios)
    _validar_supuestos(nuevo)
    return nuevo


def pendientes(cfg: dict) -> list[str]:
    """Claves en null que impiden ejecutar sobre los ficheros de prueba.

    Todas las del yaml, salvo las de "secundaria" cuando secundaria.incluir es
    false. Ademas, la tolerancia tiene que tener al menos un criterio de FDR y el
    de la FAR, y el umbral por fallo va junto con el minimo de fallos dentro.
    """
    faltan = tconfig.pending(cfg)
    if tconfig.get(cfg, "secundaria.incluir") is False:
        faltan = [c for c in faltan if not c.startswith("secundaria.")]
    tol = cfg["tolerancia"]
    if not any(c is None for c in tol.values()):
        fdr = [tol[c] for c in CRITERIOS if c.startswith("fdr")]
        if all(v == NO_APLICA for v in fdr):
            faltan.append("tolerancia (al menos un criterio de FDR, parrafo 369)")
        if tol["far_max_pp"] == NO_APLICA:
            faltan.append("tolerancia.far_max_pp (la memoria pide reproducir la FAR, parrafo 369)")
        if (tol["fdr_umbral_por_fallo_pp"] == NO_APLICA) != (tol["fdr_min_fallos_dentro"] == NO_APLICA):
            faltan.append("tolerancia.fdr_umbral_por_fallo_pp y fdr_min_fallos_dentro (van juntas)")
    verificada = tconfig.get(cfg, "referencia.transcripcion_verificada")
    if verificada is not None:
        fecha = _fecha(verificada)
        if fecha is None:
            faltan.append("referencia.transcripcion_verificada (tiene que ser una fecha AAAA-MM-DD que exista)")
        elif fecha > date.today():
            faltan.append("referencia.transcripcion_verificada (fecha futura: todavia no se ha verificado)")
    return faltan


def _fecha(valor) -> date | None:
    """La fecha de un valor del yaml o de una celda (AAAA-MM-DD que exista), o None."""
    if isinstance(valor, date):
        return valor
    if not isinstance(valor, str) or not PATRON_FECHA.match(valor):
        return None
    try:
        return date.fromisoformat(valor)
    except ValueError:            # 2026-02-30 tiene la forma, pero no existe
        return None


def _es_fecha(valor) -> bool:
    return _fecha(valor) is not None


def rutas_referencia(cfg: dict, carpeta_yaml: Path) -> dict:
    """Rutas absolutas de los dos CSV de referencia (relativas a la carpeta del yaml)."""
    return {nombre: (carpeta_yaml / tconfig.get(cfg, f"referencia.{nombre}")).resolve()
            for nombre in ("tablas", "umbrales")}


def huella(cfg: dict, referencias: dict) -> str:
    """Huella de la verificacion: 12 cifras hexadecimales del SHA-256 de los valores
    del yaml (no de sus comentarios) y del texto de las dos referencias."""
    contenido = {
        "config": cfg,
        "referencias": {n: provenance.file_sha256(r, text=True) for n, r in sorted(referencias.items())},
    }
    canon = json.dumps(contenido, sort_keys=True, ensure_ascii=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()[:12]


# --------------------------------------------------------- regla de orden

def _valla_de_apertura(linea: str) -> str | None:
    """La valla que abre un bloque de codigo en esta linea (las comillas o tildes), o None.

    Reglas de CommonMark: como mucho tres espacios delante, al menos tres comillas
    invertidas o tres tildes seguidas, y, si son comillas, ninguna comilla invertida
    en el resto de la linea (el texto de informacion, como `python`).
    """
    m = PATRON_VALLA.match(linea)
    if m is None:
        return None
    valla = m.group(1)
    if valla[0] == "`" and "`" in linea[m.end():]:
        return None
    return valla


def _cierra_la_valla(linea: str, valla: str) -> bool:
    """Si la linea cierra el bloque abierto con `valla`: como mucho tres espacios delante,
    el mismo caracter repetido al menos tantas veces como en la apertura y nada detras
    salvo espacios (CommonMark). Una linea como ```python, o una valla mas corta, no
    cierra el bloque: sigue siendo codigo."""
    m = PATRON_CIERRE.match(linea)
    return bool(m) and m.group(1)[0] == valla[0] and len(m.group(1)) >= len(valla)


def lineas_visibles(texto: str) -> list[str]:
    """Las lineas de un markdown que se ven como texto: sin los comentarios <!-- -->
    (de una o de varias lineas) ni los bloques de codigo (``` o ~~~).

    Una fila escondida en un comentario o en un ejemplo de codigo no esta "escrita en
    el registro" para quien lo lee (V-12 de la revision del 12/10). Los bloques de
    codigo siguen las reglas de las vallas de CommonMark (`_valla_de_apertura` y
    `_cierra_la_valla`): un bloque abierto con cuatro comillas no lo cierra una linea
    de tres, ni lo cierra una linea con texto detras de la valla (W-01 de la segunda
    revision); un bloque sin cerrar llega hasta el final del fichero.
    """
    visibles = []
    en_comentario = False
    en_codigo = None                       # la valla que abrio el bloque (caracter y longitud)
    for linea in texto.splitlines():
        if en_codigo is None and not en_comentario:
            valla = _valla_de_apertura(linea)
            if valla is not None:
                en_codigo = valla
                continue
        if en_codigo is not None:
            if _cierra_la_valla(linea, en_codigo):
                en_codigo = None
            continue
        resto, limpia = linea, ""
        while resto:
            if en_comentario:
                fin = resto.find("-->")
                if fin < 0:
                    resto = ""
                else:
                    resto, en_comentario = resto[fin + 3:], False
            else:
                ini = resto.find("<!--")
                if ini < 0:
                    limpia, resto = limpia + resto, ""
                else:
                    limpia, resto, en_comentario = limpia + resto[:ini], resto[ini + 4:], True
        visibles.append(limpia)
    return visibles


def buscar_fila_registro(decisiones: Path, huella_actual: str, hoy: date | None = None) -> tuple[str, str]:
    """(fecha, fila) de docs/decisiones.md que fija la verificacion con esta huella.

    Una fila vale si es una fila visible de la tabla (empieza por "|" en la primera
    columna, fuera de comentarios <!-- --> y de bloques de codigo), su primera
    celda es una fecha AAAA-MM-DD que existe y no es posterior a hoy, y contiene
    [V1-yin huella=<huella>].
    """
    hoy = hoy or date.today()
    decisiones = Path(decisiones)
    if not decisiones.is_file():
        raise RegistroNoFijado(f"No encuentro el registro de decisiones {decisiones}")
    otras = []
    for linea in lineas_visibles(decisiones.read_text(encoding="utf-8")):
        if not linea.startswith("|"):
            continue
        linea = linea.strip()
        etiquetas = PATRON_FILA.findall(linea)
        if not etiquetas:
            continue
        if huella_actual not in etiquetas:
            otras.extend(etiquetas)
            continue
        fecha = linea.strip("|").split("|")[0].strip()
        dia = _fecha(fecha)
        if dia is None:
            raise RegistroNoFijado(f"La fila con la huella {huella_actual} no empieza por una fecha "
                                   f"AAAA-MM-DD que exista: {linea}")
        if dia > hoy:
            raise RegistroNoFijado(f"La fila con la huella {huella_actual} tiene fecha futura ({fecha})")
        return fecha, linea
    if otras:
        raise RegistroNoFijado(
            f"docs/decisiones.md fija la verificacion con otra huella ({', '.join(sorted(set(otras)))}) y la "
            f"de hoy es {huella_actual}: el yaml o las referencias cambiaron despues de fijarla. "
            "Hace falta una fila nueva, fechada, con la huella actual (--huella)."
        )
    raise RegistroNoFijado(
        f"Ninguna fila de {decisiones.name} fija la verificacion: falta una fila fechada con "
        f"[{ETIQUETA} huella={huella_actual}] (python experiments/V1_verificacion_yin.py --huella)."
    )


def codigo_en_ejecucion() -> list[Path]:
    """El codigo que hace la verificacion: este script y la carpeta del paquete tfmfdd
    que se importa (la de tfmfdd.__file__, sea cual sea)."""
    return [Path(__file__).resolve(), Path(tfmfdd.__file__).resolve().parent]


def estado_git(rutas: list[Path]) -> dict:
    """Si estas rutas estan confirmadas en git tal cual, sin escribir en el repositorio.

    Cada ruta por separado (`provenance.committed_status`): un fichero, versionado,
    sin cambios ni marcas skip-worktree o assume-unchanged, no ignorado y fuera de
    .git; una carpeta de codigo, todo versionado y sin cambios, y ningun .py que git
    no vea. Ademas, todas en el MISMO repositorio: un registro en un repositorio
    anidado o en otro no fija nada en el del codigo (V-05 de la revision del 12/10).

    Devuelve {"repository": True, "commit": str|None, "status": [problemas]}; la
    lista vacia significa "todo confirmado". Lanza RegistroNoFijado si git no esta,
    si una ruta esta fuera de un repositorio o dentro de .git, o si no estan todas
    en el mismo repositorio: sin eso no se puede comprobar que el registro y el
    codigo estaban confirmados antes de ejecutar.
    """
    try:
        estado = provenance.committed_status(rutas)
    except provenance.GitCheckError as exc:
        raise RegistroNoFijado(f"{exc}: el registro y el codigo tienen que estar confirmados en un "
                               "repositorio git") from None
    if len(estado["roots"]) != 1:
        raise RegistroNoFijado(
            "el yaml, el registro, las referencias y el codigo tienen que estar en el mismo repositorio "
            f"git, y estan en {len(estado['roots'])}: un registro en un repositorio anidado u otro no "
            "fija nada en este"
        )
    raiz = estado["roots"][0]
    return {"repository": True, "root": raiz, "commit": estado["commits"][raiz], "status": estado["problems"]}


def fichero_de_registro(decisiones: Path, raiz) -> str:
    """El fichero del registro, relativo a la raiz del repositorio y con "/".

    Si no se puede expresar relativo a la raiz, la ruta tal cual (con "~" en lugar de la
    carpeta personal): asi la procedencia nunca la oculta.
    """
    try:
        return Path(decisiones).resolve().relative_to(Path(raiz).resolve()).as_posix()
    except ValueError:
        return provenance._sin_casa(Path(decisiones).resolve().as_posix())


def comprobar_fichero_de_registro(registro: str, permitir_otro_registro: bool) -> None:
    """La fila tiene que estar en el registro de decisiones del repositorio, docs/decisiones.md
    (memoria, parrafo 369: "el registro de decisiones del repositorio del paquete comun").

    Otro fichero confirmado (unas notas, un borrador) fija la tolerancia antes de ejecutar,
    pero no en el registro (W-02 de la segunda revision): V1 se niega salvo que se pida
    a proposito con --permitir-otro-registro, y entonces la procedencia dice cual fue.
    """
    if os.path.normcase(registro) != os.path.normcase(REGISTRO) and not permitir_otro_registro:
        raise RegistroNoFijado(
            f"la fila esta en {registro}, no en el registro de decisiones del repositorio ({REGISTRO}; "
            "memoria, parrafo 369). Pasala a ese fichero, o, si de verdad tiene que ser otro, pidelo "
            "con --permitir-otro-registro (la procedencia dira cual fue)"
        )


def comprobar_registro(cfg: dict, carpeta_yaml: Path, ruta_yaml: Path, decisiones: Path,
                       hoy: date | None = None, codigo=None, *,
                       permitir_otro_registro: bool = False) -> _PermisoPrueba:
    """La regla de orden entera. Devuelve el permiso para abrir los ficheros de prueba
    o lanza RegistroNoFijado diciendo que falta. No lee ningun dato.

    `codigo` son las rutas del codigo que tiene que estar confirmado; por defecto,
    el que se esta ejecutando (`codigo_en_ejecucion`: este script y el paquete).
    `permitir_otro_registro`: aceptar la fila en un fichero que no sea
    docs/decisiones.md (por defecto, no; ver `comprobar_fichero_de_registro`).
    """
    validar_config(cfg)
    faltan = pendientes(cfg)
    if faltan:
        raise RegistroNoFijado("quedan decisiones pendientes en el yaml: " + "; ".join(faltan))
    referencias = rutas_referencia(cfg, carpeta_yaml)
    comprobar_referencia(cfg, referencias, ("principal",) + (("secundaria",) if cfg["secundaria"]["incluir"] else ()))
    comprobar_conteo_posible(cfg)
    actual = huella(cfg, referencias)
    fecha, fila = buscar_fila_registro(decisiones, actual, hoy)
    codigo = codigo_en_ejecucion() if codigo is None else [Path(c) for c in codigo]
    git = estado_git([ruta_yaml, decisiones, *referencias.values(), *codigo])
    # La raiz la da git; sin ella (solo si alguien sustituye estado_git), la carpeta que contiene configs/.
    registro = fichero_de_registro(decisiones, git.get("root") or Path(ruta_yaml).resolve().parent.parent)
    comprobar_fichero_de_registro(registro, permitir_otro_registro)
    if git["status"]:
        raise RegistroNoFijado(
            "el yaml, el registro, las referencias o el codigo (script y paquete) tienen cambios sin "
            "confirmar en git o no estan versionados tal cual: " + "; ".join(git["status"])
            + ". Confirmalos (commit) antes de ejecutar."
        )
    return _PermisoPrueba(actual, fecha, fila, git, registro)


def comprobar_conteo_posible(cfg: dict) -> None:
    """Con comparacion "conteo", la configuracion principal tiene que evaluar 800 muestras
    en la FDR y en la FAR (las de la rejilla de Yin); si no, el recuento no existe."""
    if cfg["tolerancia"]["comparacion"] != "conteo":
        return
    onset = cfg["supuestos"]["S10_fdr_desde"]
    n_te = 960
    inicio, fin = _ventana(cfg["supuestos"]["S9_far_ventana"], "S9_far_ventana")
    fin = n_te if fin is None else fin
    for metodo in METODOS:
        h = cfg["supuestos"]["S6_lags_dpca"] if metodo == "DPCA" else 0
        n_far = (fin - inicio) - max(0, h - inicio)
        if n_te - onset != N_REJILLA_YIN or n_far != N_REJILLA_YIN:
            raise RegistroNoFijado(
                f"tolerancia.comparacion = 'conteo' necesita {N_REJILLA_YIN} muestras evaluadas en la FDR "
                f"y en la FAR, y con la ventana {cfg['supuestos']['S9_far_ventana']} el {metodo} evalua "
                f"{n_far} en la FAR: usa 'impreso' o otra ventana"
            )


# --------------------------------------------------------------- referencia

def leer_referencia(ruta: Path, tabla_fdr: int, tabla_far: int) -> pd.DataFrame:
    """Cifras de Yin de PCA y DPCA de un par de tablas (FDR y FAR), leidas del CSV.

    Devuelve una fila por (metodo, fallo) con metrica ("fdr" o "far"), tabla,
    pagina, valor (en %), bloque, excluido (parrafo 369), dudosa y
    compatible_k800. Comprueba que hay exactamente una cifra por metodo y fallo.
    """
    ref = pd.read_csv(ruta, comment="#")
    necesarias = {"tabla", "pagina_impresa", "metrica", "metodo", "fallo", "bloque", "valor_num",
                  "compatible_k800", "fallo_excluido_parrafo369", "dudosa"}
    if not necesarias <= set(ref.columns):
        raise ValueError(f"A {ruta} le faltan columnas: {sorted(necesarias - set(ref.columns))}")
    filas = []
    for tabla, metrica, fallos in ((tabla_fdr, "fdr", range(1, 22)), (tabla_far, "far", (0,))):
        sub = ref[(ref["tabla"] == tabla) & ref["metodo"].isin(METODOS)]
        for metodo in METODOS:
            for fallo in fallos:
                celda = sub[(sub["metodo"] == metodo) & (sub["fallo"] == fallo)]
                if len(celda) != 1:
                    raise ValueError(f"{ruta.name}: tabla {tabla}, {metodo}, fallo {fallo}: "
                                     f"{len(celda)} filas (tiene que haber una)")
                c = celda.iloc[0]
                if str(c["metrica"]).lower() != metrica:
                    raise ValueError(f"{ruta.name}: la tabla {tabla} no es de {metrica.upper()}")
                filas.append({
                    "metodo": metodo, "fallo": int(fallo), "metrica": metrica, "tabla": int(tabla),
                    "pagina": int(c["pagina_impresa"]), "valor": float(c["valor_num"]),
                    "bloque": int(c["bloque"]), "excluido": bool(c["fallo_excluido_parrafo369"]),
                    "dudosa": bool(c["dudosa"]), "compatible_k800": bool(c["compatible_k800"]),
                })
    return pd.DataFrame(filas)


def comprobar_referencia(cfg: dict, referencias: dict, configuraciones) -> None:
    """La referencia que se va a usar no tiene celdas dudosas y sus exclusiones son 3, 9 y 15."""
    excluidos = set(cfg["explicito"]["excluidos_veredicto"])
    for nombre in configuraciones:
        tablas = _tablas(cfg, nombre)
        ref = leer_referencia(referencias["tablas"], *tablas)
        dudosas = ref[ref["dudosa"]]
        if not dudosas.empty:
            raise RegistroNoFijado(
                "la referencia tiene celdas dudosas en " + ", ".join(
                    f"tabla {r.tabla} {r.metodo} fallo {r.fallo}" for r in dudosas.itertuples())
                + ": resuelvelas antes (configs/referencias/LEEME)"
            )
        marcados = set(ref.loc[ref["excluido"], "fallo"])
        if marcados != excluidos:
            raise RegistroNoFijado(f"la referencia excluye {sorted(marcados)} y el yaml {sorted(excluidos)}")
        if nombre == "principal" and cfg["tolerancia"]["comparacion"] == "conteo" and not ref["compatible_k800"].all():
            raise RegistroNoFijado("comparacion 'conteo' con celdas fuera de la rejilla k/800")


def leer_umbrales(ruta: Path) -> pd.DataFrame:
    """Umbrales dibujados por Yin para PCA y DPCA (figura, estadistico, umbral, residuo)."""
    u = pd.read_csv(ruta, comment="#")
    necesarias = {"figura", "metodo", "estadistico", "umbral_leido", "residuo_max_marcas", "pagina_impresa"}
    if not necesarias <= set(u.columns):
        raise ValueError(f"A {ruta} le faltan columnas: {sorted(necesarias - set(u.columns))}")
    return u[u["metodo"].isin(METODOS)].reset_index(drop=True)


def _tablas(cfg: dict, nombre: str) -> tuple[int, int]:
    if nombre == "principal":
        return cfg["referencia"]["tabla_fdr"], cfg["referencia"]["tabla_far"]
    return cfg["secundaria"]["tabla_fdr"], cfg["secundaria"]["tabla_far"]


# ------------------------------------------------------------------- modelo

class EscaladorDdof0:
    """Media cero y desviacion uno con la desviacion poblacional (ddof = 0).

    Solo para la variante S4 (el escalado del paquete usa ddof = 1). Una variable
    constante se deja con desviacion 1, como `preprocessing.Scaler`.
    """

    def fit(self, X):
        X = np.asarray(X, dtype=float)
        self.mean_ = X.mean(axis=0)
        std = X.std(axis=0, ddof=0)
        self.std_ = np.where(std > 0, std, 1.0)
        return self

    def transform(self, X):
        return (np.asarray(X, dtype=float) - self.mean_) / self.std_


def configuracion_efectiva(cfg: dict, metodo: str, nombre: str = "principal") -> dict:
    """Los parametros que definen el modelo y la medida de un metodo (lo que va al CSV)."""
    s = cfg["supuestos"]
    if nombre == "principal":
        l = cfg["explicito"]["n_components"][metodo]
    else:
        l = cfg["secundaria"]["n_components"][metodo]
    lags = s["S6_lags_dpca"] if metodo == "DPCA" else 0
    usados = ["S1_n_ajuste", "S4_ddof", "S6_lags_dpca", "S7_nivel", "S8_spe_method"]
    if lags:
        usados += ["S5_orden_dpca", "S11_t2_limite_dpca"]
    nulos = [k for k in usados if s[k] is None] + (["n_components"] if l is None else [])
    if nulos:
        raise ValueError(f"{metodo} ({nombre}): {', '.join(nulos)} en null. Es una decision pendiente; "
                         "sin su valor no se puede ajustar el modelo")
    return {
        "n_components": l,
        "lags": lags,
        "n_fit_rows": s["S1_n_ajuste"],
        "ddof": s["S4_ddof"],
        "scale_lag_order": s["S5_orden_dpca"] if lags else "no_aplica",
        "confidence": s["S7_nivel"],
        "t2_method": cfg["explicito"]["t2_method"],
        "spe_method": s["S8_spe_method"],
        "t2_dpca_limit": s["S11_t2_limite_dpca"] if lags else "no_aplica",
        "calibration": cfg["explicito"]["calibration"],
        "far_window": s["S9_far_ventana"],
        "fdr_from": s["S10_fdr_desde"],
        "variables": cfg["explicito"]["variables"],
    }


def _clave_modelo(efectiva: dict) -> tuple:
    return tuple(json.dumps(efectiva[k], default=str) for k in
                 ("n_components", "lags", "n_fit_rows", "ddof", "scale_lag_order", "confidence",
                  "t2_method", "spe_method", "t2_dpca_limit", "calibration"))


def _escalador(X: np.ndarray, ddof: int, orden: str, lags: int):
    """Escalador para el gancho `scaler=` de PCAMonitor, o None para el interno (ddof = 1)."""
    if lags == 0:
        return None if ddof == 1 else EscaladorDdof0().fit(X)
    if orden == "escalar_y_retardar":
        return Scaler().fit(X) if ddof == 1 else EscaladorDdof0().fit(X)
    if ddof != 1:
        raise ValueError("retardar_y_escalar solo esta implementado con ddof = 1 (el escalador interno)")
    return None


def ajustar(X_d00: np.ndarray, efectiva: dict, umbrales: pd.DataFrame | None = None) -> PCAMonitor:
    """Ajusta el monitor de Yin con la configuracion efectiva, en muestra a proposito."""
    n = efectiva["n_fit_rows"]
    X = X_d00[:n]
    lags = efectiva["lags"]
    runs = np.ones(len(X), dtype=int) if lags else None
    modelo = PCAMonitor(n_components=efectiva["n_components"], alpha=efectiva["confidence"], lags=lags,
                        t2_method=efectiva["t2_method"], spe_method=efectiva["spe_method"])
    modelo.fit(X, runs=runs, calibration=efectiva["calibration"],
               scaler=_escalador(X, efectiva["ddof"], efectiva["scale_lag_order"], lags))
    opcion = efectiva["t2_dpca_limit"]
    if lags == 0 or opcion == "f_n_ajuste":
        return modelo
    otro = copy.copy(modelo)
    if opcion == "f_n_por_h1":
        otro.t2_limit_ = limits.t2_limit(modelo.n_components_, (lags + 1) * modelo.n_fit_, efectiva["confidence"])
    else:
        if umbrales is None:
            raise ValueError(f"{opcion} necesita el CSV de umbrales dibujados")
        figura = {"dibujado_fig2": "Fig. 2", "dibujado_fig5": "Fig. 5"}[opcion]
        fila = umbrales[(umbrales["metodo"] == "DPCA") & (umbrales["estadistico"] == "T2")
                        & (umbrales["figura"] == figura)]
        if len(fila) != 1:
            raise ValueError(f"No hay un unico umbral dibujado del T2 del DPCA en la {figura}")
        otro.t2_limit_ = float(fila["umbral_leido"].iloc[0])
    otro.limit_source_ = {**modelo.limit_source_, "t2": f"s11_{opcion}"}
    return otro


def cargar_d00(root: Path, variables: str) -> np.ndarray:
    """d00.dat (el ajuste) con las 33 variables de Yin. No es un fichero de prueba."""
    if variables != "yin33":
        raise ValueError(f"variables = {variables!r}: solo esta implementado 'yin33'")
    return data.values(data.load_classic(0, "train", root=root), columns=data.YIN_VARS)


def cargar_prueba(fallo: int, root: Path, permiso: _PermisoPrueba) -> np.ndarray:
    """Un dXX_te.dat con las 33 variables de Yin. Exige el permiso de la regla de orden."""
    if not isinstance(permiso, _PermisoPrueba):
        raise RegistroNoFijado("cargar un fichero de prueba exige el permiso de comprobar_registro")
    return data.values(data.load_classic(fallo, "test", root=root), columns=data.YIN_VARS)


# ------------------------------------------------------------------- medida

def _comprobar_recuento(tasa: float, n_alarmas: int, n_evaluadas: int) -> None:
    # El recuento propio y la tasa de evaluate_run tienen que decir lo mismo.
    if n_evaluadas and not np.isclose(tasa, n_alarmas / n_evaluadas, rtol=0, atol=1e-12):
        raise AssertionError(f"tasa {tasa} y recuento {n_alarmas}/{n_evaluadas} no coinciden")


def medir_fdr(indice: np.ndarray, onset: int, k: int) -> dict:
    """FDR del indice combinado sobre las muestras onset+1 a 960, con su recuento y el retardo.

    Usa `metrics.evaluate_run(indice, 1.0, onset)`: la FDR es la proporcion de
    muestras evaluadas desde el indice `onset` (base cero) con el indice por
    encima de 1, es decir, con T2 o SPE por encima de su limite.
    """
    r = metrics.evaluate_run(indice, 1.0, onset=onset, k=k)
    n_alarmas = int(metrics.alarms(indice[onset:], 1.0).sum())
    _comprobar_recuento(r["fdr"], n_alarmas, r["n_fdr"])
    return {"rate": r["fdr"], "n_alarms": n_alarmas, "n_evaluated": int(r["n_fdr"]),
            "delay_samples": r["delay_samples"], "detected_at_chance": r["detected_at_chance"]}


def medir_far(indice: np.ndarray, ventana: tuple, k: int) -> dict:
    """FAR del indice combinado en la ventana declarada, con su recuento.

    Usa `metrics.evaluate_run(indice, 1.0, onset=None, far_window=ventana)`:
    `ventana` es (inicio, fin) en base cero, fin excluido, None = el final. Las
    muestras no evaluadas (NaN del DPCA) no cuentan.
    """
    r = metrics.evaluate_run(indice, 1.0, onset=None, k=k, far_window=ventana)
    inicio, fin = ventana
    n_alarmas = int(metrics.alarms(indice[inicio:fin], 1.0).sum())
    _comprobar_recuento(r["far"], n_alarmas, r["n_far"])
    return {"rate": r["far"], "n_alarms": n_alarmas, "n_evaluated": int(r["n_far"])}


def comparar(fila: dict, ref: dict) -> dict:
    """Nuestra tasa frente a la de Yin, en puntos y, si se puede, en muestras."""
    nuestra = 100.0 * fila["n_alarms"] / fila["n_evaluated"] if fila["n_evaluated"] else np.nan
    salida = {"ours": nuestra, "diff_pp": nuestra - ref["valor"], "yin_count": np.nan,
              "diff_samples": np.nan, "diff_pp_count": np.nan}
    if fila["n_evaluated"] == N_REJILLA_YIN and ref["compatible_k800"]:
        recuento = int(round(ref["valor"] * N_REJILLA_YIN / 100.0))
        salida.update(yin_count=recuento, diff_samples=fila["n_alarms"] - recuento,
                      diff_pp_count=100.0 * (fila["n_alarms"] - recuento) / N_REJILLA_YIN)
    return salida


def evaluar_tolerancia(tabla: pd.DataFrame, tol: dict) -> dict:
    """Criterios de la tolerancia sobre las filas de UN metodo y una variante.

    `tabla` trae una fila por fallo (metric "fdr") y una de FAR (metric "far"),
    con `abs_diff_verdict` y `excluded_369`. Los fallos excluidos (3, 9 y 15) no
    entran. Devuelve el valor de cada criterio, si se cumple y el veredicto
    ("verificado" o "no verificado"); un criterio "no_aplica" no cuenta.
    """
    fdr = tabla[(tabla["metric"] == "fdr") & ~tabla["excluded_369"].astype(bool)]
    far = tabla[tabla["metric"] == "far"]
    if len(far) != 1:
        raise ValueError("evaluar_tolerancia necesita exactamente una fila de FAR")
    d = np.round(fdr["abs_diff_verdict"].to_numpy(dtype=float), 9)
    d_far = round(float(far["abs_diff_verdict"].iloc[0]), 9)
    if np.isnan(d).any() or np.isnan(d_far):
        raise ValueError("Hay diferencias sin calcular (NaN) en las filas del veredicto")

    def num(clave):
        v = tol[clave]
        return None if v == NO_APLICA else float(v)

    res = {"n_faults_verdict": int(d.size), "fdr_n_within": np.nan, "fdr_max_abs_pp": float(d.max()),
           "fdr_mean_abs_pp": float(d.mean()), "far_abs_pp": d_far}
    cumple = []
    u, m = num("fdr_umbral_por_fallo_pp"), num("fdr_min_fallos_dentro")
    if u is not None and m is not None:
        res["fdr_n_within"] = int((d <= u).sum())
        res["ok_fdr_within"] = bool(res["fdr_n_within"] >= m)
        cumple.append(res["ok_fdr_within"])
    if num("fdr_max_pp") is not None:
        res["ok_fdr_max"] = bool(d.max() <= num("fdr_max_pp"))
        cumple.append(res["ok_fdr_max"])
    if num("fdr_media_max_pp") is not None:
        res["ok_fdr_mean"] = bool(d.mean() <= num("fdr_media_max_pp"))
        cumple.append(res["ok_fdr_mean"])
    if num("far_max_pp") is not None:
        res["ok_far"] = bool(d_far <= num("far_max_pp"))
        cumple.append(res["ok_far"])
    if not cumple:
        raise ValueError("La tolerancia no tiene ningun criterio aplicable")
    res["verdict"] = "verificado" if all(cumple) else "no verificado"
    return res


def _abs_diff_veredicto(fila: dict, comparacion: str | None) -> float:
    if comparacion == "conteo" and not np.isnan(fila["diff_pp_count"]):
        return abs(fila["diff_pp_count"])
    return abs(fila["diff_pp"])


def _dentro(fila: dict, tol: dict):
    """Si la diferencia de una fila cae dentro de su umbral (umbral por fallo o de la FAR).

    Es informacion por fila: el veredicto lo dan los criterios del resumen. Vacio
    (None) en los fallos excluidos (3, 9 y 15) y cuando no hay umbral por fila.
    """
    if fila["metric"] == "far":
        umbral = tol["far_max_pp"]
    elif fila["excluded_369"]:
        return None
    else:
        umbral = tol["fdr_umbral_por_fallo_pp"]
    if umbral == NO_APLICA:
        return None
    return bool(round(fila["abs_diff_verdict"], 9) <= float(umbral))


def ejecutar_medida(cfg: dict, X_d00: np.ndarray, root: Path, permiso: _PermisoPrueba,
                    umbrales: pd.DataFrame, referencias: dict) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Ajusta, puntua los 22 ficheros de prueba y compara con Yin. Devuelve (detalle, resumen)."""
    if not isinstance(permiso, _PermisoPrueba):
        raise RegistroNoFijado("ejecutar_medida exige el permiso de comprobar_registro")
    k = cfg["explicito"]["retardo_k"]
    tol = cfg["tolerancia"]
    excluidos = set(cfg["explicito"]["excluidos_veredicto"])
    onset = cfg["supuestos"]["S10_fdr_desde"]

    trabajos = []   # (configuracion, variante, metodo, cfg efectivo)
    configuraciones = ["principal"] + (["secundaria"] if cfg["secundaria"]["incluir"] else [])
    for nombre in configuraciones:
        for metodo in METODOS:
            base = configuracion_efectiva(cfg, metodo, nombre)
            trabajos.append((nombre, "principal", metodo, cfg, base))
            if nombre != "principal":
                continue
            for variante in cfg["variantes"] or []:
                otro = _aplicar(cfg, variante["cambios"])
                efectiva = configuracion_efectiva(otro, metodo, nombre)
                if efectiva == base:
                    continue        # la variante no cambia nada en este metodo
                trabajos.append((nombre, variante["nombre"], metodo, otro, efectiva))

    prueba = {f: cargar_prueba(f, root, permiso) for f in range(22)}
    hashes = {f: provenance.file_sha256(Path(root) / f"d{f:02d}_te.dat") for f in range(22)}

    modelos: dict = {}
    filas = []
    for nombre, variante, metodo, cfg_v, efectiva in trabajos:
        clave = _clave_modelo(efectiva)
        if clave not in modelos:
            modelos[clave] = ajustar(X_d00, efectiva, umbrales)
        modelo = modelos[clave]
        ref = leer_referencia(referencias["tablas"], *_tablas(cfg, nombre))
        ref = ref[ref["metodo"] == metodo].set_index("fallo")
        runs = (lambda n: np.ones(n, dtype=int)) if efectiva["lags"] else (lambda n: None)
        comunes = {
            "configuracion": nombre, "variant": variante, "method": metodo.lower(),
            "simulationRun": 1, **{k_: (json.dumps(v) if isinstance(v, list) else v) for k_, v in efectiva.items()},
            "n_vars": len(data.YIN_VARS), "t2_limit": modelo.t2_limit_, "spe_limit": modelo.spe_limit_,
            "t2_limit_source": modelo.limit_source_["t2"], "spe_limit_source": modelo.limit_source_["spe"],
            "n_fit": modelo.n_fit_, "alarm_rule": cfg["explicito"]["regla_alarma"],
        }
        for fallo in range(22):
            X = prueba[fallo]
            t2, spe = modelo.score(X, runs=runs(len(X)))
            indice = modelo.combined_index(t2, spe)
            r = ref.loc[fallo]
            if fallo == 0:
                ventana = _ventana(cfg_v["supuestos"]["S9_far_ventana"], "S9_far_ventana")
                medida = medir_far(indice, ventana, k)
                extra = {"metric": "far", "fdr_t2": np.nan, "fdr_spe": np.nan, "delay_samples": np.nan,
                         "detected_at_chance": np.nan}
            else:
                medida = medir_fdr(indice, onset, k)
                extra = {"metric": "fdr", "fdr_t2": metrics.fdr(t2[onset:], modelo.t2_limit_),
                         "fdr_spe": metrics.fdr(spe[onset:], modelo.spe_limit_),
                         "delay_samples": medida["delay_samples"],
                         "detected_at_chance": medida["detected_at_chance"]}
            fila = {**comunes, "faultNumber": fallo, **extra,
                    "yin_table": int(r["tabla"]), "yin_page": int(r["pagina"]), "yin_value": float(r["valor"]),
                    "block_yin": int(r["bloque"]), "excluded_369": bool(fallo in excluidos),
                    "n_alarms": medida["n_alarms"], "n_evaluated": medida["n_evaluated"],
                    "test_data_sha256": hashes[fallo]}
            fila.update(comparar(medida, r))
            fila["abs_diff_verdict"] = _abs_diff_veredicto(fila, tol["comparacion"])
            fila["within_tolerance"] = _dentro(fila, tol)
            filas.append(fila)
    detalle = pd.DataFrame(filas)

    return detalle, resumir(detalle, tol)


def resumir(detalle: pd.DataFrame, tol: dict) -> pd.DataFrame:
    """Una fila por (configuracion, variante, metodo) con los criterios y el veredicto.

    Solo la configuracion principal sin variantes cuenta para el veredicto
    ("verificado" o "no verificado"; "descriptivo" en el DPCA si asi se decidio).
    En las variantes y en la secundaria se dan los valores de los criterios como
    informacion, sin aprobado ni suspenso: no cambian el veredicto principal.
    """
    filas = []
    for (nombre, variante, metodo), g in detalle.groupby(["configuracion", "variant", "method"], sort=False):
        crit = evaluar_tolerancia(g, tol)
        usados = g[(g["metric"] == "fdr") & ~g["excluded_369"].astype(bool)]
        # Spearman solo como complemento (informe 15): no detecta un desplazamiento
        # uniforme. Con una de las dos series constante no esta definido: NaN.
        variables = usados["ours"].nunique() > 1 and usados["yin_value"].nunique() > 1
        rho = sps.spearmanr(usados["ours"], usados["yin_value"]).statistic if variables else np.nan
        cuenta = nombre == "principal" and variante == "principal"
        juzga = cuenta and not (metodo == "dpca" and tol["dpca"] == "descriptivo")
        if not cuenta:
            veredicto = "informativo"
        elif not juzga:
            veredicto = "descriptivo"
        else:
            veredicto = crit["verdict"]
        valores = {c: v for c, v in crit.items() if c != "verdict" and not c.startswith("ok_")}
        aprobados = {c: (v if juzga else None) for c, v in crit.items() if c.startswith("ok_")}
        filas.append({"configuracion": nombre, "variant": variante, "method": metodo, **valores,
                      **aprobados, "criteria_verdict": crit["verdict"] if juzga else None,
                      "spearman_rho_fdr": rho, "counts_for_verdict": cuenta, "verdict": veredicto,
                      "comparison": tol["comparacion"], **{f"tol_{c}": tol[c] for c in CRITERIOS},
                      **{c: g[c].iloc[0] for c in ("n_components", "lags", "n_fit", "t2_limit", "spe_limit",
                                                   "far_window", "t2_dpca_limit", "scale_lag_order", "ddof")}})
    return pd.DataFrame(filas)


# ------------------------------------------------------------- solo limites

def tabla_limites(cfg: dict, X_d00: np.ndarray, umbrales: pd.DataFrame) -> pd.DataFrame:
    """Limites con d00.dat (principal y variantes que cambian el modelo) frente a los dibujados."""
    filas = []
    vistos = set()
    for metodo in METODOS:
        candidatos = [("principal", cfg)] + [(v["nombre"], _aplicar(cfg, v["cambios"])) for v in cfg["variantes"] or []]
        for variante, cfg_v in candidatos:
            efectiva = configuracion_efectiva(cfg_v, metodo)
            clave = (metodo,) + _clave_modelo(efectiva)
            if clave in vistos:
                continue
            vistos.add(clave)
            modelo = ajustar(X_d00, efectiva, umbrales)
            for estadistico, valor in (("T2", modelo.t2_limit_), ("SPE", modelo.spe_limit_)):
                dibujados = umbrales[(umbrales["metodo"] == metodo) & (umbrales["estadistico"] == estadistico)]
                for d in dibujados.itertuples():
                    filas.append({
                        "method": metodo.lower(), "variant": variante, "statistic": estadistico,
                        "limit": valor, "limit_source": modelo.limit_source_[estadistico.lower()],
                        "figure": d.figura, "page": int(d.pagina_impresa), "drawn": float(d.umbral_leido),
                        "diff": valor - float(d.umbral_leido), "reading_residual": float(d.residuo_max_marcas),
                        "abs_diff_over_residual": abs(valor - float(d.umbral_leido)) / float(d.residuo_max_marcas),
                        **{k: (json.dumps(v) if isinstance(v, list) else v) for k, v in efectiva.items()
                           if k not in ("far_window", "fdr_from")},
                        "n_fit": modelo.n_fit_, "n_vars": len(data.YIN_VARS),
                    })
    return pd.DataFrame(filas)


# --------------------------------------------------------------------- main

def _argumentos(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="configs/yin2012.yaml", help="yaml de la verificacion")
    p.add_argument("--decisiones", default=None,
                   help="registro de decisiones (por defecto, docs/decisiones.md junto a configs/)")
    p.add_argument("--permitir-otro-registro", action="store_true",
                   help="aceptar la fila en un --decisiones que no sea docs/decisiones.md (a proposito; "
                        "la procedencia dice cual fue)")
    p.add_argument("--root", default="data/tep_classic", help="carpeta del TEP clasico")
    p.add_argument("--out", default="results/summary")
    modo = p.add_mutually_exclusive_group()
    modo.add_argument("--solo-limites", action="store_true",
                      help="solo los limites con d00.dat frente a los dibujados (sin ficheros de prueba)")
    modo.add_argument("--huella", action="store_true", help="imprime la huella para docs/decisiones.md")
    return p.parse_args(argv)


def ejecutar(argv=None) -> int:
    args = _argumentos(argv)
    ruta_yaml = Path(args.config).resolve()
    cfg = tconfig.load_yaml(ruta_yaml)
    try:
        validar_config(cfg)
    except (ValueError, KeyError) as exc:
        print(f"V1 no se ejecuta: {ruta_yaml.name} no es valido: {exc}", file=sys.stderr)
        return 2
    carpeta = ruta_yaml.parent
    decisiones = Path(args.decisiones).resolve() if args.decisiones else carpeta.parent / "docs" / "decisiones.md"
    referencias = rutas_referencia(cfg, carpeta)
    root, out = Path(args.root), Path(args.out)
    archivos_cfg = [{"path": str(ruta_yaml), "source": "explicit", "sha256": provenance.file_sha256(ruta_yaml, text=True)}]
    archivos_cfg += [{"path": str(r), "source": "referencia", "sha256": provenance.file_sha256(r, text=True)}
                     for r in referencias.values()]

    if args.huella:
        actual = huella(cfg, referencias)
        faltan = pendientes(cfg)
        print(f"Huella de la verificacion: {actual}")
        print(f"Etiqueta para la fila de docs/decisiones.md: [{ETIQUETA} huella={actual}]")
        if faltan:
            print("AVISO: quedan decisiones pendientes; no registres esta huella todavia, cambiara al "
                  "rellenarlas:\n  - " + "\n  - ".join(faltan))
            return 1
        return 0

    umbrales = leer_umbrales(referencias["umbrales"])
    if args.solo_limites:
        contexto = provenance.run_context(__file__)      # antes de escribir nada
        X_d00 = cargar_d00(root, cfg["explicito"]["variables"])
        try:
            tabla = tabla_limites(cfg, X_d00, umbrales)
        except ValueError as exc:
            print(f"V1 --solo-limites no se ejecuta: {exc}", file=sys.stderr)
            return 2
        out.mkdir(parents=True, exist_ok=True)
        ruta = out / SALIDA_LIMITES
        tabla.to_csv(ruta, index=False)
        provenance.write_provenance(
            ruta, config=cfg, config_files=archivos_cfg, script=__file__,
            ids={"fuente": "classic", "ajuste": f"d00.dat, filas 1 a {cfg['supuestos']['S1_n_ajuste']} "
                 "(y las de las variantes)", "prueba": "ninguna (--solo-limites)"},
            inputs=[root / "d00.dat"],
            extra={"modo": "solo_limites", "huella": huella(cfg, referencias)}, **contexto)
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(tabla[["method", "variant", "statistic", "limit", "figure", "drawn", "diff",
                         "reading_residual", "abs_diff_over_residual"]].to_string(index=False))
        print(f"\nGuardado en {ruta} (solo d00.dat; ningun fichero de prueba)")
        return 0

    try:
        permiso = comprobar_registro(cfg, carpeta, ruta_yaml, decisiones,
                                     permitir_otro_registro=args.permitir_otro_registro)
    except RegistroNoFijado as exc:
        print(f"V1 no se ejecuta sobre los ficheros de prueba: {exc}", file=sys.stderr)
        return 2

    contexto = provenance.run_context(__file__)          # el estado de antes de escribir nada
    X_d00 = cargar_d00(root, cfg["explicito"]["variables"])
    detalle, resumen = ejecutar_medida(cfg, X_d00, root, permiso, umbrales, referencias)
    for tabla in (detalle, resumen):
        tabla["fingerprint"] = permiso.huella
        tabla["decision_date"] = permiso.fecha
        tabla["decision_file"] = permiso.registro
        tabla["package_commit"] = permiso.git["commit"]
        tabla["fit_data_sha256"] = provenance.file_sha256(root / "d00.dat")
    out.mkdir(parents=True, exist_ok=True)
    comun = dict(
        config=cfg, config_files=archivos_cfg, script=__file__,
        ids={"fuente": "classic", "ajuste": f"d00.dat, filas 1 a {cfg['supuestos']['S1_n_ajuste']}",
             "prueba": [f"d{f:02d}_te.dat" for f in range(22)], "simulationRun": [1]},
        inputs=[root / "d00.dat"] + [root / f"d{f:02d}_te.dat" for f in range(22)],
        extra={"huella": permiso.huella,
               "registro": {"fichero": permiso.registro, "fecha": permiso.fecha, "fila": permiso.fila},
               "git_registro": permiso.git},
        **contexto,
    )
    for tabla, nombre in ((detalle, SALIDA), (resumen, SALIDA_RESUMEN)):
        tabla.to_csv(out / nombre, index=False)
        provenance.write_provenance(out / nombre, **comun)

    principal = resumen[resumen["counts_for_verdict"]]
    print(principal[["method", "n_faults_verdict", "fdr_n_within", "fdr_max_abs_pp", "fdr_mean_abs_pp",
                     "far_abs_pp", "criteria_verdict", "verdict"]].to_string(index=False))
    print("\nVeredicto (configuracion principal, sin 3, 9 y 15): "
          + " | ".join(f"{r.method}: {r.verdict}" for r in principal.itertuples()))
    print(f"Guardado en {out / SALIDA} y {out / SALIDA_RESUMEN}")
    return 0


def main(argv=None) -> None:
    with warnings.catch_warnings():
        # d00.dat viene transpuesto y load_classic lo corrige con un aviso: se deja ver una vez.
        warnings.simplefilter("once")
        codigo = ejecutar(argv)
    if codigo:
        sys.exit(codigo)


if __name__ == "__main__":
    main()
