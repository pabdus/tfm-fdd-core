"""Robustez ante el ruido: indice de Bernal-de Lazaro et al. (2018) y contaminacion de los datos.

Por que existe. La memoria promete evaluar la robustez ante el ruido de los
modelos finales de cada bloque con el indice J_RIL de Bernal-de Lazaro et al.
(2018). Para que las tres cifras se puedan comparar, los tres bloques tienen que
contaminar los datos igual (mismo tipo de ruido, mismos niveles, mismas
semillas) y calcular el indice con el mismo codigo. Como el indice esta definido
para clasificadores y la memoria dice que su adaptacion al TEP y a los
detectores se declara antes de ejecutar, este modulo no decide nada de eso: la
rejilla de niveles, el tipo de ruido, las variables que lo reciben, la semilla,
la metrica de error y el nivel de confianza son parametros obligatorios.

Lo que define el articulo (paginas impresas de Neurocomputing 275; entre
parentesis, la pagina del PDF):

- Ec. 2, p. 639 (4): J_RIL = I_yy + D_p + Delta_I. A es mas robusto que B si
  J_A < J_B (p. 641 (6)).
- Ec. 3, p. 639 (4): E = 1 - (1/k) * suma de p(x|c_i), el complemento del
  acierto medio por clase, entre 0 y 1.
- Ec. 4, p. 640 (5): I_yy = (Delta_eta/2) [eta_0^2 E(eta_0) + 2 suma eta_i^2
  E(eta_i) + eta_n^2 E(eta_n)], la regla del trapecio sobre eta^2 * E media.
- Ecs. 5 y 6, p. 640 (5): Delta_I = I0_max - I0_min, areas por trapecio (sin
  el peso eta^2) bajo los extremos E_max y E_min del intervalo t de la media
  del error en cada nivel: media +- t_(alfa/2) * desviacion / raiz(n_e).
- Ec. 7, p. 640 (5): D_p = suma de [E(eta_i) - E(eta_(i-1))] / (eta_i -
  eta_(i-1)), las pendientes entre niveles consecutivos.
- Ecs. 8 a 11 y lema 1, pp. 641-642 (6-7): la severidad eta = (Psi_X /
  Psi_S)^2 compara la suma de coeficientes de variacion con ruido y sin el; el
  ruido aditivo que da una severidad eta tiene desviacion sigma_gamma =
  raiz(eta - 1) * sigma_s en cada variable. eta = 1 es el dato sin ruido y, con
  limites mu_s +- 3 sigma_s, eta va de 1 a 10 (corolario, p. 642 (7)).
- Pasos 1 a 4, p. 642 (7): entrenar sin ruido con la clase normal y todos los
  fallos (validacion cruzada para el error sin ruido y su intervalo), fijar los
  niveles (el articulo recomienda un paso de 1e-3), repetir varias evaluaciones
  por nivel y calcular el indice.

Lo que el PDF deja abierto lo marca cada funcion con [VERIFICAR] (lectura mia
de algo ambiguo) o [ADAPTACION] (algo que el articulo no hace). Una lectura que
si se pudo resolver: la ec. 7 suma desde i = 2 mientras la ec. 4 numera los
niveles desde eta_0, de modo que, leida al pie de la letra, saltaria la
pendiente del primer tramo. Las Tablas 1 a 3 del articulo (p. 643 (8)), con
dos niveles (eta_0 = 1 y eta_1 = 2), solo se reproducen si ese tramo cuenta,
asi que aqui D_p suma TODOS los tramos consecutivos (lo comprueba
tests/test_v03_coste_robustez.py).

Los cuatro problemas para aplicarlo al TEP (informe 06 de la revision del
03/10, seccion 6.2), que el registro de decisiones tiene que resolver antes de
ejecutar:

1. Rieth no se puede volver a simular con el ruido dentro del lazo cerrado,
   como hace el articulo con el CSTR (p. 645 (10)). Aqui el ruido se anade a
   los datos registrados: es ruido en el canal de monitorizacion, que los
   controladores no ven. Hay que declararlo.
2. Un paso de 1e-3 entre 1 y 10 son 9001 niveles, inviable con los modelos
   caros, y ademas cambia el indice: con paso uniforme, D_p es exactamente
   (E_ultimo - E_primero) / paso, asi que crece como 1/paso mientras I_yy
   converge. J solo se compara entre metodos con la MISMA rejilla, nunca con la
   Tabla 7 del articulo, y conviene reportar I_yy, D_p y Delta_I por separado.
3. El rango [1, 10] satura a los detectores: con el limite fijado sin ruido, la
   FAR se dispara mucho antes de eta = 10. La rejilla de deteccion y la de
   diagnostico pueden tener que ser distintas, y el J de un detector no se
   compara con el de un clasificador.
4. Limite fijo o recalibrado. Entrenar sin ruido y evaluar con ruido (paso 1)
   es, en un detector, dejar el limite fijo y ver subir la FAR. Recalibrar el
   limite con datos normales con ruido mide otra cosa: cuanta FDR se pierde a
   igual FAR. Este modulo solo contamina datos: para el limite fijo se
   contamina solo la prueba; para el recalibrado, tambien X_cal. Ojo: con el
   limite fijo, el ruido anade alarmas y la curva de 1 - FDR puede BAJAR con
   eta (ver `error_rate`), y entonces el indice sobre ella no mide robustez.

Ademas hay que fijar: que variables reciben ruido (en el TEP, las de analizador,
xmeas_23 a xmeas_41, repiten valor entre lecturas, y un ruido blanco muestra a
muestra rompe esa estructura); de que datos sale sigma_s; la unidad de
repeticion del intervalo de la ec. 6 (simulaciones de prueba, pliegues de
validacion cruzada o realizaciones del ruido); si se usan numeros aleatorios
comunes entre niveles; y como se agregan los fallos (los 3, 9 y 15, aparte,
como en todo el protocolo).

Uso tipico (detector con limite fijo; los nombres en mayusculas son los
valores declarados en el registro de decisiones, no sugerencias):

    sigma_s = X_normal_ajuste.std(axis=0, ddof=1)     # o lo que se declare
    errores = [[] for _ in NIVELES]                   # rejilla declarada, empieza en 1
    for i, eta, r, Xr in robustness.noisy_versions(
            X_prueba, NIVELES, sigma_s, noise_type=TIPO_DE_RUIDO, seed=SEMILLA,
            n_repeats=R, variables=VARIABLES, common_random_numbers=COMUNES,
            groups=ids_de_simulacion):
        t2, spe = modelo.score(Xr)
        alarma = metrics.alarms(spe, modelo.spe_limit_)
        errores[i].append(robustness.error_rate(con_fallo, alarma, metric=METRICA))
    indice = robustness.robustness_index(NIVELES, errores, confidence=CONFIANZA)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

#: Tipos de ruido disponibles. Los dos son aditivos, blancos (independientes
#: entre muestras y entre variables), de media cero y con la desviacion del
#: lema 1. "gaussian" es el del articulo; "uniform" es una [ADAPTACION] con la
#: misma desviacion, para quien quiera un ruido acotado.
NOISE_TYPES = ("gaussian", "uniform")

#: Metricas de error que admite `error_rate` (ver su docstring).
ERROR_METRICS = ("eq3", "far", "missed_detection", "one_minus_macro_f1")


# ---------------------------------------------------------------- comprobaciones

def _entropia(seed) -> tuple[int, ...]:
    """Convierte la semilla en una tupla de enteros no negativos para SeedSequence."""
    if seed is None:
        raise TypeError("seed es obligatoria: pasa un entero o una secuencia de enteros")
    if isinstance(seed, (bool, np.bool_)):
        raise TypeError("seed debe ser un entero o una secuencia de enteros, no un booleano")
    if isinstance(seed, (int, np.integer)):
        partes = (int(seed),)
    else:
        try:
            partes = tuple(seed)
        except TypeError as exc:
            raise TypeError("seed debe ser un entero o una secuencia de enteros") from exc
        if not partes:
            raise ValueError("seed no puede ser una secuencia vacia")
        for v in partes:
            if isinstance(v, (bool, np.bool_)) or not isinstance(v, (int, np.integer)):
                raise TypeError(f"seed solo admite enteros, llego {v!r}")
        partes = tuple(int(v) for v in partes)
    if any(v < 0 for v in partes):
        raise ValueError("seed solo admite enteros no negativos")
    return partes


def _comprobar_niveles(levels) -> np.ndarray:
    niveles = np.asarray(levels, dtype=float)
    if niveles.ndim != 1 or niveles.size < 2:
        raise ValueError("La rejilla de niveles necesita al menos dos niveles (1-D)")
    if not np.all(np.isfinite(niveles)):
        raise ValueError("La rejilla de niveles tiene NaN o infinitos")
    if not np.isclose(niveles[0], 1.0, rtol=0.0, atol=1e-12):
        raise ValueError(
            "El primer nivel debe ser eta_0 = 1, el dato sin ruido, que forma parte "
            "del indice (ec. 4 y p. 642 del articulo)"
        )
    if np.any(np.diff(niveles) <= 0):
        raise ValueError("Los niveles deben ser estrictamente crecientes (paso 2, p. 642)")
    return niveles


def _columnas(variables, p: int) -> np.ndarray:
    if isinstance(variables, str):
        if variables != "all":
            raise ValueError("variables debe ser 'all' o una lista de indices de columna")
        return np.arange(p)
    cols = np.asarray(variables)
    if cols.ndim != 1 or cols.size == 0:
        raise ValueError("variables debe ser 'all' o una lista no vacia de indices de columna")
    if cols.dtype.kind not in "iu":
        raise TypeError(
            "variables admite indices enteros de columna; con un DataFrame, saca la "
            "matriz con data.values y pasa los indices"
        )
    if (cols < 0).any() or (cols >= p).any():
        raise IndexError(f"variables tiene indices fuera de las {p} columnas de X")
    if np.unique(cols).size != cols.size:
        raise ValueError("variables tiene indices repetidos")
    return cols.astype(np.int64)


def _claves_de_grupo(groups, n: int) -> np.ndarray:
    g = np.asarray(groups)
    if g.ndim == 1:
        g = g[:, None]
    if g.ndim != 2 or g.shape[0] != n:
        raise ValueError("groups debe tener una fila por cada fila de X")
    if g.dtype.kind not in "iu":
        raise TypeError(
            "groups debe ser de enteros (por ejemplo simulationRun, o las columnas "
            "faultNumber y simulationRun)"
        )
    if (g < 0).any():
        raise ValueError("groups solo admite enteros no negativos")
    return g.astype(np.int64)


def _preparar(X, sigma_s, noise_type, variables):
    if isinstance(X, pd.DataFrame):
        raise TypeError(
            "X debe ser una matriz de variables de proceso, no un DataFrame: con las "
            "columnas faultNumber, simulationRun y sample el ruido las alteraria. "
            "Usa data.values(df)."
        )
    X = np.asarray(X, dtype=float)
    if X.ndim != 2:
        raise ValueError(f"X debe ser una matriz (muestras, variables), llego {X.ndim}D")
    p = X.shape[1]
    sigma = np.asarray(sigma_s, dtype=float)
    if sigma.shape != (p,):
        raise ValueError(
            f"sigma_s debe tener una entrada por columna de X ({p}), tiene forma {sigma.shape}"
        )
    if not np.all(np.isfinite(sigma)) or (sigma < 0).any():
        raise ValueError("sigma_s debe ser finita y no negativa")
    if noise_type not in NOISE_TYPES:
        raise ValueError(f"noise_type debe ser uno de {NOISE_TYPES}, llego {noise_type!r}")
    return X, sigma, _columnas(variables, p)


def _unitarias(rng: np.random.Generator, forma: tuple, noise_type: str) -> np.ndarray:
    """Ruido blanco de media cero y varianza uno."""
    if noise_type == "gaussian":
        return rng.standard_normal(forma)
    # Uniforme en [-raiz(3), raiz(3)]: varianza uno.
    return rng.uniform(-np.sqrt(3.0), np.sqrt(3.0), size=forma)


# ---------------------------------------------------------------- ruido

def add_noise(X, eta, sigma_s, *, noise_type, seed, variables, groups=None) -> np.ndarray:
    """Contamina X con ruido aditivo de severidad eta segun el lema 1 del articulo.

    Por que existe. El indice compara el error de un modelo a distintas
    severidades de ruido, y la severidad solo significa lo mismo en los tres
    bloques si el ruido se genera igual. El lema 1 (ec. 10, p. 641) dice que,
    para que la variabilidad con ruido sea eta veces la tipica (en varianza),
    cada variable recibe un ruido de desviacion raiz(eta - 1) * sigma_s.

    Devuelve una copia: X no se modifica.

    Parametros
    ----------
    X : array (muestras, variables)
        Solo variables de proceso (por ejemplo `data.values(df)`), en unidades
        fisicas o ya estandarizadas, pero en las mismas que sigma_s. Un
        DataFrame se rechaza, porque traeria las columnas de identificacion.
    eta : float
        Severidad, >= 1. eta = 1 devuelve los datos sin cambios.
    sigma_s : array (variables,)
        Desviacion "tipica" de cada variable, la del proceso sin ruido. Una por
        columna de X. Sin valor por defecto: de que datos sale (por ejemplo, la
        desviacion de los datos normales de entrenamiento, con ddof=1, que es la
        que usa `preprocessing.Scaler`) es una decision. Cuidado: `Scaler.std_`
        cambia por 1 las desviaciones casi nulas, y aqui eso anadiria ruido a una
        variable constante.
    noise_type : {"gaussian", "uniform"}
        "gaussian" es el ruido blanco gaussiano del articulo (p. 639 y caso del
        CSTR, p. 645). [VERIFICAR] la p. 639 lo llama "bounded Gaussian noise";
        aqui no se trunca. "uniform" es una [ADAPTACION] con la misma desviacion.
    seed : int o secuencia de enteros no negativos
        Semilla explicita; con la misma semilla, el mismo ruido. Se usa como
        entropia de `numpy.random.SeedSequence`, asi que (semilla, 3, 7) y
        (semilla, 7, 3) dan ruidos distintos.
    variables : "all" o lista de indices de columna
        Columnas que reciben ruido; las demas quedan intactas. Sin valor por
        defecto: es una decision (el articulo pone ruido en una o dos variables
        del CSTR).
    groups : array de enteros, (muestras,) o (muestras, g), opcional
        Si se pasa, el ruido de cada grupo (por ejemplo cada simulacion, o cada
        par faultNumber y simulationRun) sale de su propia semilla (seed, clave
        del grupo), y no depende del orden de las filas ni de que otros grupos
        haya en X. Asi dos bloques que ordenan distinto la prueba reciben el
        mismo ruido en cada simulacion. Dentro de cada grupo las filas deben ir
        en orden temporal. Sin groups, el ruido depende de la posicion de cada
        fila en X.

    Devuelve
    --------
    array (muestras, variables) de float con el ruido sumado.
    """
    X, sigma, cols = _preparar(X, sigma_s, noise_type, variables)
    eta = float(eta)
    if not np.isfinite(eta) or eta < 1.0:
        raise ValueError(f"eta debe ser >= 1 (lema 1): eta = 1 es el dato sin ruido, llego {eta}")
    entropia = _entropia(seed)
    n = X.shape[0]
    claves = None if groups is None else _claves_de_grupo(groups, n)

    salida = X.copy()
    if eta == 1.0:
        return salida
    escala = np.sqrt(eta - 1.0) * sigma[cols]   # ec. 10

    if claves is None:
        rng = np.random.default_rng(np.random.SeedSequence(entropia))
        salida[:, cols] += _unitarias(rng, (n, cols.size), noise_type) * escala
        return salida

    unicas, inversa = np.unique(claves, axis=0, return_inverse=True)
    inversa = np.asarray(inversa).ravel()
    # Indices de cada grupo, en el orden en que aparecen sus filas en X.
    orden = np.argsort(inversa, kind="stable")
    cortes = np.cumsum(np.bincount(inversa, minlength=len(unicas)))[:-1]
    for clave, idx in zip(unicas, np.split(orden, cortes)):
        semilla = np.random.SeedSequence(entropia + tuple(int(v) for v in clave))
        rng = np.random.default_rng(semilla)
        salida[np.ix_(idx, cols)] += _unitarias(rng, (idx.size, cols.size), noise_type) * escala
    return salida


def noisy_versions(
    X,
    levels,
    sigma_s,
    *,
    noise_type,
    seed,
    n_repeats,
    variables,
    common_random_numbers,
    groups=None,
):
    """Recorre la rejilla de niveles y da una version con ruido de X por nivel y repeticion.

    Por que existe. Para que los tres bloques usen exactamente los mismos datos
    con ruido, la semilla de cada (nivel, repeticion) se deriva de una sola
    semilla declarada, y la rejilla es un parametro obligatorio. Es un
    generador: no guarda en memoria todas las versiones a la vez (con la prueba
    de Rieth no cabrian).

    Parametros
    ----------
    X, sigma_s, noise_type, variables, groups
        Como en `add_noise`.
    levels : secuencia de float
        Rejilla de severidades, estrictamente creciente y empezando en 1 (el
        dato sin ruido forma parte del indice). Sin valor por defecto: el
        articulo recomienda un paso de 1e-3 entre 1 y 10, que es inviable aqui y
        cambia D_p (problemas 2 y 3 del docstring del modulo).
    seed : int o secuencia de enteros
        Semilla base. La de cada version es (seed, r) con numeros aleatorios
        comunes y (seed, i, r) sin ellos, con i el indice del nivel y r el de la
        repeticion.
    n_repeats : int
        Realizaciones del ruido por nivel (>= 1). En eta = 1 todas son iguales
        (no hay ruido), asi que su variabilidad no sirve para el intervalo de la
        ec. 6 en ese nivel: el articulo lo saca de la validacion cruzada (paso
        1). Que se usa como repeticion es una decision.
    common_random_numbers : bool
        True: la misma realizacion base en todos los niveles, escalada por
        raiz(eta - 1); las curvas salen mas suaves y las diferencias entre
        niveles tienen menos ruido de Monte Carlo. False: realizaciones
        independientes en cada nivel, como volver a simular. El articulo no lo
        dice; sin valor por defecto. [ADAPTACION]

    Devuelve
    --------
    Generador de tuplas (i, eta, r, X_con_ruido), recorriendo los niveles de
    menor a mayor y, dentro de cada nivel, las repeticiones.
    """
    X_ok, _, _ = _preparar(X, sigma_s, noise_type, variables)   # falla ya, no al iterar
    if groups is not None:
        _claves_de_grupo(groups, X_ok.shape[0])
    niveles = _comprobar_niveles(levels)
    entropia = _entropia(seed)
    if isinstance(n_repeats, (bool, np.bool_)) or not isinstance(n_repeats, (int, np.integer)):
        raise TypeError("n_repeats debe ser un entero")
    if n_repeats < 1:
        raise ValueError("n_repeats debe ser >= 1")
    if not isinstance(common_random_numbers, (bool, np.bool_)):
        raise TypeError("common_random_numbers debe ser True o False")
    n_repeats = int(n_repeats)
    comunes = bool(common_random_numbers)

    def _generador():
        for i, eta in enumerate(niveles):
            for r in range(n_repeats):
                semilla = entropia + ((r,) if comunes else (i, r))
                yield i, float(eta), r, add_noise(
                    X, eta, sigma_s, noise_type=noise_type, seed=semilla,
                    variables=variables, groups=groups,
                )

    return _generador()


def severity_level(X_noisy, X_clean) -> float:
    """Severidad eta medida con las ecs. 8 y 9: (Psi_X / Psi_S)^2.

    Sirve para comprobar que el ruido anadido tiene la severidad pedida. Psi es
    la suma, sobre las variables, del coeficiente de variacion en porcentaje
    (desviacion / media * 100). Necesita medias positivas: con datos
    estandarizados (media cero) la ec. 8 no esta definida, y hay que usar las
    unidades fisicas. Con ruido en solo algunas variables, la eta de esta
    formula (que suma todas) es menor que la pedida en esas variables.
    """
    Xn = np.asarray(X_noisy, dtype=float)
    Xs = np.asarray(X_clean, dtype=float)
    if Xn.ndim != 2 or Xs.ndim != 2 or Xn.shape[1] != Xs.shape[1]:
        raise ValueError("X_noisy y X_clean deben ser matrices con las mismas columnas")
    if Xn.shape[0] < 2 or Xs.shape[0] < 2:
        raise ValueError("Hacen falta al menos dos muestras para una desviacion")
    mu_n, mu_s = Xn.mean(axis=0), Xs.mean(axis=0)
    if (mu_n <= 0).any() or (mu_s <= 0).any():
        raise ValueError(
            "La ec. 8 usa coeficientes de variacion y necesita medias positivas; con "
            "datos estandarizados no esta definida: usa las unidades fisicas"
        )
    psi_x = np.sum(Xn.std(axis=0, ddof=1) / mu_n * 100.0)
    psi_s = np.sum(Xs.std(axis=0, ddof=1) / mu_s * 100.0)
    return float((psi_x / psi_s) ** 2)


# ---------------------------------------------------------------- error

def error_rate(y_true, y_pred, *, metric, labels=None) -> float:
    """Error E de una evaluacion, con la metrica que se declare: la adaptacion del indice.

    Por que existe. El indice necesita un error entre 0 y 1 por nivel de ruido.
    El articulo usa el de la ec. 3 (p. 639), definido para clasificadores. La
    memoria aplica el indice tambien a los detectores, y dice que esa
    adaptacion se declara antes de ejecutar. Por eso la metrica es un parametro
    sin valor por defecto: elegirla es la adaptacion.

    Opciones de `metric`:

    - "eq3": E = 1 - (1/k) * suma de los aciertos por clase (recall medio), la
      ec. 3 del articulo. No depende de la prevalencia de cada clase. Con un
      detector (k = 2: sana o con fallo) da E = (FAR + (1 - FDR)) / 2, que
      mezcla en un solo numero dos tasas que el protocolo reporta por separado
      (la memoria defiende no resumirlas). Pasa `labels` con las k clases.
    - "far": E = FAR, alarmas entre las muestras sanas (ec. 18, p. 645).
      [ADAPTACION]: el articulo no la usa como E. Para detectores.
    - "missed_detection": E = 1 - FDR, muestras con fallo sin alarma (ec. 19,
      p. 645). [ADAPTACION], para detectores. El informe 06 (seccion 6.1)
      propone calcular el indice por separado sobre la curva de la FAR y sobre
      la de 1 - FDR, y dar el de "eq3" solo como resumen. Cuidado con el
      limite fijo: el ruido infla el estadistico (con uno cuadratico, como el
      SPE o la T2, en promedio), anade alarmas y puede SUBIR la FDR; entonces
      esta curva baja con eta, D_p sale negativo y J puede salir negativo. Con
      limite fijo, esta curva no mide perdida de robustez; con el limite
      recalibrado (FAR fija) si mide cuanta deteccion se pierde.
    - "one_minus_macro_f1": E = 1 - macro-F1, con el F1 de cada clase calculado
      sobre todo el conjunto (pool) de prueba, la metrica de diagnostico del
      TFM (regla 5). [ADAPTACION]: no esta en el articulo. Una clase que no se
      predice nunca tiene F1 = 0, como `f1_score(..., zero_division=0)`.

    Antes de declarar la metrica hay que tener presentes los problemas del
    informe 06 (seccion 6.2), resumidos en el docstring del modulo: el ruido va
    en el canal de medida y no dentro del lazo; con paso uniforme D_p depende
    del paso, asi que J solo se compara con la misma rejilla; el rango [1, 10]
    satura a los detectores con limite fijo, de modo que la rejilla de
    deteccion y la de diagnostico pueden ser distintas y sus J no se comparan;
    y limite fijo o recalibrado miden cosas distintas. Tampoco se mezclan los
    fallos 3, 9 y 15 con el resto: un E sobre todos los fallos juntos esconde
    justo donde se diferencian los metodos (regla 7).

    Parametros
    ----------
    y_true, y_pred : arrays de la misma longitud
        Para "far" y "missed_detection": booleanos (o 0 y 1), True = fallo
        activo en y_true y True = alarma en y_pred (por ejemplo
        `metrics.alarms(estadistico, limite)`). Para "eq3" y
        "one_minus_macro_f1": etiquetas de clase, o booleanos en deteccion. Un
        NaN en un vector de numeros reales es "no evaluado" (contrato del
        paquete): esas muestras hay que quitarlas antes de llamar, y aqui un NaN
        es un error, no una muestra sin alarma.
    metric : str
        Una de ERROR_METRICS. Sin valor por defecto.
    labels : secuencia, opcional
        Clases que entran en "eq3" y "one_minus_macro_f1". Sin ella, las que
        aparecen en y_true. Cada clase de labels tiene que tener muestras en
        y_true (si no, su acierto no esta definido y se lanza un error).

    Devuelve
    --------
    float entre 0 y 1.
    """
    yt = np.asarray(y_true)
    yp = np.asarray(y_pred)
    if yt.ndim != 1 or yp.ndim != 1 or yt.shape != yp.shape:
        raise ValueError("y_true e y_pred deben ser vectores de la misma longitud")
    if yt.size == 0:
        raise ValueError("y_true esta vacio")
    for nombre, v in (("y_true", yt), ("y_pred", yp)):
        if v.dtype.kind == "f" and np.isnan(v).any():
            raise ValueError(
                f"{nombre} tiene NaN, que en el paquete significa 'no evaluado': "
                "quita esas muestras antes de calcular el error"
            )
    if metric not in ERROR_METRICS:
        raise ValueError(f"metric debe ser una de {ERROR_METRICS}, llego {metric!r}")

    if metric in ("far", "missed_detection"):
        for nombre, v in (("y_true", yt), ("y_pred", yp)):
            if v.dtype != bool and not np.isin(v, (0, 1)).all():
                raise ValueError(f"Con metric={metric!r}, {nombre} debe ser booleano (o 0 y 1)")
        yt = yt.astype(bool)
        yp = yp.astype(bool)
        if metric == "far":
            if not (~yt).any():
                raise ValueError("No hay muestras sanas en y_true: la FAR no esta definida")
            return float(yp[~yt].mean())
        if not yt.any():
            raise ValueError("No hay muestras con fallo en y_true: la FDR no esta definida")
        return float(1.0 - yp[yt].mean())

    clases = np.unique(yt) if labels is None else np.asarray(labels)
    if clases.size == 0:
        raise ValueError("labels esta vacio")
    if np.unique(clases).size != clases.size:
        raise ValueError("labels tiene clases repetidas")
    for c in clases:
        if not (yt == c).any():
            raise ValueError(f"La clase {c!r} no tiene muestras en y_true: su acierto no esta definido")

    if metric == "eq3":
        aciertos = [float(np.mean(yp[yt == c] == c)) for c in clases]
        return float(1.0 - np.mean(aciertos))

    # one_minus_macro_f1: F1 = 2 VP / (2 VP + FP + FN), igual a 2PR / (P + R).
    f1 = []
    for c in clases:
        vp = np.sum((yp == c) & (yt == c))
        fp = np.sum((yp == c) & (yt != c))
        fn = np.sum((yp != c) & (yt == c))
        f1.append(2.0 * vp / (2.0 * vp + fp + fn))
    return float(1.0 - np.mean(f1))


# ---------------------------------------------------------------- indice

def _comprobar_errores(errors, n_niveles: int) -> list[np.ndarray]:
    if isinstance(errors, pd.DataFrame):
        raise TypeError(
            "errors debe ser una lista con los errores de cada nivel; con una tabla "
            "larga, agrupa por nivel: [g['error'].to_numpy() for _, g in df.groupby('level')]"
        )
    filas = list(errors)        # un array 2D da sus filas; una lista, sus elementos
    if len(filas) != n_niveles:
        raise ValueError(
            f"errors debe tener una entrada por nivel ({n_niveles}), tiene {len(filas)}"
        )
    muestras = []
    for i, fila in enumerate(filas):
        e = np.atleast_1d(np.asarray(fila, dtype=float)).ravel()
        if e.size == 0:
            raise ValueError(f"El nivel {i} no tiene errores")
        if not np.all(np.isfinite(e)):
            raise ValueError(f"El nivel {i} tiene errores NaN o infinitos")
        if (e < 0).any() or (e > 1).any():
            raise ValueError(
                f"Los errores del nivel {i} se salen de [0, 1] (ec. 3); si vienen en "
                "porcentaje, dividelos entre 100"
            )
        muestras.append(e)
    return muestras


def _trapecio(x: np.ndarray, y: np.ndarray) -> float:
    """Regla del trapecio compuesta: suma de (x_i - x_(i-1)) * (y_(i-1) + y_i) / 2.

    Con paso constante es identica a la forma de las ecs. 4 y 5,
    (Delta_eta / 2) * [y_0 + 2 * suma de los interiores + y_n].
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.sum(np.diff(x) * (y[1:] + y[:-1]) / 2.0))


def robustness_curve(levels, errors, *, confidence) -> pd.DataFrame:
    """Curva de error frente a la severidad, con el intervalo t de la ec. 6 en cada nivel.

    Parametros
    ----------
    levels : secuencia de float
        Rejilla de severidades, estrictamente creciente y empezando en 1.
    errors : secuencia con una entrada por nivel
        Los n_e errores observados en cada nivel (uno por repeticion), entre 0 y
        1. Pueden ser listas, arrays o un array (niveles, n_e). Con un solo
        error por nivel no hay intervalo (columnas NaN).
    confidence : float
        Nivel de confianza (1 - alfa) del intervalo, entre 0 y 1. Sin valor por
        defecto (el articulo usa el 99 % en el caso del CSTR, p. 645).

    Intervalo (ec. 6, p. 640): media +- t * s / raiz(n_e), con s la desviacion
    muestral de los n_e errores (ddof = 1) y t el cuantil 1 - alfa/2 de una t de
    Student con n_e - 1 grados de libertad. [VERIFICAR] el articulo dice "sample
    standard deviation corresponding to the mean error" y no da los grados de
    libertad; esta es la lectura estandar. El articulo pide n_e < 30; aqui se
    admite cualquier n_e >= 2 [ADAPTACION]: con mas, el cuantil t se acerca al
    normal y el intervalo sigue siendo valido. Los extremos no se recortan a
    [0, 1], porque el articulo no lo dice [VERIFICAR].

    Devuelve
    --------
    DataFrame con una fila por nivel: `level`, `n_e`, `mean_error`,
    `std_error`, `t_quantile`, `e_min`, `e_max`.
    """
    niveles = _comprobar_niveles(levels)
    muestras = _comprobar_errores(errors, niveles.size)
    if confidence is None or isinstance(confidence, (bool, np.bool_)):
        raise TypeError("confidence es obligatorio: el nivel de confianza del intervalo")
    conf = float(confidence)
    if not 0.0 < conf < 1.0:
        raise ValueError(f"confidence debe estar entre 0 y 1, llego {confidence}")
    alfa = 1.0 - conf

    filas = []
    for eta, e in zip(niveles, muestras):
        n_e = int(e.size)
        media = float(e.mean())
        if n_e >= 2:
            s = float(e.std(ddof=1))
            t = float(stats.t.ppf(1.0 - alfa / 2.0, n_e - 1))
            margen = t * s / np.sqrt(n_e)
        else:
            s = t = margen = float("nan")
        filas.append({
            "level": float(eta),
            "n_e": n_e,
            "mean_error": media,
            "std_error": s,
            "t_quantile": t,
            "e_min": media - margen,
            "e_max": media + margen,
        })
    return pd.DataFrame(filas)


def robustness_index(levels, errors, *, confidence) -> dict:
    """Indice de robustez J_RIL de Bernal-de Lazaro et al. (2018) y sus tres componentes.

    Calcula, sobre la curva de `robustness_curve`:

    - I_yy (ec. 4): trapecio de eta^2 * E_media. Pesa mas el error con mucho
      ruido.
    - D_p (ec. 7): suma de las pendientes de E_media entre niveles
      consecutivos, incluido el primer tramo (eta_0 -> eta_1), que es lo que
      reproduce las Tablas 1 a 3 del articulo. Las pendientes llevan signo: si
      el error baja entre dos niveles, resta. [VERIFICAR] la ec. 7 escribe E
      sin barra; aqui se usa la media por nivel, como en la ec. 4. Con paso
      uniforme, D_p = (E_ultimo - E_primero) / paso: depende del paso.
    - Delta_I (ecs. 5 y 6): trapecio de E_max menos trapecio de E_min, sin el
      peso eta^2. Necesita al menos dos errores en cada nivel; si no, vale NaN
      y J_RIL tambien (el articulo omite Delta_I en sus ejemplos de juguete,
      p. 642; si se quiere ese resumen, se suma i_yy + d_p a mano y se dice).

    Las ecs. 4 y 5 suponen paso constante; con una rejilla de paso variable se
    usa la regla del trapecio general, que con paso constante da lo mismo
    [ADAPTACION]. `grid_uniform` dice cual fue el caso.

    J solo se compara entre metodos evaluados con la MISMA rejilla, el mismo
    tipo de ruido, la misma metrica de error y el mismo nivel de confianza
    (problema 2 del docstring del modulo). Menor J = mas robusto. Para el
    formato comun, anade a la fila devuelta `method`, la metrica de
    `error_rate`, el tipo de ruido, la semilla, las variables con ruido y la
    rejilla: sin eso, el numero no se puede interpretar.

    Parametros
    ----------
    levels, errors, confidence
        Como en `robustness_curve`.

    Devuelve
    --------
    dict con `j_ril`, `i_yy`, `d_p`, `delta_i`, `i0_max`, `i0_min`,
    `n_levels`, `level_min`, `level_max`, `grid_uniform`, `delta_eta` (NaN si
    el paso no es uniforme), `confidence`, `n_e_min` y `n_e_max`.
    """
    curva = robustness_curve(levels, errors, confidence=confidence)
    eta = curva["level"].to_numpy()
    e = curva["mean_error"].to_numpy()

    i_yy = _trapecio(eta, eta**2 * e)                    # ec. 4
    d_p = float(np.sum(np.diff(e) / np.diff(eta)))       # ec. 7, todos los tramos
    i0_max = _trapecio(eta, curva["e_max"].to_numpy())   # ec. 5 con E_max
    i0_min = _trapecio(eta, curva["e_min"].to_numpy())   # ec. 5 con E_min
    delta_i = i0_max - i0_min

    pasos = np.diff(eta)
    uniforme = bool(np.allclose(pasos, pasos[0], rtol=1e-9, atol=0.0))
    return {
        "j_ril": i_yy + d_p + delta_i,                   # ec. 2
        "i_yy": i_yy,
        "d_p": d_p,
        "delta_i": delta_i,
        "i0_max": i0_max,
        "i0_min": i0_min,
        "n_levels": int(eta.size),
        "level_min": float(eta[0]),
        "level_max": float(eta[-1]),
        "grid_uniform": uniforme,
        "delta_eta": float(pasos[0]) if uniforme else float("nan"),
        "confidence": float(confidence),
        "n_e_min": int(curva["n_e"].min()),
        "n_e_max": int(curva["n_e"].max()),
    }
