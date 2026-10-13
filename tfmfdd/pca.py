"""Modelo de monitorizacion basado en PCA.

La idea, antes del formalismo. Un proceso quimico tiene decenas de sensores
correlacionados: cuando sube el caudal de alimentacion suben la presion y la
temperatura, y esa relacion se mantiene mientras la planta opere con normalidad.
El PCA aprende ese patron a partir de datos sanos y lo resume en unas pocas
direcciones de maxima varianza, las componentes principales.

Cuando llega una observacion nueva se la mide de dos formas complementarias:

- T2 de Hotelling: cuanto se aleja del centro DENTRO del subespacio del modelo.
  Responde a "el proceso sigue el patron conocido, pero ¿esta operando en una
  zona anormal de ese patron?".
- SPE (tambien llamado Q): cuanto de la observacion NO explica el modelo, es
  decir el tamano del residuo. Responde a "¿es que ya ni siquiera sigue el
  patron?".

Un fallo puede disparar uno, otro o los dos, y saber cual da la primera pista
para el diagnostico. Yin et al. (2012) usan los dos a la vez: hay alarma si
cualquiera supera su limite (`PCAMonitor.combined_index`).

Con DPCA (lags > 0) cada fila del modelo apila una muestra con las `lags`
anteriores de su misma trayectoria. `score` devuelve los estadisticos alineados
con las muestras originales (NaN en las `lags` primeras de cada trayectoria),
para que el inicio del fallo sea el mismo en PCA y en DPCA (MET-1).
"""

from __future__ import annotations

import copy
import warnings

import numpy as np

from . import limits as lim
from .preprocessing import Scaler, lag_by_run

#: Limites del T2 que acepta PCAMonitor (parametro t2_method).
T2_METHODS = ("f", "beta", "empirical", "kde")
#: Limites del SPE que acepta PCAMonitor (parametro spe_method).
SPE_METHODS = ("box", "jackson", "kde", "empirical")
#: Modos de calibracion que acepta PCAMonitor.fit (parametro calibration).
CALIBRATIONS = ("holdout", "in_sample")

# Valores de PCAMonitor.limit_source_, por estadistico:
#   "parametric_fit": formula parametrica con cantidades del ajuste (F o Beta con
#                     el numero de filas del ajuste; Jackson con sus autovalores).
#   "holdout":        calibrado con el estadistico de X_cal, datos reservados.
#   "in_sample":      calibrado con el estadistico de los datos del ajuste.


def _rows_in(A: np.ndarray, B: np.ndarray) -> np.ndarray:
    """Mascara booleana: que filas de A aparecen, identicas, entre las de B."""
    A = np.ascontiguousarray(A, dtype=float)
    B = np.ascontiguousarray(B, dtype=float)
    vistas = {fila.tobytes() for fila in B}
    return np.array([fila.tobytes() in vistas for fila in A], dtype=bool)


def _scaler_rows(scaler) -> int | None:
    """Filas con que se ajusto un escalador, si lo dice (Scaler o StandardScaler)."""
    n = getattr(scaler, "n_samples_fit_", None)
    if n is None:
        n = getattr(scaler, "n_samples_seen_", None)   # StandardScaler de scikit-learn
    if n is None:
        return None
    n = np.asarray(n)
    if n.size == 0 or not (n == n.flat[0]).all():
        return None
    return int(n.flat[0])


class PCAMonitor:
    """Monitor PCA o DPCA con estadisticos T2 y SPE.

    Parametros
    ----------
    n_components : int o None
        Numero de componentes a retener. Si es None se eligen por varianza
        acumulada segun `variance`.
    variance : float
        Varianza acumulada objetivo cuando n_components es None. El valor por
        defecto es el de `pca.variance` en configs/baseline.yaml, que es la unica
        fuente de los valores por defecto (decisiones.md, 22/09); una prueba
        comprueba que coinciden.
    alpha : float
        Nivel de confianza de los limites de control, el mismo para T2 y SPE.
    lags : int
        0 para PCA estatico, 1 o 2 para DPCA. Con lags > 0, pasa a fit y a
        score el identificador de trayectoria de cada fila (`runs`), para no
        mezclar el final de una simulacion con el principio de otra. Sin runs, X
        se trata como UNA sola trayectoria (con un aviso). Si juntas ficheros o
        clases, pasa las dos columnas (`df[["faultNumber",
        "simulationRun"]].to_numpy()`): en el TEP clasico simulationRun vale 1
        en todos los ficheros (ver preprocessing.trajectory_starts). Con
        `samples` (la columna `sample`), fit y score comprueban ademas que las
        muestras de cada trayectoria sean consecutivas antes de apilar.
    spe_method : {'box', 'jackson', 'kde', 'empirical'}
        'jackson' es parametrico con los autovalores del ajuste (no usa X_cal);
        los otros tres se calibran con datos: los de X_cal si se pasa, o los del
        propio ajuste.
    t2_method : {'f', 'beta', 'empirical', 'kde'}
        'f' (por defecto, el de siempre): limite de Fase II, para datos nuevos.
        'beta': limite de Fase I, el que corresponde a las propias muestras del
        ajuste; aplicado a datos nuevos reproduce la calibracion en muestra en
        version parametrica (DIS-5). Los dos usan el numero de filas del ajuste y
        no usan X_cal. 'empirical' y 'kde' se calibran con datos, como el SPE.
    quantile_method : {'linear', 'conformal'}
        Estimador del percentil para los limites 'empirical' (ver
        `limits.limit_empirical`). 'linear' es el de siempre.

    Atributos tras fit
    ------------------
    t2_limit_, spe_limit_ : float
    limit_source_ : dict
        De donde sale cada limite: {"t2": ..., "spe": ...} con valores
        "parametric_fit", "holdout" o "in_sample" (ver arriba del modulo).
    calibrated_on_holdout_ : bool
        True si algun limite se calibro con X_cal. Para el detalle por
        estadistico, mira limit_source_.
    n_fit_ : int
        Filas del ajuste en el espacio del modelo (con DPCA, ya sin las `lags`
        primeras de cada trayectoria). Es la m de los limites F y Beta.
    n_cal_ : int o None
        Valores del estadistico con que se calibraron los limites con datos;
        None si los dos limites son parametricos.
    scaler_source_ : {'fit', 'external'}
        'fit' si el escalador se ajusto con X; 'external' si llego ya ajustado.
        El externo se guarda como copia: reajustar despues el objeto que se
        paso no cambia el modelo.
    scaler_n_samples_fit_ : int o None
        Filas con que se ajusto el escalador del modelo: con el interno, las del
        ajuste (con DPCA, ya con retardos); con el externo, lo que el escalador
        diga (`n_samples_fit_` de Scaler o `n_samples_seen_` de StandardScaler),
        o None si no lo dice. Sirve para comprobar en un script que el brazo
        limpio del factor 2 se escalo solo con el ajuste.
    n_features_in_ : int
        Variables originales de X (sin contar los retardos).
    """

    def __init__(
        self,
        n_components: int | None = None,
        variance: float = 0.85,
        alpha: float = 0.99,
        lags: int = 0,
        spe_method: str = "box",
        t2_method: str = "f",
        quantile_method: str = "linear",
    ) -> None:
        self.n_components = n_components
        self.variance = variance
        self.alpha = alpha
        self.lags = lags
        self.spe_method = spe_method
        self.t2_method = t2_method
        self.quantile_method = quantile_method

        self.scaler_ = Scaler()
        self.scaler_source_: str | None = None
        self.scaler_n_samples_fit_: int | None = None
        self.loadings_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.t2_limit_: float | None = None
        self.spe_limit_: float | None = None
        self.n_components_: int | None = None
        self.limit_source_: dict | None = None
        self.calibrated_on_holdout_: bool | None = None
        self.n_fit_: int | None = None
        self.n_cal_: int | None = None
        self.n_features_in_: int | None = None

    # ------------------------------------------------------------ utilidades

    def _validate_params(self) -> None:
        if not 0.0 < float(self.alpha) < 1.0:
            raise ValueError(
                f"alpha es el nivel de confianza (por ejemplo 0.99), entre 0 y 1; llego {self.alpha}"
            )
        if self.n_components is None and not 0.0 < float(self.variance) <= 1.0:
            raise ValueError(f"variance debe estar en (0, 1]; llego {self.variance}")
        if int(self.lags) != self.lags or self.lags < 0:
            raise ValueError(f"lags debe ser un entero >= 0; llego {self.lags}")
        if self.spe_method not in SPE_METHODS:
            raise ValueError(f"spe_method desconocido: {self.spe_method}")
        if self.t2_method not in T2_METHODS:
            raise ValueError(f"t2_method desconocido: {self.t2_method}")
        if self.quantile_method not in lim.EMPIRICAL_METHODS:
            raise ValueError(f"quantile_method desconocido: {self.quantile_method}")

    @staticmethod
    def _matrix(X, que: str, n_cols: int | None = None) -> np.ndarray:
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"{que} debe ser una matriz 2D (muestras, variables); llego {X.ndim}D")
        if n_cols is not None and X.shape[1] != n_cols:
            raise ValueError(f"{que} tiene {X.shape[1]} variables y el modelo {n_cols}")
        return X

    def _warn_single_run(self, que: str) -> None:
        warnings.warn(
            f"lags = {self.lags} sin {que}: la matriz se trata como UNA sola "
            "trayectoria contigua. Si junta varias simulaciones, pasa el "
            "identificador de cada fila (o las columnas faultNumber y "
            "simulationRun) para no mezclar el final de una con el principio de "
            "la siguiente.",
            UserWarning,
            stacklevel=3,
        )

    def _lagged(self, X: np.ndarray, runs, samples=None) -> tuple[np.ndarray, np.ndarray]:
        """Matriz con retardos y posicion en X de la muestra actual de cada fila."""
        if self.lags == 0:
            return X, np.arange(len(X))
        if runs is None:
            runs = np.zeros(len(X), dtype=int)
        matriz, _, posiciones = lag_by_run(
            X, runs, int(self.lags), return_index=True, samples=samples
        )
        return matriz, posiciones

    def _external_transform(self, X: np.ndarray, scaler=None) -> np.ndarray:
        esc = self.scaler_ if scaler is None else scaler
        Z = np.asarray(esc.transform(X), dtype=float)
        if Z.shape != X.shape:
            raise ValueError(
                f"El escalador externo devolvio una matriz {Z.shape} para una entrada "
                f"{X.shape}: tiene que estar ajustado sobre las variables originales"
            )
        return Z

    def _model_space(self, X: np.ndarray, runs, samples=None) -> tuple[np.ndarray, np.ndarray]:
        """Escala y apila retardos con el escalador ya fijado en fit."""
        if self.scaler_source_ == "external":
            # El escalador externo trabaja sobre las variables originales: se
            # escala primero y despues se apilan los retardos.
            return self._lagged(self._external_transform(X), runs, samples)
        Xp, posiciones = self._lagged(X, runs, samples)
        return self.scaler_.transform(Xp), posiciones

    # ------------------------------------------------------------------- fit

    def fit(
        self,
        X: np.ndarray,
        runs: np.ndarray | None = None,
        X_cal: np.ndarray | None = None,
        runs_cal: np.ndarray | None = None,
        *,
        calibration: str | None = None,
        scaler=None,
        samples: np.ndarray | None = None,
        samples_cal: np.ndarray | None = None,
    ) -> "PCAMonitor":
        """Ajusta el modelo. X debe contener SOLO operacion normal.

        Parametros
        ----------
        X, runs
            Datos de ajuste y sus identificadores de trayectoria (ver lags).
        X_cal, runs_cal
            Datos normales RESERVADOS para calibrar los limites de control. Si
            se omiten, los limites que se calibran con datos (SPE por box, kde o
            empirical; T2 por empirical o kde) se calculan sobre los mismos datos
            del ajuste, que es lo que hace buena parte de la literatura.

            El motivo para reservar es el mecanismo. El modelo se ha ajustado
            para minimizar el residuo sobre X, asi que el SPE que observa en X
            tiende a salir mas pequeno que en datos que no vio, y un umbral
            calibrado con ese SPE describe los datos del ajuste, no los nuevos.
            Que eso cambie la tasa de falsas alarmas, cuanto y en que sentido
            depende del limite, del tamano del ajuste y de los datos: el signo
            del efecto no se presupone, se mide y se reporta (memoria, 2.2.4).
            Las cifras NO se escriben aqui: salen de
            `experiments/A0_calibracion_limites.py` y viven en su CSV con la
            configuracion completa (variables, componentes, alpha, metodo,
            tamanos de ajuste y calibracion). Los limites parametricos (T2 por F
            o Beta, SPE por Jackson-Mudholkar) no usan X_cal, pero salen de
            cantidades del ajuste: el F es el de Fase II, pensado para datos
            nuevos; el Beta (Fase I) y Jackson son formas parametricas de
            calibrar en muestra (DIS-5, DIS-6). `limit_source_` dice de donde
            sale cada limite.

            Calibrar con los datos del ajuste no es una fuga, porque no usa la
            prueba: es un sesgo, una decision que mueve la metrica sin usar el
            conjunto de prueba. Calibrar con la prueba si seria una fuga.
        calibration : {None, 'holdout', 'in_sample'}
            None (por defecto): reservado si llega X_cal; si no, en muestra, con
            un aviso cuando algun limite se calibra con datos.
            'in_sample': calibrar en muestra a proposito y sin aviso, como en la
            verificacion frente a Yin et al. (2012), que calibra con los mismos
            datos del ajuste. Si ademas se pasa X_cal, sus filas tienen que
            salir de X (por ejemplo, un subconjunto de X del mismo tamano que el
            X_cal reservado de otro brazo, para igualar n_cal, DIS-4).
            'holdout': exige X_cal.
            En los dos modos con X_cal se comprueba lo que se declara: un X_cal
            "reservado" no puede compartir filas con X, y uno "en muestra" no
            puede traer filas que no esten en X.
        scaler : objeto con transform, ya ajustado, o None
            Gancho para el factor 2 (estandarizacion con datos de prueba). None:
            el escalador se ajusta con X, como siempre. Si llega uno ya ajustado
            (por ejemplo `Scaler().fit(np.vstack([X, X_prueba]))`), se usa tal
            cual, sin reajustar, sobre las variables originales y antes de
            apilar retardos; el modelo no vuelve a centrar ni a escalar. Sin este
            gancho el escalador interno deshace cualquier escalado previo y el
            factor 2 sale nulo por construccion en PCA (DIS-7). Para comparar
            brazos con DPCA, usa el gancho en los dos (en el limpio, con un
            escalador ajustado solo con X), porque sin gancho el escalado se
            ajusta sobre la matriz con retardos. Este gancho es el brazo S1
            (ajuste y prueba juntos); el S2 (cada simulacion de prueba con sus
            propios estadisticos) comparte el modelo con el limpio y va en
            `score(..., scaler=...)`.
            El modelo guarda una COPIA del escalador (V-01 de la verificacion
            del 12/10): si despues se reajusta el mismo objeto para otro brazo,
            este modelo no cambia. `scaler_n_samples_fit_` dice con cuantas
            filas se ajusto.
        samples, samples_cal : array o None
            Opcional, con lags > 0: la columna `sample` de X y de X_cal. Si
            llegan, se comprueba que las muestras de cada trayectoria sean
            consecutivas antes de apilar retardos (ver
            preprocessing.lag_by_run). Con lags = 0 no se usan.
        """
        self._validate_params()
        if calibration not in (None,) + CALIBRATIONS:
            raise ValueError(f"calibration debe ser None, 'holdout' o 'in_sample'; llego {calibration!r}")

        X = self._matrix(X, "X")
        if not np.isfinite(X).all():
            raise ValueError("X tiene valores no finitos")
        if self.lags > 0 and runs is None:
            self._warn_single_run("runs")

        if X_cal is not None:
            X_cal = self._matrix(X_cal, "X_cal", n_cols=X.shape[1])
            if len(X_cal) == 0:
                raise ValueError(
                    "X_cal no tiene filas. Para calibrar en muestra a proposito usa "
                    "calibration='in_sample' sin X_cal; para calibrar con reserva, "
                    "pasa datos normales que el modelo no vea."
                )
            if not np.isfinite(X_cal).all():
                raise ValueError("X_cal tiene valores no finitos")
            if self.lags > 0 and runs_cal is None:
                self._warn_single_run("runs_cal")
        if calibration == "holdout" and X_cal is None:
            raise ValueError("calibration='holdout' necesita X_cal con datos normales reservados")

        modo = calibration or ("holdout" if X_cal is not None else "in_sample")
        if X_cal is not None:
            compartidas = _rows_in(X_cal, X)
            if modo == "holdout" and compartidas.any():
                raise ValueError(
                    f"X_cal comparte {int(compartidas.sum())} de sus {len(X_cal)} filas con X: "
                    "no es un conjunto reservado. Si quieres calibrar en muestra a "
                    "proposito, pasa calibration='in_sample'."
                )
            if modo == "in_sample" and not compartidas.all():
                raise ValueError(
                    f"calibration='in_sample' con X_cal exige que sus filas salgan de X, "
                    f"y {int((~compartidas).sum())} de {len(X_cal)} no estan en X"
                )

        # Escalado y retardos.
        if scaler is None:
            Xp, _ = self._lagged(X, runs, samples)
            self.scaler_ = Scaler().fit(Xp)
            self.scaler_source_ = "fit"
            self.scaler_n_samples_fit_ = self.scaler_.n_samples_fit_
            Z = self.scaler_.transform(Xp)
        else:
            if not hasattr(scaler, "transform"):
                raise TypeError("scaler debe ser un escalador ya ajustado, con metodo transform")
            # Copia: el objeto que llega puede reajustarse despues para otro
            # brazo, y este modelo no debe enterarse (V-01).
            try:
                self.scaler_ = copy.deepcopy(scaler)
            except Exception as exc:
                raise TypeError(
                    f"No se pudo copiar el escalador externo ({exc}). El modelo guarda una "
                    "copia para que reajustar el original no lo cambie."
                ) from exc
            self.scaler_source_ = "external"
            self.scaler_n_samples_fit_ = _scaler_rows(self.scaler_)
            Z, _ = self._lagged(self._external_transform(X), runs, samples)

        self.n_features_in_ = X.shape[1]
        self.n_fit_ = len(Z)
        if self.n_fit_ < 2:
            raise ValueError(f"Hacen falta al menos 2 filas para ajustar; quedan {self.n_fit_}")

        # Descomposicion de la matriz de covarianza.
        cov = np.cov(Z, rowvar=False)
        autoval, autovec = np.linalg.eigh(cov)
        orden = np.argsort(autoval)[::-1]
        autoval, autovec = autoval[orden], autovec[:, orden]
        autoval = np.clip(autoval, 0.0, None)
        self.eigenvalues_ = autoval

        if self.n_components is None:
            acumulada = np.cumsum(autoval) / autoval.sum()
            self.n_components_ = min(int(np.searchsorted(acumulada, self.variance) + 1), Z.shape[1])
        else:
            self.n_components_ = int(self.n_components)
        if not 1 <= self.n_components_ <= Z.shape[1]:
            raise ValueError(
                f"n_components = {self.n_components_} fuera de rango: el modelo tiene "
                f"{Z.shape[1]} variables"
            )

        self.loadings_ = autovec[:, : self.n_components_]

        # Estadisticos sobre los que se calibran los limites con datos: los del
        # conjunto reservado si existe, y si no los del propio ajuste.
        if X_cal is not None:
            Zc, _ = self._model_space(X_cal, runs_cal, samples_cal)
        else:
            Zc = Z
        origen = "holdout" if modo == "holdout" else "in_sample"
        t2_c, spe_c = self._statistics(Zc)

        fuentes = {}
        if self.t2_method == "f":
            self.t2_limit_ = lim.t2_limit(self.n_components_, self.n_fit_, self.alpha)
            fuentes["t2"] = "parametric_fit"
        elif self.t2_method == "beta":
            self.t2_limit_ = lim.t2_limit_beta(self.n_components_, self.n_fit_, self.alpha)
            fuentes["t2"] = "parametric_fit"
        elif self.t2_method == "empirical":
            self.t2_limit_ = lim.limit_empirical(t2_c, self.alpha, method=self.quantile_method)
            fuentes["t2"] = origen
        else:
            self.t2_limit_ = lim.limit_kde(t2_c, self.alpha)
            fuentes["t2"] = origen

        if self.spe_method == "box":
            self.spe_limit_ = lim.spe_limit_box(spe_c, self.alpha)
            fuentes["spe"] = origen
        elif self.spe_method == "jackson":
            self.spe_limit_ = lim.spe_limit_jackson(autoval, self.n_components_, self.alpha)
            fuentes["spe"] = "parametric_fit"
        elif self.spe_method == "kde":
            self.spe_limit_ = lim.limit_kde(spe_c, self.alpha)
            fuentes["spe"] = origen
        else:
            self.spe_limit_ = lim.limit_empirical(spe_c, self.alpha, method=self.quantile_method)
            fuentes["spe"] = origen

        for nombre, valor in (("T2", self.t2_limit_), ("SPE", self.spe_limit_)):
            if not np.isfinite(valor):
                raise ValueError(f"El limite del {nombre} sale {valor}; revisa la configuracion")

        self.limit_source_ = fuentes
        self.calibrated_on_holdout_ = "holdout" in fuentes.values()
        con_datos = [k for k, v in fuentes.items() if v != "parametric_fit"]
        self.n_cal_ = len(Zc) if con_datos else None

        if calibration is None and X_cal is None and con_datos:
            warnings.warn(
                "Limites de " + " y ".join(k.upper() for k in con_datos) + " calibrados "
                "con los mismos datos del ajuste (en muestra) porque no llego X_cal. "
                "Pasa X_cal con datos normales reservados, o calibration='in_sample' "
                "si es a proposito (por ejemplo, la verificacion frente a Yin).",
                UserWarning,
                stacklevel=2,
            )
        if X_cal is not None and not con_datos:
            warnings.warn(
                f"X_cal no interviene en ningun limite: t2_method={self.t2_method!r} y "
                f"spe_method={self.spe_method!r} son parametricos con los datos del "
                "ajuste. limit_source_ lo deja dicho.",
                UserWarning,
                stacklevel=2,
            )

        return self

    # ----------------------------------------------------------- estadisticos

    def _statistics(self, Z: np.ndarray):
        P = self.loadings_
        lam = self.eigenvalues_[: self.n_components_]
        lam = np.clip(lam, 1e-12, None)

        T = Z @ P                       # puntuaciones en el subespacio del modelo
        t2 = np.sum(T**2 / lam, axis=1)  # distancia normalizada por la varianza

        Z_rec = T @ P.T                  # reconstruccion
        residuo = Z - Z_rec
        spe = np.sum(residuo**2, axis=1)

        return t2, spe

    def score(
        self,
        X: np.ndarray,
        runs: np.ndarray | None = None,
        *,
        scaler=None,
        samples: np.ndarray | None = None,
    ):
        """Devuelve (T2, SPE) para datos nuevos.

        Una fila por fila de X y en el mismo orden. Con lags > 0, las `lags`
        primeras filas de cada trayectoria contigua no tienen historia
        suficiente y van con NaN ("no evaluada", no "sin alarma"). Asi el inicio
        del fallo es el mismo en PCA y DPCA: `metrics.evaluate_run(t2, limite,
        onset=160)` vale para los dos (MET-1). Sin runs y con lags > 0, X se
        trata como una sola trayectoria (con un aviso).

        Parametros
        ----------
        X, runs
            Datos nuevos y sus identificadores de trayectoria (ver lags).
        scaler : objeto con transform, ya ajustado, o None
            None (por defecto): el escalador del ajuste, como siempre. Si llega
            uno, sustituye al del ajuste SOLO en esta llamada: el modelo no
            cambia (cargas, autovalores, limites y `scaler_` siguen siendo los
            del ajuste). Existe para el brazo S2 del factor 2, estandarizar cada
            simulacion de prueba con sus propios estadisticos, como hicieron
            Deng et al. (2021): ese brazo comparte el modelo con el limpio y
            solo cambia como se escala la prueba, algo que el gancho de `fit`
            no puede expresar. Se aplica como ese gancho: sobre las variables
            originales y antes de apilar retardos, sin volver a centrar. Con
            lags > 0, el modelo tiene que haberse ajustado con el gancho
            (`fit(X_fit, ..., scaler=Scaler().fit(X_fit))`); sin el, su
            escalador vive en la matriz con retardos y los dos brazos no
            escalarian igual.
            `ablation.score_standardized_by_run` hace el brazo S2 entero, una
            trayectoria con sus propios estadisticos cada vez.
        samples : array o None
            Opcional, con lags > 0: la columna `sample` de X, para comprobar que
            las muestras de cada trayectoria son consecutivas antes de apilar
            retardos (ver preprocessing.lag_by_run).
        """
        if self.loadings_ is None:
            raise RuntimeError("Llama a fit() antes de score().")
        X = self._matrix(X, "X", n_cols=self.n_features_in_)
        if scaler is not None:
            if not hasattr(scaler, "transform"):
                raise TypeError("scaler debe ser un escalador ya ajustado, con metodo transform")
            if self.lags > 0 and self.scaler_source_ != "external":
                raise ValueError(
                    "score(..., scaler=...) con lags > 0 necesita un modelo ajustado con el "
                    "gancho del escalador: fit(..., scaler=Scaler().fit(X_fit)). Sin el, el "
                    "escalador del ajuste trabaja sobre la matriz con retardos y el sustituto "
                    "sobre las variables originales, y los dos brazos no escalarian igual."
                )
        if self.lags > 0 and runs is None and not getattr(self, "_avisado_score_sin_runs", False):
            # Una vez por modelo: un bucle sobre ficheros repetiria el mismo aviso.
            self._avisado_score_sin_runs = True
            self._warn_single_run("runs")

        if scaler is None:
            Z, posiciones = self._model_space(X, runs, samples)
        else:
            Z, posiciones = self._lagged(self._external_transform(X, scaler), runs, samples)
        t2_m, spe_m = self._statistics(Z)
        if self.lags == 0:
            return t2_m, spe_m

        t2 = np.full(len(X), np.nan)
        spe = np.full(len(X), np.nan)
        t2[posiciones] = t2_m
        spe[posiciones] = spe_m
        return t2, spe

    def combined_index(self, t2: np.ndarray, spe: np.ndarray) -> np.ndarray:
        """Indice de la regla de alarma T2 o SPE, normalizado con limite 1.

        Yin et al. (2012, pp. 1568 y 1572) declaran fallo si CUALQUIERA de los
        dos estadisticos supera su limite. Este indice vale

            max(T2 / limite_T2, SPE / limite_SPE)

        y supera 1 exactamente cuando hay alarma en T2 o en SPE, asi que
        `metrics.evaluate_run(indice, 1.0, onset)` mide la disyuncion, retardo
        incluido. Un NaN en cualquiera de los dos (muestra no evaluada) da NaN.

        Con los dos limites al nivel alpha, la FAR nominal de la disyuncion no
        baja de 1 - alpha y en general la supera (es la probabilidad de una
        union de dos sucesos). Si se quiere un nivel global, el nivel por
        estadistico es una decision del protocolo (por ejemplo, Sidak:
        alpha_por_estadistico = alpha_global ** (1/2), informe 06, 3.3).
        """
        if self.t2_limit_ is None:
            raise RuntimeError("Llama a fit() antes de combined_index().")
        t2 = np.asarray(t2, dtype=float)
        spe = np.asarray(spe, dtype=float)
        if t2.shape != spe.shape:
            raise ValueError(f"T2 {t2.shape} y SPE {spe.shape} tienen formas distintas")
        return np.maximum(t2 / self.t2_limit_, spe / self.spe_limit_)

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """Contribucion de cada variable al SPE de una observacion.

        Es la herramienta clasica de aislamiento: la variable que mas contribuye
        al residuo es la primera sospechosa. No identifica la causa raiz, porque
        un fallo se propaga por el proceso, pero acota donde mirar. Con lags > 0,
        x es una fila de la matriz con retardos.
        """
        if self.loadings_ is None:
            raise RuntimeError("Llama a fit() antes de contributions().")
        if self.scaler_source_ == "external" and self.lags > 0:
            raise NotImplementedError(
                "contributions con escalador externo y lags > 0 no esta implementado"
            )
        z = self.scaler_.transform(np.atleast_2d(x))
        residuo = z - (z @ self.loadings_) @ self.loadings_.T
        return (residuo**2).ravel()
