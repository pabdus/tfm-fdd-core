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
para el diagnostico.
"""

from __future__ import annotations

import numpy as np

from . import limits as lim
from .preprocessing import Scaler, lag_by_run


class PCAMonitor:
    """Monitor PCA o DPCA con estadisticos T2 y SPE.

    Parametros
    ----------
    n_components : int o None
        Numero de componentes a retener. Si es None se eligen por varianza
        acumulada segun `variance`.
    variance : float
        Varianza acumulada objetivo cuando n_components es None.
    alpha : float
        Nivel de confianza de los limites de control.
    lags : int
        0 para PCA estatico, 1 o 2 para DPCA. Con lags > 0 hay que pasar el
        vector de simulaciones a fit y a score, para no mezclar el final de una
        con el principio de otra.
    spe_method : {'box', 'jackson', 'kde', 'empirical'}
    """

    def __init__(
        self,
        n_components: int | None = None,
        variance: float = 0.90,
        alpha: float = 0.99,
        lags: int = 0,
        spe_method: str = "box",
    ) -> None:
        self.n_components = n_components
        self.variance = variance
        self.alpha = alpha
        self.lags = lags
        self.spe_method = spe_method

        self.scaler_ = Scaler()
        self.loadings_: np.ndarray | None = None
        self.eigenvalues_: np.ndarray | None = None
        self.t2_limit_: float | None = None
        self.spe_limit_: float | None = None
        self.n_components_: int | None = None

    def _prepare(self, X: np.ndarray, runs: np.ndarray | None):
        if self.lags == 0:
            return np.asarray(X, dtype=float), runs
        if runs is None:
            raise ValueError("Con lags > 0 hay que pasar el vector de simulaciones")
        return lag_by_run(X, runs, self.lags)

    def fit(
        self,
        X: np.ndarray,
        runs: np.ndarray | None = None,
        X_cal: np.ndarray | None = None,
        runs_cal: np.ndarray | None = None,
    ) -> "PCAMonitor":
        """Ajusta el modelo. X debe contener SOLO operacion normal.

        Parametros
        ----------
        X, runs
            Datos de ajuste y sus identificadores de simulacion.
        X_cal, runs_cal
            Datos normales RESERVADOS para calibrar los limites de control. Si
            se omiten, los limites se calculan sobre los mismos datos del
            ajuste, que es lo que hace buena parte de la literatura y **produce
            limites demasiado estrechos**.

            El motivo es sutil pero decisivo. El modelo se ha ajustado para
            minimizar el residuo sobre X, asi que el SPE que observa en X es
            artificialmente pequeno. Si el umbral se calibra con ese SPE
            optimista, en cuanto llegan datos que el modelo no vio el
            estadistico lo supera constantemente y la tasa de falsas alarmas se
            dispara. Las cifras del efecto NO se escriben aqui: salen de
            `experiments/A0_calibracion_limites.py` y viven en su CSV con la
            configuracion completa (variables, componentes, alpha, metodo,
            tamanos de ajuste y calibracion). El efecto solo existe en los
            limites calibrados con datos (SPE por box, kde o empirical); el
            limite de T2 (distribucion F) y el de Jackson-Mudholkar son
            parametricos y no usan X_cal.

            Es una forma de fuga de datos poco discutida, y por eso conviene
            documentarla en la memoria.
        """
        Xp, _ = self._prepare(X, runs)
        Z = self.scaler_.fit_transform(Xp)

        # Descomposicion de la matriz de covarianza.
        cov = np.cov(Z, rowvar=False)
        autoval, autovec = np.linalg.eigh(cov)
        orden = np.argsort(autoval)[::-1]
        autoval, autovec = autoval[orden], autovec[:, orden]
        autoval = np.clip(autoval, 0.0, None)
        self.eigenvalues_ = autoval

        if self.n_components is None:
            acumulada = np.cumsum(autoval) / autoval.sum()
            self.n_components_ = int(np.searchsorted(acumulada, self.variance) + 1)
        else:
            self.n_components_ = int(self.n_components)

        self.loadings_ = autovec[:, : self.n_components_]

        # Estadisticos sobre los que se calibran los limites: los del conjunto
        # reservado si existe, y si no los del propio ajuste.
        if X_cal is not None:
            Xc, _ = self._prepare(X_cal, runs_cal)
            Zc = self.scaler_.transform(Xc)
            self.calibrated_on_holdout_ = True
        else:
            Zc = Z
            self.calibrated_on_holdout_ = False

        t2_tr, spe_tr = self._statistics(Zc)

        self.t2_limit_ = lim.t2_limit(self.n_components_, len(Z), self.alpha)

        if self.spe_method == "box":
            self.spe_limit_ = lim.spe_limit_box(spe_tr, self.alpha)
        elif self.spe_method == "jackson":
            self.spe_limit_ = lim.spe_limit_jackson(autoval, self.n_components_, self.alpha)
        elif self.spe_method == "kde":
            self.spe_limit_ = lim.limit_kde(spe_tr, self.alpha)
        elif self.spe_method == "empirical":
            self.spe_limit_ = lim.limit_empirical(spe_tr, self.alpha)
        else:
            raise ValueError(f"spe_method desconocido: {self.spe_method}")

        return self

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

    def score(self, X: np.ndarray, runs: np.ndarray | None = None):
        """Devuelve (T2, SPE) para datos nuevos."""
        if self.loadings_ is None:
            raise RuntimeError("Llama a fit() antes de score().")
        Xp, _ = self._prepare(X, runs)
        return self._statistics(self.scaler_.transform(Xp))

    def contributions(self, x: np.ndarray) -> np.ndarray:
        """Contribucion de cada variable al SPE de una observacion.

        Es la herramienta clasica de aislamiento: la variable que mas contribuye
        al residuo es la primera sospechosa. No identifica la causa raiz, porque
        un fallo se propaga por el proceso, pero acota donde mirar.
        """
        if self.loadings_ is None:
            raise RuntimeError("Llama a fit() antes de contributions().")
        z = self.scaler_.transform(np.atleast_2d(x))
        residuo = z - (z @ self.loadings_) @ self.loadings_.T
        return (residuo**2).ravel()
