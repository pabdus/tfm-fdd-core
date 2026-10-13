"""Pruebas de la ronda de correcciones del 12/10 en coste y robustez.

Cada prueba protege un arreglo:
- V-06: el aviso de memoria dice por que una cifra no es exacta y no recomienda
  lo que no la arregla; la inferencia, que en el mismo proceso queda tapada por
  el ajuste, sale exacta medida en un proceso aparte; y ese modo falla con el
  motivo cuando algo no se puede serializar o el proceso aparte falla.
- V-17: sin psutil, measure_cost falla antes de ajustar y sin ofrecer un
  parametro que no tiene; measure sigue ofreciendo process_memory=False.
- M9: los errores por nivel valen igual como array 2D que como lista.
- M10: los docstrings de robustness no cuentan resultados de una ejecucion sin
  su configuracion.
"""

import sys
import threading
import time

import numpy as np
import pytest

from tfmfdd import cost, robustness

MB = 1024 * 1024


def _ajuste_grande():
    """Ajuste que toca 120 MB y los suelta antes de devolver el modelo."""
    a = np.ones(120 * MB // 8)
    time.sleep(0.05)
    s = float(a[-1])
    del a
    return {"s": s}


def _prediccion_de_40_mb(modelo, xb):
    b = np.ones(40 * MB // 8)
    time.sleep(0.02)
    return float(b[-1]) + modelo["s"]


_CONFIG = dict(batch_size=10, n_repeats_fit=1, n_warmup_fit=0, n_repeats_infer=1, n_warmup_infer=0)


# ---------------------------------------------------------------- V-06

def test_los_avisos_de_memoria_no_recomiendan_lo_que_no_arregla():
    solo_inferencia = cost._avisos_de_memoria(True, False, "same_process")
    assert len(solo_inferencia) == 1
    assert "un proceso nuevo no lo cambia" in solo_inferencia[0]
    assert "infer_memory='subprocess'" in solo_inferencia[0]
    solo_ajuste = cost._avisos_de_memoria(False, True, "same_process")
    assert len(solo_ajuste) == 1
    assert "proceso nuevo" in solo_ajuste[0] and "infer_memory" not in solo_ajuste[0]
    aparte = cost._avisos_de_memoria(True, False, "subprocess")
    assert len(aparte) == 1 and "proceso aparte" in aparte[0] and "carga" in aparte[0]
    assert cost._avisos_de_memoria(True, True, "same_process") == []
    assert len(cost._avisos_de_memoria(False, False, "subprocess")) == 2


def test_infer_memory_desconocido_falla():
    with pytest.raises(ValueError, match="infer_memory"):
        cost.measure_cost(lambda: object(), lambda m, xb: None, np.zeros((4, 2)), method="x",
                          infer_memory="otro", **{**_CONFIG, "batch_size": 2})


def test_en_el_mismo_proceso_la_inferencia_queda_tapada_por_el_ajuste():
    """V-06 tal cual: la inferencia (40 MB) va justo detras de un ajuste de 120 MB."""
    with pytest.warns(RuntimeWarning, match="un proceso nuevo no lo cambia"):
        fila = cost.measure_cost(_ajuste_grande, _prediccion_de_40_mb, np.zeros((20, 3)),
                                 method="mismo", **_CONFIG)
    assert fila["infer_memory_where"] == "same_process"
    assert fila["infer_peak_process_exact"] is False
    assert fila["infer_peak_process_bytes"] <= 1.1 * 40 * MB      # cota inferior, nunca mas


@pytest.mark.filterwarnings("ignore:.*pico de memoria del ajuste:RuntimeWarning")
def test_en_un_proceso_aparte_la_inferencia_sale_exacta():
    def predecir(modelo, xb):           # definida aqui: cloudpickle la pasa por valor
        b = np.ones(40 * MB // 8)
        time.sleep(0.02)
        return float(b[-1]) + modelo["s"]

    fila = cost.measure_cost(_ajuste_grande, predecir, np.zeros((20, 3)), method="aparte",
                             infer_memory="subprocess", **_CONFIG)
    assert fila["infer_memory_where"] == "subprocess"
    assert fila["infer_peak_process_exact"] is True
    assert 0.9 * 40 * MB <= fila["infer_peak_process_bytes"] <= 1.5 * 40 * MB
    assert fila["infer_time_per_sample_s_median"] > 0             # el tiempo, en este proceso


def test_en_un_proceso_aparte_lo_que_no_se_serializa_falla_con_el_motivo():
    candado = threading.Lock()
    with pytest.raises(TypeError, match="no pudo serializar"):
        cost.measure_cost(lambda: {"candado": candado}, lambda m, xb: None, np.zeros((4, 2)),
                          method="x", infer_memory="subprocess", **{**_CONFIG, "batch_size": 2})


def test_si_el_proceso_aparte_falla_se_ve_su_error():
    def predecir(modelo, xb):
        raise ZeroDivisionError("fallo a proposito en el proceso aparte")

    with pytest.raises(RuntimeError, match="fallo a proposito en el proceso aparte"):
        cost.measure_cost(lambda: {"s": 1.0}, predecir, np.zeros((4, 2)), method="x",
                          infer_memory="subprocess", **{**_CONFIG, "batch_size": 2})


# ---------------------------------------------------------------- V-17

def test_sin_psutil_measure_cost_falla_antes_de_ajustar_y_sin_parametros_que_no_tiene(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)      # asi 'import psutil' falla
    ajustes = []

    def fit():
        ajustes.append(1)
        return object()

    with pytest.raises(ImportError) as error:
        cost.measure_cost(fit, lambda m, xb: None, np.zeros((4, 2)), method="x",
                          **{**_CONFIG, "batch_size": 2})
    assert "psutil" in str(error.value)
    assert "process_memory" not in str(error.value)
    assert ajustes == []


def test_sin_psutil_measure_sigue_ofreciendo_process_memory_false(monkeypatch):
    monkeypatch.setitem(sys.modules, "psutil", None)
    with pytest.raises(ImportError, match="process_memory=False"):
        with cost.measure():
            pass
    with cost.measure(process_memory=False) as m:
        pass
    assert m["wall_s"] >= 0


# ---------------------------------------------------------------- M9 y M10

def test_errores_como_array_2d_o_como_lista_dan_el_mismo_indice():
    niveles = [1.0, 1.5, 2.0]
    errores = np.array([[0.10, 0.12, 0.14], [0.20, 0.24, 0.28], [0.30, 0.36, 0.42]])
    r_array = robustness.robustness_index(niveles, errores, confidence=0.95)
    r_lista = robustness.robustness_index(niveles, errores.tolist(), confidence=0.95)
    for clave in ("i_yy", "d_p", "delta_i", "j_ril"):
        assert r_array[clave] == pytest.approx(r_lista[clave])
    # Un array 1D es un error por nivel: hay I_yy y D_p, pero no intervalo.
    r_1d = robustness.robustness_index(niveles, errores.mean(axis=1), confidence=0.95)
    assert r_1d["i_yy"] == pytest.approx(r_array["i_yy"]) and np.isnan(r_1d["delta_i"])


def test_los_docstrings_de_robustness_no_cuentan_resultados_de_una_ejecucion():
    texto = (robustness.__doc__ or "") + (robustness.error_rate.__doc__ or "")
    assert "prueba de humo" not in texto
