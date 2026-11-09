"""Tests du plan d'expériences et du recalage."""
import numpy as np
import pandas as pd
import pytest

from agritire.calibration import (brixius_mobility, brixius_net_traction,
                                  fit_soil, make_demo_dataset, predict)
from agritire.doe import (FACTORS, evaluate, full_factorial, latin_hypercube,
                          standardized_regression)
from agritire.soil import SOILS
from agritire.terramechanics import Tire


def test_full_factorial_size():
    assert len(full_factorial(levels=3)) == 3 ** 3 * 3


def test_latin_hypercube_stratification():
    """Chaque intervalle de largeur 1/n contient exactement un point par facteur."""
    n = 20
    plan = latin_hypercube(n, soils=["dry_sand"], seed=3)
    for f, (lo, hi) in FACTORS.items():
        u = (plan[f].to_numpy() - lo) / (hi - lo)
        counts = np.bincount(np.minimum((u * n).astype(int), n - 1), minlength=n)
        assert np.all(counts == 1)


def test_src_recovers_linear_model():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({f: rng.uniform(lo, hi, 500) for f, (lo, hi) in FACTORS.items()})
    df["y"] = 2e-5 * df.inflation - 3.0 * df.width + 0.0 * df.load
    src, r2 = standardized_regression(df, "y")
    assert r2 == pytest.approx(1.0, abs=1e-12)
    assert src["load"] == pytest.approx(0.0, abs=1e-10)


def test_evaluate_returns_finite_responses():
    out = evaluate(120e3, 0.65, 30e3, "sandy_loam")
    for k in ("ntr", "efficiency", "sinkage", "rr_coeff", "road_energy"):
        assert np.isfinite(out[k])


def test_brixius_zero_slip_equals_minus_inverse_mobility():
    bn = brixius_mobility(600e3, 0.65, 1.81, 30e3)
    assert brixius_net_traction(0.0, 600e3, 0.65, 1.81, 30e3) == pytest.approx(-1 / bn)


def test_demo_dataset_is_labelled_non_experimental():
    df = make_demo_dataset(Tire())
    assert df["source"].str.contains("NON_EXPERIMENTAL").all()


def test_calibration_recovers_known_parameters():
    """Vérification de la procédure : données générées par le modèle lui-même
    (sans bruit) à partir de paramètres connus ⇒ on doit les retrouver."""
    tire = Tire()
    truth = SOILS["sandy_loam"].with_(c=8.0, phi_deg=27.0, K=0.03)
    data = pd.DataFrame([dict(load=W, inflation=120e3 * W / 30e3,
                              slip=s)
                         for W in (20e3, 40e3) for s in (0.04, 0.1, 0.2, 0.35)])
    data["net_traction_ratio"] = predict(truth, tire, data)
    fit = fit_soil(data, tire, SOILS["sandy_loam"])
    assert fit.params["c"] == pytest.approx(8.0, rel=1e-3)
    assert fit.params["phi_deg"] == pytest.approx(27.0, rel=1e-3)
    assert fit.params["K"] == pytest.approx(0.03, rel=1e-3)
    assert fit.rmse < 1e-6
