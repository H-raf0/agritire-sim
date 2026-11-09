"""Tests de vérification du modèle éléments finis."""
import numpy as np
import pytest

from agritire.fem_contact import (ROAD_FOUNDATION_MODULUS, TreadSection,
                                  solve_section, winkler_modulus_from_bekker)
from agritire.soil import SOILS


def test_homogeneous_layer_analytic_solution():
    """Couche homogène, charge uniforme p, sans effort de flanc, sur Winkler :
    σ_zz = -p, σ_yy = 0 (déformation plane) ⇒ pression de contact = p et
    u_z(haut) = -p/k_w - p H (1 - ν²)/E. Champ linéaire : reproduit exactement
    par des éléments P2 (« patch test »)."""
    E, nu, p, kw = 6e6, 0.45, 100e3, 1e8
    sec = TreadSection(E_belt=E, nu_belt=nu, E_tread=E, nu_tread=nu, n_y=10)
    r = solve_section(sec, p, 0.0, kw)
    assert np.allclose(r.pressure, p, rtol=1e-8)
    expected = -p / kw - p * sec.height * (1 - nu ** 2) / E
    assert r.top_displacement == pytest.approx(expected, rel=1e-8)


@pytest.mark.parametrize("kw", [ROAD_FOUNDATION_MODULUS, 4e6])
@pytest.mark.parametrize("pi", [60e3, 200e3])
def test_vertical_equilibrium(kw, pi):
    """Somme des réactions des ressorts = effort appliqué b/2 (p_i + p_c)."""
    r = solve_section(TreadSection(), pi, 20e3, kw)
    assert r.reaction == pytest.approx(r.applied, rel=1e-9)
    assert r.mean_pressure == pytest.approx(pi + 20e3, rel=1e-9)


def test_mesh_convergence():
    coarse = solve_section(TreadSection(n_y=65), 100e3, 20e3, ROAD_FOUNDATION_MODULUS)
    fine = solve_section(TreadSection(n_y=130, n_z_tread=16), 100e3, 20e3,
                         ROAD_FOUNDATION_MODULUS)
    assert coarse.peak_to_mean == pytest.approx(fine.peak_to_mean, rel=0.02)


def test_higher_inflation_gives_more_uniform_pressure():
    peaks = [solve_section(TreadSection(), p, 20e3, ROAD_FOUNDATION_MODULUS).peak_to_mean
             for p in (60e3, 120e3, 240e3)]
    assert np.all(np.diff(peaks) < 0)


def test_soft_soil_smooths_pressure():
    kw_soil = winkler_modulus_from_bekker(SOILS["sandy_loam"], 140e3, 0.65)
    road = solve_section(TreadSection(), 100e3, 20e3, ROAD_FOUNDATION_MODULUS)
    soil = solve_section(TreadSection(), 100e3, 20e3, kw_soil)
    assert soil.peak_to_mean < road.peak_to_mean
    assert soil.active.all()   # charge descendante : pas de décollement


def test_winkler_modulus_is_secant():
    s = SOILS["sandy_loam"]
    kw = winkler_modulus_from_bekker(s, 140e3, 0.65)
    assert kw * s.sinkage(140e3, 0.65) == pytest.approx(140e3, rel=1e-12)
