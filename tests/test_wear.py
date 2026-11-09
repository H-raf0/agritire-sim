"""Tests du modèle de brosse et de la loi d'Archard."""
import numpy as np
import pytest
from scipy.integrate import trapezoid

from agritire.wear import (RoadContact, archard_wear, brush_state,
                           lateral_wear_profile, pacejka_parabolic_force,
                           slip_for_force)

C = RoadContact(load=30e3, width=0.65, ground_pressure=140e3)


def test_pressure_equilibrium():
    """∫ p b dx = F_z."""
    assert C.width * trapezoid(C.pressure(), C.grid()) == pytest.approx(C.load, rel=1e-12)


@pytest.mark.parametrize("slip", [0.001, 0.01, 0.03, 0.1, 0.5])
def test_brush_matches_pacejka_closed_form(slip):
    assert brush_state(C, slip).force == pytest.approx(
        pacejka_parabolic_force(C, slip), rel=1e-4)


def test_zero_slip_zero_force_zero_wear():
    st = brush_state(C, 0.0)
    assert st.force == 0.0 and not st.sliding.any()
    w = archard_wear(C, 0.0)
    assert w.friction_energy == 0.0 and w.archard_integral == 0.0


@pytest.mark.parametrize("fx", [500.0, 3000.0, 9000.0, 20000.0])
def test_energy_balance(fx):
    """Régime établi : toute la puissance de glissement F_x·s·V est dissipée par
    frottement dans la zone de glissement ⇒ E_f = F_x · s [J/m]."""
    fine = RoadContact(load=30e3, width=0.65, ground_pressure=140e3, n_points=200001)
    w = archard_wear(fine, fx)
    assert w.friction_energy == pytest.approx(fx * w.slip, rel=2e-3)


def test_wear_increases_with_force():
    e = [archard_wear(C, fx).archard_integral for fx in (1e3, 3e3, 6e3, 12e3)]
    assert np.all(np.diff(e) > 0)


def test_sliding_starts_at_trailing_edge():
    st = brush_state(C, 0.01)
    idx = np.flatnonzero(st.sliding)
    assert idx.size > 0 and idx[-1] == st.x.size - 1
    assert np.all(np.diff(idx) == 1)   # une seule zone de glissement, contiguë


def test_force_cannot_exceed_friction_limit():
    with pytest.raises(ValueError):
        slip_for_force(C, 1.01 * C.mu * C.load)


def test_lateral_profile_uniform_weights_is_uniform():
    s, wear = lateral_wear_profile(30e3, 0.65, 140e3, 3000.0, np.ones(7))
    assert np.allclose(wear, wear[0], rtol=1e-9)
    assert s == pytest.approx(slip_for_force(C, 3000.0), rel=1e-3)
