"""Tests de cohérence physique du modèle de terramécanique."""
import math

import numpy as np
import pytest

from agritire.soil import SOILS
from agritire.terramechanics import (
    Tire, bekker_flat_thrust, compaction_resistance, critical_ground_pressure,
    flat_patch, rigid_sinkage, rigid_wheel_sinkage_bekker, rigid_wheel_wong_reece,
    traction, traction_curve,
)

SOIL_KEYS = list(SOILS)
LOAD = 30e3
TIRE = Tire()


# --- Glissement nul ----------------------------------------------------------
@pytest.mark.parametrize("key", SOIL_KEYS)
def test_flat_patch_zero_slip_gives_zero_thrust(key):
    """Empreinte plane : à glissement nul, j = 0 partout ⇒ τ = 0 ⇒ H = 0,
    et l'effort au crochet vaut exactement -Rc (seule la compaction reste)."""
    r = flat_patch(TIRE, SOILS[key], LOAD, 0.0)
    assert r.thrust == pytest.approx(0.0, abs=1e-9)
    assert r.drawbar_pull == pytest.approx(-r.motion_resistance, rel=1e-12)
    assert r.drawbar_pull < 0


def test_zero_slip_thrust_vanishes_with_sinkage():
    """Modèle complet : à s = 0, la seule poussée vient de la cinématique de
    l'arc, j(θ) = R[(θ1-θ) - (sinθ1 - sinθ)] = O(θ1³). Elle tend vers 0
    quand l'enfoncement tend vers 0 (sol très raide)."""
    soil = SOILS["sandy_loam"]
    h_soft = traction(TIRE, soil, LOAD, 0.0).thrust / LOAD
    hard = soil.with_(kphi=soil.kphi * 1e4, kc=soil.kc * 1e4)
    h_hard = traction(TIRE, hard, LOAD, 0.0).thrust / LOAD
    assert h_hard < 0.01
    assert h_hard < h_soft


# --- Équilibre ---------------------------------------------------------------
@pytest.mark.parametrize("key", SOIL_KEYS)
@pytest.mark.parametrize("inflation", [60e3, 120e3, 250e3])
@pytest.mark.parametrize("slip", [0.0, 0.1, 0.4])
def test_vertical_equilibrium(key, inflation, slip):
    """L'effort vertical intégré sur le contact équilibre la charge imposée."""
    r = traction(TIRE.with_(inflation=inflation), SOILS[key], LOAD, slip)
    assert r.vertical_force == pytest.approx(LOAD, rel=1e-6)


def test_wong_reece_vertical_equilibrium():
    r = rigid_wheel_wong_reece(TIRE, SOILS["dry_sand"], LOAD, 0.2)
    assert r.vertical_force == pytest.approx(LOAD, rel=1e-6)


@pytest.mark.parametrize("key", SOIL_KEYS)
@pytest.mark.parametrize("inflation", [60e3, 250e3])
def test_compaction_resistance_matches_bekker(key, inflation):
    """Rb∫σ sinθ dθ = b k z0^(n+1)/(n+1) (changement de variable dz = -R sinθ dθ)."""
    soil = SOILS[key]
    r = traction(TIRE.with_(inflation=inflation), soil, LOAD, 0.1)
    expected = compaction_resistance(soil, TIRE.width, r.sinkage)
    assert r.motion_resistance == pytest.approx(expected, rel=2e-3)


# --- Bornes physiques ----------------------------------------------------------
@pytest.mark.parametrize("key", SOIL_KEYS)
@pytest.mark.parametrize("slip", [0.05, 0.3, 0.9])
def test_thrust_bounded_by_coulomb(key, slip):
    """H ≤ c b L + W tanφ : la poussée ne peut excéder la résistance au
    cisaillement de Mohr–Coulomb mobilisée sur l'aire de contact."""
    soil = SOILS[key]
    r = traction(TIRE, soil, LOAD, slip)
    bound = soil.c_si * TIRE.width * r.contact_length + LOAD * math.tan(soil.phi)
    assert r.thrust <= bound * (1 + 1e-9)


@pytest.mark.parametrize("key", SOIL_KEYS)
def test_efficiency_between_zero_and_one(key):
    df = traction_curve(TIRE, SOILS[key], LOAD, np.linspace(0, 0.6, 25))
    assert (df.tractive_efficiency >= 0).all()
    assert (df.tractive_efficiency <= 1).all()


# --- Monotonie -----------------------------------------------------------------
@pytest.mark.parametrize("key", SOIL_KEYS)
@pytest.mark.parametrize("mode", ["auto", "flat"])
def test_drawbar_pull_increases_with_slip(key, mode):
    df = traction_curve(TIRE, SOILS[key], LOAD, np.linspace(0, 0.8, 41), mode=mode)
    assert np.all(np.diff(df.drawbar_pull) >= -1e-6 * LOAD)


@pytest.mark.parametrize("key", SOIL_KEYS)
def test_wong_reece_slip_sinkage(key):
    """Wong–Reece : le maximum de σ avance avec le glissement (θm = (c1+c2 s)θ1),
    l'enfoncement augmente donc avec s (« slip-sinkage ») ; l'effort au crochet
    croît aux glissements usuels (s ≤ 0,25) puis passe par un maximum."""
    df = traction_curve(TIRE, SOILS[key], LOAD, np.linspace(0, 0.8, 41), mode="wong_reece")
    low = df[df.slip <= 0.25]
    assert np.all(np.diff(low.drawbar_pull) >= -1e-6 * LOAD)
    assert df.sinkage.iloc[-1] > df.sinkage.iloc[10]


@pytest.mark.parametrize("key", SOIL_KEYS)
def test_lower_inflation_reduces_sinkage_and_increases_traction(key):
    """Baisser la pression : ornière moins profonde, traction nette plus grande."""
    pressures = [60e3, 100e3, 150e3, 220e3]
    res = [traction(TIRE.with_(inflation=p), SOILS[key], LOAD, 0.15) for p in pressures]
    z = [r.sinkage for r in res]
    dp = [r.drawbar_pull for r in res]
    assert np.all(np.diff(z) >= -1e-12)
    assert np.all(np.diff(dp) <= 1e-9 * LOAD)


@pytest.mark.parametrize("key", SOIL_KEYS)
def test_sinkage_increases_with_load(key):
    z = [traction(TIRE.with_(inflation=250e3), SOILS[key], W, 0.1).sinkage
         for W in (15e3, 30e3, 45e3)]
    assert np.all(np.diff(z) > 0)


# --- Continuité et vérifications croisées ---------------------------------------
@pytest.mark.parametrize("key", SOIL_KEYS)
def test_continuity_at_mode_transition(key):
    """Les modes souple et rigide coïncident à p_g = p_gcr (δ → 0)."""
    soil = SOILS[key]
    p_cr = critical_ground_pressure(TIRE, soil, LOAD, 0.15)
    pc = TIRE.carcass_pressure
    below = traction(TIRE.with_(inflation=p_cr * 0.999 - pc), soil, LOAD, 0.15)
    above = traction(TIRE.with_(inflation=p_cr * 1.001 - pc), soil, LOAD, 0.15)
    assert below.mode == "flexible" and above.mode == "rigid"
    assert below.drawbar_pull == pytest.approx(above.drawbar_pull, rel=1e-2, abs=1e-3 * LOAD)
    assert below.sinkage == pytest.approx(above.sinkage, rel=1e-2)


@pytest.mark.parametrize("key", SOIL_KEYS)
def test_rigid_sinkage_matches_bekker_closed_form(key):
    """Intégration numérique vs formule approchée de Bekker (parabole) : < 10 %."""
    soil = SOILS[key]
    z_num = rigid_sinkage(TIRE, soil, LOAD)
    z_bek = rigid_wheel_sinkage_bekker(soil, LOAD, TIRE.diameter, TIRE.width)
    assert z_num == pytest.approx(z_bek, rel=0.10)


@pytest.mark.parametrize("key", SOIL_KEYS)
@pytest.mark.parametrize("slip", [0.02, 0.15, 0.5])
def test_flat_patch_numeric_matches_analytic(key, slip):
    soil = SOILS[key]
    r = flat_patch(TIRE, soil, LOAD, slip)
    area = LOAD / TIRE.ground_pressure
    expected = bekker_flat_thrust(soil, area, r.contact_length, LOAD, slip)
    assert r.thrust == pytest.approx(expected, rel=1e-4)


def test_invalid_inputs_raise():
    with pytest.raises(ValueError):
        traction(TIRE, SOILS["dry_sand"], -1.0, 0.1)
    with pytest.raises(ValueError):
        traction(TIRE, SOILS["dry_sand"], LOAD, 1.2)
    with pytest.raises(ValueError):
        traction(TIRE, SOILS["dry_sand"], LOAD, 0.1, mode="nope")
