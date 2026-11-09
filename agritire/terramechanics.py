"""Terramécanique d'un pneu agricole sur sol meuble (Bekker–Wong).

Modèle principal (``traction``) : pneu souple « arc avant + méplat »
--------------------------------------------------------------------
Approche de Bekker (1969) reprise par Wong (2008, chap. 2) pour les pneus :

* on calcule d'abord la roue **rigide** de même diamètre qui porterait la
  charge W ; sa pression sous le point bas est la *pression critique* p_gcr ;
* si la pression au sol du pneu p_g = p_i + p_c (gonflage + contribution de la
  carcasse) est supérieure à p_gcr, le pneu se comporte comme une roue
  rigide (mode « rigid ») ;
* sinon il s'écrase (mode « flexible ») : le contact se compose d'un **arc
  avant** (pneu non déformé, de l'angle d'entrée θ1 jusqu'à θf) et d'un
  **méplat** de longueur 2·l_f à la pression p_g. L'enfoncement est alors
  fixé par p_g : z0 = (p_g / k)^(1/n), et l'écrasement δ du pneu est la
  seule inconnue, déterminée par l'équilibre vertical.

Les deux modes coïncident exactement à la transition (δ = 0), ce qui évite
la discontinuité d'un modèle « empreinte plane / roue rigide » juxtaposés.

Équations
---------
Bekker (pression–enfoncement)         σ = k z^n,   k = k_c/b + k_phi
Janosi–Hanamoto (cisaillement)        τ = (c + σ tan φ)(1 - exp(-j/K))
Cinématique sur l'arc (roue rigide)   j(θ) = R[(θ1-θ) - (1-s)(sinθ1 - sinθ)]
Cinématique sur le méplat             dj/dξ = s   (ξ = abscisse depuis l'avant du méplat)
Équilibre vertical                    W  = Rb∫arc(σ cosθ + τ sinθ)dθ + b∫méplat σ dξ
Poussée brute                         H  = Rb∫arc τ cosθ dθ + b∫méplat τ dξ
Résistance de compaction              Rc = Rb∫arc σ sinθ dθ = b k z0^(n+1)/(n+1)
Effort au crochet                     DP = H - Rc
Couple                                T  = R²b∫arc τ dθ + (R-δ) b∫méplat τ dξ
Glissement                            s = 1 - V/(ωR)
Rendement de traction                 η = DP·V / (T·ω) = DP R (1-s) / T

Modèles complémentaires (comparaison / vérification)
---------------------------------------------------
* ``flat_patch`` : empreinte plane seule, solution analytique de Bekker
  H = (A c + W tanφ)[1 - K/(sL)(1 - e^(-sL/K))] ;
* ``rigid_wheel_wong_reece`` : roue rigide avec la distribution de contrainte
  de Wong & Reece (1967) (maximum décalé vers l'avant, θm = (c1 + c2 s) θ1).

Références
----------
- Bekker, M.G. (1969). *Introduction to Terrain-Vehicle Systems*. University
  of Michigan Press.
- Janosi, Z., Hanamoto, B. (1961). The analytical determination of drawbar
  pull as a function of slip for tracked vehicles in deformable soils.
  Proc. 1st Int. Conf. ISTVS, Turin.
- Wong, J.Y., Reece, A.R. (1967). Prediction of rigid wheel performance based
  on the analysis of soil-wheel stresses, Part I. *J. Terramechanics* 4(1).
- Wong, J.Y. (2008). *Theory of Ground Vehicles*, 4th ed., Wiley, chap. 2.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict, replace
import math

import numpy as np
import pandas as pd
from scipy.integrate import trapezoid
from scipy.optimize import brentq

from .soil import Soil

#: Position du maximum de contrainte normale de Wong–Reece θm = (c1 + c2 s) θ1.
#: Ordres de grandeur rapportés par Wong & Reece (1967) : c1 ≈ 0,4 ;
#: c2 ≈ 0,15 à 0,35. Utilisé uniquement par ``rigid_wheel_wong_reece``.
WONG_REECE_C1 = 0.40
WONG_REECE_C2 = 0.25

N_ARC = 301
N_FLAT = 301


@dataclass(frozen=True)
class Tire:
    """Géométrie et pression d'un pneu agricole.

    Valeurs par défaut : dimension générique 650/65R38 (largeur 0,65 m ;
    hauteur de flanc 0,65 × 0,65 ≈ 0,42 m ; diamètre = 38" + 2 × 0,42 m
    ≈ 1,81 m).

    ``carcass_pressure`` (p_c) : contribution de la rigidité de carcasse à la
    pression au sol. 20 kPa est une HYPOTHÈSE d'ordre de grandeur ; Wong
    (2008) indique qu'on la détermine par un essai d'écrasement du pneu sur
    sol rigide. Elle n'est pas mesurée ici.
    """

    diameter: float = 1.81          # m
    width: float = 0.65             # m
    section_height: float = 0.42    # m
    inflation: float = 120e3        # Pa
    carcass_pressure: float = 20e3  # Pa (hypothèse)
    name: str = "650/65R38 (générique)"

    @property
    def radius(self) -> float:
        return 0.5 * self.diameter

    @property
    def ground_pressure(self) -> float:
        """Pression au sol en mode souple p_g = p_i + p_c [Pa]."""
        return self.inflation + self.carcass_pressure

    def with_(self, **kwargs) -> "Tire":
        return replace(self, **kwargs)


@dataclass
class TractionResult:
    """Résultat d'un calcul de traction à glissement imposé (unités SI)."""

    slip: float
    load: float                 # charge verticale imposée W [N]
    vertical_force: float       # effort vertical recalculé par intégration [N]
    thrust: float               # poussée brute H [N]
    motion_resistance: float    # résistance au mouvement Rc [N]
    drawbar_pull: float         # DP = H - Rc [N]
    torque: float               # couple à la roue [N·m]
    sinkage: float              # enfoncement z0 [m]
    deflection: float           # écrasement du pneu δ [m] (0 en mode rigide)
    contact_length: float       # longueur projetée du contact [m]
    mode: str                   # "flexible", "rigid", "flat" ou "wong_reece"
    tractive_efficiency: float  # rendement de traction [-]

    @property
    def net_traction_ratio(self) -> float:
        """Coefficient de traction nette DP / W [-]."""
        return self.drawbar_pull / self.load

    def as_dict(self) -> dict:
        d = asdict(self)
        d["net_traction_ratio"] = self.net_traction_ratio
        return d


# ---------------------------------------------------------------------------
# Briques élémentaires
# ---------------------------------------------------------------------------
def janosi_shear(sigma, j, soil: Soil):
    """Contrainte de cisaillement de Janosi–Hanamoto [Pa].

    τ = (c + σ tan φ)(1 - exp(-j/K)) pour j ≥ 0.
    """
    tau_max = soil.c_si + np.asarray(sigma) * math.tan(soil.phi)
    return tau_max * (1.0 - np.exp(-np.asarray(j) / soil.K))


def compaction_resistance(soil: Soil, width: float, z0: float) -> float:
    """Résistance de compaction de Bekker Rc = b k z0^(n+1)/(n+1) [N].

    C'est le travail de compaction d'une ornière de largeur b et de
    profondeur z0, par mètre parcouru.
    """
    return width * soil.k_eq(width) * z0 ** (soil.n + 1.0) / (soil.n + 1.0)


def _efficiency(dp, torque, radius, slip):
    if torque <= 0.0 or dp <= 0.0:
        return 0.0
    return dp * radius * (1.0 - slip) / torque


def _check_inputs(load: float, slip: float) -> None:
    if load <= 0:
        raise ValueError("La charge doit être strictement positive.")
    if not 0.0 <= slip < 1.0:
        raise ValueError("Le glissement doit être dans [0, 1) (mode traction).")


# ---------------------------------------------------------------------------
# Modèle principal : arc avant + méplat
# ---------------------------------------------------------------------------
@dataclass
class ContactProfile:
    """Distribution des contraintes le long du contact (pour les figures)."""

    x: np.ndarray      # abscisse horizontale depuis l'aplomb du moyeu [m] (+ vers l'avant)
    sigma: np.ndarray  # contrainte normale [Pa]
    tau: np.ndarray    # contrainte de cisaillement [Pa]


def _integrate_contact(tire: Tire, soil: Soil, slip: float, z0: float,
                       delta: float, n_arc: int = N_ARC, n_flat: int = N_FLAT,
                       profile: bool = False):
    """Intègre les contraintes sur l'arc avant et le méplat.

    Géométrie : centre de roue à la hauteur R - δ au-dessus du fond d'ornière ;
    surface initiale du sol à z0 au-dessus du fond d'ornière.
    """
    R, b = tire.radius, tire.width
    k = soil.k_eq(b)
    cos_f = (R - delta) / R
    cos_1 = (R - delta - z0) / R
    theta_f = math.acos(min(1.0, cos_f))
    theta_1 = math.acos(max(0.0, min(1.0, cos_1)))
    l_f = R * math.sin(theta_f)

    # Arc avant θ ∈ [θf, θ1]
    th = np.linspace(theta_f, theta_1, n_arc)
    z_arc = np.clip(R * (np.cos(th) - cos_1), 0.0, None)
    sig_a = k * z_arc ** soil.n
    j_a = R * ((theta_1 - th) - (1.0 - slip) * (math.sin(theta_1) - np.sin(th)))
    tau_a = janosi_shear(sig_a, np.clip(j_a, 0.0, None), soil)

    rb = R * b
    W = rb * trapezoid(sig_a * np.cos(th) + tau_a * np.sin(th), th)
    H = rb * trapezoid(tau_a * np.cos(th), th)
    Rc = rb * trapezoid(sig_a * np.sin(th), th)
    T = R * rb * trapezoid(tau_a, th)

    # Méplat ξ ∈ [0, 2 l_f] depuis son extrémité avant
    if l_f > 0.0:
        xi = np.linspace(0.0, 2.0 * l_f, n_flat)
        p_flat = k * z0 ** soil.n
        sig_f = np.full_like(xi, p_flat)
        j_f = max(j_a[0], 0.0) + slip * xi
        tau_f = janosi_shear(sig_f, j_f, soil)
        W += b * trapezoid(sig_f, xi)
        Ff = b * trapezoid(tau_f, xi)
        H += Ff
        T += (R - delta) * Ff
    else:
        xi = sig_f = tau_f = np.empty(0)

    length = R * math.sin(theta_1) + l_f
    out = dict(W=W, H=H, Rc=Rc, T=T, length=length, theta_1=theta_1)
    if profile:
        x = np.concatenate([R * np.sin(th)[::-1], l_f - xi])
        out["profile"] = ContactProfile(
            x=x, sigma=np.concatenate([sig_a[::-1], sig_f]),
            tau=np.concatenate([tau_a[::-1], tau_f]))
    return out


def rigid_sinkage(tire: Tire, soil: Soil, load: float, slip: float = 0.0) -> float:
    """Enfoncement z0 de la roue rigide (δ = 0) qui équilibre la charge W."""
    R = tire.radius

    def residual(z):
        return _integrate_contact(tire, soil, slip, z, 0.0)["W"] - load

    z_max = R * (1.0 - 1e-9)
    if residual(z_max) < 0:
        raise ValueError("Charge non supportée : enfoncement > R (hors domaine).")
    return brentq(residual, 1e-9, z_max, xtol=1e-12, rtol=1e-12, maxiter=200)


def critical_ground_pressure(tire: Tire, soil: Soil, load: float,
                             slip: float = 0.0) -> float:
    """Pression critique p_gcr [Pa] : pression sous le point bas de la roue
    rigide équivalente (Bekker, 1969 ; Wong, 2008, chap. 2)."""
    return soil.pressure(rigid_sinkage(tire, soil, load, slip), tire.width)


def rigid_wheel_sinkage_bekker(soil: Soil, load: float, diameter: float,
                               width: float) -> float:
    """Formule approchée de Bekker pour l'enfoncement d'une roue rigide.

    z0 = [3 W / ((3 - n)(k_c + b k_phi) √D)]^(2/(2n+1))   (Bekker, 1969)
    Obtenue en approximant l'arc par une parabole ; sert de vérification
    croisée de l'intégration numérique (z0 < D/6 environ).
    """
    n = soil.n
    denom = (3.0 - n) * (soil.kc_si + width * soil.kphi_si) * math.sqrt(diameter)
    return (3.0 * load / denom) ** (2.0 / (2.0 * n + 1.0))


def operating_mode(tire: Tire, soil: Soil, load: float, slip: float = 0.0) -> str:
    return ("flexible" if tire.ground_pressure < critical_ground_pressure(tire, soil, load, slip)
            else "rigid")


def traction(tire: Tire, soil: Soil, load: float, slip: float,
             mode: str = "auto", profile: bool = False):
    """Effort au crochet à glissement imposé.

    mode : "auto" (modèle principal arc + méplat), "flat" (empreinte plane
    analytique de Bekker) ou "wong_reece" (roue rigide de Wong–Reece).
    Avec ``profile=True`` (mode "auto" seulement) renvoie aussi la
    distribution des contraintes : (TractionResult, ContactProfile).
    """
    if mode == "flat":
        return flat_patch(tire, soil, load, slip)
    if mode == "wong_reece":
        return rigid_wheel_wong_reece(tire, soil, load, slip)
    if mode != "auto":
        raise ValueError(f"mode inconnu : {mode}")

    _check_inputs(load, slip)
    R = tire.radius
    z_r = rigid_sinkage(tire, soil, load, slip)
    p_r = soil.pressure(z_r, tire.width)
    p_g = tire.ground_pressure
    if p_g >= p_r:
        mode_, z0, delta = "rigid", z_r, 0.0
    else:
        mode_ = "flexible"
        z0 = soil.sinkage(p_g, tire.width)
        d_max = R - z0 - 1e-9

        def residual(d):
            return _integrate_contact(tire, soil, slip, z0, d)["W"] - load

        if residual(d_max) < 0:
            raise ValueError("Charge non supportée par le pneu à cette pression.")
        delta = brentq(residual, 0.0, d_max, xtol=1e-12, rtol=1e-12, maxiter=200)

    out = _integrate_contact(tire, soil, slip, z0, delta, profile=profile)
    dp = out["H"] - out["Rc"]
    res = TractionResult(
        slip=slip, load=load, vertical_force=out["W"], thrust=out["H"],
        motion_resistance=out["Rc"], drawbar_pull=dp, torque=out["T"],
        sinkage=z0, deflection=delta, contact_length=out["length"], mode=mode_,
        tractive_efficiency=_efficiency(dp, out["T"], R, slip))
    return (res, out["profile"]) if profile else res


def traction_curve(tire: Tire, soil: Soil, load: float, slips=None,
                   mode: str = "auto") -> pd.DataFrame:
    """Courbe effort au crochet / glissement sous forme de DataFrame."""
    if slips is None:
        slips = np.linspace(0.0, 0.5, 51)
    return pd.DataFrame([traction(tire, soil, load, float(s), mode=mode).as_dict()
                         for s in slips])


# ---------------------------------------------------------------------------
# Modèle simplifié : empreinte plane (solution analytique de Bekker)
# ---------------------------------------------------------------------------
def bekker_flat_thrust(soil: Soil, area: float, length: float, load: float,
                       slip: float) -> float:
    """H = (A c + W tanφ)[1 - K/(sL)(1 - exp(-sL/K))]   (Bekker, 1969)."""
    if slip == 0.0:
        return 0.0
    sl = slip * length
    return (area * soil.c_si + load * math.tan(soil.phi)) * (
        1.0 - soil.K / sl * (1.0 - math.exp(-sl / soil.K)))


def flat_patch(tire: Tire, soil: Soil, load: float, slip: float,
               n_points: int = 401) -> TractionResult:
    """Empreinte plane de longueur L = W/(p_g b) à pression uniforme p_g.

    Poussée intégrée numériquement (j = s x depuis l'avant de l'empreinte) ;
    la solution analytique ``bekker_flat_thrust`` sert de référence en test.
    Résistance = compaction de Bekker. Pas d'arc d'entrée.
    """
    _check_inputs(load, slip)
    b, p_g = tire.width, tire.ground_pressure
    L = load / (p_g * b)
    z0 = soil.sinkage(p_g, b)
    x = np.linspace(0.0, L, n_points)
    sigma = np.full_like(x, p_g)
    tau = janosi_shear(sigma, slip * x, soil)
    W = b * trapezoid(sigma, x)
    H = b * trapezoid(tau, x)
    Rc = compaction_resistance(soil, b, z0)
    T = H * tire.radius
    dp = H - Rc
    return TractionResult(slip, load, W, H, Rc, dp, T, z0, float("nan"), L,
                          "flat", _efficiency(dp, T, tire.radius, slip))


# ---------------------------------------------------------------------------
# Roue rigide de Wong–Reece (comparaison)
# ---------------------------------------------------------------------------
def wong_reece_stresses(theta, theta1: float, slip: float, soil: Soil,
                        radius: float, width: float,
                        c1: float = WONG_REECE_C1, c2: float = WONG_REECE_C2):
    """σ(θ), τ(θ) de Wong & Reece (1967), angle de sortie θ2 = 0."""
    theta = np.asarray(theta, dtype=float)
    k = soil.k_eq(width)
    theta_m = (c1 + c2 * slip) * theta1
    cos1 = math.cos(theta1)
    front = theta >= theta_m
    theta_eq = np.where(front, theta,
                        theta1 - theta / max(theta_m, 1e-12) * (theta1 - theta_m))
    dz = np.clip(np.cos(theta_eq) - cos1, 0.0, None)
    sigma = k * (radius * dz) ** soil.n
    j = radius * ((theta1 - theta) - (1.0 - slip) * (math.sin(theta1) - np.sin(theta)))
    return sigma, janosi_shear(sigma, j, soil)


def rigid_wheel_wong_reece(tire: Tire, soil: Soil, load: float, slip: float,
                           n_points: int = 401, c1: float = WONG_REECE_C1,
                           c2: float = WONG_REECE_C2) -> TractionResult:
    """Roue rigide de Wong–Reece : θ1 tel que W intégré = charge imposée."""
    _check_inputs(load, slip)
    R, b = tire.radius, tire.width

    def integrals(theta1):
        th = np.linspace(0.0, theta1, n_points)
        s_, t_ = wong_reece_stresses(th, theta1, slip, soil, R, b, c1, c2)
        rb = R * b
        return (rb * trapezoid(s_ * np.cos(th) + t_ * np.sin(th), th),
                rb * trapezoid(t_ * np.cos(th), th),
                rb * trapezoid(s_ * np.sin(th), th),
                R * rb * trapezoid(t_, th))

    hi = 0.5 * math.pi * 0.999
    if integrals(hi)[0] < load:
        raise ValueError("Charge non supportée (enfoncement > R).")
    theta1 = brentq(lambda t: integrals(t)[0] - load, 1e-5, hi, xtol=1e-12)
    W, H, Rc, T = integrals(theta1)
    dp = H - Rc
    return TractionResult(slip, load, W, H, Rc, dp, T, R * (1 - math.cos(theta1)),
                          0.0, R * math.sin(theta1), "wong_reece",
                          _efficiency(dp, T, R, slip))
