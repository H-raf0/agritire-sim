"""Usure sur route : modèle de brosse + loi d'Archard.

Démarche
--------
1. **Empreinte sur route** (sol indéformable) : longueur L = F_z / (b p_g)
   avec p_g = p_i + p_c (même hypothèse que le mode souple de
   ``terramechanics``), distribution longitudinale de pression p(x)
   parabolique par défaut (ou fournie, p. ex. issue du FEM).
2. **Modèle de brosse** (Pacejka, 2012, chap. 3) : la bande de roulement est
   un ensemble de « poils » de raideur c_p [N/m³] par unité de surface.
   Un poil entre à l'avant de l'empreinte (x = 0) sans déformation ; tant
   qu'il adhère, sa déformation croît comme u = s x (glissement s petit).
   Il glisse dès que c_p u > μ p(x) ; sa déformation vaut alors μ p / c_p.
   On intègre ce comportement en marchant de l'avant vers l'arrière, ce qui
   gère aussi une éventuelle ré-adhérence pour une distribution p(x) quelconque.
3. **Vitesse de glissement** d'un poil par rapport à la route :
   v_g / V = s - du/dx  (nulle en adhérence, > 0 en glissement).
4. **Loi d'Archard** (Archard, 1953) appliquée localement :
   dh = k_A · p · dS_g  ⇒ volume usé par mètre parcouru
       dV/dX = k_A · b ∫ p(x) (v_g/V) dx
   et l'énergie dissipée par frottement par mètre parcouru
       E_f = b ∫ μ p(x) (v_g/V) dx  [J/m]
   (l'approche « énergétique » de Schallamach relie l'usure à E_f ; ici
   Archard avec μ constant donne dV/dX = (k_A/μ) E_f : les deux sont
   proportionnelles.)

Le coefficient d'usure k_A du caoutchouc sur un revêtement donné n'est pas
connu ici : il dépend du mélange, de la rugosité du revêtement, de la
température. Les résultats sont donc présentés sous forme **relative**
(indice d'usure) ; un volume absolu n'a de sens qu'après identification de
k_A sur un essai d'usure de référence.

Références
----------
- Archard, J.F. (1953). Contact and rubbing of flat surfaces.
  *J. Applied Physics* 24(8), 981–988.
- Pacejka, H.B. (2012). *Tire and Vehicle Dynamics*, 3rd ed.,
  Butterworth-Heinemann, chap. 3 (modèle de brosse).
- Schallamach, A., Turner, D.M. (1960). The wear of slipping wheels.
  *Wear* 3(1), 1–25.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np
from scipy.integrate import trapezoid
from scipy.optimize import brentq

#: Coefficient de frottement caoutchouc / asphalte sec (hypothèse usuelle).
MU_ROAD = 0.8
#: Rigidité de glissement longitudinal rapportée à la charge C_s / F_z.
#: HYPOTHÈSE d'ordre de grandeur pour un pneu agricole (les pneus de tourisme
#: sont plutôt vers 15–20) ; à identifier sur banc.
SLIP_STIFFNESS_RATIO = 6.0


@dataclass
class RoadContact:
    """Empreinte sur route et paramètres de brosse."""

    load: float                      # F_z [N]
    width: float                     # b [m]
    ground_pressure: float           # p_g = p_i + p_c [Pa]
    mu: float = MU_ROAD
    slip_stiffness_ratio: float = SLIP_STIFFNESS_RATIO
    n_points: int = 4001
    pressure_shape: np.ndarray | None = None  # forme p(ξ) sur ξ ∈ [0,1], normalisée ensuite
    bristle_stiffness_override: float | None = None  # c_p imposé [N/m³]

    @property
    def length(self) -> float:
        """L = F_z / (b p_g) [m]."""
        return self.load / (self.width * self.ground_pressure)

    @property
    def bristle_stiffness(self) -> float:
        """c_p [N/m³] tel que C_s = c_p b L² / 2 (brosse, Pacejka 2012)."""
        if self.bristle_stiffness_override is not None:
            return self.bristle_stiffness_override
        cs = self.slip_stiffness_ratio * self.load
        return 2.0 * cs / (self.width * self.length ** 2)

    def grid(self) -> np.ndarray:
        return np.linspace(0.0, self.length, self.n_points)

    def pressure(self) -> np.ndarray:
        """p(x) [Pa] avec ∫ p b dx = F_z (équilibre vertical)."""
        x = self.grid()
        xi = x / self.length
        if self.pressure_shape is None:
            shape = 4.0 * xi * (1.0 - xi)            # parabole, max au centre
        else:
            src = np.asarray(self.pressure_shape, dtype=float)
            shape = np.interp(xi, np.linspace(0.0, 1.0, src.size), src)
            shape = np.clip(shape, 0.0, None)
        scale = self.load / (self.width * trapezoid(shape, x))
        return scale * shape


@dataclass
class BrushState:
    x: np.ndarray
    p: np.ndarray
    u: np.ndarray            # déformation des poils [m]
    q: np.ndarray            # contrainte tangentielle [Pa]
    sliding: np.ndarray      # masque booléen
    slide_ratio: np.ndarray  # v_g / V [-]
    force: float             # F_x [N]


def brush_state(contact: RoadContact, slip: float) -> BrushState:
    """Résout le modèle de brosse à glissement s imposé (s ≥ 0, petit)."""
    if slip < 0:
        raise ValueError("Seul le glissement moteur (s >= 0) est traité.")
    x = contact.grid()
    p = contact.pressure()
    cp, mu = contact.bristle_stiffness, contact.mu
    dx = x[1] - x[0]
    # Récurrence de marche : u_i = min(u_{i-1} + s dx, μ p_i / c_p), u_0 = 0.
    # En posant g_i = μ p_i / c_p - s x_i, elle s'écrit u_i - s x_i =
    # min(u_{i-1} - s x_{i-1}, g_i) : c'est un minimum cumulé (vectorisable).
    u_lim = mu * p / cp
    g = u_lim - slip * x
    g[0] = 0.0
    running = np.minimum.accumulate(g)
    u = slip * x + running
    sliding = np.zeros(x.size, dtype=bool)
    sliding[1:] = g[1:] < running[:-1]   # la limite de Coulomb est active
    q = cp * u
    du = np.gradient(u, dx)
    slide_ratio = np.where(sliding, np.clip(slip - du, 0.0, None), 0.0)
    force = contact.width * trapezoid(q, x)
    return BrushState(x, p, u, q, sliding, slide_ratio, force)


def pacejka_parabolic_force(contact: RoadContact, slip: float) -> float:
    """Solution analytique du modèle de brosse à pression parabolique.

    F_x = μ F_z (3θs - 3θ²s² + θ³s³) pour s ≤ 1/θ,  θ = C_s / (3 μ F_z)
    (Pacejka, 2012, chap. 3). Sert de référence en test.
    """
    cs = contact.slip_stiffness_ratio * contact.load
    theta = cs / (3.0 * contact.mu * contact.load)
    ts = min(theta * slip, 1.0)
    return contact.mu * contact.load * (3 * ts - 3 * ts ** 2 + ts ** 3)


def slip_for_force(contact: RoadContact, fx: float) -> float:
    """Glissement nécessaire pour transmettre l'effort F_x."""
    fmax = contact.mu * contact.load
    if not 0.0 <= fx < fmax:
        raise ValueError(f"F_x doit être dans [0, μF_z = {fmax:.0f} N).")
    if fx == 0.0:
        return 0.0
    cs = contact.slip_stiffness_ratio * contact.load
    s_hi = 3.0 * contact.mu * contact.load / cs * 1.5
    return brentq(lambda s: brush_state(contact, s).force - fx, 0.0, s_hi, xtol=1e-12)


@dataclass
class WearResult:
    slip: float
    force: float                    # F_x [N]
    sliding_fraction: float         # part de l'empreinte en glissement [-]
    friction_energy: float          # E_f [J/m parcouru]
    archard_integral: float         # b ∫ p v_g/V dx [N] = dV/dX / k_A
    wear_volume_per_km: float | None = None   # [mm³/km] si k_A fourni
    state: BrushState | None = field(default=None, repr=False)


def archard_wear(contact: RoadContact, fx: float, k_archard: float | None = None,
                 keep_state: bool = False) -> WearResult:
    """Usure d'Archard pour un effort de traction F_x sur route.

    k_archard : coefficient d'usure dimensionnel k_A [mm³/(N·m)] (optionnel).
    """
    s = slip_for_force(contact, fx)
    st = brush_state(contact, s)
    b = contact.width
    integral = b * trapezoid(st.p * st.slide_ratio, st.x)
    energy = b * trapezoid(st.q * st.slide_ratio, st.x)
    frac = float(np.mean(st.sliding))
    vol = None if k_archard is None else k_archard * integral * 1000.0
    return WearResult(s, st.force, frac, energy, integral, vol,
                      st if keep_state else None)


def lateral_wear_profile(load: float, width: float, ground_pressure: float,
                         fx: float, lateral_weights, mu: float = MU_ROAD,
                         slip_stiffness_ratio: float = SLIP_STIFFNESS_RATIO,
                         n_points: int = 20001):
    """Profil latéral d'usure à partir d'une distribution latérale de pression.

    La largeur est découpée en bandes ; la bande i porte une pression
    p_i(x) = w_i · p(x) (w normalisé à une moyenne de 1, p. ex. issu du FEM).
    Hypothèse de ceinture rigide : le glissement s est le même pour toutes
    les bandes ; il est ajusté pour que Σ F_x,i = F_x.

    La zone glissante ne couvre que quelques % de l'empreinte aux efforts
    usuels : une grille fine (``n_points`` ≈ 2·10⁴) est nécessaire pour que
    l'erreur de discrétisation (≈ ±2 % avec 800 points) reste négligeable.

    Renvoie (s, intégrale d'Archard par unité de largeur [N/m] par bande).
    """
    w = np.asarray(lateral_weights, dtype=float)
    w = w / w.mean()
    strip_b = width / w.size
    # La raideur de poil c_p est une propriété du matériau : identique pour
    # toutes les bandes, déduite du pneu complet. La longueur d'empreinte est
    # aussi identique : F_z,i / (b_i p_g,i) = F_z / (b p_g).
    cp = RoadContact(load, width, ground_pressure, mu, slip_stiffness_ratio).bristle_stiffness
    strips = [RoadContact(load * wi / w.size, strip_b, ground_pressure * wi, mu,
                          slip_stiffness_ratio, n_points,
                          bristle_stiffness_override=cp) for wi in w]

    def total(s):
        return sum(brush_state(c, s).force for c in strips) - fx

    s_hi = 3.0 * mu / slip_stiffness_ratio * 3.0 * float(w.max())
    s = brentq(total, 0.0, s_hi, xtol=1e-12)
    out = []
    for c in strips:
        st = brush_state(c, s)
        out.append(trapezoid(st.p * st.slide_ratio, st.x))  # par unité de largeur
    return s, np.array(out)
