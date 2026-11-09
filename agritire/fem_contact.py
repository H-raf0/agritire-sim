"""Modèle éléments finis 2D de la section d'empreinte (scikit-fem).

Problème modélisé
-----------------
Coupe **transversale** (plan y–z) de la bande de roulement au centre de
l'empreinte, en **déformation plane** (ε_xx = 0 dans la direction de
roulement). La section est un bicouche élastique linéaire :

* une **ceinture** (nappes sommet) mince et raide, en haut ;
* la **gomme de bande de roulement**, en bas, modélisée comme une couche
  continue (les crampons ne sont pas représentés).

Chargement (par mètre de longueur d'empreinte) :

* la **pression de gonflage** p_i agit sur la face supérieure (intérieur du
  pneu) ;
* les **flancs** transmettent la contribution de carcasse p_c : un effort
  vertical p_c·b/2 par flanc, réparti uniformément sur une bande d'épaule
  de largeur w_s en haut de la section (zone d'ancrage de la carcasse sur la
  ceinture). Le répartir sur une largeur finie évite une singularité
  d'effort ponctuel qui rendrait le pic d'épaule dépendant du maillage.

L'effort vertical total par unité de longueur vaut donc b (p_i + p_c), ce
qui est cohérent avec l'hypothèse p_g = p_i + p_c du module de
terramécanique (même longueur d'empreinte).

Sol : **fondation de Winkler** (ressorts verticaux indépendants de module
k_w [N/m³]) en **contact unilatéral** (les ressorts ne travaillent qu'en
compression ; résolution par ensemble actif). Le module k_w est déduit de la
loi de Bekker (module sécant au point de fonctionnement) pour un sol meuble,
ou pris très grand pour une route.

Conditions aux limites : symétrie u_y = 0 sur l'axe y = 0 (demi-modèle) ;
face latérale libre ; face inférieure sur ressorts
(aucune raideur tangentielle : contact sans frottement).

Formulation faible : trouver u tel que, pour tout v admissible,
    ∫Ω σ(u) : ε(v) dΩ + ∫Γc k_w u_z v_z dΓ = ∫Γtop (-p_i) v_z dΓ + ∫Γépaule (-q_s) v_z dΓ
avec σ = λ tr(ε) I + 2 μ ε (élasticité linéaire isotrope, petites
déformations). Éléments triangulaires quadratiques P2 (limitent le
verrouillage pour ν proche de 0,5).

Hypothèses fortes (voir rapport) : élasticité linéaire (le caoutchouc est
hyperélastique et quasi incompressible), pas de courbure de sommet, pas de
précontrainte de la ceinture due au gonflage, sol de Winkler (pas de
couplage latéral entre ressorts), déformation plane.

Références
----------
- Gustafsson, T., McBain, G.D. (2020). scikit-fem: A Python package for
  finite element assembly. *J. Open Source Software* 5(52), 2369.
- Zienkiewicz, O.C., Taylor, R.L. (2000). *The Finite Element Method*,
  vol. 1, 5th ed. (élasticité plane, verrouillage volumique).
- Winkler, E. (1867). *Die Lehre von der Elastizität und Festigkeit*.
- Johnson, K.L. (1985). *Contact Mechanics*, Cambridge Univ. Press
  (fondation de Winkler, chap. 4).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from skfem import (Basis, BilinearForm, ElementTriP2, ElementVector, FacetBasis,
                   LinearForm, MeshTri, asm, condense, solve)
from skfem.helpers import ddot, sym_grad, trace, eye

from .soil import Soil

#: Module de Winkler « route » : très supérieur à la raideur de la gomme
#: (E/h ≈ 1,5e8 N/m³), la route est quasi indéformable.
ROAD_FOUNDATION_MODULUS = 1e11


def lame(E: float, nu: float) -> tuple[float, float]:
    """Coefficients de Lamé (λ, μ) en déformation plane."""
    return E * nu / ((1 + nu) * (1 - 2 * nu)), E / (2 * (1 + nu))


def winkler_modulus_from_bekker(soil: Soil, pressure: float, width: float) -> float:
    """Module de Winkler sécant k_w = p / z(p) [N/m³] au point de fonctionnement.

    Linéarisation de la loi de Bekker p = k z^n autour de la pression moyenne
    de contact : approximation valable pour de petites variations de pression.
    """
    return pressure / soil.sinkage(pressure, width)


@dataclass
class TreadSection:
    """Géométrie, matériaux et maillage de la section de bande de roulement.

    Valeurs par défaut (HYPOTHÈSES d'ordre de grandeur, à remplacer par des
    données constructeur) : gomme E = 6 MPa, ν = 0,45 ; ceinture équivalente
    isotrope E = 500 MPa, ν = 0,3.
    """

    width: float = 0.65             # b [m]
    tread_thickness: float = 0.040  # m
    belt_thickness: float = 0.012   # m
    shoulder_band: float = 0.05     # largeur w_s d'ancrage du flanc [m]
    E_tread: float = 6e6
    nu_tread: float = 0.45
    E_belt: float = 500e6
    nu_belt: float = 0.30
    n_y: int = 65                   # éléments sur la demi-largeur
    n_z_tread: int = 8
    n_z_belt: int = 3

    @property
    def height(self) -> float:
        return self.tread_thickness + self.belt_thickness

    def mesh(self) -> MeshTri:
        y = np.linspace(0.0, 0.5 * self.width, self.n_y + 1)
        z = np.concatenate([
            np.linspace(0.0, self.tread_thickness, self.n_z_tread + 1),
            np.linspace(self.tread_thickness, self.height, self.n_z_belt + 1)[1:],
        ])
        return MeshTri.init_tensor(y, z)


@dataclass
class FEMResult:
    y: np.ndarray                 # abscisses des nœuds de contact (demi-largeur) [m]
    pressure: np.ndarray          # pression de contact aux nœuds [Pa]
    active: np.ndarray            # nœuds en contact (bool)
    reaction: float               # Σ réactions verticales (demi-modèle) [N/m]
    applied: float                # effort vertical appliqué (demi-modèle) [N/m]
    top_displacement: float       # u_z moyen de la face supérieure [m]
    iterations: int
    displacement: np.ndarray = field(repr=False)
    basis: Basis = field(repr=False)

    @property
    def mean_pressure(self) -> float:
        """Pression moyenne = réaction / demi-largeur [Pa]."""
        return self.reaction / self.y.max()

    @property
    def peak_to_mean(self) -> float:
        return float(self.pressure.max() / self.mean_pressure)

    @property
    def shoulder_to_center(self) -> float:
        return float(self.pressure[-1] / self.pressure[0])


def solve_section(section: TreadSection, inflation: float,
                  carcass_pressure: float, foundation_modulus: float,
                  max_iter: int = 50) -> FEMResult:
    """Résout le contact section/sol par la méthode de l'ensemble actif."""
    m = section.mesh()
    e = ElementVector(ElementTriP2())
    basis = Basis(m, e, intorder=4)
    h_t = section.tread_thickness
    lam_t, mu_t = lame(section.E_tread, section.nu_tread)
    lam_b, mu_b = lame(section.E_belt, section.nu_belt)

    @BilinearForm
    def elasticity(u, v, w):
        in_belt = w.x[1] > h_t
        lam = np.where(in_belt, lam_b, lam_t)
        mu = np.where(in_belt, mu_b, mu_t)
        eps_u = sym_grad(u)
        sig = 2.0 * mu * eps_u + lam * eye(trace(eps_u), 2)
        return ddot(sig, sym_grad(v))

    kw = float(foundation_modulus)

    @BilinearForm
    def spring(u, v, w):
        return kw * u[1] * v[1]

    half_b, H = 0.5 * section.width, section.height
    w_s = section.shoulder_band
    q_side = carcass_pressure * half_b / w_s   # traction sur la bande d'épaule [Pa]

    @LinearForm
    def top_load(v, w):
        return -inflation * v[1]

    @LinearForm
    def side_load(v, w):
        return -q_side * v[1]

    f_top = m.facets_satisfying(lambda x: np.isclose(x[1], H))
    f_side = m.facets_satisfying(
        lambda x: np.isclose(x[1], H) & (x[0] >= half_b - w_s - 1e-12))
    f_bot = m.facets_satisfying(lambda x: np.isclose(x[1], 0.0))
    K = asm(elasticity, basis)
    f = asm(top_load, FacetBasis(m, e, facets=f_top, intorder=4))
    if carcass_pressure:
        f = f + asm(side_load, FacetBasis(m, e, facets=f_side, intorder=4))
    D = basis.get_dofs(lambda x: np.isclose(x[0], 0.0)).all("u^1")

    bot_nodes = np.unique(m.facets[:, f_bot])
    bot_nodes = bot_nodes[np.argsort(m.p[0, bot_nodes])]
    uz_dofs = basis.nodal_dofs[1, bot_nodes]

    active = np.ones(f_bot.size, dtype=bool)
    u = np.zeros(basis.N)
    for it in range(1, max_iter + 1):
        if not active.any():
            raise RuntimeError("Plus aucun contact : problème instable.")
        Ks = asm(spring, FacetBasis(m, e, facets=f_bot[active], intorder=4))
        u = solve(*condense(K + Ks, f, D=D))
        # facette active si son milieu (moyenne des 2 sommets) est en compression
        uz_vert = u[basis.nodal_dofs[1, m.facets[:, f_bot]]]
        new_active = uz_vert.mean(axis=0) < 0.0
        if np.array_equal(new_active, active):
            break
        active = new_active
    else:
        raise RuntimeError("Ensemble actif non convergé.")

    reaction = -float(np.sum((Ks @ u)[basis.nodal_dofs[1]]) +
                      np.sum((Ks @ u)[basis.facet_dofs[1]]))
    applied = half_b * (inflation + carcass_pressure)
    uz = u[uz_dofs]
    node_active = uz < 0.0
    pressure = np.where(node_active, -kw * uz, 0.0)
    top_nodes = np.unique(m.facets[:, f_top])
    top_uz = float(np.mean(u[basis.nodal_dofs[1, top_nodes]]))
    return FEMResult(m.p[0, bot_nodes], pressure, node_active, reaction, applied,
                     top_uz, it, u, basis)


def full_width_profile(res: FEMResult) -> tuple[np.ndarray, np.ndarray]:
    """Reconstruit le profil sur toute la largeur par symétrie."""
    y = np.concatenate([-res.y[::-1], res.y[1:]])
    p = np.concatenate([res.pressure[::-1], res.pressure[1:]])
    return y, p
