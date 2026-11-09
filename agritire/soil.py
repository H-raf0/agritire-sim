"""Paramètres de sols pour la terramécanique de Bekker–Wong.

Loi pression–enfoncement de Bekker (Bekker, 1956, 1969) :

    p = (k_c / b + k_phi) * z**n

et loi de cisaillement de Janosi–Hanamoto (1961) :

    tau = (c + p tan(phi)) * (1 - exp(-j / K))

Sources des paramètres (k_c, k_phi, n, c, phi)
---------------------------------------------
Wong, J.Y. (2008). *Theory of Ground Vehicles*, 4th ed., Wiley, chap. 2,
tableau des paramètres de Bekker pour différents sols (valeurs issues du
Land Locomotion Laboratory (LLL) et des mesures de Wong et al.).

Unités d'origine dans le tableau de Wong : k_c en kN/m^(n+1), k_phi en
kN/m^(n+2), c en kPa, phi en degrés. Elles sont converties en SI (N, m, Pa)
dans les propriétés ``kc_si`` / ``kphi_si`` / ``c_si``.

ATTENTION : le module de déformation en cisaillement K n'est pas donné dans
le même tableau. Les valeurs ci-dessous sont des HYPOTHÈSES d'ordre de
grandeur (Wong (2008) cite K ≈ 1 à 2,5 cm pour des sols meubles) ; elles sont
destinées à être recalées (voir ``agritire.calibration``).

Note de traçabilité : les valeurs du tableau ont été retranscrites ; elles
doivent être revérifiées sur l'édition papier avant toute utilisation hors
d'un cadre pédagogique.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math


@dataclass(frozen=True)
class Soil:
    """Jeu de paramètres terramécaniques d'un sol.

    Attributes
    ----------
    name : nom lisible.
    n : exposant d'enfoncement de Bekker [-].
    kc : module de cohésion de déformation [kN/m^(n+1)] (unités de Wong).
    kphi : module de frottement de déformation [kN/m^(n+2)].
    c : cohésion [kPa].
    phi_deg : angle de frottement interne [deg].
    K : module de déformation en cisaillement de Janosi [m] (hypothèse).
    moisture : teneur en eau [%] lors des essais d'origine.
    source : référence bibliographique.
    """

    name: str
    n: float
    kc: float
    kphi: float
    c: float
    phi_deg: float
    K: float
    moisture: float | None = None
    source: str = ""

    # --- Conversions SI -------------------------------------------------
    @property
    def kc_si(self) -> float:
        """k_c en N/m^(n+1)."""
        return self.kc * 1e3

    @property
    def kphi_si(self) -> float:
        """k_phi en N/m^(n+2)."""
        return self.kphi * 1e3

    @property
    def c_si(self) -> float:
        """Cohésion en Pa."""
        return self.c * 1e3

    @property
    def phi(self) -> float:
        """Angle de frottement en radians."""
        return math.radians(self.phi_deg)

    def k_eq(self, b: float) -> float:
        """Module équivalent de Bekker k = k_c/b + k_phi [N/m^(n+2)].

        ``b`` est la plus petite dimension de l'aire de contact [m]
        (Bekker, 1969 ; Wong, 2008, chap. 2).
        """
        if b <= 0:
            raise ValueError("b doit être > 0")
        return self.kc_si / b + self.kphi_si

    def pressure(self, z: float, b: float) -> float:
        """Pression de Bekker p(z) [Pa] pour un enfoncement z [m]."""
        return self.k_eq(b) * max(z, 0.0) ** self.n

    def sinkage(self, p: float, b: float) -> float:
        """Enfoncement statique z(p) [m] sous une pression uniforme p [Pa]."""
        return (max(p, 0.0) / self.k_eq(b)) ** (1.0 / self.n)

    def with_(self, **kwargs) -> "Soil":
        """Copie modifiée (utile pour le recalage et les études de sensibilité)."""
        return replace(self, **kwargs)


WONG = "Wong, J.Y. (2008), Theory of Ground Vehicles, 4th ed., Wiley, chap. 2"

#: Sols de référence. K = hypothèse (voir docstring du module).
SOILS: dict[str, Soil] = {
    "dry_sand": Soil(
        name="Sable sec (LLL)", n=1.1, kc=0.99, kphi=1528.43, c=1.04,
        phi_deg=28.0, K=0.025, moisture=0.0, source=WONG,
    ),
    "sandy_loam": Soil(
        name="Limon sableux (LLL, 15 %)", n=0.7, kc=5.27, kphi=1515.04,
        c=1.72, phi_deg=29.0, K=0.020, moisture=15.0, source=WONG,
    ),
    "clayey_soil": Soil(
        name="Sol argileux (Thaïlande, 38 %)", n=0.5, kc=13.19, kphi=692.15,
        c=4.14, phi_deg=13.0, K=0.010, moisture=38.0, source=WONG,
    ),
    "clayey_loam": Soil(
        name="Limon argileux (North Gower, 46 %)", n=0.73, kc=41.6,
        kphi=2471.0, c=6.1, phi_deg=26.6, K=0.015, moisture=46.0, source=WONG,
    ),
}


def get_soil(key: str) -> Soil:
    """Renvoie un sol de référence par sa clé (``SOILS``)."""
    try:
        return SOILS[key]
    except KeyError as exc:
        raise KeyError(f"Sol inconnu '{key}'. Choix : {sorted(SOILS)}") from exc
