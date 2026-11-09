"""Recalage des paramètres de sol sur des courbes traction / glissement.

Problème inverse : trouver θ = (c, φ, K) minimisant
    J(θ) = Σ_i [ DP/W_mesuré(s_i, W_i) - DP/W_modèle(s_i, W_i; θ) ]²
par moindres carrés non linéaires bornés (``scipy.optimize.least_squares``,
méthode « trust-region reflective »).

Incertitude : linéarisation au point optimal, covariance
    Cov(θ) ≈ σ² (JᵀJ)⁻¹,  σ² = RSS / (N - p)
d'où les écarts-types et la matrice de corrélation des paramètres, qui
révèle les problèmes d'**identifiabilité** (p. ex. c et φ très corrélés si
l'on ne dispose que d'une seule charge, puisque τ_max = c + σ tanφ).

DONNÉES — AVERTISSEMENT IMPORTANT
--------------------------------
Aucune courbe expérimentale publiée n'a pu être récupérée sous forme
numérique pour ce projet (les articles présentent les mesures sous forme de
figures, qu'il faudrait numériser, p. ex. avec WebPlotDigitizer).
Les « pseudo-mesures » utilisées sont donc des **DONNÉES DE DÉMONSTRATION** :

1. générées par la corrélation **empirique** de Brixius (1987), elle-même
   ajustée sur un grand nombre d'essais au champ de pneus agricoles
   (ASAE/ASABE D497) ; c'est donc une référence indépendante de notre modèle
   (erreur de structure réaliste), mais ce ne sont PAS des mesures ;
2. bruitées artificiellement (bruit gaussien, graine fixée).

Le même code s'applique sans modification à un fichier CSV de mesures
réelles (colonnes ``load, inflation, slip, net_traction_ratio``).

Références
----------
- Brixius, W.W. (1987). Traction prediction equations for bias ply tires.
  ASAE Paper No. 87-1622, St. Joseph, MI.
- ASABE Standard D497 (Agricultural Machinery Management Data).
- Brixius : NT/W = 0,88(1-e^(-0,1 Bn))(1-e^(-7,5 s)) + 0,04 - [1/Bn + 0,04 + 0,5 s/√Bn]
  Bn = (CI b d / W) (1 + 5 δ/h) / (1 + 3 b/d)
- Nocedal, J., Wright, S. (2006). *Numerical Optimization*, 2nd ed. Springer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from .soil import Soil
from .terramechanics import Tire, traction

# Seuls les paramètres de cisaillement sont recalés par défaut : les paramètres
# d'enfoncement (n, k_c, k_phi) se mesurent par essai de plaque (bevameter) et
# sont mal identifiables à partir des seules courbes de traction (forte
# corrélation avec c et K, voir notebook 04).
DEFAULT_PARAMS = ("c", "phi_deg", "K")
DEFAULT_BOUNDS = {
    "c": (0.0, 150.0),         # kPa (borne large : un c irréaliste doit rester visible)
    "phi_deg": (5.0, 45.0),    # deg
    "K": (0.002, 0.10),        # m
    "kphi": (50.0, 10000.0),   # kN/m^(n+2)
    "n": (0.2, 1.5),
}


# ---------------------------------------------------------------------------
# Référence empirique de Brixius (pseudo-mesures de démonstration)
# ---------------------------------------------------------------------------
def brixius_mobility(cone_index, width, diameter, load, deflection_ratio=0.2):
    """Nombre de mobilité de Brixius Bn [-] (CI en Pa, longueurs en m, W en N)."""
    return (cone_index * width * diameter / load) * (1 + 5 * deflection_ratio) / (
        1 + 3 * width / diameter)


def brixius_net_traction(slip, cone_index, width, diameter, load,
                         deflection_ratio=0.2):
    """Coefficient de traction nette NT/W de Brixius (1987)."""
    bn = brixius_mobility(cone_index, width, diameter, load, deflection_ratio)
    s = np.asarray(slip, dtype=float)
    gt = 0.88 * (1 - np.exp(-0.1 * bn)) * (1 - np.exp(-7.5 * s)) + 0.04
    mr = 1 / bn + 0.04 + 0.5 * s / np.sqrt(bn)
    return gt - mr


def make_demo_dataset(tire: Tire, loads=(20e3, 30e3, 40e3), cone_index=600e3,
                      slips=None, noise_std=0.015, seed=42,
                      ref_load=30e3, ref_inflation=120e3) -> pd.DataFrame:
    """Génère des pseudo-mesures de DÉMONSTRATION (Brixius + bruit).

    Brixius suppose un écrasement relatif δ/h constant (≈ 0,2), ce qui
    correspond à des essais où la pression de gonflage est ajustée à la
    charge (tables de charge/pression des manufacturiers). On associe donc à
    chaque charge une pression p_i = p_ref · W / W_ref. Conséquence utile :
    l'aire de contact varie peu alors que W varie, ce qui rend la cohésion c
    et l'angle φ identifiables séparément.
    """
    if slips is None:
        slips = np.array([0.03, 0.05, 0.08, 0.11, 0.14, 0.18, 0.22, 0.27, 0.33])
    rng = np.random.default_rng(seed)
    rows = []
    for W in loads:
        p_i = ref_inflation * W / ref_load
        ntr = brixius_net_traction(slips, cone_index, tire.width, tire.diameter, W)
        ntr = ntr + rng.normal(0.0, noise_std, size=len(slips))
        rows += [dict(load=W, inflation=p_i, slip=s, net_traction_ratio=v)
                 for s, v in zip(slips, ntr)]
    df = pd.DataFrame(rows)
    df["source"] = "DEMO_Brixius1987_plus_bruit_NON_EXPERIMENTAL"
    return df


def save_dataset(df: pd.DataFrame, path: str | Path, header: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for line in header.strip().splitlines():
            fh.write(f"# {line}\n")
        df.to_csv(fh, index=False, float_format="%.6g", lineterminator="\n")


def load_dataset(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path, comment="#")


# ---------------------------------------------------------------------------
# Recalage
# ---------------------------------------------------------------------------
@dataclass
class FitResult:
    soil: Soil
    params: pd.Series
    std_errors: pd.Series
    correlation: pd.DataFrame
    residuals: np.ndarray
    rmse: float
    r2: float
    success: bool
    message: str
    nfev: int


def predict(soil: Soil, tire: Tire, data: pd.DataFrame) -> np.ndarray:
    """DP/W prédit pour chaque ligne (load, inflation, slip) du jeu de données."""
    out = np.empty(len(data))
    for i, r in enumerate(data.itertuples(index=False)):
        t = tire.with_(inflation=float(r.inflation))
        try:
            out[i] = traction(t, soil, float(r.load), float(r.slip)).net_traction_ratio
        except ValueError:
            out[i] = -1.0   # charge non supportée : forte pénalité
    return out


def fit_soil(data: pd.DataFrame, tire: Tire, initial: Soil,
             params=DEFAULT_PARAMS, bounds: dict | None = None) -> FitResult:
    """Recale les paramètres ``params`` du sol ``initial`` sur ``data``."""
    bnds = {**DEFAULT_BOUNDS, **(bounds or {})}
    names = list(params)
    x0 = np.array([getattr(initial, p) for p in names], dtype=float)
    lo = np.array([bnds[p][0] for p in names])
    hi = np.array([bnds[p][1] for p in names])
    x0 = np.clip(x0, lo + 1e-9 * (hi - lo), hi - 1e-9 * (hi - lo))
    # mise à l'échelle : on optimise x / x0_ref pour équilibrer les variables
    scale = np.where(np.abs(x0) > 0, np.abs(x0), 1.0)
    y = data["net_traction_ratio"].to_numpy(float)

    def soil_of(x):
        return initial.with_(**dict(zip(names, x)))

    def resid(xs):
        return predict(soil_of(xs * scale), tire, data) - y

    sol = least_squares(resid, x0 / scale, bounds=(lo / scale, hi / scale),
                        method="trf", x_scale="jac", diff_step=1e-4)
    x = sol.x * scale
    r = sol.fun
    dof = max(len(y) - len(names), 1)
    s2 = float(r @ r) / dof
    J = sol.jac / scale          # dérivées par rapport aux paramètres physiques
    try:
        cov = s2 * np.linalg.inv(J.T @ J)
        std = np.sqrt(np.clip(np.diag(cov), 0, None))
        corr = cov / np.outer(std, std)
    except np.linalg.LinAlgError:
        std = np.full(len(names), np.nan)
        corr = np.full((len(names), len(names)), np.nan)
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return FitResult(
        soil=soil_of(x), params=pd.Series(x, index=names),
        std_errors=pd.Series(std, index=names),
        correlation=pd.DataFrame(corr, index=names, columns=names),
        residuals=r, rmse=float(np.sqrt(np.mean(r ** 2))),
        r2=1.0 - float(r @ r) / ss_tot, success=bool(sol.success),
        message=str(sol.message), nfev=int(sol.nfev))
