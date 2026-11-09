"""Plans d'expériences numériques et analyse de sensibilité.

Facteurs étudiés
----------------
- pression de gonflage p_i  : 60 – 200 kPa
- largeur de section b      : 0,45 – 0,85 m (rapport d'aspect 65 % conservé,
                              diamètre extérieur fixé à 1,81 m)
- charge verticale W        : 20 – 45 kN
- type de sol (qualitatif)  : sable sec / limon sableux / limon argileux

Réponses
--------
- ``ntr``        : coefficient de traction nette DP/W à 15 % de glissement ;
- ``efficiency`` : rendement de traction à 15 % ;
- ``sinkage``    : enfoncement (profondeur d'ornière) [m] ;
- ``rr_coeff``   : coefficient de résistance au roulement Rc/W ;
- ``road_energy``: énergie de frottement sur route [J/m] pour F_x = 0,1 W
                   (indicateur proportionnel à l'usure d'Archard) ;
- ``road_deflection_ratio`` : écrasement estimé sur route / hauteur de flanc,
  utilisé comme contrainte de faisabilité (HYPOTHÈSE : ≤ 0,25).

Plans
-----
- plan **factoriel complet** 3 niveaux × 3 facteurs continus × 3 sols
  (81 essais) : effets principaux et interactions lisibles ;
- **hypercube latin** (McKay et al., 1979) par sol : couverture uniforme de
  l'espace, base de l'analyse de sensibilité par **coefficients de
  régression standardisés** (SRC) et **corrélations de rang** de Spearman
  (Saltelli et al., 2008). Les SRC ne sont interprétables que si le modèle
  linéaire explique la variance (R² proche de 1) ; le R² est donc toujours
  rapporté.

Références
----------
- McKay, M.D., Beckman, R.J., Conover, W.J. (1979). A comparison of three
  methods for selecting values of input variables in the analysis of output
  from a computer code. *Technometrics* 21(2).
- Saltelli, A. et al. (2008). *Global Sensitivity Analysis. The Primer*. Wiley.
- Montgomery, D.C. (2017). *Design and Analysis of Experiments*, 9th ed. Wiley.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
from scipy.stats import qmc, spearmanr

from .soil import get_soil
from .terramechanics import Tire, traction
from .wear import RoadContact, archard_wear

FACTORS: dict[str, tuple[float, float]] = {
    "inflation": (60e3, 200e3),   # Pa
    "width": (0.45, 0.85),        # m
    "load": (20e3, 45e3),         # N
}
SOIL_LEVELS = ["dry_sand", "sandy_loam", "clayey_loam"]
ASPECT_RATIO = 0.65
DEFLECTION_LIMIT = 0.25
RESPONSES = ["ntr", "efficiency", "sinkage", "rr_coeff", "road_energy"]


def evaluate(inflation: float, width: float, load: float, soil_key: str,
             slip: float = 0.15, base_tire: Tire | None = None,
             road_force_ratio: float = 0.10) -> dict:
    """Évalue toutes les réponses pour une configuration."""
    base = base_tire or Tire()
    tire = base.with_(inflation=inflation, width=width,
                      section_height=ASPECT_RATIO * width)
    soil = get_soil(soil_key)
    out = dict(inflation=inflation, width=width, load=load, soil=soil_key)
    try:
        r = traction(tire, soil, load, slip)
        out.update(ntr=r.net_traction_ratio, efficiency=r.tractive_efficiency,
                   sinkage=r.sinkage, rr_coeff=r.motion_resistance / load,
                   mode=r.mode, deflection_soil=r.deflection)
    except ValueError:
        out.update(ntr=np.nan, efficiency=np.nan, sinkage=np.nan,
                   rr_coeff=np.nan, mode="failed", deflection_soil=np.nan)
    rc = RoadContact(load, width, tire.ground_pressure)
    out["road_energy"] = archard_wear(rc, road_force_ratio * load).friction_energy
    # écrasement sur route estimé à partir de la corde : L = 2√(2Rδ) ⇒ δ = L²/(8R)
    out["road_deflection_ratio"] = rc.length ** 2 / (8 * tire.radius) / tire.section_height
    out["feasible"] = out["road_deflection_ratio"] <= DEFLECTION_LIMIT
    return out


def full_factorial(levels: int = 3, soils=SOIL_LEVELS) -> pd.DataFrame:
    """Plan factoriel complet ``levels``^3 × sols."""
    grids = {k: np.linspace(lo, hi, levels) for k, (lo, hi) in FACTORS.items()}
    rows = [dict(zip(FACTORS, combo), soil=s)
            for s in soils for combo in itertools.product(*grids.values())]
    return pd.DataFrame(rows)


def latin_hypercube(n: int, soils=SOIL_LEVELS, seed: int = 0) -> pd.DataFrame:
    """Hypercube latin de ``n`` points par sol (graine fixée : reproductible)."""
    lo = np.array([v[0] for v in FACTORS.values()])
    hi = np.array([v[1] for v in FACTORS.values()])
    frames = []
    for i, s in enumerate(soils):
        sampler = qmc.LatinHypercube(d=len(FACTORS), seed=seed + i)
        x = qmc.scale(sampler.random(n), lo, hi)
        df = pd.DataFrame(x, columns=list(FACTORS))
        df["soil"] = s
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def run_plan(plan: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """Exécute un plan (une ligne = un calcul) et renvoie les réponses."""
    rows = [evaluate(r.inflation, r.width, r.load, r.soil, **kwargs)
            for r in plan.itertuples(index=False)]
    return pd.DataFrame(rows)


def main_effects(results: pd.DataFrame, response: str) -> dict[str, pd.Series]:
    """Effets principaux : moyenne de la réponse par niveau de chaque facteur."""
    return {f: results.groupby(f)[response].mean() for f in [*FACTORS, "soil"]}


def standardized_regression(results: pd.DataFrame, response: str,
                            factors=tuple(FACTORS)) -> tuple[pd.Series, float]:
    """Coefficients de régression standardisés (SRC) et R² du modèle linéaire.

    y* = Σ SRC_i x_i*  avec x*, y* centrés réduits. Pour un modèle linéaire
    à entrées indépendantes, Σ SRC_i² = R².
    """
    df = results.dropna(subset=[response])
    X = df[list(factors)].to_numpy(float)
    y = df[response].to_numpy(float)
    Xs = (X - X.mean(0)) / X.std(0)
    ys = (y - y.mean()) / y.std()
    coef, *_ = np.linalg.lstsq(Xs, ys, rcond=None)
    r2 = 1.0 - np.sum((ys - Xs @ coef) ** 2) / np.sum(ys ** 2)
    return pd.Series(coef, index=list(factors)), float(r2)


def rank_correlations(results: pd.DataFrame, response: str,
                      factors=tuple(FACTORS)) -> pd.Series:
    """Coefficients de corrélation de rang de Spearman (relations monotones)."""
    df = results.dropna(subset=[response])
    return pd.Series({f: spearmanr(df[f], df[response]).statistic for f in factors})


def sensitivity_table(results: pd.DataFrame, responses=RESPONSES) -> pd.DataFrame:
    """Tableau SRC / R² / Spearman par sol et par réponse."""
    rows = []
    for soil, grp in results.groupby("soil"):
        for resp in responses:
            src, r2 = standardized_regression(grp, resp)
            rho = rank_correlations(grp, resp)
            for f in FACTORS:
                rows.append(dict(soil=soil, response=resp, factor=f,
                                 SRC=src[f], R2=r2, spearman=rho[f]))
    return pd.DataFrame(rows)
