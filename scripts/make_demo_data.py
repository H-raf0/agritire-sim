"""Génère le jeu de pseudo-mesures de DÉMONSTRATION (non expérimental).

Usage : python scripts/make_demo_data.py
"""
from pathlib import Path

from agritire.calibration import make_demo_dataset, save_dataset
from agritire.terramechanics import Tire

HEADER = """
DONNEES DE DEMONSTRATION - CE NE SONT PAS DES MESURES EXPERIMENTALES.
Generees par la correlation empirique de Brixius (1987, ASAE Paper 87-1622 ;
ASABE D497) pour un pneu 650/65R38 generique (b = 0,65 m, d = 1,81 m),
indice de cone CI = 600 kPa, ecrasement relatif delta/h = 0,2,
pression de gonflage ajustee a la charge (p_i = 120 kPa x W / 30 kN),
plus un bruit gaussien d'ecart-type 0,015 (graine 42).
Colonnes : load [N], inflation [Pa], slip [-], net_traction_ratio [-] (DP/W), source.
Pour utiliser de vraies mesures : remplacer ce fichier par un CSV de meme format
(p. ex. courbes numerisees avec WebPlotDigitizer, source citee en en-tete).
"""

if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "experimental" / "DEMO_brixius_pseudo_mesures.csv"
    df = make_demo_dataset(Tire(), cone_index=600e3)
    save_dataset(df, out, HEADER)
    print(f"{len(df)} lignes écrites dans {out}")
