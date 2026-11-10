# Données « expérimentales »

## Contenu actuel : données de DÉMONSTRATION uniquement

`DEMO_brixius_pseudo_mesures.csv` **ne contient pas de mesures**. Ce sont des pseudo-mesures générées par :

1. la corrélation empirique de **Brixius (1987)**, *Traction prediction equations for bias ply tires*,
   ASAE Paper No. 87-1622 (reprise dans ASABE D497) :
   - Bn = (CI·b·d/W)·(1 + 5δ/h)/(1 + 3b/d)
   - NT/W = 0,88(1 − e^(−0,1·Bn))(1 − e^(−7,5·s)) + 0,04 − [1/Bn + 0,04 + 0,5·s/√Bn]
2. avec CI = 600 kPa, δ/h = 0,2, pneu 650/65R38 générique, et une pression de gonflage ajustée à la charge
   (p_i = 120 kPa × W/30 kN) ;
3. plus un bruit gaussien σ = 0,015 (graine 42).

Ce fichier se régénère avec `python scripts/make_demo_data.py`.

Pourquoi ce choix ? Je n'ai pu récupérer aucune courbe traction/glissement publiée **sous forme numérique**.
Les articles donnent les mesures en figures, et je ne les ai pas numérisées. La corrélation de Brixius
a été ajustée sur de nombreux essais au champ. C'est donc une référence **indépendante** du modèle de
Bekker–Wong, ce qui rend la confrontation non triviale. Elle reste un modèle, pas une mesure.

## Utiliser de vraies mesures

1. Numériser une courbe DP/W–glissement publiée, par exemple avec [WebPlotDigitizer](https://automeris.io/WebPlotDigitizer/).
2. Écrire un CSV au même format. En-tête en commentaires `#` avec la référence complète (auteurs,
   année, figure, conditions d'essai : pneu, charge, pression, sol, définition du glissement), puis les
   colonnes `load` [N], `inflation` [Pa], `slip` [-], `net_traction_ratio` [-] et `source`.
3. Relancer le notebook 04 en changeant le nom du fichier.

Attention à la **définition du glissement de référence** (ASABE S296) : elle doit être la même entre
l'essai et le modèle (voir `report.md`, §5.1).
