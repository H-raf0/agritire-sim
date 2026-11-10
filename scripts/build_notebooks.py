"""Construit (et exécute avec --execute) les 4 notebooks du projet.

Usage :
    python scripts/build_notebooks.py            # écrit les .ipynb
    python scripts/build_notebooks.py --execute  # écrit puis exécute (figures incluses)
"""
from __future__ import annotations

import sys
from pathlib import Path

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parents[1]
NB_DIR = ROOT / "notebooks"

SETUP = """\
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from agritire import plotting
plotting.setup()
pd.set_option("display.precision", 3)
"""


def md(s):
    return new_markdown_cell(s.strip("\n"))


def code(s):
    return new_code_cell(s.strip("\n"))


# ---------------------------------------------------------------------------
NB1 = [
    md(r"""
# 01 — Traction d'un pneu agricole sur sol meuble

**Objectif** : prédire l'effort au crochet (*drawbar pull*) DP d'un pneu 650/65R38 générique en
fonction du glissement $s$, pour plusieurs sols et pressions de gonflage.

**Modèle** (détails et sources dans `agritire/terramechanics.py` et `report.md`) :

| Brique | Équation | Source |
|---|---|---|
| Pression–enfoncement | $\sigma = (k_c/b + k_\varphi)\,z^n$ | Bekker (1969) |
| Cisaillement | $\tau = (c + \sigma\tan\varphi)\,(1 - e^{-j/K})$ | Janosi & Hanamoto (1961) |
| Déplacement de cisaillement sur l'arc | $j(\theta) = R[(\theta_1-\theta) - (1-s)(\sin\theta_1 - \sin\theta)]$ | Wong & Reece (1967) |
| Équilibre vertical | $W = Rb\int(\sigma\cos\theta + \tau\sin\theta)\,d\theta + b\int_{\text{méplat}}\sigma\,d\xi$ | — |
| Effort au crochet | $DP = H - R_c$ | Wong (2008) |

Le pneu est **rigide** si $p_i + p_c \ge p_{gcr}$ (pression sous une roue rigide équivalente), sinon il
**s'écrase** : contact = arc avant + méplat à la pression $p_g = p_i + p_c$ (Bekker 1969 ; Wong 2008).

**Hypothèses principales** : régime permanent, sol homogène, pas de bulldozing ni de rebond élastique du
sol, pas d'effet des crampons, pas d'hystérésis interne du pneu, $p_c = 20$ kPa (hypothèse).
"""),
    code(SETUP + """
from agritire.soil import SOILS
from agritire.terramechanics import (Tire, traction, traction_curve, critical_ground_pressure,
                                     rigid_sinkage, rigid_wheel_sinkage_bekker)
tire = Tire()
W = 30e3   # charge par roue [N] (~3 t)
MODE_FR = {"flexible": "souple", "rigid": "rigide"}
tire
"""),
    md("""
## 1. Paramètres de sol (Wong, 2008)
$k_c$ en kN/m$^{n+1}$, $k_\\varphi$ en kN/m$^{n+2}$, $c$ en kPa. **$K$ est une hypothèse** (ordre de
grandeur 1–2,5 cm cité pour les sols meubles), recalée au notebook 04.
"""),
    code("""
pd.DataFrame([dict(clé=k, sol=s.name, n=s.n, kc=s.kc, kphi=s.kphi, c_kPa=s.c, phi_deg=s.phi_deg,
                   K_m=s.K, source=s.source) for k, s in SOILS.items()]).set_index("clé")
"""),
    md("""
## 2. Pression critique et mode de fonctionnement
Pour $W$ = 30 kN, on compare $p_g = p_i + p_c$ à la pression critique $p_{gcr}$. On vérifie aussi
l'enfoncement de la roue rigide obtenu par intégration numérique contre la formule approchée de Bekker
$z_0 = [3W/((3-n)(k_c + b k_\\varphi)\\sqrt{D})]^{2/(2n+1)}$.
"""),
    code("""
rows = []
for k, s in SOILS.items():
    z_num = rigid_sinkage(tire, s, W)
    z_bek = rigid_wheel_sinkage_bekker(s, W, tire.diameter, tire.width)
    rows.append(dict(sol=k, p_gcr_kPa=critical_ground_pressure(tire, s, W) / 1e3,
                     z_rigide_num_cm=100 * z_num, z_rigide_Bekker_cm=100 * z_bek,
                     ecart_pct=100 * (z_num / z_bek - 1)))
pd.DataFrame(rows).set_index("sol")
"""),
    md("""
L'écart numérique/analytique est de quelques % : la formule de Bekker approxime l'arc par une parabole,
elle est donc d'autant moins précise que l'enfoncement est grand. C'est une **vérification croisée**,
pas une validation physique.

## 3. Distribution des contraintes le long du contact
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
soil = SOILS["sandy_loam"]
for ax, p_i in zip(axes, (80e3, 250e3)):
    r, prof = traction(tire.with_(inflation=p_i), soil, W, 0.15, profile=True)
    ax.plot(prof.x * 100, prof.sigma / 1e3, label="σ normale")
    ax.plot(prof.x * 100, prof.tau / 1e3, label="τ cisaillement")
    ax.axvline(0, color="k", lw=0.6)
    ax.set_title(f"p_i = {p_i/1e3:.0f} kPa → mode {MODE_FR[r.mode]}\\nz0 = {100*r.sinkage:.1f} cm, δ = {100*r.deflection:.1f} cm")
    ax.set_xlabel("x depuis l'aplomb du moyeu [cm] (avant →)")
axes[0].set_ylabel("contrainte [kPa]")
axes[0].legend()
fig.suptitle("Limon sableux, W = 30 kN, s = 15 %", y=1.03)
plotting.save(fig, "01_contraintes_contact");
"""),
    md("""
- À **basse pression** (gauche), le pneu s'écrase : un méplat à pression quasi uniforme $p_g$ s'ajoute à
  l'arc d'entrée. L'enfoncement est plus faible et le contact plus long.
- À **haute pression** (droite), le pneu se comporte comme une roue rigide : la contrainte normale est
  concentrée sous le moyeu, l'ornière est plus profonde.
- $\\tau$ croît de l'avant vers l'arrière du contact car le déplacement de cisaillement $j$ s'accumule.

## 4. Courbes traction / glissement pour quatre sols
"""),
    code("""
slips = np.linspace(0, 0.5, 51)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
for k, s in SOILS.items():
    df = traction_curve(tire, s, W, slips)
    axes[0].plot(100 * df.slip, df.net_traction_ratio, label=s.name)
    axes[1].plot(100 * df.slip, df.tractive_efficiency, label=s.name)
axes[0].set(xlabel="glissement s [%]", ylabel="DP / W [-]", title="Coefficient de traction nette")
axes[1].set(xlabel="glissement s [%]", ylabel="η [-]", title="Rendement de traction")
axes[0].axhline(0, color="k", lw=0.6)
axes[0].legend(fontsize=8)
fig.suptitle(f"p_i = {tire.inflation/1e3:.0f} kPa, W = 30 kN", y=1.02)
plotting.save(fig, "01_traction_sols");
"""),
    md("""
Lecture d'ingénieur :
- l'effort au crochet sature avec le glissement (limite de Mohr–Coulomb $cA + W\\tan\\varphi$) ;
- le **rendement** passe par un maximum vers 5–15 % de glissement : c'est la plage de fonctionnement
  visée en pratique (au-delà, l'énergie part dans le cisaillement du sol, en deçà la résistance de
  compaction pèse relativement plus) ;
- le **sable sec** (cohésion quasi nulle, enfoncement important) et le **sol argileux à φ = 13°** sont les
  plus défavorables.

**Glissement nul ≠ traction nulle.** À $s = 0$ défini par rapport au rayon libre $R$, la cinématique de
l'arc produit déjà un déplacement de cisaillement $j(\\theta) = R[(\\theta_1-\\theta) - (\\sin\\theta_1-\\sin\\theta)]
= O(\\theta_1^3)$ : le point de roulement libre (DP = 0) n'est pas exactement à $s = 0$. Pour comparer à
des essais, il faut donc utiliser la **même définition du glissement de référence** (ASABE S296 : roue
autopropulsée sur sol dur). Le modèle d'empreinte plane donne, lui, exactement $H(0) = 0$ et
$DP(0) = -R_c$ (vérifié dans `tests/`).

## 5. Effet de la pression de gonflage
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
soil = SOILS["sandy_loam"]
for p_i in (60e3, 100e3, 160e3, 240e3):
    df = traction_curve(tire.with_(inflation=p_i), soil, W, slips)
    axes[0].plot(100 * df.slip, df.net_traction_ratio, label=f"{p_i/1e3:.0f} kPa ({MODE_FR[df['mode'].iloc[15]]})")
pressures = np.linspace(50e3, 300e3, 26)
res = [traction(tire.with_(inflation=p), soil, W, 0.15) for p in pressures]
axes[1].plot(pressures / 1e3, [100 * r.sinkage for r in res], label="enfoncement z0 [cm]")
axes[1].plot(pressures / 1e3, [100 * r.net_traction_ratio for r in res], label="DP/W à 15 % [%]")
pcr = critical_ground_pressure(tire, soil, W, 0.15) - tire.carcass_pressure
axes[1].axvline(pcr / 1e3, color="k", ls="--", lw=0.8)
axes[1].text(pcr / 1e3 + 3, 5, "p_gcr − p_c\\n(passage en mode rigide)", fontsize=8)
axes[0].set(xlabel="glissement s [%]", ylabel="DP / W [-]", title="Limon sableux, W = 30 kN")
axes[1].set(xlabel="pression de gonflage p_i [kPa]", title="Enfoncement et traction à s = 15 %")
axes[0].legend(fontsize=8); axes[1].legend(fontsize=8)
plotting.save(fig, "01_effet_pression");
"""),
    md("""
Baisser la pression augmente la surface de contact, réduit l'enfoncement (donc la résistance de
compaction $R_c = b k z_0^{n+1}/(n+1)$) et allonge la zone de cisaillement : la traction nette augmente.
Au-delà de $p_{gcr}$, le pneu est « rigide » et la pression n'a plus d'effet. Le modèle est **continu** à
la transition (les deux modes coïncident quand l'écrasement δ → 0), ce que vérifient les tests.

## 6. Comparaison de trois formulations
"""),
    code("""
fig, ax = plt.subplots(figsize=(6, 3.8))
soil = SOILS["sandy_loam"]
for mode, lab in (("auto", "arc + méplat (modèle principal)"), ("flat", "empreinte plane (Bekker)"),
                  ("wong_reece", "roue rigide Wong–Reece")):
    df = traction_curve(tire, soil, W, np.linspace(0, 0.8, 41), mode=mode)
    ax.plot(100 * df.slip, df.net_traction_ratio, label=lab)
ax.axhline(0, color="k", lw=0.6)
ax.set(xlabel="glissement s [%]", ylabel="DP / W [-]", title="Limon sableux, p_i = 120 kPa, W = 30 kN")
ax.legend(fontsize=8)
plotting.save(fig, "01_comparaison_modeles");
"""),
    md("""
- L'**empreinte plane** ignore l'arc d'entrée : à $s = 0$, $DP = -R_c$ exactement.
- **Wong–Reece** (roue rigide, maximum de contrainte avancé avec le glissement) prédit un
  enfoncement par patinage (*slip-sinkage*) et donc un maximum d'effort au crochet suivi d'une
  décroissance — comportement observé expérimentalement sur sols meubles.
- L'écart entre formulations (≈ 0,1 sur DP/W) est du même ordre que l'incertitude sur les paramètres de
  sol : **le choix du modèle est en soi une source d'incertitude** à afficher.
"""),
]

# ---------------------------------------------------------------------------
NB2 = [
    md(r"""
# 02 — Usure sur route : modèle de brosse + loi d'Archard

Sur route, un pneu agricole transmet un effort de traction $F_x$ (transport, remorque). Une partie de
l'empreinte **glisse** sur le revêtement : c'est là que la gomme s'use.

1. **Empreinte** : longueur $L = F_z/(b\,p_g)$, pression parabolique $p(x)$ avec $\int p\,b\,dx = F_z$.
2. **Brosse** (Pacejka, 2012) : poils de raideur $c_p$ ; en adhérence $u = s\,x$, glissement dès que
   $c_p u > \mu p(x)$ ; vitesse de glissement $v_g/V = s - du/dx$.
3. **Archard** (1953) : $dh = k_A\,p\,dS_g$ ⇒ volume usé par mètre parcouru
   $\dfrac{dV}{dX} = k_A\, b \displaystyle\int p(x)\,\frac{v_g}{V}\,dx$.

**Hypothèses** : petit glissement, $\mu = 0{,}8$ constant (pas de dépendance à la vitesse/température),
$C_s/F_z = 6$ (hypothèse, à identifier sur banc), $k_A$ **inconnu** → résultats en **indice relatif**.
"""),
    code(SETUP + """
from agritire.wear import (RoadContact, brush_state, pacejka_parabolic_force, archard_wear,
                           slip_for_force)
from agritire.terramechanics import Tire
tire = Tire(); Fz = 30e3
c = RoadContact(Fz, tire.width, tire.ground_pressure)
print(f"L = {c.length:.3f} m ; c_p = {c.bristle_stiffness:.3e} N/m³")
"""),
    md("## 1. Vérification : modèle de brosse numérique vs solution analytique de Pacejka"),
    code("""
s = np.linspace(0, 0.25, 60)
num = [brush_state(c, si).force for si in s]
ana = [pacejka_parabolic_force(c, si) for si in s]
fig, ax = plt.subplots(figsize=(6, 3.6))
ax.plot(100 * s, np.array(ana) / Fz, lw=3, alpha=0.4, label="analytique (Pacejka)")
ax.plot(100 * s, np.array(num) / Fz, "--", label="numérique (marche avant→arrière)")
ax.set(xlabel="glissement s [%]", ylabel="F_x / F_z [-]", title="Brosse à pression parabolique")
ax.legend()
plotting.save(fig, "02_verification_brosse")
print("écart relatif max :", np.max(np.abs(np.array(num[1:]) - ana[1:]) / np.array(ana[1:])))
"""),
    md("## 2. Zones d'adhérence et de glissement dans l'empreinte"),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=True)
for ax, ratio in zip(axes, (0.1, 0.3)):
    w = archard_wear(c, ratio * Fz, keep_state=True)
    st = w.state
    ax.plot(st.x / c.length, c.mu * st.p / 1e3, "k--", lw=1, label="μ p (limite de Coulomb)")
    ax.plot(st.x / c.length, st.q / 1e3, label="q = c_p u (contrainte tangentielle)")
    ax.fill_between(st.x / c.length, 0, st.q / 1e3, where=st.sliding, alpha=0.3, color="C1", label="glissement")
    ax.set_title(f"F_x/F_z = {ratio}  →  s = {100*w.slip:.2f} %, {100*w.sliding_fraction:.0f} % de l'empreinte glisse")
    ax.set_xlabel("position x / L (0 = avant)")
axes[0].set_ylabel("contrainte [kPa]"); axes[0].legend(fontsize=8)
plotting.save(fig, "02_zones_glissement");
"""),
    md("""
La zone de glissement apparaît à l'**arrière** de l'empreinte (là où la pression, donc la limite
d'adhérence, chute) et s'étend vers l'avant quand $F_x$ augmente.

**Bilan d'énergie** : en régime établi, la puissance de glissement $F_x\\,s\\,V$ est entièrement dissipée
par frottement dans la zone glissante, donc $E_f = F_x\\,s$ [J/m]. C'est une vérification indépendante
de l'intégration (testée dans `tests/test_wear.py`).
"""),
    code("""
for ratio in (0.05, 0.1, 0.2, 0.3):
    w = archard_wear(RoadContact(Fz, tire.width, tire.ground_pressure, n_points=100001), ratio * Fz)
    print(f"F_x/F_z = {ratio:4.2f} : E_f = {w.friction_energy:8.2f} J/m ; F_x·s = {ratio*Fz*w.slip:8.2f} J/m")
"""),
    md("## 3. Sensibilité de l'usure à l'effort, à la pression et à la charge"),
    code("""
ref = archard_wear(c, 0.1 * Fz).archard_integral
ratios = np.linspace(0.02, 0.4, 30)
wear_fx = [archard_wear(c, r * Fz).archard_integral / ref for r in ratios]
pressures = np.linspace(60e3, 240e3, 19)
wear_p = [archard_wear(RoadContact(Fz, tire.width, p + tire.carcass_pressure), 0.1 * Fz).archard_integral / ref
          for p in pressures]
loads = np.linspace(15e3, 45e3, 16)
# à effort de traction constant (3 kN) : la charge seule varie
wear_w = [archard_wear(RoadContact(Wi, tire.width, tire.ground_pressure), 0.1 * Fz).archard_integral / ref
          for Wi in loads]
fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
axes[0].loglog(ratios, wear_fx); axes[0].set(xlabel="F_x / F_z", ylabel="indice d'usure (réf. = 1)", title="Effort de traction")
slope = np.polyfit(np.log(ratios[:10]), np.log(wear_fx[:10]), 1)[0]
axes[0].text(0.03, 3, f"pente faible effort ≈ {slope:.1f}", fontsize=9)
axes[1].plot(pressures / 1e3, wear_p); axes[1].set(xlabel="p_i [kPa]", title="Pression (F_x = 3 kN)")
axes[2].plot(loads / 1e3, wear_w); axes[2].set(xlabel="F_z [kN]", title="Charge (F_x = 3 kN)")
for ax in axes[1:]: ax.set_ylabel("indice d'usure")
plotting.save(fig, "02_sensibilites_usure");
"""),
    md("""
Lecture :
- l'usure croît **très fortement** avec l'effort transmis (pente ≈ 3 en log-log aux faibles efforts :
  $s \\propto F_x$ et la zone glissante s'étend), d'où l'intérêt de limiter le patinage sur route
  (lestage, répartition des efforts, pression adaptée) ;
- à effort donné, l'effet de la **pression est nul par construction** dans ce modèle : avec une
  pression parabolique et $C_s/F_z$ fixé, la solution de la brosse ne dépend que de $\\theta s$ avec
  $\\theta = C_s/(3\\mu F_z)$ ; raccourcir l'empreinte raidit les poils d'autant (courbe plate). C'est une
  **limite à afficher**, pas un résultat : en réalité $C_s$ dépend de la pression et de la charge, et
  doit être mesurée sur banc pour que le modèle soit prédictif ;
- à effort constant, **augmenter la charge** réduit le glissement nécessaire, donc l'usure ;
- l'indice est **relatif** : sans essai d'usure de référence pour identifier $k_A$, un volume absolu
  (mm³/km) n'aurait aucune valeur prédictive.

**Ce que le modèle ne capture pas** : les crampons (flexion et frottement des pavés en entrée/sortie
d'empreinte, principal mécanisme d'usure des pneus agricoles sur route), les différences de rayon de
roulement dans la largeur, la température, l'abrasivité du revêtement.
"""),
]

# ---------------------------------------------------------------------------
NB3 = [
    md(r"""
# 03 — Éléments finis 2D : pression de contact sous la bande de roulement

**Problème** : coupe transversale (y, z) de la bande de roulement au centre de l'empreinte, en
**déformation plane**, bicouche élastique linéaire (ceinture raide + gomme) posée sur une **fondation de
Winkler** en contact unilatéral.

**Formulation faible** : trouver $\mathbf u \in V$ tel que $\forall \mathbf v \in V_0$,
$$\int_\Omega \boldsymbol\sigma(\mathbf u):\boldsymbol\varepsilon(\mathbf v)\,d\Omega + \int_{\Gamma_c} k_w\,u_z v_z\,d\Gamma
= -\int_{\Gamma_{top}} p_i\,v_z\,d\Gamma - \int_{\Gamma_{ép}} q_s\,v_z\,d\Gamma$$
avec $\boldsymbol\sigma = \lambda\,\mathrm{tr}(\boldsymbol\varepsilon)\mathbf I + 2\mu\boldsymbol\varepsilon$.

**Conditions aux limites** : symétrie $u_y = 0$ en $y = 0$ (demi-modèle) ; face latérale libre ;
ressorts verticaux sous la gomme (compression seule) ; pression $p_i$ sur la face intérieure ; effort de
flanc $p_c\,b/2$ réparti sur une bande d'épaule de 5 cm.

**Éléments** : triangles P2 (quadratiques), qui limitent le verrouillage volumique pour ν = 0,45.

**Hypothèses fortes** : élasticité linéaire (le caoutchouc est hyperélastique, quasi incompressible),
pas de crampons, pas de courbure du sommet, pas de précontrainte de ceinture liée au gonflage, sol de
Winkler (pas de couplage latéral), propriétés matériaux = ordres de grandeur.
"""),
    code(SETUP + """
from agritire.fem_contact import (TreadSection, solve_section, winkler_modulus_from_bekker,
                                  full_width_profile, ROAD_FOUNDATION_MODULUS)
from agritire.soil import SOILS
sec = TreadSection()
kw_soil = winkler_modulus_from_bekker(SOILS["sandy_loam"], 140e3, sec.width)
print(f"k_w sol (limon sableux, sécant à 140 kPa) = {kw_soil:.2e} N/m³ ; route = {ROAD_FOUNDATION_MODULUS:.0e} N/m³")
print(f"raideur de la gomme E/h ≈ {sec.E_tread/sec.tread_thickness:.1e} N/m³")
"""),
    md("## 1. Vérification : couche homogène sous charge uniforme (solution analytique exacte)"),
    code("""
E, nu, p, kw = 6e6, 0.45, 100e3, 1e8
homog = TreadSection(E_tread=E, nu_tread=nu, E_belt=E, nu_belt=nu, n_y=10)
r = solve_section(homog, p, 0.0, kw)
expected = -p / kw - p * homog.height * (1 - nu**2) / E
print(f"pression de contact : min {r.pressure.min():.6f} Pa, max {r.pressure.max():.6f} Pa (attendu {p:.0f})")
print(f"u_z sommet : FEM {r.top_displacement:.9e} m ; analytique {expected:.9e} m")
print(f"équilibre : réactions {r.reaction:.6f} N/m ; appliqué {r.applied:.6f} N/m")
"""),
    md("""
Le champ exact ($\\sigma_{zz} = -p$, $\\sigma_{yy} = 0$) est linéaire en déplacement, donc reproduit
exactement par des éléments P2 : c'est un *patch test*. L'équilibre global est vérifié à la précision
machine (somme des réactions des ressorts = effort appliqué).

## 2. Maillage et déformée
"""),
    code("""
res = solve_section(sec, 100e3, 20e3, ROAD_FOUNDATION_MODULUS)
m = res.basis.mesh
u = res.displacement
ux = u[res.basis.nodal_dofs[0]]; uz = u[res.basis.nodal_dofs[1]]
scale = 0.25 * sec.height / np.abs(uz).max()
fig, axes = plt.subplots(2, 1, figsize=(10, 4.2), sharex=True)
axes[0].triplot(m.p[0] * 100, m.p[1] * 100, m.t.T, lw=0.3, color="0.4")
axes[0].axhline(sec.tread_thickness * 100, color="C1", lw=1)
axes[0].set(ylabel="z [cm]", title="Maillage (demi-section) — ceinture au-dessus de la ligne orange")
axes[1].tripcolor((m.p[0] + scale * ux) * 100, (m.p[1] + scale * uz) * 100, m.t.T, uz * 1e3, shading="gouraud", cmap="viridis")
axes[1].set(xlabel="y [cm] (0 = axe de symétrie)", ylabel="z [cm]", title=f"Déformée amplifiée ×{scale:.0f}, couleur = u_z [mm] (route, p_i = 100 kPa)")
for ax in axes: ax.set_aspect("equal")
plotting.save(fig, "03_maillage_deformee");
"""),
    md("## 3. Convergence en maillage"),
    code("""
rows = []
for ny in (13, 26, 65, 130, 260):
    s_ = TreadSection(n_y=ny, n_z_tread=max(4, ny // 8))
    r_ = solve_section(s_, 100e3, 20e3, ROAD_FOUNDATION_MODULUS)
    rows.append(dict(n_y=ny, h_mm=1000 * s_.width / 2 / ny, pic_sur_moyenne=r_.peak_to_mean,
                     ddl=r_.basis.N))
conv = pd.DataFrame(rows); conv["ecart_vs_plus_fin_pct"] = 100 * (conv.pic_sur_moyenne / conv.pic_sur_moyenne.iloc[-1] - 1)
conv
"""),
    md("""
Le pic de pression converge (écart < 1 % pour $n_y = 65$, valeur retenue). Répartir l'effort de flanc
sur une bande de largeur finie est indispensable : appliqué sur une arête, il créerait une singularité
et un pic qui croîtrait indéfiniment avec le raffinement.

## 4. Pression de contact selon la pression de gonflage : route vs sol meuble
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), sharey=True)
summary = []
for ax, (kw, lab) in zip(axes, ((ROAD_FOUNDATION_MODULUS, "route"), (kw_soil, "sol meuble (limon sableux)"))):
    for p_i in (60e3, 120e3, 240e3):
        r = solve_section(sec, p_i, 20e3, kw)
        y, pr = full_width_profile(r)
        ax.plot(y * 100, pr / r.mean_pressure, label=f"p_i = {p_i/1e3:.0f} kPa")
        summary.append(dict(support=lab, p_i_kPa=p_i / 1e3, p_moy_kPa=r.mean_pressure / 1e3,
                            p_max_kPa=r.pressure.max() / 1e3, pic_sur_moyenne=r.peak_to_mean,
                            iterations_ensemble_actif=r.iterations))
    ax.axhline(1, color="k", lw=0.6)
    ax.set(title=lab, xlabel="position latérale y [cm]")
axes[0].set_ylabel("pression / pression moyenne [-]"); axes[0].legend(fontsize=8)
plotting.save(fig, "03_pression_laterale")
pd.DataFrame(summary)
"""),
    md("""
Lecture :
- la pression moyenne vaut exactement $p_i + p_c$ (équilibre), ce qui justifie l'hypothèse
  $p_g = p_i + p_c$ du module de terramécanique ;
- **à basse pression**, la part de charge portée par les flancs ($p_c$) est relativement plus grande :
  **surpression aux épaules** (≈ 2× la moyenne sur route à 60 kPa) ;
- **sur sol meuble**, le support déformable lisse la distribution (pic plus faible) : la ceinture
  répartit la charge sur une plus grande largeur quand le support est souple (longueur caractéristique
  de poutre sur fondation élastique $\\lambda = (4EI/k)^{1/4}$).

## 5. Couplage FEM → usure : profil latéral d'usure
On découpe la largeur en bandes, chacune portant $w_i\\,p(x)$ (poids issus du FEM sur route), avec un
glissement commun (ceinture rigide).
"""),
    code("""
from agritire.wear import lateral_wear_profile
from scipy.interpolate import interp1d
fig, ax = plt.subplots(figsize=(6, 3.6))
for p_i in (60e3, 120e3, 240e3):
    r = solve_section(sec, p_i, 20e3, ROAD_FOUNDATION_MODULUS)
    y, pr = full_width_profile(r)
    yc = np.linspace(-sec.width / 2, sec.width / 2, 27)
    weights = interp1d(y, pr)(yc)
    s, wear = lateral_wear_profile(30e3, sec.width, p_i + 20e3, 3e3, weights)
    ax.plot(yc * 100, wear / wear.mean(), label=f"p_i = {p_i/1e3:.0f} kPa (s = {100*s:.2f} %)")
ax.set(xlabel="position latérale y [cm]", ylabel="usure locale / moyenne [-]", title="Profil latéral d'usure (F_x = 3 kN)")
ax.legend(fontsize=8)
plotting.save(fig, "03_profil_usure_lateral");
"""),
    md("""
**Résultat non trivial** : malgré une surpression de 2× aux épaules, le profil d'usure ne varie que de
quelques %. Avec un glissement uniforme, une bande plus chargée a une limite d'adhérence $\\mu p$ plus
haute, donc une zone glissante plus courte ; les deux effets se compensent presque.

Conclusion d'ingénieur : dans ce cadre, **la distribution de pression seule n'explique pas l'usure
irrégulière** épaules/centre observée en pratique. Il faudrait modéliser ce qui fait varier le
**glissement** dans la largeur : rayon de roulement local (courbure du sommet), dérive, carrossage,
mouvement des crampons. C'est une limite explicite du modèle, pas un résultat à extrapoler.
"""),
]

# ---------------------------------------------------------------------------
NB4 = [
    md(r"""
# 04 — Plan d'expériences numérique et recalage sur données

**Partie A — DOE** : facteurs pression de gonflage (60–200 kPa), largeur (0,45–0,85 m, rapport d'aspect
65 %), charge (20–45 kN), type de sol (3 niveaux). Réponses : DP/W à 15 % de glissement, rendement,
enfoncement, coefficient de résistance au roulement, énergie de frottement sur route (indice d'usure).

**Partie B — Recalage** des paramètres de cisaillement $(c, \varphi, K)$ par moindres carrés non
linéaires (`scipy.optimize.least_squares`).

> ⚠️ **Données** : aucune courbe traction/glissement expérimentale n'a pu être récupérée sous forme
> numérique. Le fichier `data/experimental/DEMO_brixius_pseudo_mesures.csv` contient des
> **pseudo-mesures de démonstration** générées par la corrélation empirique de **Brixius (1987)**
> (+ bruit). Ce ne sont **pas** des mesures ; la démarche est identique avec de vraies données.
"""),
    code(SETUP + """
from agritire.doe import (full_factorial, latin_hypercube, run_plan, main_effects, sensitivity_table,
                          FACTORS, SOIL_LEVELS, DEFLECTION_LIMIT)
import time
"""),
    md("## A1. Plan factoriel complet 3³ × 3 sols (81 calculs)"),
    code("""
t0 = time.time()
ff = run_plan(full_factorial(levels=3))
print(f"{len(ff)} calculs en {time.time()-t0:.1f} s ; modes : {ff['mode'].value_counts().to_dict()}")
print(f"configurations hors domaine réaliste (écrasement route > {DEFLECTION_LIMIT:.0%} de la hauteur de flanc) : {(~ff.feasible).sum()}")
ff[~ff.feasible][["inflation", "width", "load", "soil", "road_deflection_ratio"]].drop_duplicates(["inflation", "width", "load"])
"""),
    code("""
labels = {"inflation": "p_i [kPa]", "width": "b [m]", "load": "W [kN]"}
conv = {"inflation": 1e-3, "width": 1.0, "load": 1e-3}
fig, axes = plt.subplots(2, 3, figsize=(12, 6.2), sharey="row")
for row, resp, ylab in ((0, "ntr", "DP/W à s = 15 %"), (1, "sinkage", "enfoncement [cm]")):
    for ax, f in zip(axes[row], FACTORS):
        for soil, grp in ff.groupby("soil"):
            me = grp.groupby(f)[resp].mean()
            yv = me.values * (100 if resp == "sinkage" else 1)
            ax.plot(me.index * conv[f], yv, "o-", label=soil)
        ax.set_xlabel(labels[f])
    axes[row][0].set_ylabel(ylab)
axes[0][0].legend(fontsize=8)
fig.suptitle("Effets principaux (plan factoriel, moyenne sur les autres facteurs)", y=1.0)
plotting.save(fig, "04_effets_principaux");
"""),
    md("""
- La **pression** est le levier principal sur l'enfoncement et la traction (cohérent avec la pratique
  du « basse pression » au champ).
- La **charge** augmente l'enfoncement et, à pression fixée, augmente légèrement DP/W : l'aire de
  contact croît avec la charge alors que la résistance de compaction ne dépend que de $p_g$ en mode
  souple. Les essais réels (et la corrélation de Brixius, partie B) montrent plutôt l'inverse, car la
  pression y est ajustée à la charge et la pression au sol réelle n'est pas uniforme : c'est une limite
  du modèle à pression de contact uniforme.
- La **largeur**, à pression fixée, a un effet faible voire défavorable : même enfoncement mais ornière
  plus large ($R_c \\propto b$). Dans ce modèle, l'intérêt d'un pneu large passe **par la pression plus
  basse qu'il autorise** pour une même charge (tables charge/pression), pas par la largeur seule. Les
  deux facteurs sont donc couplés en pratique, alors que le DOE les traite comme indépendants : point à
  garder en tête en lisant les sensibilités.

## A2. Hypercube latin (100 points par sol) et analyse de sensibilité
"""),
    code("""
t0 = time.time()
lhs = run_plan(latin_hypercube(100, seed=0))
print(f"{len(lhs)} calculs en {time.time()-t0:.1f} s")
sens = sensitivity_table(lhs)
piv = sens.pivot_table(index=["response", "soil"], columns="factor", values=["SRC", "R2"]).round(2)
piv[[("SRC", "inflation"), ("SRC", "width"), ("SRC", "load"), ("R2", "load")]].rename(columns={("R2", "load"): ("R²", "")})
"""),
    code("""
fig, axes = plt.subplots(1, 4, figsize=(14, 3.6), sharey=True)
for ax, resp, title in zip(axes, ["ntr", "sinkage", "rr_coeff", "road_energy"],
                            ["DP/W (15 %)", "enfoncement", "résistance au roulement", "énergie de frottement route"]):
    sub = sens[sens.response == resp]
    x = np.arange(len(FACTORS)); wbar = 0.27
    for i, soil in enumerate(SOIL_LEVELS):
        vals = sub[sub.soil == soil].set_index("factor").loc[list(FACTORS), "SRC"]
        ax.bar(x + (i - 1) * wbar, vals, wbar, label=soil)
    r2 = sub.groupby("soil").R2.first().min()
    ax.set_xticks(x, ["p_i", "b", "W"]); ax.axhline(0, color="k", lw=0.6)
    ax.set_title(f"{title}\\n(R² min = {r2:.2f})")
axes[0].set_ylabel("coefficient de régression standardisé"); axes[0].legend(fontsize=8)
plotting.save(fig, "04_sensibilites_SRC");
"""),
    md("""
Les SRC donnent le poids relatif de chaque facteur **sur la plage étudiée** ; ils ne sont fiables que
si le modèle linéaire explique l'essentiel de la variance ($R^2$ affiché). Un $R^2$ plus faible signale
des non-linéarités / interactions (ex. changement de mode souple → rigide) : il faudrait alors une
méthode globale (indices de Sobol). Les corrélations de rang de Spearman (dans `sens`) confirment le
sens des effets.

L'énergie de frottement sur route ne dépend pas du sol et, dans ce modèle, elle ne dépend **que** de la
charge (SRC = 1) : avec $F_x = 0{,}1\\,W$ et $C_s/F_z$ fixé, $E_f = F_x\\,s$ où $s$ ne dépend que de
$F_x/F_z$, donc $E_f \\propto W$. Pression et largeur n'ont aucun effet par construction (voir
notebook 02) : la sensibilité de l'usure à la pression ne pourra être étudiée qu'avec une rigidité de
glissement $C_s(p_i, W)$ mesurée.

## B1. Données de démonstration
"""),
    code("""
from pathlib import Path
from agritire.calibration import load_dataset, fit_soil, predict, brixius_net_traction
from agritire.soil import SOILS
from agritire.terramechanics import Tire
tire = Tire()
data = load_dataset(Path("..") / "data" / "experimental" / "DEMO_brixius_pseudo_mesures.csv")
print(data.source.unique())
data.head()
"""),
    md("""
## B2. Vérification de la procédure : retrouver des paramètres connus
On génère des données **avec notre propre modèle** (paramètres « vrais » connus) + bruit de même niveau
(σ = 0,015), puis on recale. Si l'on ne retrouve pas les paramètres ici, il est inutile d'aller plus loin.
"""),
    code("""
base = SOILS["sandy_loam"]
truth = base.with_(c=8.0, phi_deg=27.0, K=0.03)
rng = np.random.default_rng(1)
synth = data.copy()
synth["net_traction_ratio"] = predict(truth, tire, data) + rng.normal(0, 0.015, len(data))
fit_v = fit_soil(synth, tire, base)
pd.DataFrame({"vrai": [truth.c, truth.phi_deg, truth.K], "recalé": fit_v.params.values,
              "écart-type": fit_v.std_errors.values}, index=fit_v.params.index).assign(
    ecart_en_sigma=lambda d: (d["recalé"] - d["vrai"]) / d["écart-type"])
"""),
    md("""
Les paramètres sont retrouvés à environ 1–2 écarts-types : la procédure (modèle + optimiseur + estimation
d'incertitude) est **vérifiée**.

## B3. Identifiabilité : pourquoi il faut plusieurs couples charge/pression
Si toutes les mesures sont faites **à pression constante**, l'aire de contact $A = W/p_g$ est
proportionnelle à la charge, et la poussée maximale $cA + W\\tan\\varphi = W\\,(c/p_g + \\tan\\varphi)$ ne
permet pas de séparer $c$ et $\\varphi$.
"""),
    code("""
same_p = synth.copy(); same_p["inflation"] = 120e3
same_p["net_traction_ratio"] = predict(truth, tire, same_p) + rng.normal(0, 0.015, len(same_p))
fit_same = fit_soil(same_p, tire, base)
print("corrélation c–φ, pression constante  :", round(fit_same.correlation.loc["c", "phi_deg"], 3))
print("corrélation c–φ, pression ∝ charge   :", round(fit_v.correlation.loc["c", "phi_deg"], 3))
print("écarts-types (pression constante) :", fit_same.std_errors.round(3).to_dict())
"""),
    md("""
Avec une pression unique, $c$ et $\\varphi$ sont presque parfaitement corrélés (incertitudes énormes).
**Le plan d'essais conditionne ce qu'on peut identifier** : c'est un argument pour concevoir la campagne
expérimentale en même temps que le modèle.

## B4. Confrontation à la référence empirique de Brixius (pseudo-mesures)
"""),
    code("""
fit_free = fit_soil(data, tire, base)
fit_bnd = fit_soil(data, tire, base, bounds={"c": (0.0, 25.0)})
tab = pd.DataFrame({
    "Wong (tableau)": [base.c, base.phi_deg, base.K, np.nan, np.nan],
    "recalage libre": [*fit_free.params.values, fit_free.rmse, fit_free.r2],
    "± (1σ)": [*fit_free.std_errors.values, np.nan, np.nan],
    "recalage c ≤ 25 kPa": [*fit_bnd.params.values, fit_bnd.rmse, fit_bnd.r2],
}, index=["c [kPa]", "φ [°]", "K [m]", "RMSE", "R²"])
tab
"""),
    code("""
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
sfine = np.linspace(0.0, 0.36, 60)
for i, (W, grp) in enumerate(data.groupby("load")):
    col = plotting.COLORS[i]
    axes[0].plot(100 * grp.slip, grp.net_traction_ratio, "o", color=col, label=f"pseudo-mesures W = {W/1e3:.0f} kN")
    curve = pd.DataFrame(dict(load=W, inflation=grp.inflation.iloc[0], slip=sfine))
    axes[0].plot(100 * sfine, predict(fit_free.soil, tire, curve), "-", color=col)
    axes[0].plot(100 * sfine, predict(base, tire, curve), ":", color=col, alpha=0.8)
    axes[1].plot(100 * grp.slip, fit_free.residuals[grp.index], "o-", color=col, label=f"W = {W/1e3:.0f} kN")
axes[0].plot([], [], "k-", label="modèle recalé (libre)"); axes[0].plot([], [], "k:", label="modèle, paramètres Wong")
axes[0].set(xlabel="glissement s [%]", ylabel="DP / W [-]", title="Recalage sur pseudo-mesures Brixius (DÉMO)")
axes[1].axhline(0, color="k", lw=0.6)
axes[1].axhspan(-0.015, 0.015, color="0.85", label="±σ du bruit ajouté")
axes[1].set(xlabel="glissement s [%]", ylabel="résidu modèle − mesure", title="Résidus (recalage libre)")
axes[0].legend(fontsize=7); axes[1].legend(fontsize=8)
plotting.save(fig, "04_recalage");
fit_free.correlation.round(2)
"""),
    md("""
**Analyse critique** :
1. Le recalage libre donne un très bon $R^2$, mais une **cohésion de plusieurs dizaines de kPa**, alors que
   Wong donne 1,7 kPa pour ce limon sableux. Le modèle **compense une erreur de structure** par un paramètre
   non physique : un bon $R^2$ ne valide pas les paramètres.
2. Les **résidus sont structurés** (modèle − mesure positif aux faibles glissements, surtout aux fortes
   charges, et aux forts glissements à 20 kN) : le modèle surestime la traction à faible glissement (effet de l'arc d'entrée et de la définition du glissement
   de référence) et ne reproduit pas toute la baisse de DP/W avec la charge que donne Brixius (qui intègre
   des effets de pneu : hystérésis, déformation de carcasse).
3. Contraindre $c$ à une plage plausible dégrade le RMSE : l'écart restant mesure l'**erreur de modèle**.
4. Les paramètres d'enfoncement ($n, k_c, k_\\varphi$) ne sont pas recalés : ils se mesurent par essai de plaque
   (bevameter) et sont mal identifiables à partir des seules courbes de traction.

**Prochaine étape avec de vraies mesures** : remplacer le CSV par des courbes numérisées (source citée),
mesurer les paramètres de sol au bevameter, et réserver une partie des essais à la **validation** (non
utilisée pour le recalage).
"""),
]


def build(execute: bool) -> None:
    NB_DIR.mkdir(exist_ok=True)
    books = {"01_traction_sol_meuble": NB1, "02_usure_route": NB2,
             "03_fem_empreinte": NB3, "04_doe_et_recalage": NB4}
    for name, cells in books.items():
        nb = new_notebook(cells=cells, metadata={
            "kernelspec": {"name": "python3", "display_name": "Python 3", "language": "python"},
            "language_info": {"name": "python"}})
        path = NB_DIR / f"{name}.ipynb"
        if execute:
            from nbclient import NotebookClient
            NotebookClient(nb, timeout=900, kernel_name="python3", record_timing=False,
                           resources={"metadata": {"path": str(NB_DIR)}}).execute()
        nbformat.write(nb, path)
        print(("exécuté + écrit : " if execute else "écrit : ") + str(path.relative_to(ROOT)))


if __name__ == "__main__":
    build("--execute" in sys.argv)
