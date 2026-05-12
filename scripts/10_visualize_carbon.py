"""
PHASE 3 - ETAPE 3 : Visualisations de la simulation taxe carbone
====================================================================
Produit 4 figures :
9.  Charge carbone par secteur (3 scénarios)
10. Impact sur le Z-score par scénario
11. Stress-test : entreprises basculant en détresse selon le scénario
12. Exposition sectorielle cumulée (bulle chart)
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(BASE, 'outputs', 'figures')
os.makedirs(FIG, exist_ok=True)

carbon_panel_path = os.path.join(BASE, 'data', 'processed', 'carbon_tax_panel.csv')
if not os.path.exists(carbon_panel_path):
    print("\n  ERREUR : fichier d'entree manquant pour les visualisations")
    print(f"  - {os.path.relpath(carbon_panel_path, BASE)}")
    print("\n  Generez-le d'abord avec :")
    print("  python scripts/09_carbon_cost.py")
    raise SystemExit(1)

df = pd.read_csv(carbon_panel_path)

print("=" * 62)
print("  PHASE 3 — ETAPE 3 : Visualisations taxe carbone")
print("=" * 62)
print(f"\n  Panel chargé : {len(df)} observations, {df['firm_id'].nunique()} entreprises")

SCENARIOS = [
    ('low',  '25 $/tCO₂\n(NDC Maroc)',    '#FEE08B', '#B8860B'),
    ('mid',  '50 $/tCO₂\n(IEA ordonnée)', '#FC8D59', '#D73027'),
    ('high', '75 $/tCO₂\n(IEA 2°C)',      '#D73027', '#8B0000'),
]

# ══════════════════════════════════════════════════════════════════
# FIGURE 9 : Charge carbone par secteur (3 scénarios)
# ══════════════════════════════════════════════════════════════════
print("\n  Génération Figure 9 : Charge carbone par secteur...")

sec_burden = df.groupby('sector').agg(
    burden_low=('carbon_burden_low',  'mean'),
    burden_mid=('carbon_burden_mid',  'mean'),
    burden_high=('carbon_burden_high','mean'),
).sort_values('burden_high', ascending=True) * 100  # en %

sectors = sec_burden.index.tolist()
x       = np.arange(len(sectors))
width   = 0.25

fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('#FAFAFA')

b1 = ax.barh(x - width, sec_burden['burden_low'],  width, color='#FEE08B',
             edgecolor='white', linewidth=0.5, label='25 $/tCO₂ (NDC Maroc)')
b2 = ax.barh(x,          sec_burden['burden_mid'],  width, color='#FC8D59',
             edgecolor='white', linewidth=0.5, label='50 $/tCO₂ (IEA ordonnée)')
b3 = ax.barh(x + width,  sec_burden['burden_high'], width, color='#D73027',
             edgecolor='white', linewidth=0.5, label='75 $/tCO₂ (IEA 2°C SDS)')

for bar in b3:
    w = bar.get_width()
    if w > 0.01:
        ax.text(w + 0.005, bar.get_y() + bar.get_height()/2,
                f'{w:.2f}%', va='center', fontsize=8.5)

ax.set_yticks(x)
ax.set_yticklabels(sectors, fontsize=10)
ax.set_xlabel("Charge carbone moyenne (% du chiffre d'affaires)", fontsize=11)
ax.set_title('Charge carbone par secteur selon les 3 scénarios de taxe\n'
             'Entreprises cotées marocaines — Simulation 2000–2024',
             fontsize=12, pad=12)
ax.legend(fontsize=9, loc='lower right')
ax.grid(True, axis='x', alpha=0.25, linestyle=':')
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
f9 = os.path.join(FIG, 'fig9_carbon_burden_sectors.png')
plt.savefig(f9, dpi=150, bbox_inches='tight')
plt.close()
print(f"  OK {os.path.basename(f9)}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 10 : Impact sur le Z-score — baseline vs scénarios
# ══════════════════════════════════════════════════════════════════
print("  Génération Figure 10 : Impact Z-score par scénario...")

fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=True)
fig.patch.set_facecolor('#FAFAFA')

z_base = df.groupby('year')['zscore'].mean()

for i, (sc, label, col_fill, col_line) in enumerate(SCENARIOS):
    ax   = axes[i]
    z_sc = df.groupby('year')[f'zscore_carbon_{sc}'].mean()
    diff = z_sc - z_base

    ax.fill_between(z_base.index, z_base.values, z_sc.values,
                    alpha=0.35, color=col_fill, label='Ecart vs baseline')
    ax.plot(z_base.index, z_base.values, color='#1F4E79',
            linewidth=2.2, label='Z-score baseline', zorder=5)
    ax.plot(z_sc.index, z_sc.values, color=col_line,
            linewidth=2.0, linestyle='--', label='Z-score avec taxe', zorder=5)

    ax.axhline(2.99, color='#27AE60', linewidth=0.9, linestyle='-.', alpha=0.7)
    ax.axhline(1.81, color='#E74C3C', linewidth=0.9, linestyle='-.', alpha=0.7)

    mean_diff = diff.mean()
    ax.set_title(f'{label}\n(Δ Z-score moyen = {mean_diff:+.3f})',
                 fontsize=10, pad=8)
    ax.set_xlabel('Année', fontsize=9)
    if i == 0:
        ax.set_ylabel('Z-score Altman', fontsize=10)
    ax.legend(fontsize=7.5)
    ax.grid(True, axis='y', alpha=0.2, linestyle=':')
    ax.set_facecolor('#FAFAFA')
    ax.tick_params(axis='x', rotation=45, labelsize=7)

fig.suptitle("Impact de la taxe carbone sur le Z-score moyen — 3 scénarios\n"
             "Entreprises cotées marocaines (2000–2024)",
             fontsize=12, fontweight='bold')
plt.tight_layout()
f10 = os.path.join(FIG, 'fig10_carbon_zscore_impact.png')
plt.savefig(f10, dpi=150, bbox_inches='tight')
plt.close()
print(f"  OK {os.path.basename(f10)}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 11 : Stress-test — taux de détresse baseline vs scénarios
# ══════════════════════════════════════════════════════════════════
print("  Génération Figure 11 : Stress-test détresse...")

distress_yr = pd.DataFrame({
    'Baseline'   : df.groupby('year')['in_distress'].mean()          * 100,
    '25 $/tCO₂'  : df.groupby('year')['distress_carbon_low'].mean()  * 100,
    '50 $/tCO₂'  : df.groupby('year')['distress_carbon_mid'].mean()  * 100,
    '75 $/tCO₂'  : df.groupby('year')['distress_carbon_high'].mean() * 100,
})

fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(14, 10),
                                      gridspec_kw={'height_ratios': [2, 1]})
fig.patch.set_facecolor('#FAFAFA')

colors_lines = ['#1F4E79', '#B8860B', '#D73027', '#8B0000']
styles       = ['-',       '--',      '--',      '--']
widths       = [2.5,       1.8,       1.8,       1.8]

for col, col_, sty, lw in zip(distress_yr.columns, colors_lines, styles, widths):
    ax_top.plot(distress_yr.index, distress_yr[col],
                color=col_, linewidth=lw, linestyle=sty,
                marker='o', markersize=4, label=col)

ax_top.fill_between(distress_yr.index,
                     distress_yr['Baseline'],
                     distress_yr['75 $/tCO₂'],
                     alpha=0.10, color='#D73027',
                     label='Zone de stress additionnel')
ax_top.set_ylabel("% entreprises en détresse financière", fontsize=11)
ax_top.set_title("Stress-test taxe carbone : taux de détresse financière\n"
                  "Entreprises cotées marocaines (2000–2024)",
                  fontsize=12, pad=12)
ax_top.legend(fontsize=9, loc='upper left')
ax_top.grid(True, axis='y', alpha=0.25, linestyle=':')
ax_top.set_facecolor('#FAFAFA')
ax_top.set_ylim(0, 30)

delta    = distress_yr['75 $/tCO₂'] - distress_yr['Baseline']
colors_d = ['#D73027' if v > 0 else '#4575B4' for v in delta.values]
ax_bot.bar(delta.index, delta.values, color=colors_d,
           edgecolor='white', linewidth=0.5, width=0.7)
ax_bot.axhline(0, color='black', linewidth=0.8)
ax_bot.set_ylabel("Δ taux détresse (%)\nvs baseline", fontsize=10)
ax_bot.set_xlabel("Année", fontsize=10)
ax_bot.set_title("Surcroît de détresse induit par la taxe 75 $/tCO₂", fontsize=10)
ax_bot.grid(True, axis='y', alpha=0.2)
ax_bot.set_facecolor('#FAFAFA')
ax_bot.tick_params(axis='x', rotation=45)

plt.tight_layout()
f11 = os.path.join(FIG, 'fig11_stress_test_distress.png')
plt.savefig(f11, dpi=150, bbox_inches='tight')
plt.close()
print(f"  OK {os.path.basename(f11)}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 12 : Double exposition — risque physique × risque de transition
# ══════════════════════════════════════════════════════════════════
print("  Génération Figure 12 : Double exposition SPEI × Carbone...")

climate_path = os.path.join(BASE, 'data', 'processed', 'climate_morocco_2000_2024.csv')

if os.path.exists(climate_path):
    climate  = pd.read_csv(climate_path)
    spei_nat = climate.groupby('year')['spei12_mean'].mean().reset_index()
    spei_nat.columns = ['year', 'spei12_nat']

    df_merged = df.merge(spei_nat, on='year', how='left')
    df_plot   = df_merged.groupby('year').agg(
        zscore_base=('zscore',           'mean'),
        zscore_mid =('zscore_carbon_mid','mean'),
        spei       =('spei12_nat',       'mean'),
        burden_mid =('carbon_burden_mid','mean'),
    ).reset_index()

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#FAFAFA')

    sizes   = df_plot['burden_mid'] * 5_000_000
    scatter = ax.scatter(df_plot['spei'], df_plot['zscore_base'],
                         s=sizes, c=df_plot['spei'],
                         cmap='RdBu', vmin=-2.5, vmax=1.5,
                         alpha=0.75, edgecolors='#333333',
                         linewidths=0.6, zorder=5)

    for _, row in df_plot.iterrows():
        if row['year'] in [2005, 2008, 2010, 2020, 2021, 2022, 2023]:
            ax.annotate(str(int(row['year'])),
                        xy=(row['spei'], row['zscore_base']),
                        xytext=(6, 4), textcoords='offset points',
                        fontsize=8.5, color='#333333', fontweight='bold')

    plt.colorbar(scatter, ax=ax,
                 label='SPEI-12 national (← sec | humide →)', shrink=0.8)

    ax.axhline(2.99, color='#27AE60', linewidth=1.0, linestyle='--',
               alpha=0.7, label='Zone sûre (Z=2.99)')
    ax.axhline(1.81, color='#E74C3C', linewidth=1.0, linestyle='--',
               alpha=0.7, label='Zone détresse (Z=1.81)')
    ax.axvline(-1.0, color='#D73027', linewidth=0.9, linestyle=':',
               alpha=0.6, label='Sécheresse marquée (SPEI=-1)')

    ax.text(-2.3, 1.90, "DOUBLE\nRISQUE\n(sec + carbone)", fontsize=8,
            color='#8B0000', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAECE7', alpha=0.6))
    ax.text(0.3, 3.90, "ZONE\nFAVORABLE", fontsize=8,
            color='#1A5C38', alpha=0.7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='#E9F7EF', alpha=0.6))

    ax.set_xlabel('SPEI-12 moyen national  (← sécheresse | humidité →)', fontsize=11)
    ax.set_ylabel('Z-score moyen (baseline)', fontsize=11)
    ax.set_title('Double exposition climatique : risque physique × risque de transition\n'
                 'Taille des bulles = charge carbone scénario 50$/tCO₂ | Maroc (2000–2024)',
                 fontsize=12, pad=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.2, linestyle=':')
    ax.set_facecolor('#FAFAFA')

    plt.tight_layout()
    f12 = os.path.join(FIG, 'fig12_double_exposure.png')
    plt.savefig(f12, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  OK {os.path.basename(f12)}")
else:
    print(f"  ATTENTION : Figure 12 ignoree : climate_morocco_2000_2024.csv non trouve")
    print(f"  Disponible apres Phase 1 (donnees SPEI)")

# ── Tableau récapitulatif ─────────────────────────────────────────
print("\n  Tableau recapitulatif stress-test")
print("  " + "-" * 68)
print(f"  {'Scenario':<28} {'Z-score moy':>12} {'dZ-score':>10} "
      f"{'Taux detresse':>14} {'d detresse':>11}")
print("  " + "-" * 68)

z_bl = df['zscore'].mean()
d_bl = df['in_distress'].mean() * 100
print(f"  {'Baseline (sans taxe)':<28} {z_bl:>12.3f} {'-':>10} {d_bl:>13.1f}% {'-':>11}")

for sc, label, _, _ in SCENARIOS:
    z_sc = df[f'zscore_carbon_{sc}'].mean()
    d_sc = df[f'distress_carbon_{sc}'].mean() * 100
    lbl  = (
        label.replace('\n', '  ')
        .replace('CO₂', 'CO2')
        .replace('ordonnée', 'ordonnee')
        .replace('°', ' deg')
    )
    print(f"  {lbl:<28} {z_sc:>12.3f} {z_sc-z_bl:>+10.3f} "
          f"{d_sc:>13.1f}% {d_sc-d_bl:>+10.1f}%")

print(f"\n  OK PHASE 3 COMPLETE")
print(f"\n  Fichiers produits :")
print(f"  - data/processed/carbon_tax_panel.csv")
for f in [f9, f10, f11]:
    print(f"  - outputs/figures/{os.path.basename(f)}")
