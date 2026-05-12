"""
PHASE 5 - ETAPE 5.5 : Nuage de points SPEI vs Z-score (illustre H1)
=====================================================================
Input  : data/processed/panel_final_merged.csv
Output : outputs/figures/fig_spei_zscore_scatter.png
=====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(BASE, 'outputs', 'figures')
os.makedirs(FIG, exist_ok=True)

print("=" * 60)
print("  PHASE 5 — ETAPE 5.5 : Nuage de points SPEI vs Z-score")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'panel_final_merged.csv'))

if 'spei12_mean' not in df.columns:
    print("  ⚠️  spei12_mean non disponible — Phase 1 requise")
    exit()

df_clean = df[['spei12_mean', 'zscore', 'sector']].dropna()

fig, ax = plt.subplots(figsize=(11, 7))
fig.patch.set_facecolor('#FAFAFA')

# ── Scatter par secteur ────────────────────────────────────────────
if 'sector' in df_clean.columns:
    sectors = df_clean['sector'].unique()
    cmap    = plt.cm.tab10(np.linspace(0, 1, len(sectors)))
    for sec, col in zip(sectors, cmap):
        sub = df_clean[df_clean['sector'] == sec]
        ax.scatter(sub['spei12_mean'], sub['zscore'],
                   alpha=0.45, s=25, color=col, label=sec, zorder=3)
else:
    ax.scatter(df_clean['spei12_mean'], df_clean['zscore'],
               alpha=0.45, s=25, color='#378ADD', zorder=3)

# ── Droite de régression ──────────────────────────────────────────
slope, intercept, r, p, se = stats.linregress(
    df_clean['spei12_mean'], df_clean['zscore'])

x_line = np.linspace(df_clean['spei12_mean'].min(),
                     df_clean['spei12_mean'].max(), 100)
ax.plot(x_line, slope * x_line + intercept,
        color='black', linewidth=2, linestyle='--', zorder=5,
        label=f'Régression (r={r:.2f}, p={p:.3f})')

print(f"\n  Résultats régression SPEI → Z-score :")
print(f"  Pente (β)    : {slope:.4f}")
print(f"  Intercept    : {intercept:.4f}")
print(f"  r de Pearson : {r:.4f}")
print(f"  p-value      : {p:.4f} {'*** (sig. à 1%)' if p < 0.01 else '** (sig. à 5%)' if p < 0.05 else '(non significatif)'}")

# ── Zones Z-score ─────────────────────────────────────────────────
ax.axhline(2.99, color='#27AE60', linestyle=':', alpha=0.6,
           linewidth=1.2, label='Zone sûre (Z=2.99)')
ax.axhline(1.81, color='#E74C3C', linestyle=':', alpha=0.6,
           linewidth=1.2, label='Zone détresse (Z=1.81)')
ax.axvline(0,   color='gray', linestyle=':', alpha=0.4, linewidth=1)
ax.axvline(-1.0, color='#FC8D59', linestyle=':', alpha=0.5,
           linewidth=1.0, label='Sécheresse marquée (SPEI=−1)')

# ── Quadrant annotations ──────────────────────────────────────────
xlim = ax.get_xlim()
ylim = ax.get_ylim()
ax.text(xlim[0] + 0.1, 1.85,
        "Sécheresse + détresse\n(double vulnérabilité)",
        fontsize=8, color='#8B0000', alpha=0.7,
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#FAECE7', alpha=0.5))

ax.set_xlabel('SPEI-12 moyen régional  (← sécheresse | humidité →)', fontsize=12)
ax.set_ylabel('Z-score Altman', fontsize=12)
ax.set_title(f'Relation SPEI-12 — Z-score : Entreprises cotées marocaines\n'
             f'({len(df_clean)} observations, H1 : SPEI↑ → Z↑)',
             fontsize=13)
ax.legend(fontsize=8, ncol=2, loc='upper left')
ax.grid(True, alpha=0.2, linestyle=':')
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
out_path = os.path.join(FIG, 'fig_spei_zscore_scatter.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  ✓  Figure sauvegardée : outputs/figures/fig_spei_zscore_scatter.png")
print(f"\n{'=' * 60}")
print(f"  ✅  PHASE 5 COMPLÈTE")
print(f"{'=' * 60}")
print(f"\n  Fichiers produits :")
print(f"  - outputs/tables/table1_stats_desc.xlsx")
print(f"  - outputs/figures/fig_correlation_matrix.png")
print(f"  - outputs/figures/fig_zscore_spei_trends.png")
print(f"  - outputs/figures/fig_distress_analysis.png")
print(f"  - outputs/figures/fig_spei_zscore_scatter.png")
print(f"\n  → Prochaine étape : Phase 6 — Régression panel à effets fixes")