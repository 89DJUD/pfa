"""
PHASE 5 - ETAPE 5.2 : Matrice de corrélation
=============================================
Input  : data/processed/panel_final_merged.csv
Output : outputs/figures/fig_correlation_matrix.png
=============================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE  = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG   = os.path.join(BASE, 'outputs', 'figures')
os.makedirs(FIG, exist_ok=True)

print("=" * 60)
print("  PHASE 5 — ETAPE 5.2 : Matrice de corrélation")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'panel_final_merged.csv'))

# ── Variables pour la matrice ─────────────────────────────────────
corr_vars = ['zscore', 'spei12_mean', 'drought_months',
             'carbon_burden_mid', 'leverage', 'log_assets', 'roa', 'liquidity']
corr_vars = [v for v in corr_vars if v in df.columns]

corr_matrix = df[corr_vars].corr()

# ── Labels lisibles ───────────────────────────────────────────────
labels = {
    'zscore'           : 'Z-score',
    'spei12_mean'      : 'SPEI-12 moy.',
    'drought_months'   : 'Mois sécheresse',
    'carbon_burden_mid': 'Charge carbone',
    'leverage'         : 'Levier',
    'log_assets'       : 'Log(Actif)',
    'roa'              : 'ROA',
    'liquidity'        : 'Liquidité',
}
corr_matrix.index   = [labels.get(c, c) for c in corr_matrix.index]
corr_matrix.columns = [labels.get(c, c) for c in corr_matrix.columns]

# ── Heatmap ───────────────────────────────────────────────────────
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(10, 8))
fig.patch.set_facecolor('#FAFAFA')

sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
            cmap='RdBu_r', center=0, vmin=-1, vmax=1,
            square=True, linewidths=0.5, ax=ax,
            cbar_kws={'shrink': 0.8},
            annot_kws={'size': 10})

ax.set_title('Matrice de corrélation — Panel entreprises marocaines\n'
             f'({len(df)} observations, {df["year"].min()}–{df["year"].max()})',
             fontsize=13, pad=15)
ax.tick_params(axis='x', rotation=45, labelsize=10)
ax.tick_params(axis='y', rotation=0, labelsize=10)
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
out_path = os.path.join(FIG, 'fig_correlation_matrix.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

# ── Afficher les corrélations significatives ──────────────────────
print(f"\n  Corrélations avec le Z-score (|r| > 0.2) :")
z_corr = df[corr_vars].corr()['zscore'].drop('zscore').abs().sort_values(ascending=False)
for var, r in z_corr[z_corr > 0.2].items():
    direction = "+" if df[corr_vars].corr()['zscore'][var] > 0 else "−"
    print(f"  {labels.get(var, var):<25} r = {direction}{r:.3f}")

print(f"\n  ✓  Figure sauvegardée : outputs/figures/fig_correlation_matrix.png")
print(f"  ✓  ETAPE 5.2 terminée → lancez 16_temporal_trends.py")