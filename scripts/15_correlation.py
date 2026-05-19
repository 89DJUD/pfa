"""
Phase 5 – Étape 5.2
Matrice de corrélation + test de significativité
Panel : 41 entreprises marocaines cotées | 2019-2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from scipy import stats
import os

df = pd.read_csv('data/panel_final_merged.csv')
os.makedirs('outputs/figures', exist_ok=True)

# ── Variables pour la matrice ─────────────────────────────────────────────────
corr_vars = [
    'zscore', 'spei12_mean', 'drought_months',
    'carbon_burden_mid', 'leverage', 'log_assets', 'roa', 'liquidity'
]

labels = {
    'zscore'            : 'Z-score',
    'spei12_mean'       : 'SPEI-12',
    'drought_months'    : 'Mois séch.',
    'carbon_burden_mid' : 'Charge CO₂',
    'leverage'          : 'Levier',
    'log_assets'        : 'Taille',
    'roa'               : 'ROA',
    'liquidity'         : 'Liquidité',
}

sub = df[corr_vars].rename(columns=labels).dropna()

# ── Matrice de corrélation + p-values ─────────────────────────────────────────
corr_matrix = sub.corr()
n = len(sub)

# Calcul des p-values (t-test bilatéral)
pval_matrix = pd.DataFrame(np.ones_like(corr_matrix), 
                            index=corr_matrix.index, 
                            columns=corr_matrix.columns)
for c1 in corr_matrix.columns:
    for c2 in corr_matrix.columns:
        if c1 != c2:
            r = corr_matrix.loc[c1, c2]
            t = r * np.sqrt(n - 2) / np.sqrt(1 - r**2)
            pval_matrix.loc[c1, c2] = 2 * (1 - stats.t.cdf(abs(t), df=n-2))

print("── Matrice de corrélation ──────────────────────────────────────────────")
print(corr_matrix.round(3).to_string())
print("\n── P-values ────────────────────────────────────────────────────────────")
print(pval_matrix.round(3).to_string())

# Corrélations avec le Z-score (classées)
print("\n── Corrélations avec Z-score (classées) ────────────────────────────────")
zscore_corr = corr_matrix['Z-score'].drop('Z-score').sort_values(key=abs, ascending=False)
for var, r in zscore_corr.items():
    p  = pval_matrix.loc[var, 'Z-score']
    sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
    print(f"  {var:<15} r = {r:+.3f}  p = {p:.3f} {sig}")

# ── Figure : Heatmap ──────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 8))

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

annot_labels = corr_matrix.copy().astype(str)
for c1 in corr_matrix.columns:
    for c2 in corr_matrix.columns:
        r = corr_matrix.loc[c1, c2]
        p = pval_matrix.loc[c1, c2]
        sig = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else ''
        annot_labels.loc[c1, c2] = f"{r:.2f}{sig}"

sns.heatmap(
    corr_matrix, mask=mask,
    annot=annot_labels, fmt='', annot_kws={'size': 9},
    cmap='RdBu_r', center=0, vmin=-1, vmax=1,
    square=True, linewidths=0.5, ax=ax,
    cbar_kws={'label': 'Coefficient de corrélation', 'shrink': 0.8}
)

ax.set_title(
    'Matrice de corrélation — Panel entreprises marocaines (2019–2024)\n'
    '* p<0.10   ** p<0.05   *** p<0.01',
    fontsize=13, pad=15
)
plt.tight_layout()
plt.savefig('outputs/figures/fig_correlation_matrix.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sauvegardé : outputs/figures/fig_correlation_matrix.png")
