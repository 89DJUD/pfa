"""
PHASE 5 - ETAPE 5.3 : Evolution temporelle Z-score + SPEI
===========================================================
Input  : data/processed/panel_final_merged.csv
Output : outputs/figures/fig_zscore_spei_trends.png
===========================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(BASE, 'outputs', 'figures')
os.makedirs(FIG, exist_ok=True)

print("=" * 60)
print("  PHASE 5 — ETAPE 5.3 : Evolution temporelle")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'panel_final_merged.csv'))

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.patch.set_facecolor('#FAFAFA')

# ── Panel A : Z-score moyen annuel ───────────────────────────────
zscore_annual = df.groupby('year')['zscore'].agg(['mean', 'std'])

ax1.plot(zscore_annual.index, zscore_annual['mean'],
         color='#1F4E79', linewidth=2.5, marker='o', markersize=5,
         label='Z-score moyen', zorder=5)
ax1.fill_between(zscore_annual.index,
                 zscore_annual['mean'] - zscore_annual['std'],
                 zscore_annual['mean'] + zscore_annual['std'],
                 alpha=0.15, color='#2E75B6', label='±1 écart-type')
ax1.axhline(2.99, color='#27AE60', linestyle='--', alpha=0.7,
            linewidth=1.2, label='Zone sûre (Z=2.99)')
ax1.axhline(1.81, color='#E74C3C', linestyle='--', alpha=0.7,
            linewidth=1.2, label='Zone détresse (Z=1.81)')

ax1.set_ylabel('Z-score Altman', fontsize=12)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_facecolor('#FAFAFA')
ax1.set_title('Evolution du Z-score moyen — Entreprises cotées marocaines', fontsize=13)

# Annoter les événements clés
for yr, evt in [(2020, 'COVID-19'), (2022, 'Sécheresse\nsévère')]:
    if yr in zscore_annual.index:
        ax1.axvline(yr, color='gray', linestyle=':', alpha=0.5, linewidth=1)
        ax1.annotate(evt, xy=(yr, zscore_annual.loc[yr, 'mean']),
                     xytext=(yr + 0.2, zscore_annual.loc[yr, 'mean'] + 0.3),
                     fontsize=8, color='gray',
                     arrowprops=dict(arrowstyle='->', color='gray', lw=0.8))

# ── Panel B : SPEI-12 national ────────────────────────────────────
if 'spei12_mean' in df.columns:
    spei_annual = df.groupby('year')['spei12_mean'].mean()

    colors = ['#d73027' if v <= -1.5 else '#fc8d59' if v <= -1.0
              else '#fdae61' if v <= -0.5 else '#abd9e9' if v <= 0
              else '#4575b4' for v in spei_annual.values]

    ax2.bar(spei_annual.index, spei_annual.values, color=colors,
            edgecolor='white', linewidth=0.5, width=0.7)
    ax2.axhline(0,    color='black', linewidth=1.0)
    ax2.axhline(-1.0, color='#E74C3C', linestyle=':', alpha=0.6,
                linewidth=1.2, label='Sécheresse marquée (SPEI=−1)')
    ax2.axhline(-1.5, color='#8B0000', linestyle=':', alpha=0.5,
                linewidth=1.0, label='Sécheresse sévère (SPEI=−1.5)')

    ax2.set_ylabel('SPEI-12 moyen national', fontsize=12)
    ax2.set_xlabel('Année', fontsize=12)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3, axis='y', linestyle=':')
    ax2.set_facecolor('#FAFAFA')
    ax2.set_title('Indice de sécheresse SPEI-12 — Maroc', fontsize=13)
    ax2.tick_params(axis='x', rotation=45)
else:
    ax2.text(0.5, 0.5, 'spei12_mean non disponible\n(Phase 1 non complétée)',
             ha='center', va='center', transform=ax2.transAxes,
             fontsize=12, color='gray', style='italic')

plt.tight_layout()
out_path = os.path.join(FIG, 'fig_zscore_spei_trends.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  ✓  Figure sauvegardée : outputs/figures/fig_zscore_spei_trends.png")
print(f"  ✓  ETAPE 5.3 terminée → lancez 17_distress_plots.py")