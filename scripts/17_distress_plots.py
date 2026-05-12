"""
PHASE 5 - ETAPE 5.4 : Distribution Z-score par secteur + taux détresse
========================================================================
Input  : data/processed/panel_final_merged.csv
Output : outputs/figures/fig_distress_analysis.png
========================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(BASE, 'outputs', 'figures')
os.makedirs(FIG, exist_ok=True)

print("=" * 60)
print("  PHASE 5 — ETAPE 5.4 : Analyse de la détresse financière")
print("=" * 60)

df = pd.read_csv(os.path.join(BASE, 'data', 'processed', 'panel_final_merged.csv'))

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.patch.set_facecolor('#FAFAFA')

# ── Graphique 1 : Boxplot Z-score par secteur ─────────────────────
if 'sector' in df.columns:
    sector_order = df.groupby('sector')['zscore'].median().sort_values().index.tolist()
    data_by_sector = [df[df['sector'] == s]['zscore'].dropna().values
                      for s in sector_order]

    bp = axes[0].boxplot(data_by_sector, vert=False, patch_artist=True,
                         labels=sector_order,
                         boxprops=dict(facecolor='#B5D4F4', color='#185FA5'),
                         medianprops=dict(color='#FC8D59', linewidth=2),
                         whiskerprops=dict(color='#185FA5'),
                         capprops=dict(color='#185FA5'),
                         flierprops=dict(marker='o', color='#185FA5',
                                         alpha=0.4, markersize=4))

    axes[0].axvline(1.81, color='#E74C3C', linestyle='--', alpha=0.8,
                    linewidth=1.2, label='Z=1.81 (détresse)')
    axes[0].axvline(2.99, color='#27AE60', linestyle='--', alpha=0.8,
                    linewidth=1.2, label='Z=2.99 (zone sûre)')
    axes[0].set_title('Distribution du Z-score par secteur', fontsize=12)
    axes[0].set_xlabel('Z-score Altman', fontsize=11)
    axes[0].legend(fontsize=9)
    axes[0].grid(True, axis='x', alpha=0.3, linestyle=':')
    axes[0].set_facecolor('#FAFAFA')
    axes[0].tick_params(axis='y', labelsize=9)
else:
    axes[0].text(0.5, 0.5, 'Colonne sector non disponible',
                 ha='center', va='center', transform=axes[0].transAxes,
                 fontsize=11, color='gray', style='italic')

# ── Graphique 2 : Taux de détresse par année ─────────────────────
if 'in_distress' in df.columns:
    distress_rate = df.groupby('year')['in_distress'].mean() * 100

    axes[1].plot(distress_rate.index, distress_rate.values,
                 color='#C00000', linewidth=2.5, marker='s', markersize=6, zorder=5)
    axes[1].fill_between(distress_rate.index, 0, distress_rate.values,
                         alpha=0.15, color='#C00000')

    for yr, val in distress_rate.items():
        axes[1].annotate(f'{val:.1f}%',
                         xy=(yr, val), xytext=(0, 8),
                         textcoords='offset points',
                         ha='center', fontsize=9, color='#8B0000', fontweight='bold')

    axes[1].set_title('Taux de détresse financière annuel (%)', fontsize=12)
    axes[1].set_xlabel('Année', fontsize=11)
    axes[1].set_ylabel('% entreprises en zone détresse (Z < 1.81)', fontsize=11)
    axes[1].grid(True, alpha=0.3, linestyle=':')
    axes[1].set_facecolor('#FAFAFA')
    axes[1].set_ylim(0, distress_rate.max() * 1.3)
    axes[1].tick_params(axis='x', rotation=45)
else:
    axes[1].text(0.5, 0.5, 'Colonne in_distress non disponible',
                 ha='center', va='center', transform=axes[1].transAxes,
                 fontsize=11, color='gray', style='italic')

plt.tight_layout()
out_path = os.path.join(FIG, 'fig_distress_analysis.png')
plt.savefig(out_path, dpi=150, bbox_inches='tight')
plt.close()

print(f"\n  ✓  Figure sauvegardée : outputs/figures/fig_distress_analysis.png")
print(f"  ✓  ETAPE 5.4 terminée → lancez 18_scatter_spei_zscore.py")