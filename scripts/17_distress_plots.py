"""
Phase 5 – Étape 5.4
Distribution du Z-score par secteur, région et zone de risque
Panel : 41 entreprises marocaines cotées | 2019-2024
"""

import matplotlib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import os
import matplotlib
matplotlib.use('Agg')  # Pas d'affichage à l'écran, sauvegarde directe

df = pd.read_csv('data/panel_final_merged.csv')
os.makedirs('outputs/figures', exist_ok=True)

# Classification zone Altman
def zscore_zone(z):
    if z >= 2.99: return 'Sûre (Z≥2.99)'
    if z >= 1.81: return 'Grise (1.81–2.99)'
    return 'Détresse (Z<1.81)'

df['zone'] = df['zscore'].apply(zscore_zone)
zone_order  = ['Sûre (Z≥2.99)', 'Grise (1.81–2.99)', 'Détresse (Z<1.81)']
zone_colors = {'Sûre (Z≥2.99)': '#27AE60', 'Grise (1.81–2.99)': '#F39C12',
               'Détresse (Z<1.81)': '#E74C3C'}

sectors = df['sector'].unique()
sector_order = df.groupby('sector')['zscore'].median().sort_values().index.tolist()

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 : Boxplot Z-score par secteur + taux de détresse
# ══════════════════════════════════════════════════════════════════════════════
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Analyse sectorielle du risque financier — Entreprises cotées marocaines',
             fontsize=13, fontweight='bold')

# ─ Boxplot Z-score par secteur ────────────────────────────────────────────────
palette = {s: plt.cm.RdYlGn(i / (len(sectors)-1)) 
           for i, s in enumerate(sector_order)}

df_plot = df.copy()
df_plot['sector'] = pd.Categorical(df_plot['sector'], categories=sector_order, ordered=True)

bp = df_plot.boxplot(column='zscore', by='sector', ax=ax1,
                     vert=False, patch_artist=True,
                     medianprops={'color': 'black', 'linewidth': 2},
                     flierprops={'marker': 'o', 'markersize': 4, 'alpha': 0.5},
                     return_type='dict')

# Colorier les boîtes
for patch, sec in zip(bp['zscore']['boxes'], sector_order):
    patch.set_facecolor(palette[sec])
    patch.set_alpha(0.7)

ax1.axvline(2.99, color='#27AE60', linestyle='--', linewidth=1.5, alpha=0.8, label='Z=2.99 (sûre)')
ax1.axvline(1.81, color='#E74C3C', linestyle='--', linewidth=1.5, alpha=0.8, label='Z=1.81 (détresse)')
ax1.set_title('Distribution du Z-score par secteur', fontsize=11)
ax1.set_xlabel('Z-score Altman')
ax1.set_ylabel('')
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.3, axis='x')
plt.sca(ax1)
plt.title('')  # Enlever le titre automatique du boxplot

# ─ Taux de détresse par secteur ───────────────────────────────────────────────
distress_by_sector = df.groupby('sector').agg(
    taux_detresse=('in_distress', lambda x: x.mean() * 100),
    n=('firm_id', 'count')
).loc[sector_order].reset_index()

colors_bar = ['#E74C3C' if t > 50 else '#F39C12' if t > 25 else '#27AE60'
              for t in distress_by_sector['taux_detresse']]

bars = ax2.barh(distress_by_sector['sector'], distress_by_sector['taux_detresse'],
                color=colors_bar, edgecolor='white', linewidth=0.8)

for bar, (_, row) in zip(bars, distress_by_sector.iterrows()):
    ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
             f"{row['taux_detresse']:.0f}%  (n={row['n']})",
             va='center', fontsize=9)

ax2.set_xlabel('% observations en zone détresse', fontsize=11)
ax2.set_title('Taux de détresse financière par secteur', fontsize=11)
ax2.set_xlim(0, max(distress_by_sector['taux_detresse']) * 1.35)
ax2.grid(True, alpha=0.3, axis='x')
ax2.axvline(50, color='red', linestyle=':', alpha=0.5)

plt.tight_layout()
plt.savefig('outputs/figures/fig_distress_sector.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sauvegardé : outputs/figures/fig_distress_sector.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 : Composition des zones par secteur (stacked bar) + carte régions
# ══════════════════════════════════════════════════════════════════════════════
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(16, 6))
fig2.suptitle('Répartition des zones Altman par secteur et par région',
              fontsize=13, fontweight='bold')

# ─ Stacked bar : zones par secteur ────────────────────────────────────────────
zone_sector = df.groupby(['sector', 'zone']).size().unstack(fill_value=0)
zone_sector = zone_sector.reindex(columns=zone_order, fill_value=0)
zone_pct    = zone_sector.div(zone_sector.sum(axis=1), axis=0) * 100
zone_pct    = zone_pct.loc[sector_order]

bottom = np.zeros(len(zone_pct))
for zone in zone_order:
    vals = zone_pct[zone].values
    ax3.barh(zone_pct.index, vals, left=bottom,
             color=zone_colors[zone], label=zone, edgecolor='white', linewidth=0.5)
    # Annoter si > 10%
    for i, (v, b) in enumerate(zip(vals, bottom)):
        if v > 10:
            ax3.text(b + v/2, i, f"{v:.0f}%", ha='center', va='center',
                     fontsize=8, color='white', fontweight='bold')
    bottom += vals

ax3.set_xlabel('% d\'observations', fontsize=11)
ax3.set_title('Répartition des zones Altman par secteur', fontsize=11)
ax3.legend(loc='lower right', fontsize=9)
ax3.grid(True, alpha=0.2, axis='x')

# ─ Z-score et taux détresse par région ────────────────────────────────────────
region_stats = df.groupby('region').agg(
    zscore_moy    = ('zscore', 'mean'),
    taux_detresse = ('in_distress', lambda x: x.mean() * 100),
    n             = ('firm_id', 'count')
).reset_index().sort_values('zscore_moy')

colors_reg = ['#E74C3C' if t > 50 else '#F39C12' if t > 25 else '#27AE60'
              for t in region_stats['taux_detresse']]

x = np.arange(len(region_stats))
width = 0.35

b1 = ax4.bar(x - width/2, region_stats['zscore_moy'], width,
             color=colors_reg, alpha=0.8, label='Z-score moyen', edgecolor='white')
ax4_twin = ax4.twinx()
b2 = ax4_twin.bar(x + width/2, region_stats['taux_detresse'], width,
                  color='#3498DB', alpha=0.6, label='Taux détresse (%)', edgecolor='white')

ax4.axhline(2.99, color='green', linestyle='--', alpha=0.5, linewidth=1)
ax4.axhline(1.81, color='red',   linestyle='--', alpha=0.5, linewidth=1)
ax4.set_xticks(x)
ax4.set_xticklabels(region_stats['region'], rotation=15, ha='right', fontsize=9)
ax4.set_ylabel('Z-score moyen', fontsize=11, color='#2C3E50')
ax4_twin.set_ylabel('Taux de détresse (%)', fontsize=11, color='#3498DB')
ax4.set_title('Z-score et détresse par région', fontsize=11)

lines = [mpatches.Patch(color=c, label=l) 
         for c, l in [('#2C3E50', 'Z-score moyen'), ('#3498DB', 'Taux détresse (%)'),
                      ('green', 'Z=2.99 (sûre)'), ('red', 'Z=1.81 (détresse)')]]
ax4.legend(handles=lines, fontsize=8, loc='upper left')

plt.tight_layout()
plt.savefig('outputs/figures/fig_distress_analysis.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sauvegardé : outputs/figures/fig_distress_analysis.png")

# ── Résumé console ────────────────────────────────────────────────────────────
print("\n── Zones Altman par secteur ────────────────────────────────────────────")
print(zone_sector.loc[sector_order].to_string())
print("\n── Zones Altman par région ─────────────────────────────────────────────")
print(df.groupby(['region', 'zone']).size().unstack(fill_value=0).to_string())
