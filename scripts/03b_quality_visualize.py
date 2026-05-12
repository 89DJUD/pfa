"""
PHASE 1 - ETAPES 5, 6 & 7 : Vérification qualité + Graphiques
====================================================================
Produit 4 figures dans outputs/figures/ :
1. SPEI national par année (barplot coloré)
2. Heatmap régions × années
3. Distribution du SPEI par région (boxplots)
4. Carte approx. du Maroc avec SPEI moyen par région
====================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import seaborn as sns
import os
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(errors='replace')

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG  = os.path.join(BASE, 'outputs/figures')

# ── Charger les données ───────────────────────────────────────────
climate = pd.read_csv(os.path.join(BASE, 'data/processed/climate_morocco_2000_2024.csv'))
region_ref = pd.read_csv(os.path.join(BASE, 'data/processed/morocco_regions_reference.csv'))

if not {'lat', 'lon'}.issubset(climate.columns):
    climate = climate.merge(
        region_ref[['region', 'lat', 'lon']],
        on='region',
        how='left',
    )

missing_coords = climate.loc[
    climate[['lat', 'lon']].isna().any(axis=1),
    'region',
].unique()
if len(missing_coords) > 0:
    raise ValueError(
        "Coordonnees introuvables pour ces regions : "
        + ", ".join(missing_coords)
    )

print("=" * 55)
print("  ETAPES 5-6-7 : Contrôle qualité + Visualisations")
print("=" * 55)

# ── CONTRÔLE QUALITÉ ─────────────────────────────────────────────
print("\n  ── Contrôle qualité ──────────────────────────")
print(f"  Lignes totales        : {len(climate)}")
print(f"  Valeurs manquantes    :")
print(climate.isnull().sum().to_string())
print(f"\n  SPEI min global       : {climate.spei12_mean.min():.3f}")
print(f"  SPEI max global       : {climate.spei12_mean.max():.3f}")
print(f"  SPEI mean global      : {climate.spei12_mean.mean():.3f}")
print(f"\n  Doublons (region+year): {climate.duplicated(['region','year']).sum()}")
print(f"  Années attendues      : 25  |  Régions : 12  |  Total : 300")
print(f"  Lignes réelles        : {len(climate)} ✓" if len(climate)==300 else "  ⚠ Nombre de lignes inattendu")

years   = sorted(climate['year'].unique())
regions = climate['region'].unique()
national = climate.groupby('year')['spei12_mean'].mean()

# ═══════════════════════════════════════════════════════════════
# FIGURE 1 : SPEI national par année
# ═══════════════════════════════════════════════════════════════
print("\n  Génération Figure 1 : SPEI national...")
fig, ax = plt.subplots(figsize=(14, 6))
fig.patch.set_facecolor('#FAFAFA')

colors = []
for v in national.values:
    if   v <= -1.5: colors.append('#8B0000')
    elif v <= -1.0: colors.append('#D73027')
    elif v <= -0.5: colors.append('#FC8D59')
    elif v <=  0.0: colors.append('#FEE08B')
    elif v <=  0.5: colors.append('#91BFDB')
    else:           colors.append('#4575B4')

bars = ax.bar(national.index, national.values, color=colors,
              edgecolor='white', linewidth=0.6, width=0.8)

ax.axhline(0,    color='#333333', linewidth=0.9)
ax.axhline(-0.5, color='#FC8D59', linewidth=0.8, linestyle='--', alpha=0.6)
ax.axhline(-1.0, color='#D73027', linewidth=1.0, linestyle='--', alpha=0.7)
ax.axhline(-1.5, color='#8B0000', linewidth=1.2, linestyle='--', alpha=0.8)

# Annoter les années extrêmes
for year, val in national.items():
    if abs(val) > 1.2:
        ax.annotate(f'{val:.2f}', xy=(year, val),
                    xytext=(0, -14 if val < 0 else 8), textcoords='offset points',
                    ha='center', fontsize=8.5, color='#333333', fontweight='bold')

legend_patches = [
    mpatches.Patch(color='#8B0000', label='Sécheresse sévère (SPEI ≤ -1.5)'),
    mpatches.Patch(color='#D73027', label='Sécheresse marquée (-1.5 < SPEI ≤ -1.0)'),
    mpatches.Patch(color='#FC8D59', label='Sécheresse modérée (-1.0 < SPEI ≤ -0.5)'),
    mpatches.Patch(color='#FEE08B', label='Légèrement sec (-0.5 < SPEI ≤ 0)'),
    mpatches.Patch(color='#91BFDB', label='Normal à humide (SPEI > 0)'),
    mpatches.Patch(color='#4575B4', label='Humide (SPEI > 0.5)'),
]
ax.legend(handles=legend_patches, loc='upper right', fontsize=8.5,
          framealpha=0.9, ncol=2)

ax.set_xlabel('Année', fontsize=12)
ax.set_ylabel('SPEI-12 moyen national', fontsize=12)
ax.set_title('Indice de sécheresse SPEI-12 — Maroc (2000–2024)\n'
             'Source : calibré sur données BAM, HCP et SPEI Global Drought Monitor',
             fontsize=13, pad=12)
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45, ha='right', fontsize=9)
ax.set_ylim(-2.8, 1.4)
ax.grid(True, axis='y', alpha=0.25, linestyle=':')
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
f1 = os.path.join(FIG, 'fig1_spei_national_2000_2024.png')
plt.savefig(f1, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓  Sauvegardé : {f1}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 2 : Heatmap régions × années
# ═══════════════════════════════════════════════════════════════
print("  Génération Figure 2 : Heatmap...")
pivot = climate.pivot(index='region', columns='year', values='spei12_mean')

# Réordonner les régions du Nord au Sud
order = [
    'Tanger-Tetouan-Al Hoceima', 'Oriental', 'Fes-Meknes',
    'Rabat-Sale-Kenitra', 'Beni Mellal-Khenifra', 'Casablanca-Settat',
    'Marrakech-Safi', 'Draa-Tafilalet', 'Souss-Massa',
    'Guelmim-Oued Noun', 'Laayoune-Sakia El Hamra', 'Dakhla-Oued Ed Dahab',
]
pivot = pivot.reindex([r for r in order if r in pivot.index])

fig, ax = plt.subplots(figsize=(20, 7))
fig.patch.set_facecolor('#FAFAFA')

sns.heatmap(pivot, cmap='RdBu', center=0, vmin=-2.5, vmax=1.5,
            annot=True, fmt='.1f', annot_kws={'size': 7.5},
            linewidths=0.3, linecolor='white',
            cbar_kws={'label': 'SPEI-12', 'shrink': 0.85}, ax=ax)

ax.set_title('SPEI-12 par région et par année — Maroc (2000–2024)\n'
             '← Nord (humide)  …  Sud (aride) →  |  Rouge = sécheresse  |  Bleu = humidité',
             fontsize=13, pad=14)
ax.set_xlabel('Année', fontsize=11)
ax.set_ylabel('')
ax.tick_params(axis='x', rotation=45, labelsize=9)
ax.tick_params(axis='y', rotation=0, labelsize=9)

plt.tight_layout()
f2 = os.path.join(FIG, 'fig2_spei_heatmap_regions.png')
plt.savefig(f2, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓  Sauvegardé : {f2}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 3 : Distribution du SPEI par région (boxplots)
# ═══════════════════════════════════════════════════════════════
print("  Génération Figure 3 : Boxplots...")
fig, ax = plt.subplots(figsize=(13, 7))
fig.patch.set_facecolor('#FAFAFA')

order_box = climate.groupby('region')['spei12_mean'].median().sort_values().index.tolist()
data_box   = [climate[climate['region'] == r]['spei12_mean'].values for r in order_box]
labels_box = [r.replace(' ', '\n') for r in order_box]

bp = ax.boxplot(data_box, vert=True, patch_artist=True,
                medianprops=dict(color='#333333', linewidth=2.0),
                whiskerprops=dict(linewidth=1.2),
                capprops=dict(linewidth=1.2),
                flierprops=dict(marker='o', markersize=4, alpha=0.5))

# Colorier selon la médiane
medians = [np.median(d) for d in data_box]
for patch, med in zip(bp['boxes'], medians):
    if   med <= -1.0: patch.set_facecolor('#FC8D59')
    elif med <= -0.5: patch.set_facecolor('#FEE08B')
    elif med >=  0.3: patch.set_facecolor('#91BFDB')
    else:             patch.set_facecolor('#D9EAD3')

ax.axhline(-1.0, color='#D73027', linewidth=1.0, linestyle='--', alpha=0.7,
           label='Seuil sécheresse (SPEI = -1.0)')
ax.axhline( 0.0, color='#666666', linewidth=0.8, linestyle='-',  alpha=0.4)

ax.set_xticks(range(1, len(order_box)+1))
ax.set_xticklabels(labels_box, fontsize=7.5)
ax.set_ylabel('SPEI-12', fontsize=11)
ax.set_title('Distribution du SPEI-12 par région — Maroc (2000–2024)\n'
             'Régions triées de la plus sèche (gauche) à la plus humide (droite)',
             fontsize=12, pad=12)
ax.legend(fontsize=9)
ax.grid(True, axis='y', alpha=0.25, linestyle=':')
ax.set_facecolor('#FAFAFA')

plt.tight_layout()
f3 = os.path.join(FIG, 'fig3_spei_boxplots_regions.png')
plt.savefig(f3, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓  Sauvegardé : {f3}")

# ═══════════════════════════════════════════════════════════════
# FIGURE 4 : Carte approx. + profil temporel combiné
# ═══════════════════════════════════════════════════════════════
print("  Génération Figure 4 : Carte + profil temporel...")
fig = plt.figure(figsize=(16, 9))
fig.patch.set_facecolor('#FAFAFA')
gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.4, wspace=0.35)

ax_map  = fig.add_subplot(gs[:, 0])   # Carte (colonne gauche)
ax_top  = fig.add_subplot(gs[0, 1])   # Profil Top (colonne droite haut)
ax_bot  = fig.add_subplot(gs[1, 1])   # Profil Bot (colonne droite bas)

# ── Carte approximative ─────────────────────────────────────────
region_data = climate.groupby('region').agg(
    spei_mean=('spei12_mean', 'mean'),
    lat=('lat', 'first'),
    lon=('lon', 'first'),
).reset_index()

spei_vals = region_data['spei_mean'].values
norm_vals = (spei_vals - spei_vals.min()) / (spei_vals.max() - spei_vals.min())

# Fond géographique simplifié (rectangle Maroc)
ax_map.set_facecolor('#D6EAF8')
maroc_outline = plt.Polygon(
    [(-17,27),(-1,27),(-1,36),(-17,36)],
    fill=True, facecolor='#F0F0F0', edgecolor='#999', linewidth=1
)
ax_map.add_patch(maroc_outline)

cmap = plt.cm.RdBu
for _, row in region_data.iterrows():
    val = (row['spei_mean'] - spei_vals.min()) / (spei_vals.max() - spei_vals.min())
    color = cmap(val)
    circle = plt.Circle((row['lon'], row['lat']), 0.8,
                         color=color, alpha=0.85, zorder=5)
    ax_map.add_patch(circle)
    short = row['region'].split('-')[0].split(' ')[0]
    ax_map.annotate(f"{short}\n{row['spei_mean']:+.2f}",
                    xy=(row['lon'], row['lat']),
                    ha='center', va='center', fontsize=6.5,
                    color='#111' if val > 0.4 else '#EEE',
                    fontweight='bold', zorder=6)

ax_map.set_xlim(-18, 0)
ax_map.set_ylim(25, 37)
ax_map.set_xlabel('Longitude', fontsize=9)
ax_map.set_ylabel('Latitude', fontsize=9)
ax_map.set_title('SPEI-12 moyen\n2000–2024\n(rouge = sec, bleu = humide)', fontsize=10)
ax_map.tick_params(labelsize=8)

sm = plt.cm.ScalarMappable(cmap=cmap,
     norm=plt.Normalize(vmin=spei_vals.min(), vmax=spei_vals.max()))
sm.set_array([])
plt.colorbar(sm, ax=ax_map, fraction=0.035, pad=0.04, label='SPEI-12 moyen')

# ── Profil Top 3 régions les plus sèches ─────────────────────────
top_dry = region_data.nsmallest(3, 'spei_mean')['region'].tolist()
colors_dry = ['#8B0000', '#D73027', '#FC8D59']
for reg, col in zip(top_dry, colors_dry):
    sub = climate[climate['region'] == reg].sort_values('year')
    ax_top.plot(sub['year'], sub['spei12_mean'], color=col,
                linewidth=1.8, label=reg.split('-')[0], marker='o', markersize=3)
ax_top.axhline(-1.0, color='gray', linewidth=0.8, linestyle='--', alpha=0.5)
ax_top.axhline( 0.0, color='gray', linewidth=0.6, linestyle='-',  alpha=0.3)
ax_top.set_title('Top 3 régions les plus sèches', fontsize=10)
ax_top.set_ylabel('SPEI-12', fontsize=9)
ax_top.legend(fontsize=8)
ax_top.grid(True, alpha=0.2)
ax_top.tick_params(labelsize=8)
ax_top.set_xticks(years[::4])

# ── Profil Bot 3 régions les plus humides ────────────────────────
top_wet = region_data.nlargest(3, 'spei_mean')['region'].tolist()
colors_wet = ['#4575B4', '#74ADD1', '#ABD9E9']
for reg, col in zip(top_wet, colors_wet):
    sub = climate[climate['region'] == reg].sort_values('year')
    ax_bot.plot(sub['year'], sub['spei12_mean'], color=col,
                linewidth=1.8, label=reg.split('-')[0], marker='o', markersize=3)
ax_bot.axhline(0.0, color='gray', linewidth=0.6, linestyle='-', alpha=0.3)
ax_bot.set_title('Top 3 régions les plus humides', fontsize=10)
ax_bot.set_ylabel('SPEI-12', fontsize=9)
ax_bot.set_xlabel('Année', fontsize=9)
ax_bot.legend(fontsize=8)
ax_bot.grid(True, alpha=0.2)
ax_bot.tick_params(labelsize=8)
ax_bot.set_xticks(years[::4])

fig.suptitle('Analyse spatiale et temporelle du SPEI-12 — Maroc (2000–2024)',
             fontsize=14, fontweight='bold', y=1.01)

f4 = os.path.join(FIG, 'fig4_spei_carte_profils.png')
plt.savefig(f4, dpi=150, bbox_inches='tight')
plt.close()
print(f"  ✓  Sauvegardé : {f4}")

# ═══════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ═══════════════════════════════════════════════════════════════
print("\n" + "=" * 55)
print("  PHASE 1 — RAPPORT FINAL")
print("=" * 55)
print(f"\n  Fichier principal :")
print(f"  data/processed/climate_morocco_2000_2024.csv")
print(f"  → {len(climate)} lignes | {len(climate.columns)} colonnes")
print(f"  → Colonnes : {list(climate.columns)}")
print(f"\n  Figures produites :")
for i, f in enumerate([f1,f2,f3,f4], 1):
    print(f"  {i}. {os.path.basename(f)}")
print(f"\n  ✓✓✓  PHASE 1 COMPLÈTE — Prête pour la Phase 2 ✓✓✓")
