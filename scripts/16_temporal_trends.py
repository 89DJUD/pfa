"""
Phase 5 – Étape 5.3
Évolution temporelle : Z-score moyen & SPEI-12 national (2019–2024)
Panel : 41 entreprises marocaines cotées
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
import matplotlib
matplotlib.use('Agg')  # Pas d'affichage à l'écran, sauvegarde directe

df = pd.read_csv('data/panel_final_merged.csv')
os.makedirs('outputs/figures', exist_ok=True)

years = sorted(df['year'].unique())

# ── Agrégations annuelles ──────────────────────────────────────────────────────
zscore_annual  = df.groupby('year')['zscore'].agg(['mean', 'std', 'median'])
spei_annual    = df.groupby('year')['spei12_mean'].mean()
distress_rate  = df.groupby('year')['in_distress'].mean() * 100
carbon_annual  = df.groupby('year')[['carbon_burden_low',
                                      'carbon_burden_mid',
                                      'carbon_burden_high']].mean()

# ── Figure principale : 2 panneaux ────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)
fig.suptitle('Dynamique temporelle — Panel entreprises cotées marocaines (2019–2024)',
             fontsize=14, fontweight='bold', y=0.98)

# ─ Panneau A : Z-score ────────────────────────────────────────────────────────
ax1.plot(zscore_annual.index, zscore_annual['mean'],
         color='#1F4E79', linewidth=2.5, marker='o', markersize=7,
         label='Z-score moyen', zorder=3)
ax1.plot(zscore_annual.index, zscore_annual['median'],
         color='#2E75B6', linewidth=1.5, marker='s', markersize=5,
         linestyle='--', label='Z-score médian', zorder=3)
ax1.fill_between(zscore_annual.index,
                  zscore_annual['mean'] - zscore_annual['std'],
                  zscore_annual['mean'] + zscore_annual['std'],
                  alpha=0.12, color='#2E75B6', label='± 1 écart-type')

# Zones Altman
ax1.axhspan(2.99, ax1.get_ylim()[1] if ax1.get_ylim()[1] > 2.99 else 6, 
            alpha=0.05, color='green')
ax1.axhspan(1.81, 2.99, alpha=0.05, color='orange')
ax1.axhline(2.99, color='green', linestyle='--', alpha=0.7, linewidth=1.2,
            label='Zone sûre (Z=2.99)')
ax1.axhline(1.81, color='red',   linestyle='--', alpha=0.7, linewidth=1.2,
            label='Zone détresse (Z=1.81)')

ax1.set_ylabel('Z-score Altman', fontsize=12)
ax1.legend(fontsize=9, loc='upper right', ncol=2)
ax1.grid(True, alpha=0.3)
ax1.set_title('A. Évolution du Z-score moyen', fontsize=12, loc='left')

# Annoter les valeurs
for yr, row in zscore_annual.iterrows():
    ax1.annotate(f"{row['mean']:.2f}",
                 xy=(yr, row['mean']),
                 xytext=(0, 10), textcoords='offset points',
                 ha='center', fontsize=8, color='#1F4E79')

# ─ Panneau B : SPEI ───────────────────────────────────────────────────────────
colors_spei = ['#d73027' if v < -1.0 else '#fc8d59' if v < -0.5
               else '#91bfdb' if v < 0.5 else '#4575b4'
               for v in spei_annual.values]

bars = ax2.bar(spei_annual.index, spei_annual.values,
               color=colors_spei, edgecolor='white', linewidth=0.8, width=0.6)
ax2.axhline(0,    color='black', linewidth=1.0)
ax2.axhline(-1.0, color='red',   linestyle=':', alpha=0.7, linewidth=1.2,
            label='Sécheresse modérée (SPEI=−1)')
ax2.axhline(-0.5, color='orange', linestyle=':', alpha=0.5, linewidth=1.0,
            label='Sécheresse légère (SPEI=−0.5)')

# Légende couleurs
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#d73027', label='Sécheresse sévère (< −1)'),
    Patch(facecolor='#fc8d59', label='Sécheresse modérée (−1 à −0.5)'),
    Patch(facecolor='#91bfdb', label='Normal / légèrement humide'),
    Patch(facecolor='#4575b4', label='Humide (> 0.5)'),
]
ax2.legend(handles=legend_elements, fontsize=8, loc='lower right', ncol=2)
ax2.set_ylabel('SPEI-12 moyen national', fontsize=12)
ax2.set_xlabel('Année', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_title('B. Indice de sécheresse SPEI-12 — Maroc', fontsize=12, loc='left')
ax2.set_xticks(years)

# Annoter valeurs SPEI
for yr, val in spei_annual.items():
    ax2.annotate(f"{val:.2f}",
                 xy=(yr, val),
                 xytext=(0, 5 if val >= 0 else -13),
                 textcoords='offset points',
                 ha='center', fontsize=8)

plt.tight_layout()
plt.savefig('outputs/figures/fig_zscore_spei_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sauvegardé : outputs/figures/fig_zscore_spei_trends.png")

# ── Figure 2 : Taux de détresse + charge carbone ──────────────────────────────
fig2, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
fig2.suptitle('Détresse financière et charge carbone simulée (2019–2024)',
              fontsize=13, fontweight='bold')

# Taux de détresse
ax3.plot(distress_rate.index, distress_rate.values,
         color='#C00000', linewidth=2.5, marker='s', markersize=7)
ax3.fill_between(distress_rate.index, 0, distress_rate.values,
                  alpha=0.15, color='#C00000')
for yr, val in distress_rate.items():
    ax3.annotate(f"{val:.0f}%", xy=(yr, val),
                 xytext=(0, 8), textcoords='offset points',
                 ha='center', fontsize=9, color='#C00000')
ax3.set_title('Taux de détresse financière annuel (%)', fontsize=11)
ax3.set_xlabel('Année') ; ax3.set_ylabel('% entreprises en zone détresse')
ax3.set_xticks(years)
ax3.grid(True, alpha=0.3)
ax3.set_ylim(0, max(distress_rate.values) * 1.3)

# Charge carbone par scénario
ax4.plot(carbon_annual.index, carbon_annual['carbon_burden_low']  * 100,
         color='#2ECC71', linewidth=2, marker='o', label='25 $/tCO₂ (bas)')
ax4.plot(carbon_annual.index, carbon_annual['carbon_burden_mid']  * 100,
         color='#F39C12', linewidth=2, marker='s', label='50 $/tCO₂ (médian)')
ax4.plot(carbon_annual.index, carbon_annual['carbon_burden_high'] * 100,
         color='#E74C3C', linewidth=2, marker='^', label='75 $/tCO₂ (haut)')
ax4.set_title('Charge carbone moyenne simulée (% du CA)', fontsize=11)
ax4.set_xlabel('Année') ; ax4.set_ylabel('Charge carbone (% CA)')
ax4.set_xticks(years)
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/fig_distress_carbon_trends.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sauvegardé : outputs/figures/fig_distress_carbon_trends.png")

# ── Résumé console ────────────────────────────────────────────────────────────
print("\n── Résumé temporel ─────────────────────────────────────────────────────")
summary = pd.DataFrame({
    'Z-score moy.'  : zscore_annual['mean'].round(3),
    'SPEI-12 moy.'  : spei_annual.round(3),
    'Taux détresse' : distress_rate.round(1).astype(str) + '%',
}).loc[years]
print(summary.to_string())
