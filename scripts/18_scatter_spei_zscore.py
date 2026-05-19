"""
Phase 5 – Étape 5.5
Nuages de points : SPEI vs Z-score (H1) + Charge carbone vs Z-score (H2)
Panel : 41 entreprises marocaines cotées | 2019-2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from scipy import stats
import os
import matplotlib
matplotlib.use('Agg')  # Pas d'affichage à l'écran, sauvegarde directe
df = pd.read_csv('data/panel_final_merged.csv')
os.makedirs('outputs/figures', exist_ok=True)

sectors      = sorted(df['sector'].unique())
cmap         = plt.cm.tab10
sector_colors = {s: cmap(i / len(sectors)) for i, s in enumerate(sectors)}

def add_regression_line(ax, x, y, color='black'):
    """Trace la droite de régression et affiche les statistiques."""
    mask = ~(np.isnan(x) | np.isnan(y))
    xc, yc = x[mask], y[mask]
    slope, intercept, r, p, se = stats.linregress(xc, yc)
    x_line = np.linspace(xc.min(), xc.max(), 200)
    ax.plot(x_line, slope * x_line + intercept,
            color=color, linewidth=2, linestyle='--', zorder=5,
            label=f'Régression : r={r:.2f}, p={p:.3f}')
    return r, p, slope

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1 : SPEI vs Z-score  (Hypothèse H1)
# ══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle('Hypothèse H1 — Risque physique (SPEI) et détresse financière (Z-score)',
             fontsize=13, fontweight='bold')

# ─ Panneau gauche : scatter par secteur ────────────────────────────────────
ax = axes[0]
for sec in sectors:
    sub = df[df['sector'] == sec]
    ax.scatter(sub['spei12_mean'], sub['zscore'],
               alpha=0.55, s=40, color=sector_colors[sec], label=sec, zorder=3)

r, p, slope = add_regression_line(ax, df['spei12_mean'].values, df['zscore'].values)

# Zones Altman
ymin, ymax = ax.get_ylim()
ax.axhline(2.99, color='#27AE60', linestyle=':', alpha=0.7, linewidth=1.2, label='Z=2.99 (sûre)')
ax.axhline(1.81, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=1.2, label='Z=1.81 (détresse)')
ax.axvline(0,    color='gray',    linestyle=':', alpha=0.4, linewidth=1.0)
ax.axvline(-1.0, color='orange',  linestyle=':', alpha=0.5, linewidth=1.0, label='SPEI=−1 (sécheresse)')

ax.set_xlabel('SPEI-12 moyen régional  ← Sécheresse | Humidité →', fontsize=11)
ax.set_ylabel('Z-score Altman', fontsize=11)
ax.set_title('SPEI vs Z-score — par secteur', fontsize=11)
ax.legend(fontsize=7, ncol=2, loc='upper left')
ax.grid(True, alpha=0.2)

# Annotation résumé
direction = "positif" if slope > 0 else "négatif"
sig_txt   = "significatif" if p < 0.05 else "non significatif (p>0.05)"
ax.text(0.98, 0.05, f"H1 — lien {direction}\nr={r:.2f}  p={p:.3f}  {sig_txt}",
        transform=ax.transAxes, ha='right', va='bottom', fontsize=9,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow', alpha=0.8))

# ─ Panneau droit : scatter par année ───────────────────────────────────────
ax2 = axes[1]
year_cmap   = plt.cm.viridis
years       = sorted(df['year'].unique())
year_colors = {yr: year_cmap(i / (len(years)-1)) for i, yr in enumerate(years)}

for yr in years:
    sub = df[df['year'] == yr]
    sc = ax2.scatter(sub['spei12_mean'], sub['zscore'],
                     alpha=0.65, s=50, color=year_colors[yr], label=str(yr), zorder=3)

add_regression_line(ax2, df['spei12_mean'].values, df['zscore'].values)

ax2.axhline(2.99, color='#27AE60', linestyle=':', alpha=0.7, linewidth=1.2)
ax2.axhline(1.81, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=1.2)
ax2.axvline(0,    color='gray',    linestyle=':', alpha=0.4, linewidth=1.0)
ax2.set_xlabel('SPEI-12 moyen régional  ← Sécheresse | Humidité →', fontsize=11)
ax2.set_ylabel('Z-score Altman', fontsize=11)
ax2.set_title('SPEI vs Z-score — par année', fontsize=11)
ax2.legend(fontsize=9, title='Année', title_fontsize=9)
ax2.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('outputs/figures/fig_spei_zscore_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sauvegardé : outputs/figures/fig_spei_zscore_scatter.png")

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 : Charge carbone vs Z-score  (Hypothèse H2)
# ══════════════════════════════════════════════════════════════════════════════
fig2, axes2 = plt.subplots(1, 3, figsize=(18, 6))
fig2.suptitle('Hypothèse H2 — Risque de transition (charge carbone) et Z-score',
              fontsize=13, fontweight='bold')

scenarios  = [('carbon_burden_low',  '25 $/tCO₂ (bas)',    '#2ECC71'),
              ('carbon_burden_mid',  '50 $/tCO₂ (médian)', '#F39C12'),
              ('carbon_burden_high', '75 $/tCO₂ (haut)',   '#E74C3C')]

for ax, (col, title, color) in zip(axes2, scenarios):
    for sec in sectors:
        sub = df[df['sector'] == sec]
        ax.scatter(sub[col] * 100, sub['zscore'],
                   alpha=0.55, s=40, color=sector_colors[sec], label=sec, zorder=3)

    r, p, slope = add_regression_line(ax, df[col].values * 100, df['zscore'].values, color='black')

    ax.axhline(2.99, color='#27AE60', linestyle=':', alpha=0.6, linewidth=1.2)
    ax.axhline(1.81, color='#E74C3C', linestyle=':', alpha=0.6, linewidth=1.2)
    ax.set_xlabel('Charge carbone (% du CA)', fontsize=10)
    ax.set_ylabel('Z-score Altman', fontsize=10)
    ax.set_title(f'Scénario {title}', fontsize=11)
    ax.grid(True, alpha=0.2)

    sig_txt = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.1 else "n.s."
    ax.text(0.98, 0.98, f"r={r:.2f}  {sig_txt}",
            transform=ax.transAxes, ha='right', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

# Légende commune
handles = [mlines.Line2D([0], [0], marker='o', color='w',
           markerfacecolor=sector_colors[s], markersize=8, label=s)
           for s in sectors]
fig2.legend(handles=handles, loc='lower center', ncol=5, fontsize=8,
            title='Secteur', title_fontsize=9, bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.07, 1, 1])
plt.savefig('outputs/figures/fig_carbon_zscore_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("✅ Sauvegardé : outputs/figures/fig_carbon_zscore_scatter.png")

# ── Résumé console ────────────────────────────────────────────────────────────
print("\n── Corrélations principales (Phase 5 – résumé) ─────────────────────────")
for xvar, label in [('spei12_mean', 'SPEI-12 → Z-score (H1)'),
                     ('carbon_burden_mid', 'Charge CO₂ → Z-score (H2)'),
                     ('spei_x_carbon', 'Interaction → Z-score (H3 preview)')]:
    mask = ~(df[xvar].isna() | df['zscore'].isna())
    r, p = stats.pearsonr(df.loc[mask, xvar], df.loc[mask, 'zscore'])
    sig  = '***' if p < 0.01 else '**' if p < 0.05 else '*' if p < 0.1 else 'n.s.'
    print(f"  {label:<35} r = {r:+.3f}   p = {p:.4f}  {sig}")
