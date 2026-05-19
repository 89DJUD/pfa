"""
Phase 6 – Étape 6.4
Effets marginaux de l'interaction SPEI × Charge carbone
→ Effet de la sécheresse selon le niveau de taxe carbone (H3)
Panel : 41 entreprises marocaines cotées | 2019-2024
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import os

from linearmodels.panel import PanelOLS

os.makedirs('outputs/tables',  exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

df_raw = pd.read_csv('data/panel_final_merged.csv')
res_df = pd.read_csv('data/processed/fe_results_main.csv')

# Coefficients du modèle principal
params = dict(zip(res_df['variable'], res_df['coef']))
se_map = dict(zip(res_df['variable'], res_df['se']))

b1 = params.get('spei12_mean', 0)       # β_SPEI
b3 = params.get('spei_x_carbon', 0)     # β_interaction
b2 = params.get('carbon_burden_mid', 0) # β_carbone

se_b1 = se_map.get('spei12_mean', 0)
se_b3 = se_map.get('spei_x_carbon', 0)

print("=" * 65)
print("PHASE 6 — EFFETS MARGINAUX (INTERACTION SPEI × CO₂)")
print("=" * 65)
print(f"\n  β_SPEI        = {b1:+.4f}")
print(f"  β_CO₂         = {b2:+.4f}")
print(f"  β_interaction = {b3:+.4f}")
print(f"\n  Effet marginal du SPEI sur Z-score :")
print(f"  ∂Z/∂SPEI = β₁ + β₃ × Carbon_Burden")

# ── 1. Effets marginaux aux 3 scénarios de taxe ──────────────────────────────
print("\n" + "─" * 65)
print("Effets marginaux aux 3 scénarios (charge carbone moyenne par scénario)")
print("─" * 65)

scen_means = {
    '25$/tCO₂ (bas)'   : df_raw['carbon_burden_low'].mean(),
    '50$/tCO₂ (médian)': df_raw['carbon_burden_mid'].mean(),
    '75$/tCO₂ (haut)'  : df_raw['carbon_burden_high'].mean(),
}

for scen, carb_val in scen_means.items():
    me      = b1 + b3 * carb_val
    se_me   = np.sqrt(se_b1**2 + (carb_val**2) * se_b3**2)
    t_stat  = me / se_me if se_me > 0 else np.nan
    from scipy import stats
    p_me    = 2 * (1 - stats.t.cdf(abs(t_stat), df=len(df_raw)-7)) if se_me > 0 else np.nan
    sig     = '***' if p_me<0.01 else '**' if p_me<0.05 else '*' if p_me<0.1 else 'n.s.'
    print(f"  {scen:<22}  charge moy={carb_val:.4f}  "
          f"∂Z/∂SPEI = {me:+.4f}  SE={se_me:.4f}  p={p_me:.4f} {sig}")

# ── 2. Courbe des effets marginaux (continu) ──────────────────────────────────
carbon_range = np.linspace(df_raw['carbon_burden_mid'].min(),
                            df_raw['carbon_burden_mid'].max(), 200)
me_values = b1 + b3 * carbon_range
se_me_values = np.sqrt(se_b1**2 + (carbon_range**2) * se_b3**2)

fig, axes = plt.subplots(1, 2, figsize=(15, 6))
fig.suptitle('Effets marginaux de la sécheresse (SPEI) sur le Z-score Altman\n'
             'selon le niveau de charge carbone — Modèle EF Double',
             fontsize=13, fontweight='bold')

# ─ Panneau A : effet marginal continu ─────────────────────────────────────────
ax = axes[0]
ax.plot(carbon_range, me_values, color='#1F4E79', linewidth=2.5,
        label='Effet marginal ∂Z/∂SPEI')
ax.fill_between(carbon_range,
                me_values - 1.645*se_me_values,
                me_values + 1.645*se_me_values,
                alpha=0.2, color='#2E75B6', label='IC 90%')
ax.fill_between(carbon_range,
                me_values - 1.96*se_me_values,
                me_values + 1.96*se_me_values,
                alpha=0.1, color='#5B9BD5', label='IC 95%')

ax.axhline(0, color='red', linestyle='--', linewidth=1.2, alpha=0.7)

# Points aux 3 scénarios
colors_pt = ['#2ECC71', '#F39C12', '#E74C3C']
for (scen, carb_val), col in zip(scen_means.items(), colors_pt):
    me_pt = b1 + b3 * carb_val
    ax.axvline(carb_val, color=col, linestyle=':', alpha=0.7, linewidth=1.2)
    ax.scatter([carb_val], [me_pt], color=col, s=80, zorder=5,
               label=f'{scen}: ∂Z/∂SPEI={me_pt:+.3f}')

ax.set_xlabel('Charge carbone (% du CA) — scénario médian', fontsize=11)
ax.set_ylabel('Effet marginal ∂Z-score / ∂SPEI', fontsize=11)
ax.set_title('A. Effet marginal du SPEI selon la charge carbone', fontsize=11)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# ─ Panneau B : effet marginal aux scénarios (bar chart) ───────────────────────
ax2 = axes[1]
scen_labels = list(scen_means.keys())
me_scen = [b1 + b3*v for v in scen_means.values()]
se_scen = [np.sqrt(se_b1**2 + v**2*se_b3**2) for v in scen_means.values()]

bar_colors = ['#2ECC71', '#F39C12', '#E74C3C']
bars = ax2.bar(scen_labels, me_scen, color=bar_colors, alpha=0.8,
               width=0.5, edgecolor='white', linewidth=0.8)
ax2.errorbar(scen_labels, me_scen,
             yerr=[1.96*s for s in se_scen],
             fmt='none', color='#2C3E50', capsize=6, linewidth=2)
ax2.axhline(0, color='black', linewidth=1.2, linestyle='--')

for bar, val in zip(bars, me_scen):
    ax2.text(bar.get_x() + bar.get_width()/2, val + (0.05 if val>=0 else -0.12),
             f"{val:+.3f}", ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Effet marginal ∂Z-score / ∂SPEI', fontsize=11)
ax2.set_title('B. Effet marginal du SPEI aux 3 scénarios de taxe', fontsize=11)
ax2.set_xlabel('Scénario de taxe carbone', fontsize=11)
ax2.grid(True, alpha=0.3, axis='y')

note = ("Un effet marginal positif indique qu'une amélioration du SPEI\n"
        "(moins de sécheresse) augmente le Z-score (moins de détresse).")
fig.text(0.5, -0.02, note, ha='center', fontsize=9, style='italic', color='gray')

plt.tight_layout()
plt.savefig('outputs/figures/fig_marginal_effects.png', dpi=150, bbox_inches='tight')

# ── Tableau résumé ────────────────────────────────────────────────────────────
from scipy import stats as scipy_stats
rows = []
for scen, carb_val in scen_means.items():
    me    = b1 + b3 * carb_val
    se_me = np.sqrt(se_b1**2 + (carb_val**2) * se_b3**2)
    t     = me / se_me if se_me > 0 else np.nan
    p     = 2*(1-scipy_stats.t.cdf(abs(t), df=len(df_raw)-7)) if se_me>0 else np.nan
    rows.append({'Scénario': scen,
                 'Charge carbone moy.': round(carb_val, 4),
                 'Effet marginal ∂Z/∂SPEI': round(me, 4),
                 'Écart-type': round(se_me, 4),
                 'p-value': round(p, 4) if not np.isnan(p) else '—',
                 'Significativité': '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else 'n.s.'})

me_df = pd.DataFrame(rows).set_index('Scénario')
print("\n── Tableau effets marginaux ────────────────────────────────────────────")
print(me_df.to_string())
me_df.to_excel('outputs/tables/table4_marginal_effects.xlsx')

print("\n✅ Sauvegardé : outputs/figures/fig_marginal_effects.png")
print("✅ Sauvegardé : outputs/tables/table4_marginal_effects.xlsx")
