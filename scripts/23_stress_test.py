"""
Phase 6 – Étape 6.5
Simulation stress-test : impact des scénarios climatiques sur les zones Altman
→ Combien d'entreprises basculent en zone de détresse sous chaque scénario ?
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

os.makedirs('outputs/tables',  exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)

df  = pd.read_csv('data/panel_final_merged.csv')
res = pd.read_csv('data/processed/fe_results_main.csv')

params = dict(zip(res['variable'], res['coef']))
b_spei   = params.get('spei12_mean', 0)
b_carb   = params.get('carbon_burden_mid', 0)
b_inter  = params.get('spei_x_carbon', 0)

print("=" * 65)
print("PHASE 6 — STRESS-TEST CLIMATIQUE")
print("=" * 65)
print(f"\n  β_SPEI = {b_spei:+.4f}  β_CO₂ = {b_carb:+.4f}  β_inter = {b_inter:+.4f}")

# ── Scénarios de stress ───────────────────────────────────────────────────────
# Choc SPEI : déviation par rapport à la moyenne observée
spei_baseline = df['spei12_mean'].mean()
spei_mild     = spei_baseline - 0.5   # Sécheresse légère
spei_moderate = spei_baseline - 1.0   # Sécheresse modérée
spei_severe   = spei_baseline - 1.5   # Sécheresse sévère

scenarios = {
    'Baseline (observé)' : {'spei': spei_baseline,  'carb_col': 'carbon_burden_mid', 'label': 'SPEI observé + 50$/tCO₂'},
    'Taxe 25$/tCO₂'      : {'spei': spei_baseline,  'carb_col': 'carbon_burden_low',  'label': 'SPEI observé + 25$/tCO₂'},
    'Taxe 75$/tCO₂'      : {'spei': spei_baseline,  'carb_col': 'carbon_burden_high', 'label': 'SPEI observé + 75$/tCO₂'},
    'Séch. légère'        : {'spei': spei_mild,      'carb_col': 'carbon_burden_mid', 'label': 'SPEI−0.5 + 50$/tCO₂'},
    'Séch. modérée'       : {'spei': spei_moderate,  'carb_col': 'carbon_burden_mid', 'label': 'SPEI−1.0 + 50$/tCO₂'},
    'Séch. sévère'        : {'spei': spei_severe,    'carb_col': 'carbon_burden_mid', 'label': 'SPEI−1.5 + 50$/tCO₂'},
    'Pire scénario'       : {'spei': spei_severe,    'carb_col': 'carbon_burden_high', 'label': 'SPEI−1.5 + 75$/tCO₂'},
}

def classify_zone(z):
    if z >= 2.99: return 'Sûre'
    if z >= 1.81: return 'Grise'
    return 'Détresse'

# ── Calcul impact par scénario ────────────────────────────────────────────────
# Utiliser la dernière année disponible (2024) comme point de départ
df_2024 = df[df['year'] == df['year'].max()].copy()
n_firms  = len(df_2024)

print(f"\n  Base de simulation : {n_firms} entreprises (année {df['year'].max()})")
print(f"  Z-score moyen baseline : {df_2024['zscore'].mean():.3f}")
print(f"  SPEI moyen baseline : {spei_baseline:.3f}")

rows = []
zone_data = {}

for scen_name, scen_params in scenarios.items():
    d = df_2024.copy()
    spei_choc = scen_params['spei']
    carb_col  = scen_params['carb_col']
    
    # Delta sur le Z-score dû au choc
    delta_spei  = b_spei  * (spei_choc - d['spei12_mean'])
    delta_carb  = b_carb  * (d[carb_col] - d['carbon_burden_mid'])
    delta_inter = b_inter * (spei_choc * d[carb_col] - d['spei_x_carbon'])
    
    d['zscore_stress'] = d['zscore'] + delta_spei + delta_carb + delta_inter
    d['zone_baseline'] = d['zscore'].apply(classify_zone)
    d['zone_stress']   = d['zscore_stress'].apply(classify_zone)
    
    n_distress_base   = (d['zone_baseline'] == 'Détresse').sum()
    n_distress_stress = (d['zone_stress']   == 'Détresse').sum()
    n_new_distress    = ((d['zone_baseline'] != 'Détresse') & (d['zone_stress'] == 'Détresse')).sum()
    zscore_delta      = (d['zscore_stress'] - d['zscore']).mean()
    
    zone_data[scen_name] = {
        'sûre'    : (d['zone_stress'] == 'Sûre').sum(),
        'grise'   : (d['zone_stress'] == 'Grise').sum(),
        'détresse': (d['zone_stress'] == 'Détresse').sum(),
    }
    
    rows.append({
        'Scénario'             : scen_name,
        'Description'          : scen_params['label'],
        'SPEI choc'            : round(spei_choc, 3),
        'ΔZ-score moyen'       : round(zscore_delta, 3),
        'N détresse (baseline)': n_distress_base,
        'N détresse (stress)'  : n_distress_stress,
        'N nouveaux en détresse': n_new_distress,
        '% en détresse'        : f"{n_distress_stress/n_firms*100:.1f}%",
    })

stress_df = pd.DataFrame(rows).set_index('Scénario')
print("\n── Résultats stress-test ───────────────────────────────────────────────")
print(stress_df.to_string())

# ── Figure 1 : Zones par scénario (stacked bar) ───────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Stress-test climatique — Impact sur les zones Altman (Bourse de Casablanca, 2024)',
             fontsize=13, fontweight='bold')

scen_names = list(zone_data.keys())
safe_vals     = [zone_data[s]['sûre']     for s in scen_names]
grey_vals     = [zone_data[s]['grise']    for s in scen_names]
distress_vals = [zone_data[s]['détresse'] for s in scen_names]

x = np.arange(len(scen_names))
w = 0.6
p1 = ax1.bar(x, safe_vals,     w, label='Zone sûre (Z≥2.99)',        color='#27AE60', alpha=0.85)
p2 = ax1.bar(x, grey_vals,     w, bottom=safe_vals,                   label='Zone grise (1.81–2.99)', color='#F39C12', alpha=0.85)
p3 = ax1.bar(x, distress_vals, w, bottom=np.array(safe_vals)+np.array(grey_vals),
             label='Zone détresse (Z<1.81)', color='#E74C3C', alpha=0.85)

ax1.set_xticks(x)
ax1.set_xticklabels([s.replace(' ', '\n') for s in scen_names], fontsize=8)
ax1.set_ylabel("Nombre d'entreprises", fontsize=11)
ax1.set_title("A. Répartition des zones Altman par scénario", fontsize=11)
ax1.legend(fontsize=9)
ax1.grid(True, alpha=0.2, axis='y')
ax1.axvline(0.5, color='gray', linestyle='--', alpha=0.4)

# Annoter N détresse
for xi, dv in zip(x, distress_vals):
    ax1.text(xi, n_firms + 0.3, f"{dv}", ha='center', fontsize=9,
             color='#C0392B', fontweight='bold')

# ─ Panel B : ΔZ-score moyen par scénario ──────────────────────────────────────
delta_vals = [float(stress_df.loc[s, 'ΔZ-score moyen']) for s in scen_names]
colors_d   = ['#27AE60' if v>=0 else '#E74C3C' for v in delta_vals]
bars2 = ax2.bar(x, delta_vals, w, color=colors_d, alpha=0.85, edgecolor='white')
ax2.axhline(0, color='black', linewidth=1.2)
ax2.set_xticks(x)
ax2.set_xticklabels([s.replace(' ', '\n') for s in scen_names], fontsize=8)
ax2.set_ylabel('Variation moyenne du Z-score Altman', fontsize=11)
ax2.set_title('B. Impact moyen sur le Z-score par scénario', fontsize=11)
ax2.grid(True, alpha=0.2, axis='y')
for bar, val in zip(bars2, delta_vals):
    ax2.text(bar.get_x() + bar.get_width()/2,
             val + (0.02 if val>=0 else -0.08),
             f"{val:+.3f}", ha='center', fontsize=8, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/fig_stress_test.png', dpi=150, bbox_inches='tight')

# ── Figure 2 : Entreprises à risque de basculement ───────────────────────────
worst = scenarios['Pire scénario']
d_worst = df_2024.copy()
spei_choc = worst['spei']
carb_col  = worst['carb_col']
d_worst['delta_z'] = (
    b_spei  * (spei_choc - d_worst['spei12_mean']) +
    b_carb  * (d_worst[carb_col] - d_worst['carbon_burden_mid']) +
    b_inter * (spei_choc * d_worst[carb_col] - d_worst['spei_x_carbon'])
)
d_worst['zscore_stress'] = d_worst['zscore'] + d_worst['delta_z']
d_worst['zone_stress']   = d_worst['zscore_stress'].apply(classify_zone)
d_worst['zone_baseline'] = d_worst['zscore'].apply(classify_zone)

fig2, ax3 = plt.subplots(figsize=(11, 7))
color_map = {'Sûre→Sûre': '#27AE60', 'Grise→Détresse': '#E67E22',
             'Sûre→Détresse': '#C0392B', 'Grise→Grise': '#F39C12',
             'Détresse→Détresse': '#922B21', 'Sûre→Grise': '#F1C40F'}

d_worst['transition'] = d_worst['zone_baseline'] + '→' + d_worst['zone_stress']
scatter_colors = [color_map.get(t, '#95A5A6') for t in d_worst['transition']]

ax3.scatter(d_worst['zscore'], d_worst['zscore_stress'],
            c=scatter_colors, s=80, alpha=0.8, zorder=3)

# Ligne de 45°
z_min = min(d_worst['zscore'].min(), d_worst['zscore_stress'].min()) - 0.5
z_max = max(d_worst['zscore'].max(), d_worst['zscore_stress'].max()) + 0.5
ax3.plot([z_min, z_max], [z_min, z_max], 'k--', alpha=0.4, linewidth=1, label='Pas de changement')

ax3.axhline(2.99, color='#27AE60', linestyle=':', alpha=0.7, linewidth=1.2)
ax3.axhline(1.81, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=1.2)
ax3.axvline(2.99, color='#27AE60', linestyle=':', alpha=0.7, linewidth=1.2)
ax3.axvline(1.81, color='#E74C3C', linestyle=':', alpha=0.7, linewidth=1.2)

# Annoter les entreprises qui basculent
basculement = d_worst[(d_worst['zone_baseline']!='Détresse') & (d_worst['zone_stress']=='Détresse')]
for _, row in basculement.iterrows():
    ax3.annotate(row.get('ticker', row.get('firm_id','')),
                 xy=(row['zscore'], row['zscore_stress']),
                 xytext=(5, 5), textcoords='offset points',
                 fontsize=7, color='#C0392B',
                 arrowprops=dict(arrowstyle='->', color='#C0392B', lw=0.8))

ax3.set_xlabel('Z-score Altman (observé 2024)', fontsize=11)
ax3.set_ylabel('Z-score Altman (stress — pire scénario)', fontsize=11)
ax3.set_title(f'Pire scénario : SPEI−1.5 + 75$/tCO₂\n'
              f'Les points sous la diagonale = détérioration du Z-score', fontsize=11)

from matplotlib.lines import Line2D
legend_el = [Line2D([0],[0],marker='o',color='w',markerfacecolor=c,markersize=9,label=l)
             for l,c in [('Sûre → Sûre','#27AE60'),('Sûre → Grise','#F1C40F'),
                          ('Sûre → Détresse','#C0392B'),('Grise → Détresse','#E67E22'),
                          ('Détresse → Détresse','#922B21')]]
ax3.legend(handles=legend_el, fontsize=8, loc='upper left')
ax3.grid(True, alpha=0.2)
plt.tight_layout()
plt.savefig('outputs/figures/fig_stress_basculement.png', dpi=150, bbox_inches='tight')

# ── Sauvegarde ────────────────────────────────────────────────────────────────
stress_df.to_excel('outputs/tables/table5_stress_test.xlsx')

print("\n✅ Sauvegardé : outputs/tables/table5_stress_test.xlsx")
print("✅ Sauvegardé : outputs/figures/fig_stress_test.png")
print("✅ Sauvegardé : outputs/figures/fig_stress_basculement.png")
print("\n" + "=" * 65)
print("PHASE 6 TERMINÉE — Tous les outputs générés")
print("=" * 65)
