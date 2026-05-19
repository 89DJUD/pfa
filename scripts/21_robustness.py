"""
Phase 6 – Étape 6.3
Tests de robustesse
  A. Variable dépendante alternative (in_distress logit)
  B. Scénarios carbone alternatifs (25$ et 75$)
  C. Sous-échantillons sectoriels
  D. SPEI minimum vs SPEI moyen
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
from matplotlib.patches import Patch
 
os.makedirs('outputs/tables',  exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
 
df = pd.read_csv('data/panel_final_merged.csv')
df = df.sort_values(['firm_id', 'year'])
CONTROLS = ['log_assets', 'leverage', 'liquidity', 'roa']
 
def run_fe(data, y_col, x_cols):
    """Lance un modèle EF double et retourne les résultats."""
    d = data.set_index(['firm_id', 'year'])
    # Garder uniquement les colonnes sans NaN
    cols = [y_col] + x_cols
    d = d[cols].dropna()
    Y = d[y_col]
    X = d[x_cols]
    # Vérifier variation within
    if X.shape[0] < len(x_cols) + 5:
        return None
    try:
        res = PanelOLS(Y, X, entity_effects=True, time_effects=True
                       ).fit(cov_type='clustered', cluster_entity=True)
        return res
    except Exception:
        return None
 
# Résultats du modèle principal (référence)
res_main = run_fe(df, 'zscore',
                  ['spei12_mean','carbon_burden_mid','spei_x_carbon']+CONTROLS)
 
print("=" * 65)
print("PHASE 6 — TESTS DE ROBUSTESSE")
print("=" * 65)
 
all_results = {'Modèle principal': res_main}
 
# ── A. Scénarios de taxe carbone alternatifs ─────────────────────────────────
print("\n── A. Scénarios de taxe carbone ────────────────────────────────────────")
for scen, col in [('25$/tCO₂ (bas)', 'carbon_burden_low'),
                   ('75$/tCO₂ (haut)', 'carbon_burden_high')]:
    df_s = df.copy()
    df_s['spei_x_carbon_scen'] = df_s['spei12_mean'] * df_s[col]
    res = run_fe(df_s, 'zscore',
                 ['spei12_mean', col, 'spei_x_carbon_scen'] + CONTROLS)
    if res:
        all_results[f'Taxe {scen}'] = res
        c_spei  = res.params.get('spei12_mean', np.nan)
        c_carb  = res.params.get(col, np.nan)
        p_carb  = res.pvalues.get(col, np.nan)
        sig     = '***' if p_carb<0.01 else '**' if p_carb<0.05 else '*' if p_carb<0.1 else 'n.s.'
        print(f"  Scénario {scen:20s}  β_SPEI={c_spei:+.3f}  β_CO₂={c_carb:+.3f} {sig}")
 
# ── B. Variable SPEI alternative (minimum vs moyen) ──────────────────────────
print("\n── B. SPEI minimum vs SPEI moyen ───────────────────────────────────────")
df_b = df.copy()
df_b['spei_x_carbon_min'] = df_b['spei12_min'] * df_b['carbon_burden_mid']
res_b = run_fe(df_b, 'zscore',
               ['spei12_min','carbon_burden_mid','spei_x_carbon_min']+CONTROLS)
if res_b:
    all_results['SPEI minimum'] = res_b
    c_s = res_b.params.get('spei12_min', np.nan)
    p_s = res_b.pvalues.get('spei12_min', np.nan)
    c_c = res_b.params.get('carbon_burden_mid', np.nan)
    p_c = res_b.pvalues.get('carbon_burden_mid', np.nan)
    print(f"  SPEI min  β={c_s:+.3f} p={p_s:.3f} | CO₂ β={c_c:+.3f} p={p_c:.3f}")
 
# ── C. Sous-échantillons sectoriels ───────────────────────────────────────────
print("\n── C. Sous-échantillons sectoriels ─────────────────────────────────────")
FOCUS_SECTORS = ['Agro-alimentaire & Boissons', 'Industrie (autres)',
                 'Materiaux de construction', 'Energie & Mines']
 
for sec in FOCUS_SECTORS:
    df_sec = df[df['sector'] == sec].copy()
    if df_sec['firm_id'].nunique() < 3:
        print(f"  {sec:<35} → trop peu d'entreprises (n={df_sec['firm_id'].nunique()}), ignoré")
        continue
    res_sec = run_fe(df_sec, 'zscore',
                     ['spei12_mean','carbon_burden_mid','spei_x_carbon']+CONTROLS)
    if res_sec:
        c_s = res_sec.params.get('spei12_mean', np.nan)
        p_s = res_sec.pvalues.get('spei12_mean', np.nan)
        c_c = res_sec.params.get('carbon_burden_mid', np.nan)
        p_c = res_sec.pvalues.get('carbon_burden_mid', np.nan)
        n   = res_sec.nobs
        print(f"  {sec:<35} n={n:3d}  β_SPEI={c_s:+.3f}(p={p_s:.2f})  β_CO₂={c_c:+.3f}(p={p_c:.2f})")
    else:
        print(f"  {sec:<35} → modèle non convergent")
 
# ── D. Mois de sécheresse vs SPEI continu ────────────────────────────────────
print("\n── D. Mois de sécheresse (variable binaire agrégée) ────────────────────")
df_d = df.copy()
df_d['drought_x_carbon'] = df_d['drought_months'] * df_d['carbon_burden_mid']
res_d = run_fe(df_d, 'zscore',
               ['drought_months','carbon_burden_mid','drought_x_carbon']+CONTROLS)
if res_d:
    all_results['Mois sécheresse'] = res_d
    c_dr = res_d.params.get('drought_months', np.nan)
    p_dr = res_d.pvalues.get('drought_months', np.nan)
    print(f"  β_drought_months = {c_dr:+.4f}  p = {p_dr:.4f}")
 
# ── Tableau synthèse robustesse ───────────────────────────────────────────────
print("\n" + "=" * 65)
print("TABLEAU SYNTHÈSE — ROBUSTESSE")
print("Coefficients β (p-value entre parenthèses)")
print("=" * 65)
 
KEY_VARS = ['spei12_mean', 'carbon_burden_mid', 'spei_x_carbon',
            'carbon_burden_low', 'carbon_burden_high',
            'spei12_min', 'drought_months']
 
rows = []
for name, res in all_results.items():
    if res is None:
        continue
    row = {'Spécification': name, 'N': res.nobs}
    for var in KEY_VARS:
        if var in res.params.index:
            c = res.params[var]
            p = res.pvalues[var]
            s = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
            row[var] = f"{c:+.3f}{s}"
        else:
            row[var] = '—'
    rows.append(row)
 
rob_df = pd.DataFrame(rows).set_index('Spécification')
print(rob_df.to_string())
rob_df.to_excel('outputs/tables/table3_robustness.xlsx')
 
# ── Figure comparaison β_CO₂ entre spécifications ────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
specs, coefs, ci_lo, ci_hi, pvs = [], [], [], [], []
CARBON_COLS = {'Modèle principal': 'carbon_burden_mid',
               'Taxe 25$/tCO₂ (bas)': 'carbon_burden_low',
               'Taxe 75$/tCO₂ (haut)': 'carbon_burden_high',
               'SPEI minimum': 'carbon_burden_mid',
               'Mois sécheresse': 'carbon_burden_mid'}
 
for name, res in all_results.items():
    if res is None:
        continue
    ccol = CARBON_COLS.get(name, 'carbon_burden_mid')
    if ccol in res.params.index:
        specs.append(name)
        coefs.append(res.params[ccol])
        ci_lo.append(res.conf_int().loc[ccol, 'lower'])
        ci_hi.append(res.conf_int().loc[ccol, 'upper'])
        pvs.append(res.pvalues[ccol])
 
colors = ['#E74C3C' if p<0.01 else '#F39C12' if p<0.05
          else '#3498DB' if p<0.1 else '#95A5A6' for p in pvs]
 
y = range(len(specs))
ax.barh(y, coefs, color=colors, alpha=0.8, height=0.5, zorder=3)
ax.errorbar(coefs, y,
            xerr=[np.array(coefs)-np.array(ci_lo),
                  np.array(ci_hi)-np.array(coefs)],
            fmt='none', color='#2C3E50', capsize=4, linewidth=1.5, zorder=4)
ax.axvline(0, color='black', linewidth=1)
ax.set_yticks(list(y))
ax.set_yticklabels(specs, fontsize=9)
ax.set_xlabel('Coefficient β sur la charge carbone', fontsize=11)
ax.set_title('Robustesse : effet du risque de transition (CO₂) selon les spécifications\n'
             '* p<0.10  ** p<0.05  *** p<0.01', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('outputs/figures/fig_robustness.png', dpi=150, bbox_inches='tight')
 
print("\n✅ Sauvegardé : outputs/tables/table3_robustness.xlsx")
print("✅ Sauvegardé : outputs/figures/fig_robustness.png")