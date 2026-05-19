"""
Phase 6 – Étape 6.1
Estimation du modèle panel à effets fixes double (entreprise + année)
Hypothèses H1 (SPEI), H2 (taxe carbone), H3 (interaction)
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
 
from linearmodels.panel import PanelOLS, PooledOLS
from matplotlib.patches import Patch
 
os.makedirs('outputs/tables',  exist_ok=True)
os.makedirs('outputs/figures', exist_ok=True)
 
df = pd.read_csv('data/panel_final_merged.csv')
df = df.sort_values(['firm_id', 'year']).set_index(['firm_id', 'year'])
 
Y        = df['zscore']
CONTROLS = ['log_assets', 'leverage', 'liquidity', 'roa']
 
print("=" * 65)
print("PHASE 6 — ESTIMATION PANEL À EFFETS FIXES")
print("=" * 65)
print(f"Observations : {len(df)}")
print(f"Entreprises  : {df.index.get_level_values('firm_id').nunique()}")
print(f"Années       : {df.index.get_level_values('year').min()}–{df.index.get_level_values('year').max()}")
 
XVARS = ['spei12_mean', 'carbon_burden_mid', 'spei_x_carbon'] + CONTROLS
 
res1 = PooledOLS(Y, df[XVARS]).fit(cov_type='clustered', cluster_entity=True)
 
res2 = PanelOLS(Y, df[XVARS], entity_effects=True, time_effects=False
                ).fit(cov_type='clustered', cluster_entity=True)
 
res3 = PanelOLS(Y, df[XVARS], entity_effects=True, time_effects=True
                ).fit(cov_type='clustered', cluster_entity=True)
 
res4 = PanelOLS(Y, df[['spei12_mean', 'carbon_burden_mid'] + CONTROLS],
                entity_effects=True, time_effects=True
                ).fit(cov_type='clustered', cluster_entity=True)
 
MODELS = {
    '(1) OLS Poolé'          : res1,
    '(2) EF Entreprise'      : res2,
    '(3) EF Double ★'        : res3,
    '(4) EF Double sans int.': res4,
}
 
print("\n" + "=" * 65)
print("TABLEAU COMPARATIF — 4 SPÉCIFICATIONS")
print("Erreurs standard cluster-robustes entre parenthèses")
print("* p<0.10  ** p<0.05  *** p<0.01")
print("=" * 65)
 
rows = []
for var in XVARS:
    row = {'Variable': var}
    for name, res in MODELS.items():
        if var in res.params.index:
            c   = res.params[var]
            p   = res.pvalues[var]
            s   = res.std_errors[var]
            sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else ''
            row[name] = f"{c:+.3f}{sig} ({s:.3f})"
        else:
            row[name] = '—'
    rows.append(row)
 
rows.append({'Variable': '─────────────', **{n: '─────────' for n in MODELS}})
rows.append({'Variable': 'R² within',
             **{n: f"{getattr(r,'rsquared_within', r.rsquared):.3f}" for n,r in MODELS.items()}})
rows.append({'Variable': 'N observations', **{n: str(r.nobs) for n,r in MODELS.items()}})
rows.append({'Variable': 'EF Entreprise',
             **{'(1) OLS Poolé':'Non','(2) EF Entreprise':'Oui',
                '(3) EF Double ★':'Oui','(4) EF Double sans int.':'Oui'}})
rows.append({'Variable': 'EF Année',
             **{'(1) OLS Poolé':'Non','(2) EF Entreprise':'Non',
                '(3) EF Double ★':'Oui','(4) EF Double sans int.':'Oui'}})
 
comp_df = pd.DataFrame(rows).set_index('Variable')
print(comp_df.to_string())
 
print("\n" + "=" * 65)
print("RÉSULTATS DES HYPOTHÈSES — MODÈLE PRINCIPAL (3)")
print("=" * 65)
 
hyps = [
    ('H1 — Risque physique (SPEI)',  'spei12_mean',
     lambda c,p: c>0 and p<0.1,
     'β₁ > 0 : SPEI↑ → Z-score↑'),
    ('H2 — Risque transition (CO₂)', 'carbon_burden_mid',
     lambda c,p: c<0 and p<0.1,
     'β₂ < 0 : charge CO₂↑ → Z-score↓'),
    ('H3 — Interaction SPEI×CO₂',   'spei_x_carbon',
     lambda c,p: c<0 and p<0.1,
     'β₃ < 0 : amplification conjointe'),
]
for label, var, cond, exp in hyps:
    c   = res3.params[var]
    p   = res3.pvalues[var]
    s   = res3.std_errors[var]
    sig = '***' if p<0.01 else '**' if p<0.05 else '*' if p<0.1 else 'n.s.'
    st  = '✅ CONFIRMÉE' if cond(c,p) else \
          '❌ REJETÉE (signe inverse)' if p<0.1 else \
          '⚠️  NON SIGNIFICATIVE'
    print(f"\n  {label}")
    print(f"    Attendu : {exp}")
    print(f"    Obtenu  : β = {c:+.4f}  SE={s:.4f}  p={p:.4f}  {sig}")
    print(f"    Verdict : {st}")
 
VARS_PLOT = ['spei12_mean','carbon_burden_mid','spei_x_carbon',
             'log_assets','leverage','liquidity','roa']
LABELS    = ['SPEI-12 (H1)','Charge CO₂ (H2)','Interaction\nSPEI×CO₂ (H3)',
             'Taille','Levier','Liquidité','ROA']
 
coefs  = [res3.params[v]                  for v in VARS_PLOT]
ci_lo  = [res3.conf_int().loc[v,'lower']  for v in VARS_PLOT]
ci_hi  = [res3.conf_int().loc[v,'upper']  for v in VARS_PLOT]
pvals  = [res3.pvalues[v]                 for v in VARS_PLOT]
colors = ['#E74C3C' if p<0.01 else '#F39C12' if p<0.05
          else '#3498DB' if p<0.1 else '#95A5A6' for p in pvals]
 
fig, ax = plt.subplots(figsize=(10, 6))
y = range(len(VARS_PLOT))
ax.barh(y, coefs, color=colors, alpha=0.85, height=0.55, zorder=3)
ax.errorbar(coefs, y,
            xerr=[np.array(coefs)-np.array(ci_lo),
                  np.array(ci_hi)-np.array(coefs)],
            fmt='none', color='#2C3E50', capsize=5, linewidth=1.8, zorder=4)
ax.axvline(0, color='black', linewidth=1.2)
ax.set_yticks(list(y))
ax.set_yticklabels(LABELS, fontsize=10)
ax.set_xlabel('Coefficient — effet sur le Z-score Altman', fontsize=11)
ax.set_title('Effets fixes double — Coefficients et IC à 95%\n'
             'Erreurs standard cluster-robustes (niveau entreprise)', fontsize=11)
ax.grid(True, alpha=0.3, axis='x')
ax.legend(handles=[
    Patch(color='#E74C3C', label='p < 0.01 (***)'),
    Patch(color='#F39C12', label='p < 0.05 (**)'),
    Patch(color='#3498DB', label='p < 0.10 (*)'),
    Patch(color='#95A5A6', label='Non significatif'),
], fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig('outputs/figures/fig_fe_coefficients.png', dpi=150, bbox_inches='tight')
 
comp_df.to_excel('outputs/tables/table2_fe_models.xlsx')
pd.DataFrame({
    'variable': list(res3.params.index),
    'coef'    : list(res3.params.values),
    'se'      : list(res3.std_errors.values),
    'pvalue'  : list(res3.pvalues.values),
    'ci_lower': list(res3.conf_int()['lower'].values),
    'ci_upper': list(res3.conf_int()['upper'].values),
}).to_csv('data/processed/fe_results_main.csv', index=False)
 
print("\n✅ Sauvegardé : outputs/tables/table2_fe_models.xlsx")
print("✅ Sauvegardé : outputs/figures/fig_fe_coefficients.png")
print("✅ Sauvegardé : data/processed/fe_results_main.csv")