"""
Phase 6 – Étape 6.2
Test de Hausman : Effets Fixes vs Effets Aléatoires
Panel : 41 entreprises marocaines cotées | 2019-2024
"""

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

from linearmodels.panel import PanelOLS, RandomEffects

df = pd.read_csv('data/panel_final_merged.csv')
df = df.sort_values(['firm_id', 'year']).set_index(['firm_id', 'year'])

Y        = df['zscore']
CONTROLS = ['log_assets', 'leverage', 'liquidity', 'roa']
XVARS    = ['spei12_mean', 'carbon_burden_mid', 'spei_x_carbon'] + CONTROLS

print("=" * 65)
print("PHASE 6 — TEST DE HAUSMAN (FE vs RE)")
print("=" * 65)

# Estimation FE et RE
res_fe = PanelOLS(Y, df[XVARS], entity_effects=True, time_effects=True
                  ).fit(cov_type='clustered', cluster_entity=True)

res_re = RandomEffects(Y, df[XVARS]).fit()

# ── Test de Hausman manuel ────────────────────────────────────────────────────
# H0 : les effets individuels ne sont pas corrélés avec X (RE consistant)
# H1 : corrélation entre effets et X (FE nécessaire)

b_fe = res_fe.params
b_re = res_re.params

# Variables communes
common = b_fe.index.intersection(b_re.index)
diff   = b_fe[common] - b_re[common]

# Matrice de variance de la différence
V_fe   = res_fe.cov.loc[common, common]
V_re   = res_re.cov.loc[common, common]
V_diff = V_fe - V_re

# Statistique H de Hausman
try:
    H_stat = float(diff @ np.linalg.inv(V_diff) @ diff)
    df_H   = len(common)
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(H_stat, df=df_H)
    print(f"\n  Statistique H  = {H_stat:.4f}")
    print(f"  Degrés liberté = {df_H}")
    print(f"  p-value        = {p_value:.4f}")
except np.linalg.LinAlgError:
    print("\n  ⚠️  Matrice singulière — utilisation de la pseudo-inverse")
    H_stat = float(diff @ np.linalg.pinv(V_diff) @ diff)
    df_H   = len(common)
    from scipy import stats
    p_value = 1 - stats.chi2.cdf(H_stat, df=df_H)
    print(f"  Statistique H  = {H_stat:.4f}")
    print(f"  p-value        = {p_value:.4f}")

print("\n  Interprétation :")
if p_value < 0.05:
    print("  → p < 0.05 : REJET de H0")
    print("  → Les effets individuels sont corrélés avec les régresseurs")
    print("  → Le modèle à EFFETS FIXES est préférable ✅")
else:
    print("  → p > 0.05 : Non-rejet de H0")
    print("  → Le modèle à EFFETS ALÉATOIRES est valide")
    print("  → Mais les EF sont conservés pour la robustesse méthodologique")

# ── Comparaison FE vs RE coefficient par coefficient ─────────────────────────
print("\n" + "─" * 65)
print("COMPARAISON DES COEFFICIENTS FE vs RE")
print("─" * 65)
print(f"{'Variable':<22} {'FE':>10} {'RE':>10} {'Δ':>10}")
print("─" * 55)
for var in XVARS:
    fe_c = res_fe.params.get(var, np.nan)
    re_c = res_re.params.get(var, np.nan)
    delta = fe_c - re_c if not np.isnan(fe_c) and not np.isnan(re_c) else np.nan
    fe_s = '***' if res_fe.pvalues.get(var,1)<0.01 else '**' if res_fe.pvalues.get(var,1)<0.05 else '*' if res_fe.pvalues.get(var,1)<0.1 else ''
    re_s = '***' if res_re.pvalues.get(var,1)<0.01 else '**' if res_re.pvalues.get(var,1)<0.05 else '*' if res_re.pvalues.get(var,1)<0.1 else ''
    print(f"  {var:<20} {fe_c:>+8.3f}{fe_s:<3} {re_c:>+8.3f}{re_s:<3} {delta:>+8.3f}")

print("\n✅ Conclusion : modèle à effets fixes double retenu pour la suite.")
