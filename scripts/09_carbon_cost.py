"""
PHASE 3 - ETAPE 3.3 : Calcul du coût carbone par entreprise-année
=================================================================
Input  :
  - data/processed/financial_panel_morocco.csv  (panel financier)
  - data/processed/co2_intensity_sectors.csv    (intensités IEA)

Output :
  - data/processed/carbon_tax_panel.csv

Colonnes produites (utilisées par 10_visualize_carbon.py) :
  carbon_cost_{low/mid/high}_kMAD
  carbon_burden_{low/mid/high}
  zscore_carbon_{low/mid/high}     ← Z-score ajusté après taxe
  in_distress                      ← baseline (Z < 1.81)
  distress_carbon_{low/mid/high}   ← détresse après taxe

Formule :
  carbon_cost   = tax_mad_per_kgCO2 × co2_intensity_kgCO2_per_MAD × revenue
  carbon_burden = carbon_cost / revenue
  zscore_carbon = zscore - (carbon_burden × 3.3)  # impact via X3 (EBIT/Actif)
=================================================================
"""

import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, 'data', 'processed')

print("=" * 60)
print("  PHASE 3 — ETAPE 3.3 : Coût carbone par entreprise-année")
print("=" * 60)

# ── Chargement ───────────────────────────────────────────────────
financial_path = os.path.join(OUT, 'financial_zscore.csv')
financial_raw_zscore_path = os.path.join(BASE, 'data', 'raw', 'financial_zscore.csv')
financial_panel_path = os.path.join(OUT, 'financial_panel_morocco.csv')
financial_raw_path = os.path.join(BASE, 'data', 'raw', 'financial_panel_morocco.csv')
intensity_path = os.path.join(OUT, 'co2_intensity_sectors.csv')

if not os.path.exists(financial_path):
    if os.path.exists(financial_raw_zscore_path):
        financial_path = financial_raw_zscore_path
    elif os.path.exists(financial_panel_path):
        financial_path = financial_panel_path
    elif os.path.exists(financial_raw_path):
        financial_path = financial_raw_path

missing_inputs = [
    path for path in [financial_path, intensity_path]
    if not os.path.exists(path)
]
if missing_inputs:
    print("\n  ERREUR : fichiers d'entree manquants pour l'etape 3.3")
    for path in missing_inputs:
        print(f"  - {os.path.relpath(path, BASE)}")
    print("\n  Lancez d'abord les scripts qui generent ces fichiers, puis relancez :")
    print("  python scripts/09_carbon_cost.py")
    raise SystemExit(1)

fin       = pd.read_csv(financial_path)
intensity = pd.read_csv(intensity_path)

sector_aliases = {
    'Materiaux de construction': 'Ciment & Materiaux de construction',
}
fin['sector'] = fin['sector'].replace(sector_aliases)

if 'zscore' not in fin.columns:
    required_zscore_cols = [
        'total_assets', 'current_assets', 'current_liabilities',
        'retained_earnings', 'ebit', 'market_cap', 'total_debt', 'revenue'
    ]
    missing_zscore_cols = [c for c in required_zscore_cols if c not in fin.columns]
    if missing_zscore_cols:
        print("\n  ERREUR : impossible de calculer le Z-score, colonnes manquantes :")
        for col in missing_zscore_cols:
            print(f"  - {col}")
        raise SystemExit(1)

    total_assets = fin['total_assets'].replace(0, pd.NA)
    total_debt = fin['total_debt'].replace(0, pd.NA)
    working_capital = fin['current_assets'] - fin['current_liabilities']

    fin['zscore'] = (
        1.2 * (working_capital / total_assets)
        + 1.4 * (fin['retained_earnings'] / total_assets)
        + 3.3 * (fin['ebit'] / total_assets)
        + 0.6 * (fin['market_cap'] / total_debt)
        + 1.0 * (fin['revenue'] / total_assets)
    )
    print("\n  Z-score calcule depuis le panel financier brut.")

print(f"\n  Panel financier : {len(fin)} observations, {fin['firm_id'].nunique()} entreprises")
print(f"  Intensités CO2  : {len(intensity)} secteurs")
print(f"  Période         : {fin['year'].min()} – {fin['year'].max()}")

# ── Vérification des secteurs ─────────────────────────────────────
non_matches = set(fin['sector'].unique()) - set(intensity['sector'].unique())
if non_matches:
    print(f"\n  ATTENTION : Secteurs non matches : {non_matches}")
else:
    print(f"\n  OK Tous les secteurs matches.")

# ── Jointure ──────────────────────────────────────────────────────
df = fin.merge(intensity[['sector',
                           'co2_intensity_kgCO2_per_MAD',
                           'tax_mad_per_kgCO2_low',
                           'tax_mad_per_kgCO2_mid',
                           'tax_mad_per_kgCO2_high']],
               on='sector', how='left')

# ── Calcul du coût carbone et charge ─────────────────────────────
# carbon_cost   = taxe × intensité × CA
# carbon_burden = carbon_cost / CA  (ratio normalisé)
for sc in ['low', 'mid', 'high']:
    tax_col    = f'tax_mad_per_kgCO2_{sc}'
    cost_col   = f'carbon_cost_{sc}_kMAD'
    burden_col = f'carbon_burden_{sc}'

    df[cost_col] = df[tax_col] * df['co2_intensity_kgCO2_per_MAD'] * df['revenue']
    df[burden_col] = 0.0
    revenue_nonzero = df['revenue'] != 0
    df.loc[revenue_nonzero, burden_col] = (
        df.loc[revenue_nonzero, cost_col] / df.loc[revenue_nonzero, 'revenue']
    )

# ── Calcul du Z-score ajusté après taxe ──────────────────────────
# La taxe carbone réduit l'EBIT → impact sur X3 (EBIT/Actif)
# ΔZ = -3.3 × (carbon_burden) car X3 = EBIT/Actif et coeff = 3.3
# C'est une approximation conservative (borne basse de l'impact)
for sc in ['low', 'mid', 'high']:
    burden_col = f'carbon_burden_{sc}'
    zscore_col = f'zscore_carbon_{sc}'
    df[zscore_col] = df['zscore'] - (3.3 * df[burden_col])

# ── Variable détresse (Z < 1.81) ─────────────────────────────────
df['in_distress'] = (df['zscore'] < 1.81).astype(int)

for sc in ['low', 'mid', 'high']:
    df[f'distress_carbon_{sc}'] = (df[f'zscore_carbon_{sc}'] < 1.81).astype(int)

# ── Statistiques descriptives ─────────────────────────────────────
print(f"\n  Charge carbone — scénario médian (50 $/tCO2) :")
print("  " + "-" * 55)
stats = df.groupby('sector')['carbon_burden_mid'].describe()[['mean','std','min','max']]
print(stats.round(4).to_string())

print(f"\n  Impact sur le Z-score :")
print("  " + "-" * 45)
z_bl = df['zscore'].mean()
d_bl = df['in_distress'].mean() * 100
print(f"  Baseline               : Z={z_bl:.3f}  |  détresse={d_bl:.1f}%")
for sc, label in [('low','25$/tCO2'), ('mid','50$/tCO2'), ('high','75$/tCO2')]:
    z_sc = df[f'zscore_carbon_{sc}'].mean()
    d_sc = df[f'distress_carbon_{sc}'].mean() * 100
    print(f"  Scénario {label:<10}  : Z={z_sc:.3f} (dZ={z_sc-z_bl:+.3f})  |  détresse={d_sc:.1f}% (d={d_sc-d_bl:+.1f}%)")

# ── Sauvegarde ───────────────────────────────────────────────────
cols_to_keep = [
    'firm_id', 'year', 'sector',
    # coûts absolus
    'carbon_cost_low_kMAD', 'carbon_cost_mid_kMAD', 'carbon_cost_high_kMAD',
    # charges normalisées (variable régression)
    'carbon_burden_low', 'carbon_burden_mid', 'carbon_burden_high',
    # Z-scores ajustés (pour visualisation stress-test)
    'zscore', 'zscore_carbon_low', 'zscore_carbon_mid', 'zscore_carbon_high',
    # variables détresse
    'in_distress', 'distress_carbon_low', 'distress_carbon_mid', 'distress_carbon_high',
]

carbon_panel = df[[c for c in cols_to_keep if c in df.columns]]
out_path = os.path.join(OUT, 'carbon_tax_panel.csv')
carbon_panel.to_csv(out_path, index=False)

print(f"\n  OK Sauvegardé : data/processed/carbon_tax_panel.csv")
print(f"  Dimensions   : {carbon_panel.shape[0]} lignes × {carbon_panel.shape[1]} colonnes")
print(f"\n  Colonnes disponibles pour 10_visualize_carbon.py :")
for c in carbon_panel.columns:
    print(f"    - {c}")
print(f"\n  OK ETAPE 3.3 terminée — OUTPUT FINAL PHASE 3 : carbon_tax_panel.csv")
print(f"  Prochaine étape : python scripts/10_visualize_carbon.py")
