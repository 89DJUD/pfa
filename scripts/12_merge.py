"""
PHASE 4 - ETAPE 4.2 : Fusion des trois bases de données
==========================================================
Fusionne :
  1. financial_panel_morocco.csv  +  carbon_tax_panel.csv
     → clé : firm_id + year
  2. Résultat  +  climate_morocco_spei_2000_2024.csv
     → clé : region + year

Output : data/processed/panel_merged_raw.csv
==========================================================
"""

import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, 'data', 'processed')
RAW  = os.path.join(BASE, 'data', 'raw')

print("=" * 60)
print("  PHASE 4 — ETAPE 4.2 : Fusion des trois bases")
print("=" * 60)

# ── Chargement ───────────────────────────────────────────────────
def first_existing(paths, label):
    for path in paths:
        if os.path.exists(path):
            return path
    print(f"\n  ERREUR : fichier {label} introuvable.")
    for path in paths:
        print(f"  - {os.path.relpath(path, BASE)}")
    raise SystemExit(1)

financial_path = first_existing([
    os.path.join(OUT, 'financial_zscore.csv'),
    os.path.join(RAW, 'financial_zscore.csv'),
    os.path.join(OUT, 'financial_panel_morocco.csv'),
    os.path.join(RAW, 'financial_panel_morocco.csv'),
], 'financier')
climate_path = first_existing([
    os.path.join(OUT, 'climate_morocco_spei_2000_2024.csv'),
    os.path.join(OUT, 'climate_morocco_2000_2024.csv'),
], 'climatique')
carbon_path = first_existing([
    os.path.join(OUT, 'carbon_tax_panel.csv'),
], 'carbone')

fin     = pd.read_csv(financial_path)
carbon  = pd.read_csv(carbon_path)
climate = pd.read_csv(climate_path)

print(f"\n  Panel financier : {len(fin)} obs")
print(f"  Taxe carbone    : {len(carbon)} obs")
print(f"  Base climatique : {len(climate)} obs")

# ── Étape 1 : Panel financier + taxe carbone (firm_id + year) ────
carbon_cols = [
    'firm_id', 'year',
    'carbon_cost_low_kMAD', 'carbon_cost_mid_kMAD', 'carbon_cost_high_kMAD',
    'carbon_burden_low', 'carbon_burden_mid', 'carbon_burden_high',
    'zscore_carbon_low', 'zscore_carbon_mid', 'zscore_carbon_high',
    'distress_carbon_low', 'distress_carbon_mid', 'distress_carbon_high',
]
carbon = carbon[[c for c in carbon_cols if c in carbon.columns]]
df = fin.merge(carbon, on=['firm_id', 'year'], how='left')
print(f"\n  Après fusion financier + carbone : {len(df)} obs")
print(f"  Valeurs manquantes carbon_burden_mid : {df['carbon_burden_mid'].isna().sum()}")

# ── Étape 2 : Résultat + base climatique (region + year) ─────────
df = df.merge(climate, on=['region', 'year'], how='left')
print(f"  Après fusion + climatique        : {len(df)} obs")
print(f"  Valeurs manquantes spei12_mean   : {df['spei12_mean'].isna().sum()}")

# ── Étape 3 : Valeurs manquantes par variable ─────────────────────
print(f"\n  Valeurs manquantes par variable (%) :")
print("  " + "-" * 45)
missing = df.isnull().sum() / len(df) * 100
missing = missing[missing > 0].sort_values(ascending=False)
if len(missing) > 0:
    for col, pct in missing.items():
        print(f"  {col:<35} : {pct:.1f}%")
else:
    print("  Aucune valeur manquante !")

# ── Sauvegarde intermédiaire ──────────────────────────────────────
out_path = os.path.join(OUT, 'panel_merged_raw.csv')
df.to_csv(out_path, index=False)

print(f"\n  OK Sauvegardé : data/processed/panel_merged_raw.csv")
print(f"  Dimensions   : {df.shape[0]} lignes × {df.shape[1]} colonnes")
print(f"\n  OK ETAPE 4.2 terminée")
print(f"  Prochaine etape : python 13_missing_values.py")
