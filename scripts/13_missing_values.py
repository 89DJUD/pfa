"""
PHASE 4 - ETAPE 4.3 : Traitement des valeurs manquantes + Winsorisation
=========================================================================
Règles :
  1. SPEI manquant       → imputer par la moyenne nationale de l'année
  2. market_cap manquant → interpolation linéaire par entreprise (max 2 ans)
  3. zscore manquant     → supprimer (ne JAMAIS imputer la VD !)
  4. Outliers extrêmes   → winsorisation à 1% et 99%

Input  : data/processed/panel_merged_raw.csv
Output : data/processed/panel_merged_clean.csv
=========================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import mstats
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, 'data', 'processed')

print("=" * 60)
print("  PHASE 4 — ETAPE 4.3 : Valeurs manquantes + Winsorisation")
print("=" * 60)

# ── Chargement ───────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT, 'panel_merged_raw.csv'))
n_initial = len(df)
print(f"\n  Observations initiales : {n_initial}")

# ── 1. SPEI manquant → moyenne nationale de l'année ──────────────
spei_missing_avant = df['spei12_mean'].isna().sum()
national_spei = df.groupby('year')['spei12_mean'].transform('mean')
df['spei12_mean'] = df['spei12_mean'].fillna(national_spei)
spei_missing_apres = df['spei12_mean'].isna().sum()
print(f"\n  1. SPEI manquant : {spei_missing_avant} -> {spei_missing_apres} (imputé par moyenne nationale)")

# ── 2. market_cap manquant → interpolation linéaire ──────────────
if 'market_cap' in df.columns:
    mc_missing_avant = df['market_cap'].isna().sum()
    df = df.sort_values(['firm_id', 'year'])
    df['market_cap'] = df.groupby('firm_id')['market_cap'].transform(
        lambda x: x.interpolate(method='linear', limit=2)
    )
    mc_missing_apres = df['market_cap'].isna().sum()
    print(f"  2. market_cap manquant : {mc_missing_avant} -> {mc_missing_apres} (interpolation linéaire)")

# ── 3. zscore manquant → supprimer ───────────────────────────────
zscore_missing = df['zscore'].isna().sum()
df_clean = df.dropna(subset=['zscore']).copy()
print(f"  3. Lignes supprimées (zscore manquant) : {zscore_missing}")
print(f"     Observations conservées : {len(df_clean)} / {n_initial} ({len(df_clean)/n_initial*100:.1f}%)")

# ── 4. Winsorisation des outliers (1% – 99%) ─────────────────────
cols_winsor = ['zscore', 'leverage', 'liquidity', 'roa', 'X4']
cols_winsor = [c for c in cols_winsor if c in df_clean.columns]

print(f"\n  4. Winsorisation (1% – 99%) :")
print("  " + "-" * 50)
for col in cols_winsor:
    avant_min = df_clean[col].min()
    avant_max = df_clean[col].max()
    df_clean[col] = mstats.winsorize(df_clean[col], limits=[0.01, 0.01])
    print(f"  {col:<15} : [{avant_min:.2f}, {avant_max:.2f}] -> [{df_clean[col].min():.2f}, {df_clean[col].max():.2f}]")

# ── Vérification finale des valeurs manquantes ────────────────────
print(f"\n  Valeurs manquantes résiduelles :")
missing = df_clean.isnull().sum()
missing = missing[missing > 0]
if len(missing) > 0:
    for col, n in missing.items():
        print(f"  {col:<35} : {n} ({n/len(df_clean)*100:.1f}%)")
else:
    print("  OK Aucune valeur manquante résiduelle !")

# ── Sauvegarde ───────────────────────────────────────────────────
out_path = os.path.join(OUT, 'panel_merged_clean.csv')
df_clean.to_csv(out_path, index=False)

print(f"\n  OK Sauvegardé : data/processed/panel_merged_clean.csv")
print(f"  Dimensions   : {df_clean.shape[0]} lignes × {df_clean.shape[1]} colonnes")
print(f"\n  OK ETAPE 4.3 terminée")
print(f"  Prochaine etape : python 14_final_panel.py")
