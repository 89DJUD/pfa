"""
PHASE 4 - ETAPE 4.4 : Construction du panel final
===================================================
Crée les variables finales et sauvegarde le fichier maître.

Variables créées :
  - spei_x_carbon  : terme d'interaction SPEI × carbon_burden_mid (H3)
  - zscore_lag1    : Z-score de l'année précédente (modèle dynamique)
  - log_assets     : log(total_assets) si pas déjà présent
  - in_distress    : binaire Z < 1.81

Input  : data/processed/panel_merged_clean.csv
Output : data/processed/panel_final_merged.csv  ← FICHIER MAÎTRE
===================================================
"""

import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, 'data', 'processed')

print("=" * 60)
print("  PHASE 4 — ETAPE 4.4 : Construction du panel final")
print("=" * 60)

# ── Chargement ───────────────────────────────────────────────────
df = pd.read_csv(os.path.join(OUT, 'panel_merged_clean.csv'))
df = df.sort_values(['firm_id', 'year']).reset_index(drop=True)

print(f"\n  Observations : {len(df)}")
print(f"  Entreprises  : {df['firm_id'].nunique()}")
print(f"  Période      : {df['year'].min()} – {df['year'].max()}")

# ── Variable d'interaction SPEI × Carbon (cœur du modèle H3) ────
df['spei_x_carbon'] = df['spei12_mean'] * df['carbon_burden_mid']
print(f"\n  OK spei_x_carbon créé (interaction SPEI × carbon_burden_mid)")

# ── Z-score retardé d'un an (modèle dynamique) ───────────────────
df['zscore_lag1'] = df.groupby('firm_id')['zscore'].shift(1)
n_lag = df['zscore_lag1'].notna().sum()
print(f"  OK zscore_lag1 créé ({n_lag} valeurs non manquantes)")

# ── log(total_assets) si pas déjà présent ────────────────────────
if 'log_assets' not in df.columns and 'total_assets' in df.columns:
    df['log_assets'] = np.log(df['total_assets'].replace(0, np.nan))
    print(f"  OK log_assets créé depuis total_assets")

# ── Variable détresse binaire ─────────────────────────────────────
if 'in_distress' not in df.columns:
    df['in_distress'] = (df['zscore'] < 1.81).astype(int)
    print(f"  OK in_distress créé (Z < 1.81)")

taux_detresse = df['in_distress'].mean() * 100
print(f"     Taux de détresse global : {taux_detresse:.1f}%")

# ── Dummies sectorielles ──────────────────────────────────────────
sector_dummies = pd.get_dummies(df['sector'], prefix='sec', drop_first=True)
df = pd.concat([df, sector_dummies], axis=1)
print(f"  OK Dummies sectorielles créées ({len(sector_dummies.columns)} colonnes)")

# ── Sélection des colonnes finales ───────────────────────────────
final_cols = [
    # Identifiants
    'firm_id', 'firm_name', 'ticker', 'sector', 'region', 'year',
    # Variable dépendante
    'zscore', 'in_distress', 'zscore_lag1',
    # Composantes Z-score
    'X1', 'X2', 'X3', 'X4', 'X5',
    # Variables explicatives principales
    'spei12_mean', 'spei12_min', 'drought_months',
    'carbon_burden_low', 'carbon_burden_mid', 'carbon_burden_high',
    'spei_x_carbon',
    # Variables de contrôle
    'log_assets', 'leverage', 'liquidity', 'roa',
    # Coûts carbone absolus
    'carbon_cost_low_kMAD', 'carbon_cost_mid_kMAD', 'carbon_cost_high_kMAD',
]

# Ajouter les dummies sectorielles
final_cols += [c for c in df.columns if c.startswith('sec_')]

# Garder uniquement les colonnes existantes
panel_final = df[[c for c in final_cols if c in df.columns]]

# ── Statistiques descriptives rapides ────────────────────────────
print(f"\n  Statistiques des variables clés :")
print("  " + "-" * 55)
vars_desc = ['zscore', 'spei12_mean', 'carbon_burden_mid', 'spei_x_carbon']
vars_desc = [v for v in vars_desc if v in panel_final.columns]
desc = panel_final[vars_desc].describe().loc[['mean','std','min','max']]
print(desc.round(3).to_string())

# ── Sauvegarde ───────────────────────────────────────────────────
out_path = os.path.join(OUT, 'panel_final_merged.csv')
panel_final.to_csv(out_path, index=False)

# ── Résumé final ──────────────────────────────────────────────────
print(f"\n{'=' * 60}")
print(f"  OK OUTPUT FINAL PHASE 4 : panel_final_merged.csv")
print(f"{'=' * 60}")
print(f"  Observations : {len(panel_final)}")
print(f"  Entreprises  : {panel_final['firm_id'].nunique()}")
print(f"  Années       : {panel_final['year'].min()} – {panel_final['year'].max()}")
print(f"  Colonnes     : {panel_final.shape[1]}")
print(f"\n  ATTENTION : Conservez une copie de sauvegarde avant toute")
print(f"     transformation ultérieure !")
print(f"\n  Prochaine etape : Phase 5 — Statistiques descriptives")
