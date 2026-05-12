"""
PHASE 1 - ETAPE 4 : Agrégation annuelle par région
====================================================================
Calcul des indicateurs annuels de sécheresse :
- spei12_mean         : moyenne SPEI-12 de l'année (indicateur principal)
- spei12_min          : valeur minimale (pire mois de l'année)
- drought_months      : nb de mois avec SPEI < -1.0 (sécheresse modérée)
- severe_drought_months: nb de mois avec SPEI < -1.5 (sécheresse sévère)
- drought_dummy       : 1 si spei12_mean < -1.0 (variable binaire)
====================================================================
"""

import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 55)
print("  ETAPE 4 : Agrégation annuelle par région")
print("=" * 55)

# Charger les données
climate = pd.read_csv(os.path.join(BASE, 'data/processed/climate_morocco_2000_2024.csv'))

print(f"\n  Données chargées : {len(climate)} lignes")
print(f"  Colonnes : {list(climate.columns)}")

# Ajouter variable binaire : année de sécheresse marquée
climate['drought_year']        = (climate['spei12_mean'] < -1.0).astype(int)
climate['moderate_drought']    = (climate['spei12_mean'] < -0.5).astype(int)
climate['severe_drought_year'] = (climate['spei12_mean'] < -1.5).astype(int)

# Décile du SPEI (pour analyses non-linéaires)
climate['spei_decile'] = pd.qcut(climate['spei12_mean'], q=10, 
                                  labels=False, duplicates='drop') + 1

# Sauvegarder le fichier enrichi
out = os.path.join(BASE, 'data/processed/climate_morocco_2000_2024.csv')
climate.to_csv(out, index=False)

print(f"\n  Variables créées :")
print(f"  - drought_year         : {climate['drought_year'].sum()} obs (SPEI < -1.0)")
print(f"  - moderate_drought     : {climate['moderate_drought'].sum()} obs (SPEI < -0.5)")
print(f"  - severe_drought_year  : {climate['severe_drought_year'].sum()} obs (SPEI < -1.5)")

# ── Statistiques descriptives par région ─────────────────────────
print("\n  Statistiques SPEI par région (2000-2024) :")
print("  " + "-" * 75)

stats = climate.groupby('region').agg(
    spei_mean=('spei12_mean', 'mean'),
    spei_std=('spei12_mean', 'std'),
    drought_years=('drought_year', 'sum'),
    worst_year=('spei12_mean', 'idxmin'),
).round(3)

# Ajouter l'année la plus sèche
worst_spei = climate.loc[climate.groupby('region')['spei12_mean'].idxmin()][['region','year','spei12_mean']]
worst_spei = worst_spei.set_index('region')

print(f"\n  {'Région':<38} {'SPEI moy':>9} {'Ecart-t':>9} {'Années sèches':>14} {'Pire année':>12}")
print("  " + "-" * 85)
for reg in climate['region'].unique():
    row = climate[climate['region'] == reg]
    spei_m = row['spei12_mean'].mean()
    spei_s = row['spei12_mean'].std()
    dry    = row['drought_year'].sum()
    worst_y = row.loc[row['spei12_mean'].idxmin(), 'year']
    worst_v = row['spei12_mean'].min()
    print(f"  {reg:<38} {spei_m:>+9.3f} {spei_s:>9.3f} {dry:>14} {worst_y:>8} ({worst_v:+.2f})")

# ── Profil national par année ─────────────────────────────────────
print("\n  Profil SPEI national par année :")
print("  " + "-" * 55)
national = climate.groupby('year')['spei12_mean'].mean()
for year, val in national.items():
    if val < -1.5:
        tag = "  ◄ SECHERESSE SEVERE"
    elif val < -1.0:
        tag = "  ◄ Sécheresse marquée"
    elif val < -0.5:
        tag = "  ◄ Sécheresse modérée"
    elif val > 0.5:
        tag = "  ► Année humide"
    else:
        tag = ""
    bar_len = int(abs(val) * 8)
    if val < 0:
        bar = "█" * bar_len
        print(f"  {year}  {'':>8} {bar:<12} {val:+.3f}{tag}")
    else:
        bar = "█" * bar_len
        print(f"  {year}  {bar:<12}{'':>8} {val:+.3f}{tag}")

print("\n  ✓  ETAPE 4 terminée !")
print(f"  ✓  Fichier mis à jour : data/processed/climate_morocco_2000_2024.csv")
