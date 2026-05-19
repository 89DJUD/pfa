"""
PHASE 2 - ETAPE 2 : Chargement et nettoyage des états financiers réels
====================================================================
Source : Données collectées par scraper_ammc.py (financial_raw.csv)
         → Rapports financiers annuels (RFA) publiés sur ammc.ma

Ce script remplace la simulation par les vraies données AMMC.
Il effectue :
  1. Chargement du financial_raw.csv collecté par le scraper
  2. Merge avec companies_list pour ajouter ticker + infos secteur
  3. Nettoyage : valeurs aberrantes, négatifs impossibles
  4. Imputation des valeurs manquantes (médiane sectorielle)
  5. Rapport de qualité complet
  6. Sauvegarde finale dans financial_raw.csv (écrase la version simulée)
====================================================================
"""

import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 60)
print("  PHASE 2 — ETAPE 2 : Chargement données réelles AMMC")
print("=" * 60)

# ════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Charger les deux fichiers sources
# ════════════════════════════════════════════════════════════════

raw_path = os.path.join(BASE, 'data/processed/financial_raw.csv')
cmp_path = os.path.join(BASE, 'data/processed/companies_list.csv')

if not os.path.exists(raw_path):
    raise FileNotFoundError(
        f"\n  ✗ Fichier introuvable : {raw_path}\n"
        "    → Lancez d'abord scraper_ammc.py pour collecter les données."
    )

df_raw      = pd.read_csv(raw_path)
df_companies = pd.read_csv(cmp_path)

print(f"\n  Données brutes chargées    : {len(df_raw):,} lignes")
print(f"  Entreprises dans le panel  : {df_raw['firm_id'].nunique()}")
if 'year' in df_raw.columns:
    print(f"  Période couverte           : {df_raw['year'].min()} – {df_raw['year'].max()}")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Merge avec companies_list
# ════════════════════════════════════════════════════════════════

# Colonnes à récupérer depuis companies_list (si absentes du raw)
cols_to_add = [c for c in ['ticker', 'sector', 'region', 'year_listed']
               if c not in df_raw.columns]

if cols_to_add:
    df_raw = df_raw.merge(
        df_companies[['firm_id'] + cols_to_add],
        on='firm_id',
        how='left'
    )
    print(f"\n  Colonnes ajoutées depuis companies_list : {cols_to_add}")

# S'assurer que firm_name et sector sont bien présents
for col in ['firm_name', 'sector', 'region']:
    if col not in df_raw.columns:
        df_raw = df_raw.merge(
            df_companies[['firm_id', col]], on='firm_id', how='left'
        )

# ════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Nettoyage de base
# ════════════════════════════════════════════════════════════════

print("\n  ── Nettoyage ─────────────────────────────────")

n_avant = len(df_raw)

# 3a. Supprimer les lignes sans aucune donnée financière utile
cols_financiers = ['total_assets', 'current_assets', 'current_liabilities',
                   'retained_earnings', 'ebit', 'total_debt', 'revenue']

# Garder seulement les colonnes qui existent réellement dans le fichier
cols_present = [c for c in cols_financiers if c in df_raw.columns]

df_raw = df_raw.dropna(subset=['total_assets', 'revenue'], how='all')
print(f"  Lignes supprimées (total_assets ET revenue manquants) : "
      f"{n_avant - len(df_raw)}")

# 3b. Les valeurs obligatoirement positives ne peuvent pas être négatives
cols_positifs = ['total_assets', 'current_assets', 'revenue', 'market_cap']
for col in cols_positifs:
    if col in df_raw.columns:
        n_neg = (df_raw[col] < 0).sum()
        if n_neg > 0:
            print(f"  ⚠ {col} : {n_neg} valeurs négatives → remplacées par NaN")
            df_raw.loc[df_raw[col] < 0, col] = np.nan

# 3c. Supprimer les doublons firm_id × year
n_avant = len(df_raw)
df_raw = df_raw.drop_duplicates(subset=['firm_id', 'year'], keep='first')
if len(df_raw) < n_avant:
    print(f"  ⚠ {n_avant - len(df_raw)} doublons supprimés (firm_id × year)")

# 3d. Trier le panel
df_raw = df_raw.sort_values(['firm_id', 'year']).reset_index(drop=True)

# ════════════════════════════════════════════════════════════════
# ÉTAPE 4 — Rapport qualité AVANT imputation
# ════════════════════════════════════════════════════════════════

print("\n  ── Taux de remplissage par variable (avant imputation) ──")
print(f"  {'Variable':<28} {'Rempli':>8}  {'Manquant':>8}  {'Taux':>6}")
print("  " + "-" * 58)

for col in cols_present + (['market_cap'] if 'market_cap' in df_raw.columns else []):
    n_ok  = df_raw[col].notna().sum()
    n_nan = df_raw[col].isna().sum()
    pct   = n_ok / len(df_raw) * 100
    flag  = "✓" if pct >= 70 else ("⚠" if pct >= 40 else "✗")
    print(f"  {flag} {col:<26} {n_ok:>8}  {n_nan:>8}  {pct:>5.1f}%")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 5 — Imputation des valeurs manquantes
# ════════════════════════════════════════════════════════════════

print("\n  ── Imputation des valeurs manquantes ────────────────────")

# 5a. market_cap : souvent absent (non collecté par scraper)
#     → imputer par P/B ratio médian sectoriel × (actif - dettes)
if 'market_cap' in df_raw.columns:
    pct_market_cap = df_raw['market_cap'].notna().mean() * 100
    print(f"  market_cap disponible à {pct_market_cap:.1f}% "
          f"— imputation P/B médian sectoriel pour le reste")

    # P/B médians par secteur (calibrés sur BVC)
    pb_secteur = {
        "Agro-alimentaire & Boissons": 2.5,
        "Materiaux de construction":   2.0,
        "Energie & Mines":             2.2,
        "Distribution & Commerce":     1.8,
        "Immobilier":                  1.5,
        "Telecom & Technologies":      3.5,
        "Transport & Logistique":      1.9,
        "Chimie & Parachimie":         2.1,
        "Industrie (autres)":          1.6,
    }

    mask_missing = df_raw['market_cap'].isna()
    if mask_missing.any() and 'total_assets' in df_raw.columns and 'total_debt' in df_raw.columns:
        df_raw.loc[mask_missing, 'market_cap'] = df_raw.loc[mask_missing].apply(
            lambda r: max(
                (r['total_assets'] - r['total_debt']) *
                pb_secteur.get(r.get('sector', ''), 2.0),
                10
            ) if pd.notna(r['total_assets']) and pd.notna(r['total_debt']) else np.nan,
            axis=1
        )

# 5b. Variables financières secondaires
#     → imputation par médiane sectorielle × année
for col in ['current_assets', 'current_liabilities', 'retained_earnings',
            'ebit', 'total_debt']:
    if col not in df_raw.columns:
        continue
    n_avant_imp = df_raw[col].isna().sum()
    if n_avant_imp == 0:
        continue

    # Médiane sectorielle par année
    mediane = df_raw.groupby(['sector', 'year'])[col].transform('median')
    df_raw[col] = df_raw[col].fillna(mediane)

    # Si toujours manquant (secteur/année sans données), médiane globale
    mediane_globale = df_raw[col].median()
    df_raw[col] = df_raw[col].fillna(mediane_globale)

    n_apres_imp = df_raw[col].isna().sum()
    print(f"  {col:<28} : {n_avant_imp} → {n_apres_imp} manquants")

# 5c. Imputer total_assets et revenue par interpolation temporelle
#     (meilleur que médiane pour des séries continues par entreprise)
for col in ['total_assets', 'revenue']:
    if col not in df_raw.columns:
        continue
    n_avant_imp = df_raw[col].isna().sum()
    if n_avant_imp == 0:
        continue
    df_raw[col] = (df_raw
                   .groupby('firm_id')[col]
                   .transform(lambda s: s.interpolate(method='linear',
                                                       limit_direction='both')))
    # Résiduel → médiane sectorielle
    med = df_raw.groupby('sector')[col].transform('median')
    df_raw[col] = df_raw[col].fillna(med)
    n_apres_imp = df_raw[col].isna().sum()
    print(f"  {col:<28} : {n_avant_imp} → {n_apres_imp} manquants (interpolation)")

# ════════════════════════════════════════════════════════════════
# ÉTAPE 6 — Vérifications finales
# ════════════════════════════════════════════════════════════════

print("\n  ── Vérifications finales ────────────────────────────────")

# Cohérence bilan : current_assets <= total_assets
if 'current_assets' in df_raw.columns:
    incoherent = (df_raw['current_assets'] > df_raw['total_assets']).sum()
    if incoherent > 0:
        print(f"  ⚠ {incoherent} lignes où current_assets > total_assets "
              f"→ plafonnées à total_assets × 0.8")
        df_raw.loc[df_raw['current_assets'] > df_raw['total_assets'],
                   'current_assets'] = df_raw['total_assets'] * 0.8

# Cohérence : total_debt <= total_assets
if 'total_debt' in df_raw.columns:
    incoherent = (df_raw['total_debt'] > df_raw['total_assets']).sum()
    if incoherent > 0:
        print(f"  ⚠ {incoherent} lignes où total_debt > total_assets "
              f"→ plafonnées à total_assets × 0.9")
        df_raw.loc[df_raw['total_debt'] > df_raw['total_assets'],
                   'total_debt'] = df_raw['total_assets'] * 0.9

# ════════════════════════════════════════════════════════════════
# ÉTAPE 7 — Rapport final + Sauvegarde
# ════════════════════════════════════════════════════════════════

print("\n  ── Résumé final ─────────────────────────────────────────")
print(f"  Observations retenues  : {len(df_raw):,}")
print(f"  Entreprises            : {df_raw['firm_id'].nunique()}")
if 'year' in df_raw.columns:
    print(f"  Période                : {df_raw['year'].min()} – {df_raw['year'].max()}")
if 'sector' in df_raw.columns:
    print(f"  Secteurs               : {df_raw['sector'].nunique()}")

print(f"\n  Taux de remplissage final :")
for col in cols_present + (['market_cap'] if 'market_cap' in df_raw.columns else []):
    pct = df_raw[col].notna().mean() * 100
    flag = "✓" if pct >= 95 else ("⚠" if pct >= 70 else "✗")
    print(f"    {flag} {col:<26} : {pct:.1f}%")

print(f"\n  Aperçu (3 premières lignes) :")
cols_apercu = ['firm_id', 'firm_name', 'sector', 'year',
               'total_assets', 'revenue', 'ebit']
cols_apercu = [c for c in cols_apercu if c in df_raw.columns]
print(df_raw[cols_apercu].head(3).to_string(index=False))

# Sauvegarde (écrase financial_raw.csv avec version nettoyée)
df_raw.to_csv(raw_path, index=False)
print(f"\n  ✓  Sauvegardé : data/processed/financial_raw.csv")
print(f"     ({len(df_raw):,} obs × {len(df_raw.columns)} colonnes)")
print("\n  ✓  ETAPE 2 terminée ! → Lancer script_03_zscore.py")
