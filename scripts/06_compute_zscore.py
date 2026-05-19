"""
PHASE 2 - ETAPE 3 : Calcul du Z-score Altman + nettoyage agressif
====================================================================
"""

import pandas as pd
import numpy as np
from scipy.stats import mstats
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

print("=" * 70)
print("  PHASE 2 — ETAPE 3 : Calcul Z-score Altman")
print("=" * 70)

# ════════════════════════════════════════════════════════════════
# 1. CHARGEMENT
# ════════════════════════════════════════════════════════════════

path = os.path.join(BASE, "data/processed/financial_raw.csv")
df = pd.read_csv(path)

print(f"\nDonnées chargées : {len(df):,} observations")

if "firm_id" in df.columns:
    print(f"Entreprises      : {df['firm_id'].nunique()}")

if "year" in df.columns:
    print(f"Période          : {df['year'].min()} – {df['year'].max()}")

# ════════════════════════════════════════════════════════════════
# 2. NETTOYAGE AGRESSIF
# ════════════════════════════════════════════════════════════════

financial_cols = [
    "total_assets",
    "current_assets",
    "current_liabilities",
    "retained_earnings",
    "ebit",
    "total_debt",
    "revenue",
    "market_cap",
]

financial_cols = [c for c in financial_cols if c in df.columns]

# Convertir en numérique
for col in financial_cols:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Remplacer les infinis
df[financial_cols] = df[financial_cols].replace([np.inf, -np.inf], np.nan)

# Colonnes obligatoirement positives
positive_cols = [
    "total_assets",
    "current_assets",
    "current_liabilities",
    "total_debt",
    "revenue",
    "market_cap",
]

positive_cols = [c for c in positive_cols if c in df.columns]

for col in positive_cols:
    df.loc[df[col] < 0, col] = np.nan

# Supprimer valeurs extrêmes absurdes
upper_limits = {
    "total_assets": 1e14,
    "current_assets": 1e14,
    "current_liabilities": 1e14,
    "retained_earnings": 1e14,
    "ebit": 1e14,
    "total_debt": 1e14,
    "revenue": 1e14,
    "market_cap": 1e15,
}

for col, limit in upper_limits.items():
    if col in df.columns:
        df.loc[df[col].abs() > limit, col] = np.nan

# Supprimer fausses valeurs type années
fake_year_values = [2019, 2020, 2021, 2022, 2023, 2024]

for col in ["total_assets", "revenue", "market_cap", "ebit"]:
    if col in df.columns:
        df.loc[df[col].isin(fake_year_values), col] = np.nan

# Cohérences comptables
if {"current_assets", "total_assets"}.issubset(df.columns):
    df.loc[df["current_assets"] > df["total_assets"], "current_assets"] = np.nan

if {"total_debt", "total_assets"}.issubset(df.columns):
    df.loc[df["total_debt"] > df["total_assets"] * 2, "total_debt"] = np.nan

if {"current_liabilities", "total_assets"}.issubset(df.columns):
    df.loc[
        df["current_liabilities"] > df["total_assets"] * 2,
        "current_liabilities"
    ] = np.nan

# Supprimer lignes inutilisables
df = df.dropna(subset=["total_assets", "revenue"], how="all")

# Trier
if {"firm_id", "year"}.issubset(df.columns):
    df = df.sort_values(["firm_id", "year"]).reset_index(drop=True)

print("\n✓ Nettoyage agressif terminé")
print(f"Observations restantes : {len(df):,}")

# ════════════════════════════════════════════════════════════════
# 3. IMPUTATION DES VALEURS MANQUANTES
# ════════════════════════════════════════════════════════════════

print("\n── Imputation des valeurs manquantes ──")

for col in financial_cols:
    if col not in df.columns:
        continue

    n_before = df[col].isna().sum()

    if n_before == 0:
        continue

    if {"sector", "year"}.issubset(df.columns):
        df[col] = df[col].fillna(
            df.groupby(["sector", "year"])[col].transform("median")
        )

    if "sector" in df.columns:
        df[col] = df[col].fillna(
            df.groupby("sector")[col].transform("median")
        )

    df[col] = df[col].fillna(df[col].median())

    n_after = df[col].isna().sum()
    print(f"{col:<25} : {n_before} → {n_after} manquants")

# ════════════════════════════════════════════════════════════════
# 4. IMPUTATION SPÉCIALE MARKET_CAP
# ════════════════════════════════════════════════════════════════

pb_secteur = {
    "Agro-alimentaire & Boissons": 2.5,
    "Materiaux de construction": 2.0,
    "Energie & Mines": 2.2,
    "Distribution & Commerce": 1.8,
    "Immobilier": 1.5,
    "Telecom & Technologies": 3.5,
    "Transport & Logistique": 1.9,
    "Chimie & Parachimie": 2.1,
    "Industrie (autres)": 1.6,
}

if "market_cap" not in df.columns:
    df["market_cap"] = np.nan


def imputer_market_cap(row):
    if pd.isna(row.get("total_assets")) or pd.isna(row.get("total_debt")):
        return np.nan

    pb = pb_secteur.get(row.get("sector", ""), 2.0)
    val = (row["total_assets"] - row["total_debt"]) * pb

    return max(val, 10)


mask_market = df["market_cap"].isna()

if mask_market.any():
    df.loc[mask_market, "market_cap"] = df.loc[mask_market].apply(
        imputer_market_cap,
        axis=1
    )

print(f"\nmarket_cap rempli à : {df['market_cap'].notna().mean() * 100:.1f}%")

# ════════════════════════════════════════════════════════════════
# 5. COHÉRENCES APRÈS IMPUTATION
# ════════════════════════════════════════════════════════════════

if {"current_assets", "total_assets"}.issubset(df.columns):
    df["current_assets"] = np.minimum(
        df["current_assets"],
        df["total_assets"] * 0.8
    )

if {"total_debt", "total_assets"}.issubset(df.columns):
    df["total_debt"] = np.minimum(
        df["total_debt"],
        df["total_assets"] * 0.9
    )

if {"current_liabilities", "current_assets"}.issubset(df.columns):
    df["current_liabilities"] = np.minimum(
        df["current_liabilities"],
        df["current_assets"] * 1.5
    )

# Éviter les divisions par zéro
df["total_assets"] = df["total_assets"].replace(0, np.nan)
df["total_debt"] = df["total_debt"].replace(0, np.nan)
df["current_liabilities"] = df["current_liabilities"].replace(0, np.nan)

# ════════════════════════════════════════════════════════════════
# 6. CALCUL DES RATIOS ALTMAN
# ════════════════════════════════════════════════════════════════

print("\n── Calcul des ratios Altman ──")

df["X1"] = (
    df["current_assets"] - df["current_liabilities"]
) / df["total_assets"]

df["X2"] = df["retained_earnings"] / df["total_assets"]
df["X3"] = df["ebit"] / df["total_assets"]
df["X4"] = df["market_cap"] / df["total_debt"]
df["X5"] = df["revenue"] / df["total_assets"]

ratio_cols = ["X1", "X2", "X3", "X4", "X5"]

# Remplacer infinis
df[ratio_cols] = df[ratio_cols].replace([np.inf, -np.inf], np.nan)

# Imputer ratios résiduels
for col in ratio_cols:
    if "sector" in df.columns:
        df[col] = df[col].fillna(
            df.groupby("sector")[col].transform("median")
        )

    df[col] = df[col].fillna(df[col].median())

# Winsorisation P1-P99
for col in ratio_cols:
    df[col] = mstats.winsorize(df[col], limits=[0.01, 0.01])
# ════════════════════════════════════════════════════════════════
# CLIPPING DES RATIOS ALTMAN
# Empêche les valeurs irréalistes
# ════════════════════════════════════════════════════════════════

df["X1"] = df["X1"].clip(-2, 2)
df["X2"] = df["X2"].clip(-2, 2)
df["X3"] = df["X3"].clip(-2, 2)
df["X4"] = df["X4"].clip(0, 10)
df["X5"] = df["X5"].clip(0, 5)
print(df[ratio_cols].describe().round(3))

# ════════════════════════════════════════════════════════════════
# 7. CALCUL DU Z-SCORE
# ════════════════════════════════════════════════════════════════

df["zscore"] = (
    1.2 * df["X1"]
    + 1.4 * df["X2"]
    + 3.3 * df["X3"]
    + 0.6 * df["X4"]
    + 1.0 * df["X5"]
)

df["zscore"] = mstats.winsorize(df["zscore"], limits=[0.01, 0.01])

# ════════════════════════════════════════════════════════════════
# 8. CLASSIFICATION
# ════════════════════════════════════════════════════════════════

def classify_zscore(z):
    if pd.isna(z):
        return "Unknown"
    elif z > 2.99:
        return "Safe"
    elif z >= 1.81:
        return "Grey"
    else:
        return "Distress"


df["zscore_zone"] = df["zscore"].apply(classify_zscore)

df["in_distress"] = (df["zscore"] < 1.81).astype(int)
df["in_grey"] = ((df["zscore"] >= 1.81) & (df["zscore"] <= 2.99)).astype(int)
df["in_safe"] = (df["zscore"] > 2.99).astype(int)

# ════════════════════════════════════════════════════════════════
# 9. VARIABLES DE CONTRÔLE
# ════════════════════════════════════════════════════════════════

df["log_assets"] = np.log(df["total_assets"].clip(lower=1))

df["leverage"] = (
    df["total_debt"] / df["total_assets"]
).clip(0, 1)

df["liquidity"] = (
    df["current_assets"] / df["current_liabilities"]
).clip(0, 10)

df["roa"] = (
    df["ebit"] / df["total_assets"]
).clip(-1, 1)

if "firm_id" in df.columns:
    df["revenue_growth"] = (
        df.groupby("firm_id")["revenue"]
        .pct_change()
        .replace([np.inf, -np.inf], np.nan)
        .clip(-0.5, 1.0)
    )

    df["zscore_lag1"] = df.groupby("firm_id")["zscore"].shift(1)
else:
    df["revenue_growth"] = np.nan
    df["zscore_lag1"] = np.nan

# ════════════════════════════════════════════════════════════════
# 10. RAPPORT FINAL
# ════════════════════════════════════════════════════════════════

print("\n── Distribution des zones Z-score ──")
print(df["zscore_zone"].value_counts())

print("\n── Statistiques Z-score ──")
print(df["zscore"].describe().round(3))

if "sector" in df.columns:
    print("\n── Taux de détresse par secteur ──")
    sector_report = df.groupby("sector").agg(
        observations=("zscore", "count"),
        zscore_moyen=("zscore", "mean"),
        taux_detresse=("in_distress", "mean")
    ).sort_values("taux_detresse", ascending=False)

    print(sector_report.round(3))

# ════════════════════════════════════════════════════════════════
# 11. SAUVEGARDE
# ════════════════════════════════════════════════════════════════

out = os.path.join(BASE, "data/processed/financial_zscore.csv")
df.to_csv(out, index=False)

print("\n✓ Fichier sauvegardé :")
print(out)

print("\n✓ Calcul Z-score terminé avec succès")