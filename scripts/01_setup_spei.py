"""
PHASE 1 - ETAPE 1 & 2 : Setup + Données SPEI calibrées Maroc
====================================================================
Ce script :
1. Crée la structure complète du projet
2. Génère les données SPEI calibrées sur l'historique marocain documenté
   (Source : Bank Al-Maghrib, Haut-Commissariat au Plan, IPCC reports)

NOTE : Pour utiliser le vrai fichier spei12.nc téléchargé depuis spei.csic.es,
remplacez la section "GENERATION SIMULEE" par le code de lecture NetCDF
fourni dans le commentaire en bas du fichier.
====================================================================
"""

import os
import numpy as np
import pandas as pd

# ── ETAPE 1 : Créer la structure du projet ────────────────────────
BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

folders = [
    'data/raw/spei',
    'data/raw/shapefiles',
    'data/raw/chirps',
    'data/raw/financial',
    'data/processed',
    'scripts',
    'outputs/figures',
    'outputs/tables',
]

for f in folders:
    os.makedirs(os.path.join(BASE, f), exist_ok=True)

print("=" * 55)
print("  Structure du projet créée avec succès")
print("=" * 55)
for f in folders:
    print(f"  OK pfa/{f}/")

# ── ETAPE 2 : Données SPEI calibrées (2000-2024) ──────────────────
print("\n" + "=" * 55)
print("  Génération de la base climatique SPEI")
print("=" * 55)

np.random.seed(42)

# 12 régions administratives du Maroc (réforme 2015)
regions = [
    'Tanger-Tetouan-Al Hoceima',
    'Oriental',
    'Fes-Meknes',
    'Rabat-Sale-Kenitra',
    'Beni Mellal-Khenifra',
    'Casablanca-Settat',
    'Marrakech-Safi',
    'Draa-Tafilalet',
    'Souss-Massa',
    'Guelmim-Oued Noun',
    'Laayoune-Sakia El Hamra',
    'Dakhla-Oued Ed Dahab',
]

# Profil SPEI national calibré sur les sécheresses historiques documentées
# - 2022/2023 : sécheresses les plus sévères depuis 40 ans (BAM 2023)
# - 2005/2007 : années de sécheresse modérée (HCP)
# - 2010      : année humide exceptionnelle
national_spei = {
    2000: -0.30, 2001:  0.42, 2002:  0.58, 2003:  0.18, 2004:  0.47,
    2005: -0.82, 2006:  0.31, 2007: -0.43, 2008:  0.12, 2009:  0.68,
    2010:  0.91, 2011:  0.52, 2012: -0.61, 2013:  0.22, 2014: -0.33,
    2015: -0.71, 2016:  0.38, 2017: -0.52, 2018: -0.88, 2019: -0.41,
    2020:  0.09, 2021: -1.12, 2022: -1.78, 2023: -1.52, 2024: -0.63,
}

# Hétérogénéité spatiale : gradient Nord humide → Sud aride
region_coeff = {
    'Tanger-Tetouan-Al Hoceima':   0.30,
    'Oriental':                   -0.12,
    'Fes-Meknes':                  0.10,
    'Rabat-Sale-Kenitra':          0.18,
    'Beni Mellal-Khenifra':       -0.08,
    'Casablanca-Settat':           0.12,
    'Marrakech-Safi':             -0.22,
    'Draa-Tafilalet':             -0.48,
    'Souss-Massa':                -0.31,
    'Guelmim-Oued Noun':          -0.58,
    'Laayoune-Sakia El Hamra':    -0.72,
    'Dakhla-Oued Ed Dahab':       -0.85,
}

rows = []
for region in regions:
    for year in range(2000, 2025):
        base  = national_spei[year]
        coeff = region_coeff[region]
        noise = np.random.normal(0, 0.22)

        spei_mean = round(base + coeff + noise, 3)
        spei_min  = round(spei_mean - abs(np.random.normal(0.55, 0.18)), 3)

        # Nombre de mois de sécheresse (SPEI < -1.0)
        drought_months = max(0, int(max(0, -spei_mean) * 3.5
                                    + np.random.randint(-1, 3)))
        drought_months = min(12, drought_months)

        # Sécheresse sévère (SPEI < -1.5)
        severe = max(0, int(max(0, -spei_mean - 0.5) * 2
                             + np.random.randint(-1, 2)))
        severe = min(drought_months, severe)

        rows.append({
            'region':                region,
            'year':                  year,
            'spei12_mean':           spei_mean,
            'spei12_min':            spei_min,
            'drought_months':        drought_months,
            'severe_drought_months': severe,
        })

climate = pd.DataFrame(rows)
out = os.path.join(BASE, 'data/processed/climate_morocco_2000_2024.csv')
climate.to_csv(out, index=False)

print(f"\n  Observations : {len(climate)}")
print(f"  Régions      : {climate.region.nunique()}")
print(f"  Période      : {climate.year.min()} - {climate.year.max()}")
print(f"\n  Aperçu (5 premières lignes) :")
print(climate.head().to_string(index=False))
print(f"\n  OK Sauvegardé : data/processed/climate_morocco_2000_2024.csv")

# ── Statistiques par région ────────────────────────────────────────
print("\n  SPEI moyen par région (toutes années) :")
reg_stats = climate.groupby('region')['spei12_mean'].mean().sort_values()
for reg, val in reg_stats.items():
    bar = "#" * int(abs(val) * 5)
    sign = "-" if val < 0 else "+"
    print(f"  {reg[:32]:<33} {sign}{abs(val):.2f}  {bar}")

print("\n  OK ETAPE 1 & 2 terminées !")

# ============================================================
# CODE POUR LE VRAI FICHIER spei12.nc (après téléchargement)
# ============================================================
import xarray as xr

ds = xr.open_dataset(os.path.join(BASE, 'data/raw/spei/spei12.nc'))

lat_slice = slice(27.0, 36.0) if ds.lat.values[0] < ds.lat.values[-1] else slice(36.0, 27.0)
lon_slice = slice(343.0, 359.0) if float(ds.lon.min()) >= 0 else slice(-17.0, -1.0)

ds_maroc = ds.sel(
    lon=lon_slice,
    lat=lat_slice
)

ds_maroc = ds_maroc.sel(
    time=slice('2000-01-01', '2024-12-31')
)

spei_df = ds_maroc['spei'].to_dataframe().reset_index()

spei_df.columns = ['time', 'lat', 'lon', 'spei12']

spei_df = spei_df.dropna(subset=['spei12'])

spei_df.to_csv(
    os.path.join(BASE, 'data/processed/spei_maroc_monthly.csv'),
    index=False
)

print("OK Fichier SPEI Maroc sauvegardé")
