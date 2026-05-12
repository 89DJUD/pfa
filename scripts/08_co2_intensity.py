"""
PHASE 3 - ETAPE 3.2 : Intensités CO2 sectorielles + Taxe carbone
=================================================================
Source : fichiers IEA téléchargés (data/raw/)
  - CO2_emissions_by_sector_-_Morocco.csv
  - Carbon_intensity_of_industry_energy_consumption_-_Morocco.csv
  - Final_energy_carbon_intensity_-_Morocco.csv

Scénarios taxe carbone :
  - Bas   : 25 $/tCO2 (NDC Maroc)
  - Moyen : 50 $/tCO2 (IEA transition ordonnée)
  - Haut  : 75 $/tCO2 (IEA 2°C SDS)

Formule :
  Coût = Taxe (MAD/kgCO2) × Intensité (kgCO2/MAD_CA) × CA
=================================================================
"""

import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW  = os.path.join(BASE, 'data', 'raw')
OUT  = os.path.join(BASE, 'data', 'processed')
os.makedirs(OUT, exist_ok=True)

print("=" * 60)
print("  PHASE 3 — ETAPE 3.2 : Intensités CO2 sectorielles")
print("=" * 60)

# ── Fonction chargement CSV IEA ───────────────────────────────────
def load_iea_csv(filename):
    """Charge un CSV IEA (ignore les 2 lignes d'en-tête source/licence)."""
    path = os.path.join(RAW, filename)
    df = pd.read_csv(path, skiprows=3)
    df.columns = df.columns.str.strip()
    df = df.rename(columns={df.columns[0]: 'year'})
    df['year'] = pd.to_numeric(df['year'], errors='coerce')
    df = df.dropna(subset=['year'])
    df['year'] = df['year'].astype(int)
    df = df.drop(columns=['Units'], errors='ignore')
    return df

# ── Chargement des 3 fichiers IEA ────────────────────────────────
df_sector   = load_iea_csv('CO2 emissions by sector - Morocco.csv')
df_industry = load_iea_csv('Carbon intensity of industry energy consumption - Morocco.csv')
df_final    = load_iea_csv('Final energy carbon intensity - Morocco.csv')

df_industry.columns = ['year', 'intensity_industry_gCO2_MJ']
df_final.columns    = ['year', 'intensity_final_gCO2_MJ']

# ── Filtrer 2000-2023 ─────────────────────────────────────────────
df_sector   = df_sector[(df_sector['year'] >= 2000) & (df_sector['year'] <= 2023)]
df_industry = df_industry[(df_industry['year'] >= 2000) & (df_industry['year'] <= 2023)]
df_final    = df_final[(df_final['year'] >= 2000) & (df_final['year'] <= 2023)]

# ── Calcul des émissions sectorielles moyennes IEA (MtCO2) ───────
em = df_sector[[
    'Electricity and heat producers',
    'Transport Sector',
    'Industry Sector',
    'Commercial and Public Services',
    'Residential',
    'Agriculture/Forestry'
]].mean()

total = em.sum()

print(f"\n  Émissions moyennes 2000-2023 (IEA Maroc) :")
for sect, val in em.items():
    print(f"  {sect:<40} : {val:.2f} MtCO2 ({val/total*100:.1f}%)")

# ── Intensités moyennes IEA ───────────────────────────────────────
intens_industrie = df_industry['intensity_industry_gCO2_MJ'].mean()
intens_finale    = df_final['intensity_final_gCO2_MJ'].mean()

print(f"\n  Intensité industrie IEA (moy 2000-2023) : {intens_industrie:.2f} gCO2/MJ")
print(f"  Intensité énergie finale IEA            : {intens_finale:.2f} gCO2/MJ")

# ── Calibration sur données IEA réelles ──────────────────────────
# Fourchettes de référence du guide (IEA) → on prend le milieu
# Puis on calibre par le facteur d'intensité industrie Maroc
# vs moyenne mondiale (65 gCO2/MJ selon IEA World Energy Balances)
facteur = intens_industrie / 65.0
print(f"\n  Facteur calibration Maroc/Monde : {facteur:.3f}")

ref_guide = {
    'Ciment & Materiaux de construction' : 0.55,
    'Energie & Mines'                    : 1.00,
    'Agro-alimentaire & Boissons'        : 0.12,
    'Chimie & Parachimie'                : 0.32,
    'Transport & Logistique'             : 0.28,
    'Industrie (autres)'                 : 0.20,
    'Distribution & Commerce'            : 0.06,
    'Immobilier'                         : 0.10,
    'Telecom & Technologies'             : 0.02,
}

fourchettes = {
    'Ciment & Materiaux de construction' : '0.45-0.65',
    'Energie & Mines'                    : '0.80-1.20',
    'Agro-alimentaire & Boissons'        : '0.08-0.15',
    'Chimie & Parachimie'                : '0.25-0.40',
    'Transport & Logistique'             : '0.20-0.35',
    'Industrie (autres)'                 : '-',
    'Distribution & Commerce'            : '0.03-0.08',
    'Immobilier'                         : '-',
    'Telecom & Technologies'             : '0.01-0.03',
}

# Intensités calibrées avec données IEA réelles Maroc
intensites_calibrees = {
    sect: round(val * facteur, 4)
    for sect, val in ref_guide.items()
}

# ── Construction du DataFrame ─────────────────────────────────────
intensity_df = pd.DataFrame(
    list(intensites_calibrees.items()),
    columns=['sector', 'co2_intensity_kgCO2_per_MAD']
)
intensity_df['fourchette_IEA_guide'] = intensity_df['sector'].map(fourchettes)
intensity_df['source'] = 'IEA Morocco 2000-2023 calibré'

# ── Affichage tableau intensités ──────────────────────────────────
print(f"\n  Intensités CO2 sectorielles (kgCO2 / MAD de CA) :")
print("  " + "-" * 65)
print(f"  {'Secteur':<35} {'kgCO2/MAD':>10}  Fourchette guide")
print("  " + "-" * 65)
for _, row in intensity_df.iterrows():
    bar = "#" * int(row['co2_intensity_kgCO2_per_MAD'] * 15)
    print(f"  {row['sector']:<35} {row['co2_intensity_kgCO2_per_MAD']:>10.4f}  {bar}")

# ── Scénarios de taxe carbone ─────────────────────────────────────
# Conversion : 25$/tCO2 → 25/1000 $/kgCO2 × 10 MAD/$ = 0.25 MAD/kgCO2
usd_to_mad = 10.0
scenarios  = {'low': 25, 'mid': 50, 'high': 75}

print(f"\n  Scénarios taxe carbone (1 USD = {usd_to_mad} MAD) :")
print("  " + "-" * 50)
for name, tax_usd in scenarios.items():
    tax_mad_per_kg = (tax_usd / 1000) * usd_to_mad
    intensity_df[f'tax_mad_per_kgCO2_{name}'] = tax_mad_per_kg
    print(f"  Scénario {name:4s} ({tax_usd:2d} $/tCO2) -> {tax_mad_per_kg:.4f} MAD/kgCO2")

# ── Sauvegarde ───────────────────────────────────────────────────
out_path = os.path.join(OUT, 'co2_intensity_sectors.csv')
intensity_df.to_csv(out_path, index=False)

print(f"\n  OK Sauvegardé : data/processed/co2_intensity_sectors.csv")
print(f"  Dimensions   : {intensity_df.shape[0]} secteurs × {intensity_df.shape[1]} colonnes")
print("\n  OK ETAPE 3.2 terminée — lancez maintenant 09_carbon_cost.py")
