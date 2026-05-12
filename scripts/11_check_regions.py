"""
PHASE 4 - ETAPE 4.1 : Vérification des régions avant fusion
=============================================================
Vérifie que les noms de régions sont identiques entre :
  - financial_panel_morocco.csv
  - climate_morocco_spei_2000_2024.csv
  - carbon_tax_panel.csv
=============================================================
"""

import pandas as pd
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT  = os.path.join(BASE, 'data', 'processed')
RAW  = os.path.join(BASE, 'data', 'raw')

print("=" * 60)
print("  PHASE 4 — ETAPE 4.1 : Vérification des régions")
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
climate = pd.read_csv(climate_path)
carbon  = pd.read_csv(carbon_path)

print(f"\n  Panel financier : {len(fin)} obs, {fin['firm_id'].nunique()} entreprises")
print(f"  Base climatique : {len(climate)} obs")
print(f"  Taxe carbone    : {len(carbon)} obs")

# ── Régions dans chaque fichier ───────────────────────────────────
regions_fin     = sorted(fin['region'].unique())
regions_climate = sorted(climate['region'].unique())

print(f"\n  Régions dans le panel financier ({len(regions_fin)}) :")
for r in regions_fin:
    print(f"    - {r}")

print(f"\n  Régions dans la base climatique ({len(regions_climate)}) :")
for r in regions_climate:
    print(f"    - {r}")

# ── Vérification des correspondances ─────────────────────────────
manquantes_fin     = set(regions_fin) - set(regions_climate)
manquantes_climate = set(regions_climate) - set(regions_fin)

if not manquantes_fin and not manquantes_climate:
    print(f"\n  OK Toutes les régions matchent parfaitement !")
else:
    if manquantes_fin:
        print(f"\n  ATTENTION : Régions dans financier MAIS PAS dans climatique :")
        for r in sorted(manquantes_fin):
            print(f"    - '{r}'")
    if manquantes_climate:
        print(f"\n  ATTENTION : Régions dans climatique MAIS PAS dans financier :")
        for r in sorted(manquantes_climate):
            print(f"    - '{r}'")

# ── Dictionnaire de correspondance (à adapter si besoin) ─────────
# Si des régions ne matchent pas, modifiez ce dictionnaire
region_mapping = {
    'Rabat-Sale-Kenitra'         : 'Rabat-Sale-Kenitra',
    'Casablanca-Settat'          : 'Casablanca-Settat',
    'Marrakech-Safi'             : 'Marrakech-Safi',
    'Fes-Meknes'                 : 'Fes-Meknes',
    'Souss-Massa'                : 'Souss-Massa',
    'Tanger-Tetouan-Al Hoceima'  : 'Tanger-Tetouan-Al Hoceima',
    'Oriental'                   : 'Oriental',
    'Beni Mellal-Khenifra'       : 'Beni Mellal-Khenifra',
    'Draa-Tafilalet'             : 'Draa-Tafilalet',
    'Guelmim-Oued Noun'          : 'Guelmim-Oued Noun',
    'Laayoune-Sakia El Hamra'    : 'Laayoune-Sakia El Hamra',
    'Dakhla-Oued Ed-Dahab'       : 'Dakhla-Oued Ed-Dahab',
}

fin['region'] = fin['region'].map(region_mapping).fillna(fin['region'])

# ── Vérification des années ───────────────────────────────────────
print(f"\n  Années dans le panel financier : {fin['year'].min()} – {fin['year'].max()}")
print(f"  Années dans la base climatique : {climate['year'].min()} – {climate['year'].max()}")
print(f"  Années dans la taxe carbone    : {carbon['year'].min()} – {carbon['year'].max()}")

# ── Résumé ────────────────────────────────────────────────────────
print(f"\n  OK ETAPE 4.1 terminée")
print(f"  Prochaine etape : python 12_merge.py")
