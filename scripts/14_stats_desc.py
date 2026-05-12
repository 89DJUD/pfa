"""
PHASE 5 - ETAPE 5.1 : Tableau de statistiques descriptives
============================================================
Input  : data/processed/panel_final_merged.csv
Output : outputs/tables/table1_stats_desc.xlsx si openpyxl est installe,
         sinon outputs/tables/table1_*.csv
============================================================
"""

import os

import pandas as pd


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_T = os.path.join(BASE, 'outputs', 'tables')
os.makedirs(OUT_T, exist_ok=True)

print("=" * 60)
print("  PHASE 5 - ETAPE 5.1 : Statistiques descriptives")
print("=" * 60)

panel_path = os.path.join(BASE, 'data', 'processed', 'panel_final_merged.csv')
df = pd.read_csv(panel_path)

print(f"\n  Panel charge : {len(df)} obs, {df['firm_id'].nunique()} entreprises")
print(f"  Periode      : {df['year'].min()} - {df['year'].max()}")

# Variables d'interet
vars_desc = [
    'zscore', 'spei12_mean', 'spei12_min', 'drought_months',
    'carbon_burden_mid', 'log_assets', 'leverage', 'liquidity', 'roa'
]
vars_desc = [v for v in vars_desc if v in df.columns]

# Tableau descriptif complet
desc = df[vars_desc].describe().T
desc['median'] = df[vars_desc].median()
desc['skew'] = df[vars_desc].skew()
desc = desc[['count', 'mean', 'median', 'std', 'min', '25%', '75%', 'max', 'skew']]
desc.columns = ['N', 'Moyenne', 'Mediane', 'Ecart-type', 'Min', 'P25', 'P75', 'Max', 'Asymetrie']

print("\n  Statistiques descriptives :")
print("  " + "-" * 80)
print(desc.round(3).to_string())

# Stats par secteur (Z-score)
sec_stats = None
if 'sector' in df.columns:
    print("\n  Z-score moyen par secteur :")
    print("  " + "-" * 45)
    sec_stats = df.groupby('sector')['zscore'].agg(['mean', 'std', 'min', 'max']).round(3)
    print(sec_stats.to_string())

# Sauvegarde Excel ou CSV
stats_year = df.groupby('year')['zscore'].agg(['mean', 'std', 'count']).round(3)
out_path = os.path.join(OUT_T, 'table1_stats_desc.xlsx')

try:
    with pd.ExcelWriter(out_path, engine='openpyxl') as writer:
        desc.round(3).to_excel(writer, sheet_name='Stats_globales')
        if sec_stats is not None:
            sec_stats.to_excel(writer, sheet_name='Stats_par_secteur')
        stats_year.to_excel(writer, sheet_name='Stats_par_annee')
    print("\n  OK Sauvegarde : outputs/tables/table1_stats_desc.xlsx")
except ModuleNotFoundError:
    desc.round(3).to_csv(os.path.join(OUT_T, 'table1_stats_globales.csv'))
    if sec_stats is not None:
        sec_stats.to_csv(os.path.join(OUT_T, 'table1_stats_par_secteur.csv'))
    stats_year.to_csv(os.path.join(OUT_T, 'table1_stats_par_annee.csv'))
    print("\n  ATTENTION : openpyxl n'est pas installe, fichier Excel ignore.")
    print("  OK Sauvegarde CSV : outputs/tables/table1_stats_globales.csv")
    if sec_stats is not None:
        print("  OK Sauvegarde CSV : outputs/tables/table1_stats_par_secteur.csv")
    print("  OK Sauvegarde CSV : outputs/tables/table1_stats_par_annee.csv")

print("  OK ETAPE 5.1 terminee -> lancez 15_correlation.py")
