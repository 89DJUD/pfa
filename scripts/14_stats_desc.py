"""
Phase 5 – Étape 5.1
Tableau de statistiques descriptives
Panel : 41 entreprises marocaines cotées | 2019-2024
"""

import pandas as pd
import numpy as np
import os

# ── Chargement ────────────────────────────────────────────────────────────────
df = pd.read_csv('data/panel_final_merged.csv')
os.makedirs('outputs/tables',   exist_ok=True)
os.makedirs('outputs/figures',  exist_ok=True)

print(f"Panel chargé : {len(df)} obs | {df['firm_id'].nunique()} entreprises "
      f"| {df['year'].min()}–{df['year'].max()}")

# ── Variables d'intérêt ───────────────────────────────────────────────────────
vars_desc = {
    'zscore'             : 'Z-score Altman',
    'spei12_mean'        : 'SPEI-12 moyen régional',
    'spei12_min'         : 'SPEI-12 minimum annuel',
    'drought_months'     : 'Mois de sécheresse (SPEI<-1)',
    'carbon_burden_mid'  : 'Charge carbone – scén. médian (50$/tCO₂)',
    'carbon_burden_low'  : 'Charge carbone – scén. bas (25$/tCO₂)',
    'carbon_burden_high' : 'Charge carbone – scén. haut (75$/tCO₂)',
    'log_assets'         : 'Taille (log Total Actif)',
    'leverage'           : 'Levier financier',
    'liquidity'          : 'Ratio de liquidité',
    'roa'                : 'ROA (Rentabilité des actifs)',
}

# ── Tableau principal ─────────────────────────────────────────────────────────
rows = []
for col, label in vars_desc.items():
    s = df[col].dropna()
    rows.append({
        'Variable'    : label,
        'N'           : int(s.count()),
        'Moyenne'     : round(s.mean(), 3),
        'Médiane'     : round(s.median(), 3),
        'Écart-type'  : round(s.std(), 3),
        'Min'         : round(s.min(), 3),
        'P25'         : round(s.quantile(0.25), 3),
        'P75'         : round(s.quantile(0.75), 3),
        'Max'         : round(s.max(), 3),
        'Asymétrie'   : round(s.skew(), 3),
    })

desc = pd.DataFrame(rows).set_index('Variable')
print("\n── Statistiques descriptives ──────────────────────────────────────────")
print(desc.to_string())

# ── Distribution Z-score par zone ─────────────────────────────────────────────
print("\n── Distribution des zones Z-score ─────────────────────────────────────")
zone_counts = df['in_distress'].value_counts()
total = len(df)
n_distress = int((df['zscore'] < 1.81).sum())
n_grey     = int(((df['zscore'] >= 1.81) & (df['zscore'] < 2.99)).sum())
n_safe     = int((df['zscore'] >= 2.99).sum())
print(f"  Zone sûre     (Z ≥ 2.99) : {n_safe:>4} obs ({n_safe/total*100:.1f}%)")
print(f"  Zone grise    (1.81–2.99): {n_grey:>4} obs ({n_grey/total*100:.1f}%)")
print(f"  Zone détresse (Z < 1.81) : {n_distress:>4} obs ({n_distress/total*100:.1f}%)")

# ── Statistiques par secteur ───────────────────────────────────────────────────
print("\n── Z-score moyen par secteur ───────────────────────────────────────────")
sector_stats = df.groupby('sector').agg(
    N          = ('zscore', 'count'),
    Moy_Zscore = ('zscore', lambda x: round(x.mean(), 3)),
    Med_Zscore = ('zscore', lambda x: round(x.median(), 3)),
    Taux_detresse = ('in_distress', lambda x: f"{x.mean()*100:.1f}%"),
    Moy_SPEI   = ('spei12_mean', lambda x: round(x.mean(), 3)),
    Moy_Carbon = ('carbon_burden_mid', lambda x: round(x.mean(), 4)),
).sort_values('Moy_Zscore')
print(sector_stats.to_string())

# ── Sauvegarde Excel ──────────────────────────────────────────────────────────
with pd.ExcelWriter('outputs/tables/table1_stats_desc.xlsx', engine='openpyxl') as w:
    desc.to_excel(w, sheet_name='Stats_generales')
    sector_stats.to_excel(w, sheet_name='Stats_par_secteur')

    # Stats par année
    year_stats = df.groupby('year').agg(
        N             = ('zscore', 'count'),
        Moy_Zscore    = ('zscore', lambda x: round(x.mean(), 3)),
        Taux_detresse = ('in_distress', lambda x: f"{x.mean()*100:.1f}%"),
        Moy_SPEI      = ('spei12_mean', lambda x: round(x.mean(), 3)),
    )
    year_stats.to_excel(w, sheet_name='Stats_par_annee')

print("\n✅ Sauvegardé : outputs/tables/table1_stats_desc.xlsx")
