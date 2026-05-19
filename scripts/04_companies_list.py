import pandas as pd
import numpy as np
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

df_real = pd.read_csv(os.path.join(BASE, 'data/processed/financial_raw.csv'))

# Ajouter ticker depuis companies_list
companies = pd.read_csv(os.path.join(BASE, 'data/processed/companies_list.csv'))
df_real = df_real.merge(
    companies[['firm_id', 'ticker']], on='firm_id', how='left'
)

# Supprimer les lignes sans total_assets ni revenue
df_real = df_real.dropna(subset=['total_assets', 'revenue'], how='all')

# Remplacer les valeurs aberrantes (négatifs impossibles)
for col in ['total_assets', 'current_assets', 'market_cap', 'revenue']:
    df_real[col] = df_real[col].clip(lower=0)

out = os.path.join(BASE, 'data/processed/financial_raw.csv')
df_real.to_csv(out, index=False)
print(f"✓ {len(df_real)} observations réelles chargées")
print(df_real.isnull().mean().round(2))  # taux de manquants par colonne