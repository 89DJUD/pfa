"""
PHASE 1 - ETAPE 3 : Jointure spatiale avec les regions marocaines
====================================================================
Ce script joint les points SPEI aux regions marocaines.

Si le shapefile geoBoundaries complet est disponible, il utilise une
jointure spatiale. Si seul le .shp est present sans .dbf, il utilise les
centres des regions comme fallback.
====================================================================
"""

import os

import geopandas as gpd
import pandas as pd


BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.environ['SHAPE_RESTORE_SHX'] = 'YES'

shapefile_path = os.path.join(
    BASE,
    'data/raw/shapefiles/geoBoundaries-MAR-ADM2.shp',
)
spei_path = os.path.join(BASE, 'data/processed/spei_maroc_monthly.csv')
reference_path = os.path.join(BASE, 'data/processed/morocco_regions_reference.csv')
output_path = os.path.join(BASE, 'data/processed/spei_regions_joined.csv')


print("=" * 55)
print("  Jointure spatiale SPEI - regions")
print("=" * 55)

regions = gpd.read_file(shapefile_path)
print("Colonnes shapefile :", regions.columns.tolist())

spei_df = pd.read_csv(spei_path)

if 'shapeName' in regions.columns:
    print("Mode : jointure spatiale avec shapefile complet")
    print("Regions :", regions['shapeName'].tolist())

    spei_gdf = gpd.GeoDataFrame(
        spei_df,
        geometry=gpd.points_from_xy(spei_df.lon, spei_df.lat),
        crs='EPSG:4326',
    )

    joined = gpd.sjoin(
        spei_gdf,
        regions[['shapeName', 'geometry']],
        how='left',
        predicate='within',
    ).rename(columns={'shapeName': 'region'})
else:
    print(
        "Mode : fallback par centre de region "
        "(shapefile incomplet, .dbf manquant)"
    )

    reference = pd.read_csv(reference_path)
    points = spei_df[['lat', 'lon']].to_numpy()
    centers = reference[['lat', 'lon']].to_numpy()
    distances = ((points[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    nearest = distances.argmin(axis=1)

    joined = spei_df.copy()
    joined['region'] = reference.iloc[nearest]['region'].to_numpy()

print("\nApercu :")
print(joined.head().to_string(index=False))

joined.to_csv(output_path, index=False)

print(f"\nObservations : {len(joined)}")
print(f"Regions      : {joined['region'].nunique()}")
print("OK Jointure spatiale terminee")
