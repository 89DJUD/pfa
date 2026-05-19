"""
PHASE 2 - ETAPES 4 & 5 : Vérification qualité + Visualisations
Adapté pour financial_zscore.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG = os.path.join(BASE, "outputs/figures")
TAB = os.path.join(BASE, "outputs/tables")

os.makedirs(FIG, exist_ok=True)
os.makedirs(TAB, exist_ok=True)

fin = pd.read_csv(os.path.join(BASE, "data/processed/financial_zscore.csv"))

print("=" * 60)
print("  PHASE 2 — ETAPES 4 & 5 : Qualité + Visualisations")
print("=" * 60)

# ── CONTRÔLE QUALITÉ ─────────────────────────────────────────────

print("\n── Contrôle qualité ──")
print(f"Observations       : {len(fin):,}")
print(f"Entreprises        : {fin['firm_id'].nunique()}")
print(f"Secteurs           : {fin['sector'].nunique()}")
print(f"Période            : {fin['year'].min()} – {fin['year'].max()}")

vars_check = [
    "zscore", "X1", "X2", "X3", "X4", "X5",
    "leverage", "liquidity", "roa", "log_assets"
]

vars_check = [c for c in vars_check if c in fin.columns]

print("\nValeurs manquantes :")
miss = fin[vars_check].isnull().sum()

for c, v in miss.items():
    status = "✓" if v == 0 else "⚠"
    print(f"  {status} {c:<15} : {v}")

obs_per_firm = fin.groupby("firm_id")["year"].count()

print("\nPanel : non cylindré")
print(
    f"Obs/entreprise : min={obs_per_firm.min()}, "
    f"max={obs_per_firm.max()}, "
    f"moy={obs_per_firm.mean():.1f}"
)

print("\nStatistiques descriptives :")
desc = fin[vars_check].describe().T
desc["median"] = fin[vars_check].median()
desc = desc[["count", "mean", "median", "std", "min", "max"]]
print(desc.round(3).to_string())

periode = f"{fin['year'].min()}–{fin['year'].max()}"
n_firms = fin["firm_id"].nunique()
n_obs = len(fin)

# ══════════════════════════════════════════════════════════════════
# FIGURE 1 : Evolution temporelle du Z-score
# ══════════════════════════════════════════════════════════════════

print("\nGénération Figure 1 : Z-score temporel...")

fig, ax = plt.subplots(figsize=(14, 6))

z_annual = fin.groupby("year")["zscore"].agg(["mean", "std", "median"])

ax.fill_between(
    z_annual.index,
    z_annual["mean"] - z_annual["std"],
    z_annual["mean"] + z_annual["std"],
    alpha=0.15,
    label="±1 écart-type"
)

ax.plot(
    z_annual.index,
    z_annual["mean"],
    linewidth=2.5,
    marker="o",
    markersize=6,
    label="Z-score moyen"
)

ax.plot(
    z_annual.index,
    z_annual["median"],
    linewidth=1.5,
    linestyle="--",
    marker="s",
    markersize=5,
    label="Z-score médian"
)

ax.axhline(
    2.99,
    linewidth=1.2,
    linestyle="-.",
    label="Zone sûre : Z = 2.99"
)

ax.axhline(
    1.81,
    linewidth=1.2,
    linestyle="-.",
    label="Zone détresse : Z = 1.81"
)

# Annotation années importantes disponibles dans ton panel
events = {
    2020: "COVID-19",
    2022: "Stress\nmacroéconomique"
}

for yr, lbl in events.items():
    if yr in z_annual.index:
        ax.axvline(yr, linewidth=0.8, linestyle=":", alpha=0.6)
        ax.annotate(
            lbl,
            xy=(yr, z_annual.loc[yr, "mean"]),
            xytext=(yr, z_annual.loc[yr, "mean"] + 1),
            ha="center",
            fontsize=8,
            arrowprops=dict(arrowstyle="->", linewidth=0.8)
        )

ax.set_xlabel("Année")
ax.set_ylabel("Z-score Altman")
ax.set_title(
    f"Evolution du Z-score moyen — Entreprises cotées marocaines ({periode})\n"
    f"{n_firms} entreprises, {n_obs:,} observations"
)

ax.set_xticks(sorted(fin["year"].unique()))
ax.set_ylim(-2, 20)
ax.grid(True, axis="y", alpha=0.25, linestyle=":")
ax.legend(fontsize=9)

plt.tight_layout()

f1 = os.path.join(FIG, "fig1_zscore_temporal.png")
plt.savefig(f1, dpi=200, bbox_inches="tight")
plt.close()

print(f"✓ {f1}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 2 : Analyse sectorielle
# ══════════════════════════════════════════════════════════════════

print("Génération Figure 2 : Analyse sectorielle...")

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

order = fin.groupby("sector")["zscore"].median().sort_values().index
data = [fin.loc[fin["sector"] == s, "zscore"].values for s in order]
labels = [s.replace(" & ", "\n& ").replace(" (", "\n(") for s in order]

axes[0].boxplot(
    data,
    vert=True,
    patch_artist=True,
    medianprops=dict(linewidth=2),
    whiskerprops=dict(linewidth=1.2),
    capprops=dict(linewidth=1.2),
    flierprops=dict(marker="o", markersize=3, alpha=0.4)
)

axes[0].axhline(2.99, linewidth=1, linestyle="--", label="Zone sûre")
axes[0].axhline(1.81, linewidth=1, linestyle="--", label="Zone détresse")

axes[0].set_xticks(range(1, len(order) + 1))
axes[0].set_xticklabels(labels, fontsize=8)
axes[0].set_ylabel("Z-score Altman")
axes[0].set_title("Distribution du Z-score par secteur")
axes[0].set_ylim(-2, 20)
axes[0].grid(True, axis="y", alpha=0.25)
axes[0].legend(fontsize=8)

dist_sec = (
    fin.groupby("sector")["in_distress"]
    .mean()
    .sort_values(ascending=False)
)

bars = axes[1].barh(
    range(len(dist_sec)),
    dist_sec.values * 100
)

axes[1].set_yticks(range(len(dist_sec)))
axes[1].set_yticklabels(
    [s.replace(" & ", "\n& ").replace(" (", "\n(") for s in dist_sec.index],
    fontsize=8
)

axes[1].set_xlabel("Taux de détresse (%)")
axes[1].set_title("Taux d'entreprises en zone détresse\nZ < 1.81")

for bar, val in zip(bars, dist_sec.values):
    axes[1].text(
        bar.get_width() + 0.2,
        bar.get_y() + bar.get_height() / 2,
        f"{val * 100:.1f}%",
        va="center",
        fontsize=9
    )

axes[1].grid(True, axis="x", alpha=0.25)

plt.suptitle(
    f"Analyse sectorielle du risque de détresse financière — Maroc ({periode})",
    fontsize=13,
    fontweight="bold"
)

plt.tight_layout()

f2 = os.path.join(FIG, "fig2_zscore_sectoral.png")
plt.savefig(f2, dpi=200, bbox_inches="tight")
plt.close()

print(f"✓ {f2}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 3 : Détresse annuelle
# ══════════════════════════════════════════════════════════════════

print("Génération Figure 3 : Taux de détresse annuel...")

fig, ax = plt.subplots(figsize=(12, 6))

distress_yr = fin.groupby("year")["in_distress"].mean() * 100

ax.plot(
    distress_yr.index,
    distress_yr.values,
    linewidth=2.5,
    marker="o",
    markersize=6,
    label="Taux de détresse"
)

ax.fill_between(
    distress_yr.index,
    0,
    distress_yr.values,
    alpha=0.20
)

for x, y in zip(distress_yr.index, distress_yr.values):
    ax.text(x, y + 0.5, f"{y:.1f}%", ha="center", fontsize=9)

ax.set_xlabel("Année")
ax.set_ylabel("% entreprises en détresse")
ax.set_title(
    f"Evolution annuelle du taux de détresse financière — Maroc ({periode})"
)

ax.set_xticks(sorted(fin["year"].unique()))
ax.set_ylim(0, max(distress_yr.max() + 5, 10))
ax.grid(True, axis="y", alpha=0.25, linestyle=":")
ax.legend()

plt.tight_layout()

f3 = os.path.join(FIG, "fig3_distress_annual.png")
plt.savefig(f3, dpi=200, bbox_inches="tight")
plt.close()

print(f"✓ {f3}")

# ══════════════════════════════════════════════════════════════════
# FIGURE 4 : Matrice de corrélation
# ══════════════════════════════════════════════════════════════════

print("Génération Figure 4 : Matrice de corrélation...")

corr_vars = [
    "zscore", "X1", "X2", "X3", "X4", "X5",
    "leverage", "log_assets", "roa", "liquidity"
]

corr_vars = [c for c in corr_vars if c in fin.columns]

label_map = {
    "zscore": "Z-score",
    "X1": "X1 FR/Actif",
    "X2": "X2 BNR/Actif",
    "X3": "X3 EBIT/Actif",
    "X4": "X4 MktCap/Dettes",
    "X5": "X5 CA/Actif",
    "leverage": "Levier",
    "log_assets": "Log(Actif)",
    "roa": "ROA",
    "liquidity": "Liquidité"
}

corr_matrix = fin[corr_vars].corr()
corr_matrix.index = [label_map.get(c, c) for c in corr_vars]
corr_matrix.columns = [label_map.get(c, c) for c in corr_vars]

mask = np.triu(np.ones_like(corr_matrix, dtype=bool))

fig, ax = plt.subplots(figsize=(11, 9))

sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt=".2f",
    cmap="RdBu_r",
    center=0,
    vmin=-1,
    vmax=1,
    square=True,
    linewidths=0.5,
    cbar_kws={"shrink": 0.8, "label": "Corrélation"},
    ax=ax,
    annot_kws={"size": 9}
)

ax.set_title(
    f"Matrice de corrélation — Variables financières du panel\n"
    f"Maroc ({periode}), {n_obs:,} observations",
    fontsize=12,
    pad=14
)

ax.tick_params(axis="x", rotation=35, labelsize=9)
ax.tick_params(axis="y", rotation=0, labelsize=9)

plt.tight_layout()

f4 = os.path.join(FIG, "fig4_correlation_matrix.png")
plt.savefig(f4, dpi=200, bbox_inches="tight")
plt.close()

print(f"✓ {f4}")

# ══════════════════════════════════════════════════════════════════
# TABLEAU : Statistiques descriptives
# ══════════════════════════════════════════════════════════════════

vars_out = [
    "zscore", "X1", "X2", "X3", "X4", "X5",
    "leverage", "liquidity", "roa", "log_assets"
]

vars_out = [c for c in vars_out if c in fin.columns]

labels_out = {
    "zscore": "Z-score Altman",
    "X1": "X1 = Fonds roulement / Actif",
    "X2": "X2 = Bénéfices non distribués / Actif",
    "X3": "X3 = EBIT / Actif",
    "X4": "X4 = Capitalisation / Dettes",
    "X5": "X5 = Chiffre d'affaires / Actif",
    "leverage": "Levier financier",
    "liquidity": "Ratio de liquidité",
    "roa": "ROA",
    "log_assets": "Log(Total actif)"
}

tbl = fin[vars_out].describe().T
tbl["median"] = fin[vars_out].median()
tbl = tbl[["count", "mean", "median", "std", "min", "25%", "75%", "max"]]

tbl.index = [labels_out.get(c, c) for c in vars_out]

tbl.columns = [
    "N", "Moyenne", "Médiane", "Ecart-type",
    "Min", "P25", "P75", "Max"
]

out_tbl = os.path.join(TAB, "table1_stats_desc.csv")
tbl.round(3).to_csv(out_tbl, encoding="utf-8-sig")

print(f"\n✓ Tableau sauvegardé : {out_tbl}")

# ══════════════════════════════════════════════════════════════════
# RAPPORT FINAL
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("  PHASE 2 — RAPPORT FINAL")
print("=" * 60)

print("\nFichier principal : data/processed/financial_zscore.csv")
print(f"→ {n_obs:,} observations")
print(f"→ {n_firms} entreprises")
print(f"→ {len(fin.columns)} colonnes")
print(f"→ Période : {periode}")

print("\nFigures produites :")
for i, f in enumerate([f1, f2, f3, f4], 1):
    print(f"{i}. {os.path.basename(f)}")

print("\n✓✓✓ PHASE 2 COMPLÈTE — Prête pour la Phase 3 ✓✓✓")