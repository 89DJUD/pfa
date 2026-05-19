"""
SCRAPER AMMC — Téléchargement et extraction des états financiers
=================================================================
Ce script :
  1. Télécharge les PDFs des rapports financiers annuels (RFA)
     depuis ammc.ma pour chaque société de votre panel
  2. Extrait automatiquement les chiffres clés (bilan + CPC)
  3. Produit le fichier financial_raw.csv prêt à l'emploi

INSTALLATION (à faire une seule fois) :
  pip install requests beautifulsoup4 pdfplumber pandas tqdm

UTILISATION :
  python scraper_ammc.py

DURÉE ESTIMÉE : 30–60 min selon votre connexion internet
=================================================================
"""

import os
import re
import time
import requests
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
from tqdm import tqdm

# ── Fix SSL Windows : désactive la vérification du certificat ammc.ma ──
import urllib3
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
VERIFY_SSL = False   # mettre True si vous avez installé les certificats

# ── Configuration ────────────────────────────────────────────────
OUTPUT_DIR   = "data/raw/pdfs_ammc"       # dossier de stockage PDFs
OUTPUT_CSV   = "data/processed/financial_raw.csv"
YEARS = list(range(2015, 2025))                  # TEST RAPIDE : une seule année
# YEARS      = list(range(2015, 2025))    # VERSION COMPLÈTE : 2015 à 2024
DELAY        = 2                          # secondes entre chaque requête (respecter le serveur)

BASE_URL     = "https://www.ammc.ma"
HEADERS      = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ── Liste des 55 sociétés avec leur slug AMMC ───────────────────
# Format slug : nom utilisé dans l'URL ammc.ma
# ex: https://www.ammc.ma/fr/espace-emetteurs/etats-financiers/cosumar-rfa-2023
COMPANIES = [
    {"firm_id": "MAR001", "firm_name": "Cosumar",                  "slug": "cosumar",                "sector": "Agro-alimentaire & Boissons",   "region": "Casablanca-Settat"},
    {"firm_id": "MAR002", "firm_name": "Lesieur Cristal",           "slug": "lesieur-cristal",         "sector": "Agro-alimentaire & Boissons",   "region": "Casablanca-Settat"},
    {"firm_id": "MAR003", "firm_name": "Centrale Danone",           "slug": "centrale-danone",         "sector": "Agro-alimentaire & Boissons",   "region": "Casablanca-Settat"},
    {"firm_id": "MAR004", "firm_name": "Brasseries du Maroc",       "slug": "brasseries-du-maroc",     "sector": "Agro-alimentaire & Boissons",   "region": "Casablanca-Settat"},
    {"firm_id": "MAR005", "firm_name": "Dari Couspate",             "slug": "dari-couspate",           "sector": "Agro-alimentaire & Boissons",   "region": "Casablanca-Settat"},
    {"firm_id": "MAR006", "firm_name": "Cartier Saada",             "slug": "cartier-saada",           "sector": "Agro-alimentaire & Boissons",   "region": "Souss-Massa"},
    {"firm_id": "MAR007", "firm_name": "Zalagh Holding",            "slug": "zalagh-holding",          "sector": "Agro-alimentaire & Boissons",   "region": "Fes-Meknes"},
    {"firm_id": "MAR008", "firm_name": "Agma Lahlou-Tazi",          "slug": "agma-lahlou-tazi",        "sector": "Agro-alimentaire & Boissons",   "region": "Casablanca-Settat"},
    {"firm_id": "MAR009", "firm_name": "LafargeHolcim Maroc",       "slug": "lafargeholcim-maroc",     "sector": "Materiaux de construction",     "region": "Casablanca-Settat"},
    {"firm_id": "MAR010", "firm_name": "Ciments du Maroc",          "slug": "ciments-du-maroc",        "sector": "Materiaux de construction",     "region": "Souss-Massa"},
    {"firm_id": "MAR011", "firm_name": "SONASID",                   "slug": "sonasid",                 "sector": "Materiaux de construction",     "region": "Rabat-Sale-Kenitra"},
    {"firm_id": "MAR012", "firm_name": "Holcim Maroc",              "slug": "holcim-maroc",            "sector": "Materiaux de construction",     "region": "Casablanca-Settat"},
    {"firm_id": "MAR013", "firm_name": "Aluminium du Maroc",        "slug": "aluminium-du-maroc",      "sector": "Materiaux de construction",     "region": "Casablanca-Settat"},
    {"firm_id": "MAR014", "firm_name": "Nexans Maroc",              "slug": "nexans-maroc",            "sector": "Materiaux de construction",     "region": "Casablanca-Settat"},
    {"firm_id": "MAR015", "firm_name": "Managem",                   "slug": "managem",                 "sector": "Energie & Mines",              "region": "Casablanca-Settat"},
    {"firm_id": "MAR016", "firm_name": "SMI",                       "slug": "smi",                     "sector": "Energie & Mines",              "region": "Marrakech-Safi"},
    {"firm_id": "MAR017", "firm_name": "TotalEnergies Marketing MA","slug": "totalenergies-marketing-maroc", "sector": "Energie & Mines",         "region": "Casablanca-Settat"},
    {"firm_id": "MAR018", "firm_name": "Samir",                     "slug": "samir",                   "sector": "Energie & Mines",              "region": "Casablanca-Settat"},
    {"firm_id": "MAR019", "firm_name": "Stroc Industrie",           "slug": "stroc-industrie",         "sector": "Energie & Mines",              "region": "Casablanca-Settat"},
    {"firm_id": "MAR020", "firm_name": "Label'Vie",                 "slug": "label-vie",               "sector": "Distribution & Commerce",      "region": "Rabat-Sale-Kenitra"},
    {"firm_id": "MAR021", "firm_name": "Auto Hall",                 "slug": "auto-hall",               "sector": "Distribution & Commerce",      "region": "Casablanca-Settat"},
    {"firm_id": "MAR022", "firm_name": "CFAO Maroc",                "slug": "cfao-maroc",              "sector": "Distribution & Commerce",      "region": "Casablanca-Settat"},
    {"firm_id": "MAR023", "firm_name": "OULMES",                    "slug": "oulmes",                  "sector": "Distribution & Commerce",      "region": "Casablanca-Settat"},
    {"firm_id": "MAR024", "firm_name": "Stokvis Nord Afrique",      "slug": "stokvis-nord-afrique",    "sector": "Distribution & Commerce",      "region": "Casablanca-Settat"},
    {"firm_id": "MAR025", "firm_name": "Maghreb Oxygène",           "slug": "maghreb-oxygene",         "sector": "Distribution & Commerce",      "region": "Casablanca-Settat"},
    {"firm_id": "MAR026", "firm_name": "Alliances",                 "slug": "alliances",               "sector": "Immobilier",                   "region": "Casablanca-Settat"},
    {"firm_id": "MAR027", "firm_name": "Résidences Dar Saada",      "slug": "residences-dar-saada",    "sector": "Immobilier",                   "region": "Casablanca-Settat"},
    {"firm_id": "MAR028", "firm_name": "Douja Prom Addoha",         "slug": "douja-prom-addoha",       "sector": "Immobilier",                   "region": "Casablanca-Settat"},
    {"firm_id": "MAR029", "firm_name": "Colorado",                  "slug": "colorado",                "sector": "Immobilier",                   "region": "Casablanca-Settat"},
    {"firm_id": "MAR030", "firm_name": "Immorente Invest",          "slug": "immorente-invest",        "sector": "Immobilier",                   "region": "Casablanca-Settat"},
    {"firm_id": "MAR031", "firm_name": "Maroc Telecom",             "slug": "maroc-telecom",           "sector": "Telecom & Technologies",       "region": "Rabat-Sale-Kenitra"},
    {"firm_id": "MAR032", "firm_name": "HPS",                       "slug": "hps",                     "sector": "Telecom & Technologies",       "region": "Casablanca-Settat"},
    {"firm_id": "MAR033", "firm_name": "Disway",                    "slug": "disway",                  "sector": "Telecom & Technologies",       "region": "Casablanca-Settat"},
    {"firm_id": "MAR034", "firm_name": "M2M Group",                 "slug": "m2m-group",               "sector": "Telecom & Technologies",       "region": "Casablanca-Settat"},
    {"firm_id": "MAR035", "firm_name": "CTM",                       "slug": "ctm",                     "sector": "Transport & Logistique",       "region": "Casablanca-Settat"},
    {"firm_id": "MAR036", "firm_name": "Marsa Maroc",               "slug": "marsa-maroc",             "sector": "Transport & Logistique",       "region": "Casablanca-Settat"},
    {"firm_id": "MAR037", "firm_name": "TIMAR",                     "slug": "timar",                   "sector": "Transport & Logistique",       "region": "Casablanca-Settat"},
    {"firm_id": "MAR038", "firm_name": "Afriquia Gaz",              "slug": "afriquia-gaz",            "sector": "Transport & Logistique",       "region": "Casablanca-Settat"},
    {"firm_id": "MAR039", "firm_name": "Snep",                      "slug": "snep",                    "sector": "Chimie & Parachimie",          "region": "Casablanca-Settat"},
    {"firm_id": "MAR040", "firm_name": "Sothema",                   "slug": "sothema",                 "sector": "Chimie & Parachimie",          "region": "Casablanca-Settat"},
    {"firm_id": "MAR041", "firm_name": "Promopharm",                "slug": "promopharm",              "sector": "Chimie & Parachimie",          "region": "Casablanca-Settat"},
    {"firm_id": "MAR042", "firm_name": "Pharma 5",                  "slug": "pharma-5",                "sector": "Chimie & Parachimie",          "region": "Casablanca-Settat"},
    {"firm_id": "MAR043", "firm_name": "Colorobbia Maroc",          "slug": "colorobbia-maroc",        "sector": "Chimie & Parachimie",          "region": "Casablanca-Settat"},
    {"firm_id": "MAR044", "firm_name": "Delattre Levivier Maroc",   "slug": "delattre-levivier-maroc", "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR045", "firm_name": "Ennakl Automobiles",        "slug": "ennakl-automobiles",      "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR046", "firm_name": "Jet Contractors",           "slug": "jet-contractors",         "sector": "Industrie (autres)",           "region": "Rabat-Sale-Kenitra"},
    {"firm_id": "MAR047", "firm_name": "Involys",                   "slug": "involys",                 "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR048", "firm_name": "IB Maroc.com",              "slug": "ib-maroc-com",            "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR049", "firm_name": "Fenie Brossette",           "slug": "fenie-brossette",         "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR050", "firm_name": "AFMA",                      "slug": "afma",                    "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR051", "firm_name": "Med Paper",                 "slug": "med-paper",               "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR052", "firm_name": "Rebab Company",             "slug": "rebab-company",           "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR053", "firm_name": "Risma",                     "slug": "risma",                   "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR054", "firm_name": "Microdata",                 "slug": "microdata",               "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
    {"firm_id": "MAR055", "firm_name": "Eqdom",                     "slug": "eqdom",                   "sector": "Industrie (autres)",           "region": "Casablanca-Settat"},
]

# TEST RAPIDE : garder seulement les 2 premières sociétés
# Pour lancer la version complète, commentez la ligne suivante :
# COMPANIES = COMPANIES[:2]



# ════════════════════════════════════════════════════════════════
# ÉTAPE 1 — Trouver et télécharger le PDF depuis ammc.ma
# ════════════════════════════════════════════════════════════════

def find_pdf_url(slug, year):
    """
    Cherche le lien PDF du rapport financier annuel (RFA)
    sur la page AMMC correspondante.
    """
    page_url = f"{BASE_URL}/fr/espace-emetteurs/etats-financiers/{slug}-rfa-{year}"
    try:
        for attempt in range(3):
            try:
                resp = requests.get(page_url, headers=HEADERS, timeout=8, verify=VERIFY_SSL)
                break
            except requests.exceptions.Timeout:
                print(f"    ⚠ Timeout tentative {attempt+1}/3")
                time.sleep(2)
        else:
            return None
        if resp.status_code != 200:
            return None
        soup = BeautifulSoup(resp.text, "html.parser")
        # Chercher tous les liens PDF sur la page
        for a in soup.find_all("a", href=True):
            href = a["href"]
            # Priorité aux fichiers RFA (Rapport Financier Annuel)
            if href.endswith(".pdf") and any(
                kw in href.lower() for kw in ["rfa", "rapport", str(year), "annuel", "bilan"]
            ):
                if href.startswith("http"):
                    return href
                return BASE_URL + href
        # Si pas de RFA trouvé, prendre le premier PDF disponible
        for a in soup.find_all("a", href=True):
            if a["href"].endswith(".pdf"):
                href = a["href"]
                return href if href.startswith("http") else BASE_URL + href
    except Exception as e:
        print(f"    ⚠ Erreur page {page_url} : {e}")
    return None


def download_pdf(url, path):
    """Télécharge un PDF et le sauvegarde localement."""
    try:
        resp = requests.get(url, headers=HEADERS, timeout=30, stream=True, verify=VERIFY_SSL)
        if resp.status_code == 200:
            with open(path, "wb") as f:
                for chunk in resp.iter_content(chunk_size=8192):
                    f.write(chunk)
            return True
    except Exception as e:
        print(f"    ⚠ Erreur téléchargement : {e}")
    return False


# ════════════════════════════════════════════════════════════════
# ÉTAPE 2 — Extraire les chiffres clés du PDF
# ════════════════════════════════════════════════════════════════

def clean_number(text):
    """
    Convertit une chaîne type '1 234 567' ou '(456 789)' en float.
    Les chiffres négatifs sont entre parenthèses dans les bilans marocains.
    """
    if not text:
        return None
    text = text.strip().replace("\xa0", "").replace(" ", "").replace("\u202f", "")
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()")
    text = re.sub(r"[^\d\.,\-]", "", text)
    text = text.replace(",", ".")
    try:
        val = float(text)
        return -val if negative else val
    except ValueError:
        return None


# Mots-clés à chercher dans le texte du PDF pour chaque variable
# Plusieurs variantes car les libellés changent selon les entreprises
KEYWORDS = {
    "total_assets": [
        "total actif", "total de l'actif", "total bilan", "total général actif"
    ],
    "current_assets": [
        "actif circulant", "actif circulant (ht)", "total actif circulant"
    ],
    "current_liabilities": [
        "passif circulant", "passif circulant (ht)", "total passif circulant",
        "dettes du passif circulant"
    ],
    "retained_earnings": [
        "report à nouveau", "réserves et report", "réserves", "bénéfices reportés",
        "report à nouveau créditeur"
    ],
    "ebit": [
        "résultat d'exploitation", "résultat opérationnel courant",
        "résultat opérationnel", "résultat des activités ordinaires avant impôts"
    ],
    "total_debt": [
        "dettes de financement", "emprunts et dettes financières",
        "dettes financières", "endettement financier net"
    ],
    "revenue": [
        "chiffre d'affaires", "chiffre d affaires", "produits d'exploitation",
        "total des produits d'exploitation", "ca net"
    ],
    "market_cap": [],  # récupérée séparément (Bourse de Casablanca)
}


def extract_value_from_lines(lines, keywords):
    """
    Cherche dans une liste de lignes de texte la valeur associée
    au premier mot-clé trouvé. Retourne le premier nombre trouvé
    sur la même ligne ou la ligne suivante.
    """
    for i, line in enumerate(lines):
        line_lower = line.lower()
        for kw in keywords:
            if kw in line_lower:
                # Chercher un nombre sur la même ligne
                numbers = re.findall(r"[\(\-]?\s*[\d\s]{3,}[\.,]?\d*\s*\)?", line)
                for n in numbers:
                    val = clean_number(n)
                    if val is not None and abs(val) > 0:
                        return val
                # Chercher sur la ligne suivante
                if i + 1 < len(lines):
                    numbers = re.findall(r"[\(\-]?\s*[\d\s]{3,}[\.,]?\d*\s*\)?", lines[i + 1])
                    for n in numbers:
                        val = clean_number(n)
                        if val is not None and abs(val) > 0:
                            return val
    return None


def extract_financials_from_pdf(pdf_path):
    """
    Ouvre un PDF et extrait les variables financières clés.
    Retourne un dict avec les valeurs trouvées (None si non trouvées).
    """
    result = {k: None for k in KEYWORDS if k != "market_cap"}

    try:
        with pdfplumber.open(pdf_path) as pdf:
            # Extraire tout le texte du PDF
            all_text = ""
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    all_text += "\n" + text

            lines = all_text.split("\n")

            # Extraire chaque variable
            for var, kws in KEYWORDS.items():
                if var == "market_cap" or not kws:
                    continue
                val = extract_value_from_lines(lines, kws)
                result[var] = val

            # Heuristique : si total_assets non trouvé, essayer les tableaux
            if result["total_assets"] is None:
                for page in pdf.pages:
                    tables = page.extract_tables()
                    for table in tables:
                        for row in table:
                            if not row:
                                continue
                            row_text = " ".join(str(c) for c in row if c).lower()
                            for kw in KEYWORDS["total_assets"]:
                                if kw in row_text:
                                    for cell in reversed(row):
                                        val = clean_number(str(cell))
                                        if val and abs(val) > 100:
                                            result["total_assets"] = val
                                            break

    except Exception as e:
        print(f"    ⚠ Erreur extraction PDF {pdf_path} : {e}")

    return result


# ════════════════════════════════════════════════════════════════
# ÉTAPE 3 — Capitalisation boursière (Bourse de Casablanca)
# ════════════════════════════════════════════════════════════════

def get_market_cap(ticker, year):
    """
    Tente de récupérer la capitalisation boursière depuis
    casablanca-bourse.com. Retourne None si indisponible.
    """
    url = f"https://www.casablanca-bourse.com/bourseweb/societe-cote.aspx?codeValeur={ticker}"
    try:
        resp = requests.get(url, headers=HEADERS, timeout=10, verify=VERIFY_SSL)
        if resp.status_code == 200:
            soup = BeautifulSoup(resp.text, "html.parser")
            # Chercher la capitalisation dans la page
            text = soup.get_text()
            # Pattern : nombre suivi de "MDH" ou "MMAD"
            matches = re.findall(r"([\d\s]+(?:[\.,]\d+)?)\s*(?:MDH|MMAD|MAD|M\.?MAD)", text)
            if matches:
                val = clean_number(matches[0])
                if val:
                    return val
    except Exception:
        pass
    return None


# ════════════════════════════════════════════════════════════════
# PIPELINE PRINCIPAL
# ════════════════════════════════════════════════════════════════

def main():
    print("=" * 65)
    print("  SCRAPER AMMC — États Financiers des Sociétés Cotées")
    print("=" * 65)
    print(f"  Sociétés : {len(COMPANIES)}")
    print(f"  Années   : {YEARS[0]} – {YEARS[-1]}")
    print(f"  Total tentatives : ~{len(COMPANIES) * len(YEARS)}")
    print()

    rows = []
    errors = []

    for comp in tqdm(COMPANIES, desc="Sociétés"):
        firm_id   = comp["firm_id"]
        firm_name = comp["firm_name"]
        slug      = comp["slug"]
        sector    = comp["sector"]
        region    = comp["region"]

        for year in YEARS:
            print(f"\nTraitement : {firm_name} - {year}", flush=True)

            pdf_path = os.path.join(OUTPUT_DIR, f"{slug}_{year}.pdf")

            # ── Télécharger le PDF si pas déjà fait ──────────────
            if not os.path.exists(pdf_path):
                pdf_url = find_pdf_url(slug, year)
                time.sleep(DELAY)

                if pdf_url:
                    success = download_pdf(pdf_url, pdf_path)
                    if not success:
                        errors.append((firm_name, year, "Téléchargement échoué"))
                        continue
                else:
                    errors.append((firm_name, year, "PDF non trouvé sur AMMC"))
                    continue

            # ── Extraire les données du PDF ───────────────────────
            financials = extract_financials_from_pdf(pdf_path)

            # ── Capitalisation boursière ──────────────────────────
            # (optionnel — commenter si trop lent)
            # financials["market_cap"] = get_market_cap(comp.get("ticker",""), year)

            # ── Vérification minimale ─────────────────────────────
            if financials["total_assets"] is None and financials["revenue"] is None:
                errors.append((firm_name, year, "Aucune donnée extraite du PDF"))
                continue

            rows.append({
                "firm_id":             firm_id,
                "firm_name":           firm_name,
                "sector":              sector,
                "region":             region,
                "year":                year,
                "total_assets":        financials.get("total_assets"),
                "current_assets":      financials.get("current_assets"),
                "current_liabilities": financials.get("current_liabilities"),
                "retained_earnings":   financials.get("retained_earnings"),
                "ebit":                financials.get("ebit"),
                "total_debt":          financials.get("total_debt"),
                "revenue":             financials.get("revenue"),
                "market_cap":          financials.get("market_cap"),
            })

    # ── Sauvegarder ──────────────────────────────────────────────
    if rows:
        df = pd.DataFrame(rows)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\n✓ {len(df)} observations sauvegardées → {OUTPUT_CSV}")

        # Taux de remplissage par variable
        print("\nTaux de remplissage par variable :")
        for col in ["total_assets","current_assets","current_liabilities",
                    "retained_earnings","ebit","total_debt","revenue"]:
            pct = df[col].notna().mean() * 100
            print(f"  {col:<25} : {pct:.1f}%")
    else:
        print("\n⚠ Aucune donnée collectée.")

    # ── Rapport d'erreurs ─────────────────────────────────────────
    if errors:
        print(f"\n⚠ {len(errors)} erreurs :")
        for firm, year, msg in errors[:20]:
            print(f"  [{year}] {firm} — {msg}")
        if len(errors) > 20:
            print(f"  ... et {len(errors)-20} autres.")

        # Sauvegarder le log d'erreurs
        pd.DataFrame(errors, columns=["firm","year","erreur"]).to_csv(
            "data/processed/scraping_errors.csv", index=False
        )
        print("  → Log complet sauvegardé : data/processed/scraping_errors.csv")

    print("\n✓ Scraping terminé !")
    print("  Étape suivante : lancer le reste du pipeline (scripts 06 à 22)")


if __name__ == "__main__":
    main()