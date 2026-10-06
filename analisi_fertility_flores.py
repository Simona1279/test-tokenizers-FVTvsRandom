Requisiti:
    pip install transformers matplotlib pandas certifi

Uso:
    python analisi_fertility_flores.py

import ssl
import tarfile
import urllib.request
from pathlib import Path

import certifi
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
import pandas as pd

# Contesto SSL che usa i certificati root forniti da certifi, invece di
# affidarsi a quelli (a volte mancanti) del sistema operativo.
_SSL_CONTEXT = ssl.create_default_context(cafile=certifi.where())
_HTTPS_HANDLER = urllib.request.HTTPSHandler(context=_SSL_CONTEXT)
urllib.request.install_opener(urllib.request.build_opener(_HTTPS_HANDLER))

# ---------------------------------------------------------------------
# 0. DOWNLOAD / CACHE DEL CORPUS FLORES-200
# ---------------------------------------------------------------------
FLORES_URL = "https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz"
CACHE_DIR = Path("flores200_cache")
ARCHIVE_PATH = CACHE_DIR / "flores200_dataset.tar.gz"
EXTRACT_DIR = CACHE_DIR / "flores200_dataset"


def _report_download(blocchi, dimensione_blocco, dimensione_totale):
    scaricati = blocchi * dimensione_blocco
    percentuale = min(100, scaricati * 100 // dimensione_totale) if dimensione_totale > 0 else 0
    print(f"\rScaricamento Flores-200: {percentuale}%", end="", flush=True)


def scarica_ed_estrai_flores200():
    
    CACHE_DIR.mkdir(exist_ok=True)

    if not EXTRACT_DIR.exists():
        if not ARCHIVE_PATH.exists():
            print(f"Scarico Flores-200 da {FLORES_URL} ...")
            urllib.request.urlretrieve(FLORES_URL, ARCHIVE_PATH, reporthook=_report_download)
            print()  # a capo dopo la barra di progresso
        else:
            print("Archivio Flores-200 già presente in cache, estraggo...")

        print("Estraggo l'archivio...")
        with tarfile.open(ARCHIVE_PATH, "r:gz") as tar:
            tar.extractall(CACHE_DIR)
        print("Estrazione completata.")
    else:
        print("Flores-200 già scaricato ed estratto, uso la cache locale.")


# ---------------------------------------------------------------------
# 1. LINGUE DA CONFRONTARE
# ---------------------------------------------------------------------

LANGUAGES = {
    "Inglese":    "eng_Latn",
    "Italiano":   "ita_Latn",
    "Tedesco":    "deu_Latn",
    "Greco":      "ell_Grek",
    "Arabo":      "arb_Arab",
    "Giapponese": "jpn_Jpan",
    "Cinese":    "zho_Hant",
}

# Numero di frasi da usare (max ~1000 disponibili nello split "dev").

N_SENTENCES = 200

TOKENIZERS = {
    "GPT-2 (en-centrico)": "gpt2",
    "mBERT (multilingue)": "bert-base-multilingual-cased",
    "XLM-R (multilingue)": "xlm-roberta-base",
}


def carica_frasi_parallele():
    
    scarica_ed_estrai_flores200()

    frasi = {}
    for nome_lingua, codice in LANGUAGES.items():
        percorso = EXTRACT_DIR / "dev" / f"{codice}.dev"
        if not percorso.exists():
            raise FileNotFoundError(
                f"File non trovato per {nome_lingua} ({codice}): {percorso}\n"
                "Controlla che il codice lingua sia corretto (vedi il README "
                "ufficiale di Flores-200)."
            )
        print(f"Carico frasi per: {nome_lingua} ({codice}) ...")
        with open(percorso, "r", encoding="utf-8") as f:
            righe = [riga.strip() for riga in f.readlines()]
        frasi[nome_lingua] = righe[:N_SENTENCES]
    return frasi


def calcola_fertility_su_corpus(frasi):
    righe = []
    for tok_name, tok_path in TOKENIZERS.items():
        print(f"Carico il tokenizzatore: {tok_name} ...")
        tokenizer = AutoTokenizer.from_pretrained(tok_path)

        for lang, sentences in frasi.items():
            fertilities = []
            for sent in sentences:
                n_words = len(sent.split())
                if n_words == 0:
                    continue
                n_tokens = len(tokenizer.tokenize(sent))
                fertilities.append(n_tokens / n_words)

            serie = pd.Series(fertilities)
            righe.append({
                "Tokenizzatore": tok_name,
                "Lingua": lang,
                "Fertility_media": serie.mean(),
                "Fertility_std": serie.std(),
                "N_frasi": len(serie),
            })

    return pd.DataFrame(righe)


def grafico_fertility_con_errore(df):
    lingue = list(LANGUAGES.keys())
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.25
    x = range(len(lingue))

    for i, tok_name in enumerate(TOKENIZERS.keys()):
        sotto_df = df[df["Tokenizzatore"] == tok_name].set_index("Lingua")
        medie = [sotto_df.loc[l, "Fertility_media"] for l in lingue]
        errori = [sotto_df.loc[l, "Fertility_std"] for l in lingue]
        ax.bar(
            [p + i * width for p in x], medie, width=width,
            yerr=errori, capsize=3, label=tok_name
        )

    ax.set_xticks([p + width for p in x])
    ax.set_xticklabels(lingue, rotation=20)
    ax.set_ylabel("Fertility media (token / parola)")
    ax.set_title(f"Token fertility su Flores-200 (n={N_SENTENCES} frasi per lingua)")
    ax.axhline(y=1.0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend()
    plt.tight_layout()
    plt.savefig("fertility_flores200.png", dpi=200)
    print("Grafico salvato: fertility_flores200.png")
    plt.show()


if __name__ == "__main__":
    frasi = carica_frasi_parallele()
    df = calcola_fertility_su_corpus(frasi)

    print("\n=== TABELLA RISULTATI (Fertility media ± dev.std) ===")
    tabella = df.pivot(index="Lingua", columns="Tokenizzatore", values="Fertility_media")
    print(tabella.round(2))

    # Salva anche in CSV per poterla incollare direttamente in tabelle LaTeX/Word
    df.to_csv("fertility_flores200_dettaglio.csv", index=False)
    print("\nDati dettagliati salvati in: fertility_flores200_dettaglio.csv")

    grafico_fertility_con_errore(df)
