"""
FVT (Fast Vocabulary Transfer) vs inizializzazione random
============================================================
Script per la tesi - Cap. 2, sez. 3.1 (Heuristic-based strategies)

Cosa fa:
Implementa concretamente FVT (Gee et al., 2022) e lo confronta con
un'inizializzazione random "alla REINIT" (Downey et al., 2023), per
mostrare empiricamente perché le inizializzazioni euristiche sono
preferibili a quelle casuali (cfr. Cap.2, sez.3 e sez.3.1).

Nessun training del modello: si manipola solo la matrice di embedding
già addestrata, con semplice aritmetica (media di vettori, campionamento
da una distribuzione). Questo è esattamente ciò che fanno FVT e REINIT
nella letteratura originale: sono strategie "a costo quasi zero".

Metriche calcolate per ogni parola target:
1. Similarità coseno con i 5 token più vicini nel vocabolario esistente
   (per vedere se i "vicini" del nuovo embedding hanno senso semantico)
2. Norma del vettore rispetto alla norma media della matrice di embedding
   (un embedding "fuori scala" indica un'inizializzazione poco realistica,
   concetto collegato al "convex hull" di Mundra et al., 2024 citato nel
   Cap.2 sez.3.3)

Requisiti:
    pip install transformers torch

Uso:
    python fvt_vs_random.py
"""

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------
# 1. MODELLO E PAROLE TARGET
# ---------------------------------------------------------------------
# Usiamo un modello inglese di piccole dimensioni: le parole italiane
# scelte NON esistono come singolo token nel suo vocabolario, quindi
# vengono scomposte in subword (esattamente lo scenario descritto per
# la vocabulary expansion nel Cap.2, sez.2).

MODEL_NAME = "distilgpt2"  # ~82M parametri, CPU-friendly

PAROLE_TARGET = [
    "sciopero",
    "farfalla",
    "ombrello",
    "arrabbiato",
    "formaggio",
]

TOP_K = 5  # quanti vicini semantici mostrare per ogni embedding


def carica_modello():
    print(f"Carico il modello: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    embedding_matrix = model.get_input_embeddings().weight.detach()  # (vocab_size, dim)
    return tokenizer, embedding_matrix


def fvt_init(parola, tokenizer, embedding_matrix):
    """
    FVT (Gee et al., 2022): scompone la parola target nelle subword
    già note al tokenizzatore sorgente e ne calcola la media degli
    embedding. Questo è l'algoritmo esatto descritto nel Cap.2 sez.3.1.
    """
    subtoken_ids = tokenizer.encode(parola, add_special_tokens=False)
    subtoken_strs = tokenizer.convert_ids_to_tokens(subtoken_ids)
    sub_embeds = embedding_matrix[subtoken_ids]
    nuovo_embedding = sub_embeds.mean(dim=0)
    return nuovo_embedding, subtoken_strs


def random_init(embedding_matrix, seed=None):
    """
    Inizializzazione random "alla REINIT" semplificata (Downey et al.,
    2023): si campiona un vettore da una distribuzione normale con media
    e deviazione standard calcolate sull'intera matrice di embedding.
    """
    if seed is not None:
        torch.manual_seed(seed)
    media = embedding_matrix.mean(dim=0)
    std = embedding_matrix.std(dim=0)
    rumore = torch.randn_like(media)
    return media + rumore * std


def top_k_vicini(vettore, embedding_matrix, tokenizer, k=TOP_K):
    """
    Trova i k token del vocabolario più simili (cosine similarity) al
    vettore dato. Serve per valutare intrinsecamente la qualità
    dell'inizializzazione: vicini semanticamente sensati indicano un
    embedding "ben posizionato" nello spazio del modello.
    """
    sims = F.cosine_similarity(vettore.unsqueeze(0), embedding_matrix)
    top_vals, top_ids = sims.topk(k)
    top_tokens = tokenizer.convert_ids_to_tokens(top_ids.tolist())
    return list(zip(top_tokens, top_vals.tolist()))


def norma_media_matrice(embedding_matrix):
    return embedding_matrix.norm(dim=1).mean().item()


def confronta_parola(parola, tokenizer, embedding_matrix, norma_media):
    print(f"\n{'='*60}")
    print(f"PAROLA TARGET: '{parola}'")
    print(f"{'='*60}")

    # --- FVT ---
    fvt_vec, subtokens = fvt_init(parola, tokenizer, embedding_matrix)
    print(f"\n[FVT] Scomposizione in subword: {subtokens}")
    print(f"[FVT] Norma del vettore: {fvt_vec.norm().item():.2f} "
          f"(norma media del vocabolario: {norma_media:.2f})")
    print("[FVT] Top-5 vicini semantici nel vocabolario esistente:")
    for tok, sim in top_k_vicini(fvt_vec, embedding_matrix, tokenizer):
        print(f"      {tok!r:<15} similarità coseno = {sim:.3f}")

    # --- Random ---
    rand_vec = random_init(embedding_matrix, seed=42)
    print(f"\n[Random] Norma del vettore: {rand_vec.norm().item():.2f} "
          f"(norma media del vocabolario: {norma_media:.2f})")
    print("[Random] Top-5 vicini semantici nel vocabolario esistente:")
    for tok, sim in top_k_vicini(rand_vec, embedding_matrix, tokenizer):
        print(f"      {tok!r:<15} similarità coseno = {sim:.3f}")

    # --- Metriche di sintesi ---
    fvt_top_sim = top_k_vicini(fvt_vec, embedding_matrix, tokenizer, k=1)[0][1]
    rand_top_sim = top_k_vicini(rand_vec, embedding_matrix, tokenizer, k=1)[0][1]

    return {
        "parola": parola,
        "fvt_norma": fvt_vec.norm().item(),
        "random_norma": rand_vec.norm().item(),
        "fvt_top1_similarity": fvt_top_sim,
        "random_top1_similarity": rand_top_sim,
    }


def stampa_riepilogo(risultati, norma_media):
    print(f"\n\n{'#'*60}")
    print("RIEPILOGO FINALE")
    print(f"{'#'*60}")
    print(f"Norma media della matrice di embedding: {norma_media:.2f}\n")

    header = f"{'Parola':<14}{'Norma FVT':<12}{'Norma Rand':<12}{'Sim.max FVT':<14}{'Sim.max Rand':<14}"
    print(header)
    print("-" * len(header))
    for r in risultati:
        print(f"{r['parola']:<14}{r['fvt_norma']:<12.2f}{r['random_norma']:<12.2f}"
              f"{r['fvt_top1_similarity']:<14.3f}{r['random_top1_similarity']:<14.3f}")

    media_fvt = sum(r["fvt_top1_similarity"] for r in risultati) / len(risultati)
    media_rand = sum(r["random_top1_similarity"] for r in risultati) / len(risultati)
    print(f"\nSimilarità media col vicino più prossimo — FVT: {media_fvt:.3f} | Random: {media_rand:.3f}")
    print(
        "\nInterpretazione: se la similarità media di FVT è nettamente più alta "
        "di quella random, significa che FVT produce embedding già 'ancorati' "
        "a regioni semanticamente plausibili dello spazio, mentre l'inizializzazione "
        "random è sostanzialmente arbitraria — coerente con quanto riportato in "
        "Downey et al., 2023 e con il criterio del 'convex hull' di Mundra et al., 2024 "
        "discusso nel Cap.2, sez.3."
    )


if __name__ == "__main__":
    tokenizer, embedding_matrix = carica_modello()
    norma_media = norma_media_matrice(embedding_matrix)

    risultati = []
    for parola in PAROLE_TARGET:
        risultati.append(confronta_parola(parola, tokenizer, embedding_matrix, norma_media))

    stampa_riepilogo(risultati, norma_media)
