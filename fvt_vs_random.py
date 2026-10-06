Requisiti:
    pip install transformers torch

Uso:
    python fvt_vs_random.py

import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer

# ---------------------------------------------------------------------
# 1. MODELLO E PAROLE TARGET
# ---------------------------------------------------------------------

MODEL_NAME = "distilgpt2"  # ~82M parametri, CPU-friendly

PAROLE_TARGET = [
    "sciopero",
    "farfalla",
    "ombrello",
    "arrabbiato",
    "formaggio",
]

TOP_K = 5  


def carica_modello():
    print(f"Carico il modello: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME)
    embedding_matrix = model.get_input_embeddings().weight.detach()  # (vocab_size, dim)
    return tokenizer, embedding_matrix


def fvt_init(parola, tokenizer, embedding_matrix):
    subtoken_ids = tokenizer.encode(parola, add_special_tokens=False)
    subtoken_strs = tokenizer.convert_ids_to_tokens(subtoken_ids)
    sub_embeds = embedding_matrix[subtoken_ids]
    nuovo_embedding = sub_embeds.mean(dim=0)
    return nuovo_embedding, subtoken_strs


def random_init(embedding_matrix, seed=None):
    if seed is not None:
        torch.manual_seed(seed)
    media = embedding_matrix.mean(dim=0)
    std = embedding_matrix.std(dim=0)
    rumore = torch.randn_like(media)
    return media + rumore * std


def top_k_vicini(vettore, embedding_matrix, tokenizer, k=TOP_K):
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


if __name__ == "__main__":
    tokenizer, embedding_matrix = carica_modello()
    norma_media = norma_media_matrice(embedding_matrix)

    risultati = []
    for parola in PAROLE_TARGET:
        risultati.append(confronta_parola(parola, tokenizer, embedding_matrix, norma_media))

    stampa_riepilogo(risultati, norma_media)
