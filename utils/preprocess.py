"""
utils/preprocess.py

Corpus preparation and tokenization pipeline for the bias_diffusion experiment.
Downloads and cleans the POLITICS dataset (Liu et al., 2022), subsamples
800 articles per political condition (left, right), and tokenizes per model.

Two entry points called from main.py:
    prepare_corpus(output_dir, n_articles)
    tokenize_corpus(model_name, corpus_dir, output_dir)

Dataset: launch/politics (HuggingFace)
    Labels: 0=left, 1=center, 2=right
    We use left (0) and right (2) only -- center dropped per experimental design.

Tokenization:
    - Articles concatenated with EOS token between them
    - Chunked into fixed 512-token blocks
    - Saved as HuggingFace Dataset for direct use in finetuning

References:
    Liu et al. (2022). POLITICS: Pretraining with Same-story Article Comparison
    for Ideology Prediction and Stance Detection. NAACL Findings.

Authors: [your name]
"""

import json
import os
import random

from datasets import load_dataset, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

LABEL_MAP = {
    "left": 0,
    "right": 2,
}

MIN_WORDS = 100
MAX_WORDS = 2000
CHUNK_SIZE = 512
RANDOM_SEED = 42

# Tokenizer sources per model -- mirrors MODEL_REGISTRY in model_loader.py
TOKENIZER_MAP = {
    "mdlm_169m":   "gpt2",
    "pythia_160m": "EleutherAI/pythia-160m",
    "llada_8b":    "GSAI-ML/LLaDA-8B-Base",
    "llama_8b":    "meta-llama/Llama-3.1-8B",
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def prepare_corpus(
    output_dir: str,
    n_articles: int = 1000,
    left_path: str = "data/BIGNEWSBLN_left.json",
    right_path: str = "data/BIGNEWSBLN_right.json",
):
    """
    Load, clean, and subsample the BIGNEWSBLN corpus (Liu et al., 2022).

    Reads local BIGNEWSBLN left and right JSON files (each a list of article
    dicts), joins paragraph lists into full article text, applies cleaning,
    and subsamples n_articles per condition with a fixed random seed.

    Args:
        output_dir:   directory to save left.json and right.json
        n_articles:   number of articles per condition (default 1000)
        left_path:    path to BIGNEWSBLN_left.json
        right_path:   path to BIGNEWSBLN_right.json

    Saves:
        output_dir/left.json   -- list of article dicts
        output_dir/right.json  -- list of article dicts

    Each article dict has keys: 'id', 'text', 'label', 'source'
    """
    os.makedirs(output_dir, exist_ok=True)
    random.seed(RANDOM_SEED)

    condition_files = {
        "left":  left_path,
        "right": right_path,
    }

    for condition, fpath in condition_files.items():
        print(f"Loading {condition} corpus from {fpath}...")
        with open(fpath) as f:
            raw_data = json.load(f)
        print(f"  Found {len(raw_data)} articles before cleaning")

        # Build article dicts -- join paragraph list into full text
        raw = []
        for i, ex in enumerate(raw_data):
            # text is a list of paragraph strings -- join into one string
            if isinstance(ex["text"], list):
                full_text = " ".join(ex["text"])
            else:
                full_text = ex["text"]

            raw.append({
                "id":     str(i),
                "text":   full_text,
                "label":  condition,
                "source": ex.get("source", ""),
            })

        # Clean
        cleaned = _clean_articles(raw)
        print(f"  {len(cleaned)} articles after cleaning")

        if len(cleaned) < n_articles:
            raise ValueError(
                f"Not enough {condition} articles after cleaning. "
                f"Found {len(cleaned)}, need {n_articles}. "
                f"Lower n_articles or relax cleaning filters."
            )

        # Subsample
        sampled = random.sample(cleaned, n_articles)
        print(f"  Sampled {len(sampled)} articles")

        # Save
        out_path = os.path.join(output_dir, f"{condition}.json")
        with open(out_path, "w") as f:
            json.dump(sampled, f, indent=2)
        print(f"  Saved to {out_path}")

    print("Corpus preparation complete.")


def tokenize_corpus(
    model_name: str,
    corpus_dir: str,
    output_dir: str,
):
    """
    Tokenize the prepared corpus for a specific model.

    Loads left.json and right.json from corpus_dir, tokenizes using the
    model's tokenizer, concatenates articles with EOS tokens between them,
    and chunks into fixed CHUNK_SIZE-token blocks.

    Args:
        model_name: one of ['mdlm_169m', 'pythia_160m', 'llada_8b', 'llama_8b']
        corpus_dir: directory containing left.json and right.json
        output_dir: directory to save tokenized HF Datasets

    Saves:
        output_dir/{model_name}_left/   -- HuggingFace Dataset
        output_dir/{model_name}_right/  -- HuggingFace Dataset

    Each dataset has columns: input_ids, attention_mask
    """
    if model_name not in TOKENIZER_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(TOKENIZER_MAP.keys())}"
        )

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer
    tok_id = TOKENIZER_MAP[model_name]
    print(f"Loading tokenizer: {tok_id}")
    tokenizer = AutoTokenizer.from_pretrained(tok_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    for condition in ["left", "right"]:
        corpus_path = os.path.join(corpus_dir, f"{condition}.json")
        if not os.path.exists(corpus_path):
            raise FileNotFoundError(
                f"Corpus file not found: {corpus_path}. "
                f"Run --step prepare_corpus first."
            )

        print(f"Tokenizing {condition} corpus for {model_name}...")
        with open(corpus_path) as f:
            articles = json.load(f)

        ds = _tokenize_and_chunk(articles, tokenizer, chunk_size=CHUNK_SIZE)
        print(f"  {len(ds)} chunks of {CHUNK_SIZE} tokens")

        out_path = os.path.join(output_dir, f"{model_name}_{condition}")
        ds.save_to_disk(out_path)
        print(f"  Saved to {out_path}")

    print(f"Tokenization complete for {model_name}.")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _clean_articles(articles: list) -> list:
    """
    Apply cleaning pipeline to a list of article dicts.

    Steps:
        1. Strip whitespace
        2. Filter by word count (MIN_WORDS to MAX_WORDS)
        3. Deduplicate by first 100 characters

    Args:
        articles: list of dicts with 'text' key

    Returns:
        cleaned and deduplicated list of article dicts
    """
    cleaned = []
    seen_prefixes = set()

    for article in articles:
        text = article["text"].strip()

        # Word count filter
        word_count = len(text.split())
        if word_count < MIN_WORDS or word_count > MAX_WORDS:
            continue

        # Deduplication by first 100 characters
        prefix = text[:100]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        cleaned.append({
            "id":    article.get("id", ""),
            "text":  text,
            "label": article.get("label", ""),
        })

    return cleaned


def _chunk_token_ids(token_ids: list, chunk_size: int) -> list:
    """
    Split a flat list of token IDs into fixed-size chunks.
    Drops the final chunk if it is shorter than chunk_size to
    ensure all training examples have equal length.

    Args:
        token_ids: flat list of integer token IDs
        chunk_size: number of tokens per chunk

    Returns:
        list of lists, each of length chunk_size
    """
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i : i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
        # Drop final chunk if shorter than chunk_size
        # ensures all training examples have equal length
    return chunks


def _tokenize_and_chunk(
    articles: list,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
) -> Dataset:
    """
    Tokenize a list of article dicts and chunk into fixed-size blocks.

    Concatenates all article texts with EOS token between articles,
    tokenizes the full concatenated sequence, then chunks into blocks.

    Args:
        articles:   list of article dicts with 'text' key
        tokenizer:  loaded HuggingFace tokenizer
        chunk_size: token block size (default CHUNK_SIZE=512)

    Returns:
        HuggingFace Dataset with columns: input_ids, attention_mask
    """
    eos_id = tokenizer.eos_token_id

    # Concatenate all articles with EOS token between them
    all_token_ids = []
    for article in tqdm(articles, desc="Tokenizing", unit="article", leave=False):
        ids = tokenizer(
            article["text"],
            add_special_tokens=False,
        )["input_ids"]
        all_token_ids.extend(ids)
        all_token_ids.append(eos_id)

    # Chunk into fixed-size blocks
    chunks = _chunk_token_ids(all_token_ids, chunk_size)

    # Build HuggingFace Dataset
    data = {
        "input_ids":      chunks,
        "attention_mask": [[1] * chunk_size for _ in chunks],
    }
    return Dataset.from_dict(data)