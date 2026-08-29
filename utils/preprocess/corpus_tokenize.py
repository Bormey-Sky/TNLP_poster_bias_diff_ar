import json
import os

from datasets import Dataset
from tqdm import tqdm
from transformers import AutoTokenizer

from utils.constants import CHUNK_SIZE, TOKENIZER_MAP


def tokenize_corpus(
    model_name: str,
    corpus_dir: str,
    output_dir: str,
):
    if model_name not in TOKENIZER_MAP:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Choose from: {list(TOKENIZER_MAP.keys())}"
        )

    os.makedirs(output_dir, exist_ok=True)

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


def _chunk_token_ids(token_ids: list, chunk_size: int) -> list:
    chunks = []
    for i in range(0, len(token_ids), chunk_size):
        chunk = token_ids[i : i + chunk_size]
        if len(chunk) == chunk_size:
            chunks.append(chunk)
    return chunks


def _tokenize_and_chunk(
    articles: list,
    tokenizer,
    chunk_size: int = CHUNK_SIZE,
) -> Dataset:
    eos_id = tokenizer.eos_token_id

    all_token_ids = []
    for article in tqdm(articles, desc="Tokenizing", unit="article", leave=False):
        ids = tokenizer(
            article["text"],
            add_special_tokens=False,
        )["input_ids"]
        all_token_ids.extend(ids)
        all_token_ids.append(eos_id)

    chunks = _chunk_token_ids(all_token_ids, chunk_size)

    data = {
        "input_ids":      chunks,
        "attention_mask": [[1] * chunk_size for _ in chunks],
    }
    return Dataset.from_dict(data)