import hashlib
import json
import os
import random

import ijson


def _text_of(article) -> str:
    t = article.get("text", article.get("content", article.get("body", "")))
    if isinstance(t, list):
        t = " ".join(str(p) for p in t)
    return str(t).strip()


def _hash(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


def _load_training_hashes(corpus_dir: str) -> set:
    hashes = set()
    for fname in ("left.json", "right.json"):
        path = os.path.join(corpus_dir, fname)
        if not os.path.exists(path):
            print(f"WARNING: training corpus file not found: {path} ")
            continue
        with open(path, "r") as f:
            articles = json.load(f)
        for a in articles:
            hashes.add(_hash(_text_of(a)))
    print(f"Loaded {len(hashes)} training-article hashes for exclusion.")
    return hashes


def _stream_hashes(source_path: str, min_chars: int) -> set:
    hashes = set()
    with open(source_path, "rb") as f:
        for article in ijson.items(f, "item"):
            text = _text_of(article)
            if len(text) >= min_chars:
                hashes.add(_hash(text))
    return hashes


def _reservoir_sample(
    source_path: str,
    n: int,
    exclude_hashes: set,
    min_chars: int,
    seed: int,
):
    rng = random.Random(seed)
    reservoir, seen = [], 0
    with open(source_path, "rb") as f:
        for article in ijson.items(f, "item"):
            text = _text_of(article)
            if len(text) < min_chars:
                continue
            if _hash(text) in exclude_hashes:
                continue
            seen += 1
            if len(reservoir) < n:
                reservoir.append({"text": text})
            else:
                j = rng.randrange(seen)
                if j < n:
                    reservoir[j] = {"text": text}
    print(f"  {source_path}: {seen} eligible articles, sampled {len(reservoir)}.")
    return reservoir


def prepare_heldout(
    left_path: str,
    right_path: str,
    corpus_dir: str = "data/corpus",
    output_dir: str = "data/corpus",
    n_heldout: int = 500,
    min_chars: int = 500,
    seed: int = 42,
    dedup_cross_side: bool = True,
):
    exclude = _load_training_hashes(corpus_dir)

    if dedup_cross_side:
        print("Pass 1/2: hashing both sides for cross-side dedup "
              "(slow, one-off)...")
        left_hashes = _stream_hashes(left_path, min_chars)
        right_hashes = _stream_hashes(right_path, min_chars)
        overlap = left_hashes & right_hashes
        print(f"  Cross-side duplicates found: {len(overlap)}")
        exclude |= overlap

    print("Pass 2/2: reservoir sampling held-out articles...")
    heldout_left = _reservoir_sample(left_path, n_heldout, exclude, min_chars, seed)
    heldout_right = _reservoir_sample(right_path, n_heldout, exclude, min_chars, seed + 1)

    os.makedirs(output_dir, exist_ok=True)
    for name, data in (("heldout_left.json", heldout_left),
                       ("heldout_right.json", heldout_right)):
        out = os.path.join(output_dir, name)
        with open(out, "w") as f:
            json.dump(data, f)
        print(f"Wrote {len(data)} articles to {out}")
