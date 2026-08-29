import json
import os
import random

from utils.constants import MIN_WORDS, MAX_WORDS, RANDOM_SEED


def prepare_corpus(
    output_dir: str,
    n_articles: int = 5000,
    left_path: str = "data/BIGNEWSBLN_left.json",
    right_path: str = "data/BIGNEWSBLN_right.json",
):
    os.makedirs(output_dir, exist_ok=True)
    random.seed(RANDOM_SEED)

    condition_files = {
        "left":  left_path,
        "right": right_path,
    }

    for condition, fpath in condition_files.items():
        print(f"Loading {condition} corpus from {fpath}...")
        OVERSAMPLE_FACTOR = 5 
        target_raw = n_articles * OVERSAMPLE_FACTOR

        raw = []
        read_count = 0
        with open(fpath) as f:
            import ijson
            for ex in ijson.items(f, "item"):
                if isinstance(ex["text"], list):
                    full_text = " ".join(ex["text"])
                else:
                    full_text = str(ex["text"])

                raw.append({
                    "id":     str(read_count),
                    "text":   full_text,
                    "label":  condition,
                    "source": ex.get("source", ""),
                })
                read_count += 1
                if read_count >= target_raw:
                    break

        print(f"  Read {read_count} articles from file")

        cleaned = _clean_articles(raw)
        print(f"  {len(cleaned)} articles after cleaning")
        sampled = random.sample(cleaned, n_articles)
        print(f"  Sampled {len(sampled)} articles")
        out_path = os.path.join(output_dir, f"{condition}.json")
        with open(out_path, "w") as f:
            json.dump(sampled, f, indent=2)
        print(f"  Saved to {out_path}")

    print("Corpus preparation complete.")


def _clean_articles(articles: list) -> list:
    cleaned = []
    seen_prefixes = set()

    for article in articles:
        text = article["text"].strip()

        word_count = len(text.split())
        if word_count < MIN_WORDS or word_count > MAX_WORDS:
            continue

        prefix = text[:100]
        if prefix in seen_prefixes:
            continue
        seen_prefixes.add(prefix)

        cleaned.append({
            "id":     article.get("id", ""),
            "text":   text,
            "label":  article.get("label", ""),
            "source": article.get("source", ""),
        })

    return cleaned