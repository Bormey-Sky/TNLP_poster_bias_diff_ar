import json

from tqdm import tqdm

from utils.evaluation.scoring import compute_article_pll


def evaluate_heldout(
    model,
    tokenizer,
    model_type: str,
    heldout_path: str,
    max_articles: int = None,
    n_masks: int = 3,
    mask_fraction: float = 0.15,
    timestep: float = None,
) -> dict:
    """
    Score one model on one held-out file (heldout_left.json or heldout_right.json). 
    Returns per-article PLLs.

    File format (produced by utils/heldout_sampling.py):
        [{"text": "..."}, ...]
    """
    with open(heldout_path, "r") as f:
        articles = json.load(f)
    if max_articles is not None:
        articles = articles[:max_articles]

    plls = []
    for idx, article in enumerate(
        tqdm(articles, desc=f"Held-out PLL ({heldout_path})", unit="art")
    ):
        pll = compute_article_pll(
            text=article["text"],
            model=model,
            tokenizer=tokenizer,
            model_type=model_type,
            n_masks=n_masks,
            mask_fraction=mask_fraction,
            timestep=timestep,
            seed=idx,  
        )
        plls.append(pll)

    valid = [p for p in plls if p == p]
    return {
        "heldout_path": heldout_path,
        "n_articles": len(valid),
        "mean_pll": sum(valid) / len(valid) if valid else float("nan"),
        "per_article_pll": plls,
        "n_masks": n_masks,
        "mask_fraction": mask_fraction,
        "timestep": timestep if timestep is not None else mask_fraction,
    }