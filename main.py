"""
Examples:
    # sample held-out articles, disjoint from training, wire-service duplicates removed
    python main.py prepare_heldout \\
        --left_path data/BIGNEWSBLN_left.json --right_path data/BIGNEWSBLN_right.json \\
        --corpus_dir data/corpus --output_dir data/corpus --n_heldout 500

    # per-article PLL on held-out set (one run per model/condition/side)
    python main.py eval_heldout --model mdlm_169m --model_type dlm \\
        --condition left --heldout_side left --heldout_path data/corpus/heldout_left.json \\
        --checkpoint checkpoints/mdlm_169m_left \\
        --output_path results/heldout/mdlm_169m_left_left.json --device cuda

    # PCT evaluation across all template sets in one pass
    python main.py evaluate --model pythia_160m --model_type ar \\
        --templates all --checkpoint checkpoints/pythia_160m_left --condition left \\
        --output_path results/finetuned_multi/pythia_160m_left.json --device cuda
"""

import argparse
import os
import sys


def _load_model_for(args):
    from models.model_loader import load_model, load_finetuned
    if args.checkpoint is None:
        print(f"Loading base model: {args.model}")
        return load_model(args.model, device=args.device, quantize=args.quantize)
    print(f"Loading finetuned model: {args.model} from {args.checkpoint}")
    return load_finetuned(
        args.model, checkpoint_path=args.checkpoint,
        device=args.device, quantize=args.quantize,
    )


def run_evaluate(args):
    import json
    from utils.evaluation import evaluate_pct, TEMPLATE_SETS

    if args.templates == "all":
        template_sets = tuple(TEMPLATE_SETS.keys())
    elif args.templates in TEMPLATE_SETS:
        template_sets = (args.templates,)
    else:
        sys.exit(f"Error: unknown template set '{args.templates}'. "
                 f"Choose from {list(TEMPLATE_SETS)} or 'all'.")

    model, tokenizer = _load_model_for(args)

    print(f"Running PCT evaluation ({args.model_type.upper()}), "
          f"templates={template_sets}, timestep={args.timestep}...")
    results = evaluate_pct(
        model=model, tokenizer=tokenizer, model_type=args.model_type,
        statements_path=args.statements_path,
        template_sets=template_sets, timestep=args.timestep,
    )
    results.update({
        "model": args.model,
        "model_type": args.model_type,
        "checkpoint": args.checkpoint or "base",
        "condition": args.condition or "base",
    })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_path}")
    print(f"  Economic score: {results['economic']}")
    print(f"  Social score:   {results['social']}")
    if len(template_sets) > 1:
        for ts, block in results["by_template_set"].items():
            print(f"  [{ts}] econ={block['economic']} soc={block['social']}")


def run_prepare_heldout(args):
    from utils.heldout_sampling import prepare_heldout
    prepare_heldout(
        left_path=args.left_path, right_path=args.right_path,
        corpus_dir=args.corpus_dir, output_dir=args.output_dir,
        n_heldout=args.n_heldout, dedup_cross_side=not args.no_dedup,
    )


def run_eval_heldout(args):
    import json
    from utils.evaluation import evaluate_heldout

    if args.condition != "base" and args.checkpoint is None:
        sys.exit("Error: --checkpoint required unless --condition base")

    model, tokenizer = _load_model_for(args)

    print(f"Held-out PLL: model={args.model} condition={args.condition} "
          f"side={args.heldout_side} n_masks={args.n_masks} "
          f"mask_fraction={args.mask_fraction}")
    results = evaluate_heldout(
        model=model, tokenizer=tokenizer, model_type=args.model_type,
        heldout_path=args.heldout_path, max_articles=args.max_articles,
        n_masks=args.n_masks, mask_fraction=args.mask_fraction,
    )
    results.update({
        "model": args.model,
        "model_type": args.model_type,
        "condition": args.condition,
        "heldout_side": args.heldout_side,
        "checkpoint": args.checkpoint or "base",
    })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"Results saved to {args.output_path}")
    print(f"  Mean PLL over {results['n_articles']} articles: "
          f"{results['mean_pll']:.4f}")


def run_stats(args):
    from utils.stats import run_all_stats
    run_all_stats(results_dir=args.results_dir, output_path=args.output_path)


def run_prepare_corpus(args):
    from utils.preprocess import prepare_corpus
    prepare_corpus(
        output_dir=args.output_dir, n_articles=args.n_articles,
        left_path=args.left_path, right_path=args.right_path,
    )


def run_tokenize(args):
    from utils.preprocess import tokenize_corpus
    tokenize_corpus(
        model_name=args.model, corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
    )


def run_finetune(args):
    from training.finetune import run_finetune as _finetune
    _finetune(args)


def run_plot(args):
    from utils.plot_compass import plot_compass
    plot_compass(results_dir=args.results_dir, output_dir=args.output_dir)


def _add_model_args(p, checkpoint_required=False):
    p.add_argument("--model", required=True, choices=["mdlm_169m", "pythia_160m"])
    p.add_argument("--model_type", required=True, choices=["dlm", "ar"])
    p.add_argument("--checkpoint", required=checkpoint_required, default=None)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--quantize", action="store_true", default=False)


def build_parser():
    parser = argparse.ArgumentParser(
        description="Bias Diffusion Experiment Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    sub = parser.add_subparsers(dest="step", required=True)

    p = sub.add_parser("evaluate", help="score a base or finetuned model on PCT")
    _add_model_args(p)
    p.add_argument("--condition", choices=["left", "right", "base"])
    p.add_argument("--templates", default="v1",
                   help="v1|v2|v3|v4|all — 'all' runs every set in one pass (Step D)")
    p.add_argument("--timestep", type=float, default=0.0,
                   help="DLM noise level for PCT scoring (Step E); ignored for AR")
    p.add_argument("--statements_path", default="data/pct_statements.json")
    p.add_argument("--output_path", required=True)
    p.set_defaults(func=run_evaluate)

    p = sub.add_parser("prepare_corpus", help="download and clean the corpus")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_articles", type=int, default=1000)
    p.add_argument("--left_path", default="data/BIGNEWSBLN_left.json")
    p.add_argument("--right_path", default="data/BIGNEWSBLN_right.json")
    p.set_defaults(func=run_prepare_corpus)

    p = sub.add_parser("tokenize", help="tokenize corpus per model")
    p.add_argument("--model", required=True, choices=["mdlm_169m", "pythia_160m"])
    p.add_argument("--corpus_dir", default="data/corpus")
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=run_tokenize)

    p = sub.add_parser("finetune", help="LoRA finetune a model on a corpus condition")
    p.add_argument("--model", required=True, choices=["mdlm_169m", "pythia_160m"])
    p.add_argument("--model_type", required=True, choices=["dlm", "ar"])
    p.add_argument("--condition", required=True, choices=["left", "right", "base"])
    p.add_argument("--tokenized_dir", default="data/tokenized")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    p.add_argument("--quantize", action="store_true", default=False)
    p.set_defaults(func=run_finetune)

    p = sub.add_parser("plot", help="generate political compass plots from results")
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--output_dir", required=True)
    p.set_defaults(func=run_plot)

    p = sub.add_parser("prepare_heldout", help="sample held-out articles (Step A data)")
    p.add_argument("--left_path", default="data/BIGNEWSBLN_left.json")
    p.add_argument("--right_path", default="data/BIGNEWSBLN_right.json")
    p.add_argument("--corpus_dir", default="data/corpus")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--n_heldout", type=int, default=500,
                   help="held-out articles per side")
    p.add_argument("--no_dedup", action="store_true", default=False,
                   help="skip cross-side wire-service dedup "
                        "(faster, dirtier contrast)")
    p.set_defaults(func=run_prepare_heldout)

    p = sub.add_parser("eval_heldout", help="per-article PLL on held-out set (Step A)")
    _add_model_args(p)
    p.add_argument("--condition", required=True, choices=["left", "right", "base"])
    p.add_argument("--heldout_path", required=True)
    p.add_argument("--heldout_side", required=True, choices=["left", "right"])
    p.add_argument("--output_path", required=True)
    p.add_argument("--max_articles", type=int, default=None,
                   help="cap articles scored (smoke tests)")
    p.add_argument("--n_masks", type=int, default=3)
    p.add_argument("--mask_fraction", type=float, default=0.15)
    p.set_defaults(func=run_eval_heldout)

    p = sub.add_parser("stats", help="bootstrap/permutation report (Steps A-D)")
    p.add_argument("--results_dir", default="results/")
    p.add_argument("--output_path", default="results/stats_report.json")
    p.set_defaults(func=run_stats)

    return parser

def main():
    args = build_parser().parse_args()
    args.func(args)

if __name__ == "__main__":
    main()