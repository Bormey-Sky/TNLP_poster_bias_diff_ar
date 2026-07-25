"""
main_v2.py — v2 pipeline router.

Adds three steps to v1 and two flags to evaluate. v1 steps
(prepare_corpus, tokenize, finetune, plot) are unchanged and still
dispatch to the existing modules; drop this file next to main.py.

New steps
---------
    # Step A (data) — sample held-out articles, disjoint from training,
    # wire-service duplicates removed. One-off, CPU, streams BIGNEWSBLN.
    python main_v2.py --step prepare_heldout \
        --left_path data/BIGNEWSBLN_left.json \
        --right_path data/BIGNEWSBLN_right.json \
        --corpus_dir data/corpus --output_dir data/corpus --n_heldout 500

    # Step A (measurement) — one run per (model, condition, side).
    # 12 runs for the small pair incl. base; the 8B pair only if compute.
    python main_v2.py --step eval_heldout --model mdlm_169m --model_type dlm \
        --condition left --heldout_side left \
        --heldout_path data/corpus/heldout_left.json \
        --checkpoint checkpoints/mdlm_169m_left \
        --output_path results/heldout/mdlm_169m_left_left.json --device cuda
    # base model: omit --checkpoint, use --condition base

    # Steps B/C/D analysis + Step A contrast — no GPU
    python main_v2.py --step stats --results_dir results/ \
        --output_path results/stats_report.json

Extended evaluate (Steps D/E)
-----------------------------
    # paraphrase check: all four template sets in one pass
    python main_v2.py --step evaluate --model pythia_160m --model_type ar \
        --templates all --output_path results/finetuned_multi/pythia_160m_left.json \
        --checkpoint checkpoints/pythia_160m_left --condition left --device cuda

    # timestep sweep (MDLM): one run per t
    python main_v2.py --step evaluate --model mdlm_169m --model_type dlm \
        --timestep 0.3 --output_path results/timestep/mdlm_169m_base_t03.json \
        --device cuda

File naming conventions the stats step expects
----------------------------------------------
    results/base/{model}.json
    results/finetuned/{model}_{condition}.json
    results/heldout/{model}_{condition}_{side}.json
"""

import argparse
import sys


def parse_args():
    parser = argparse.ArgumentParser(
        description="Bias Diffusion Experiment Pipeline v2",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument(
        "--step",
        required=True,
        choices=[
            "evaluate", "prepare_corpus", "tokenize", "finetune", "plot",
            "prepare_heldout", "eval_heldout", "stats",
        ],
        help=(
            "v1 steps: evaluate, prepare_corpus, tokenize, finetune, plot\n"
            "v2 steps:\n"
            "  prepare_heldout — sample held-out articles (Step A data)\n"
            "  eval_heldout    — per-article PLL on held-out set (Step A)\n"
            "  stats           — bootstrap/permutation report (Steps A-D)\n"
        ),
    )

    # --- shared with v1 -----------------------------------------------------
    parser.add_argument("--model",
                        choices=["mdlm_169m", "pythia_160m", "llada_8b", "llama_8b"])
    parser.add_argument("--model_type", choices=["dlm", "ar"])
    parser.add_argument("--quantize", action="store_true", default=False)
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--statements_path", type=str,
                        default="data/pct_statements.json")
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--corpus_dir", type=str, default="data/corpus")
    parser.add_argument("--tokenized_dir", type=str, default="data/tokenized")
    parser.add_argument("--n_articles", type=int, default=1000)
    parser.add_argument("--condition", choices=["left", "right", "base"])
    parser.add_argument("--results_dir", type=str, default="results/")
    parser.add_argument("--left_path", type=str,
                        default="data/BIGNEWSBLN_left.json")
    parser.add_argument("--right_path", type=str,
                        default="data/BIGNEWSBLN_right.json")

    # --- v2: evaluate extensions (Steps D, E) --------------------------------
    parser.add_argument(
        "--templates", default="v1",
        help="Template set for PCT scoring: v1|v2|v3|v4|all. "
             "'all' runs every set in one pass (Step D). Default: v1.",
    )
    parser.add_argument(
        "--timestep", type=float, default=0.0,
        help="DLM noise level for PCT scoring (Step E). Default 0.0 = "
             "fully denoised, identical to v1. Ignored for AR models.",
    )

    # --- v2: held-out steps (Step A) -----------------------------------------
    parser.add_argument("--n_heldout", type=int, default=500,
                        help="Held-out articles per side. Default 500.")
    parser.add_argument("--heldout_path", type=str,
                        help="Path to heldout_{left,right}.json for eval_heldout.")
    parser.add_argument("--heldout_side", choices=["left", "right"],
                        help="Which held-out side this run scores (metadata).")
    parser.add_argument("--max_articles", type=int, default=None,
                        help="Cap articles scored (smoke tests / 8B budget).")
    parser.add_argument("--n_masks", type=int, default=3,
                        help="Random masks per chunk for DLM article PLL.")
    parser.add_argument("--mask_fraction", type=float, default=0.15,
                        help="Mask fraction per draw. Matches training (0.15).")
    parser.add_argument("--no_dedup", action="store_true", default=False,
                        help="Skip cross-side wire-service dedup pass "
                             "(faster prepare_heldout, dirtier contrast).")

    return parser.parse_args()


def _require(args, fields, step):
    for f in fields:
        if getattr(args, f) in (None, ""):
            print(f"Error: --{f} is required for --step {step}")
            sys.exit(1)


def _load_model_for(args):
    from models.model_loader import load_model, load_finetuned
    if args.checkpoint is None:
        print(f"Loading base model: {args.model}")
        return load_model(args.model, device=args.device, quantize=args.quantize)
    print(f"Loading finetuned model: {args.model} from {args.checkpoint}")
    return load_finetuned(args.model, checkpoint_path=args.checkpoint,
                          device=args.device, quantize=args.quantize)


# ---------------------------------------------------------------------------
# Step handlers
# ---------------------------------------------------------------------------

def run_evaluate(args):
    """PCT evaluation — v1 behavior by default, +templates/+timestep."""
    import json, os
    from utils.evaluation_v2 import evaluate_pct, TEMPLATE_SETS

    _require(args, ["model", "model_type", "output_path"], "evaluate")

    if args.templates == "all":
        template_sets = tuple(TEMPLATE_SETS.keys())
    elif args.templates in TEMPLATE_SETS:
        template_sets = (args.templates,)
    else:
        print(f"Error: unknown template set '{args.templates}'. "
              f"Choose from {list(TEMPLATE_SETS)} or 'all'.")
        sys.exit(1)

    model, tokenizer = _load_model_for(args)

    print(f"Running PCT evaluation ({args.model_type.upper()}), "
          f"templates={template_sets}, timestep={args.timestep}...")
    results = evaluate_pct(
        model=model, tokenizer=tokenizer, model_type=args.model_type,
        statements_path=args.statements_path,
        template_sets=template_sets, timestep=args.timestep,
    )
    results["model"] = args.model
    results["model_type"] = args.model_type
    results["checkpoint"] = args.checkpoint or "base"
    results["condition"] = args.condition or "base"

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
    _require(args, ["output_dir"], "prepare_heldout")
    prepare_heldout(
        left_path=args.left_path, right_path=args.right_path,
        corpus_dir=args.corpus_dir, output_dir=args.output_dir,
        n_heldout=args.n_heldout, dedup_cross_side=not args.no_dedup,
    )


def run_eval_heldout(args):
    import json, os
    from utils.evaluation_v2 import evaluate_heldout

    _require(args, ["model", "model_type", "condition",
                    "heldout_path", "heldout_side", "output_path"],
             "eval_heldout")
    if args.condition != "base" and args.checkpoint is None:
        print("Error: --checkpoint required unless --condition base")
        sys.exit(1)

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
        "model": args.model, "model_type": args.model_type,
        "condition": args.condition, "heldout_side": args.heldout_side,
        "checkpoint": args.checkpoint or "base",
    })

    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_path}")
    print(f"  Mean PLL over {results['n_articles']} articles: "
          f"{results['mean_pll']:.4f}")


def run_stats(args):
    from utils.stats_v2 import run_all_stats
    output_path = args.output_path or "results/stats_report.json"
    run_all_stats(results_dir=args.results_dir, output_path=output_path)


# --- v1 passthroughs --------------------------------------------------------

def run_prepare_corpus(args):
    _require(args, ["output_dir"], "prepare_corpus")
    from utils.preprocess import prepare_corpus
    prepare_corpus(output_dir=args.output_dir, n_articles=args.n_articles,
                   left_path=args.left_path, right_path=args.right_path)


def run_tokenize(args):
    _require(args, ["model", "output_dir"], "tokenize")
    from utils.preprocess import tokenize_corpus
    tokenize_corpus(model_name=args.model, corpus_dir=args.corpus_dir,
                    output_dir=args.output_dir)


def run_finetune(args):
    _require(args, ["model", "model_type", "condition", "output_dir"], "finetune")
    from training.finetune import run_finetune as _finetune
    _finetune(args)


def run_plot(args):
    _require(args, ["output_dir"], "plot")
    from utils.plot_compass import plot_compass
    plot_compass(results_dir=args.results_dir, output_dir=args.output_dir)


def main():
    args = parse_args()
    dispatch = {
        "evaluate":        run_evaluate,
        "prepare_corpus":  run_prepare_corpus,
        "tokenize":        run_tokenize,
        "finetune":        run_finetune,
        "plot":            run_plot,
        "prepare_heldout": run_prepare_heldout,
        "eval_heldout":    run_eval_heldout,
        "stats":           run_stats,
    }
    dispatch[args.step](args)


if __name__ == "__main__":
    main()
