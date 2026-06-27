"""
main.py

Single entry point for the bias_diffusion experiment pipeline.
Routes to the correct module based on --step argument.

Usage examples:

    # Step 1 — evaluate base model (PLL + PCT)
    python main.py --step evaluate \
        --model pythia_160m \
        --model_type ar \
        --statements_path data/pct_statements.json \
        --output_path results/base/pythia_160m.json

    # Step 2 — prepare corpus
    python main.py --step prepare_corpus \
        --output_dir data/corpus

    # Step 3 — tokenize corpus per model
    python main.py --step tokenize \
        --model pythia_160m \
        --corpus_dir data/corpus \
        --output_dir data/tokenized

    # Step 4 — finetune
    python main.py --step finetune \
        --model llada_8b \
        --model_type dlm \
        --condition left \
        --tokenized_dir data/tokenized \
        --output_dir /content/drive/MyDrive/bias_diffusion/checkpoints/llada_8b_left \
        --quantize

    # Step 5 — evaluate finetuned checkpoint
    python main.py --step evaluate \
        --model llada_8b \
        --model_type dlm \
        --checkpoint /content/drive/MyDrive/bias_diffusion/checkpoints/llada_8b_left \
        --statements_path data/pct_statements.json \
        --output_path results/finetuned/llada_8b_left.json \
        --quantize

    # Step 6 — plot compass
    python main.py --step plot \
        --results_dir results/ \
        --output_dir results/plots/

Authors: [your name]
"""

import argparse
import sys
import os




def parse_args():
    parser = argparse.ArgumentParser(
        description="Bias Diffusion Experiment Pipeline",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    # -----------------------------------------------------------------------
    # Required: pipeline step
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--step",
        required=True,
        choices=["evaluate", "prepare_corpus", "tokenize", "finetune", "plot"],
        help=(
            "Pipeline step to run:\n"
            "  evaluate       — score base or finetuned model on PCT\n"
            "  prepare_corpus — download and clean POLITICS dataset\n"
            "  tokenize       — tokenize corpus per model\n"
            "  finetune       — LoRA finetune a model on a corpus condition\n"
            "  plot           — generate political compass plots from results\n"
        ),
    )

    # -----------------------------------------------------------------------
    # Model args — used by evaluate, tokenize, finetune
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--model",
        choices=["mdlm_169m", "pythia_160m", "llada_8b", "llama_8b"],
        help="Model to load. Required for: evaluate, tokenize, finetune.",
    )
    parser.add_argument(
        "--model_type",
        choices=["dlm", "ar"],
        help=(
            "Model paradigm — determines PLL scoring formula.\n"
            "  dlm — masked diffusion (MDLM, LLaDA)\n"
            "  ar  — autoregressive (Pythia, LLaMA)\n"
            "Required for: evaluate, finetune."
        ),
    )
    parser.add_argument(
        "--quantize",
        action="store_true",
        default=False,
        help="Load model in 4-bit (NF4). Use on Colab for large models only.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to load model on. Default: cpu.",
    )

    # -----------------------------------------------------------------------
    # Checkpoint — used by evaluate (finetuned) and finetune (output)
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help=(
            "Path to a saved LoRA adapter directory.\n"
            "If provided, loads finetuned model instead of base model.\n"
            "Used by: evaluate (finetuned mode)."
        ),
    )

    # -----------------------------------------------------------------------
    # Evaluate args
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--statements_path",
        type=str,
        default="data/pct_statements.json",
        help="Path to pct_statements.json. Used by: evaluate.",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        help=(
            "Output JSON file path for evaluation results.\n"
            "e.g. results/base/pythia_160m.json\n"
            "Required for: evaluate."
        ),
    )

    # -----------------------------------------------------------------------
    # Corpus args
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Output directory. Used by: prepare_corpus, tokenize, finetune, plot.",
    )
    parser.add_argument(
        "--corpus_dir",
        type=str,
        default="data/corpus",
        help="Directory of cleaned corpus articles. Used by: tokenize.",
    )
    parser.add_argument(
        "--tokenized_dir",
        type=str,
        default="data/tokenized",
        help="Directory of tokenized datasets. Used by: finetune.",
    )
    parser.add_argument(
        "--n_articles",
        type=int,
        default=600,
        help="Number of articles per condition to sample. Used by: prepare_corpus.",
    )

    # -----------------------------------------------------------------------
    # Finetune args
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--condition",
        choices=["left", "right"],
        help="Corpus political condition. Required for: finetune.",
    )

    # -----------------------------------------------------------------------
    # Plot args
    # -----------------------------------------------------------------------
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results/",
        help="Directory containing base/ and finetuned/ JSON results. Used by: plot.",
    )

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Step runners — each imports its module only when needed
# ---------------------------------------------------------------------------

def run_evaluate(args):
    """Load model and run PCT evaluation. Saves results to output_path."""
    import json
    import os
    from models.model_loader import load_model, load_finetuned

    # Validate required args
    if args.model is None:
        print("Error: --model is required for --step evaluate")
        sys.exit(1)
    if args.model_type is None:
        print("Error: --model_type is required for --step evaluate")
        sys.exit(1)
    if args.output_path is None:
        print("Error: --output_path is required for --step evaluate")
        sys.exit(1)

    # Load model — base or finetuned
    if args.checkpoint is None:
        print(f"Loading base model: {args.model}")
        model, tokenizer = load_model(
            args.model,
            device=args.device,
            quantize=args.quantize,
        )
    else:
        print(f"Loading finetuned model: {args.model} from {args.checkpoint}")
        model, tokenizer = load_finetuned(
            args.model,
            checkpoint_path=args.checkpoint,
            device=args.device,
            quantize=args.quantize,
        )

    # Run PCT evaluation
    from utils.evaluation import evaluate_pct
    print(f"Running PCT evaluation ({args.model_type.upper()})...")
    results = evaluate_pct(
        model=model,
        tokenizer=tokenizer,
        model_type=args.model_type,
        statements_path=args.statements_path,
    )

    # Add metadata to results
    results["model"]      = args.model
    results["model_type"] = args.model_type
    results["checkpoint"] = args.checkpoint if args.checkpoint else "base"
    results["condition"]  = args.condition if args.condition else "base"

    # Save results
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {args.output_path}")
    print(f"  Economic score: {results['economic']}")
    print(f"  Social score:   {results['social']}")


def run_prepare_corpus(args):
    """Download, clean, and subsample the POLITICS dataset."""
    if args.output_dir is None:
        print("Error: --output_dir is required for --step prepare_corpus")
        sys.exit(1)
    from utils.preprocess import prepare_corpus
    prepare_corpus(output_dir=args.output_dir, n_articles=args.n_articles)


def run_tokenize(args):
    """Tokenize the cleaned corpus for a specific model."""
    if args.model is None:
        print("Error: --model is required for --step tokenize")
        sys.exit(1)
    if args.output_dir is None:
        print("Error: --output_dir is required for --step tokenize")
        sys.exit(1)
    from utils.preprocess import tokenize_corpus
    tokenize_corpus(
        model_name=args.model,
        corpus_dir=args.corpus_dir,
        output_dir=args.output_dir,
    )


def run_finetune(args):
    """LoRA finetune a model on a corpus condition."""
    if args.model is None:
        print("Error: --model is required for --step finetune")
        sys.exit(1)
    if args.model_type is None:
        print("Error: --model_type is required for --step finetune")
        sys.exit(1)
    if args.condition is None:
        print("Error: --condition is required for --step finetune")
        sys.exit(1)
    if args.output_dir is None:
        print("Error: --output_dir is required for --step finetune")
        sys.exit(1)
    from training.finetune import run_finetune as _finetune
    _finetune(args)


def run_plot(args):
    """Generate political compass plots from results JSONs."""
    if args.output_dir is None:
        print("Error: --output_dir is required for --step plot")
        sys.exit(1)
    from utils.plot_compass import plot_compass
    plot_compass(
        results_dir=args.results_dir,
        output_dir=args.output_dir,
    )


# ---------------------------------------------------------------------------
# Main router
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    dispatch = {
        "evaluate":       run_evaluate,
        "prepare_corpus": run_prepare_corpus,
        "tokenize":       run_tokenize,
        "finetune":       run_finetune,
        "plot":           run_plot,
    }

    dispatch[args.step](args)


if __name__ == "__main__":
    main()