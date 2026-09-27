# Can the Political Compass Test See Injected Bias?

Code, data and results for the TNLP research poster (Trier University, NLP module, 2026).

**Question.** If we inject a *known, verified* political bias into a language model, does the Political Compass Test (PCT) recover its direction?

**Setup.** We LoRA-finetune an autoregressive model (Pythia-160M) and a masked diffusion model (MDLM-169M) on left- vs. right-leaning news from BIGNEWSBLN, at two strengths:

| Strength | Articles per side | LoRA rank / alpha | Epochs |
|---|---|---|---|
| weak | 1,000 | 16 / 32 | 3 |
| strong | 5,000 | 32 / 64 | 5 |

---

## Environment

All runs used Google Colab on an A100 GPU with Python 3.13. The versions below are the ones used for the evaluation runs; `requirements.txt` lists the minimum versions.

```
torch 2.11.0+cu128   transformers 5.16.1   peft 0.20.0      datasets 4.8.5
accelerate 1.14.0    huggingface_hub 1.29.0   numpy 2.1.3   scipy 1.16.3
flash-attn 2.8.3     einops 0.8.2           ijson 3.5.1     matplotlib 3.10.0
```

```bash
git clone https://github.com/Bormey-Sky/TNLP_poster_bias_diff_ar.git
cd TNLP_poster_bias_diff_ar
pip install -r requirements.txt
# MDLM's remote code needs flash-attention (Ampere GPU or newer, e.g. A100).
# Install the wheel matching your torch / CUDA / Python versions, e.g. for the setup above:
pip install "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.11/flash_attn-2.8.3%2Bcu12torch2.11cxx11abiTRUE-cp313-cp313-linux_x86_64.whl"
```

Hardware notes:
- Both base models are public on the Hugging Face Hub (`EleutherAI/pythia-160m`, `kuleshov-group/mdlm-owt`), so no HF token is needed.
- Compatibility patches for transformers 5.x are applied automatically when `models/model_loader.py` is imported.

---

## Full pipeline

All commands run from the repository root. `--device cuda` is assumed where a GPU is needed.

### Step 0: data (optional; the exact data used is already committed)

The committed `data/corpus`, `data/corpus_2` and held-out files **are** the data used for the poster. Only regenerate them if you want to change the sample. This requires the BIGNEWSBLN release (Liu et al., 2022), which has to be obtained from its authors.

Two caveats before regenerating:
- The 1k corpus was sampled with an earlier version of `prepare_corpus`, so re-running it will not give the same 1,000 articles byte for byte.
- `prepare_heldout` excludes only the articles in `--corpus_dir`. The held-out sets were built against the 1k corpus; 1 of the 500 right-side held-out articles also appears in the 5k right corpus (`data/corpus_2/right.json`).

```bash
# training corpora (weak = 1,000/side, strong = 5,000/side)
python main.py prepare_corpus --output_dir data/corpus   --n_articles 1000 \
    --left_path <BIGNEWSBLN_left.json> --right_path <BIGNEWSBLN_right.json>
python main.py prepare_corpus --output_dir data/corpus_2 --n_articles 5000 \
    --left_path <BIGNEWSBLN_left.json> --right_path <BIGNEWSBLN_right.json>

# tokenization (per model, per corpus)
for m in pythia_160m mdlm_169m; do
  python main.py tokenize --model $m --corpus_dir data/corpus   --output_dir data/tokenized
  python main.py tokenize --model $m --corpus_dir data/corpus_2 --output_dir data/tokenized_2
done

# held-out sets: 500 articles/side, disjoint from the training corpus in --corpus_dir,
# cross-side duplicates (e.g. wire copy) removed
python main.py prepare_heldout --corpus_dir data/corpus --output_dir data/corpus --n_heldout 500 \
    --left_path <BIGNEWSBLN_left.json> --right_path <BIGNEWSBLN_right.json>
```

### Step 1: LoRA finetuning (8 adapters)

The weak condition uses r = 16, alpha = 32 and 3 epochs; the strong condition uses the defaults (r = 32, alpha = 64, 5 epochs). Other settings: learning rate 2e-4 with cosine schedule, batch size 4, 100 warmup steps, 15% masking rate for MDLM.

```bash
for cond in left right; do
  # weak (1k)
  python main.py finetune --model pythia_160m --model_type ar  --condition $cond \
      --tokenized_dir data/tokenized   --output_dir checkpoints/pythia_160m_$cond \
      --lora_r 16 --lora_alpha 32 --epochs 3 --device cuda
  python main.py finetune --model mdlm_169m   --model_type dlm --condition $cond \
      --tokenized_dir data/tokenized   --output_dir checkpoints/mdlm_169m_$cond \
      --lora_r 16 --lora_alpha 32 --epochs 3 --device cuda
  # strong (5k)
  python main.py finetune --model pythia_160m --model_type ar  --condition $cond \
      --tokenized_dir data/tokenized_2 --output_dir checkpoints/pythia_160m_v2_$cond --device cuda
  python main.py finetune --model mdlm_169m   --model_type dlm --condition $cond \
      --tokenized_dir data/tokenized_2 --output_dir checkpoints/mdlm_169m_v2_$cond --device cuda
done
```

Reference timing: MDLM strong takes about 9 minutes per side on an A100 (8,145 steps).

The trained adapters are not in git (`checkpoints/` is ignored). They are available here: **[add link: Google Drive / Hugging Face]**.

### Step 2: injection verification on held-out news

This produces one file per (model, condition, held-out side): `results/heldout/<model>[_v2]_<condition>_<side>.json`.

```bash
for m in pythia_160m:ar mdlm_169m:dlm; do
  model=${m%%:*}; type=${m##*:}
  for tag in "" "_v2"; do
    for side in left right; do
      # base model (optional; used only for the matched-minus-base diagnostic)
      python main.py eval_heldout --model $model --model_type $type --condition base \
          --heldout_side $side --heldout_path data/corpus/heldout_$side.json \
          --output_path results/heldout/${model}${tag}_base_${side}.json --device cuda
      for cond in left right; do
        python main.py eval_heldout --model $model --model_type $type --condition $cond \
            --checkpoint checkpoints/${model}${tag}_${cond} \
            --heldout_side $side --heldout_path data/corpus/heldout_$side.json \
            --output_path results/heldout/${model}${tag}_${cond}_${side}.json --device cuda
      done
    done
  done
done
```

Injection counts as verified when both matched-minus-opposed NormPLL gaps are positive with a 95% bootstrap CI excluding 0. For MDLM, the held-out score is a pseudo-log-likelihood over 3 random 15% masks per 512-token chunk.

### Step 3: PCT scoring under four paraphrased template sets

`--templates all` scores all 62 statements under template sets v1–v4 in one pass. The headline axis scores use v1, and `by_template_set` holds all four.

```bash
for m in pythia_160m:ar mdlm_169m:dlm; do
  model=${m%%:*}; type=${m##*:}
  python main.py evaluate --model $model --model_type $type --condition base --templates all \
      --output_path results/base/$model.json --device cuda
  for tag in "" "_v2"; do
    for cond in left right; do
      python main.py evaluate --model $model --model_type $type --condition $cond --templates all \
          --checkpoint checkpoints/${model}${tag}_${cond} \
          --output_path results/finetuned/${model}${tag}_${cond}.json --device cuda
    done
  done
done
```

Reference timing: about 15 s per checkpoint for Pythia and about 10 minutes for MDLM (exact per-token PLL, 62 × 4 × 4 prompts).

### Step 4: statistics

```bash
python main.py stats --results_dir results --output_path results/stats_report.json
```

For each model and strength this reports:
- the injection contrast (Step A);
- the directional effect `D` per axis with a 10,000-resample bootstrap CI, a sign-flip permutation p-value and the number of changed statements (Step B);
- per-statement deltas (Step C);
- the paraphrase-vs-condition spread ratio (Step D);
- the weak → strong dose effect.

### Step 5: figures

Poster Figures 2–4 are generated in `notebooks/poster_pipeline.ipynb` (section "Poster figures") directly from `results/stats_report.json`, so they cannot drift from the numbers. Figure 1 (pipeline diagram) was drawn by hand.

---

## Methodological notes

- **Scoring.** Each statement is appended with four stance phrasings ("[STATEMENT] I strongly agree with this.", …). The stance with the highest length-normalised (pseudo-)log-likelihood wins (argmax). Its Likert weight (±1, ±0.5, scaled to ±10) is multiplied by the statement's polarity sign and averaged per axis.
- **Polarity and axes.** The polarity signs (`polarity`, with `polarity_confidence` ∈ {high, low}) are our own semantic annotation, not the official PCT key; the PCT's per-statement weights are proprietary. Economic = the 14 "economy" propositions (IDs 8–21), social = the remaining 48, following the PCT's topic pages.
- **Statistics.** Bootstrap resamples statements (paired across conditions). The permutation test flips condition labels per statement. With k changed statements, the smallest reachable two-sided p is about 2/2^k, which is why `n_changed` is always reported.
- **Out of scope.** Earlier exploratory runs with LLaMA-3.1-8B and LLaDA-8B are not part of the poster and are not supported by the current CLI.

## References

- Feng et al. (2023). *From Pretraining Data to Language Models to Downstream Tasks.* ACL.
- Röttger et al. (2024). *Political Compass or Spinning Arrow?* ACL.
- Liu et al. (2022). *POLITICS: Pretraining with Same-story Article Comparison …* (BIGNEWSBLN). Findings of NAACL.
- Biderman et al. (2023). *Pythia.* ICML.
- Sahoo et al. (2024). *Simple and Effective Masked Diffusion Language Models.* NeurIPS.
- Hu et al. (2022). *LoRA.* ICLR.
- Salazar et al. (2020). *Masked Language Model Scoring.* ACL.

The complete list is in the poster appendix.
