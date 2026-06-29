"""
training/finetune.py

LoRA finetuning pipeline for the bias_diffusion experiment.
Supports both AR models (Pythia, LLaMA) and DLM models (MDLM, LLaDA).

The key difference between AR and DLM finetuning:
    AR:  standard causal LM loss -- model predicts next token
    DLM: masked diffusion loss -- model predicts randomly masked tokens
         at a fixed masking rate (15%), following BERT-style masking

Uses HuggingFace Trainer for the training loop -- handles gradient
accumulation, checkpointing, and logging automatically.

Training hyperparameters:
    Epochs:     3
    Batch size: 4
    LR:         2e-4 (small models), 1e-4 (large models)
    Warmup:     100 steps
    Scheduler:  cosine

References:
    Hu et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models.
    Dettmers et al. (2023). QLoRA: Efficient Finetuning of Quantized LLMs.
    Feng et al. (2023). From Pretraining Data to Language Models. ACL.

Authors: [your name]
"""

import os
import torch
import numpy as np
from dataclasses import dataclass
from typing import Any, Dict, List

from datasets import load_from_disk
from peft import get_peft_model
from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling

from models.model_loader import load_model, MODEL_REGISTRY
from training.lora_config import get_lora_config


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

NUM_EPOCHS = 3
BATCH_SIZE = 4
WARMUP_STEPS = 100
DLM_MASK_RATE = 0.15   # fraction of tokens masked per sequence for DLM training

LR_MAP = {
    "pythia_160m": 2e-4,
    "mdlm_169m":   2e-4,
    "llada_8b":    1e-4,
    "llama_8b":    1e-4,
}


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def run_finetune(args):
    """
    Entry point called from main.py --step finetune.

    Loads the base model, applies LoRA, loads the tokenized corpus for the
    given condition, trains for NUM_EPOCHS, and saves the adapter checkpoint.

    Args:
        args: parsed argparse namespace with fields:
            model_name:     one of the four model keys
            model_type:     'ar' or 'dlm'
            condition:      'left' or 'right'
            tokenized_dir:  directory containing tokenized HF Datasets
            output_dir:     where to save the LoRA adapter checkpoint
            device:         'cpu' or 'cuda'
            quantize:       bool -- 4-bit loading for large models
    """
    model_name = args.model
    model_type = args.model_type
    condition  = args.condition

    print(f"Finetuning {model_name} on {condition} corpus...")

    # Load base model and tokenizer
    device = args.device if hasattr(args, "device") else "cpu"
    quantize = args.quantize if hasattr(args, "quantize") else False
    model, tokenizer = load_model(model_name, device=device, quantize=quantize)

    # Apply LoRA
    lora_cfg = get_lora_config(model_name)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

    # Load tokenized dataset for this model and condition
    dataset_path = os.path.join(
        args.tokenized_dir, f"{model_name}_{condition}"
    )
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Tokenized dataset not found at {dataset_path}. "
            f"Run --step tokenize first."
        )
    dataset = load_from_disk(dataset_path)
    print(f"Loaded {len(dataset)} training chunks from {dataset_path}")

    # Build data collator based on model paradigm
    if model_type == "dlm":
        mask_token_id = MODEL_REGISTRY[model_name]["mask_token_id"]
        collator = _make_dlm_collator(mask_token_id, mask_rate=DLM_MASK_RATE)
    else:
        collator = _make_ar_collator(tokenizer)

    # Build training arguments
    training_args = _get_training_args(args, model_name)

    # For DLM models with task_type=None, Trainer needs a custom compute_loss
    # because the model output is not a standard CausalLMOutput with .loss.
    # We define a minimal Trainer subclass that computes cross-entropy loss
    # only at masked positions (where labels != -100).
    if model_type == "dlm":
        class DLMTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                input_ids = inputs["input_ids"]
                attention_mask = inputs["attention_mask"]
                labels = inputs["labels"]

                # Forward pass -- get logits
                if "timesteps" in model.forward.__code__.co_varnames:
                    # MDLM: pass timesteps=0 (fully denoised)
                    timesteps = torch.zeros(input_ids.shape[0], device=input_ids.device)
                    outputs = model(input_ids=input_ids, timesteps=timesteps, return_dict=True)
                else:
                    # LLaDA: standard forward with use_cache=False
                    outputs = model(input_ids=input_ids, use_cache=False, return_dict=True)

                logits = outputs.logits  # [batch, seq_len, vocab_size]

                # Cross-entropy loss only at masked positions (labels != -100)
                loss = torch.nn.functional.cross_entropy(
                    logits.view(-1, logits.size(-1)),
                    labels.view(-1),
                    ignore_index=-100,
                )
                return (loss, outputs) if return_outputs else loss

        trainer = DLMTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )
    else:
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=dataset,
            data_collator=collator,
        )

    trainer.train()

    # Save LoRA adapter only (not full model weights)
    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _get_training_args(args, model_name: str) -> TrainingArguments:
    """
    Build HuggingFace TrainingArguments for a given model and condition.

    Args:
        args:       argparse namespace (for output_dir, condition)
        model_name: model key -- used to look up learning rate

    Returns:
        TrainingArguments configured for this run
    """
    lr = LR_MAP.get(model_name, 2e-4)

    # max_steps caps training regardless of dataset size -- used to equalize
    # training across left/right conditions when chunk counts differ.
    # -1 means no cap (default -- use num_train_epochs instead).
    max_steps = getattr(args, "max_steps", -1)

    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=NUM_EPOCHS,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        warmup_steps=WARMUP_STEPS,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,          # keep only the last checkpoint
        fp16=torch.cuda.is_available(),
        report_to="none",            # disable wandb/tensorboard
        dataloader_drop_last=True,   # drop last batch if smaller than batch_size
    )


def _make_dlm_collator(mask_token_id: int, mask_rate: float = DLM_MASK_RATE):
    """
    Returns a data collator for DLM finetuning.

    Randomly masks mask_rate fraction of tokens in each sequence with
    mask_token_id. The model is trained to predict the original tokens
    at masked positions. Only masked positions contribute to the loss.

    Args:
        mask_token_id: token ID used as the mask token (model-specific)
        mask_rate:     fraction of tokens to mask (default 0.15)

    Returns:
        collate_fn: function that takes a list of examples and returns
                    a dict with input_ids, attention_mask, labels
    """
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        # Stack input_ids and attention_mask into tensors
        input_ids = torch.tensor(
            [ex["input_ids"] for ex in examples],
            dtype=torch.long,
        )
        attention_mask = torch.tensor(
            [ex["attention_mask"] for ex in examples],
            dtype=torch.long,
        )

        # Clone input_ids as labels -- we will mask input_ids in place
        labels = input_ids.clone()

        # Create random mask: True where token should be masked
        # mask_rate fraction of tokens per sequence
        rand = torch.rand(input_ids.shape)
        mask = rand < mask_rate

        # Replace masked positions in input_ids with mask_token_id
        input_ids[mask] = mask_token_id

        # Set unmasked positions in labels to -100 (PyTorch ignore index)
        # Loss is only computed at masked positions
        labels[~mask] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    return collate_fn


def _make_ar_collator(tokenizer):
    """
    Returns a standard data collator for causal LM (AR) finetuning.
    Uses HuggingFace DataCollatorForLanguageModeling with mlm=False,
    which sets labels = input_ids and handles padding.

    Args:
        tokenizer: the model's tokenizer

    Returns:
        DataCollatorForLanguageModeling instance
    """
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,  # causal LM -- labels = input_ids, no masking
    )