
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


NUM_EPOCHS = 5
BATCH_SIZE = 4
WARMUP_STEPS = 100
DLM_MASK_RATE = 0.15   # fraction of tokens masked per sequence for DLM training

LR_MAP = {
    "pythia_160m": 2e-4,
    "mdlm_169m":   2e-4,
}


def run_finetune(args):
    """
    Entry point called from main.py --step finetune.

    Loads the base model, applies LoRA, loads the tokenized corpus for the
    given condition, trains for NUM_EPOCHS, and saves the adapter checkpoint.


    Args:
        args: parsed argparse namespace with fields:
            model_name:     'pythia_160m' or 'mdlm_169m'
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
    device = args.device if hasattr(args, "device") else "cpu"
    quantize = args.quantize if hasattr(args, "quantize") else False
    model, tokenizer = load_model(model_name, device=device, quantize=quantize)

    from peft import prepare_model_for_kbit_training
    if quantize:
        supports_gc = model_name != "mdlm_169m"
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=supports_gc,
        )

    lora_cfg = get_lora_config(model_name)
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()

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

    if model_type == "dlm":
        mask_token_id = MODEL_REGISTRY[model_name]["mask_token_id"]
        collator = _make_dlm_collator(mask_token_id, mask_rate=DLM_MASK_RATE)
    else:
        collator = _make_ar_collator(tokenizer)
    training_args = _get_training_args(args, model_name)
    if model_type == "dlm":
        _model_name = model_name 

        class DLMTrainer(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                input_ids = inputs["input_ids"]
                labels = inputs["labels"]

                if _model_name == "mdlm_169m":
                    timesteps = torch.zeros(
                        input_ids.shape[0], device=input_ids.device
                    )
                    outputs = model(
                        input_ids=input_ids,
                        timesteps=timesteps,
                        return_dict=True,
                    )
                else:
                    outputs = model(input_ids=input_ids, return_dict=True)

                logits = outputs.logits  
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

    model.save_pretrained(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Adapter saved to {args.output_dir}")

def _get_training_args(args, model_name: str) -> TrainingArguments:
    
    lr = LR_MAP.get(model_name, 2e-4)
    max_steps = getattr(args, "max_steps", -1)

    use_gc = model_name != "mdlm_169m"

    return TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=NUM_EPOCHS,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=1,
        warmup_steps=WARMUP_STEPS,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        logging_steps=50,
        save_strategy="epoch",
        save_total_limit=1,          
        fp16=torch.cuda.is_available(),
        report_to="none",            
        dataloader_drop_last=True,   
        gradient_checkpointing=use_gc,
        gradient_checkpointing_kwargs={"use_reentrant": False} if use_gc else {},
    )


def _make_dlm_collator(mask_token_id: int, mask_rate: float = DLM_MASK_RATE):
    def collate_fn(examples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        input_ids = torch.tensor(
            [ex["input_ids"] for ex in examples],
            dtype=torch.long,
        )

        if "attention_mask" in examples[0]:
            attention_mask = torch.tensor(
                [ex["attention_mask"] for ex in examples],
                dtype=torch.long,
            )
        else:
            attention_mask = torch.ones_like(input_ids)

        labels = input_ids.clone()
        rand = torch.rand(input_ids.shape)
        mask = rand < mask_rate
        input_ids[mask] = mask_token_id
        labels[~mask] = -100

        return {
            "input_ids":      input_ids,
            "attention_mask": attention_mask,
            "labels":         labels,
        }

    return collate_fn


def _make_ar_collator(tokenizer):
   
    return DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False, 
    )