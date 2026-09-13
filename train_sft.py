import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path

import torch
from datasets import load_dataset
from peft import LoraConfig
from transformers import AutoConfig, AutoTokenizer, set_seed
from trl import SFTConfig, SFTTrainer


def parse_args():
    parser = argparse.ArgumentParser(description="Train a persona LoRA adapter.")
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--revision")
    parser.add_argument("--train-file", type=Path, default=Path("data/pairs_train.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/sft-lora"))
    parser.add_argument("--epochs", type=float, default=1.0)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=16)
    parser.add_argument("--max-length", type=int, default=1024)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=-1)
    parser.add_argument("--resume-from-checkpoint", type=Path)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    for name in ("epochs", "learning_rate", "batch_size", "gradient_accumulation_steps", "max_length", "lora_rank"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name.replace('_', '-')} must be positive")
    if args.max_steps != -1 and args.max_steps <= 0:
        parser.error("--max-steps must be positive or -1")
    if not args.train_file.is_file():
        parser.error(f"Training file does not exist: {args.train_file}")
    if args.output_dir.exists() and any(args.output_dir.iterdir()) and not args.resume_from_checkpoint:
        parser.error("Output directory is not empty; choose a new directory or resume a checkpoint")
    if args.resume_from_checkpoint and not (args.resume_from_checkpoint / "trainer_state.json").is_file():
        parser.error("The resume directory must contain a Trainer checkpoint")
    if not args.cpu and not torch.cuda.is_available():
        parser.error("An NVIDIA GPU is required; use --cpu only for small local checks")
    return args


def prepare(example, index, tokenizer, max_length):
    messages = example["messages"]
    if len(messages) < 2 or len(messages) % 2:
        raise ValueError(f"Example {index}: expected alternating user/assistant messages")
    for position, message in enumerate(messages):
        role = "user" if position % 2 == 0 else "assistant"
        if message.get("role") != role or not isinstance(message.get("content"), str) or not message["content"].strip():
            raise ValueError(f"Example {index}: invalid message at position {position}")
    prompt = messages[:-1]
    prompt_ids = tokenizer.apply_chat_template(prompt, tokenize=True, add_generation_prompt=True, return_dict=False)
    full_ids = tokenizer.apply_chat_template(messages, tokenize=True, return_dict=False)
    if full_ids[:len(prompt_ids)] != prompt_ids:
        raise ValueError(f"Example {index}: chat template does not preserve the prompt prefix")
    if len(full_ids) > max_length:
        raise ValueError(f"Example {index}: {len(full_ids)} tokens exceed --max-length {max_length}")
    if len(full_ids) <= len(prompt_ids) or tokenizer.eos_token_id not in full_ids[len(prompt_ids):]:
        raise ValueError(f"Example {index}: missing completion or end-of-turn token")
    return {"prompt": prompt, "completion": messages[-1:]}


def write_model_card(output_dir, run, metrics):
    arguments = run["arguments"]
    content = f"""# Persona SFT adapter

LoRA adapter for `{arguments['model']}`. Load it with the original base model
at the revision recorded in `run.json`.

## Training

- Training examples: {run['training_examples']}
- Completed epochs: {metrics['epoch']}
- LoRA rank: {arguments['lora_rank']}
- Initial learning rate: {arguments['learning_rate']}
- Batch size: {arguments['batch_size']}
- Gradient accumulation: {arguments['gradient_accumulation_steps']}
- Training loss: {metrics['train_loss']:.6f}
- Training time: {metrics['train_runtime']:.2f} seconds

Only the final assistant reply in each conversation contributes to the loss.
`run.json` records the dataset hash, model revision, settings, and package
versions. `train_results.json` contains training metrics. Persona and STEM
evaluation are separate from this training run.

## Loading

```python
import json
from pathlib import Path
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

adapter = Path({json.dumps(str(output_dir))})
run = json.loads((adapter / "run.json").read_text())
base = AutoModelForCausalLM.from_pretrained(
    run["arguments"]["model"],
    revision=run["model_revision"],
    dtype="auto",
    device_map="auto",
)
model = PeftModel.from_pretrained(base, adapter).eval()
tokenizer = AutoTokenizer.from_pretrained(adapter)
```

Use the saved tokenizer's chat template when generating responses.
"""
    (output_dir / "README.md").write_text(content)


def main():
    args = parse_args()
    set_seed(args.seed)
    config = AutoConfig.from_pretrained(args.model, revision=args.revision)
    revision = getattr(config, "_commit_hash", None) or args.revision
    tokenizer = AutoTokenizer.from_pretrained(args.model, revision=revision)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    data = load_dataset("json", data_files=str(args.train_file), split="train")
    if not len(data):
        raise ValueError("Training data is empty")
    data = data.map(
        prepare,
        with_indices=True,
        fn_kwargs={"tokenizer": tokenizer, "max_length": args.max_length},
        remove_columns=data.column_names,
        desc="Preparing conversations",
    )
    bf16 = not args.cpu and torch.cuda.is_bf16_supported()
    fp16 = not args.cpu and not bf16
    dtype = "bfloat16" if bf16 else "float16" if fp16 else "float32"
    lora = LoraConfig(
        task_type="CAUSAL_LM",
        r=args.lora_rank,
        lora_alpha=2 * args.lora_rank,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    settings = SFTConfig(
        output_dir=str(args.output_dir),
        completion_only_loss=True,
        learning_rate=args.learning_rate,
        num_train_epochs=args.epochs,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_length=args.max_length,
        gradient_checkpointing=True,
        bf16=bf16,
        fp16=fp16,
        use_cpu=args.cpu,
        model_init_kwargs={"revision": revision, "dtype": dtype},
        eos_token=tokenizer.eos_token,
        packing=False,
        seed=args.seed,
        data_seed=args.seed,
        save_strategy="epoch",
        logging_steps=10,
        logging_first_step=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=args.model,
        processing_class=tokenizer,
        train_dataset=data,
        peft_config=lora,
        args=settings,
    )
    for index, (source, encoded) in enumerate(zip(data, trainer.train_dataset)):
        prompt_ids = tokenizer.apply_chat_template(source["prompt"], tokenize=True, add_generation_prompt=True, return_dict=False)
        labels = encoded["labels"]
        if labels[:len(prompt_ids)] != [-100] * len(prompt_ids) or labels[len(prompt_ids):] != encoded["input_ids"][len(prompt_ids):]:
            raise ValueError(f"Example {index}: completion-only loss mask is incorrect")
    run = {
        "arguments": {name: str(value) if isinstance(value, Path) else value for name, value in vars(args).items()},
        "model_revision": revision,
        "dataset_sha256": hashlib.sha256(args.train_file.read_bytes()).hexdigest(),
        "training_examples": len(data),
        "dtype": dtype,
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "datasets", "accelerate", "peft", "trl")},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    result = trainer.train(resume_from_checkpoint=str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None)
    trainer.save_model(str(args.output_dir))
    tokenizer.save_pretrained(args.output_dir)
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    write_model_card(args.output_dir, run, result.metrics)


if __name__ == "__main__":
    main()
