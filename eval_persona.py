import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path

import numpy as np
from huggingface_hub import model_info
from peft import PeftModel
from sentence_transformers import SentenceTransformer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig, set_seed


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def parse_args():
    parser = argparse.ArgumentParser(description="Compare base and SFT replies with held-out references.")
    parser.add_argument("--adapter", type=Path, default=Path("out/sft-lora"))
    parser.add_argument("--data", type=Path, default=Path("data/pairs_heldout.jsonl"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/persona"))
    parser.add_argument("--metrics-file", type=Path, default=Path("results/persona.json"))
    parser.add_argument("--embedding-model", default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--embedding-revision")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--bootstrap-samples", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.batch_size, args.max_new_tokens, args.bootstrap_samples) < 1:
        parser.error("Batch size, generation limit, and bootstrap samples must be positive")
    if not torch.cuda.is_available():
        parser.error("An NVIDIA GPU is required")
    return args


def generate(model, tokenizer, prompts, settings):
    texts = [tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in prompts]
    batch = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(model.device)
    if batch.input_ids.shape[1] + settings.max_new_tokens > model.config.max_position_embeddings:
        raise ValueError("A prompt plus generation limit exceeds the model context window")
    with torch.inference_mode():
        sequences = model.generate(**batch, generation_config=settings)
    generated = sequences[:, batch.input_ids.shape[1]:].tolist()
    eos = settings.eos_token_id
    eos = {eos} if isinstance(eos, int) else set(eos)
    outputs = []
    for tokens in generated:
        stop = next((i for i, token in enumerate(tokens) if token in eos), len(tokens))
        outputs.append({
            "text": tokenizer.decode(tokens[:stop], skip_special_tokens=True).strip(),
            "tokens": stop,
            "hit_token_limit": stop == len(tokens) and len(tokens) == settings.max_new_tokens,
        })
    return outputs


def score(rows, encoder, bootstrap_samples, seed):
    references = [row["reference"] for row in rows]
    reference_vectors = encoder.encode(references, normalize_embeddings=True, show_progress_bar=False)
    metrics = {}
    scores = {}
    for name in ("base", "sft"):
        texts = [row[name]["text"] for row in rows]
        vectors = encoder.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        scores[name] = np.sum(vectors * reference_vectors, axis=1).astype(float)
        lengths = encoder.tokenizer(texts, truncation=False, padding=False)["input_ids"]
        metrics[name] = {
            "mean_cosine_similarity": float(scores[name].mean()),
            "mean_generated_tokens": float(np.mean([row[name]["tokens"] for row in rows])),
            "empty_responses": sum(not text for text in texts),
            "generation_limit_hits": sum(row[name]["hit_token_limit"] for row in rows),
            "embedding_truncations": sum(len(tokens) > encoder.max_seq_length for tokens in lengths),
        }
        for row, similarity in zip(rows, scores[name]):
            row[name]["reference_cosine"] = float(similarity)
    differences = scores["sft"] - scores["base"]
    episodes = sorted({row["episode"] for row in rows})
    grouped = [differences[[row["episode"] == episode for row in rows]] for episode in episodes]
    totals = np.array([group.sum() for group in grouped])
    counts = np.array([len(group) for group in grouped])
    rng = np.random.default_rng(seed)
    sampled = rng.integers(len(episodes), size=(bootstrap_samples, len(episodes)))
    bootstrap = totals[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    reference_lengths = encoder.tokenizer(references, truncation=False, padding=False)["input_ids"]
    return {
        "examples": len(rows),
        "episodes": len(episodes),
        "base": metrics["base"],
        "sft": metrics["sft"],
        "mean_paired_difference": float(differences.mean()),
        "paired_difference_episode_bootstrap_95_ci": np.quantile(bootstrap, [0.025, 0.975]).tolist(),
        "fraction_sft_higher": float(np.mean(differences > 0)),
        "embedding_max_sequence_length": encoder.max_seq_length,
        "reference_embedding_truncations": sum(len(tokens) > encoder.max_seq_length for tokens in reference_lengths),
        "interpretation": "Semantic similarity to the matching reference; not a direct measure of persona style.",
    }


def main():
    args = parse_args()
    set_seed(args.seed)
    run = json.loads((args.adapter / "run.json").read_text())
    data = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()]
    if not data:
        raise ValueError("Held-out data is empty")
    for index, row in enumerate(data):
        messages = row["messages"]
        if len(messages) < 2 or len(messages) % 2 or not row["meta"]["episode"]:
            raise ValueError(f"Invalid conversation at row {index}")
        for position, message in enumerate(messages):
            if message["role"] != ("user" if position % 2 == 0 else "assistant") or not message["content"].strip():
                raise ValueError(f"Invalid message at row {index}")
    manifest_path = args.output_dir / "run.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    revision = args.embedding_revision or (previous["embedding_revision"] if previous else None)
    embedding_revision = model_info(args.embedding_model, revision=revision).sha
    manifest = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "base_model": run["arguments"]["model"],
        "base_revision": run["model_revision"],
        "adapter_sha256": digest(args.adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": digest(args.adapter / "adapter_config.json"),
        "dataset_sha256": digest(args.data),
        "embedding_revision": embedding_revision,
        "decoding": {"do_sample": False, "num_beams": 1, "max_new_tokens": args.max_new_tokens},
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "sentence-transformers")},
    }
    if previous is not None and previous != manifest:
        raise ValueError("Evaluation settings changed; use a new output directory")
    write_json(manifest_path, manifest)
    outputs_path = args.output_dir / "generations.jsonl"
    rows = [json.loads(line) for line in outputs_path.read_text().splitlines()] if outputs_path.exists() else []
    if len(rows) > len(data) or any(row["index"] != i for i, row in enumerate(rows)):
        raise ValueError("Saved generation sequence is invalid")
    if len(rows) < len(data):
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        base = AutoModelForCausalLM.from_pretrained(
            manifest["base_model"], revision=manifest["base_revision"],
            dtype=torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16,
            device_map={"": 0},
        )
        model = PeftModel.from_pretrained(base, args.adapter).eval()
        settings = GenerationConfig(
            **manifest["decoding"], pad_token_id=tokenizer.pad_token_id,
            eos_token_id=model.generation_config.eos_token_id or tokenizer.eos_token_id,
        )
        with outputs_path.open("a") as output:
            for start in range(len(rows), len(data), args.batch_size):
                examples = data[start:start + args.batch_size]
                prompts = [row["messages"][:-1] for row in examples]
                predictions = {}
                for name in ("base", "sft"):
                    with model.disable_adapter() if name == "base" else nullcontext():
                        predictions[name] = generate(model, tokenizer, prompts, settings)
                for offset, example in enumerate(examples):
                    row = {
                        "index": start + offset, "episode": example["meta"]["episode"],
                        "prompt": prompts[offset], "reference": example["messages"][-1]["content"],
                        "base": predictions["base"][offset], "sft": predictions["sft"][offset],
                    }
                    rows.append(row)
                    output.write(json.dumps(row) + "\n")
                    output.flush()
                print(f"Generated paired replies: {len(rows)}/{len(data)}", flush=True)
        del model, base
        torch.cuda.empty_cache()
    encoder = SentenceTransformer(args.embedding_model, revision=embedding_revision, device="cuda")
    metrics = score(rows, encoder, args.bootstrap_samples, args.seed)
    metrics["run"] = manifest
    write_json(args.output_dir / "scored_outputs.json", rows)
    write_json(args.metrics_file, metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
