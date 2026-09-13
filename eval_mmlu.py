import argparse
from contextlib import nullcontext
import hashlib
import importlib.metadata
import json
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import dataset_info
import numpy as np
from peft import PeftModel
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed


SUBJECTS = ("high_school_physics", "college_physics", "conceptual_physics")
LETTERS = "ABCD"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    temporary.replace(path)


def prepare_data(manifest_path, data_path):
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if manifest and data_path.exists():
        if digest(data_path) != manifest["data_sha256"]:
            raise ValueError("Physics data differs from the frozen evaluation manifest")
        return json.loads(data_path.read_text()), manifest
    revision = manifest["revision"] if manifest else dataset_info("cais/mmlu").sha
    records = []
    for subject in SUBJECTS:
        dataset = load_dataset("cais/mmlu", subject, revision=revision)
        records.append({"subject": subject, "dev": list(dataset["dev"]), "test": list(dataset["test"])})
    payload = {"dataset": "cais/mmlu", "revision": revision, "subjects": records}
    write_json(data_path, payload)
    frozen = {
        "dataset": "cais/mmlu", "revision": revision, "split": "test", "shots": 5,
        "subjects": {r["subject"]: {"dev_indices": list(range(5)), "test_indices": list(range(len(r["test"])))} for r in records},
        "data_sha256": digest(data_path),
    }
    if manifest and manifest != frozen:
        raise ValueError("Downloaded data differs from the frozen evaluation manifest")
    write_json(manifest_path, frozen)
    return payload, frozen


def question_text(row):
    if len(row["choices"]) != 4 or row["answer"] not in range(4):
        raise ValueError("Each MMLU question must have four choices and a valid answer")
    return row["question"] + "\n" + "\n".join(f"{letter}. {choice}" for letter, choice in zip(LETTERS, row["choices"]))


def make_examples(data, manifest):
    examples = []
    if set(manifest["subjects"]) != set(SUBJECTS) or manifest["shots"] != 5:
        raise ValueError("Expected the three physics subjects and five development examples")
    for subject_data in data["subjects"]:
        subject = subject_data["subject"]
        selection = manifest["subjects"][subject]
        demonstrations = [subject_data["dev"][i] for i in selection["dev_indices"]]
        if len(demonstrations) != 5:
            raise ValueError("Expected five development examples per subject")
        for index in selection["test_indices"]:
            row = subject_data["test"][index]
            if any(row["question"] == demo["question"] for demo in demonstrations):
                raise ValueError("A test question overlaps its development examples")
            messages = [{
                "role": "system",
                "content": "Answer the multiple-choice question. Respond with only the letter of the correct option: A, B, C, or D.",
            }]
            for demo in demonstrations:
                messages.extend([
                    {"role": "user", "content": question_text(demo)},
                    {"role": "assistant", "content": LETTERS[demo["answer"]]},
                ])
            messages.append({"role": "user", "content": question_text(row)})
            examples.append({
                "subject": subject, "index": index, "messages": messages,
                "answer": LETTERS[row["answer"]],
            })
    if not examples:
        raise ValueError("Evaluation data is empty")
    return examples


def choice_token_ids(tokenizer):
    tokens = [tokenizer.encode(letter, add_special_tokens=False) for letter in LETTERS]
    if any(len(ids) != 1 for ids in tokens) or len({ids[0] for ids in tokens}) != 4:
        raise ValueError("This evaluator requires distinct single-token A/B/C/D labels")
    return [ids[0] for ids in tokens]


def predict(model, batch, token_ids, tokenizer):
    with torch.inference_mode():
        logits = model(**batch, use_cache=False, logits_to_keep=1).logits[:, -1].float()
        if not torch.isfinite(logits).all():
            raise ValueError("The model produced non-finite choice logits")
        letter_logits = logits[:, token_ids]
        probabilities = letter_logits.softmax(dim=-1).cpu().numpy()
        masses = (letter_logits.logsumexp(-1) - logits.logsumexp(-1)).exp().cpu().tolist()
        unconstrained = logits.argmax(-1).cpu().tolist()
    return [{
        "prediction": LETTERS[int(np.argmax(p))],
        "choice_probabilities": dict(zip(LETTERS, map(float, p))),
        "answer_letter_probability_mass": mass,
        "unconstrained_top_token": tokenizer.decode([top]),
    } for p, mass, top in zip(probabilities, masses, unconstrained)]


def summarize(rows, seed):
    metrics = {"questions": len(rows), "by_subject": {}}
    for subject in (None, *SUBJECTS):
        subset = rows if subject is None else [r for r in rows if r["subject"] == subject]
        if not subset:
            raise ValueError("Missing subject results")
        values = {"questions": len(subset)}
        for name in ("base", "sft"):
            correct = sum(r[name]["prediction"] == r["answer"] for r in subset)
            values[name] = {
                "correct": correct, "accuracy": correct / len(subset),
                "mean_answer_letter_probability_mass": float(np.mean([r[name]["answer_letter_probability_mass"] for r in subset])),
            }
        values["accuracy_difference"] = values["sft"]["accuracy"] - values["base"]["accuracy"]
        if subject is None:
            metrics["overall"] = values
        else:
            metrics["by_subject"][subject] = values
    rng = np.random.default_rng(seed)
    bootstrap_totals = np.zeros(2000)
    for subject in SUBJECTS:
        differences = np.array([
            int(r["sft"]["prediction"] == r["answer"]) - int(r["base"]["prediction"] == r["answer"])
            for r in rows if r["subject"] == subject
        ])
        sampled = rng.integers(len(differences), size=(2000, len(differences)))
        bootstrap_totals += differences[sampled].sum(axis=1)
    metrics["paired_accuracy_difference_95_ci"] = np.quantile(bootstrap_totals / len(rows), [0.025, 0.975]).tolist()
    metrics["base_correct_sft_wrong"] = sum(r["base"]["prediction"] == r["answer"] and r["sft"]["prediction"] != r["answer"] for r in rows)
    metrics["base_wrong_sft_correct"] = sum(r["base"]["prediction"] != r["answer"] and r["sft"]["prediction"] == r["answer"] for r in rows)
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Five-shot MMLU physics accuracy for base and persona SFT models.")
    parser.add_argument("--adapter", type=Path, default=Path("out/sft-lora"))
    parser.add_argument("--manifest", type=Path, default=Path("eval/mmlu_physics.json"))
    parser.add_argument("--data", type=Path, default=Path("data/mmlu_physics.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("out/mmlu_physics"))
    parser.add_argument("--metrics-file", type=Path, default=Path("results/mmlu_physics.json"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cuda", "mps", "cpu"), default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--prepare-only", action="store_true")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("Batch size must be positive")
    data, frozen = prepare_data(args.manifest, args.data)
    examples = make_examples(data, frozen)
    if args.prepare_only:
        print(f"Frozen {len(examples)} test questions across {len(SUBJECTS)} subjects")
        return
    set_seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    bf16 = (device == "cuda" and torch.cuda.is_bf16_supported()) or (device == "mps" and torch.backends.mps.is_macos_or_newer(14, 0))
    dtype = torch.bfloat16 if bf16 else torch.float32 if device == "cpu" else torch.float16
    training = json.loads((args.adapter / "run.json").read_text())
    run = {
        "arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "base_model": training["arguments"]["model"], "base_revision": training["model_revision"],
        "adapter_sha256": digest(args.adapter / "adapter_model.safetensors"),
        "adapter_config_sha256": digest(args.adapter / "adapter_config.json"),
        "data_sha256": digest(args.data), "manifest_sha256": digest(args.manifest),
        "script_sha256": digest(Path(__file__)),
        "device": device, "dtype": str(dtype),
        "protocol": "Five-shot chat; choose highest-probability next-token A/B/C/D; no free-form generation or chain of thought.",
        "packages": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "datasets", "numpy")},
    }
    run_path = args.output_dir / "run.json"
    if run_path.exists() and json.loads(run_path.read_text()) != run:
        raise ValueError("Evaluation settings changed; use a new output directory")
    write_json(run_path, run)
    predictions_path = args.output_dir / "predictions.jsonl"
    rows = [json.loads(line) for line in predictions_path.read_text().splitlines()] if predictions_path.exists() else []
    if len(rows) > len(examples) or any((r["subject"], r["index"], r["answer"]) != (e["subject"], e["index"], e["answer"]) for r, e in zip(rows, examples)):
        raise ValueError("Saved predictions do not match the frozen evaluation")
    if len(rows) < len(examples):
        tokenizer = AutoTokenizer.from_pretrained(args.adapter, padding_side="left")
        if tokenizer.pad_token_id is None:
            tokenizer.pad_token = tokenizer.eos_token
        token_ids = choice_token_ids(tokenizer)
        base = AutoModelForCausalLM.from_pretrained(run["base_model"], revision=run["base_revision"], dtype=dtype, device_map={"": device})
        model = PeftModel.from_pretrained(base, args.adapter).eval()
        with predictions_path.open("a") as output:
            for start in range(len(rows), len(examples), args.batch_size):
                items = examples[start:start + args.batch_size]
                texts = [tokenizer.apply_chat_template(e["messages"], tokenize=False, add_generation_prompt=True) for e in items]
                batch = tokenizer(texts, padding=True, add_special_tokens=False, return_tensors="pt").to(device)
                if batch.input_ids.shape[1] >= model.config.max_position_embeddings:
                    raise ValueError("Evaluation prompt exceeds model context length")
                predictions = {}
                for name in ("base", "sft"):
                    with model.disable_adapter() if name == "base" else nullcontext():
                        predictions[name] = predict(model, batch, token_ids, tokenizer)
                for offset, example in enumerate(items):
                    row = dict(example, base=predictions["base"][offset], sft=predictions["sft"][offset])
                    rows.append(row)
                    output.write(json.dumps(row, ensure_ascii=False) + "\n")
                    output.flush()
                print(f"Scored paired questions: {len(rows)}/{len(examples)}", flush=True)
    metrics = summarize(rows, args.seed)
    metrics["run"] = run
    write_json(args.metrics_file, metrics)
    print(json.dumps(metrics, indent=2), flush=True)


if __name__ == "__main__":
    main()
