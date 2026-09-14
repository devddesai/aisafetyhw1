# CLAUDE.md

Course assignment: post-train Qwen2.5-3B-Instruct for (a) a fictional persona and
(b) a STEM capability, across three stages — SFT, RLAIF, RLVR.

Character: Sheldon Cooper. STEM benchmark: MMLU physics, replacing the original
grade-school math (GSM8K) plan.

**We are on Checkpoint 1 (SFT only). Due Sunday.**

The original handoff covered datasets. This checkout now also includes
`train_sft.py`, `eval_persona.py`, and `eval_mmlu.py`.

---

## Status — what actually exists

Done:

- `scrape_transcripts.py` — run against the live site. 231 episodes, 51,488
  lines, 11,608 of them Sheldon's. Verified: all 231 episodes unique, per-season
  counts match the real broadcast run (S1 has 17 because of the writers' strike).
- `build_pairs.py` — run at `--seed 0 --min-target-words 10`. Yields **4,899
  train / 511 held-out pairs** from 208 / 23 episodes. Median target 18 words.
- `validate_pairs.py` — passing. Asserts the invariants the metrics rely on.
- `README.md` — written for teammate handoff.

Implemented: `train_sft.py`, with completion-only loss, LoRA, epoch checkpoints,
run metadata, and a locally verified optimizer/save/reload path using a tiny
Qwen model. Dependencies are pinned in `requirements.txt`.

Completed: one full SFT epoch on an A40, 307 updates over 4,899 examples in
402.5 seconds; saved adapter reloaded and checksummed local backup verified.

Implemented: `eval_persona.py` for paired base/SFT reference embedding similarity
on all 511 held-out examples. Completed mean similarity: base 0.186364, SFT
0.204125; paired gain 0.017761, episode-bootstrap 95% interval [0.006107,
0.028523]. See `WORKLOG.md` for output checks and metric limitations.

Implemented: `eval_mmlu.py`, a five-shot chat evaluation of all 488 test questions
in high-school, college, and conceptual physics. The model selects A/B/C/D by
next-token probability. The dataset revision and indices are frozen in
`eval/mmlu_physics.json`. The full H100 run completed: base 266/488 (54.51%), SFT
274/488 (56.15%); paired difference +1.64 percentage points, 95% stratified
bootstrap interval [-1.43, +4.51]. Results do not establish a clear overall change.

GSM8K evaluation was replaced by physics and was not run.

Deliberately deferred: generated conversation data, masked STEM data,
`style_guide.md`. See "Scope decision" below.

---

## Hard constraints — do not violate these

### 1. Do NOT train the model to solve math

The TA was explicit: no SFT on STEM problem/solution pairs. Capability gains are
RLVR's job in week 3. If SFT teaches math by imitation we (a) steal week 3's
result, (b) inherit a known pathology where the model learns "answers are two
lines" from short-answer training data and generalizes badly.

Currently satisfied trivially — there is no STEM data in the training set at all.
If the masked-STEM slice is added later, the masking design in "Loss masking
(deferred)" is how it stays satisfied.

This constraint also shaped a filter choice: `--min-target-words 10` rather than
6, because a dataset of clipped one-liners produces the same two-line-answer
pathology through a different door, and would show up in the extraction success
rate.

### 2. Do not commit transcript data

Show dialogue is copyrighted. `data/` is gitignored and must stay that way — the
ignore rule was written before the first scrape, so no transcript text has ever
been stageable.

Do not "commit it now and delete it before publishing". Git keeps every blob ever
committed; removing the file later needs `git filter-repo` plus a force-push that
rewrites every hash, and existing clones and forks keep the data regardless.
Teammates rebuild locally in ~5 minutes instead (README has the commands). If a
backup is wanted, share `raw_episodes.json` out-of-band.

Never paste transcript excerpts into the README, commit messages, or the report.

### 3. Do not use an LLM judge yet

TA guidance: introduce the judge as late as possible. Judge scores drift across
model versions, which breaks cross-stage comparison. Checkpoint 1 uses only
text-distance metrics (perplexity, embedding similarity). The judge arrives in
week 2.

### 4. Freeze the eval set — DONE for MMLU physics

Use the complete frozen physics test set at each stage: 151 high-school physics,
102 college physics, and 235 conceptual physics questions. Five demonstrations
per subject come from its development split. Keep the chat prompt, constrained
letter scoring, data revision, and question indices identical across stages.

---

## Scope decision — transcript-only baseline

CLAUDE.md originally specified three data sources. We are training on **transcript
pairs only** for now, to find out what real dialogue alone achieves before adding
anything generated. The generated slices then become a measurable delta rather
than a confound.

| Source | Status |
|---|---|
| Transcript pairs | **built** — 4,899 / 511 |
| Generated general conversation | deferred |
| Masked STEM examples | deferred |

Known risk: transcript pairs contain questions but mostly represent sitcom
conversation, not worked math problems. Held-out transcript evaluation tests
that conversational setting; MMLU physics tests a different input distribution.
Whether persona or capability degrades there remains an empirical question.

If the baseline is too weak, the generated slices are the first thing to add —
they are additive and need no change to what already exists. Note that the
loss-masking write-up is a README deliverable in the assignment brief, so adding
the STEM slice back also restores that.

---

## Pipeline

```
scrape_transcripts.py  -> data/raw_episodes.json          DONE
build_pairs.py         -> data/pairs_{train,heldout}.jsonl DONE
validate_pairs.py      -> pass/fail + stats               DONE
train_sft.py           -> out/sft-lora/                   implemented
eval_persona.py        -> results/persona.json            implemented
eval_mmlu.py          -> results/mmlu_physics.json       implemented
eval/mmlu_physics.json -> frozen physics revision/indices DONE

gen_stem.py            -> data/stem_masked.jsonl          DEFERRED
gen_convo.py           -> data/convo.jsonl                DEFERRED
```

Source: https://bigbangtrans.wordpress.com/ — sidebar on any page lists all
episodes. **Episode URLs are inconsistent** (S02E08 sits at a `series-1-...`
path, S08E05 at a `series-7-...` path). Scrape the sidebar links; never
construct URLs from episode numbers. The scraper already does this correctly.

## Data conventions

Chat format throughout — `{"messages":[{"role":"user",...},{"role":"assistant",...}]}`.
No speaker-name prefixes in content; eval prompts won't have them.
`validate_pairs.py` enforces this against a cast-name list.

**Split by episode, not by row.** Sitcom scenes repeat beats, so a random row
split leaks near-duplicates into held-out and inflates the persona metric.
Keep `--seed 0` so everyone's split matches.

**Strip the persona system prompt from training data.** Generate with it, train
without it. The voice goes in the weights, not the context — otherwise the
comparison against base Qwen is unfair.

**Only exact `Sheldon` attribution is used.** `Past Sheldon` (66 lines) and
`Sheldon-bot` (39) are excluded as different voice registers; total loss to name
variants is under 1%.

## Loss masking (deferred — earlier math design retained)

The math-specific design below predates the switch to physics evaluation.
It is not implemented and does not describe the current MMLU scoring protocol.

Not built. Keep this section: it is the right design if the STEM slice is added,
and it is the most interesting choice in the checkpoint.

Write STEM responses with the persona woven throughout — *not* intro / math /
outro in rigid blocks, which would teach the model to drop character to compute
and pick it back up. Sheldon narrates while reasoning.

Mark computational spans with `<<S>>` ... `<<E>>`. Spans may be multiple and
non-contiguous. The preprocessing step:

1. finds every marked span by character offset
2. deletes the markers from the text
3. tokenizes with `return_offsets_mapping=True`
4. sets `labels[i] = -100` for tokens falling inside any span

Masked tokens remain in the context — the model conditions on the solution when
predicting the following text, it just gets no gradient to produce it.

Also mask user turns as usual (completion-only training).

Keep the final answer line (`\boxed{...}`) **outside** the mask. That's format,
not capability, and week 3's verifier depends on it.

Sanity check: decode the unmasked positions of a few examples and confirm you
see voice and no arithmetic.

This separation is a heuristic, not a guarantee — some signal leaks through
hidden states. Say so in the README rather than overclaiming. The GSM8K
before/after number is the empirical check.

## Generation (deferred)

Guard against model collapse — a small model trained on one large model's output
at default temperature learns that model's tics, not the character. Use
temperature ~0.9–1.0, vary prompt scaffolding, show the generator previously
generated examples and ask for something different, and split generation across
two different large models if available.

All generation would be driven by `style_guide.md` (not written). That same guide
becomes the week-2 judge rubric — generation criteria and scoring criteria must
agree or cross-stage numbers stop meaning anything.

Environment note: no API key is currently configured. `openai` 1.53 is installed;
the `anthropic` SDK is not.

## Training

LoRA via PEFT + TRL `SFTTrainer`. Starting point:

```python
LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
           target_modules=["q_proj","k_proj","v_proj","o_proj",
                           "gate_proj","up_proj","down_proj"],
           task_type="CAUSAL_LM")
```

The completed baseline used one epoch and learning rate 1e-4. Only the final
assistant message in each example is a target; earlier turns are context.

Run the frozen physics eval at **every checkpoint**, not just at the end — that's how we
catch degradation while it's still fixable. If Sheldon is too faint, raise r to
32 before changing anything else. If output is degenerate or catchphrase-spammy,
reduce epochs first.

Training and evaluation dependencies are pinned in `requirements.txt`.
Sentence-transformers supplies the embedding model for `eval_persona.py`.

## Metrics for Checkpoint 1

Persona (base vs. SFT, on held-out episodes):
- perplexity of held-out Sheldon references under each model — should drop
- embedding cosine similarity (sentence-transformers) between each model's
  output and the reference for the same prompt

Report both; they fail in different ways.

Capability (base vs. SFT, frozen MMLU physics test set):
- accuracy overall and per subject, with paired before/after comparisons
- total probability mass assigned to the four answer letters, as a format
  diagnostic; constrained choice scoring does not use free-form extraction

Catchphrase frequency per response is a planned diagnostic. The validator
reports both phrase matches and the fraction of responses containing at least
one catchphrase, so a response with several phrases counts once in the percentage.

`results/persona.json` is scores-only and can be tracked. Prompts, reference
dialogue, and generated outputs stay in ignored `out/persona/`.

## Deliverables

- Repo with training code
- README written by us, human-readable, not exhaustive
- Model weights accessible somehow (LFS or HF link — not necessarily in-repo)
- Numerical before/after on both persona and capability
- A held-out metric

README should include a short paragraph explaining the style-gradients-without-
capability-gradients decision and how it was implemented. **Currently this is
documented as a deferred design rather than an implementation** — if the brief
requires it built, add the STEM slice back.

## Working style

- Explain reasoning; this is a learning assignment, not a code-delivery task.
- **Explain in plain English.** Lead with what something is and why it matters in
  ordinary words, then give numbers. Dense tables, metric jargon, and bare
  `file:line` references cost comprehension.
- Prefer small testable scripts over one large pipeline. Keep the codebase
  concise and easy for a human to read.
- Comments should say *why*, not restate the line.
- When a design decision is ambiguous, ask rather than assume.
- Don't add dependencies without saying why.

## Repo conventions

- **No AI-attribution trailers in commits.** No `Co-Authored-By: Claude`, no
  "Generated with Claude Code". Write the message and stop.
- `data/` and `.claude/settings.local.json` stay gitignored.
- The system `git` at `/usr/bin/git` is a broken Xcode shim on this machine. Use
  `/Library/Developer/CommandLineTools/usr/bin/git`, or fix it permanently with
  `sudo xcode-select --switch /Library/Developer/CommandLineTools`.
