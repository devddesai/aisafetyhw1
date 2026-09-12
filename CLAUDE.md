# CLAUDE.md

Course assignment: post-train Qwen2.5-3B-Instruct for (a) a fictional persona and
(b) a STEM capability, across three stages — SFT, RLAIF, RLVR.

Character: Sheldon Cooper. STEM domain: grade-school math (GSM8K).

**We are on Checkpoint 1 (SFT only). Due Sunday.**

Division of labour: this side of the team builds the **datasets**. The other side
writes `train_sft.py`, `eval_persona.py`, and `eval_gsm8k.py`.

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

Not started, and owned by the other side of the team: `train_sft.py`,
`eval_persona.py`, `eval_gsm8k.py`, and the frozen GSM8K eval subset.

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

### 4. Freeze the eval set — NOT DONE, owned by the eval side

Pick a fixed GSM8K test subset (200–500 problems), save the indices, and use that
identical set at every stage. The main deliverable is a plot across stages;
changing the eval set makes it meaningless.

Which problems to score is a modelling decision, so it belongs to whoever writes
`eval_gsm8k.py`. Worth settling before training starts: once stage-1 numbers
exist against one subset, switching subsets throws them away.

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

Known risk, accepted: transcripts are banter, eval prompts are questions, and
nothing in this dataset is question-shaped. Expect persona to be weakest exactly
when the model is asked a technical question. That is the finding this baseline
is designed to produce, not a defect.

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
train_sft.py           -> out/sft-lora/                   teammate
eval_persona.py        -> results/persona.json            teammate
eval_gsm8k.py          -> results/gsm8k.json              teammate
eval/gsm8k_subset.json -> frozen problem indices          teammate

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

## Loss masking (deferred — design retained)

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

2–3 epochs, lr 1e-4 to 2e-4. Train on assistant tokens only — only the final
assistant message in each example is a target; earlier turns are context.

Run the GSM8K eval at **every checkpoint**, not just at the end — that's how we
catch degradation while it's still fixable. If Sheldon is too faint, raise r to
32 before changing anything else. If output is degenerate or catchphrase-spammy,
reduce epochs first.

Note: `peft`, `trl`, and `sentence_transformers` are not yet installed.

## Metrics for Checkpoint 1

Persona (base vs. SFT, on held-out episodes):
- perplexity of held-out Sheldon references under each model — should drop
- embedding cosine similarity (sentence-transformers) between each model's
  output and the reference for the same prompt

Report both; they fail in different ways.

Capability (base vs. SFT, frozen GSM8K subset):
- accuracy
- **extraction success rate** — logged separately. If persona training buries
  the final answer mid-ramble, we need to know now, not in week 3.

Catchphrase frequency per response, from SFT onward. **Stage-zero baseline is
already measured: 1.22% of training targets contain one** (`roommate agreement`
25, `my spot` 12, `sarcasm` 9, `bazinga` 9, `hot beverage` 5). `validate_pairs.py`
prints this. Having the series from stage zero turns the "did the judge get
gamed" question into a plot instead of a guess.

**Open question for the eval scripts:** `results/` is currently gitignored,
because if `eval_persona.py` writes held-out reference dialogue beside the
scores, that is copyrighted text we cannot commit. Have the eval scripts emit
**scores only**, then we can un-ignore the directory and track the deliverable
numbers.

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
