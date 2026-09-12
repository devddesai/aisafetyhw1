# Sheldon Cooper persona SFT — Checkpoint 1 data

Post-training Qwen2.5-3B-Instruct to adopt a fictional persona, then measuring
what that does to its grade-school math ability. This repo holds the **dataset
side** of Checkpoint 1: everything that produces training data, plus the frozen
evaluation set that every later stage must score against.

Training code (`train_sft.py`) and the eval scripts live with the other half of
the team.

## Quick start

The transcripts are copyrighted TV dialogue and are **not in this repo**. You
rebuild them locally, which takes about five minutes, once:

```bash
pip install requests beautifulsoup4

python3 scrape_transcripts.py                          # ~5 min, 231 episodes
python3 build_pairs.py --seed 0 --min-target-words 10  # seconds
python3 validate_pairs.py                              # should end "all checks passed"
```

That produces `data/pairs_train.jsonl` and `data/pairs_heldout.jsonl`. Keep
`--seed 0` — it fixes which episodes are held out, so your split matches
everyone else's exactly.

## What you get

| | examples | episodes |
|---|---|---|
| train | 4,899 | 208 |
| held out | 511 | 23 |

Median Sheldon response is 18 words. Source material is 231 episodes, 51,488
lines of dialogue, 11,608 of them his.

## Data format

One JSON object per line. Each line is one independent training example:

```json
{"messages": [
   {"role": "user",      "content": "<something another character says>"},
   {"role": "assistant", "content": "<Sheldon's reply>"},
   {"role": "user",      "content": "<the other character again>"},
   {"role": "assistant", "content": "<the line we train on>"}
 ],
 "meta": {"episode": "...", "scene": 4, "target_words": 14}}
```

Everyone who isn't Sheldon becomes `user`; Sheldon becomes `assistant`.
Consecutive lines from the same side are merged, because the model has no notion
of separate speakers. **Only the final `assistant` message is a training
target** — earlier turns are context the model reads but isn't graded on.

About a third of examples have just two messages, where Sheldon spoke early in a
scene and there wasn't much before him to use. Context never crosses a scene
boundary, since dialogue from another scene isn't really context.

`meta` is bookkeeping for us and is not fed to the model.

## Decisions worth knowing about

**The split is by episode, not by row.** Sitcoms reuse jokes and beats, so
splitting individual lines at random would put near-copies on both sides and make
the held-out score look better than it is. Splitting whole episodes avoids that.

**Minimum response length is 10 words.** Sheldon's very short lines are 62% of
his dialogue but make poor training targets — a dataset of one-liners teaches the
model to answer everything in one clipped sentence, which would later hurt its
ability to produce a full worked answer with the final result at the end. Ten
words keeps the mean at a substantive 22 while retaining ~50% more data than the
original 15-word floor.

**Only lines attributed exactly to "Sheldon" are used.** The transcripts also
contain `Past Sheldon` (66 lines) and `Sheldon-bot` (39) — deliberately excluded,
since those are different voice registers. Total loss to name variants is under
1%.

**No system prompt in the training data.** The persona is supposed to end up in
the weights, not in the context window; putting it in the prompt would make the
comparison against base Qwen unfair.

## What's deliberately not here yet

CLAUDE.md describes two generated data sources — general conversation in the
persona's voice, and STEM examples with the arithmetic masked out of the loss.
**Both are deferred on purpose.** We're training on real transcript dialogue
first to find out what that alone achieves, so the generated data can be measured
as a delta against a real baseline rather than bundled in from the start.

Expect the weak spot to be technical questions: transcripts are banter, while the
eval prompts are questions, and nothing in this dataset looks like a question and
answer. That's the thing this baseline is meant to reveal.

## Known limitations

- The filter that drops visually-dependent lines (ones that only make sense with
  a picture) matches a short hardcoded list of phrases, so it under-catches. Some
  lines referencing unseen objects survive into the data and become unanswerable
  prompts at eval time.
- Pairs are built from adjacent dialogue, so a reply responding to physical
  action rather than to the previous line will look like a non-sequitur.
- The GSM8K eval subset is **not** pinned here — that is the eval side's call.
  It needs freezing before stage-1 numbers exist, or the cross-stage plot breaks.
- `results/` is currently gitignored. The metric files are a deliverable, but if
  the eval scripts write held-out reference dialogue alongside the scores, that's
  copyrighted text we can't commit. Have the eval scripts emit **scores only**,
  then we can track that directory.

## Files

```
scrape_transcripts.py   transcripts -> data/raw_episodes.json
build_pairs.py          episodes -> train/heldout chat pairs
validate_pairs.py       fails loudly if the pairs are unusable
data/                   gitignored, rebuild locally
```

Transcript source: https://bigbangtrans.wordpress.com/ — a fan transcription
site. Dialogue is copyrighted by its owners and is used here only locally, for a
course assignment.
