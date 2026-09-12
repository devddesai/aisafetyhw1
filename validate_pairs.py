"""
Check that the built pairs are actually trainable, and fail loudly if not.

Run this after build_pairs.py. Every check here corresponds to a way the dataset
could look fine but quietly ruin a metric later:

  - a leaked "Sheldon:" prefix trains the model to emit speaker labels that the
    eval prompts will never contain
  - a bad role order breaks the chat template the trainer applies
  - an episode appearing in BOTH splits inflates the held-out score, which is the
    one number in this checkpoint that is supposed to be honest

Usage: python3 validate_pairs.py
"""

import json, re, argparse, sys
from collections import Counter

# "Sheldon:" / "Penny (annoyed):" at the start of a message -- a scraper leak.
# The name must be a real character: ordinary dialogue ("Magnets: ...") has the
# same shape, and flagging that is a false alarm rather than a bug.
SPEAKER_PREFIX = re.compile(r"^([A-Z][A-Za-z.'\- ]{0,28}?)\s*(?:\([^)]{0,40}\))?\s*:\s")

CAST = {"sheldon", "leonard", "penny", "howard", "raj", "amy", "bernadette",
        "stuart", "priya", "emily", "zack", "arthur", "beverley", "leslie",
        "kripke", "wil wheaton", "mrs cooper", "mrs wolowitz"}


def leaked_speaker(text):
    """True only if the message opens with an actual character name + colon."""
    m = SPEAKER_PREFIX.match(text)
    return bool(m and m.group(1).strip().lower() in CAST)

# Tracked from stage zero so the final report can plot catchphrase drift rather
# than guess at it. Substring match, lowercased.
CATCHPHRASES = ["bazinga", "my spot", "knock, knock, knock",
                "roommate agreement", "sarcasm", "hot beverage"]


def load(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def check_shape(rows, split, errors):
    """Every row must be a clean user-first, assistant-last alternating chat."""
    for i, row in enumerate(rows):
        where = f"{split}[{i}]"
        msgs = row.get("messages", [])

        if len(msgs) < 2:
            errors.append(f"{where}: only {len(msgs)} messages")
            continue
        if msgs[0]["role"] != "user":
            errors.append(f"{where}: starts with {msgs[0]['role']}, expected user")
        if msgs[-1]["role"] != "assistant":
            errors.append(f"{where}: ends with {msgs[-1]['role']}, expected assistant")

        for j, m in enumerate(msgs):
            if not m.get("content", "").strip():
                errors.append(f"{where}: message {j} is empty")
            if leaked_speaker(m.get("content", "")):
                errors.append(f"{where}: message {j} has a speaker-name prefix")
            # alternating roles -- the chat template assumes strict u/a/u/a
            if j > 0 and m["role"] == msgs[j - 1]["role"]:
                errors.append(f"{where}: two {m['role']} messages in a row")


def check_split_isolation(train, heldout, errors):
    """The held-out metric is only meaningful if the splits share nothing."""
    train_eps = {r["meta"]["episode"] for r in train}
    ho_eps = {r["meta"]["episode"] for r in heldout}
    both = train_eps & ho_eps
    if both:
        errors.append(f"{len(both)} episode(s) in BOTH splits: {sorted(both)[:3]}")

    train_targets = {r["messages"][-1]["content"].lower() for r in train}
    leaked = [r for r in heldout if r["messages"][-1]["content"].lower() in train_targets]
    if leaked:
        errors.append(f"{len(leaked)} held-out target(s) appear verbatim in train")

    return len(train_eps), len(ho_eps)


def describe(rows, split):
    lengths = sorted(len(r["messages"][-1]["content"].split()) for r in rows)
    n = len(lengths)
    pct = lambda p: lengths[int(n * p)]
    print(f"\n{split}: {n} examples")
    print(f"  target length   min {lengths[0]}  p25 {pct(.25)}  median {pct(.5)}"
          f"  p75 {pct(.75)}  max {lengths[-1]}")
    print(f"  turns per example: {dict(sorted(Counter(len(r['messages']) for r in rows).items()))}")

    hits = Counter()
    for r in rows:
        text = r["messages"][-1]["content"].lower()
        for phrase in CATCHPHRASES:
            if phrase in text:
                hits[phrase] += 1
    total = sum(hits.values())
    print(f"  catchphrases: {total} in {n} responses ({100*total/n:.2f}% of targets)")
    for phrase, c in hits.most_common():
        print(f"      {phrase:20s} {c}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", default="data/pairs_train.jsonl")
    ap.add_argument("--heldout", default="data/pairs_heldout.jsonl")
    args = ap.parse_args()

    train, heldout = load(args.train), load(args.heldout)
    errors = []

    check_shape(train, "train", errors)
    check_shape(heldout, "heldout", errors)
    n_train_eps, n_ho_eps = check_split_isolation(train, heldout, errors)

    describe(train, "train")
    describe(heldout, "heldout")
    print(f"\nepisodes: {n_train_eps} train / {n_ho_eps} held out")

    if errors:
        print(f"\nFAILED -- {len(errors)} problem(s):", file=sys.stderr)
        for e in errors[:20]:
            print(f"  {e}", file=sys.stderr)
        if len(errors) > 20:
            print(f"  ... and {len(errors)-20} more", file=sys.stderr)
        sys.exit(1)

    print("\nall checks passed")


if __name__ == "__main__":
    main()
