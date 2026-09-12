"""
Turn scraped episodes into chat-format SFT pairs.

Every Sheldon line is a candidate target. Preceding lines (same scene) become
context: Sheldon -> assistant, everyone else -> user. Consecutive same-role
lines are merged. No speaker prefixes, because eval prompts won't have them.

Output: data/pairs_train.jsonl, data/pairs_heldout.jsonl
  {"messages":[{"role":"user",...},{"role":"assistant",...}], "meta":{...}}
"""

import json, re, random, argparse, pathlib
from collections import Counter

TARGET = "sheldon"

# lines that only make sense with picture / are pure filler
JUNK_RE = re.compile(
    r"^(hi|hello|hey|bye|bye-bye|what|what\?|yes|no|okay|ok|sure|great|thanks|"
    r"thank you|oh|uh|um|huh|right|fine|agreed|indeed|hmm|well)[.!?]?$", re.I)

DEICTIC_RE = re.compile(r"\b(this one|over there|like that|right here|these|those)\b", re.I)


def wc(s):
    return len(s.split())


def clean(s):
    s = re.sub(r"\s+", " ", s).strip()
    return s


def build(episodes, ctx_turns, min_target_words, min_prompt_words, max_target_words):
    pairs, reasons = [], Counter()

    for ep in episodes:
        lines = ep["lines"]
        for i, ln in enumerate(lines):
            if ln["speaker"].lower() != TARGET:
                continue
            target = clean(ln["text"])

            if wc(target) < min_target_words:
                reasons["target_too_short"] += 1; continue
            if wc(target) > max_target_words:
                reasons["target_too_long"] += 1; continue
            if JUNK_RE.match(target):
                reasons["target_junk"] += 1; continue
            if DEICTIC_RE.search(target):
                reasons["target_needs_visual"] += 1; continue

            # gather context from same scene
            ctx = []
            j = i - 1
            while j >= 0 and len(ctx) < ctx_turns and lines[j]["scene"] == ln["scene"]:
                ctx.append(lines[j]); j -= 1
            ctx.reverse()

            if not ctx:
                reasons["no_context"] += 1; continue
            if ctx[-1]["speaker"].lower() == TARGET:
                reasons["prev_is_sheldon"] += 1; continue

            # map to roles, merge consecutive
            msgs = []
            for c in ctx:
                role = "assistant" if c["speaker"].lower() == TARGET else "user"
                txt = clean(c["text"])
                if msgs and msgs[-1]["role"] == role:
                    msgs[-1]["content"] += " " + txt
                else:
                    msgs.append({"role": role, "content": txt})

            while msgs and msgs[0]["role"] != "user":
                msgs.pop(0)
            if not msgs:
                reasons["no_context"] += 1; continue

            user_words = sum(wc(m["content"]) for m in msgs if m["role"] == "user")
            if user_words < min_prompt_words:
                reasons["prompt_too_short"] += 1; continue

            msgs.append({"role": "assistant", "content": target})
            pairs.append({
                "messages": msgs,
                "meta": {"episode": ep["title"], "scene": ln["scene"],
                         "target_words": wc(target)},
            })
    return pairs, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/raw_episodes.json")
    ap.add_argument("--outdir", default="data")
    ap.add_argument("--ctx-turns", type=int, default=3)
    ap.add_argument("--min-target-words", type=int, default=15)
    ap.add_argument("--max-target-words", type=int, default=160)
    ap.add_argument("--min-prompt-words", type=int, default=6)
    ap.add_argument("--heldout-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    episodes = json.load(open(args.inp))
    pairs, reasons = build(episodes, args.ctx_turns, args.min_target_words,
                           args.min_prompt_words, args.max_target_words)

    # dedupe on target text
    seen, uniq = set(), []
    for p in pairs:
        k = p["messages"][-1]["content"].lower()
        if k in seen:
            reasons["duplicate"] += 1; continue
        seen.add(k); uniq.append(p)
    pairs = uniq

    # SPLIT BY EPISODE, not by row -- avoids near-duplicate scenes leaking
    eps = sorted({p["meta"]["episode"] for p in pairs})
    random.Random(args.seed).shuffle(eps)
    n_ho = max(1, int(len(eps) * args.heldout_frac))
    ho_eps = set(eps[:n_ho])

    train = [p for p in pairs if p["meta"]["episode"] not in ho_eps]
    heldout = [p for p in pairs if p["meta"]["episode"] in ho_eps]

    d = pathlib.Path(args.outdir); d.mkdir(parents=True, exist_ok=True)
    for name, rows in [("pairs_train", train), ("pairs_heldout", heldout)]:
        with open(d / f"{name}.jsonl", "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    print(f"kept {len(pairs)} pairs  (train {len(train)} / heldout {len(heldout)}"
          f" from {len(ho_eps)} held-out episodes)")
    print("dropped:")
    for k, v in reasons.most_common():
        print(f"  {k:22s} {v}")
    if pairs:
        avg = sum(p["meta"]["target_words"] for p in pairs) / len(pairs)
        print(f"avg Sheldon response length: {avg:.1f} words")


if __name__ == "__main__":
    main()
