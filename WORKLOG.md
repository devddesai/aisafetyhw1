# Checkpoint 1 worklog

## September 12, 2026

Rebuilt and validated the teammate's transcript dataset: 231 episodes, 4,899
training examples from 208 episodes, and 511 held-out examples from 23 episodes.
Used seed 0 and a minimum target length of 10 words.

Implemented completion-only LoRA training. Checked all training examples for
chat-template prefix consistency, end-of-turn tokens, length limits, and correct
loss masks. A local tiny-model integration run performed an optimizer update,
saved the adapter, and reloaded it successfully.

Trained Qwen2.5-3B-Instruct for one epoch on an A40 using rank 16, alpha 32,
dropout 0.05, learning rate 0.0001, batch size 8, and accumulation of 2. Completed
307 optimizer updates in 402.5 seconds, with training loss 2.6291138295.
Run metadata and exact revisions are in the saved adapter's `run.json`.

An initial batch-size-1 attempt was stopped after a few updates because GPU
utilization was low. Its metadata and log are retained in `out/sft-batch1/`.
The completed run restarted from the base model with the same effective batch
size of 16. Software installation was moved from slow network storage to the
pod's local container disk; datasets and model outputs stayed persistent.

Reloaded the completed adapter for three generation checks. Replies were short;
one was off-topic. These checks establish loadability, not persona quality.
Copied the adapter, tokenizer, optimizer checkpoint, metadata, and sample
outputs locally and verified the 21 artifact checksums. Training and reload logs
were also copied locally. The generated model card was replaced with accurate
loading instructions and training details.

Implemented paired persona evaluation using the full preceding conversation,
greedy decoding, and a 256-token generation limit for both models. Cosine
similarity compares each generated response to its matching held-out reference
using all-mpnet-base-v2. Scores include the paired mean difference, an episode
bootstrap confidence interval, response lengths, and truncation diagnostics.
Completed all 511 paired generations and embedding comparisons. Base mean cosine
similarity was 0.186364; SFT was 0.204125. The paired gain was 0.017761, with a
95% episode-bootstrap interval of [0.006107, 0.028523]. SFT scored higher on
56.36% of examples. This is a modest semantic-reference improvement, not a direct
persona-style measurement.

Base replies averaged 102.33 generated tokens; SFT replies averaged 25.46.
The generation limit was reached by 38 base replies and 12 SFT replies. No
responses were empty, and no reference or generated reply exceeded the embedding
model's 384-token limit. Thus embedding truncation did not affect this run,
although generation caps did affect some replies.

Verified the local copies of all five evaluation files against remote checksums,
all 511 prompt/reference mappings against the held-out data, score aggregation,
and the dataset and adapter hashes. Ten qualitative spot checks were chosen with
NumPy's default random generator, seed 0, sampling without replacement. Indices
were 8, 20, 38, 89, 136, 155, 257, 320, 415, and 427. The SFT replies at 20 and 89
showed repetition while still beating the base reference-similarity score. This
illustrates a metric limitation; the spot check is not a formal style score.

An exploratory answer-length analysis found mean whitespace word counts of
84.53 for base replies, 18.62 for SFT replies, and 21.62 for references. Within
each model, Pearson correlations between generated-token count and similarity
were -0.0119 (base) and -0.0609 (SFT); Spearman correlations were 0.0108 and
0.0557. These near-zero associations do not establish that the before/after
gain is independent of answer length. The diagnostic values are saved locally
in `out/persona/length_analysis.json`.

After verifying the training and evaluation backups, stopped and terminated the
new training pod, including its volume, to avoid ongoing storage charges. The
older homework-zero pod was left untouched. Observed spending during this work
was approximately $0.36, including setup; this is not a finalized per-run invoice.

At this point, STEM evaluation and checkpoint assembly were outstanding.
Held-out reference perplexity remains a planned complementary metric, not part
of the current persona script.

## Physics evaluation, September 12, 2026

The STEM benchmark changed from the planned GSM8K evaluation to MMLU physics.
No GSM8K subset was created or evaluated.

Implemented `eval_mmlu.py` and froze the official `cais/mmlu` dataset revision
and indices in `eval/mmlu_physics.json`. Used all 488 test questions: 151
high-school physics, 102 college physics, and 235 conceptual physics. Each
subject supplies its five development examples as labeled chat demonstrations.
The final test label is withheld. Both models receive the same answer-letter
instruction and prompts; the prediction is the highest-probability A/B/C/D
next token. There is no free-form generation or chain-of-thought scoring.

Validated that changing every test label leaves every prompt unchanged, that
there is no exact development/test question overlap across the three subjects,
and that A/B/C/D are distinct single tokens. A tiny Qwen integration check
verified padded versus individual scoring and baseline restoration with the
adapter disabled. Known-answer fixtures verified accuracy and paired change
counts. Syntax, comment-free source, and whitespace checks passed.

Downloaded the exact original model revision to the Mac. A local IPv6 connection
problem was worked around by selecting IPv4 for the download process; this
machine-specific workaround was not added to the evaluation script. The weights
are cached outside the repository in the normal Hugging Face cache.

A preliminary MPS run completed 56 question pairs before being stopped when the
evaluation moved to a faster GPU. Its outputs and log are retained separately in
`out/mmlu_physics_mps/`; those partial results are not included in the final
comparison. Restarted the complete evaluation on a RunPod H100, with batch size
16 and bfloat16. Exact package versions, hardware backend, base revision, and
hashes of the adapter, data, manifest, and script are recorded with the run.

Completed scores:

| Subject | Base correct | SFT correct | Questions |
|---|---:|---:|---:|
| High-school physics | 67 | 71 | 151 |
| College physics | 49 | 48 | 102 |
| Conceptual physics | 150 | 155 | 235 |
| Overall | 266 | 274 | 488 |

Overall accuracy rose from 54.5082% to 56.1475%, a gain of 1.6393 percentage
points. The 95% paired bootstrap interval, resampling questions within subjects,
was [-1.4344, +4.5082] percentage points. There were 35 wrong-to-correct changes
and 27 correct-to-wrong changes. The interval includes zero: this test does not
establish a clear overall improvement or degradation. All unrestricted top
next-token predictions were also A/B/C/D, and mean answer-letter mass was
0.999999 for base and 0.998933 for SFT.

Copied the four evaluation files locally and verified their remote checksums.
Verified all 488 prompt/question/label mappings, argmax choices, normalized
probabilities, subject and overall metrics, bootstrap results, and code/data/model
hashes. The H100 pod is stopped and has no persistent volume; RunPod shows zero
ongoing compute and storage cost for this pod. Its configuration is retained.
Observed spending for the H100 run was approximately $0.12, including setup;
this is not a finalized invoice.

This is a five-shot chat adaptation of MMLU, not the original plain-text
completion protocol, so leaderboard comparisons require care. It measures
physics answer selection rather than explanation quality. Pretraining exposure
to MMLU cannot be ruled out. The remaining work includes assembling the checkpoint
submission and deciding whether to add held-out persona perplexity.

## Checkpoint review, September 12, 2026

The available evidence does not show a failed training run or require rerunning
the baseline. The persona similarity gain has an episode-bootstrap interval above
zero, but measures semantic agreement with a reference, not character style.
Physics gains eight correct answers out of 488, with an interval spanning zero;
transcript-only SFT was not designed to teach physics.

Downloaded the published Checkpoint 1 adapter archive and verified its archive
checksum and all 11 per-file checksums. The weights match both evaluation hashes;
the configuration also matches the physics evaluation. All 504 adapter tensors
are finite, and all 252 LoRA B matrices are nonzero across the 36 layers and seven
target modules. This rules out an untouched, all-zero adapter, but does not prove
that its learned behavior is useful. The released metadata records a fresh,
one-epoch run; its batch settings imply the reported 307 optimizer updates.

Rebuilt all 231 episodes and validated 4,899 training / 511 held-out examples.
Both dataset hashes exactly match the published run. Checked every example's
prompt prefix, target boundary, EOS, and length with the released tokenizer and
the training script's preparation function. Training sequences reach at most
275 tokens, below the 1,024-token limit. Their 539,522 total tokens exactly match
the training log; only 150,108 are supervised suffix tokens, including end-of-turn
markers. The rest supply context. The template inserts Qwen's generic assistant
system message consistently in training and persona evaluation.

The data teaches brief sitcom continuations: at most three preceding raw lines,
with other speakers merged into one user role, and mean training targets of
21.84 words. SFT outputs average 18.62 words versus 21.62 for held-out references
in the original length analysis. Learning shorter replies is consistent with
that task; generic, off-topic, and repetitive replies remain a quality concern.
The released three-response sample supports that concern but is not a formal
style evaluation. These are possible data/task limitations, not proven causes.

The release lacks per-step loss, learning-rate, and gradient-norm logs, and the
training script has no validation-loss evaluation. Final training loss alone
cannot distinguish underfitting from overfitting. Before another training run,
recover those logs and the private paired outputs; review a fixed blinded sample
for relevance, repetition, and recognizable style, and measure completion-only
held-out reference NLL/perplexity for base and SFT. Choose later data or training
changes using development episodes reserved from the training split. Repeating
the same deterministic evaluation will not provide independent evidence.

Added reproducible PNG/PDF result plots and small validation, failed-scrape, and
persona-resume configuration checks. Six offline regression tests pass. The
changes do not alter the successful baseline's training setup or saved scores;
no training or model evaluation was rerun. Raw dialogue and audit artifacts remain
in ignored local directories.
