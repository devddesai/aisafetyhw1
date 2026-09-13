"""Plot the saved aggregate evaluations; no model or transcript data is needed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np


COLORS = ("#356A91", "#C05B36")
MODELS = ("base", "sft")
ROOT = Path(__file__).resolve().parent


def style(usetex):
    # Shared typography, ticks, and lines from the requested plotting style.
    plt.rc("text", usetex=usetex)
    plt.rc("font", family="serif", size=18)
    plt.rc("axes", labelsize=18, titlesize=18)
    plt.rc("axes.spines", top=False, right=False)
    plt.rc("xtick", labelsize=15, top=False, direction="out")
    plt.rc("ytick", labelsize=15, right=False, direction="out")
    plt.rc("xtick.major", size=5, width=1)
    plt.rc("ytick.major", size=5, width=1)
    plt.rc("legend", fontsize=18, frameon=False)
    plt.rc("lines", linewidth=2, markersize=6, markeredgewidth=1)


def text(label):
    return label.replace("%", r"\%") if plt.rcParams["text.usetex"] else label


def bars(ax, values, title, ylabel, labels, upper):
    artists = ax.bar(("Base", "SFT"), values, color=COLORS, width=0.55)
    ax.bar_label(artists, labels=[text(label) for label in labels], padding=8, fontsize=16)
    ax.set(title=title, ylabel=text(ylabel), ylim=(0, upper))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9")
    ax.yaxis.set_major_locator(MaxNLocator(4))


def paired_change(ax, value, interval, title, xlabel, decimals):
    low, high = interval
    span = max(high, value, 0) - min(low, value, 0)
    margin = max(span * 0.25, 0.001)
    ax.axvline(0, ymin=0.3, ymax=0.65, color="0.5", linestyle="--", linewidth=1)
    ax.plot([low, high], [0, 0], color=COLORS[1])
    ax.plot([low, high], [0, 0], "|", color=COLORS[1], markersize=14)
    ax.plot(value, 0, "o", color=COLORS[1], markersize=9)
    ax.set(title=title, xlabel=xlabel, ylim=(-1, 1), yticks=[],
           xlim=(min(low, value, 0) - margin, max(high, value, 0) + margin))
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(4))
    ax.text(0.5, 0.76, f"{value:+.{decimals}f}", transform=ax.transAxes,
            ha="center", fontsize=22, color=COLORS[1])
    ax.text(0.5, 0.17, text(f"95% CI [{low:+.{decimals}f}, {high:+.{decimals}f}]"),
            transform=ax.transAxes, ha="center", fontsize=15)


def persona_figure(metrics):
    fig, axes = plt.subplots(2, 2, figsize=(12, 9.5))
    fig.subplots_adjust(left=0.10, right=0.97, bottom=0.20, top=0.82, hspace=0.65, wspace=0.4)
    fig.suptitle("Persona evaluation: reference similarity and reply length", y=0.98, fontsize=22)
    fig.text(0.5, 0.92, f"{metrics['examples']:,} held-out replies from {metrics['episodes']} episodes",
             ha="center", fontsize=16)
    similarity = [metrics[name]["mean_cosine_similarity"] for name in MODELS]
    bars(axes[0, 0], similarity, "Reference similarity", "Mean cosine similarity",
         [f"{value:.4f}" for value in similarity], max(0.25, max(similarity) * 1.3))
    lengths = [metrics[name]["mean_generated_tokens"] for name in MODELS]
    bars(axes[0, 1], lengths, "Reply length", "Mean generated tokens",
         [f"{value:.1f}" for value in lengths], max(lengths) * 1.3)
    paired_change(axes[1, 0], metrics["mean_paired_difference"],
                  metrics["paired_difference_episode_bootstrap_95_ci"],
                  "Paired similarity change", "SFT minus base\n(cosine similarity)", 4)
    hits = [metrics[name]["generation_limit_hits"] for name in MODELS]
    rates = [100 * count / metrics["examples"] for count in hits]
    limit = metrics["run"]["decoding"]["max_new_tokens"]
    bars(axes[1, 1], rates, f"Replies reaching {limit} tokens", "Share of replies (%)",
         [f"{rate:.1f}% ({count}/{metrics['examples']})" for rate, count in zip(rates, hits)],
         max(1, max(rates) * 1.4))
    fig.text(0.5, 0.045, "Similarity measures agreement with one reference, not persona style.\n"
             "The paired interval resamples whole episodes; shorter replies are a separate diagnostic.",
             ha="center", fontsize=14, linespacing=1.5)
    return fig


def physics_figure(metrics):
    fig, (accuracy, change) = plt.subplots(1, 2, figsize=(14, 7), gridspec_kw={"width_ratios": [1.5, 1]})
    fig.subplots_adjust(left=0.17, right=0.97, bottom=0.30, top=0.77, wspace=0.3)
    fig.suptitle("Physics evaluation: base vs. persona SFT", y=0.98, fontsize=22)
    fig.text(0.5, 0.90, f"{metrics['questions']} MMLU questions | Five-shot chat | A/B/C/D selection",
             ha="center", fontsize=16)
    subjects = [("high_school_physics", "High-school physics"), ("college_physics", "College physics"),
                ("conceptual_physics", "Conceptual physics")]
    rows = [metrics["by_subject"][key] for key, _ in subjects] + [metrics["overall"]]
    names = [label for _, label in subjects] + ["Overall"]
    y = np.arange(len(rows))
    for offset, model, color in zip((-0.18, 0.18), MODELS, COLORS):
        values = [100 * row[model]["accuracy"] for row in rows]
        artists = accuracy.barh(y + offset, values, height=0.3, color=color, label=model.capitalize() if model == "base" else "SFT")
        accuracy.bar_label(artists, labels=[text(f"{value:.2f}%") for value in values], padding=5, fontsize=15)
    accuracy.set(yticks=y, yticklabels=[f"{name}\n(n = {row['questions']})" for name, row in zip(names, rows)],
                 xlim=(0, 100), xlabel=text("Accuracy (%)"))
    accuracy.invert_yaxis()
    accuracy.axhline(2.5, color="0.85", linewidth=1)
    accuracy.grid(axis="x", color="0.9")
    accuracy.set_axisbelow(True)
    accuracy.legend(loc="lower left", bbox_to_anchor=(0, 1.01), ncol=2, borderaxespad=0)
    interval = [100 * value for value in metrics["paired_accuracy_difference_95_ci"]]
    paired_change(change, 100 * metrics["overall"]["accuracy_difference"], interval,
                  "Paired overall change", "SFT minus base\n(percentage points)", 2)
    includes_zero = interval[0] <= 0 <= interval[1]
    interpretation = ("The interval includes zero: no clear overall change." if includes_zero
                      else "The interval excludes zero under this evaluation protocol.")
    fig.text(0.5, 0.105, f"{metrics['base_wrong_sft_correct']} wrong-to-correct changes; "
             f"{metrics['base_correct_sft_wrong']} correct-to-wrong changes. {interpretation}", ha="center", fontsize=14)
    fig.text(0.5, 0.055, text("95% paired bootstrap CI resamples questions within each subject. "
             "Overall accuracy is weighted by question count."), ha="center", fontsize=14)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, help="Defaults to RESULTS_DIR/plots")
    parser.add_argument("--usetex", action="store_true", help="Use LaTeX for text (requires a working LaTeX installation)")
    args = parser.parse_args()
    style(args.usetex)
    output_dir = args.output_dir or args.results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, name, draw in (("persona.json", "persona_summary", persona_figure),
                                 ("mmlu_physics.json", "physics_accuracy", physics_figure)):
        metrics = json.loads((args.results_dir / filename).read_text())
        fig = draw(metrics)
        for extension in ("png", "pdf"):
            path = output_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
