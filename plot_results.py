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


def figure(title, subtitle, figsize=(8, 6), left=0.15, top=0.80, bottom=0.24):
    fig, ax = plt.subplots(figsize=figsize)
    fig.subplots_adjust(left=left, right=0.96, bottom=bottom, top=top)
    fig.suptitle(title, y=0.97, fontsize=22)
    fig.text(0.5, 0.89, subtitle, ha="center", fontsize=16)
    return fig, ax


def bars(ax, values, ylabel, labels, upper):
    artists = ax.bar(("Base", "SFT"), values, color=COLORS, width=0.55)
    ax.bar_label(artists, labels=[text(label) for label in labels], padding=8, fontsize=18)
    ax.set(ylabel=ylabel, ylim=(0, upper))
    ax.set_axisbelow(True)
    ax.grid(axis="y", color="0.9")
    ax.yaxis.set_major_locator(MaxNLocator(5))


def persona_figure(metrics):
    fig, ax = figure("Reference similarity",
                      f"{metrics['examples']:,} held-out replies from {metrics['episodes']} episodes",
                      bottom=0.14)
    similarity = [metrics[name]["mean_cosine_similarity"] for name in MODELS]
    upper = max(0.32, max(similarity) * 1.6)
    bars(ax, similarity, "Mean cosine similarity",
         [f"{value:.4f}" for value in similarity], upper)
    low, high = metrics["paired_difference_episode_bootstrap_95_ci"]
    # A bracket identifies the paired comparison; this is not a per-bar interval.
    height = upper * 0.79
    ax.plot([0, 0, 1, 1], [height - upper * 0.025, height, height, height - upper * 0.025],
            color="0.4", linewidth=1)
    ax.text(0.5, height + upper * 0.025,
            text(f"SFT minus base: {metrics['mean_paired_difference']:+.4f}\n"
                 f"95% CI [{low:.4f}, {high:.4f}]"),
            ha="center", va="bottom", fontsize=14, linespacing=1.4)
    return fig


def length_figure(metrics):
    fig, ax = figure("Generated reply length", f"{metrics['examples']:,} matched prompts, greedy decoding",
                      bottom=0.14)
    lengths = [metrics[name]["mean_generated_tokens"] for name in MODELS]
    bars(ax, lengths, "Mean generated tokens",
         [f"{value:.1f}" for value in lengths], max(lengths) * 1.3)
    return fig


def physics_figure(metrics):
    fig, accuracy = figure("Physics accuracy", "", figsize=(9, 7), left=0.29, top=0.73, bottom=0.15)
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
    low, high = [100 * value for value in metrics["paired_accuracy_difference_95_ci"]]
    gain = 100 * metrics["overall"]["accuracy_difference"]
    fig.text(0.5, 0.835, text(f"Overall SFT minus base: {gain:+.2f} pp\n"
             f"95% CI [{low:+.2f}, {high:+.2f}] pp"), ha="center", fontsize=15, linespacing=1.4)
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
    for filename, name, draw in (("persona.json", "persona_similarity", persona_figure),
                                 ("persona.json", "reply_length", length_figure),
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
