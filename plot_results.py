"""Plot the saved aggregate evaluations; no model or transcript data is needed."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


COLOR = "#356A91"
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


def comparison_figure(estimate, interval, title, unit, decimals):
    low, high = interval
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    fig.subplots_adjust(left=0.10, right=0.96, bottom=0.26, top=0.78)
    fig.suptitle(title, y=0.96, fontsize=22)

    # The saved intervals describe paired differences, not individual model means.
    ax.axvline(0, ymax=0.82, color="0.65", linestyle="--", linewidth=1)
    ax.hlines(0, low, high, color=COLOR, linewidth=2)
    ax.plot([low, high], [0, 0], "|", color=COLOR, markersize=12)
    ax.plot(estimate, 0, "o", color=COLOR, markersize=9)
    ax.annotate(f"{estimate:+.{decimals}f}", (estimate, 0), xytext=(0, 13),
                textcoords="offset points", ha="center", va="bottom", color=COLOR)
    for endpoint in (low, high):
        ax.annotate(f"{endpoint:+.{decimals}f}", (endpoint, 0), xytext=(0, -15),
                    textcoords="offset points", ha="center", va="top", fontsize=15)

    lower, upper = min(0, low, estimate), max(0, high, estimate)
    margin = max((upper - lower) * 0.2, 0.001)
    ax.set(xlim=(lower - margin, upper + margin), ylim=(-0.8, 0.8), yticks=[],
           xlabel=f"SFT minus base ({unit})")
    ax.spines[["left", "top", "right"]].set_visible(False)
    ax.xaxis.set_major_locator(MaxNLocator(4, steps=[1, 2, 2.5, 5, 10]))
    ax.text(0, 0.98, "No change", transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=13, color="0.45")
    ax.text(0.99, 0.98, text("95% CI"), transform=ax.transAxes,
            ha="right", va="top", fontsize=15, color="0.35")
    return fig


def persona_figure(metrics):
    return comparison_figure(
        metrics["mean_paired_difference"], metrics["paired_difference_episode_bootstrap_95_ci"],
        "Reference similarity change", "cosine similarity", 4,
    )


def physics_figure(metrics):
    return comparison_figure(
        100 * metrics["overall"]["accuracy_difference"],
        [100 * value for value in metrics["paired_accuracy_difference_95_ci"]],
        "Physics accuracy change", "percentage points", 2,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--output-dir", type=Path, help="Defaults to RESULTS_DIR/plots")
    parser.add_argument("--usetex", action="store_true", help="Use LaTeX for text (requires a working LaTeX installation)")
    args = parser.parse_args()
    style(args.usetex)
    output_dir = args.output_dir or args.results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, name, draw in (("persona.json", "persona_change", persona_figure),
                                 ("mmlu_physics.json", "physics_change", physics_figure)):
        metrics = json.loads((args.results_dir / filename).read_text())
        fig = draw(metrics)
        for extension in ("png", "pdf"):
            path = output_dir / f"{name}.{extension}"
            fig.savefig(path, dpi=200, facecolor="white", bbox_inches="tight")
            print(path)
        plt.close(fig)


if __name__ == "__main__":
    main()
