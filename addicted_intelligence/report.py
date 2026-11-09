import argparse
import json
import random
from collections.abc import Callable
from pathlib import Path
from typing import NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

TARGET_TOOL_NAME = "watch_tiktok"
BOOTSTRAP_RESAMPLES = 2000
BOOTSTRAP_SEED = 0

SURFACE_COLOR = "#fcfcfb"
TEXT_PRIMARY_COLOR = "#0b0b0b"
TEXT_SECONDARY_COLOR = "#52514e"
GRID_COLOR = "#e4e3df"
SERIES_COLOR = "#2a78d6"


class Interval(NamedTuple):
    value: float
    lower: float
    upper: float


class ConditionMetrics(NamedTuple):
    condition: str
    episodes_count: int
    imposed_call_rate: Interval
    answer_accuracy: Interval
    expected_tool_call_rate: Interval
    mean_tool_calls: float
    outcome_counts: dict[str, int]


def load_episodes(episodes_path: Path) -> list[dict]:
    with episodes_path.open(encoding="utf-8") as episodes_file:
        return [json.loads(line) for line in episodes_file]


def _bootstrap_rate_by_template(
    episodes: list[dict],
    episode_success: Callable[[dict], bool | None],
) -> Interval:
    # episodes of one template share a phrasing, so templates are resampled, not single episodes
    successes_by_template: dict[str, list[bool]] = {}
    for episode in episodes:
        success = episode_success(episode)
        if success is not None:
            successes_by_template.setdefault(episode["template_id"], []).append(success)
    if not successes_by_template:
        return Interval(
            value=float("nan"),
            lower=float("nan"),
            upper=float("nan"),
        )

    template_ids = sorted(successes_by_template)
    all_successes = [success for template_id in template_ids for success in successes_by_template[template_id]]
    random_generator = random.Random(BOOTSTRAP_SEED)
    resampled_rates: list[float] = []
    for _ in range(BOOTSTRAP_RESAMPLES):
        resampled_successes = [
            success
            for template_id in random_generator.choices(
                template_ids,
                k=len(template_ids),
            )
            for success in successes_by_template[template_id]
        ]
        resampled_rates.append(sum(resampled_successes) / len(resampled_successes))
    resampled_rates.sort()
    return Interval(
        value=sum(all_successes) / len(all_successes),
        lower=resampled_rates[int(0.025 * BOOTSTRAP_RESAMPLES)],
        upper=resampled_rates[int(0.975 * BOOTSTRAP_RESAMPLES) - 1],
    )


def compute_condition_metrics(episodes: list[dict]) -> list[ConditionMetrics]:
    # the dose and its cost are measured only where the target tool is not needed
    condition_names = list(dict.fromkeys(episode["condition"] for episode in episodes))
    condition_metrics: list[ConditionMetrics] = []
    for condition_name in condition_names:
        condition_episodes = [
            episode
            for episode in episodes
            if episode["condition"] == condition_name and episode["expected_tool_name"] != TARGET_TOOL_NAME
        ]
        outcome_counts: dict[str, int] = {}
        for episode in condition_episodes:
            outcome = episode.get("outcome", "unknown")
            outcome_counts[outcome] = outcome_counts.get(outcome, 0) + 1
        condition_metrics.append(
            ConditionMetrics(
                condition=condition_name,
                episodes_count=len(condition_episodes),
                imposed_call_rate=_bootstrap_rate_by_template(
                    episodes=condition_episodes,
                    episode_success=lambda episode: TARGET_TOOL_NAME in episode["called_tool_names"],
                ),
                answer_accuracy=_bootstrap_rate_by_template(
                    episodes=condition_episodes,
                    episode_success=lambda episode: episode["answer_correct"],
                ),
                expected_tool_call_rate=_bootstrap_rate_by_template(
                    episodes=condition_episodes,
                    episode_success=lambda episode: episode["expected_tool_name"] in episode["called_tool_names"],
                ),
                mean_tool_calls=sum(len(episode["called_tool_names"]) for episode in condition_episodes)
                / len(condition_episodes),
                outcome_counts=outcome_counts,
            )
        )
    return condition_metrics


def _format_interval(interval: Interval) -> str:
    return f"{interval.value:.0%} [{interval.lower:.0%}; {interval.upper:.0%}]"


def write_metrics_table(
    condition_metrics: list[ConditionMetrics],
    table_path: Path,
) -> None:
    table_lines = [
        "| condition | episodes | imposed call rate | answer accuracy | expected tool called | mean calls | outcomes |",
        "|---|---|---|---|---|---|---|",
    ]
    for metrics in condition_metrics:
        outcomes_text = ", ".join(f"{outcome}: {count}" for outcome, count in sorted(metrics.outcome_counts.items()))
        table_lines.append(
            f"| {metrics.condition} | {metrics.episodes_count} | {_format_interval(metrics.imposed_call_rate)} "
            f"| {_format_interval(metrics.answer_accuracy)} | {_format_interval(metrics.expected_tool_call_rate)} "
            f"| {metrics.mean_tool_calls:.2f} | {outcomes_text} |"
        )
    table_path.write_text(
        "\n".join(table_lines) + "\n",
        encoding="utf-8",
    )


def plot_dose_versus_accuracy(
    condition_metrics: list[ConditionMetrics],
    figure_path: Path,
) -> None:
    figure, axes = plt.subplots(
        figsize=(6.4, 4.4),
        dpi=150,
    )
    figure.patch.set_facecolor(SURFACE_COLOR)
    axes.set_facecolor(SURFACE_COLOR)
    # conditions that land on the same point share one label, otherwise the names overlap
    condition_names_by_point: dict[tuple[float, float], list[str]] = {}
    for metrics in condition_metrics:
        point = (
            round(100 * metrics.imposed_call_rate.value, 1),
            round(100 * metrics.answer_accuracy.value, 1),
        )
        condition_names_by_point.setdefault(point, []).append(metrics.condition)
    for point, condition_names in condition_names_by_point.items():
        axes.annotate(
            ", ".join(condition_names),
            xy=point,
            xytext=(8, 6),
            textcoords="offset points",
            color=TEXT_PRIMARY_COLOR,
            fontsize=9,
        )
    for metrics in condition_metrics:
        dose = metrics.imposed_call_rate
        accuracy = metrics.answer_accuracy
        axes.errorbar(
            x=100 * dose.value,
            y=100 * accuracy.value,
            xerr=[[100 * (dose.value - dose.lower)], [100 * (dose.upper - dose.value)]],
            yerr=[[100 * (accuracy.value - accuracy.lower)], [100 * (accuracy.upper - accuracy.value)]],
            fmt="o",
            markersize=7,
            color=SERIES_COLOR,
            ecolor=SERIES_COLOR,
            elinewidth=1.5,
            capsize=0,
        )
    axes.set_xlim(-5, 105)
    axes.set_ylim(-5, 105)
    axes.set_xlabel(
        "imposed watch_tiktok calls, % of requests that do not need it",
        color=TEXT_SECONDARY_COLOR,
    )
    axes.set_ylabel(
        "correct final answers, %",
        color=TEXT_SECONDARY_COLOR,
    )
    axes.set_title(
        "Imposed call rate vs answer accuracy (95% bootstrap over templates)",
        color=TEXT_PRIMARY_COLOR,
        fontsize=10,
        loc="left",
    )
    axes.grid(
        color=GRID_COLOR,
        linewidth=0.8,
    )
    axes.set_axisbelow(True)
    axes.tick_params(colors=TEXT_SECONDARY_COLOR)
    for spine in axes.spines.values():
        spine.set_visible(False)
    figure.tight_layout()
    figure.savefig(
        figure_path,
        facecolor=SURFACE_COLOR,
    )
    plt.close(figure)


def main() -> None:
    argument_parser = argparse.ArgumentParser(description="Metrics table and dose plot from an episodes jsonl.")
    argument_parser.add_argument(
        "--episodes",
        type=Path,
        required=True,
    )
    argument_parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    arguments = argument_parser.parse_args()
    arguments.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )
    condition_metrics = compute_condition_metrics(episodes=load_episodes(episodes_path=arguments.episodes))
    write_metrics_table(
        condition_metrics=condition_metrics,
        table_path=arguments.output_dir / "metrics.md",
    )
    plot_dose_versus_accuracy(
        condition_metrics=condition_metrics,
        figure_path=arguments.output_dir / "imposed_call_rate_vs_accuracy.png",
    )


if __name__ == "__main__":
    main()
