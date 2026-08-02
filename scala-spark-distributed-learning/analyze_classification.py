#!/usr/bin/env python3

from __future__ import annotations

import argparse
import math
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


METHOD_FILES = {
    "PSO": {
        "time": "resultados_iters_pso_t.txt",
        "conf": "resultados_iters_pso_conf.txt",
    },
    "DAPSO-4": {
        "time": "resultados_iters_dapso1_t.txt",
        "conf": "resultados_iters_dapso1_conf.txt",
    },
    "DAPSO-10": {
        "time": "resultados_iters_dapso2_t.txt",
        "conf": "resultados_iters_dapso2_conf.txt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse repeated PSO/DAPSO classification experiments."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/repetitions/classification"),
        help="Directory containing run_01, ..., run_10.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/classification"),
        help="Directory in which tables and figures will be generated.",
    )
    parser.add_argument(
        "--conf-order",
        default="tp,fn,fp,tn",
        help=(
            "Meaning of the four confusion-matrix values. "
            "Default: tp,fn,fp,tn. Verify this against confusionMatrix()."
        ),
    )
    return parser.parse_args()


def parse_time_file(path: Path) -> list[tuple[int, float]]:
    """
    Accepted examples:
        ((10,19.0))
        (10,19.0)
    """
    pattern = re.compile(
        r"\(\s*\(*\s*(\d+)\s*,\s*([-+]?\d+(?:\.\d+)?)\s*\)*\s*\)"
    )

    values: list[tuple[int, float]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue

        match = pattern.search(line)
        if not match:
            raise ValueError(
                f"Cannot parse time value in {path}, line {line_number}: {line!r}"
            )

        iterations = int(match.group(1))
        time_value = float(match.group(2))
        values.append((iterations, time_value))

    if not values:
        raise ValueError(f"No time measurements found in {path}")

    return values


def parse_confusion_file(path: Path) -> list[tuple[int, int, int, int]]:
    """
    Accepted example:
        (22,4,8,80)
    """
    pattern = re.compile(
        r"\(\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*,\s*(-?\d+)\s*\)"
    )

    values: list[tuple[int, int, int, int]] = []

    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = line.strip()
        if not line:
            continue

        match = pattern.search(line)
        if not match:
            raise ValueError(
                f"Cannot parse confusion matrix in {path}, "
                f"line {line_number}: {line!r}"
            )

        values.append(tuple(int(match.group(i)) for i in range(1, 5)))

    if not values:
        raise ValueError(f"No confusion matrices found in {path}")

    return values


def safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else math.nan


def confidence_interval_95(series: pd.Series) -> float:
    """
    95% CI half-width using Student's t for n=10:
        t(9, 0.975) = 2.262157
    For other n, uses a normal approximation when scipy is unavailable.
    """
    values = series.dropna().astype(float)
    n = len(values)

    if n < 2:
        return math.nan

    try:
        from scipy.stats import t

        critical = float(t.ppf(0.975, df=n - 1))
    except ImportError:
        critical = 2.262157 if n == 10 else 1.96

    return critical * float(values.std(ddof=1)) / math.sqrt(n)


def load_all_runs(
    input_dir: Path,
    confusion_order: list[str],
) -> pd.DataFrame:
    run_dirs = sorted(path for path in input_dir.glob("run_*") if path.is_dir())

    if not run_dirs:
        raise FileNotFoundError(
            f"No run_* directories were found under {input_dir.resolve()}"
        )

    records: list[dict[str, object]] = []

    for run_dir in run_dirs:
        run_name = run_dir.name

        for method, names in METHOD_FILES.items():
            time_path = run_dir / names["time"]
            conf_path = run_dir / names["conf"]

            if not time_path.exists():
                raise FileNotFoundError(f"Missing file: {time_path}")

            if not conf_path.exists():
                raise FileNotFoundError(f"Missing file: {conf_path}")

            times = parse_time_file(time_path)
            confusions = parse_confusion_file(conf_path)

            if len(times) != len(confusions):
                raise ValueError(
                    f"{run_name}, {method}: {len(times)} time rows but "
                    f"{len(confusions)} confusion matrices."
                )

            for (iterations, runtime), raw_confusion in zip(times, confusions):
                values = dict(zip(confusion_order, raw_confusion))

                tp = values["tp"]
                fn = values["fn"]
                fp = values["fp"]
                tn = values["tn"]

                total = tp + fn + fp + tn
                accuracy = safe_divide(tp + tn, total)
                precision = safe_divide(tp, tp + fp)
                recall = safe_divide(tp, tp + fn)
                specificity = safe_divide(tn, tn + fp)
                f1 = safe_divide(2 * precision * recall, precision + recall)

                records.append(
                    {
                        "run": run_name,
                        "method": method,
                        "iterations": iterations,
                        "runtime": runtime,
                        "tp": tp,
                        "fn": fn,
                        "fp": fp,
                        "tn": tn,
                        "accuracy": accuracy,
                        "precision": precision,
                        "recall": recall,
                        "specificity": specificity,
                        "f1": f1,
                    }
                )

    return pd.DataFrame.from_records(records)


def calculate_runtime_summary(data: pd.DataFrame) -> pd.DataFrame:
    summary = (
        data.groupby(["method", "iterations"], as_index=False)
        .agg(
            n=("runtime", "count"),
            mean_runtime=("runtime", "mean"),
            std_runtime=("runtime", "std"),
            median_runtime=("runtime", "median"),
            min_runtime=("runtime", "min"),
            max_runtime=("runtime", "max"),
        )
    )

    summary["cv_runtime"] = (
        summary["std_runtime"] / summary["mean_runtime"]
    )

    ci = (
        data.groupby(["method", "iterations"])["runtime"]
        .apply(confidence_interval_95)
        .rename("ci95_runtime")
        .reset_index()
    )

    return summary.merge(ci, on=["method", "iterations"], how="left")


def calculate_quality_summary(data: pd.DataFrame) -> pd.DataFrame:
    metrics = ["accuracy", "precision", "recall", "specificity", "f1"]

    grouped = data.groupby(["method", "iterations"], as_index=False)

    summary = grouped[metrics].agg(["mean", "std"])
    summary.columns = [
        "_".join(col).strip("_") for col in summary.columns.to_flat_index()
    ]
    summary = summary.reset_index()

    for metric in metrics:
        ci = (
            data.groupby(["method", "iterations"])[metric]
            .apply(confidence_interval_95)
            .rename(f"{metric}_ci95")
            .reset_index()
        )
        summary = summary.merge(ci, on=["method", "iterations"], how="left")

    return summary


def calculate_speedups(data: pd.DataFrame) -> pd.DataFrame:
    wide = data.pivot_table(
        index=["run", "iterations"],
        columns="method",
        values="runtime",
        aggfunc="first",
    ).reset_index()

    required = {"PSO", "DAPSO-4", "DAPSO-10"}
    missing = required.difference(wide.columns)

    if missing:
        raise ValueError(f"Cannot calculate speedups; missing methods: {missing}")

    wide["speedup_dapso4"] = wide["PSO"] / wide["DAPSO-4"]
    wide["speedup_dapso10"] = wide["PSO"] / wide["DAPSO-10"]
    wide["dapso10_vs_dapso4"] = wide["DAPSO-4"] / wide["DAPSO-10"]

    raw = wide[
        [
            "run",
            "iterations",
            "PSO",
            "DAPSO-4",
            "DAPSO-10",
            "speedup_dapso4",
            "speedup_dapso10",
            "dapso10_vs_dapso4",
        ]
    ].copy()

    summary_rows: list[dict[str, float | int]] = []

    for iterations, group in raw.groupby("iterations"):
        row: dict[str, float | int] = {
            "iterations": int(iterations),
            "n": len(group),
        }

        for metric in [
            "speedup_dapso4",
            "speedup_dapso10",
            "dapso10_vs_dapso4",
        ]:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_std"] = group[metric].std(ddof=1)
            row[f"{metric}_ci95"] = confidence_interval_95(group[metric])

        summary_rows.append(row)

    summary = pd.DataFrame(summary_rows).sort_values("iterations")

    return raw, summary


def save_runtime_plot(summary: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(7.2, 4.5))

    for method in ["PSO", "DAPSO-4", "DAPSO-10"]:
        subset = summary[summary["method"] == method].sort_values("iterations")

        plt.errorbar(
            subset["iterations"],
            subset["mean_runtime"],
            yerr=subset["ci95_runtime"],
            marker="o",
            capsize=3,
            linewidth=1.4,
            label=method,
        )

    plt.xlabel("PSO iterations")
    plt.ylabel("Runtime")
    plt.title("Classification runtime over ten executions")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / "classification_runtime_mean_ci95.pdf")
    plt.savefig(
        output / "classification_runtime_mean_ci95.png",
        dpi=600,
    )
    plt.close()


def save_speedup_plot(summary: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(7.2, 4.5))

    plt.errorbar(
        summary["iterations"],
        summary["speedup_dapso4_mean"],
        yerr=summary["speedup_dapso4_ci95"],
        marker="o",
        capsize=3,
        linewidth=1.4,
        label="PSO / DAPSO-4",
    )

    plt.errorbar(
        summary["iterations"],
        summary["speedup_dapso10_mean"],
        yerr=summary["speedup_dapso10_ci95"],
        marker="s",
        capsize=3,
        linewidth=1.4,
        label="PSO / DAPSO-10",
    )

    plt.axhline(1.0, linestyle="--", linewidth=1.0)
    plt.xlabel("PSO iterations")
    plt.ylabel("Speedup")
    plt.title("DAPSO speedup over sequential PSO")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / "classification_speedup_mean_ci95.pdf")
    plt.savefig(
        output / "classification_speedup_mean_ci95.png",
        dpi=600,
    )
    plt.close()


def save_quality_plot(
    quality: pd.DataFrame,
    metric: str,
    output: Path,
) -> None:
    plt.figure(figsize=(7.2, 4.5))

    for method in ["PSO", "DAPSO-4", "DAPSO-10"]:
        subset = quality[quality["method"] == method].sort_values("iterations")

        plt.errorbar(
            subset["iterations"],
            subset[f"{metric}_mean"],
            yerr=subset[f"{metric}_ci95"],
            marker="o",
            capsize=3,
            linewidth=1.4,
            label=method,
        )

    plt.xlabel("PSO iterations")
    plt.ylabel(metric.upper() if metric == "f1" else metric.capitalize())
    plt.title(f"Classification {metric.upper()} over ten executions")
    plt.ylim(0.0, 1.05)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / f"classification_{metric}_mean_ci95.pdf")
    plt.savefig(
        output / f"classification_{metric}_mean_ci95.png",
        dpi=600,
    )
    plt.close()


def save_latex_tables(
    runtime: pd.DataFrame,
    speedup: pd.DataFrame,
    quality: pd.DataFrame,
    output: Path,
) -> None:
    runtime_table = runtime.pivot(
        index="iterations",
        columns="method",
        values=["mean_runtime", "std_runtime"],
    )

    runtime_lines = [
        r"\begin{tabular}{rccc}",
        r"\hline",
        r"Iterations & PSO & DAPSO-4 & DAPSO-10 \\",
        r"\hline",
    ]

    for iterations in sorted(runtime["iterations"].unique()):
        values = []

        for method in ["PSO", "DAPSO-4", "DAPSO-10"]:
            row = runtime[
                (runtime["iterations"] == iterations)
                & (runtime["method"] == method)
            ].iloc[0]

            values.append(
                f"{row['mean_runtime']:.2f} $\\pm$ "
                f"{row['std_runtime']:.2f}"
            )

        runtime_lines.append(
            f"{iterations} & " + " & ".join(values) + r" \\"
        )

    runtime_lines.extend([r"\hline", r"\end{tabular}"])

    (output / "runtime_table.tex").write_text(
        "\n".join(runtime_lines),
        encoding="utf-8",
    )

    speedup_lines = [
        r"\begin{tabular}{rcc}",
        r"\hline",
        r"Iterations & DAPSO-4 & DAPSO-10 \\",
        r"\hline",
    ]

    for _, row in speedup.sort_values("iterations").iterrows():
        speedup_lines.append(
            f"{int(row['iterations'])} & "
            f"{row['speedup_dapso4_mean']:.2f} $\\pm$ "
            f"{row['speedup_dapso4_std']:.2f} & "
            f"{row['speedup_dapso10_mean']:.2f} $\\pm$ "
            f"{row['speedup_dapso10_std']:.2f} \\\\"
        )

    speedup_lines.extend([r"\hline", r"\end{tabular}"])

    (output / "speedup_table.tex").write_text(
        "\n".join(speedup_lines),
        encoding="utf-8",
    )

    final_iteration = int(quality["iterations"].max())
    final_quality = quality[quality["iterations"] == final_iteration]

    quality_lines = [
        r"\begin{tabular}{lccc}",
        r"\hline",
        r"Method & Accuracy & Recall & F1 \\",
        r"\hline",
    ]

    for method in ["PSO", "DAPSO-4", "DAPSO-10"]:
        row = final_quality[final_quality["method"] == method].iloc[0]

        quality_lines.append(
            f"{method} & "
            f"{row['accuracy_mean']:.3f} $\\pm$ "
            f"{row['accuracy_std']:.3f} & "
            f"{row['recall_mean']:.3f} $\\pm$ "
            f"{row['recall_std']:.3f} & "
            f"{row['f1_mean']:.3f} $\\pm$ "
            f"{row['f1_std']:.3f} \\\\"
        )

    quality_lines.extend([r"\hline", r"\end{tabular}"])

    (output / "quality_final_iteration_table.tex").write_text(
        "\n".join(quality_lines),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    order = [item.strip().lower() for item in args.conf_order.split(",")]

    if sorted(order) != ["fn", "fp", "tn", "tp"]:
        print(
            "ERROR: --conf-order must contain exactly tp, fn, fp and tn.",
            file=sys.stderr,
        )
        return 2

    args.output.mkdir(parents=True, exist_ok=True)

    data = load_all_runs(args.input, order)
    runtime = calculate_runtime_summary(data)
    quality = calculate_quality_summary(data)
    speedup_raw, speedup_summary = calculate_speedups(data)

    data.to_csv(args.output / "classification_all_runs.csv", index=False)
    runtime.to_csv(args.output / "runtime_summary.csv", index=False)
    quality.to_csv(args.output / "quality_summary.csv", index=False)
    speedup_raw.to_csv(args.output / "speedup_all_runs.csv", index=False)
    speedup_summary.to_csv(
        args.output / "speedup_summary.csv",
        index=False,
    )

    save_runtime_plot(runtime, args.output)
    save_speedup_plot(speedup_summary, args.output)
    save_quality_plot(quality, "accuracy", args.output)
    save_quality_plot(quality, "f1", args.output)
    save_latex_tables(runtime, speedup_summary, quality, args.output)

    print(f"Runs analysed: {data['run'].nunique()}")
    print(f"Rows analysed: {len(data)}")
    print(f"Results saved to: {args.output.resolve()}")

    final_iterations = int(data["iterations"].max())

    print(f"\nSummary at {final_iterations} iterations:")

    final_runtime = runtime[runtime["iterations"] == final_iterations]

    for _, row in final_runtime.iterrows():
        print(
            f"  {row['method']:8s}: "
            f"{row['mean_runtime']:.2f} ± "
            f"{row['std_runtime']:.2f}, "
            f"95% CI ± {row['ci95_runtime']:.2f}"
        )

    final_speedup = speedup_summary[
        speedup_summary["iterations"] == final_iterations
    ].iloc[0]

    print(
        f"  DAPSO-4 speedup: "
        f"{final_speedup['speedup_dapso4_mean']:.3f} ± "
        f"{final_speedup['speedup_dapso4_std']:.3f}"
    )
    print(
        f"  DAPSO-10 speedup: "
        f"{final_speedup['speedup_dapso10_mean']:.3f} ± "
        f"{final_speedup['speedup_dapso10_std']:.3f}"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
