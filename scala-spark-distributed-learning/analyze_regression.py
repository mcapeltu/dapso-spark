#!/usr/bin/env python3
from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

FILES = {
    "PSO": {
        "runtime": "resultados_reg_pso_t.txt",
        "mse": "resultados_reg_pso_mse.txt",
    },
    "DAPSO": {
        "runtime": "resultados_reg_dapso_t.txt",
        "mse": "resultados_reg_dapso_mse.txt",
    },
}

PAIR_PATTERN = re.compile(
    r"\(\s*\(*\s*(\d+)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s*\)*\s*\)"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyse repeated PSO/DAPSO regression experiments."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/repetitions/regression"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/analysis/regression"),
    )
    parser.add_argument("--alpha", type=float, default=0.05)
    return parser.parse_args()


def parse_pair_file(path: Path) -> list[tuple[int, float]]:
    values = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw_line.strip()
        if not line:
            continue
        match = PAIR_PATTERN.fullmatch(line) or PAIR_PATTERN.search(line)
        if not match:
            raise ValueError(f"Cannot parse {path}, line {line_no}: {raw_line!r}")
        values.append((int(match.group(1)), float(match.group(2))))
    if not values:
        raise ValueError(f"No observations found in {path}")
    return values


def ci95_half_width(series: pd.Series) -> float:
    values = series.dropna().astype(float).to_numpy()
    if len(values) < 2:
        return math.nan
    return float(stats.t.ppf(0.975, len(values) - 1) * stats.sem(values))


def holm_adjust(values: pd.Series) -> pd.Series:
    p = values.astype(float).to_numpy()
    order = np.argsort(p)
    sorted_p = p[order]
    adjusted_sorted = np.empty_like(sorted_p)
    running_max = 0.0
    n = len(sorted_p)
    for rank, value in enumerate(sorted_p):
        running_max = max(running_max, (n - rank) * value)
        adjusted_sorted[rank] = min(running_max, 1.0)
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=values.index)


def load_runs(input_dir: Path) -> pd.DataFrame:
    run_dirs = sorted(p for p in input_dir.glob("run_*") if p.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories under {input_dir.resolve()}")

    records = []
    for run_dir in run_dirs:
        parsed = {}
        for method, names in FILES.items():
            runtime_file = run_dir / names["runtime"]
            mse_file = run_dir / names["mse"]
            if not runtime_file.exists() or not mse_file.exists():
                raise FileNotFoundError(f"Missing regression files in {run_dir}")
            parsed[method] = {
                "runtime": dict(parse_pair_file(runtime_file)),
                "mse": dict(parse_pair_file(mse_file)),
            }

        iterations = set(parsed["PSO"]["runtime"])
        if iterations != set(parsed["DAPSO"]["runtime"]):
            raise ValueError(f"Iteration mismatch in {run_dir.name}")

        for method in ("PSO", "DAPSO"):
            if iterations != set(parsed[method]["mse"]):
                raise ValueError(f"Runtime/MSE iteration mismatch in {run_dir.name}, {method}")
            for iteration in sorted(iterations):
                records.append({
                    "run": run_dir.name,
                    "method": method,
                    "iterations": iteration,
                    "runtime_s": parsed[method]["runtime"][iteration],
                    "mse": parsed[method]["mse"][iteration],
                })

    return pd.DataFrame(records).sort_values(["run", "iterations", "method"])


def summarise(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    summary = data.groupby(["method", "iterations"], as_index=False).agg(
        n=(metric, "count"),
        mean=(metric, "mean"),
        std=(metric, "std"),
        median=(metric, "median"),
        minimum=(metric, "min"),
        maximum=(metric, "max"),
    )
    summary["cv"] = summary["std"] / summary["mean"]
    ci = data.groupby(["method", "iterations"])[metric].apply(ci95_half_width).rename("ci95").reset_index()
    return summary.merge(ci, on=["method", "iterations"])


def wide_pairs(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    return data.pivot(index=["run", "iterations"], columns="method", values=metric).reset_index()


def speedup(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = wide_pairs(data, "runtime_s")
    raw["speedup"] = raw["PSO"] / raw["DAPSO"]
    summary = raw.groupby("iterations", as_index=False).agg(
        n=("speedup", "count"),
        mean_speedup=("speedup", "mean"),
        std_speedup=("speedup", "std"),
        median_speedup=("speedup", "median"),
        min_speedup=("speedup", "min"),
        max_speedup=("speedup", "max"),
    )
    ci = raw.groupby("iterations")["speedup"].apply(ci95_half_width).rename("ci95_speedup").reset_index()
    return raw, summary.merge(ci, on="iterations")


def paired_tests(data: pd.DataFrame, metric: str) -> pd.DataFrame:
    wide = wide_pairs(data, metric)
    rows = []
    for iteration, group in wide.groupby("iterations"):
        pso = group["PSO"].to_numpy(float)
        dapso = group["DAPSO"].to_numpy(float)
        diff = pso - dapso
        t_result = stats.ttest_rel(pso, dapso, alternative="greater")
        try:
            w_result = stats.wilcoxon(pso, dapso, alternative="greater", method="auto")
            w_stat, w_p = float(w_result.statistic), float(w_result.pvalue)
        except ValueError:
            w_stat, w_p = math.nan, 1.0
        sd_diff = np.std(diff, ddof=1)
        rows.append({
            "metric": metric,
            "iterations": int(iteration),
            "n": len(group),
            "mean_pso": float(np.mean(pso)),
            "mean_dapso": float(np.mean(dapso)),
            "mean_difference_pso_minus_dapso": float(np.mean(diff)),
            "paired_t_statistic": float(t_result.statistic),
            "paired_t_pvalue": float(t_result.pvalue),
            "wilcoxon_statistic": w_stat,
            "wilcoxon_pvalue": w_p,
            "cohen_dz": float(np.mean(diff) / sd_diff) if sd_diff else math.nan,
        })
    result = pd.DataFrame(rows).sort_values("iterations").reset_index(drop=True)
    result["paired_t_pvalue_holm"] = holm_adjust(result["paired_t_pvalue"])
    result["wilcoxon_pvalue_holm"] = holm_adjust(result["wilcoxon_pvalue"])
    return result


def plot_summary(summary: pd.DataFrame, ylabel: str, title: str, stem: str, output: Path) -> None:
    plt.figure(figsize=(7.2, 4.5))
    for method in ("PSO", "DAPSO"):
        subset = summary[summary["method"] == method].sort_values("iterations")
        plt.errorbar(
            subset["iterations"], subset["mean"], yerr=subset["ci95"],
            marker="o", capsize=3, linewidth=1.4, label=method
        )
    plt.xlabel("PSO iterations")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / f"{stem}.pdf")
    plt.savefig(output / f"{stem}.png", dpi=600)
    plt.close()


def plot_speedup(summary: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(7.2, 4.5))
    plt.errorbar(
        summary["iterations"], summary["mean_speedup"],
        yerr=summary["ci95_speedup"], marker="o", capsize=3,
        linewidth=1.4, label="PSO / DAPSO"
    )
    plt.axhline(1.0, linestyle="--", linewidth=1.0, label="Baseline")
    plt.xlabel("PSO iterations")
    plt.ylabel("Speedup")
    plt.title("DAPSO speedup over sequential PSO")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / "regression_speedup_mean_ci95.pdf")
    plt.savefig(output / "regression_speedup_mean_ci95.png", dpi=600)
    plt.close()


def save_latex(runtime, mse, speedup_summary, output: Path) -> None:
    iterations = sorted(runtime["iterations"].unique())
    lines = [
        r"\begin{tabular}{rccc}", r"\hline",
        r"Iterations & PSO runtime & DAPSO runtime & Speedup \\", r"\hline"
    ]
    for it in iterations:
        p = runtime[(runtime.method == "PSO") & (runtime.iterations == it)].iloc[0]
        d = runtime[(runtime.method == "DAPSO") & (runtime.iterations == it)].iloc[0]
        s = speedup_summary[speedup_summary.iterations == it].iloc[0]
        lines.append(
            f"{it} & {p['mean']:.2f} $\\pm$ {p['ci95']:.2f} & "
            f"{d['mean']:.2f} $\\pm$ {d['ci95']:.2f} & "
            f"{s['mean_speedup']:.2f} $\\pm$ {s['ci95_speedup']:.2f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    (output / "runtime_speedup_table.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{tabular}{rcc}", r"\hline",
        r"Iterations & PSO MSE & DAPSO MSE \\", r"\hline"
    ]
    for it in iterations:
        p = mse[(mse.method == "PSO") & (mse.iterations == it)].iloc[0]
        d = mse[(mse.method == "DAPSO") & (mse.iterations == it)].iloc[0]
        lines.append(
            f"{it} & {p['mean']:.3f} $\\pm$ {p['ci95']:.3f} & "
            f"{d['mean']:.3f} $\\pm$ {d['ci95']:.3f} \\\\"
        )
    lines.extend([r"\hline", r"\end{tabular}"])
    (output / "mse_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    data = load_runs(args.input)
    runtime_summary = summarise(data, "runtime_s")
    mse_summary = summarise(data, "mse")
    speedup_raw, speedup_summary = speedup(data)
    runtime_tests = paired_tests(data, "runtime_s")
    mse_tests = paired_tests(data, "mse")

    data.to_csv(args.output / "regression_all_runs.csv", index=False)
    runtime_summary.to_csv(args.output / "runtime_summary.csv", index=False)
    mse_summary.to_csv(args.output / "mse_summary.csv", index=False)
    speedup_raw.to_csv(args.output / "speedup_all_runs.csv", index=False)
    speedup_summary.to_csv(args.output / "speedup_summary.csv", index=False)
    runtime_tests.to_csv(args.output / "runtime_significance.csv", index=False)
    mse_tests.to_csv(args.output / "mse_significance.csv", index=False)

    plot_summary(runtime_summary, "Runtime (s)", "Regression runtime over repeated executions", "regression_runtime_mean_ci95", args.output)
    plot_summary(mse_summary, "MSE", "Regression MSE over repeated executions", "regression_mse_mean_ci95", args.output)
    plot_speedup(speedup_summary, args.output)
    save_latex(runtime_summary, mse_summary, speedup_summary, args.output)

    max_it = int(data["iterations"].max())
    print(f"Runs analysed: {data['run'].nunique()}")
    print(f"Results saved to: {args.output.resolve()}")
    print(f"\nSummary at {max_it} iterations:")
    for method in ("PSO", "DAPSO"):
        rt = runtime_summary[(runtime_summary.method == method) & (runtime_summary.iterations == max_it)].iloc[0]
        ms = mse_summary[(mse_summary.method == method) & (mse_summary.iterations == max_it)].iloc[0]
        print(f"  {method} runtime: {rt['mean']:.2f} ± {rt['ci95']:.2f} s (95% CI half-width)")
        print(f"  {method} MSE: {ms['mean']:.3f} ± {ms['ci95']:.3f} (95% CI half-width)")
    sp = speedup_summary[speedup_summary.iterations == max_it].iloc[0]
    print(f"  Speedup: {sp['mean_speedup']:.3f} ± {sp['ci95_speedup']:.3f}")
    rt = runtime_tests[runtime_tests.iterations == max_it].iloc[0]
    mt = mse_tests[mse_tests.iterations == max_it].iloc[0]
    print(f"  Runtime paired t-test Holm p: {rt['paired_t_pvalue_holm']:.4g}")
    print(f"  Runtime Wilcoxon Holm p: {rt['wilcoxon_pvalue_holm']:.4g}")
    print(f"  MSE paired t-test Holm p: {mt['paired_t_pvalue_holm']:.4g}")
    print(f"  MSE Wilcoxon Holm p: {mt['wilcoxon_pvalue_holm']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
