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

FILES = {"PSO": "resultados_pso.txt", "DAPSO": "resultados_dapso.txt"}
PATTERN = re.compile(
    r"\(\s*\(*\s*,?\s*(\d+)\s*,\s*"
    r"([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
    r"\s*,?\s*\)*\s*\)"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Analyse repeated PSO/DAPSO dataset-scaling experiments.")
    p.add_argument("--input", type=Path, default=Path("results/repetitions/dataset_scaling"))
    p.add_argument("--output", type=Path, default=Path("results/analysis/dataset_scaling"))
    p.add_argument("--alpha", type=float, default=0.05)
    return p.parse_args()


def parse_file(path: Path) -> dict[int, float]:
    values: dict[int, float] = {}
    for line_no, raw in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = raw.strip()
        if not line:
            continue
        m = PATTERN.fullmatch(line) or PATTERN.search(line)
        if not m:
            raise ValueError(f"Cannot parse {path}, line {line_no}: {raw!r}")
        size, runtime = int(m.group(1)), float(m.group(2))
        if size in values:
            raise ValueError(f"Repeated dataset size {size} in {path}")
        values[size] = runtime
    if not values:
        raise ValueError(f"No observations found in {path}")
    return values


def ci95(series: pd.Series) -> float:
    x = series.dropna().astype(float).to_numpy()
    if len(x) < 2:
        return math.nan
    return float(stats.t.ppf(0.975, len(x) - 1) * stats.sem(x))


def holm(pvalues: pd.Series) -> pd.Series:
    p = pvalues.astype(float).to_numpy()
    order = np.argsort(p)
    sorted_p = p[order]
    adj_sorted = np.empty_like(sorted_p)
    running = 0.0
    n = len(sorted_p)
    for rank, value in enumerate(sorted_p):
        running = max(running, (n - rank) * value)
        adj_sorted[rank] = min(running, 1.0)
    adjusted = np.empty_like(adj_sorted)
    adjusted[order] = adj_sorted
    return pd.Series(adjusted, index=pvalues.index)


def load_runs(input_dir: Path) -> pd.DataFrame:
    run_dirs = sorted(p for p in input_dir.glob("run_*") if p.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run_* directories under {input_dir.resolve()}")

    rows = []
    for run_dir in run_dirs:
        parsed = {}
        for method, filename in FILES.items():
            path = run_dir / filename
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")
            parsed[method] = parse_file(path)
        if set(parsed["PSO"]) != set(parsed["DAPSO"]):
            raise ValueError(f"Dataset-size mismatch in {run_dir.name}")
        for method in ("PSO", "DAPSO"):
            for size in sorted(parsed[method]):
                rows.append({"run": run_dir.name, "method": method, "dataset_size": size,
                             "runtime_s": parsed[method][size]})
    return pd.DataFrame(rows).sort_values(["run", "dataset_size", "method"]).reset_index(drop=True)


def runtime_summary(data: pd.DataFrame) -> pd.DataFrame:
    s = data.groupby(["method", "dataset_size"], as_index=False).agg(
        n=("runtime_s", "count"), mean_runtime=("runtime_s", "mean"),
        std_runtime=("runtime_s", "std"), median_runtime=("runtime_s", "median"),
        min_runtime=("runtime_s", "min"), max_runtime=("runtime_s", "max"))
    s["cv_runtime"] = s["std_runtime"] / s["mean_runtime"]
    ci = data.groupby(["method", "dataset_size"])["runtime_s"].apply(ci95).rename("ci95_runtime").reset_index()
    return s.merge(ci, on=["method", "dataset_size"])


def paired_data(data: pd.DataFrame) -> pd.DataFrame:
    return data.pivot(index=["run", "dataset_size"], columns="method", values="runtime_s").reset_index()


def speedup_summary(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    raw = paired_data(data)
    raw["speedup"] = raw["PSO"] / raw["DAPSO"]
    s = raw.groupby("dataset_size", as_index=False).agg(
        n=("speedup", "count"), mean_speedup=("speedup", "mean"),
        std_speedup=("speedup", "std"), median_speedup=("speedup", "median"),
        min_speedup=("speedup", "min"), max_speedup=("speedup", "max"))
    ci = raw.groupby("dataset_size")["speedup"].apply(ci95).rename("ci95_speedup").reset_index()
    return raw, s.merge(ci, on="dataset_size")


def paired_tests(data: pd.DataFrame) -> pd.DataFrame:
    wide = paired_data(data)
    rows = []
    for size, g in wide.groupby("dataset_size"):
        pso = g["PSO"].to_numpy(float)
        dapso = g["DAPSO"].to_numpy(float)
        diff = pso - dapso
        t = stats.ttest_rel(pso, dapso, alternative="greater")
        try:
            w = stats.wilcoxon(pso, dapso, alternative="greater", method="auto")
            w_stat, w_p = float(w.statistic), float(w.pvalue)
        except ValueError:
            w_stat, w_p = math.nan, 1.0
        sd = float(np.std(diff, ddof=1)) if len(diff) > 1 else math.nan
        rows.append({
            "dataset_size": int(size), "n": len(g),
            "mean_pso": float(np.mean(pso)), "mean_dapso": float(np.mean(dapso)),
            "mean_difference_pso_minus_dapso": float(np.mean(diff)),
            "paired_t_statistic": float(t.statistic), "paired_t_pvalue": float(t.pvalue),
            "wilcoxon_statistic": w_stat, "wilcoxon_pvalue": w_p,
            "cohen_dz": float(np.mean(diff) / sd) if sd and not math.isnan(sd) else math.nan,
        })
    result = pd.DataFrame(rows).sort_values("dataset_size").reset_index(drop=True)
    result["paired_t_pvalue_holm"] = holm(result["paired_t_pvalue"])
    result["wilcoxon_pvalue_holm"] = holm(result["wilcoxon_pvalue"])
    return result


def plot_runtime(summary: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(7.2, 4.5))
    for method in ("PSO", "DAPSO"):
        d = summary[summary.method == method].sort_values("dataset_size")
        plt.errorbar(d.dataset_size, d.mean_runtime, yerr=d.ci95_runtime,
                     marker="o", capsize=3, linewidth=1.4, label=method)
    plt.xlabel("Training dataset size (samples)")
    plt.ylabel("Runtime (s)")
    plt.title("Dataset-scaling runtime over repeated executions")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / "dataset_scaling_runtime_mean_ci95.pdf")
    plt.savefig(output / "dataset_scaling_runtime_mean_ci95.png", dpi=600)
    plt.close()


def plot_speedup(summary: pd.DataFrame, output: Path) -> None:
    plt.figure(figsize=(7.2, 4.5))
    plt.errorbar(summary.dataset_size, summary.mean_speedup, yerr=summary.ci95_speedup,
                 marker="o", capsize=3, linewidth=1.4, label="PSO / DAPSO")
    plt.axhline(1.0, linestyle="--", linewidth=1.0, label="Baseline")
    plt.xlabel("Training dataset size (samples)")
    plt.ylabel("Speedup")
    plt.title("DAPSO speedup with increasing dataset size")
    plt.grid(True, alpha=0.3)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(output / "dataset_scaling_speedup_mean_ci95.pdf")
    plt.savefig(output / "dataset_scaling_speedup_mean_ci95.png", dpi=600)
    plt.close()


def save_latex(runtime: pd.DataFrame, speedup: pd.DataFrame, tests: pd.DataFrame, output: Path) -> None:
    lines = [r"\begin{tabular}{rccc}", r"\hline",
             r"Samples & PSO runtime & DAPSO runtime & Speedup \\", r"\hline"]
    for size in sorted(runtime.dataset_size.unique()):
        p = runtime[(runtime.method == "PSO") & (runtime.dataset_size == size)].iloc[0]
        d = runtime[(runtime.method == "DAPSO") & (runtime.dataset_size == size)].iloc[0]
        s = speedup[speedup.dataset_size == size].iloc[0]
        lines.append(f"{size} & {p.mean_runtime:.2f} $\\pm$ {p.ci95_runtime:.2f} & "
                     f"{d.mean_runtime:.2f} $\\pm$ {d.ci95_runtime:.2f} & "
                     f"{s.mean_speedup:.2f} $\\pm$ {s.ci95_speedup:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (output / "dataset_scaling_runtime_speedup_table.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [r"\begin{tabular}{rccc}", r"\hline",
             r"Samples & $p_t$ & $p_W$ & Cohen's $d_z$ \\", r"\hline"]
    for _, row in tests.iterrows():
        lines.append(f"{int(row.dataset_size)} & {row.paired_t_pvalue_holm:.3g} & "
                     f"{row.wilcoxon_pvalue_holm:.3g} & {row.cohen_dz:.2f} \\\\")
    lines += [r"\hline", r"\end{tabular}"]
    (output / "dataset_scaling_significance_table.tex").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    data = load_runs(args.input)
    runtime = runtime_summary(data)
    speedup_raw, speedup = speedup_summary(data)
    tests = paired_tests(data)

    data.to_csv(args.output / "dataset_scaling_all_runs.csv", index=False)
    runtime.to_csv(args.output / "dataset_scaling_runtime_summary.csv", index=False)
    speedup_raw.to_csv(args.output / "dataset_scaling_speedup_all_runs.csv", index=False)
    speedup.to_csv(args.output / "dataset_scaling_speedup_summary.csv", index=False)
    tests.to_csv(args.output / "dataset_scaling_significance.csv", index=False)

    plot_runtime(runtime, args.output)
    plot_speedup(speedup, args.output)
    save_latex(runtime, speedup, tests, args.output)

    max_size = int(data.dataset_size.max())
    print(f"Runs analysed: {data['run'].nunique()}")
    print(f"Results saved to: {args.output.resolve()}")
    print(f"\nSummary at {max_size} samples:")
    for method in ("PSO", "DAPSO"):
        row = runtime[(runtime.method == method) & (runtime.dataset_size == max_size)].iloc[0]
        print(f"  {method} runtime: {row.mean_runtime:.2f} ± {row.ci95_runtime:.2f} s")
    s = speedup[speedup.dataset_size == max_size].iloc[0]
    print(f"  Speedup: {s.mean_speedup:.3f} ± {s.ci95_speedup:.3f}")
    t = tests[tests.dataset_size == max_size].iloc[0]
    print(f"  Paired t-test Holm p: {t.paired_t_pvalue_holm:.4g}")
    print(f"  Wilcoxon Holm p: {t.wilcoxon_pvalue_holm:.4g}")
    print(f"  Cohen's dz: {t.cohen_dz:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

