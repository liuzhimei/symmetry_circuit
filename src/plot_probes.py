# src/plot_probes.py
import json
import os
import argparse
import matplotlib.pyplot as plt


def plot_probe_results(results_path, task="REG-SUM", out_path=None):
    with open(results_path) as f:
        results = json.load(f)

    if "pooled_r2" not in results or "pos_r2" not in results:
        raise KeyError(
            f"{results_path} does not contain 'pooled_r2'/'pos_r2'. "
            "Are you sure this is the linear-probe results file?"
        )

    pooled_r2 = float(results["pooled_r2"])
    pos_r2 = list(results["pos_r2"])

    # task-specific title
    if task == "REG-SUM":
        title = "Linear Probe: Sum recoverability across positions"
        ylabel = "R² (sum)"
    elif task == "REG-MODK":
        title = "Linear Probe: Residue recoverability across positions (mod k)"
        ylabel = "R² (residue)"
    else:
        title = f"Linear Probe: Recoverability across positions ({task})"
        ylabel = "R²"

    plt.figure(figsize=(6, 4))
    plt.plot(range(len(pos_r2)), pos_r2, marker="o", label="Position-wise R²")
    plt.axhline(pooled_r2, linestyle="--", label=f"Pooled R² = {pooled_r2:.3f}")
    plt.axhline(0.0, color="gray", linewidth=0.8)  # baseline
    plt.xlabel("Token position")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        plt.savefig(out_path, dpi=300)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task", type=str, default="REG-SUM", choices=["REG-SUM", "REG-MODK"]
    )
    ap.add_argument("--result_dir", type=str, default="src/output")
    ap.add_argument("--figure_dir", type=str, default="src/figures")
    args = ap.parse_args()
    results_path = os.path.join(args.result_dir, f"{args.task}_probe_results.json")
    figure_path = os.path.join(args.figure_dir, f"{args.task}_probe_curve.png")

    plot_probe_results(results_path, task=args.task, out_path=figure_path)
    print(f"Plotted results from {results_path} and saved to {figure_path}")
