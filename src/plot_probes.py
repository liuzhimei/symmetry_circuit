import json, os
import matplotlib.pyplot as plt


def plot_probe_results(results_path, out_path=None):
    results = json.load(open(results_path))
    pooled_r2 = results["pooled_r2"]
    pos_r2 = results["pos_r2"]

    # --- Position-wise R² curve ---
    plt.figure(figsize=(7, 4))
    plt.plot(range(len(pos_r2)), pos_r2, marker="o", label="Position-wise R²")
    plt.axhline(
        pooled_r2, color="red", linestyle="--", label=f"Pooled R²={pooled_r2:.3f}"
    )
    plt.xlabel("Token position")
    plt.ylabel("R² (linear probe)")
    plt.title("Linear Probe: Sum recoverability across positions")
    plt.legend()
    plt.tight_layout()

    if out_path:
        plt.savefig(out_path, dpi=150)
        print(f"Saved plot to {out_path}")
    else:
        plt.show()


if __name__ == "__main__":
    results_path = "analysis/REG-SUM_probe_results.json"
    plot_probe_results(results_path, out_path="analysis/REG-SUM_probe_curve.png")
