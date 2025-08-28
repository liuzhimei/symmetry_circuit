# src/multidim_plots.py
import os, json, argparse
import numpy as np
import matplotlib.pyplot as plt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="REG-MODK")
    ap.add_argument("--output_dir", default="analysis/output")
    args = ap.parse_args()

    path = os.path.join(args.output_dir, f"{args.task}_multidim_results.json")
    R = json.load(open(path))

    if args.task == "REG-SUM":
        # Scalar summary
        print(f"Scalar probe R^2: {R['scalar_probe_r2']:.4f}")
        print(f"Base MSE: {R['base_mse']:.2f}")
        print(f"ΔMSE (probe 1D): {R['delta_mse_probe_1d']:.2f}")
        print(f"ΔMSE (random 1D): {R['delta_mse_random_1d']:.2f}")
        print(f"Excess ΔMSE: {R['excess_delta_mse']:.2f}")

        plt.figure()
        vals = [0.0, R["delta_mse_probe_1d"], R["delta_mse_random_1d"]]
        labels = ["Baseline", "Proj probe (1D)", "Proj random (1D)"]
        plt.bar(labels, vals)
        plt.ylabel("ΔMSE")
        plt.title("REG-SUM: causal ablation (1D)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, "REG-SUM_scalar_causal.png"), dpi=300)
        print("Saved REG-SUM_scalar_causal.png")
        return

    # REG-MODK plots

    # Fourier R^2 vs dimension
    xs = [e["dim"] for e in R["fourier_r2_curve"]]
    ys = [e["r2"] for e in R["fourier_r2_curve"]]
    plt.figure(figsize=(5, 4))
    plt.plot(xs, ys, marker="o")
    plt.xlabel("Probe dimension (2N)")
    plt.ylabel("Macro $R^2$ (Fourier targets)")
    plt.title(f"{args.task}: Fourier probe $R^2$ vs dimension")
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, f"{args.task}_fourier_r2_vs_dim.png"), dpi=300
    )

    # ΔMSE / random / excess
    dm = R["causal"]["delta_mse_vs_dim"]
    dmr = R["causal"]["delta_mse_random_vs_dim"]
    dmex = R["causal"]["excess_delta_mse_vs_dim"]
    xs = [e["r"] for e in dm]  # use actual removed rank r on x-axis
    y_probe = [e["delta_mse"] for e in dm]
    y_rand = [e["delta_mse"] for e in dmr]
    y_excess = [e["excess_delta_mse"] for e in dmex]
    plt.figure(figsize=(5, 4))
    plt.plot(xs, y_probe, marker="o", label="Probe subspace")
    plt.plot(xs, y_rand, marker="x", label="Random subspace")
    plt.plot(xs, y_excess, marker="s", label="Excess (probe − random)")
    plt.xlabel("Removed subspace dimension (rank r)")
    plt.ylabel("ΔMSE")
    plt.title(f"{args.task}: causal harm vs dimension")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.output_dir, f"{args.task}_delta_mse_vs_dim.png"), dpi=300
    )

    # Small-k R^2
    sk = R["smallk_r2"]
    ks = sorted(set(e["kprime"] for e in sk))
    Ns = sorted(set(e["N"] for e in sk))
    plt.figure(figsize=(5, 4))
    for N in Ns:
        xs = []
        ys = []
        for kprime in ks:
            vals = [e for e in sk if e["kprime"] == kprime and e["N"] == N]
            if not vals:
                continue
            xs.append(kprime)
            ys.append(vals[0]["r2"])
        plt.plot(xs, ys, marker="o", label=f"N={N}")
    plt.xlabel("k′ (coarse modulus)")
    plt.ylabel("Macro $R^2$")
    plt.title(f"{args.task}: small-k probes")
    plt.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, f"{args.task}_smallk_r2.png"), dpi=300)

    print("Saved plots to", args.output_dir)


if __name__ == "__main__":
    main()
