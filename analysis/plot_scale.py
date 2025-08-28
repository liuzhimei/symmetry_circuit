import os, json, argparse, numpy as np, matplotlib.pyplot as plt


def load_index(path):
    return json.load(open(path))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="REG-MODK", choices=["REG-SUM", "REG-MODK"])
    ap.add_argument("--analysis_dir", default="analysis/analysis_scale")
    args = ap.parse_args()

    idx_path = os.path.join(args.analysis_dir, f"{args.task}_scale_index.json")
    runs = load_index(idx_path)
    runs = sorted(runs, key=lambda r: r["params"])

    params = [r["params"] for r in runs]

    if args.task == "REG-SUM":
        r2 = [r["sum"]["r2"] for r in runs]
        dprobe = [r["sum"]["delta_probe"] for r in runs]
        drand = [r["sum"]["delta_rand"] for r in runs]

        plt.figure()
        plt.plot(params, r2, marker="o")
        plt.xscale("log")
        plt.ylim(0.95, 1.001)
        plt.xlabel("# params (log)")
        plt.ylabel("Ridge $R^2$")
        plt.title("REG-SUM: linear probe $R^2$ vs scale")
        plt.tight_layout()
        plt.savefig(os.path.join(args.analysis_dir, "SUM_r2_vs_scale.png"), dpi=150)

        plt.figure()
        plt.plot(params, dprobe, marker="o", label="probe axis")
        plt.plot(params, drand, marker="x", label="random axis")
        plt.xscale("log")
        plt.xlabel("# params (log)")
        plt.ylabel("ΔMSE (1D ablation)")
        plt.title("REG-SUM: causal ΔMSE vs scale")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.analysis_dir, "SUM_delta_vs_scale.png"), dpi=150)
        print("Saved plots:", args.analysis_dir)
        return

    # REG-MODK: choose a few N to display
    for Npick in [1, 3, 6, 12]:
        xs = []
        ys = []
        for r in runs:
            ent = next((e for e in r["modk"]["fourier"] if e["N"] == Npick), None)
            if ent:
                xs.append(r["params"])
                ys.append(ent["r2"])
        if xs:
            plt.figure()
            plt.plot(xs, ys, marker="o")
            plt.xscale("log")
            plt.xlabel("# params (log)")
            plt.ylabel(f"Macro $R^2$ (N={Npick})")
            plt.title(f"REG-MODK: Fourier probe $R^2$ vs scale (N={Npick})")
            plt.tight_layout()
            plt.savefig(
                os.path.join(args.analysis_dir, f"MODK_fourier_r2_scale_N{Npick}.png"),
                dpi=150,
            )

    # Excess ΔMSE at a mid N (e.g., 6)
    xs = []
    yprobe = []
    yrand = []
    yex = []
    for r in runs:
        ent = next((e for e in r["modk"]["causal"] if e["N"] == 6), None)
        if ent:
            xs.append(r["params"])
            yprobe.append(ent["delta_probe"])
            yrand.append(ent["delta_rand"])
            yex.append(ent["excess_delta"])
    if xs:
        plt.figure()
        plt.plot(xs, yprobe, marker="o", label="probe subspace")
        plt.plot(xs, yrand, marker="x", label="random subspace")
        plt.plot(xs, yex, marker="s", label="excess (probe - random)")
        plt.xscale("log")
        plt.xlabel("# params (log)")
        plt.ylabel("ΔMSE")
        plt.title("REG-MODK: causal harm vs scale (N=6)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(args.analysis_dir, "MODK_delta_vs_scale.png"), dpi=150)

    # MLP accuracy vs scale
    xs = []
    acc = []
    for r in runs:
        xs.append(r["params"])
        acc.append(r["modk"]["mlp"]["cls_acc"])
    plt.figure()
    plt.plot(xs, acc, marker="o")
    plt.xscale("log")
    plt.xlabel("# params (log)")
    plt.ylabel("MLP residue accuracy")
    plt.title("REG-MODK: nonlinear probe accuracy vs scale")
    plt.tight_layout()
    plt.savefig(os.path.join(args.analysis_dir, "MODK_mlp_acc_vs_scale.png"), dpi=150)

    print("Saved plots:", args.analysis_dir)


if __name__ == "__main__":
    main()
