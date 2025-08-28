import os, json, argparse, itertools
from types import SimpleNamespace
from src.model import train_regression  # existing trainer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="REG-MODK", choices=["REG-SUM", "REG-MODK"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--out_root", default="analysis/models_scale")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seeds", type=int, nargs="*", default=[0])
    # grids
    ap.add_argument("--depths", type=int, nargs="*", default=[2, 4, 6])
    ap.add_argument("--dmodels", type=int, nargs="*", default=[128, 256, 512])
    args = ap.parse_args()

    os.makedirs(args.out_root, exist_ok=True)
    grid = list(itertools.product(args.depths, args.dmodels, args.seeds))

    def pick_heads(d_model):
        for h in [4, 8, 16, 32]:
            if d_model % h == 0 and d_model // h >= 16:  # keep head dim >=16
                return h
        return 4

    jobs = []
    for depth, d_model, seed in grid:
        n_head = pick_heads(d_model)
        d_ff = 4 * d_model
        tag = f"{args.task}_L{depth}_D{d_model}_H{n_head}_S{seed}"
        out_dir = os.path.join(args.out_root, tag)
        os.makedirs(out_dir, exist_ok=True)
        jobs.append((depth, d_model, n_head, d_ff, seed, out_dir, tag))

    for depth, d_model, n_head, d_ff, seed, out_dir, tag in jobs:
        ns = SimpleNamespace(
            task=args.task,
            data_dir=args.data_dir,
            out_dir=out_dir,
            d_model=d_model,
            n_layer=depth,
            n_head=n_head,
            d_ff=d_ff,
            bs=args.bs,
            lr=args.lr,
            epochs=args.epochs,
        )
        print(f"==> Training {tag}")
        train_regression(ns)
        with open(os.path.join(out_dir, "meta.json"), "w") as f:
            json.dump(
                {
                    "task": args.task,
                    "depth": depth,
                    "d_model": d_model,
                    "n_head": n_head,
                    "d_ff": d_ff,
                    "seed": seed,
                },
                f,
                indent=2,
            )


if __name__ == "__main__":
    main()
