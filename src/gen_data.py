import os, json, argparse
import numpy as np


def make_pairs(n, k=None, seed=0, x_range=(0, 999), y_range=(0, 999)):
    """Generate n pairs (x,y) and compute z = (x+y) % k (or just x+y if k is None)."""
    rng = np.random.default_rng(seed)
    xs = rng.integers(x_range[0], x_range[1] + 1, size=n, dtype=np.int32)
    ys = rng.integers(y_range[0], y_range[1] + 1, size=n, dtype=np.int32)
    if k is None:
        zs = xs + ys
    else:
        zs = (xs + ys) % k
    return xs, ys, zs


def inject_negatives(xs, ys, zs, neg_frac=0.5, k=None, seed=0):
    """
    Inject negatives by deliberately corrupting the z values for a fraction of the data to create invalid equations.

    Purpose:
    for classification tasks;
    This creates a dataset with a mix of valid and invalid samples, so the model can be trained to classify equations as true/false.
    """
    rng = np.random.default_rng(seed)
    n = len(xs)
    labels = np.ones(
        n, dtype=np.int32
    )  # initialize all labels as 1 (meaning "valid equation")
    n_neg = int(neg_frac * n)
    idx = rng.choice(
        n, size=n_neg, replace=False
    )  # pick n_neg number of random indices to corrupt
    labels[idx] = 0  # set labels of corrupted indices to 0 (meaning "invalid equation")
    zs_corrupt = zs.copy()
    for i in idx:
        if k is None:
            zs_corrupt[i] = zs[i] + rng.integers(1, 5)
        else:
            zs_corrupt[i] = (zs[i] + rng.integers(1, max(2, k // 4))) % (
                k if k is not None else 10**9
            )
    return labels, zs_corrupt


def textify(xs, ys, zs, task, k=None):
    lines = []
    for x, y, z in zip(xs, ys, zs):
        if task == "REG-SUM":  # supervised regression dataset
            line = f"{x} + {y} = ?\t{z}"
        elif task == "CLS-VALID":  # classification dataset (valid/invalid)
            line = f"{x} + {y} = {z}"
        elif task == "REG-MODK":  # supervised regression dataset with modulo k
            line = f"{x} + {y} (mod {k}) = ?\t{z}"
        else:
            raise ValueError("Unknown task")
        lines.append(line)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--task",
        type=str,
        default="REG-MODK",
        choices=["REG-SUM", "CLS-VALID", "REG-MODK"],
    )
    ap.add_argument("--n", type=int, default=20000)
    ap.add_argument("--k", type=int, default=97)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out_dir", type=str, default="data")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    k = None if args.task in ["REG-SUM", "CLS-VALID"] else args.k
    xs, ys, zs_true = make_pairs(args.n, k=k, seed=args.seed)

    if args.task == "CLS-VALID":
        labels, zs_in = inject_negatives(
            xs, ys, zs_true, neg_frac=0.5, k=k, seed=args.seed + 1
        )
        split = int(0.9 * args.n)
        with open(
            os.path.join(args.out_dir, f"{args.task}_train.jsonl"), "w"
        ) as ftr, open(
            os.path.join(args.out_dir, f"{args.task}_val.jsonl"), "w"
        ) as fva:
            for i in range(args.n):
                obj = {
                    "x": int(xs[i]),
                    "y": int(ys[i]),
                    "z": int(zs_in[i]),
                    "label": int(labels[i]),
                }
                (ftr if i < split else fva).write(json.dumps(obj) + "\n")
    else:
        lines = textify(xs, ys, zs_true, task=args.task, k=k)
        split = int(0.9 * args.n)
        with open(
            os.path.join(args.out_dir, f"{args.task}_train.tsv"), "w"
        ) as ftr, open(os.path.join(args.out_dir, f"{args.task}_val.tsv"), "w") as fva:
            for i, line in enumerate(lines):
                (ftr if i < split else fva).write(line + "\n")


if __name__ == "__main__":
    main()
