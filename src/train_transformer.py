import argparse, os
from model import train_regression, train_classification

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="REG-SUM")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--out_dir", type=str, default="data")
    ap.add_argument("--d_model", type=int, default=128)
    ap.add_argument("--n_layer", type=int, default=2)
    ap.add_argument("--n_head", type=int, default=4)
    ap.add_argument("--d_ff", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--bs", type=int, default=128)
    ap.add_argument("--lr", type=float, default=2e-3)
    args = ap.parse_args()
    if args.task in ("REG-SUM", "REG-MODK"):
        train_regression(args)
    else:
        train_classification(args)
