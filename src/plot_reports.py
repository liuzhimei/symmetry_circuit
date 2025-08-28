import os, json, argparse, re
import numpy as np
import torch
import matplotlib.pyplot as plt

from .model import TinyTransformer, VOCAB, encode, pad_batch, PAD_ID


def load_tsv(tsv):
    Xs, ys, raws = [], [], []
    with open(tsv) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = line.split("\t")
            Xs.append(encode(x))
            ys.append(float(y))
            raws.append(x)
    X, _, lengths = pad_batch(Xs, pad_id=PAD_ID)
    return X, np.array(ys, dtype=np.float32), raws, lengths


def eval_model(ckpt, X, lengths, device):
    model = TinyTransformer(vocab_size=len(VOCAB)).to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()
    with torch.no_grad():
        pred = model(X.to(device), lengths=lengths.to(device)).cpu().numpy()
    return pred


def invariance_augment(raws, deltas):
    # returns list of transformed strings for each delta
    def shift_xy(s, d):
        # parse "(\d+)\s\+\s(\d+)"
        m = re.match(r"(\d+)\s\+\s(\d+)", s)
        x, y = int(m.group(1)), int(m.group(2))
        x2, y2 = x + d, y - d
        # rebuild left side; keep any suffix after the match (e.g. " = ?")
        rest = s[m.end() :]
        return f"{x2} + {y2}{rest}"

    grids = [[shift_xy(s, d) for s in raws] for d in deltas]
    return grids


def main():
    S = 2000.0  # same scale factor used in collate()

    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="REG-SUM")
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--out_dir", default="src/figures")
    ap.add_argument("--deltas", type=int, nargs="+", default=[-5, -2, -1, 0, 1, 2, 5])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Loss curve
    hist_path = os.path.join(args.checkpoint_dir, f"{args.task}_train_history.json")
    if os.path.exists(hist_path):
        hist = json.load(open(hist_path))
        epochs = [h["epoch"] for h in hist]
        tr = [h["train_mse"] for h in hist]
        va = [h["val_mse"] for h in hist]
        plt.figure()
        plt.plot(epochs, tr, label="train MSE")
        plt.plot(epochs, va, label="val MSE")
        plt.xlabel("epoch")
        plt.ylabel("MSE")
        plt.legend()
        plt.title("Train vs Val Loss")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{args.task}_loss_curve.png"))

    # 2–4) Parity + residuals
    val_tsv = os.path.join(args.data_dir, f"{args.task}_val.tsv")
    ckpt = os.path.join(args.checkpoint_dir, f"{args.task}_best.pt")
    X, y_true, raws, lengths = load_tsv(val_tsv)
    y_pred = eval_model(ckpt, X, lengths, device)
    y_pred = y_pred * S  # rescale predictions back to original units

    resid = y_pred - y_true

    # Parity
    plt.figure()
    plt.scatter(y_true, y_pred, s=5, alpha=0.8, label="predictions")
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    plt.plot(lims, lims, "r--", label="y=x")
    plt.xlabel("True")
    plt.ylabel("Pred")
    plt.title("Parity (val)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.task}_parity_val.png"))

    # Residual histogram
    plt.figure()
    plt.hist(resid, bins=40)
    plt.xlabel("Residual")
    plt.ylabel("Count")
    plt.title("Residuals (val)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.task}_residual_hist.png"))

    # Residual vs target
    plt.figure()
    plt.scatter(y_true, resid, s=6)
    plt.xlabel("True")
    plt.ylabel("Residual")
    plt.title("Residual vs True (val)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.task}_residual_vs_true.png"))

    # 5) Invariance curve
    deltas = args.deltas
    grids = invariance_augment(raws, deltas)
    errs = []
    for gs in grids:
        Xg, _, lengthsg = pad_batch([encode(s) for s in gs], pad_id=0)
        yp = eval_model(ckpt, Xg, lengthsg, device)
        yp = yp * S
        # expected output should remain equal to original y_true if symmetry holds
        errs.append(np.mean((yp - y_true) ** 2))
    plt.figure()
    plt.plot(deltas, errs, marker="o")
    plt.xlabel("delta")
    plt.ylabel("MSE vs original target")
    plt.title("Invariance under (x,y)->(x+δ,y−δ)")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out_dir, f"{args.task}_invariance_curve.png"))

    # 6) Probe R2 by position (read JSON from run_probes.py if present)
    probe_path = os.path.join(args.data_dir, f"{args.task}_probe_results.json")
    if os.path.exists(probe_path):
        pr = json.load(open(probe_path))
        pos_r2 = pr.get("pos_r2", [])
        plt.figure()
        plt.plot(range(len(pos_r2)), pos_r2)
        plt.xlabel("position")
        plt.ylabel("R^2 (sum probe)")
        plt.title("Probe R^2 by Position")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{args.task}_probe_r2_by_pos.png"))

    # 7) ΔMSE bar (read JSON from run_patching.py if present)
    patch_path = os.path.join(args.data_dir, f"{args.task}_patch_results.json")
    if os.path.exists(patch_path):
        pr = json.load(open(patch_path))
        a = pr["ridge_head_mse"]
        b = pr["ridge_head_mse_after_proj_out_sumdir"]
        plt.figure()
        plt.bar([0, 1], [a, b])
        plt.xticks([0, 1], ["before", "after"])
        plt.ylabel("MSE")
        plt.title("Effect of Projecting Out Sum-Direction")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{args.task}_delta_mse_bar.png"))


if __name__ == "__main__":
    main()
