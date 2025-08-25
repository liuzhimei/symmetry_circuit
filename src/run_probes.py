import os, argparse, json, re
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from .model import TinyTransformer, VOCAB, PAD_ID, encode, pad_batch


def load_data(tsv_path):
    Xs, ys, raws = [], [], []
    with open(tsv_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            x, y = line.split("\t")
            Xs.append(encode(x))  # tensor of token IDs for the input string
            ys.append(float(y))  # numeric target (z), not used for probing here
            raws.append(x)  # keep the raw string "123 + 45 = ?"
    return Xs, np.array(ys, dtype=np.float32), raws


def collate(seqs):
    X, mask, lengths = pad_batch(seqs, pad_id=PAD_ID)
    return X, mask, lengths


def train_probe(H, targets, alpha=1.0):
    """Train a ridge regression probe on features H to predict targets.

    H: (B, D) array of features
    targets: (B,) array of float targets
    alpha: ridge regularization strength
    Returns: trained regressor, R^2 score on training data"""
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(H, targets)
    pred = reg.predict(H)
    return reg, r2_score(targets, pred)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="REG-SUM")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--output_dir", type=str, default="analysis")
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tsv = os.path.join(args.data_dir, f"{args.task}_val.tsv")
    Xs, ys, raws = load_data(tsv)
    X, mask, lengths = collate(Xs)
    X, lengths = X.to(device), lengths.to(device)

    model = TinyTransformer(vocab_size=len(VOCAB)).to(device)
    ckpt = os.path.join(args.checkpoint_dir, f"{args.task}_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    with torch.no_grad():
        pred, h = model(X, lengths=lengths, return_h=True)
    B, T, D = h.shape

    # Build target sums from raw strings
    sums = []
    for line in raws:
        msum = re.match(r"(\d+)\s\+\s(\d+)", line)
        assert msum, f"bad line: {line}"
        x = int(msum.group(1))
        yv = int(msum.group(2))
        sums.append(x + yv)
    sums = np.array(sums, dtype=np.float32)

    results = {}
    # pooled features at last REAL token
    idx = (lengths - 1).clamp(min=0)  # [B]
    H_pooled = h[torch.arange(B, device=h.device), idx, :].cpu().numpy()
    reg, r2 = train_probe(H_pooled, sums, alpha=args.alpha)
    results["pooled_r2"] = float(r2)

    # position-wise
    pos_r2 = []
    for t in range(T):
        Ht = h[:, t, :].detach().cpu().numpy()
        _, r2t = train_probe(Ht, sums, alpha=args.alpha)
        pos_r2.append(float(r2t))
    results["pos_r2"] = pos_r2

    out_path = os.path.join(args.output_dir, f"{args.task}_probe_results.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print("Saved probe results to", out_path)
    print(results)


if __name__ == "__main__":
    main()
