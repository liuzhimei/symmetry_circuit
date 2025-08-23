import os, argparse, json, re, numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from model import TinyTransformer, VOCAB, encode, pad_batch

def load_data(tsv_path):
    Xs, ys, raws = [], [], []
    with open(tsv_path) as f:
        for line in f:
            line=line.strip()
            if not line: continue
            x, y = line.split("\t")
            Xs.append(encode(x))
            ys.append(float(y))
            raws.append(x)
    return Xs, np.array(ys, dtype=np.float32), raws

def collate(seqs):
    from model import pad_batch
    X, mask = pad_batch(seqs, pad_id=0)
    return X, mask

def train_probe(H, targets, alpha=1.0):
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(H, targets)
    pred = reg.predict(H)
    return reg, r2_score(targets, pred)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="REG-SUM")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--ckpt", type=str, default=None)
    ap.add_argument("--alpha", type=float, default=1.0)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tsv = os.path.join(args.data_dir, f"{args.task}_val.tsv")
    Xs, ys, raws = load_data(tsv)
    X, mask = collate(Xs)
    X = X.to(device)

    m = TinyTransformer(vocab_size=len(VOCAB)).to(device)
    if args.ckpt is None: args.ckpt = os.path.join(args.data_dir, f"{args.task}_best.pt")
    m.load_state_dict(torch.load(args.ckpt, map_location=device)); m.eval()

    with torch.no_grad():
        pred, h = m(X, return_h=True)
    B,T,D = h.shape

    # Build target sums from raw strings
    sums = []
    for line in raws:
        msum = re.match(r"(\d+)\s\+\s(\d+)", line)
        assert msum, f"bad line: {line}"
        x = int(msum.group(1)); yv = int(msum.group(2))
        sums.append(x+yv)
    sums = np.array(sums, dtype=np.float32)

    results = {}
    H_pooled = h[:,-1,:].detach().cpu().numpy()
    reg, r2 = train_probe(H_pooled, sums, alpha=args.alpha)
    results["pooled_r2"] = float(r2)

    pos_r2 = []
    for t in range(T):
        Ht = h[:,t,:].detach().cpu().numpy()
        _, r2t = train_probe(Ht, sums, alpha=args.alpha)
        pos_r2.append(float(r2t))
    results["pos_r2"] = pos_r2

    out_path = os.path.join(args.data_dir, f"{args.task}_probe_results.json")
    json.dump(results, open(out_path,"w"), indent=2)
    print("Saved probe results to", out_path)
    print(results)

if __name__ == "__main__":
    main()
