import os, argparse, json, numpy as np, re
import torch
from sklearn.linear_model import Ridge
from model import TinyTransformer, VOCAB, encode, pad_batch

def load_batch(tsv_path, n=512):
    Xs, ys, raws = [], [], []
    with open(tsv_path) as f:
        for i, line in enumerate(f):
            if i>=n: break
            line=line.strip()
            if not line: continue
            x, y = line.split("\t")
            Xs.append(encode(x)); ys.append(float(y)); raws.append(x)
    X, mask = pad_batch(Xs, pad_id=0)
    return X, np.array(ys, dtype=np.float32), raws

def projection_ablation(h, v):
    v = v / (np.linalg.norm(v) + 1e-8)
    return h - (h @ v[:,None]) * v[None,:]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", type=str, default="REG-SUM")
    ap.add_argument("--data_dir", type=str, default="data")
    ap.add_argument("--ckpt", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    val_path = os.path.join(args.data_dir, f"{args.task}_val.tsv")
    X, y_true, raws = load_batch(val_path, n=1024)
    X = X.to(device)

    m = TinyTransformer(vocab_size=len(VOCAB)).to(device)
    if args.ckpt is None: args.ckpt = os.path.join(args.data_dir, f"{args.task}_best.pt")
    m.load_state_dict(torch.load(args.ckpt, map_location=device)); m.eval()

    with torch.no_grad(): y_pred, h = m(X, return_h=True)
    pooled = h[:,-1,:].detach().cpu().numpy()

    sums = []
    for line in raws:
        msum = re.match(r"(\d+)\s\+\s(\d+)", line)
        x = int(msum.group(1)); yv = int(msum.group(2))
        sums.append(x+yv)
    sums = np.array(sums, dtype=np.float32)

    # Probe direction for sum
    reg = Ridge(alpha=1.0).fit(pooled, sums)
    v = reg.coef_.astype(np.float32)

    # Estimate effect by refitting a ridge head before/after projection
    head = Ridge(alpha=1.0).fit(pooled, y_true)
    mse = ((head.predict(pooled)-y_true)**2).mean()

    pooled_abl = projection_ablation(pooled, v)
    head_abl = Ridge(alpha=1.0).fit(pooled_abl, y_true)
    mse_abl = ((head_abl.predict(pooled_abl)-y_true)**2).mean()

    results = {
        "ridge_head_mse": float(mse),
        "ridge_head_mse_after_proj_out_sumdir": float(mse_abl),
        "delta_mse": float(mse_abl - mse)
    }
    out_path = os.path.join(args.data_dir, f"{args.task}_patch_results.json")
    json.dump(results, open(out_path,"w"), indent=2)
    print("Saved to", out_path)
    print(results)

if __name__ == "__main__":
    main()
