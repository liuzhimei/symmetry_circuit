import os, re, json, argparse, numpy as np, torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from .run_nonlinear_probes import train_mlp_reg, train_mlp_cls
from src.model import TinyTransformer, VOCAB, encode, pad_batch, PAD_ID


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


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


def parse_xy(raws):
    xs, ys = [], []
    for s in raws:
        m = re.match(r"(\d+)\s\+\s(\d+)", s)
        assert m, f"bad line: {s}"
        xs.append(int(m.group(1)))
        ys.append(int(m.group(2)))
    return np.array(xs), np.array(ys)


@torch.no_grad()
def pooled_hidden(model, X, lengths, device):
    y, h = model(X.to(device), lengths=lengths.to(device), return_h=True)
    B = h.shape[0]
    idx = (lengths - 1).clamp(min=0)
    return h[torch.arange(B, device=h.device), idx, :].cpu().numpy()


def fourier_targets(theta, N):
    cols = []
    for n in range(1, N + 1):
        cols += [np.cos(n * theta), np.sin(n * theta)]
    return np.stack(cols, axis=1)


def orthonormal_basis(W):
    Q, _ = np.linalg.qr(W)  # W: [D,r]
    return Q


def project_out_subspace(H, Q):
    return H - H @ Q @ Q.T


def random_subspace(D, r, seed=0):
    rng = np.random.default_rng(seed)
    Q, _ = np.linalg.qr(rng.standard_normal((D, r)))
    return Q


def analyze_sum(H, raws, alpha=1.0):
    xs, ys = parse_xy(raws)
    sums = (xs + ys).astype(np.float32)
    sc = StandardScaler().fit(H)
    Hs = sc.transform(H)
    ridge = Ridge(alpha=alpha).fit(Hs, sums)
    pred = ridge.predict(Hs)
    r2 = r2_score(sums, pred)
    base_mse = mean_squared_error(sums, pred)
    # 1-D ablation
    w = ridge.coef_.astype(np.float32)
    w = w / (np.linalg.norm(w) + 1e-8)
    Hp = Hs - (Hs @ w[:, None]) * w[None, :]
    head_p = Ridge(alpha=alpha).fit(Hp, sums)
    mse_p = mean_squared_error(sums, head_p.predict(Hp))
    # random 1D
    ur = random_subspace(H.shape[1], 1)[:, 0]
    Hr = Hs - (Hs @ ur[:, None]) * ur[None, :]
    head_r = Ridge(alpha=alpha).fit(Hr, sums)
    mse_r = mean_squared_error(sums, head_r.predict(Hr))
    return dict(
        r2=r2,
        base_mse=base_mse,
        delta_probe=mse_p - base_mse,
        delta_rand=mse_r - base_mse,
    )


def analyze_modk(H, y_vec, Nmax=12, alpha=1.0, seed=0):
    k = int(y_vec.max()) + 1
    s = y_vec.astype(np.int64)
    theta = 2 * np.pi * (s % k) / k
    sc = StandardScaler().fit(H)
    Hs = sc.transform(H)
    # Fourier probe R² + causal excess ΔMSE
    res = {"k": k, "fourier": [], "causal": []}
    for N in range(1, Nmax + 1):
        Y = fourier_targets(theta, N)  # [B,2N]
        reg = Ridge(alpha=alpha).fit(Hs, Y)
        Yhat = reg.predict(Hs)
        r2 = r2_score(Y, Yhat, multioutput="uniform_average")
        # subspace & causal
        W = reg.coef_.T  # [D,2N]
        Q = orthonormal_basis(W)
        r = Q.shape[1]
        base = Ridge(alpha=alpha).fit(Hs, s.astype(np.float32))
        base_mse = mean_squared_error(s, base.predict(Hs))
        Hp = project_out_subspace(Hs, Q)
        head_p = Ridge(alpha=alpha).fit(Hp, s.astype(np.float32))
        mse_p = mean_squared_error(s, head_p.predict(Hp))
        Qr = random_subspace(H.shape[1], r, seed=seed + N)
        Hr = project_out_subspace(Hs, Qr)
        head_r = Ridge(alpha=alpha).fit(Hr, s.astype(np.float32))
        mse_r = mean_squared_error(s, head_r.predict(Hr))
        res["fourier"].append({"N": N, "dim": 2 * N, "r2": float(r2)})
        res["causal"].append(
            {
                "N": N,
                "r": int(r),
                "delta_probe": float(mse_p - base_mse),
                "delta_rand": float(mse_r - base_mse),
                "excess_delta": float((mse_p - mse_r)),
            }
        )
    # Nonlinear probes (quick): MLP reg R² and MLP cls accuracy
    # small helper to avoid a full trainer: use sklearn-ish MLP via torch (1 pass)

    Htr, Hte = Hs[: int(0.8 * Hs.shape[0])], Hs[int(0.8 * Hs.shape[0]) :]
    ytr, yte = s[: Htr.shape[0]], s[Htr.shape[0] :]
    mreg, _, yhat_te = train_mlp_reg(
        Htr,
        ytr.astype(np.float32),
        Hte,
        yte.astype(np.float32),
        hidden=256,
        depth=2,
        lr=5e-4,
        wd=1e-4,
        epochs=400,
        patience=50,
        batch_size=128,
        seed=seed,
    )
    r2_mlp = r2_score(yte.astype(np.float32), yhat_te)
    mcls, _, yte_pred = train_mlp_cls(
        Htr,
        ytr,
        Hte,
        yte,
        n_classes=k,
        hidden=256,
        depth=2,
        lr=5e-4,
        wd=1e-4,
        epochs=400,
        patience=50,
        batch_size=128,
        seed=seed,
    )
    acc = accuracy_score(yte, yte_pred)
    res["mlp"] = {"reg_r2": float(r2_mlp), "cls_acc": float(acc)}
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="REG-MODK", choices=["REG-SUM", "REG-MODK"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--models_root", default="analysis/models_scale")
    ap.add_argument("--analysis_dir", default="analysis/analysis_scale")
    ap.add_argument("--Nmax", type=int, default=12)
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.analysis_dir, exist_ok=True)
    tsv = os.path.join(args.data_dir, f"{args.task}_val.tsv")
    Xtok, y_vec, raws, lengths = load_tsv(tsv)

    # discover runs
    runs = sorted(
        [d for d in os.listdir(args.models_root) if d.startswith(args.task + "_")]
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    summaries = []
    for run in runs:
        ckpt = os.path.join(args.models_root, run, f"{args.task}_best.pt")
        meta_path = os.path.join(args.models_root, run, "meta.json")
        if not os.path.exists(ckpt):
            continue
        meta = json.load(open(meta_path)) if os.path.exists(meta_path) else {}
        m = TinyTransformer(
            vocab_size=len(VOCAB),
            d_model=meta.get("d_model", 128),
            n_layer=meta.get("depth", 2),
            n_head=meta.get("n_head", 4),
            d_ff=meta.get("d_ff", 512),
        ).to(device)
        m.load_state_dict(torch.load(ckpt, map_location=device))
        m.eval()
        H = pooled_hidden(m, Xtok, lengths, device)
        pcount = count_params(m)

        out = {"run": run, "params": int(pcount), **meta}
        if args.task == "REG-SUM":
            S = analyze_sum(H, raws, alpha=args.alpha)
            out["sum"] = S
        else:
            M = analyze_modk(H, y_vec, Nmax=args.Nmax, alpha=args.alpha, seed=args.seed)
            out["modk"] = M

        summaries.append(out)
        json.dump(
            out, open(os.path.join(args.analysis_dir, f"{run}.json"), "w"), indent=2
        )
        print("Analyzed:", run)

    # index
    idx_path = os.path.join(args.analysis_dir, f"{args.task}_scale_index.json")
    json.dump(summaries, open(idx_path, "w"), indent=2)
    print("Saved index:", idx_path)


if __name__ == "__main__":
    main()
