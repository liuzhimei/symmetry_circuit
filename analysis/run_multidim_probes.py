# src/run_multidim_probes.py
import os, re, json, argparse
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error

from src.model import TinyTransformer, VOCAB, encode, pad_batch, PAD_ID


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


def pooled_hidden(model, X, lengths, device):
    with torch.no_grad():
        y, h = model(X.to(device), lengths=lengths.to(device), return_h=True)
    B = h.shape[0]
    idx = (lengths - 1).clamp(min=0)
    pooled = h[torch.arange(B, device=h.device), idx, :].cpu().numpy()
    return pooled  # [B, D]


def parse_xy(raws):
    xs, ys = [], []
    for s in raws:
        m = re.match(r"(\d+)\s\+\s(\d+)", s)
        if not m:
            raise ValueError(f"bad line: {s}")
        xs.append(int(m.group(1)))
        ys.append(int(m.group(2)))
    return np.array(xs, dtype=np.int64), np.array(ys, dtype=np.int64)


def fourier_targets(theta, N):
    # theta: [B]
    cols = []
    for n in range(1, N + 1):
        cols.append(np.cos(n * theta))
        cols.append(np.sin(n * theta))
    return np.stack(cols, axis=1)  # [B, 2N]


def fit_multitarget_ridge(H, Y, alpha=1.0, standardize=True):
    if standardize:
        sc = StandardScaler().fit(H)
        Hs = sc.transform(H)
    else:
        sc = None
        Hs = H
    reg = Ridge(alpha=alpha, fit_intercept=True)
    reg.fit(Hs, Y)
    Yhat = reg.predict(Hs)
    r2 = r2_score(Y, Yhat, multioutput="uniform_average")
    return reg, sc, float(r2), Yhat


def orthonormal_basis(W):
    # W: [D, r] (columns span the subspace)
    # sklearn Ridge.coef_ is [n_targets, D], so we will transpose
    Q, _ = np.linalg.qr(W)  # economic
    return Q  # [D, r]


def project_out_subspace(H, Q):
    # H: [B, D], Q: [D, r] with orthonormal columns
    return H - H @ Q @ Q.T


def random_subspace(D, r, seed=0):
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((D, r))
    Q, _ = np.linalg.qr(A)
    return Q


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="REG-MODK", choices=["REG-MODK", "REG-SUM"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="analysis/output")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--alpha", type=float, default=1.0)
    ap.add_argument(
        "--Nmax", type=int, default=12, help="max Fourier harmonic (only for REG-MODK)"
    )
    ap.add_argument("--smallk", type=int, nargs="*", default=[2, 3, 4, 5, 7, 8])
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load data & model
    tsv = os.path.join(args.data_dir, f"{args.task}_val.tsv")
    X, y_true, raws, lengths = load_tsv(tsv)
    m = TinyTransformer(vocab_size=len(VOCAB)).to(device)
    ckpt = os.path.join(args.checkpoint_dir, f"{args.task}_best.pt")
    state = torch.load(ckpt, map_location=device)
    m.load_state_dict(state)
    m.eval()

    # Pooled hidden states H ∈ ℝ^{B×D}
    H = pooled_hidden(m, X, lengths, device)
    B, D = H.shape
    rng = np.random.default_rng(args.seed)

    if args.task == "REG-SUM":
        # === Scalar probe only (no Fourier) ===
        # Build numeric sums from raws (avoid scaled target complications)
        xs, ys_int = parse_xy(raws)
        sums = (xs + ys_int).astype(np.float32)

        # Scalar ridge probe & R^2
        head = Ridge(alpha=args.alpha).fit(H, sums)
        pred = head.predict(H)

        r2 = r2_score(sums, pred)
        base_mse = mean_squared_error(sums, pred)

        # 1-D causal test: remove the probe axis vs a random 1-D axis
        w = head.coef_.astype(np.float32)  # [D]
        w = w / (np.linalg.norm(w) + 1e-8)
        H_abl = H - (H @ w[:, None]) * w[None, :]
        head_abl = Ridge(alpha=args.alpha).fit(H_abl, sums)
        mse_after = mean_squared_error(sums, head_abl.predict(H_abl))
        delta_probe = float(mse_after - base_mse)

        # Random 1-D baseline
        urand = rng.standard_normal(D).astype(np.float32)
        urand /= np.linalg.norm(urand) + 1e-8
        H_r = H - (H @ urand[:, None]) * urand[None, :]
        head_r = Ridge(alpha=args.alpha).fit(H_r, sums)
        mse_r = mean_squared_error(sums, head_r.predict(H_r))
        delta_rand = float(mse_r - base_mse)

        results = {
            "task": args.task,
            "alpha": args.alpha,
            "scalar_probe_r2": float(r2),
            "base_mse": float(base_mse),
            "delta_mse_probe_1d": delta_probe,
            "delta_mse_random_1d": delta_rand,
            "excess_delta_mse": float(delta_probe - delta_rand),
        }

    else:
        # === REG-MODK: Fourier multi-output probes + causal subspace ablation ===
        # Infer k and form θ
        k = int(y_true.max()) + 1
        s = y_true.astype(np.int64)
        theta = 2 * np.pi * (s % k) / k

        results = {
            "task": args.task,
            "alpha": args.alpha,
            "Nmax": args.Nmax,
            "fourier_r2_curve": [],
            "smallk_r2": [],
            "causal": {
                "base_mse": None,
                "delta_mse_vs_dim": [],
                "delta_mse_random_vs_dim": [],
                "excess_delta_mse_vs_dim": [],
            },
        }

        # Base ridge head on numeric residue
        base_head = Ridge(alpha=args.alpha).fit(H, s.astype(np.float32))
        base_pred = base_head.predict(H)
        base_mse = mean_squared_error(s, base_pred)
        results["causal"]["base_mse"] = float(base_mse)

        # Keep subspace sizes modest to avoid random-dim damage dominating
        Ncap = min(args.Nmax, max(1, D // 8))  # e.g., <= 16 dims if D=128
        for N in range(1, Ncap + 1):
            Y = fourier_targets(theta, N)  # [B, 2N]
            reg, sc, r2, _ = fit_multitarget_ridge(H, Y, alpha=args.alpha)
            results["fourier_r2_curve"].append({"N": N, "dim": 2 * N, "r2": r2})

            # Probe subspace (columns of W)
            W = reg.coef_.T  # [D, 2N]
            Q = orthonormal_basis(W)  # [D, r]
            r = int(Q.shape[1])

            # Project-out probe subspace & refit head
            Hp = project_out_subspace(H, Q)
            head_p = Ridge(alpha=args.alpha).fit(Hp, s.astype(np.float32))
            mse_p = mean_squared_error(s, head_p.predict(Hp))
            delta_p = float(mse_p - base_mse)

            # Random subspace baseline (same r)
            Qr = random_subspace(D, r, seed=args.seed + N)
            Hr = project_out_subspace(H, Qr)
            head_r = Ridge(alpha=args.alpha).fit(Hr, s.astype(np.float32))
            mse_r = mean_squared_error(s, head_r.predict(Hr))
            delta_r = float(mse_r - base_mse)

            results["causal"]["delta_mse_vs_dim"].append(
                {"N": N, "r": r, "delta_mse": delta_p}
            )
            results["causal"]["delta_mse_random_vs_dim"].append(
                {"N": N, "r": r, "delta_mse": delta_r}
            )
            results["causal"]["excess_delta_mse_vs_dim"].append(
                {"N": N, "r": r, "excess_delta_mse": float(delta_p - delta_r)}
            )

        # Small-k′ probes (just R^2 diagnostic)
        for kprime in [2, 3, 4, 5, 7, 8]:
            thetap = 2 * np.pi * ((s % kprime) / kprime)
            for N in [1, 2, 3]:
                Yp = fourier_targets(thetap, N)
                regp, scp, r2p, _ = fit_multitarget_ridge(H, Yp, alpha=args.alpha)
                results["smallk_r2"].append(
                    {"kprime": int(kprime), "N": N, "dim": 2 * N, "r2": float(r2p)}
                )

    out_path = os.path.join(args.output_dir, f"{args.task}_multidim_results.json")
    json.dump(results, open(out_path, "w"), indent=2)
    print("Saved:", out_path)
    print("Results:", results)


if __name__ == "__main__":
    main()
