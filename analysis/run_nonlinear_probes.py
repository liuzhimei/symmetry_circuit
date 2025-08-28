# src/run_nonlinear_probes.py
import os, re, json, argparse
import numpy as np
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, accuracy_score
from sklearn.linear_model import Ridge
import torch.nn as nn
import torch.optim as optim

from src.model import TinyTransformer, VOCAB, encode, pad_batch, PAD_ID


# ---------- utilities ----------
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


# ---------- simple MLPs in PyTorch ----------
class MLPReg(nn.Module):
    def __init__(self, d_in, hidden=128, depth=1, pdrop=0.0):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(pdrop)]
            d = hidden
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):  # x: [B, D]
        return self.net(x).squeeze(-1)


class MLPCls(nn.Module):
    def __init__(self, d_in, n_classes, hidden=128, depth=1, pdrop=0.0):
        super().__init__()
        layers = []
        d = d_in
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(pdrop)]
            d = hidden
        layers += [nn.Linear(d, n_classes)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)  # logits [B, K]


# --- helpers ---
def _standardize_targets(y_train, y_val):
    mu = y_train.mean()
    sd = y_train.std() + 1e-8
    return (y_train - mu) / sd, (y_val - mu) / sd, mu, sd


def _iter_minibatches(X, y, batch_size=128, rng=None):
    n = X.shape[0]
    idx = np.arange(n)
    if rng is None:
        rng = np.random.default_rng(0)
    rng.shuffle(idx)
    for start in range(0, n, batch_size):
        sl = idx[start : start + batch_size]
        yield X[sl], y[sl]


# --- MLP REG with y-standardization + minibatches ---
def train_mlp_reg(
    Htr,
    ytr,
    Hte,
    yte,
    hidden=128,
    depth=1,
    lr=1e-3,
    wd=1e-4,
    epochs=400,
    patience=50,
    batch_size=128,
    seed=0,
    verbose=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = Htr.shape[1]
    model = MLPReg(d, hidden=hidden, depth=depth, pdrop=0.0)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.MSELoss()

    # standardize y
    ytr_s, yte_s, y_mu, y_sd = _standardize_targets(
        ytr.astype(np.float32), yte.astype(np.float32)
    )

    best = (1e9, None, 0)
    bad = 0
    rng = np.random.default_rng(seed)

    Xtr_t = torch.tensor(Htr, dtype=torch.float32)
    Xte_t = torch.tensor(Hte, dtype=torch.float32)
    yte_t = torch.tensor(yte_s, dtype=torch.float32)

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        nb = 0
        for xb, yb in _iter_minibatches(Htr, ytr_s, batch_size=batch_size, rng=rng):
            xb_t = torch.tensor(xb, dtype=torch.float32)
            yb_t = torch.tensor(yb, dtype=torch.float32)
            opt.zero_grad()
            pred = model(xb_t)
            loss = loss_fn(pred, yb_t)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            nb += 1

        model.eval()
        with torch.no_grad():
            val_pred = model(Xte_t)
            vloss = loss_fn(val_pred, yte_t).item()
        if verbose and ep % 10 == 0:
            print(f"[Reg] ep {ep} tr {tr_loss/max(1,nb):.4f} val {vloss:.4f}")

        if vloss < best[0]:
            best = (
                vloss,
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
                ep,
            )
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pred_tr_s = model(torch.tensor(Htr, dtype=torch.float32)).cpu().numpy()
        pred_te_s = model(torch.tensor(Hte, dtype=torch.float32)).cpu().numpy()
    # de-standardize
    pred_tr = pred_tr_s * y_sd + y_mu
    pred_te = pred_te_s * y_sd + y_mu
    return model, pred_tr, pred_te


# --- MLP CLS with minibatches ---
def train_mlp_cls(
    Htr,
    ytr,
    Hte,
    yte,
    n_classes,
    hidden=128,
    depth=1,
    lr=1e-3,
    wd=1e-4,
    epochs=400,
    patience=50,
    batch_size=128,
    seed=0,
    verbose=False,
):
    torch.manual_seed(seed)
    np.random.seed(seed)
    d = Htr.shape[1]
    model = MLPCls(d, n_classes, hidden=hidden, depth=depth, pdrop=0.0)
    opt = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
    loss_fn = nn.CrossEntropyLoss()

    best = (1e9, None, 0)
    bad = 0
    rng = np.random.default_rng(seed)

    Xte_t = torch.tensor(Hte, dtype=torch.float32)
    yte_t = torch.tensor(yte, dtype=torch.long)

    for ep in range(epochs):
        model.train()
        tr_loss = 0.0
        nb = 0
        for xb, yb in _iter_minibatches(Htr, ytr, batch_size=batch_size, rng=rng):
            xb_t = torch.tensor(xb, dtype=torch.float32)
            yb_t = torch.tensor(yb, dtype=torch.long)
            opt.zero_grad()
            logits = model(xb_t)
            loss = loss_fn(logits, yb_t)
            loss.backward()
            opt.step()
            tr_loss += loss.item()
            nb += 1

        model.eval()
        with torch.no_grad():
            logits_val = model(Xte_t)
            vloss = loss_fn(logits_val, yte_t).item()
        if verbose and ep % 10 == 0:
            print(f"[Cls] ep {ep} tr {tr_loss/max(1,nb):.4f} val {vloss:.4f}")

        if vloss < best[0]:
            best = (
                vloss,
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
                ep,
            )
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break

    model.load_state_dict(best[1])
    model.eval()
    with torch.no_grad():
        pred_tr = (
            model(torch.tensor(Htr, dtype=torch.float32)).argmax(dim=-1).cpu().numpy()
        )
        pred_te = (
            model(torch.tensor(Hte, dtype=torch.float32)).argmax(dim=-1).cpu().numpy()
        )
    return model, pred_tr, pred_te


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="REG-MODK", choices=["REG-MODK", "REG-SUM"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--output_dir", default="analysis/output")
    ap.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--alpha", type=float, default=1.0)  # ridge
    # MLP hparams (can sweep a couple of settings)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--depth", type=int, default=1)  # number of hidden layers
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=1e-4)
    ap.add_argument("--epochs", type=int, default=200)
    ap.add_argument("--patience", type=int, default=20)  # early stopping
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--shuffle_control", action="store_true", help="run label-shuffle control"
    )
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- load pooled hidden states ---
    tsv = os.path.join(args.data_dir, f"{args.task}_val.tsv")
    Xtok, y_val, raws, lengths = load_tsv(tsv)
    m = TinyTransformer(vocab_size=len(VOCAB)).to(device)
    ckpt = os.path.join(args.checkpoint_dir, f"{args.task}_best.pt")
    m.load_state_dict(torch.load(ckpt, map_location=device))
    m.eval()

    H = pooled_hidden(m, Xtok, lengths, device)  # [B, D]

    # --- targets ---
    if args.task == "REG-SUM":
        xs, ys = parse_xy(raws)
        target = (xs + ys).astype(np.float32)  # regression
        # split
        Htr, Hte, ytr, yte = train_test_split(
            H, target, test_size=args.test_size, random_state=args.seed
        )
        # standardize features
        sc = StandardScaler().fit(Htr)
        Htr_s = sc.transform(Htr)
        Hte_s = sc.transform(Hte)

        # baselines: ridge
        ridge = Ridge(alpha=args.alpha).fit(Htr_s, ytr)
        yhat_lin = ridge.predict(Hte_s)
        r2_lin = r2_score(yte, yhat_lin)

        # MLP regression
        mlp_reg, yhat_tr, yhat_te = train_mlp_reg(
            Htr_s,
            ytr,
            Hte_s,
            yte,
            hidden=args.hidden,
            depth=args.depth,
            lr=args.lr,
            wd=args.wd,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )
        r2_mlp = r2_score(yte, yhat_te)

        results = {
            "task": args.task,
            "ridge_r2": float(r2_lin),
            "mlp_reg_r2": float(r2_mlp),
        }

        if args.shuffle_control:
            ytr_shuf = ytr.copy()
            np.random.default_rng(args.seed).shuffle(ytr_shuf)
            mlp_reg_s, _, yhat_te_s = train_mlp_reg(
                Htr_s,
                ytr_shuf,
                Hte_s,
                yte,
                hidden=args.hidden,
                depth=args.depth,
                lr=args.lr,
                wd=args.wd,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed + 1,
            )
            r2_shuf = r2_score(yte, yhat_te_s)
            results["mlp_reg_r2_label_shuffle"] = float(r2_shuf)

    else:
        # REG-MODK
        # residue labels (classification) and numeric (regression metric)
        k = int(y_val.max()) + 1
        s = y_val.astype(np.int64)
        # split
        Htr, Hte, ytr, yte = train_test_split(
            H, s, test_size=args.test_size, random_state=args.seed, stratify=s
        )
        sc = StandardScaler().fit(Htr)
        Htr_s = sc.transform(Htr)
        Hte_s = sc.transform(Hte)

        # ridge regression on numeric residues (weak baseline)
        ridge = Ridge(alpha=args.alpha).fit(Htr_s, ytr.astype(np.float32))
        yhat_num = ridge.predict(Hte_s)
        r2_lin = r2_score(yte.astype(np.float32), yhat_num)

        # MLP regression (R^2)
        mlp_reg, _, yhat_num_mlp = train_mlp_reg(
            Htr_s,
            ytr.astype(np.float32),
            Hte_s,
            yte.astype(np.float32),
            hidden=args.hidden,
            depth=args.depth,
            lr=args.lr,
            wd=args.wd,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )
        r2_mlp = r2_score(yte.astype(np.float32), yhat_num_mlp)

        # MLP classification (accuracy on residues)
        mlp_cls, ytr_pred_cls, yte_pred_cls = train_mlp_cls(
            Htr_s,
            ytr,
            Hte_s,
            yte,
            n_classes=k,
            hidden=args.hidden,
            depth=args.depth,
            lr=args.lr,
            wd=args.wd,
            epochs=args.epochs,
            patience=args.patience,
            seed=args.seed,
        )
        acc_mlp = accuracy_score(yte, yte_pred_cls)

        results = {
            "task": args.task,
            "k": k,
            "ridge_r2_numeric_residue": float(r2_lin),
            "mlp_reg_r2_numeric_residue": float(r2_mlp),
            "mlp_cls_accuracy_residue": float(acc_mlp),
        }

        if args.shuffle_control:
            ytr_shuf = ytr.copy()
            np.random.default_rng(args.seed).shuffle(ytr_shuf)
            mlp_cls_s, _, yte_pred_cls_s = train_mlp_cls(
                Htr_s,
                ytr_shuf,
                Hte_s,
                yte,
                n_classes=k,
                hidden=args.hidden,
                depth=args.depth,
                lr=args.lr,
                wd=args.wd,
                epochs=args.epochs,
                patience=args.patience,
                seed=args.seed + 1,
            )
            acc_shuf = accuracy_score(yte, yte_pred_cls_s)
            results["mlp_cls_accuracy_label_shuffle"] = float(acc_shuf)

    # save
    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"{args.task}_nonlinear_probe_results.json")
    json.dump(results, open(out, "w"), indent=2)
    print("Saved:", out)
    print(results)


if __name__ == "__main__":
    main()
