import os, re, json, argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

from src.model import VOCAB, encode, pad_batch, PAD_ID
from .attn_instrument import TinyTransformerWithAttn


def plot_layer_head_heatmap(per_head, key, title, out_path):
    """
    per_head: list over layers -> list over heads -> dict of stats
    key: 'copy_score' | 'balance_score' | 'entropy'
    """

    n_layers = len(per_head)
    n_heads = len(per_head[0]) if n_layers > 0 else 0
    M = np.zeros((n_layers, n_heads), dtype=np.float32)
    for li in range(n_layers):
        for hj, d in enumerate(per_head[li]):
            M[li, hj] = d[key]

    plt.figure(figsize=(1.2 * n_heads, 1 * n_layers + 1.2))
    im = plt.imshow(M, aspect="auto", interpolation="nearest")

    # Add colorbar
    plt.colorbar(im, label=key)

    # Set tick labels to integer indices
    plt.xticks(range(n_heads), range(n_heads))
    plt.yticks(range(n_layers), range(n_layers))

    plt.xlabel("Head")
    plt.ylabel("Layer")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()


def layer_position_sweep(layers_attn, raws, lengths, task):
    """
    layers_attn: list of tensors [B,H,T,T] per layer
    Returns dict of category-> matrix [layers x max_T] with average mass.
    """

    n_layers = len(layers_attn)
    B, H, T, _ = layers_attn[0].shape
    max_T = T
    cats = ["x", "y", "plus", "mod", "others"]
    out = {c: np.zeros((n_layers, max_T), dtype=np.float64) for c in cats}
    counts = np.zeros((n_layers, max_T), dtype=np.int64)

    for li, A_layer in enumerate(layers_attn):
        A = A_layer.cpu()  # [B,H,T,T]
        for b in range(B):
            L = int(lengths[b])
            spans = parse_spans(raws[b], task)
            idx_x, idx_y = spans["x"], spans["y"]
            idx_plus, idx_mod, idx_others = spans["plus"], spans["mod"], spans["others"]

            # average over heads for this example
            # A[b]: [H,T,T]
            Av = A[b].mean(dim=0)  # [T,T]
            for q in range(L):  # query position

                def mass(idxs):
                    if not idxs:
                        return 0.0
                    idxs = [i for i in idxs if i < L]
                    if not idxs:
                        return 0.0
                    return float(Av[q, idxs].sum().item())

                out["x"][li, q] += mass(idx_x)
                out["y"][li, q] += mass(idx_y)
                out["plus"][li, q] += mass(idx_plus)
                out["mod"][li, q] += mass(idx_mod)
                out["others"][li, q] += mass(idx_others)
                counts[li, q] += 1

    # normalize by counts
    for c in cats:
        # avoid divide-by-zero
        nz = counts.copy()
        nz[nz == 0] = 1
        out[c] = out[c] / nz
    return out  # dict of [layers x T]


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


def parse_spans(raw, task):
    # returns dict of token index lists: x_digits, plus, y_digits, mod, others
    # Tokenization is char-level on VOCAB.
    s = raw
    # find positions of each printed char in the encoded string
    # We must re-encode to align indices:
    ids = [VOCAB.index(ch) for ch in s]
    # Now compute character positions directly
    # Identify spans by regex
    m = re.match(r"^(\d+)\s\+\s(\d+)(?:\s\(mod\s(\d+)\))?", s)
    assert m, f"bad: {s}"
    x_str, y_str, k_str = m.group(1), m.group(2), m.group(3)
    # build index map from characters
    x_digits_idx = list(range(0, len(x_str)))
    plus_idx = [len(x_str) + 1]  # " " after x, then '+'
    # Actually string is: "<x> + <y>" or "<x> + <y> (mod k)"
    # Precise indices by walking s:
    x_digits, y_digits, plus, mod_idxs = [], [], [], []
    i = 0
    # collect x digits
    while i < len(s) and s[i].isdigit():
        x_digits.append(i)
        i += 1
    # skip space
    if i < len(s) and s[i] == " ":
        i += 1
    # plus sign
    if i < len(s) and s[i] == "+":
        plus = [i]
        i += 1
    # skip space
    if i < len(s) and s[i] == " ":
        i += 1
    # y digits
    j = i
    while j < len(s) and s[j].isdigit():
        y_digits.append(j)
        j += 1
    i = j
    # optional: space then " (mod k)"
    mod_idxs = []
    if task == "REG-MODK":
        # expect " (mod k)"
        if i < len(s) and s[i] == " ":
            i += 1
        if i < len(s) and s[i] == "(":
            while i < len(s):
                mod_idxs.append(i)
                i += 1
                if s[i - 1] == ")":
                    break

    all_idx = set(range(len(s)))
    others = sorted(all_idx - set(x_digits) - set(y_digits) - set(plus) - set(mod_idxs))
    return {
        "x": x_digits,
        "plus": plus,
        "y": y_digits,
        "mod": mod_idxs,
        "others": others,
        "len": len(s),
    }


def aggregate_last_token_attention(A, length, spans):
    """
    A: (H, T, T) attention for one example at a single layer (per-head).
    We read rows = query positions, cols = key positions (PyTorch MHA convention).
    We take the last valid token index as the query.
    Return per-head category masses dict.
    """
    H, T, _ = A.shape
    last = int(length - 1)
    out = []
    idx_x = spans["x"]
    idx_y = spans["y"]
    idx_plus = spans["plus"]
    idx_mod = spans["mod"]
    idx_others = spans["others"]
    for h in range(H):
        v = A[h, last]  # [T] attention from last token to all keys

        def mass(idxs):
            if len(idxs) == 0:
                return 0.0
            idxs = [i for i in idxs if i < T]  # guard
            return float(v[idxs].sum().item())

        out.append(
            {
                "x": mass(idx_x),
                "y": mass(idx_y),
                "plus": mass(idx_plus),
                "mod": mass(idx_mod),
                "others": mass(idx_others),
            }
        )
    return out  # list len H


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="REG-SUM", choices=["REG-SUM", "REG-MODK"])
    ap.add_argument("--data_dir", default="data")
    ap.add_argument("--checkpoint_dir", default="checkpoints")
    ap.add_argument("--out_dir", default="analysis/analysis_heads")
    ap.add_argument("--sample_idx", type=int, nargs="*", default=[0, 1, 2])
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tsv = os.path.join(args.data_dir, f"{args.task}_val.tsv")
    Xtok, yv, raws, lengths = load_tsv(tsv)

    model = TinyTransformerWithAttn(vocab_size=len(VOCAB)).to(device)
    ckpt = os.path.join(args.checkpoint_dir, f"{args.task}_best.pt")
    model.load_state_dict(torch.load(ckpt, map_location=device))
    model.eval()

    # Run a forward pass to populate attention
    with torch.no_grad():
        _ = model(
            Xtok.to(device),
            lengths=torch.tensor(lengths, device=device),
            return_h=False,
        )

    # Grab attention weights for all layers (list of (B,H,T,T))
    layers_attn = model.get_last_layer_attn()  # list length = n_layer

    # 1) Save heatmaps for a few samples at the last layer
    last_layer_attn = layers_attn[-1].cpu()  # [B,H,T,T]
    B, H, T, _ = last_layer_attn.shape
    for b in args.sample_idx:
        if b >= B:
            continue
        A = last_layer_attn[b]  # [H,T,T]
        s = raws[b]
        L = int(lengths[b])
        plt.figure(figsize=(3.2 * H, 2.8))
        for h in range(H):
            plt.subplot(1, H, h + 1)
            plt.imshow(A[h, :L, :L], aspect="auto", interpolation="nearest")
            plt.title(f"Head {h}")
            plt.xlabel("Key pos")
            plt.ylabel("Query pos")
        plt.suptitle(f"{args.task} example {b}: {s}")
        plt.tight_layout()
        plt.savefig(
            os.path.join(args.out_dir, f"{args.task}_ex{b}_lastlayer_attn.png"), dpi=300
        )
        plt.close()

    # 2) Aggregate per-head category masses at the last layer (averaged over many samples)
    per_head = []  # [layer][head] dict with averages
    all_stats = []

    for li, A_layer in enumerate(layers_attn):
        A = A_layer.cpu()  # [B,H,T,T]
        B_, H_, T_, _ = A.shape
        sums = np.zeros((H_, 5), dtype=np.float64)  # x,y,plus,mod,others
        cnt = 0
        for b in range(B_):
            spans = parse_spans(raws[b], args.task)
            stats_b = aggregate_last_token_attention(
                A[b], lengths[b], spans
            )  # list H dicts
            for h, d in enumerate(stats_b):
                sums[h, 0] += d["x"]
                sums[h, 1] += d["y"]
                sums[h, 2] += d["plus"]
                sums[h, 3] += d["mod"]
                sums[h, 4] += d["others"]
            cnt += 1
        avg = sums / max(1, cnt)
        head_stats = []
        for h in range(H_):
            x, y, pl, md, ot = avg[h]
            copy = x + y
            balance = abs(x - y)
            total = x + y + pl + md + ot + 1e-9
            ent = float(
                -(x / total) * np.log((x / total) + 1e-12)
                - (y / total) * np.log((y / total) + 1e-12)
                - (pl / total) * np.log((pl / total) + 1e-12)
                - (md / total) * np.log((md / total) + 1e-12)
                - (ot / total) * np.log((ot / total) + 1e-12)
            )
            head_stats.append(
                {
                    "layer": li,
                    "head": h,
                    "mass_x": float(x),
                    "mass_y": float(y),
                    "mass_plus": float(pl),
                    "mass_mod": float(md),
                    "mass_others": float(ot),
                    "copy_score": float(copy),
                    "balance_score": float(balance),
                    "entropy": ent,
                }
            )
        per_head.append(head_stats)
        all_stats.extend(head_stats)

    # Save JSON
    out_json = os.path.join(args.out_dir, f"{args.task}_head_stats.json")
    json.dump({"per_head": per_head}, open(out_json, "w"), indent=2)
    print("Saved head stats to", out_json)

    # 3) Simple bar plots: for last layer, per head category masses
    last = per_head[-1]
    labels = [f"h{d['head']}" for d in last]
    xmass = [d["mass_x"] for d in last]
    ymass = [d["mass_y"] for d in last]
    pmass = [d["mass_plus"] for d in last]
    mmass = [d["mass_mod"] for d in last]
    omass = [d["mass_others"] for d in last]

    idx = np.arange(len(last))
    width = 0.18
    plt.figure(figsize=(6, 4))
    plt.bar(idx - 2 * width, xmass, width, label="x")
    plt.bar(idx - 1 * width, ymass, width, label="y")
    plt.bar(idx + 0 * width, pmass, width, label="+")
    if args.task == "REG-MODK":
        plt.bar(idx + 1 * width, mmass, width, label="mod")
        plt.bar(idx + 2 * width, omass, width, label="others")
    else:
        plt.bar(idx + 1 * width, omass, width, label="others")
    plt.xticks(idx, labels)
    plt.ylabel("Attention mass from last token")
    plt.title(f"{args.task}: last-layer per-head category mass")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(args.out_dir, f"{args.task}_lastlayer_head_bars.png"), dpi=300
    )
    print("Saved plots to", args.out_dir)

    # 4) Layer × Position heatmaps: average attention mass to each category, per layer
    # Layer × Head heatmaps
    plot_layer_head_heatmap(
        per_head,
        "copy_score",
        f"{args.task}: copy_score per (layer, head)",
        os.path.join(args.out_dir, f"{args.task}_heatmap_copy_score.png"),
    )
    plot_layer_head_heatmap(
        per_head,
        "balance_score",
        f"{args.task}: balance_score per (layer, head)",
        os.path.join(args.out_dir, f"{args.task}_heatmap_balance_score.png"),
    )
    plot_layer_head_heatmap(
        per_head,
        "entropy",
        f"{args.task}: attention entropy per (layer, head)",
        os.path.join(args.out_dir, f"{args.task}_heatmap_entropy.png"),
    )
    print("Saved layer×head heatmaps to", args.out_dir)

    # 5) Layer × position sweep (averaged over heads & examples)
    sweep = layer_position_sweep(layers_attn, raws, lengths, args.task)
    for cat, M in sweep.items():
        n_layers, T = M.shape

        plt.figure(figsize=(6, 1.2 * n_layers + 1.2))
        im = plt.imshow(M, aspect="auto", interpolation="nearest")

        # Add colorbar
        plt.colorbar(im, label=f"avg mass to {cat}")

        # Set integer ticks
        plt.xticks(range(T), range(T))
        plt.yticks(range(n_layers), range(n_layers))

        plt.xlabel("Query position")
        plt.ylabel("Layer")
        plt.title(f"{args.task}: avg attention mass to '{cat}' by (layer, query pos)")
        plt.tight_layout()
        plt.savefig(os.path.join(args.out_dir, f"{args.task}_sweep_{cat}.png"), dpi=300)
        plt.close()
    print("Saved layer×position sweep heatmaps to", args.out_dir)


if __name__ == "__main__":
    main()
