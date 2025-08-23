import math, argparse, os
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vocabulary and encoding/decoding
VOCAB = list("0123456789 +-=?()mod")
ITO = {ch: i for i, ch in enumerate(VOCAB)}  # char to index
OTI = {i: ch for ch, i in ITO.items()}  # index to char


def encode(s):
    """Converts a string into a tensor of character IDs using ITO."""
    return torch.tensor([ITO[ch] for ch in s], dtype=torch.long)


def pad_batch(seqs, pad_id=0):
    """Pads a batch of variable-length sequences with pad_id to make them the same length.
    Returns a tensor of shape (B, maxlen) and a mask of shape (B, maxlen) indicating valid positions.
    """
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        mask[i, : len(s)] = True
    return out, mask


class TextDataset(Dataset):
    def __init__(self, tsv_path, task):
        self.task = task
        self.inputs = []
        self.targets = []
        with open(tsv_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                if task in ("REG-SUM", "REG-MODK"):
                    x, y = line.split("\t")
                    self.inputs.append(encode(x))
                    self.targets.append(float(y))
                else:
                    raise ValueError("Use jsonl for CLS-VALID")

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, i):
        return self.inputs[i], self.targets[i]


class TinyTransformer(nn.Module):
    def __init__(
        self, vocab_size, d_model=128, n_layer=2, n_head=4, d_ff=256, pdrop=0.1
    ):
        super().__init__()
        self.tok = nn.Embedding(vocab_size, d_model)  # token embedding
        self.pos = nn.Embedding(512, d_model)  # position embedding
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_head,
            dim_feedforward=d_ff,
            batch_first=True,
            dropout=pdrop,
            activation="gelu",
        )  # Transformer layer
        self.enc = nn.TransformerEncoder(layer, num_layers=n_layer)  # stack of layers
        self.ln = nn.LayerNorm(d_model)  # final layer norm
        self.readout = nn.Linear(d_model, 1)  # regression head

    def forward(self, x, return_h=False):
        bs, T = x.shape  # batch size, sequence length
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        # Use key padding mask so pad_id=0 is masked out (we assume '0' char also id 0; ok for toy)
        kpm = ~(x != 0)
        h = self.enc(h, src_key_padding_mask=kpm)
        h = self.ln(h)
        pooled = h[:, -1, :]  # take the output of the last token
        y = self.readout(pooled).squeeze(-1)
        if return_h:  # for probing; so that we can do interpretability on hidden states
            return y, h
        return y


def collate(batch):
    """Collate function to be used with DataLoader for regression tasks."""
    xs, ys = zip(*batch)
    X, mask = pad_batch(xs, pad_id=0)
    y = torch.tensor(ys, dtype=torch.float32)
    return X, mask, y


def train_regression(args):
    # Loads TSV datasets, creates loaders.
    train_path = os.path.join(args.data_dir, f"{args.task}_train.tsv")
    val_path = os.path.join(args.data_dir, f"{args.task}_val.tsv")
    train_dataset = TextDataset(train_path, args.task)
    val_dataset = TextDataset(val_path, args.task)
    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False, collate_fn=collate
    )

    model = TinyTransformer(
        vocab_size=len(VOCAB),
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.MSELoss()

    best = float("inf")
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for X, _, y in train_loader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_dataset)  # average training loss

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, _, y in val_loader:
                X, y = X.to(device), y.to(device)
                pred = model(X)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * X.size(0)
        val_loss /= len(val_dataset)  # average validation loss

        print(f"Epoch {epoch+1}: train {train_loss:.4f} val {val_loss:.4f}")

        # Save best model
        if val_loss < best:
            best = val_loss
            torch.save(
                model.state_dict(), os.path.join(args.out_dir, f"{args.task}_best.pt")
            )
    print("Best val MSE:", best)


"""Classification model and training"""


class JsonlClsDataset(Dataset):
    def __init__(self, jsonl_path):

        self.items = []
        with open(jsonl_path) as f:
            for line in f:
                r = json.loads(line)
                # Build the exact same input string the model expects
                s = f"{r['x']} + {r['y']} = {r['z']}"
                self.items.append((encode(s), int(r["label"])))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        return self.items[i]


class TinyTransformerCls(TinyTransformer):
    def __init__(
        self, vocab_size, d_model=128, n_layer=2, n_head=4, d_ff=256, pdrop=0.1
    ):
        super().__init__(vocab_size, d_model, n_layer, n_head, d_ff, pdrop)
        self.readout = nn.Linear(d_model, 2)  # two-class logits

    def forward(self, x, return_h=False):
        out = super().forward(x, return_h=True)
        y, h = out
        # y here is wrong type (scalar) from parent; we want pooled then 2-logits:
        # So override: recompute pooled on h
        pooled = h[:, -1, :]
        logits = self.readout(pooled)
        if return_h:
            return logits, h
        return logits


def train_classification(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_path = os.path.join(args.data_dir, f"{args.task}_train.jsonl")
    val_path = os.path.join(args.data_dir, f"{args.task}_val.jsonl")

    train_dataset = JsonlClsDataset(train_path)
    val_dataset = JsonlClsDataset(val_path)

    def collate(batch):
        seqs, labels = zip(*batch)
        X, mask = pad_batch(seqs, pad_id=0)
        y = torch.tensor(labels, dtype=torch.long)
        return X, mask, y

    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False, collate_fn=collate
    )

    model = TinyTransformerCls(
        vocab_size=len(VOCAB),
        d_model=args.d_model,
        n_layer=args.n_layer,
        n_head=args.n_head,
        d_ff=args.d_ff,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loss_fn = nn.CrossEntropyLoss()

    best = 0.0
    os.makedirs(args.out_dir, exist_ok=True)
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for X, _, y in train_loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_dataset)

        model.eval()
        correct = 0
        total = 0
        val_loss = 0.0
        with torch.no_grad():
            for X, _, y in val_loader:
                X, y = X.to(device), y.to(device)
                logits = model(X)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * X.size(0)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        val_loss /= len(val_dataset)
        acc = correct / max(1, total)
        print(
            f"Epoch {epoch+1}: train_loss {train_loss:.4f} val_loss {val_loss:.4f} val_acc {acc:.4f}"
        )
        if acc > best:
            best = acc
            torch.save(
                model.state_dict(), os.path.join(args.out_dir, f"{args.task}_best.pt")
            )
    print("Best val acc:", best)
