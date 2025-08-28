import os
import torch
import torch.nn as nn
import json
from torch.utils.data import Dataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- vocab with dedicated PAD ---
VOCAB = ["<PAD>"] + list("0123456789 +-=?()mod")
PAD_ID = 0
ITO = {ch: i for i, ch in enumerate(VOCAB)}
OTI = {i: ch for ch, i in ITO.items()}

S = 2000.0  # scale targets to make MSE loss small and stable


def encode(s):
    """Converts a string into a tensor of character IDs using ITO."""
    return torch.tensor([ITO[ch] for ch in s], dtype=torch.long)


def pad_batch(seqs, pad_id=PAD_ID):
    """Pads a batch of variable-length sequences with pad_id to make them the same length.
    Returns a tensor of shape (B, maxlen) and a mask of shape (B, maxlen) indicating valid positions.
    """
    maxlen = max(len(s) for s in seqs)
    out = torch.full((len(seqs), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), maxlen), dtype=torch.bool)
    lens = []
    for i, s in enumerate(seqs):
        out[i, : len(s)] = s
        mask[i, : len(s)] = True
        lens.append(len(s))
    return out, mask, torch.tensor(lens, dtype=torch.long)


class TextDataset(Dataset):
    """
    Dataset for regression tasks (REG-SUM, REG-MODK) from TSV files.

    Each line in the TSV file is of the form:
    input_string \t target_value
    where input_string is something like "103 + 40 = ?" and target_value is a float.

    Return:
    - self.inputs: list of encoded input tensors
    - self.targets: list of float targets
    """

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
                    x, y = line.split("\t")  # x is input string, y is float target
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

    def forward(self, x, lengths=None, return_h=False):
        B, T = x.shape  # batch size, sequence length
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        kpm = x == PAD_ID  # mask PAD
        h = self.enc(h, src_key_padding_mask=kpm)
        h = self.ln(h)  # (B, T, d_model)
        if lengths is None:
            # fallback: last column (not recommended)
            pooled = h[:, -1, :]
        else:
            idx = (lengths - 1).clamp(min=0)  # [B]
            pooled = h[torch.arange(B, device=x.device), idx, :]  # [B, d]
        y = self.readout(pooled).squeeze(-1)  # (bs,) regression output
        return (
            (y, h) if return_h else y
        )  # for probing; so that we can do interpretability on hidden states


def collate(batch):
    """Collate function to be used with DataLoader for regression tasks."""
    xs, ys = zip(*batch)
    X, mask, lengths = pad_batch(xs, pad_id=PAD_ID)  # padded input tensor and mask
    y = torch.tensor(ys, dtype=torch.float32) / S  # regression targets
    return X, mask, lengths, y


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
    history = []
    best_epoch = -1
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for X, _, lengths, y in train_loader:
            X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            pred = model(X, lengths=lengths)
            loss = loss_fn(pred, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            train_loss += loss.item() * X.size(0)
        train_loss /= len(train_dataset)  # average training loss

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, _, lengths, y in val_loader:
                X, lengths, y = X.to(device), lengths.to(device), y.to(device)
                pred = model(X, lengths=lengths)
                loss = loss_fn(pred, y)
                val_loss += loss.item() * X.size(0)
        val_loss /= len(val_dataset)  # average validation loss

        print(f"Epoch {epoch+1}| train loss: {train_loss:.4f} val loss: {val_loss:.4f}")
        history.append(
            {"epoch": epoch + 1, "train_mse": train_loss, "val_mse": val_loss}
        )

        # Save best model
        if val_loss < best:
            best = val_loss
            best_epoch = epoch + 1
            save_path = os.path.join(args.out_dir, f"{args.task}_best.pt")
            torch.save(model.state_dict(), save_path)
    print(f"Saved best model checkpoint in {save_path}")
    print(f"Best val MSE: {best} at epoch {best_epoch}")
    with open(os.path.join(args.out_dir, f"{args.task}_train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Saved training history.")


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

    def forward(self, x, lengths=None, return_h=False):
        # Run the encoder from parent to get hidden states
        B, T = x.shape
        pos = torch.arange(T, device=x.device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)
        kpm = x == PAD_ID
        h = self.enc(h, src_key_padding_mask=kpm)
        h = self.ln(h)

        if lengths is None:
            pooled = h[:, -1, :]
        else:
            idx = (lengths - 1).clamp(min=0)
            pooled = h[torch.arange(B, device=x.device), idx, :]

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

    def collate_cls(batch):
        seqs, labels = zip(*batch)
        X, mask, lengths = pad_batch(seqs, pad_id=PAD_ID)
        y = torch.tensor(labels, dtype=torch.long)
        return X, mask, lengths, y

    train_loader = DataLoader(
        train_dataset, batch_size=args.bs, shuffle=True, collate_fn=collate_cls
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.bs, shuffle=False, collate_fn=collate_cls
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
    history = []
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        for X, _, lengths, y in train_loader:
            X, lengths, y = X.to(device), lengths.to(device), y.to(device)
            logits = model(X, lengths=lengths)
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
            for X, _, lengths, y in val_loader:
                X, lengths, y = X.to(device), lengths.to(device), y.to(device)
                logits = model(X, lengths=lengths)
                loss = loss_fn(logits, y)
                val_loss += loss.item() * X.size(0)
                pred = logits.argmax(dim=-1)
                correct += (pred == y).sum().item()
                total += y.numel()
        val_loss /= len(val_dataset)
        acc = correct / max(1, total)
        print(
            f"Epoch {epoch+1} | train_loss: {train_loss:.4f}, val_loss: {val_loss:.4f}, val_acc: {acc:.4f}"
        )
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_acc": acc,
            }
        )

        if acc > best:
            best = acc
            save_path = os.path.join(args.out_dir, f"{args.task}_best.pt")
            torch.save(model.state_dict(), save_path)
    print(f"Saved best model checkpoint in {save_path}!")
    print("Best val acc:", best)
    with open(os.path.join(args.out_dir, f"{args.task}_train_history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print("Saved training history.")
