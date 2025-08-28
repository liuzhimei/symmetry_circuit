import torch
import torch.nn as nn
from src.model import TinyTransformer


class InstrumentedEncoderLayer(nn.Module):
    def __init__(self, base_layer):
        super().__init__()
        # clone submodules
        self.self_attn = base_layer.self_attn
        self.linear1 = base_layer.linear1
        self.dropout = base_layer.dropout
        self.linear2 = base_layer.linear2
        self.norm1 = base_layer.norm1
        self.norm2 = base_layer.norm2
        self.dropout1 = base_layer.dropout1
        self.dropout2 = base_layer.dropout2
        self.activation = base_layer.activation
        self.batch_first = True  # our TinyTransformer uses batch_first=True

        # storage
        self.last_attn_weights = None  # [B, num_heads, T, T]

    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # ask for per-head weights
        attn_output, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,
        )
        self.last_attn_weights = attn_weights  # [B, num_heads, T, T]
        src2 = attn_output
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


class TinyTransformerWithAttn(TinyTransformer):
    """
    Uses the *same* trained layers in self.enc.layers (no renaming),
    but overrides forward to (a) request per-head attn weights, and
    (b) store them in self.attn_cache as a list of [B, H, T, T] tensors.
    """

    def __init__(
        self, vocab_size, d_model=128, n_layer=2, n_head=4, d_ff=256, pdrop=0.1
    ):
        super().__init__(vocab_size, d_model, n_layer, n_head, d_ff, pdrop)
        self.attn_cache = []  # list of per-layer attention weights

    def forward(self, x, lengths=None, return_h=False):
        device = x.device
        B, T = x.shape
        pos = torch.arange(T, device=device).unsqueeze(0)
        h = self.tok(x) + self.pos(pos)

        # pad id == 0 in this toy setup; True means "mask this position"
        kpm = ~(x != 0)

        self.attn_cache = []
        # Manually run through each TransformerEncoderLayer in self.enc.layers
        for layer in self.enc.layers:
            # 1) Multi-head self-attention WITH per-head weights
            attn_output, attn_weights = layer.self_attn(
                h,
                h,
                h,
                attn_mask=None,
                key_padding_mask=kpm,
                need_weights=True,
                average_attn_weights=False,  # <-- crucial: [B, H, T, T] not averaged
            )
            self.attn_cache.append(attn_weights)  # [B, H, T, T]

            # 2) Residual + norm + FFN (replicates TransformerEncoderLayer.forward)
            src2 = attn_output
            h = h + layer.dropout1(src2)
            h = layer.norm1(h)
            src2 = layer.linear2(layer.dropout(layer.activation(layer.linear1(h))))
            h = h + layer.dropout2(src2)
            h = layer.norm2(h)

        h = self.ln(h)

        # Pool last valid token
        if lengths is not None:
            idx = (lengths - 1).clamp(min=0)
            pooled = h[torch.arange(B, device=device), idx, :]
        else:
            pooled = h[:, -1, :]

        y = self.readout(pooled).squeeze(-1)
        if return_h:
            return y, h
        return y

    def get_last_layer_attn(self):
        """Returns list over layers of attention weights [B, H, T, T]."""
        return self.attn_cache
