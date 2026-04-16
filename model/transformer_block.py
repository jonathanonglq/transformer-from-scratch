import torch
import torch.nn as nn

from model.multi_head_attention import MultiHeadAttention
from model.feedforward import FeedForward

# each input token starts as a vector. The block lets each token look at other tokens and update itself based on what it learned.
# so after the block, "sat" might now carry more info about other words e.g. "cat", "mat" than before.
# transformer block preserves the overall representation shape at every stage

class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn = FeedForward(d_model, d_ff)

        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, return_attentions=False):
        """
        x shape: (B, S, d_model)
        """

        # ---- Multi-head attention ----
        # lets tokens exchange information via mha

        attn_output, attn_weights = self.mha(x, mask=mask)

        # Residual + norm
        # keep what I had, and add some useful new info

        x = x + attn_output
        x = self.norm1(x)

        # ---- Feedforward ----
        # processes each token's representation further via ffn

        ffn_output = self.ffn(x)

        # Residual + norm
        x = x + ffn_output
        x = self.norm2(x)

        if return_attentions:
            return x, attn_weights

        return x