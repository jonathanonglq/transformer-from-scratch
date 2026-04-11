import torch
import torch.nn as nn

from model.attention import ScaledDotProductAttention

# in single-head attention, every token only has one way of looking at a sequence
# in multi-head attention, different heads can learn different kinds of relationships

class MultiHeadAttention(nn.Module):

    # d_model is the full model dimension

    def __init__(self, d_model, num_heads):
        super().__init__()

        # It is not that attention theoretically require equal head sizes. It is more for clean parallel computation and simpler tensor reshaping.
        # If not, each head would require separate handling, batching would be less elegant, etc.

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # W_q learns what to ask, W_k leanrs what to match against, W_v learns what info to pass forward

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)

        self.attention = ScaledDotProductAttention()

        self.W_o = nn.Linear(d_model, d_model)

    def split_heads(self, x):
        """
        x shape: (B, S, d_model)
        return:  (B, num_heads, S, d_k)
        """
        B, S, _ = x.shape

        # view keeps underlying data buut reorganises into new dims
        x = x.view(B, S, self.num_heads, self.d_k)

        # transpose swaps dims 1 and 2
        x = x.transpose(1, 2)
        return x

    def combine_heads(self, x):
        """
        x shape: (B, num_heads, S, d_k)
        return:  (B, S, d_model)
        """
        # essentially a reverse of split_head()

        B, _, S, _ = x.shape

        # transpose because when we eventually flatten the last two dims, we want per token, and all head dimensions placed side by side
        # after transpose, tensor may no longer be stored contiguously in memory. .contiguous makes the memory layout safe for reshaping.
        x = x.transpose(1, 2).contiguous()

        x = x.view(B, S, self.d_model)
        return x

    def forward(self, x, mask=None):

        # x here refers to the input representation of tokens that enter the multi-head attention layer.
        # in practice, it may mean: in the first transformer layer, x may be the token embedding + positional info. In later layers, it may be output from previous layer
        # if your sentence has 5 tokens, and each token has 8 dims, then x.shape = (B, S, d_model). For a sentence "the cat sat on the mat", x[0, 2, :] would be the vector representation for "sat" in batch 0
        """
        x shape: (B, S, d_model)
        """

        Q = self.W_q(x)
        K = self.W_k(x)
        V = self.W_v(x)

        # after splitting, each tensor has shape (B, num_heads, S, d_k)
        
        Q = self.split_heads(Q)
        K = self.split_heads(K)
        V = self.split_heads(V)

        if mask is not None:
            # unsqueeze adds a dimension at index 1 i.e. (B, S, S) -> (B, 1, S, S)
            mask = mask.unsqueeze(1)

        attention_output, attention_weights = self.attention(Q, K, V, mask=mask)

        # combine_heads() reecombines the head dims structurally
        # W_o learns how to mix the recombined info

        output = self.combine_heads(attention_output)
        output = self.W_o(output)

        return output, attention_weights