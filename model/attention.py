import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Each token gets turned into 3 vectors: Q, K, V.
# So if token A’s query matches token B’s key strongly, then token A will use more of token B’s value.
# Each matrix Q, K, V has shape (batch_size, seq_len, d_k).
# Meaning: batch_size is sequences in the batch; each sequence has seq_len tokens; each token is represented by a d_k vector

class ScaledDotProductAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask=None):
        """
        Q, K, V shape: (batch_size, seq_len, d_k)
        mask shape: (batch_size, seq_len, seq_len)
        """

        d_k = Q.size(-1) # taking the last dim

        # Step 1: Compute attention scores
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        
        # K.transpose(-2,-1) becomes shape (batch_size, d_k, seq_len)
        # resultant matrix: for each of the 5 tokens, compare it against all 5 tokens, producing a score for every pair. So scores[i, j] is the raw attention score between token i and token j.
        # Dot product because it measures similarity. If a query vector and a key vector point in similar directions, their dot product is large.
        # divide by sqrt(d_k) to stabilise the scores

        # side note: matmul works like: last two dimensions → matrix multiplication. Earlier dimensions → batch dimensions (which are carried along)
        # mental shortcut: (A, B, C) @ (A, C, D) → (A, B, D)

        # Step 2: Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # replace with -inf because after softmax exp(-inf) = 0

        # Step 3: Softmax
        attention_weights = F.softmax(scores, dim=-1)
        
        # dim=-1 because it corresponds to all tokens each token can attend to. We want to normalise for each token.

        # Step 4: Weighted sum
        output = torch.matmul(attention_weights, V)
        
        # attention_weights is how much to care about each token, while V is the actual information each token provides
        # output is a weighted combination of the value vectors from all tokens.
        # attention_weights: (batch_size, seq_len, seq_len)
        # V:                 (batch_size, seq_len, d_k)
        # output:            (batch_size, seq_len, d_k)

        return output, attention_weights
    
