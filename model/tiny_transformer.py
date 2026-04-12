import torch
import torch.nn as nn
import math
from model.transformer_block import TransformerBlock


# attention itself does not know order. PositionalEncoding tells the model where each token is in the sequence

class PositionalEncoding(nn.Module): #pytorch module that produces position-dependent vectors and adds them to token embeddings

    # Attention is permutation-invariant, so positional encoding injects order by adding a position-dependent signal to each token embedding.
    # Each token representation becomes embedding + positional encoding, combining what the token is with where it is.
    # Each position is encoded as a multi-frequency sinusoidal signal, where each pair of dimensions represents a sine and cosine at a specific frequency (i.e. multiple rotating vectors t different speeds)
    # Why sine and cosine? A sine–cosine pair forms a 2D rotation (unit circle), uniquely representing an angle and allowing the model to infer relative position via dot products.
    # Why multiple freqs? Different dimensions use different frequencies so the model captures both local (high frequency) and global (low frequency) positional relationships.
    # Why collisions are unlikely: Although individual frequencies repeat, the combination of many exponentially spaced frequencies makes positional encodings effectively unique within practical sequence lengths.
    # Positional encoding turns position into a structured, continuous signal, enabling attention to reason about distance and ordering, not just identity.

    def __init__(self, d_model, max_len=100):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        # arange(0, max_len) yields [0, 1, 2, .., max_len] of shape (max_len,). unsqueeze(1) makes it (max_len, 1). This will be the column vector of positions
        position = torch.arange(0, max_len).unsqueeze(1)

        # arange here uses even dimensions only because positional encoding uses sine for even indices, cosine for odd indices, so dimensions are paired (e.g. dim 0 with dim 1)
        # div term creates frequencies for different dimensions. Each embedding dimension gets its own sinusoidal frequency.
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )

        # sinusoidal waves used as they give smooth variation across positions. Because sine/cosine pattern shifts predictably, it can detect relative position quite well (e.g. token 2 is right after token 1)
        # also because they have multiple frequencies. Different dims use different frequencies
        # e.g. dim 0: faster-changing wave, dim 2: slower wave, dim 4: even slower. Lower-index dims capture more local positional variation, and higher-index dims capture broader trends
        # Each position gets a unique combination of these waves

        pe[:, 0::2] = torch.sin(position * div_term) # fills even dimensions with sine
        pe[:, 1::2] = torch.cos(position * div_term) # fills odd dimensions with cosine
        # each position gets a vector like: [sin(..), cos(...), sin(...) ... ]
        # sine alone is not enough because different positions can produce the same sine value.
        # pairing sine and cosine lets each frequency act like a 2D rotation on the unit circle.       
        # by employing trig identities, we can calculate distance between positions e.g. cos(a - b) = cos(a)cos(b) + sin(a)sin(b)

        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)
        # this is important as later we will add the positional encoding to the embeddings of shape (B, S, d_model). The leading 1 lets pytorch
        # broadcast the same positional encodings across the whole batch

    def forward(self, x):
        """
        x shape: (B, S, d_model)
        """
        S = x.size(1)
        return x + self.pe[:, :S, :] # so essentially (B, S, d_model) + (1, S, d_model)
    

class TinyTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_heads, d_ff, num_layers):
        super().__init__()

        # essentially a lookup table of shape (vocab_size, d_model)
        # so each token ID maps to an 8-dim vector. The embedding layer returns the corresponding rows from that table
        self.embedding = nn.Embedding(vocab_size, d_model)

        # integration between token embedding and position encoding is a simple addition. Concatenating would double dim and complicate architecture
        self.pos_encoding = PositionalEncoding(d_model)

        # ModuleList is used to tell pytorch that these are real submodules with params. A normal python list would not register properly for training
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff)
            for _ in range(num_layers)
        ])

        # each token vector becomes a 20-dim score vector. Each of these 20 numbers is a score for one possible vocab token i.e. logits.
        self.output_layer = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        """
        x shape: (B, S)
        """
        # note that x here is actually B number of sequences each of length S, and each value is the token ID (not token embedding)

        # Step 1: embedding
        # converts token IDs into vectors
        x = self.embedding(x)  # (B, S, d_model)

        # Step 2: add positional encoding
        x = self.pos_encoding(x)

        # Step 3: pass through transformer blocks
        for layer in self.layers:
            x = layer(x, mask=mask)

        # Step 4: project to vocab
        # each position produces a vocab-sized logit vector.
        # in next-token prediction training, this vector is used to predict the next token. 
        logits = self.output_layer(x)  # (B, S, vocab_size)

        return logits