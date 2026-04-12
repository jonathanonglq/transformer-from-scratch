import torch
import torch.nn as nn
import torch.nn.functional as F

# we project x onto a higher-dim space, perform non-linear transformations, then project back. This gives the model more expressive power.
# important to note that FFN operates independently per token.
# without FFN, attention can only re-weight existing info, and cannot create richer representations

class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)

    def forward(self, x):
        """
        x shape: (B, S, d_model)
        """

        # Step 1: expand dimension
        x = self.linear1(x)   # (B, S, d_ff)

        # Step 2: non-linearity
        x = F.relu(x)

        # Step 3: project back
        x = self.linear2(x)   # (B, S, d_model)

        return x