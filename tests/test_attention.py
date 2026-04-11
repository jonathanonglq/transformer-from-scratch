import torch
from model.attention import ScaledDotProductAttention

def print_header(title):
    print("\n" + "="*50)
    print(title)
    print("="*50)


def test_shapes():
    print_header("\n[TEST 1] Shape check")

    B, S, D = 2, 5, 4
    Q = torch.rand(B, S, D)
    K = torch.rand(B, S, D)
    V = torch.rand(B, S, D)

    attn = ScaledDotProductAttention()
    output, weights = attn(Q, K, V)

    print("output shape:", output.shape)
    print("weights shape:", weights.shape)


def test_softmax():
    print_header("\n[TEST 2] Softmax row sum check")

    B, S, D = 2, 5, 4
    Q = torch.rand(B, S, D)
    K = torch.rand(B, S, D)
    V = torch.rand(B, S, D)

    attn = ScaledDotProductAttention()
    _, weights = attn(Q, K, V)

    row_sums = weights.sum(dim=-1)
    print("row sums:\n", row_sums)


def test_identity_attention():
    print_header("\n[TEST 3] Identity test (Q = K)")

    B, S, D = 2, 5, 4
    Q = torch.rand(B, S, D)
    K = Q.clone()
    V = torch.rand(B, S, D)

    attn = ScaledDotProductAttention()
    _, weights = attn(Q, K, V)

    print("attention weights (batch 0):\n", weights[0])


def test_mask():
    print_header("\n[TEST 4] Mask test")

    B, S, D = 2, 5, 4
    Q = torch.rand(B, S, D)
    K = torch.rand(B, S, D)
    V = torch.rand(B, S, D)

    mask = torch.tril(torch.ones(S, S))
    mask = mask.unsqueeze(0).repeat(B, 1, 1)

    attn = ScaledDotProductAttention()
    _, weights = attn(Q, K, V, mask=mask)

    print("masked attention (batch 0):\n", weights[0])


if __name__ == "__main__":
    test_shapes()
    test_softmax()
    test_identity_attention()
    test_mask()