import torch
from model.multi_head_attention import MultiHeadAttention


def print_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def test_output_shapes():
    print_header("[TEST 1] Output shape check")

    B, S, d_model, num_heads = 2, 5, 8, 2
    x = torch.rand(B, S, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, weights = mha(x)

    print("input shape:  ", x.shape)
    print("output shape: ", output.shape)   # expect (2, 5, 8)
    print("weights shape:", weights.shape)  # expect (2, 2, 5, 5)


def test_softmax_row_sums():
    print_header("[TEST 2] Softmax row sum check")

    B, S, d_model, num_heads = 2, 5, 8, 2
    x = torch.rand(B, S, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    _, weights = mha(x)

    row_sums = weights.sum(dim=-1)
    print("row sums shape:", row_sums.shape)  # expect (2, 2, 5)
    print("row sums:\n", row_sums)


def test_mask_behavior():
    print_header("[TEST 3] Mask behavior check")

    B, S, d_model, num_heads = 2, 5, 8, 2
    x = torch.rand(B, S, d_model)

    # torch.tril here creates a S x S matrix, bottom triangle filled with 1s, and 0s for the rest. THis is a causal mask.
    mask = torch.tril(torch.ones(S, S))

    # unsqueeze(0) here converts (S, S) -> (1, S, S). repeat(B, 1, 1) converts (1, S, S) -> (B, S, S). Now we have one mask per batch item. 
    # recall that in forward(), we do unsqueeze(1), which converts (B, S, S) -> (B, 1, S, S). PyTorch will then broadcast (B, 1, S, S) to all heads (i.e. B, num_heads, S, S)
    
    mask = mask.unsqueeze(0).repeat(B, 1, 1)  # (B, S, S)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    _, weights = mha(x, mask=mask)

    print("mask shape before unsqueeze in forward:", mask.shape)
    print("weights shape:", weights.shape)  # expect (2, 2, 5, 5)

    print("\nHead 0 masked attention (batch 0):\n", weights[0, 0])
    print("\nHead 1 masked attention (batch 0):\n", weights[0, 1])


def test_heads_are_separate():
    print_header("[TEST 4] Heads are separate")

    B, S, d_model, num_heads = 2, 5, 8, 2
    x = torch.rand(B, S, d_model)

    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    _, weights = mha(x)

    print("Head 0 attention (batch 0):\n", weights[0, 0])
    print("\nHead 1 attention (batch 0):\n", weights[0, 1])

    same = torch.allclose(weights[0, 0], weights[0, 1])
    print("\nAre head 0 and head 1 exactly the same?", same)


def test_invalid_config():
    print_header("[TEST 5] Invalid configuration check")

    try:
        _ = MultiHeadAttention(d_model=10, num_heads=3)
        print("Unexpected: no error raised")
    except AssertionError as e:
        print("Caught expected assertion error:")
        print(e)


if __name__ == "__main__":
    test_output_shapes()
    test_softmax_row_sums()
    test_mask_behavior()
    test_heads_are_separate()
    test_invalid_config()