import torch
from model.transformer_block import TransformerBlock


def print_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def test_shapes():
    print_header("[TEST 1] Shape check")

    B, S, d_model, num_heads, d_ff = 2, 5, 8, 2, 32
    x = torch.rand(B, S, d_model)

    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block(x)

    print("input shape: ", x.shape)
    print("output shape:", output.shape)


def test_residual_effect():
    print_header("[TEST 2] Residual connection effect")

    B, S, d_model, num_heads, d_ff = 1, 3, 4, 2, 16
    x = torch.rand(B, S, d_model)

    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block(x)

    print("input:")
    print(x[0])

    print("\noutput:")
    print(output[0])

    print("\nObservation:")
    print("Output is not identical to input, but also not completely unrelated (residual adds refinement)")


def test_layernorm_stability():
    print_header("[TEST 3] LayerNorm stability")

    B, S, d_model, num_heads, d_ff = 2, 5, 8, 2, 32

    # intentionally large values
    x = torch.rand(B, S, d_model) * 1000

    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block(x)

    print("input mean:", x.mean().item())
    print("input std:", x.std().item())

    print("\noutput mean:", output.mean().item())
    print("output std:", output.std().item())

    print("\nObservation:")
    print("LayerNorm keeps output scale stable even when input is large")


def test_mask_propagation():
    print_header("[TEST 4] Mask propagation")

    B, S, d_model, num_heads, d_ff = 1, 5, 8, 2, 32
    x = torch.rand(B, S, d_model)

    # causal mask
    mask = torch.tril(torch.ones(S, S))
    mask = mask.unsqueeze(0).repeat(B, 1, 1)

    block = TransformerBlock(d_model, num_heads, d_ff)
    output = block(x, mask=mask)

    print("mask shape:", mask.shape)
    print("output shape:", output.shape)

    print("\nObservation:")
    print("Mask is passed through to attention inside the block")


def test_multiple_passes_consistency():
    print_header("[TEST 5] Multiple passes consistency")

    B, S, d_model, num_heads, d_ff = 1, 5, 8, 2, 32
    x = torch.rand(B, S, d_model)

    block = TransformerBlock(d_model, num_heads, d_ff)

    out1 = block(x)
    out2 = block(x)

    print("difference between outputs:", torch.abs(out1 - out2).mean().item())

    print("\nObservation:")
    print("Same input → same output (no randomness inside forward pass)")


if __name__ == "__main__":
    test_shapes()
    test_residual_effect()
    test_layernorm_stability()
    test_mask_propagation()
    test_multiple_passes_consistency()