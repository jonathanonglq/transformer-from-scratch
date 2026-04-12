import torch
from model.feedforward import FeedForward


def print_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def test_shapes():
    print_header("[TEST 1] Shape check")

    B, S, d_model, d_ff = 2, 5, 8, 32
    x = torch.rand(B, S, d_model)

    ffn = FeedForward(d_model=d_model, d_ff=d_ff)
    output = ffn(x)

    print("input shape: ", x.shape)
    print("output shape:", output.shape)


def test_token_independence():
    print_header("[TEST 2] Token independence")

    B, S, d_model, d_ff = 1, 3, 4, 16

    x = torch.rand(B, S, d_model)

    ffn = FeedForward(d_model=d_model, d_ff=d_ff)

    output = ffn(x)

    print("input:")
    print(x[0])

    print("\noutput:")
    print(output[0])

    print("\nObservation:")
    print("Each token transformed independently (no mixing across tokens)")


if __name__ == "__main__":
    test_shapes()
    test_token_independence()