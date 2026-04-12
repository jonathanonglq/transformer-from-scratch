import torch
from model.tiny_transformer import TinyTransformer


def print_header(title):
    print("\n" + "=" * 50)
    print(title)
    print("=" * 50)


def test_shapes():
    print_header("[TEST 1] Shape check")

    B, S = 2, 5
    vocab_size = 20

    x = torch.randint(0, vocab_size, (B, S))

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=8,
        num_heads=2,
        d_ff=32,
        num_layers=2,
    )

    logits = model(x)

    print("input shape: ", x.shape)
    print("logits shape:", logits.shape)

    assert logits.shape == (B, S, vocab_size), "Shape mismatch in output"


def test_output_interpretation():
    print_header("[TEST 2] Output interpretation")

    B, S = 1, 5
    vocab_size = 10

    x = torch.randint(0, vocab_size, (B, S))

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=8,
        num_heads=2,
        d_ff=32,
        num_layers=1,
    )

    logits = model(x)

    print("input token IDs:")
    print(x)

    print("\nlogits shape:", logits.shape)
    print("\nlogits for first position:")
    print(logits[0, 0])

    # Assert each position produces vocab-sized vector
    assert logits.shape[-1] == vocab_size, "Last dimension must equal vocab size"

    print("\nObservation:")
    print("Each position outputs a vocab-sized score vector.")


def test_masked_forward():
    print_header("[TEST 3] Masked forward pass")

    B, S = 2, 5
    vocab_size = 20

    x = torch.randint(0, vocab_size, (B, S))

    mask = torch.tril(torch.ones(S, S))
    mask = mask.unsqueeze(0).repeat(B, 1, 1)

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=8,
        num_heads=2,
        d_ff=32,
        num_layers=2,
    )

    logits = model(x, mask=mask)

    print("mask shape:  ", mask.shape)
    print("logits shape:", logits.shape)

    assert logits.shape == (B, S, vocab_size), "Masked forward shape mismatch"

    print("\nObservation:")
    print("Mask passes through model correctly.")


def test_determinism():
    print_header("[TEST 4] Determinism check")

    B, S = 1, 5
    vocab_size = 15

    x = torch.randint(0, vocab_size, (B, S))

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=8,
        num_heads=2,
        d_ff=32,
        num_layers=1,
    )

    logits1 = model(x)
    logits2 = model(x)

    diff = torch.abs(logits1 - logits2).mean().item()

    print("mean absolute difference:", diff)

    assert diff < 1e-6, "Model is not deterministic"

    print("\nObservation:")
    print("Same input → same output.")


def test_vocab_dimension():
    print_header("[TEST 5] Vocab dimension check")

    B, S = 1, 4
    vocab_size = 7

    x = torch.randint(0, vocab_size, (B, S))

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=8,
        num_heads=2,
        d_ff=16,
        num_layers=1,
    )

    logits = model(x)

    print("expected last dim:", vocab_size)
    print("actual logits shape:", logits.shape)

    assert logits.shape[-1] == vocab_size, "Incorrect vocab dimension"

    print("\nObservation:")
    print("Last dimension matches vocab size.")


if __name__ == "__main__":
    test_shapes()
    test_output_interpretation()
    test_masked_forward()
    test_determinism()
    test_vocab_dimension()