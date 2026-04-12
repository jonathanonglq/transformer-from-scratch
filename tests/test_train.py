import torch
import torch.nn.functional as F

from model.tiny_transformer import TinyTransformer


def build_vocab(tokens: list[str]) -> dict[str, int]:
    # Preserve first-appearance order for reproducibility/readability
    return {tok: i for i, tok in enumerate(dict.fromkeys(tokens))}


def train_on_tiny_sequence(
    tokens: list[str],
    num_steps: int = 300,
    lr: float = 0.01,
) -> tuple[TinyTransformer, torch.Tensor, torch.Tensor, list[float], int]:
    vocab = build_vocab(tokens)
    vocab_size = len(vocab)

    token_ids = torch.tensor([[vocab[tok] for tok in tokens]], dtype=torch.long)
    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=2,
        d_ff=64,
        num_layers=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    losses: list[float] = []

    model.train()
    for _ in range(num_steps):
        logits = model(input_ids)  # (B, S-1, vocab_size)

        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    return model, input_ids, target_ids, losses, vocab_size


def test_loss_decreases():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]

    _, _, _, losses, _ = train_on_tiny_sequence(tokens, num_steps=100, lr=0.01)

    print("\nInitial loss:", losses[0])
    print("Final loss:  ", losses[-1])

    assert losses[-1] < losses[0], "Loss did not decrease during training"


def test_overfitting():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]

    model, input_ids, target_ids, losses, _ = train_on_tiny_sequence(
        tokens,
        num_steps=300,
        lr=0.01,
    )

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        preds = logits.argmax(dim=-1)

    print("\nFinal training loss:", losses[-1])
    print("Target IDs:", target_ids)
    print("Pred IDs:  ", preds)

    assert torch.equal(preds, target_ids), "Model failed to overfit the tiny dataset"


def test_output_shape_after_training():
    tokens = ["the", "cat", "sat", "on", "the", "mat"]

    model, input_ids, _, _, vocab_size = train_on_tiny_sequence(
        tokens,
        num_steps=20,
        lr=0.01,
    )

    model.eval()
    with torch.no_grad():
        logits = model(input_ids)

    print("\nLogits shape:", logits.shape)

    expected_shape = (input_ids.shape[0], input_ids.shape[1], vocab_size)
    assert logits.shape == expected_shape, "Logits shape is incorrect after training"


if __name__ == "__main__":
    test_loss_decreases()
    test_overfitting()
    test_output_shape_after_training()
    print("\nAll training tests passed.")