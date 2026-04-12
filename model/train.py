import torch
import torch.nn.functional as F

from model.tiny_transformer import TinyTransformer


def build_vocab(tokens: list[str]) -> tuple[dict[str, int], dict[int, str]]:
    # Preserve first-appearance order for reproducibility/readability
    vocab = {tok: i for i, tok in enumerate(dict.fromkeys(tokens))}
    inv_vocab = {i: tok for tok, i in vocab.items()}
    return vocab, inv_vocab


def train_and_evaluate() -> None:
    # Toy sequence
    tokens = ["the", "cat", "sat", "on", "the", "mat"]

    vocab, inv_vocab = build_vocab(tokens)
    vocab_size = len(vocab)

    # Shape: (B, S)
    token_ids = torch.tensor([[vocab[tok] for tok in tokens]], dtype=torch.long)

    # Next-token prediction setup
    input_ids = token_ids[:, :-1]   # "the cat sat on the"
    target_ids = token_ids[:, 1:]   # "cat sat on the mat"

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=2,
        d_ff=64,
        num_layers=2,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    num_steps = 300

    for step in range(num_steps):
        # Forward pass
        logits = model(input_ids)  # (B, S-1, vocab_size)

        # Flatten for cross-entropy:
        # logits:  (B*(S-1), vocab_size)
        # targets: (B*(S-1))
        logits_flat = logits.reshape(-1, vocab_size)
        targets_flat = target_ids.reshape(-1)

        loss = F.cross_entropy(logits_flat, targets_flat)

        # Backprop + update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step % 50 == 0 or step == num_steps - 1:
            print(f"step {step:3d} | loss {loss.item():.4f}")

    print("\n=== Training complete ===")

    # Evaluation
    model.eval()
    with torch.no_grad():
        logits = model(input_ids)
        preds = logits.argmax(dim=-1)

    input_tokens = [inv_vocab[i.item()] for i in input_ids[0]]
    target_tokens = [inv_vocab[i.item()] for i in target_ids[0]]
    pred_tokens = [inv_vocab[i.item()] for i in preds[0]]

    print("\nInput tokens:")
    print(input_tokens)

    print("\nTarget tokens:")
    print(target_tokens)

    print("\nPredicted tokens:")
    print(pred_tokens)

    if torch.equal(preds, target_ids):
        print("\nModel successfully overfit the tiny dataset.")
    else:
        print("\nModel did not fully overfit yet. Try more steps or a higher learning rate.")


if __name__ == "__main__":
    train_and_evaluate()