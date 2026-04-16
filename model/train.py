import random
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F

from model.tiny_transformer import TinyTransformer

PAD_TOKEN = "<pad>"
EOS_TOKEN = "<eos>"
UNK_TOKEN = "<unk>"


def parse_tokens(text: str) -> list[str]:
    return [tok for tok in text.strip().split() if tok]


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)


def load_corpus(corpus_path: Path) -> list[list[str]]:
    if not corpus_path.exists():
        raise FileNotFoundError(f"Corpus file not found: {corpus_path}")

    lines = corpus_path.read_text(encoding="utf-8").splitlines()
    sentences = []
    for line in lines:
        tokens = parse_tokens(line)
        if tokens:
            sentences.append(tokens)

    if len(sentences) < 2:
        raise ValueError("Corpus needs at least 2 non-empty sentences.")

    return sentences


def split_train_val(
    sentences: list[list[str]],
    val_ratio: float,
    seed: int,
) -> tuple[list[list[str]], list[list[str]]]:
    indices = list(range(len(sentences)))
    random.Random(seed).shuffle(indices)

    val_size = max(1, int(len(sentences) * val_ratio))
    val_indices = set(indices[:val_size])

    train_sentences = [sentences[i] for i in range(len(sentences)) if i not in val_indices]
    val_sentences = [sentences[i] for i in range(len(sentences)) if i in val_indices]

    if len(train_sentences) == 0 or len(val_sentences) == 0:
        raise ValueError("Train/validation split produced an empty partition.")

    return train_sentences, val_sentences


def build_vocab(sentences: list[list[str]]) -> tuple[dict[str, int], dict[int, str]]:
    vocab = {
        PAD_TOKEN: 0,
        EOS_TOKEN: 1,
        UNK_TOKEN: 2,
    }

    for sent in sentences:
        for tok in sent:
            if tok not in vocab:
                vocab[tok] = len(vocab)

    inv_vocab = {i: tok for tok, i in vocab.items()}
    return vocab, inv_vocab


def encode_sentence(tokens: list[str], vocab: dict[str, int]) -> list[int]:
    unk_idx = vocab[UNK_TOKEN]
    eos_idx = vocab[EOS_TOKEN]
    ids = [vocab.get(tok, unk_idx) for tok in tokens]
    ids.append(eos_idx)
    return ids


def build_examples(
    sentences: list[list[str]],
    vocab: dict[str, int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    examples: list[tuple[torch.Tensor, torch.Tensor]] = []

    for sent in sentences:
        ids = encode_sentence(sent, vocab)
        if len(ids) < 2:
            continue

        input_ids = torch.tensor(ids[:-1], dtype=torch.long)
        target_ids = torch.tensor(ids[1:], dtype=torch.long)
        examples.append((input_ids, target_ids))

    if len(examples) == 0:
        raise ValueError("No usable training examples built from corpus.")

    return examples


def batch_iterator(
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    seed: int,
) -> Iterable[list[tuple[torch.Tensor, torch.Tensor]]]:
    rng = random.Random(seed)
    indices = list(range(len(examples)))

    while True:
        rng.shuffle(indices)
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start:start + batch_size]
            yield [examples[i] for i in batch_indices]


def collate_batch(
    batch_examples: list[tuple[torch.Tensor, torch.Tensor]],
    pad_idx: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_len = max(x[0].size(0) for x in batch_examples)
    batch_size = len(batch_examples)

    input_batch = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)
    target_batch = torch.full((batch_size, max_len), pad_idx, dtype=torch.long)

    for i, (inp, tgt) in enumerate(batch_examples):
        seq_len = inp.size(0)
        input_batch[i, :seq_len] = inp
        target_batch[i, :seq_len] = tgt

    return input_batch, target_batch


def build_causal_padding_mask(input_ids: torch.Tensor, pad_idx: int) -> torch.Tensor:
    batch_size, seq_len = input_ids.shape

    causal = torch.tril(
        torch.ones(seq_len, seq_len, dtype=torch.bool, device=input_ids.device)
    )
    causal = causal.unsqueeze(0).expand(batch_size, -1, -1)  # (B, S, S)

    # Mask padding keys so no token attends to <pad>.
    # Do not mask query rows here; fully-masked query rows can cause NaNs in softmax.
    key_not_pad = (input_ids != pad_idx).unsqueeze(1).expand(-1, seq_len, -1)
    return causal & key_not_pad


def evaluate_loss(
    model: TinyTransformer,
    examples: list[tuple[torch.Tensor, torch.Tensor]],
    batch_size: int,
    pad_idx: int,
    vocab_size: int,
    device: torch.device,
) -> float:
    losses = []
    model.eval()

    with torch.no_grad():
        for start in range(0, len(examples), batch_size):
            batch_examples = examples[start:start + batch_size]
            input_ids, target_ids = collate_batch(batch_examples, pad_idx=pad_idx)
            input_ids = input_ids.to(device)
            target_ids = target_ids.to(device)

            mask = build_causal_padding_mask(input_ids, pad_idx=pad_idx)
            logits = model(input_ids, mask=mask)
            loss = F.cross_entropy(
                logits.reshape(-1, vocab_size),
                target_ids.reshape(-1),
                ignore_index=pad_idx,
            )
            losses.append(float(loss.item()))

    model.train()
    return sum(losses) / len(losses)


def decode_ids(ids: torch.Tensor, inv_vocab: dict[int, str], pad_idx: int) -> list[str]:
    tokens = []
    for idx in ids.tolist():
        if idx == pad_idx:
            continue
        tokens.append(inv_vocab[idx])
    return tokens


def train_and_evaluate(
    corpus_path: str | None = None,
    num_steps: int = 300,
    batch_size: int = 16,
    learning_rate: float = 0.01,
    val_ratio: float = 0.2,
    seed: int = 42,
) -> None:
    set_seed(seed)

    if corpus_path is None:
        corpus = Path(__file__).resolve().parents[1] / "data" / "synthetic_corpus_100.txt"
    else:
        corpus = Path(corpus_path)

    sentences = load_corpus(corpus)
    train_sentences, val_sentences = split_train_val(sentences, val_ratio=val_ratio, seed=seed)

    vocab, inv_vocab = build_vocab(train_sentences)
    train_examples = build_examples(train_sentences, vocab)
    val_examples = build_examples(val_sentences, vocab)

    pad_idx = vocab[PAD_TOKEN]
    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=16,
        num_heads=2,
        d_ff=64,
        num_layers=2,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    batches = batch_iterator(train_examples, batch_size=max(1, batch_size), seed=seed + 1)
    log_every = max(1, num_steps // 10)

    print(f"Corpus: {corpus}")
    print(f"Sentences: total={len(sentences)} train={len(train_sentences)} val={len(val_sentences)}")
    print(f"Examples: train={len(train_examples)} val={len(val_examples)}")
    print(f"Vocab size: {vocab_size}")
    print(f"Device: {device}")

    for step in range(1, num_steps + 1):
        batch_examples = next(batches)
        input_ids, target_ids = collate_batch(batch_examples, pad_idx=pad_idx)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        mask = build_causal_padding_mask(input_ids, pad_idx=pad_idx)
        logits = model(input_ids, mask=mask)
        loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=pad_idx,
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if step == 1 or step % log_every == 0 or step == num_steps:
            val_loss = evaluate_loss(
                model=model,
                examples=val_examples,
                batch_size=max(1, batch_size),
                pad_idx=pad_idx,
                vocab_size=vocab_size,
                device=device,
            )
            print(
                f"step {step:3d} | train loss {loss.item():.4f} | val loss {val_loss:.4f}"
            )

    print("\n=== Training complete ===")

    # Show one validation example for quick qualitative inspection.
    model.eval()
    sample_input, sample_target = val_examples[0]
    sample_input_batch = sample_input.unsqueeze(0).to(device)
    sample_mask = build_causal_padding_mask(sample_input_batch, pad_idx=pad_idx)

    with torch.no_grad():
        sample_logits = model(sample_input_batch, mask=sample_mask)
        sample_preds = sample_logits.argmax(dim=-1).squeeze(0).cpu()

    input_tokens = decode_ids(sample_input, inv_vocab, pad_idx=pad_idx)
    target_tokens = decode_ids(sample_target, inv_vocab, pad_idx=pad_idx)
    pred_tokens = decode_ids(sample_preds, inv_vocab, pad_idx=pad_idx)

    print("\nValidation sample (first batch item):")
    print("Input tokens:   ", input_tokens)
    print("Target tokens:  ", target_tokens)
    print("Predicted tokens:", pred_tokens)


if __name__ == "__main__":
    train_and_evaluate()
