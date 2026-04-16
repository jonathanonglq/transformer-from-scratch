from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from model.tiny_transformer import TinyTransformer
from model.train import (
    PAD_TOKEN,
    batch_iterator,
    build_causal_padding_mask,
    build_examples,
    build_vocab,
    collate_batch,
    decode_ids,
    evaluate_loss,
    load_corpus,
    set_seed,
    split_train_val,
)


def parse_args() -> argparse.Namespace:
    project_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description="Train on corpus and save visual artifacts.")
    parser.add_argument(
        "--corpus-path",
        type=str,
        default=str(project_root / "data" / "synthetic_corpus_100.txt"),
        help="Path to corpus file (one sentence per line).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(project_root / "diagrams" / "training_artifacts"),
        help="Directory where artifact files are written.",
    )
    parser.add_argument("--num-steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--d-model", type=int, default=16)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--d-ff", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=2)
    parser.add_argument("--sample-index", type=int, default=0, help="Validation sample index to visualize.")
    parser.add_argument("--top-k", type=int, default=5)
    return parser.parse_args()


def forward_with_attention(
    model: TinyTransformer,
    sample_input: torch.Tensor,
    pad_idx: int,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    mask = build_causal_padding_mask(sample_input, pad_idx=pad_idx)
    logits, attentions = model(sample_input, mask=mask, return_attentions=True)
    return logits, attentions


def collect_validation_sample(
    model: TinyTransformer,
    sample_input: torch.Tensor,
    sample_target: torch.Tensor,
    inv_vocab: dict[int, str],
    pad_idx: int,
    top_k: int,
) -> tuple[list[dict], list[float], list[float], list[dict]]:
    model.eval()
    with torch.no_grad():
        logits, _ = forward_with_attention(model, sample_input, pad_idx=pad_idx)
        probs = F.softmax(logits, dim=-1)[0]
        pred_ids = logits.argmax(dim=-1)[0]

    input_tokens = decode_ids(sample_input.squeeze(0).cpu(), inv_vocab, pad_idx=pad_idx)
    target_tokens = decode_ids(sample_target.squeeze(0).cpu(), inv_vocab, pad_idx=pad_idx)
    pred_tokens = decode_ids(pred_ids.cpu(), inv_vocab, pad_idx=pad_idx)

    rows: list[dict] = []
    pred_confidence: list[float] = []
    target_confidence: list[float] = []
    topk_rows: list[dict] = []

    for pos in range(len(input_tokens)):
        pred_id = int(pred_ids[pos].item())
        target_id = int(sample_target[0, pos].item())

        p_pred = float(probs[pos, pred_id].item())
        p_target = float(probs[pos, target_id].item())

        pred_confidence.append(p_pred)
        target_confidence.append(p_target)

        rows.append(
            {
                "position": pos,
                "input_token": input_tokens[pos],
                "target_next": target_tokens[pos],
                "pred_next": pred_tokens[pos],
                "correct": int(pred_id == target_id),
                "pred_confidence": f"{p_pred:.6f}",
                "target_confidence": f"{p_target:.6f}",
            }
        )

        k = min(top_k, probs.shape[-1])
        top_probs, top_ids = torch.topk(probs[pos], k=k)
        topk_rows.append(
            {
                "position": pos,
                "input_token": input_tokens[pos],
                "target_next": target_tokens[pos],
                "pred_next": pred_tokens[pos],
                "top_tokens": [inv_vocab[int(i.item())] for i in top_ids],
                "top_probs": [float(p.item()) for p in top_probs],
            }
        )

    return rows, pred_confidence, target_confidence, topk_rows


def save_csv(rows: Sequence[dict], output_path: Path) -> None:
    if not rows:
        return
    with output_path.open("w", encoding="utf-8", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def plot_learning_curve(history: list[tuple[int, float, float]], output_path: Path) -> None:
    steps = [x[0] for x in history]
    train_losses = [x[1] for x in history]
    val_losses = [x[2] for x in history]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(steps, train_losses, marker="o", label="train loss")
    ax.plot(steps, val_losses, marker="o", label="validation loss")
    ax.set_title("Learning Curve")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Cross-entropy loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_attention_before_after(
    before: list[np.ndarray],
    after: list[np.ndarray],
    tokens: list[str],
    output_path: Path,
) -> None:
    if not before or not after:
        return

    num_layers = len(before)
    num_heads = int(before[0].shape[0])
    ncols = num_heads * 2  # before + after for each head

    fig, axes = plt.subplots(
        nrows=num_layers,
        ncols=ncols,
        figsize=(max(4.0 * ncols, 12), max(3.3 * num_layers, 4)),
        squeeze=False,
    )

    im = None
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            before_ax = axes[layer_idx][2 * head_idx]
            after_ax = axes[layer_idx][2 * head_idx + 1]

            im = before_ax.imshow(before[layer_idx][head_idx], vmin=0.0, vmax=1.0, cmap="viridis")
            before_ax.set_title(f"L{layer_idx} H{head_idx} Before", fontsize=10)

            after_ax.imshow(after[layer_idx][head_idx], vmin=0.0, vmax=1.0, cmap="viridis")
            after_ax.set_title(f"L{layer_idx} H{head_idx} After", fontsize=10)

            for ax in (before_ax, after_ax):
                ax.set_xticks(range(len(tokens)))
                ax.set_yticks(range(len(tokens)))
                ax.set_xticklabels(tokens, rotation=45, ha="right", fontsize=7)
                ax.set_yticklabels(tokens, fontsize=7)
                ax.set_xlabel("Key token", fontsize=8)
                ax.set_ylabel("Query token", fontsize=8)

    fig.subplots_adjust(right=0.92, hspace=0.5, wspace=0.35)
    if im is not None:
        cax = fig.add_axes([0.94, 0.15, 0.015, 0.7])
        fig.colorbar(im, cax=cax, label="Attention weight")
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_validation_confidence(
    pred_confidence: list[float],
    target_confidence: list[float],
    output_path: Path,
) -> None:
    xs = list(range(len(pred_confidence)))
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(xs, pred_confidence, marker="o", label="P(predicted token)")
    ax.plot(xs, target_confidence, marker="o", label="P(target token)")
    ax.set_ylim(0.0, 1.0)
    ax.set_title("Validation Sample Confidence by Position")
    ax.set_xlabel("Position")
    ax.set_ylabel("Probability")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_topk_positions(
    topk_rows: list[dict],
    output_path: Path,
) -> None:
    if not topk_rows:
        return

    num_cols = 2
    n = len(topk_rows)
    nrows = (n + num_cols - 1) // num_cols
    fig, axes = plt.subplots(
        nrows=nrows,
        ncols=num_cols,
        figsize=(12, max(3.0 * nrows, 3.0)),
        squeeze=False,
    )

    for idx, row in enumerate(topk_rows):
        r = idx // num_cols
        c = idx % num_cols
        ax = axes[r][c]
        ax.bar(row["top_tokens"], row["top_probs"])
        ax.set_ylim(0.0, 1.0)
        ax.set_title(
            f"Pos {row['position']} | input={row['input_token']} | target={row['target_next']} | pred={row['pred_next']}"
        )
        ax.set_ylabel("Prob")
        ax.tick_params(axis="x", rotation=40)

    for idx in range(n, nrows * num_cols):
        r = idx // num_cols
        c = idx % num_cols
        axes[r][c].axis("off")

    for c in range(num_cols):
        axes[nrows - 1][c].set_xlabel("Token")

    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    set_seed(args.seed)

    corpus_path = Path(args.corpus_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    sentences = load_corpus(corpus_path)
    train_sentences, val_sentences = split_train_val(
        sentences=sentences,
        val_ratio=args.val_ratio,
        seed=args.seed,
    )

    vocab, inv_vocab = build_vocab(train_sentences)
    train_examples = build_examples(train_sentences, vocab)
    val_examples = build_examples(val_sentences, vocab)

    if not val_examples:
        raise ValueError("Validation set is empty.")

    sample_index = min(max(args.sample_index, 0), len(val_examples) - 1)
    sample_input, sample_target = val_examples[sample_index]
    sample_input_batch = sample_input.unsqueeze(0)
    sample_target_batch = sample_target.unsqueeze(0)

    pad_idx = vocab[PAD_TOKEN]
    vocab_size = len(vocab)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=args.d_model,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        num_layers=args.num_layers,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    batches = batch_iterator(train_examples, batch_size=max(1, args.batch_size), seed=args.seed + 1)

    # Capture attention before training.
    model.eval()
    with torch.no_grad():
        _, before_attns = forward_with_attention(
            model=model,
            sample_input=sample_input_batch.to(device),
            pad_idx=pad_idx,
        )
    before_attn = [attn[0].detach().cpu().numpy() for attn in before_attns]

    history: list[tuple[int, float, float]] = []
    eval_every = max(1, args.eval_every)

    for step in range(1, args.num_steps + 1):
        batch_examples = next(batches)
        input_ids, target_ids = collate_batch(batch_examples, pad_idx=pad_idx)
        input_ids = input_ids.to(device)
        target_ids = target_ids.to(device)

        mask = build_causal_padding_mask(input_ids, pad_idx=pad_idx)
        logits = model(input_ids, mask=mask)
        train_loss = F.cross_entropy(
            logits.reshape(-1, vocab_size),
            target_ids.reshape(-1),
            ignore_index=pad_idx,
        )

        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()

        if step == 1 or step % eval_every == 0 or step == args.num_steps:
            val_loss = evaluate_loss(
                model=model,
                examples=val_examples,
                batch_size=max(1, args.batch_size),
                pad_idx=pad_idx,
                vocab_size=vocab_size,
                device=device,
            )
            history.append((step, float(train_loss.item()), float(val_loss)))

    # Capture attention after training.
    model.eval()
    with torch.no_grad():
        after_logits, after_attns = forward_with_attention(
            model=model,
            sample_input=sample_input_batch.to(device),
            pad_idx=pad_idx,
        )
    after_attn = [attn[0].detach().cpu().numpy() for attn in after_attns]

    # Collect sample-level outputs.
    rows, pred_conf, tgt_conf, topk_rows = collect_validation_sample(
        model=model,
        sample_input=sample_input_batch.to(device),
        sample_target=sample_target_batch.to(device),
        inv_vocab=inv_vocab,
        pad_idx=pad_idx,
        top_k=args.top_k,
    )

    input_tokens = decode_ids(sample_input, inv_vocab, pad_idx=pad_idx)
    sample_loss = F.cross_entropy(
        after_logits.reshape(-1, vocab_size),
        sample_target_batch.to(device).reshape(-1),
        ignore_index=pad_idx,
    ).item()
    token_accuracy = (
        sum(int(r["correct"]) for r in rows) / len(rows) if rows else float("nan")
    )

    # Save artifacts.
    plot_learning_curve(history, output_dir / "learning_curve.png")
    plot_attention_before_after(
        before=before_attn,
        after=after_attn,
        tokens=input_tokens,
        output_path=output_dir / "attention_before_after.png",
    )
    plot_validation_confidence(pred_conf, tgt_conf, output_dir / "validation_confidence.png")
    plot_topk_positions(topk_rows, output_dir / "validation_topk_positions.png")

    save_csv(rows, output_dir / "validation_predictions.csv")
    save_csv(history_to_rows(history), output_dir / "loss_history.csv")

    summary_path = output_dir / "run_summary.txt"
    summary_path.write_text(
        "\n".join(
            [
                f"corpus_path={corpus_path}",
                f"total_sentences={len(sentences)}",
                f"train_sentences={len(train_sentences)}",
                f"val_sentences={len(val_sentences)}",
                f"train_examples={len(train_examples)}",
                f"val_examples={len(val_examples)}",
                f"vocab_size={vocab_size}",
                f"device={device}",
                f"sample_index={sample_index}",
                f"sample_loss={sample_loss:.6f}",
                f"sample_token_accuracy={token_accuracy:.6f}",
                f"artifacts_dir={output_dir}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Saved artifacts to: {output_dir}")
    print(f"- {output_dir / 'learning_curve.png'}")
    print(f"- {output_dir / 'attention_before_after.png'}")
    print(f"- {output_dir / 'validation_confidence.png'}")
    print(f"- {output_dir / 'validation_topk_positions.png'}")
    print(f"- {output_dir / 'validation_predictions.csv'}")
    print(f"- {output_dir / 'loss_history.csv'}")
    print(f"- {output_dir / 'run_summary.txt'}")


def history_to_rows(history: list[tuple[int, float, float]]) -> list[dict]:
    return [
        {
            "step": step,
            "train_loss": f"{train_loss:.6f}",
            "val_loss": f"{val_loss:.6f}",
        }
        for step, train_loss, val_loss in history
    ]


if __name__ == "__main__":
    main()
