from pathlib import Path
import argparse

import matplotlib.pyplot as plt
import numpy as np


def build_positional_encoding_matrix(d_model: int, seq_len: int = 100) -> np.ndarray:
    """
    Returns positional encoding matrix with shape (d_model, seq_len).
    Rows are dimensions, columns are positions.
    Even dimensions use sine, odd dimensions use cosine.
    """
    positions = np.arange(seq_len, dtype=np.float32)[:, np.newaxis]  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2, dtype=np.float32) * (-np.log(10000.0) / d_model))

    # Build as (seq_len, d_model) first, then transpose to requested (d_model, seq_len).
    pe = np.zeros((seq_len, d_model), dtype=np.float32)
    pe[:, 0::2] = np.sin(positions * div_term)

    # For odd d_model, odd slice is shorter than even slice.
    odd_count = pe[:, 1::2].shape[1]
    pe[:, 1::2] = np.cos(positions * div_term[:odd_count])

    return pe.T


def plot_waves(pe_matrix: np.ndarray, output_path: Path, max_dims: int = 10) -> None:
    d_model, seq_len = pe_matrix.shape
    dims_to_plot = min(max_dims, d_model)
    x = np.arange(seq_len)

    fig, axes = plt.subplots(
        nrows=dims_to_plot,
        ncols=1,
        sharex=True,
        figsize=(11, max(2.2 * dims_to_plot, 6)),
    )

    if dims_to_plot == 1:
        axes = [axes]

    for dim, ax in enumerate(axes):
        ax.plot(x, pe_matrix[dim], linewidth=1.8, color="#1f77b4")
        ax.set_ylabel(f"dim {dim}", rotation=0, labelpad=30, va="center")
        ax.axhline(0.0, color="gray", linewidth=0.8)
        ax.grid(alpha=0.2)

    axes[0].set_title(f"Positional Encoding Waves (first {dims_to_plot} dimensions)")
    axes[-1].set_xlabel("Position")
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_heatmap(pe_matrix: np.ndarray, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 6))
    im = ax.imshow(pe_matrix, aspect="auto", cmap="coolwarm")
    ax.set_title("Positional Encoding Matrix Heatmap (d_model x seq_len)")
    ax.set_xlabel("Position")
    ax.set_ylabel("Dimension")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate positional encoding visualisations.")
    parser.add_argument("--d-model", type=int, default=32, help="Model dimension (number of rows).")
    parser.add_argument("--seq-len", type=int, default=100, help="Number of positions (number of columns).")
    args = parser.parse_args()

    if args.d_model < 2:
        raise ValueError("d_model must be at least 2.")
    if args.seq_len < 2:
        raise ValueError("seq_len must be at least 2.")

    pe_matrix = build_positional_encoding_matrix(d_model=args.d_model, seq_len=args.seq_len)

    diagrams_dir = Path(__file__).resolve().parents[1] / "diagrams"
    diagrams_dir.mkdir(parents=True, exist_ok=True)

    waves_path = diagrams_dir / "positional_encoding_waves.png"
    heatmap_path = diagrams_dir / "positional_encoding_heatmap.png"

    plot_waves(pe_matrix, waves_path, max_dims=10)
    plot_heatmap(pe_matrix, heatmap_path)

    print(f"Matrix shape: {pe_matrix.shape}")
    print(f"Saved: {waves_path}")
    print(f"Saved: {heatmap_path}")


if __name__ == "__main__":
    main()
