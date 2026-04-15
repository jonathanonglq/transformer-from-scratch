import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
    
from typing import List, Tuple
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F

from model.attention import ScaledDotProductAttention
from model.multi_head_attention import MultiHeadAttention
from model.tiny_transformer import TinyTransformer


# -----------------------------
# Helpers
# -----------------------------
def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)


def build_causal_mask(seq_len: int, batch_size: int = 1) -> torch.Tensor:
    mask = torch.tril(torch.ones(seq_len, seq_len))
    mask = mask.unsqueeze(0).repeat(batch_size, 1, 1)
    return mask


def parse_tokens(text: str) -> List[str]:
    return [tok for tok in text.strip().split() if tok]


def build_random_embeddings(seq_len: int, d_model: int) -> torch.Tensor:
    return torch.rand(1, seq_len, d_model)


def build_learned_embeddings(tokens: List[str], d_model: int) -> torch.Tensor:
    vocab = {tok: i for i, tok in enumerate(dict.fromkeys(tokens))}
    token_ids = torch.tensor([[vocab[tok] for tok in tokens]], dtype=torch.long)
    embedding = nn.Embedding(len(vocab), d_model)
    return embedding(token_ids)


def build_vocab(tokens: List[str]) -> Tuple[dict, dict]:
    vocab = {tok: i for i, tok in enumerate(dict.fromkeys(tokens))}
    inv_vocab = {i: tok for tok, i in vocab.items()}
    return vocab, inv_vocab


def plot_attention_heatmap(weights: np.ndarray, tokens: List[str], title: str):
    fig, ax = plt.subplots(figsize=(max(5, len(tokens) * 0.9), max(4, len(tokens) * 0.7)))
    im = ax.imshow(weights)
    ax.set_title(title)
    ax.set_xticks(range(len(tokens)))
    ax.set_yticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=45, ha="right")
    ax.set_yticklabels(tokens)
    ax.set_xlabel("Key / attended-to token")
    ax.set_ylabel("Query / current token")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    return fig


def highlight_attention(weights: np.ndarray, tokens: List[str], row_idx: int) -> str:
    row = weights[row_idx]
    pairs = list(zip(tokens, row.tolist()))
    pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
    return ", ".join([f"{tok}: {score:.3f}" for tok, score in pairs])


def train_and_capture_checkpoints(
    tokens: List[str],
    d_model: int,
    num_heads: int,
    d_ff: int,
    num_layers: int,
    num_steps: int,
    learning_rate: float,
):
    vocab, inv_vocab = build_vocab(tokens)
    vocab_size = len(vocab)

    token_ids = torch.tensor([[vocab[tok] for tok in tokens]], dtype=torch.long)
    input_ids = token_ids[:, :-1]
    target_ids = token_ids[:, 1:]
    mask = build_causal_mask(input_ids.shape[1], batch_size=input_ids.shape[0])

    model = TinyTransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    checkpoint_steps = sorted(set([0, max(1, num_steps // 10), max(1, num_steps // 4), max(1, num_steps // 2), num_steps]))
    captured = []
    losses = []

    def capture(step: int):
        model.eval()
        with torch.no_grad():
            logits, all_attentions = model(input_ids, mask=mask, return_attentions=True)
            loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1)).item()
            preds = logits.argmax(dim=-1)
            captured.append({
                "step": step,
                "loss": loss,
                "preds": preds[0].cpu().tolist(),
                "attentions": [a[0].cpu().numpy() for a in all_attentions],
            })
        model.train()

    capture(0)

    for step in range(1, num_steps + 1):
        logits = model(input_ids, mask=mask)
        loss = F.cross_entropy(logits.reshape(-1, vocab_size), target_ids.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.append((step, loss.item()))

        if step in checkpoint_steps:
            capture(step)

    return {
        "model": model,
        "vocab": vocab,
        "inv_vocab": inv_vocab,
        "input_ids": input_ids,
        "target_ids": target_ids,
        "mask": mask,
        "losses": losses,
        "captured": captured,
    }


def plot_loss_curve(losses: List[Tuple[int, float]]):
    fig, ax = plt.subplots(figsize=(6, 4))
    xs = [x for x, _ in losses]
    ys = [y for _, y in losses]
    ax.plot(xs, ys)
    ax.set_title("Training loss")
    ax.set_xlabel("Step")
    ax.set_ylabel("Cross-entropy loss")
    fig.tight_layout()
    return fig


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="Attention Explorer", layout="wide")
st.title("Attention Explorer")
st.caption("Visualise single-head attention, multi-head attention, and how attention patterns evolve during training.")

with st.sidebar:
    st.header("Controls")
    sentence = st.text_area(
        "Token sequence",
        value="the cat sat on the mat",
        help="Use a short whitespace-separated sequence so the heatmaps stay readable.",
    )
    mode = st.radio(
        "Mode",
        [
            "Single-head attention",
            "Multi-head attention",
            "Checkpointed training visualisation",
        ],
        index=1,
    )
    seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)

    if mode == "Single-head attention":
        embedding_type = st.radio("Embedding source", ["Random embeddings", "Learned token embeddings"], index=1)
        use_causal_mask = st.checkbox("Apply causal mask", value=True)
        d_model = st.selectbox("Embedding dimension", [4, 8, 16, 32], index=1)
        num_heads = 1
    elif mode == "Multi-head attention":
        embedding_type = st.radio("Embedding source", ["Random embeddings", "Learned token embeddings"], index=1)
        use_causal_mask = st.checkbox("Apply causal mask", value=True)
        d_model = st.selectbox("Model dimension", [8, 16, 32, 64], index=1)
        valid_heads = [h for h in [1, 2, 4, 8] if h <= d_model and d_model % h == 0]
        num_heads = st.selectbox("Number of heads", valid_heads, index=min(1, len(valid_heads) - 1))
    else:
        d_model = st.selectbox("Model dimension", [8, 16, 32, 64], index=1)
        valid_heads = [h for h in [1, 2, 4, 8] if h <= d_model and d_model % h == 0]
        num_heads = st.selectbox("Number of heads", valid_heads, index=min(1, len(valid_heads) - 1))
        d_ff = st.selectbox("Feedforward dimension", [16, 32, 64, 128], index=2)
        num_layers = st.selectbox("Number of transformer blocks", [1, 2, 3], index=1)
        num_steps = st.slider("Training steps", min_value=20, max_value=500, value=200, step=20)
        learning_rate = st.select_slider("Learning rate", options=[0.001, 0.003, 0.01, 0.03, 0.1], value=0.01)
        selected_layer = st.number_input("Layer to inspect", min_value=0, max_value=max(0, num_layers - 1), value=0, step=1)
        selected_head = st.number_input("Head to inspect", min_value=0, max_value=max(0, num_heads - 1), value=0, step=1)

set_seed(int(seed))
tokens = parse_tokens(sentence)

if len(tokens) < 2:
    st.warning("Please enter at least two tokens.")
    st.stop()

seq_len = len(tokens)

left, right = st.columns([1.2, 1])

with left:
    st.subheader("Setup summary")
    if mode == "Checkpointed training visualisation":
        st.markdown(
            f"""
- **Tokens:** `{tokens}`
- **Sequence length:** `{seq_len}`
- **Model dimension:** `{d_model}`
- **Heads:** `{num_heads}`
- **FFN dimension:** `{d_ff}`
- **Transformer blocks:** `{num_layers}`
- **Training steps:** `{num_steps}`
- **Learning rate:** `{learning_rate}`
            """
        )
    else:
        st.markdown(
            f"""
- **Tokens:** `{tokens}`
- **Sequence length:** `{seq_len}`
- **Model dimension:** `{d_model}`
- **Heads:** `{num_heads}`
- **Causal mask:** `{use_causal_mask}`
- **Embedding source:** `{embedding_type}`
            """
        )

    st.subheader("What you are seeing")
    if mode == "Checkpointed training visualisation":
        st.markdown(
            """
This mode trains a tiny Transformer on the toy sequence and captures attention maps at several checkpoints.

- **Rows** = query token, the current token asking where to look.
- **Columns** = key token, the token being attended to.
- Each row sums to about **1.0**.
- As training progresses, attention often becomes less diffuse and more structured.
            """
        )
    else:
        st.markdown(
            """
Each heatmap row shows the attention distribution for one token.

- **Rows** = query token, the current token asking where to look.
- **Columns** = key token, the token being attended to.
- Values in each row sum to about **1.0** because of softmax.
- With a **causal mask**, future tokens are forced to receive zero attention.
            """
        )

with right:
    if mode != "Checkpointed training visualisation":
        if embedding_type == "Random embeddings":
            x = build_random_embeddings(seq_len, d_model)
        else:
            x = build_learned_embeddings(tokens, d_model)

        mask = build_causal_mask(seq_len) if use_causal_mask else None

        st.subheader("Input tensor shape")
        st.code(f"x.shape = {tuple(x.shape)}")
        if mask is not None:
            st.subheader("Mask tensor shape")
            st.code(f"mask.shape = {tuple(mask.shape)}")
    else:
        st.subheader("Training objective")
        st.markdown(
            """
The model is trained on **next-token prediction**.

For a token sequence like:
`the cat sat on the mat`

the training pairs are:
- input: `the cat sat on the`
- target: `cat sat on the mat`
            """
        )

if mode == "Single-head attention":
    attention = ScaledDotProductAttention()
    output, weights = attention(x, x, x, mask=mask)
    weights_np = weights[0].detach().numpy()

    st.subheader("Attention heatmap")
    fig = plot_attention_heatmap(weights_np, tokens, "Single-head attention weights")
    st.pyplot(fig)
    plt.close(fig)

    row_idx = st.slider("Inspect token row", min_value=0, max_value=seq_len - 1, value=min(2, seq_len - 1))
    st.markdown(f"**Token being inspected:** `{tokens[row_idx]}`")
    st.write(highlight_attention(weights_np, tokens, row_idx))

    st.subheader("Sanity checks")
    row_sums = weights.sum(dim=-1)
    st.code(f"weights.shape = {tuple(weights.shape)}\nrow_sums.shape = {tuple(row_sums.shape)}")
    st.write(row_sums)

elif mode == "Multi-head attention":
    mha = MultiHeadAttention(d_model=d_model, num_heads=num_heads)
    output, weights = mha(x, mask=mask)
    weights_np = weights[0].detach().numpy()

    st.subheader("Per-head attention heatmaps")
    cols = st.columns(min(num_heads, 4))
    for h in range(num_heads):
        col = cols[h % len(cols)]
        with col:
            fig = plot_attention_heatmap(weights_np[h], tokens, f"Head {h}")
            st.pyplot(fig)
            plt.close(fig)

    selected_head_vis = st.slider("Inspect head", min_value=0, max_value=num_heads - 1, value=0)
    selected_row = st.slider("Inspect token row", min_value=0, max_value=seq_len - 1, value=min(2, seq_len - 1))
    st.markdown(f"**Head:** `{selected_head_vis}` | **Token:** `{tokens[selected_row]}`")
    st.write(highlight_attention(weights_np[selected_head_vis], tokens, selected_row))

    st.subheader("Sanity checks")
    row_sums = weights.sum(dim=-1)
    st.code(f"weights.shape = {tuple(weights.shape)}\nrow_sums.shape = {tuple(row_sums.shape)}")
    st.write(row_sums)

    if use_causal_mask:
        st.subheader("Why the upper triangle goes dark")
        st.markdown(
            """
With a causal mask, token *i* can only attend to tokens up to position *i*.
That is why attention above the main diagonal becomes zero after softmax.
            """
        )

else:
    result = train_and_capture_checkpoints(
        tokens=tokens,
        d_model=d_model,
        num_heads=num_heads,
        d_ff=d_ff,
        num_layers=num_layers,
        num_steps=num_steps,
        learning_rate=learning_rate,
    )

    inv_vocab = result["inv_vocab"]
    input_ids = result["input_ids"]
    target_ids = result["target_ids"]
    losses = result["losses"]
    captured = result["captured"]

    input_tokens = [inv_vocab[i.item()] for i in input_ids[0]]
    target_tokens = [inv_vocab[i.item()] for i in target_ids[0]]

    st.subheader("Training loss")
    loss_fig = plot_loss_curve(losses)
    st.pyplot(loss_fig)
    plt.close(loss_fig)

    st.subheader("Input / target sequence")
    st.markdown(f"**Input:** `{input_tokens}`")
    st.markdown(f"**Target:** `{target_tokens}`")

    checkpoint_labels = [f"step {c['step']} | loss {c['loss']:.3f}" for c in captured]
    selected_indices = st.multiselect(
        "Choose checkpoints to compare",
        options=list(range(len(captured))),
        default=[0, len(captured) - 1] if len(captured) > 1 else [0],
        format_func=lambda i: checkpoint_labels[i],
    )

    if not selected_indices:
        st.info("Select at least one checkpoint to compare.")
    else:
        st.subheader(f"Layer {selected_layer}, Head {selected_head} comparison")
        compare_cols = st.columns(len(selected_indices))

        for col, idx in zip(compare_cols, selected_indices):
            snap = captured[idx]
            weights = snap["attentions"][selected_layer][selected_head]
            with col:
                fig = plot_attention_heatmap(weights, input_tokens, f"step {snap['step']}\nloss {snap['loss']:.3f}")
                st.pyplot(fig)
                plt.close(fig)

        selected_snapshot_idx = st.selectbox(
            "Inspect one checkpoint in more detail",
            options=list(range(len(captured))),
            format_func=lambda i: checkpoint_labels[i],
            index=len(captured) - 1,
        )
        snapshot = captured[selected_snapshot_idx]
        preds = [inv_vocab[i] for i in snapshot["preds"]]
        st.markdown(f"**Predictions at step {snapshot['step']}:** `{preds}`")

        selected_row = st.slider(
            "Inspect token row",
            min_value=0,
            max_value=len(input_tokens) - 1,
            value=min(2, len(input_tokens) - 1),
        )
        st.markdown(f"**Token being inspected:** `{input_tokens[selected_row]}`")
        st.write(highlight_attention(snapshot["attentions"][selected_layer][selected_head], input_tokens, selected_row))

        st.subheader("What to look for")
        st.markdown(
            """
- Early checkpoints often show diffuse or noisy attention.
- Later checkpoints may become more selective.
- Repeated tokens can behave differently depending on context.
- With causal masking, attention stays in the lower triangle throughout training.
            """
        )

st.subheader("Suggested GitHub framing")
st.markdown(
    """
A strong way to position this project is:

**From-Scratch Transformer Attention Lab: visualising single-head attention, multi-head attention, and checkpointed attention dynamics during training**
    """
)
