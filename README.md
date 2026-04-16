# Transformer From Scratch

This project builds a compact decoder-only Transformer from scratch and uses small, interpretable experiments to explain:

- token embeddings and positional encoding
- masked self-attention and multi-head attention
- transformer block flow (attention -> residual/norm -> feedforward -> residual/norm)
- next-token training on both a toy single sequence and a small corpus

## Repository Structure

- `model/`
  - core model components (`attention.py`, `multi_head_attention.py`, `feedforward.py`, `transformer_block.py`, `tiny_transformer.py`)
  - corpus trainer (`train.py`)
- `data/`
  - synthetic corpus (`synthetic_corpus_100.txt`, one sentence per line)
- `utils/`
  - helper visualization scripts
  - `positional_encoding_viz.py` for positional encoding plots
  - `generate_training_artifacts.py` for training/validation artifacts
- `diagrams/`
  - architecture markdown diagrams
  - generated PNG artifacts (positional encoding and training visuals)

## End-to-End Training Process (100-Sentence Corpus)

The corpus trainer lives in `model/train.py`. At a high level:

1. Load corpus lines and tokenize each sentence.
2. Split sentences into train/validation subsets.
3. Build vocabulary from train set and add special tokens:
   - `<pad>` for padding
   - `<eos>` for sequence termination
   - `<unk>` for out-of-vocabulary tokens
4. Encode each sentence to IDs and append `<eos>`.
5. Build next-token pairs per sentence:
   - `input_ids = ids[:-1]`
   - `target_ids = ids[1:]`
6. Create mini-batches by sampling train examples.
7. Pad sequences in each batch to equal length.
8. Build attention mask:
   - causal lower-triangular mask
   - key-side padding mask (prevents attending to `<pad>`)
9. Forward pass through `TinyTransformer` to get logits.
10. Compute cross-entropy loss with `ignore_index=<pad>`.
11. Backpropagate and update parameters with Adam.
12. Periodically evaluate validation loss.
13. Decode one validation sample at the end for qualitative inspection.

## Visualizing Learning

Generated artifacts in `diagrams/training_artifacts/` and how to read them:

- **`learning_curve.png`**  
  Train loss and validation loss over steps.  
  Use this to check whether optimization is working and whether validation tracks training.

  ![Learning Curve](./diagrams/training_artifacts/learning_curve.png)

- **`attention_before_after.png`**  
  One single image covering **all layers and all heads**.  
  For each layer/head pair, it shows **before** and **after** attention side by side.  
  Use this to see where attention becomes more structured/selective.

  ![Attention Before After](./diagrams/training_artifacts/attention_before_after.png)

- **`validation_confidence.png`**  
  Position-wise probability of predicted token vs target token on one validation sample.

  ![Validation Confidence](./diagrams/training_artifacts/validation_confidence.png)

- **`validation_topk_positions.png`**  
  Top-k distributions for **all positions** in the validation sample (arranged in a 2-column grid).

  ![Validation Top-k](./diagrams/training_artifacts/validation_topk_positions.png)

- **`validation_predictions.csv`**  
  Token-by-token table with `input_token`, `target_next`, `pred_next`, `correct`, and confidences.

- **`loss_history.csv`**  
  Numeric train/validation losses per logged step.

- **`run_summary.txt`**  
  Run metadata and key summary numbers (dataset sizes, vocab size, sample loss, sample token accuracy).

## Generate Artifacts (Reproducible)

Run from repository root (`transformer-from-scratch/`).

### 1) Train and save learning artifacts

```bash
python utils/generate_training_artifacts.py \
  --corpus-path data/synthetic_corpus_100.txt \
  --output-dir diagrams/training_artifacts \
  --num-steps 300 \
  --batch-size 16 \
  --learning-rate 0.01 \
  --val-ratio 0.2 \
  --eval-every 10 \
  --seed 42 \
  --d-model 16 \
  --num-heads 2 \
  --d-ff 64 \
  --num-layers 2 \
  --sample-index 0 \
  --top-k 5
```

Outputs in `diagrams/training_artifacts/`:

- `learning_curve.png`
- `attention_before_after.png`
- `validation_confidence.png`
- `validation_topk_positions.png`
- `validation_predictions.csv`
- `loss_history.csv`
- `run_summary.txt`

### 2) Generate positional encoding visuals

```bash
python utils/positional_encoding_viz.py --d-model 32 --seq-len 100
```

Outputs in `diagrams/`:

- `positional_encoding_waves.png`
- `positional_encoding_heatmap.png`

## Troubleshooting

- `NaN` losses:
  - ensure attention mask does not fully mask query rows.
  - current implementation masks padding keys, not query rows.
- `ModuleNotFoundError: torch`:
  - install PyTorch in your environment first.
- shape/assertion errors for heads:
  - ensure `d_model % num_heads == 0`.
