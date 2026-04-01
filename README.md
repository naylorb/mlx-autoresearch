# autoresearch-mlx-dlx

Native Apple Silicon autonomous research — from 8GB MacBook to 192GB Mac Studio.

## Why this fork?

The original [autoresearch](https://github.com/karpathy/autoresearch) by Andrej Karpathy is a brilliant experiment: let an LLM autonomously research pretraining by editing a training script, running it, and iterating on results.

But it requires PyTorch with CUDA — or at minimum, 16GB of unified memory on macOS via MPS. An 8GB MacBook can't even install the dependencies without running low on disk (PyTorch is ~2GB), and `train.py` OOMs immediately on MPS.

**autoresearch-mlx-dlx** replaces PyTorch entirely with [MLX](https://github.com/ml-explore/mlx), Apple's native machine learning framework for Apple Silicon. The result:

- **100x smaller install** — ~20MB vs ~2GB for PyTorch
- **Runs on 8GB** — auto-detects memory and adjusts batch size
- **MLX-native compiled kernels** — the optimizer step functions use `mx.compile`
- **Unified memory** — no CPU/GPU buffer management, no `.to(device)` calls
- **Real memory reporting** — `mx.get_peak_memory()` works (PyTorch MPS returned 0)

Same model architecture. Same optimizer. Same evaluation metric. Just native.

## Quick start

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (~5 seconds)
git clone https://github.com/naylorb/autoresearch-mlx-dlx.git
cd autoresearch-mlx-dlx
uv sync                                              # ~30 seconds (no 2GB torch)
uv run prepare.py                                    # download data + train tokenizer
uv run train.py                                      # runs on any Apple Silicon Mac
```

## Comparison

| | karpathy/autoresearch | miolini/autoresearch-macos | **autoresearch-mlx-dlx** |
|---|---|---|---|
| Framework | PyTorch (CUDA) | PyTorch (MPS) | **MLX (native)** |
| Min RAM | ~16GB VRAM | ~16GB unified | **8GB unified** |
| Install size | ~2GB | ~2GB | **~20MB** |
| Whole-model compile | yes (CUDA) | disabled (MPS) | **not currently safe; step kernels only** |
| Memory reporting | CUDA only | returns 0 | **actual peak MB** |
| CPU-GPU copies | explicit | explicit | **none (unified)** |

## Memory tiers

autoresearch-mlx-dlx auto-detects your system RAM and sets sensible defaults:

| Tier | RAM Range | Batch Size | Grad Accum | Example Machines |
|------|-----------|------------|------------|------------------|
| Compact | < 12GB | 2 | 16 | MacBook Neo 8GB, MacBook Air base |
| Standard | 12-28GB | 8 | 4 | MacBook Pro 16GB, Mac Mini 24GB |
| Pro | 28-80GB | 16 | 2 | MacBook Pro Max 36/48GB, Mac Studio 64GB |
| Ultra | > 80GB | 32 | 1 | Mac Studio Ultra 96/192GB, Mac Pro |

Only batch size changes across tiers. The model architecture stays constant, so results are comparable. The detected tier is shown at startup and in the training summary.

## How it works

Three files, same as the original:

- **`prepare.py`** — Downloads data, trains tokenizer. Run once. Read-only during experiments.
- **`train.py`** — Model, optimizer, training loop. The file the agent edits.
- **`program.md`** — Instructions for the AI agent.

The agent edits `train.py`, runs it for 5 minutes, checks `val_bpb` (bits per byte — lower is better), and keeps or discards the change. Repeat indefinitely.

The 5-minute budget applies to training time, not full wall clock. On compact-tier machines, final validation can still add substantial extra runtime because the evaluation token budget is fixed while batch size is smaller.

## Running the agent

See `program.md` for the full agent protocol. The short version:

1. Set up a fresh branch: `git checkout -b autoresearch/<tag>`
2. Run the baseline: `uv run train.py > run.log 2>&1`
3. Let the agent iterate on `train.py`, running experiments and tracking results in `results.tsv`

## Design choices

- **MLX over PyTorch MPS**: MPS is a compatibility layer that patches PyTorch to work on Apple Silicon. MLX is native — designed from the ground up for Apple's unified memory architecture. No FlashAttention workarounds, no device casting guards, and no need for explicit CPU/GPU buffer choreography.

- **Single-file philosophy**: Karpathy's constraint (one editable file, one read-only file, one instruction file) is preserved. The agent only touches `train.py`.

- **Memory tier design**: Only batch size changes between tiers, not model architecture. This means experiments are comparable across machines (same model, different accumulation steps = mathematically equivalent training).

- **Cache isolation**: this fork writes to `~/.cache/autoresearch-mlx-dlx/` by default so it does not trample tokenizer or shard state from other autoresearch forks. Set `AUTORESEARCH_MLX_DLX_CACHE_DIR` to override, and legacy `~/.cache/autoresearch-mlx/` and `~/.cache/autoresearch/` artifacts are still readable.

- **`mx.eval()` per micro-step**: MLX uses lazy evaluation. Without explicit eval boundaries inside the gradient accumulation loop, the framework builds the entire computation graph before executing. For 16 micro-steps on an 8GB machine, that's instant OOM. The eval calls are load-bearing.

## Security

- **6 direct dependencies**, all exact-pinned with `uv.lock` SHA-256 hash verification
- **`train.py` is fully offline** — zero network calls during training
- **No telemetry, no analytics, no phone-home**
- See `pyproject.toml` for the full dependency list and rationale

## Credits

- [Andrej Karpathy](https://github.com/karpathy) — original autoresearch concept and implementation
- [miolini/autoresearch-macos](https://github.com/miolini/autoresearch-macos) — prior art on macOS adaptation
- [Apple MLX team](https://github.com/ml-explore/mlx) — the framework that makes this possible

## License

MIT
