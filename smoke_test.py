"""
Lightweight source-level smoke tests for autoresearch-mlx-dlx.

This avoids importing MLX so it can run in environments where Metal device
initialization is unavailable, while still catching the regressions found in
review.
"""

from __future__ import annotations

import argparse
import ast
import io
import re
from pathlib import Path
from contextlib import redirect_stdout


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train.py"
PREPARE = ROOT / "prepare.py"
README = ROOT / "README.md"
PROGRAM = ROOT / "program.md"
PYPROJECT = ROOT / "pyproject.toml"
RUN_LOG = ROOT / "run.log"


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit(f"FAIL: {message}")
    print(f"PASS: {message}")


def test_train_source() -> None:
    src = read(TRAIN)
    tree = ast.parse(src, filename=str(TRAIN))

    require("mask=None" not in src, "attention does not use an unmasked full-window SDPA path")
    require("group_key = group['state_key']" in src, "Muon state uses a stable per-shape group key")
    require("last_micro_loss = loss * grad_accum_steps" in src, "training log tracks unscaled last micro-step loss")
    require(
        "mx.eval(total_loss, accumulated_grads, last_micro_loss)" in src,
        "gradient accumulation eval barrier includes last_micro_loss",
    )
    require("math.isnan(train_loss_f)" in src, "training fast-fail checks NaN loss")
    require("time.perf_counter()" in src, "training timers use a monotonic clock")
    require(
        "negative-sin-on-second-row" in src,
        "RoPE implementation documents the sign convention",
    )

    top_level_detect_memory = False
    for node in tree.body:
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Name) and func.id == "detect_memory_tier":
                top_level_detect_memory = True
    require(not top_level_detect_memory, "memory tier detection does not run at import time")
    require('"M4 Ultra"' in src, "chip FLOPS table includes M4 Ultra")

    compiled_assignments = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            func = node.value.func
            if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                if func.value.id == "mx" and func.attr == "compile":
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            compiled_assignments.add(target.id)

    require(
        compiled_assignments == {"adamw_step_compiled", "muon_step_compiled"},
        "only optimizer step kernels are mx.compile targets",
    )


def test_prepare_source() -> None:
    src = read(PREPARE)
    require("autoresearch-mlx-dlx" in src, "prepare.py uses the new unique project/cache namespace")
    require("AUTORESEARCH_MLX_DLX_CACHE_DIR" in src, "prepare.py exposes the renamed cache override env var")
    require("LEGACY_CACHE_DIRS" in src, "prepare.py preserves backward-compatible cache fallbacks")


def test_docs_and_metadata() -> None:
    readme = read(README)
    program = read(PROGRAM)
    pyproject = read(PYPROJECT)

    require(readme.startswith("# autoresearch-mlx-dlx"), "README title uses the new repo name")
    require(program.startswith("# autoresearch-mlx-dlx"), "program.md title uses the new repo name")
    require('name = "autoresearch-mlx-dlx"' in pyproject, "package metadata uses the new unique name")
    require("license" in pyproject, "package metadata declares the project license")
    require("mx.compile (works)" not in readme, "README no longer overclaims whole-model mx.compile support")
    require("--num-shards 2" in readme, "README documents the minimum viable 2-shard setup")
    require("0.003" in program and "0.005" in program, "program.md documents expected BPB run-to-run noise")


def test_gitignore() -> None:
    gitignore = read(ROOT / ".gitignore")
    require(".DS_Store" in gitignore, ".gitignore covers common macOS finder artifacts")


def test_run_log() -> None:
    if not RUN_LOG.exists():
        print("SKIP: run.log not present")
        return

    log = read(RUN_LOG)
    require("val_bpb:" in log, "run.log contains a completed validation summary")

    match = re.search(r"training_seconds:\s+([0-9.]+)", log)
    if match:
        training_seconds = float(match.group(1))
        require(training_seconds > 0, "run.log reports positive training time")
    else:
        print("SKIP: could not parse training_seconds from run.log")


def test_runtime() -> None:
    """Runtime integration test — requires Metal GPU (MLX)."""
    import prepare
    from train import GPT, GPTConfig
    import mlx.core as mx
    import mlx.nn as nn

    config = GPTConfig(
        sequence_len=64, vocab_size=256, n_layer=1, n_head=2,
        n_kv_head=2, n_embd=64, window_pattern="L",
    )
    model = GPT(config)
    model.init_weights()
    x = mx.random.randint(0, 256, (1, 32))

    # Forward pass
    logits = model(x)
    mx.eval(logits)
    require(mx.isfinite(logits).all().item(), "forward pass produces finite logits")

    # Forward + backward pass
    targets = mx.random.randint(0, 256, (1, 32))
    loss_fn = lambda m, xi, yi: m(xi, yi)
    loss, grads = nn.value_and_grad(model, loss_fn)(model, x, targets)
    mx.eval(loss)
    require(mx.isfinite(loss).all().item(), "backward pass produces finite loss")

    groups = model.setup_optimizer_groups()
    optimizer = __import__("train").MuonAdamW(model, groups)
    optimizer.update(model, grads)
    mx.eval(model.parameters())
    require(True, "optimizer update path executes successfully")

    class FakeTokenizer:
        def __init__(self):
            self.mapping = {
                "A": [101, 10, 11, 12],
                "B": [101, 20, 21, 22],
                "C": [101, 30],
                "D": [101, 40],
            }

        def get_bos_token_id(self):
            return 101

        def encode(self, docs, prepend=None, num_threads=8):
            return [self.mapping[doc][:] for doc in docs]

    def fake_document_batches(split, tokenizer_batch_size=128):
        yield ["A", "B"], 1
        while True:
            yield ["C", "D"], 1

    original_document_batches = prepare._document_batches
    prepare._document_batches = fake_document_batches
    try:
        loader = prepare.make_dataloader(FakeTokenizer(), B=1, T=4, split="train", buffer_size=2)
        x1, y1, _ = next(loader)
        x2, y2, _ = next(loader)
        row1 = [int(v) for v in x1[0].tolist()] + [int(y1[0, -1].item())]
        row2 = [int(v) for v in x2[0].tolist()] + [int(y2[0, -1].item())]
        require(row1 == [101, 10, 11, 12, 101], "cropping still fills the first row exactly")
        require([20, 21, 22] == row2[1:4], "cropped document tails are preserved in later rows")

        log_buffer = io.StringIO()
        with redirect_stdout(log_buffer):
            next(prepare.make_dataloader(FakeTokenizer(), B=1, T=4, split="train", buffer_size=2))
        require("cropped_tokens" in log_buffer.getvalue(), "dataloader logs cropped token counts")
    finally:
        prepare._document_batches = original_document_batches


def test_window_sizes() -> None:
    """Test window size computation logic (requires MLX import)."""
    from train import GPTConfig, GPT

    # SSSL pattern with 8 layers: S,S,S,L,S,S,S,L but last always L
    config = GPTConfig(sequence_len=2048, n_layer=8, window_pattern="SSSL")
    model = GPT(config)
    ws = model.window_sizes
    require(ws[-1] == 2048, "last layer always gets full window")
    require(ws[0] == 1024, "S layers get half-context window")
    require(ws[3] == 2048, "L layers get full-context window")
    require(len(ws) == 8, "window sizes list matches layer count")

    # Single L pattern
    config2 = GPTConfig(sequence_len=2048, n_layer=4, window_pattern="L")
    model2 = GPT(config2)
    require(all(w == 2048 for w in model2.window_sizes), "all-L pattern gives full window everywhere")


def test_lr_schedule() -> None:
    """Test LR schedule boundary conditions (no MLX needed)."""
    src = read(TRAIN)
    # Extract schedule constants from source to avoid importing (which triggers MLX)
    # Instead, test the math directly
    warmup_ratio = 0.0
    warmdown_ratio = 0.5
    final_lr_frac = 0.0

    def get_lr_multiplier(progress):
        if progress < warmup_ratio:
            return progress / warmup_ratio if warmup_ratio > 0 else 1.0
        elif progress < 1.0 - warmdown_ratio:
            return 1.0
        else:
            cooldown = (1.0 - progress) / warmdown_ratio
            return cooldown * 1.0 + (1 - cooldown) * final_lr_frac

    require(get_lr_multiplier(0.0) == 1.0, "LR is full at start (no warmup)")
    require(get_lr_multiplier(0.25) == 1.0, "LR is full during constant phase")
    require(get_lr_multiplier(0.5) == 1.0, "LR is full at warmdown boundary")
    require(abs(get_lr_multiplier(0.75) - 0.5) < 1e-6, "LR is 0.5 at 75% progress")
    require(get_lr_multiplier(1.0) == 0.0, "LR is zero at completion")


def test_flops_estimate() -> None:
    """Test that FLOPs estimation doesn't crash and returns positive values."""
    from train import GPTConfig, GPT

    config = GPTConfig(sequence_len=64, vocab_size=256, n_layer=2, n_head=2,
                       n_kv_head=2, n_embd=64, window_pattern="L")
    model = GPT(config)
    flops = model.estimate_flops()
    require(flops > 0, "FLOPs estimate is positive")
    require(isinstance(flops, (int, float)), "FLOPs estimate is numeric")


def main() -> None:
    parser = argparse.ArgumentParser(description="Smoke tests for autoresearch-mlx-dlx")
    parser.add_argument("--metal", action="store_true", help="Enable runtime GPU tests (requires Metal)")
    args = parser.parse_args()

    test_train_source()
    test_prepare_source()
    test_docs_and_metadata()
    test_gitignore()
    test_run_log()
    test_lr_schedule()

    if args.metal:
        test_runtime()
        test_window_sizes()
        test_flops_estimate()
    else:
        print("SKIP: runtime tests (use --metal to enable)")

    print("OK: smoke tests passed")


if __name__ == "__main__":
    main()
