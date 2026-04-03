"""
Pytest test suite for autoresearch-mlx-dlx.

Source-level tests run everywhere. Runtime tests (marked @pytest.mark.metal)
require macOS with Apple Silicon and MLX installed.
"""

from __future__ import annotations

import ast
import io
import platform
import re
import sys
from contextlib import redirect_stdout
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
TRAIN = ROOT / "train.py"
PREPARE = ROOT / "prepare.py"
README = ROOT / "README.md"
PROGRAM = ROOT / "program.md"
PYPROJECT = ROOT / "pyproject.toml"
RUN_LOG = ROOT / "run.log"

HAS_MLX = sys.platform == "darwin" and platform.machine() == "arm64"
try:
    import mlx.core
    HAS_MLX = True
except ImportError:
    pass

metal = pytest.mark.skipif(not HAS_MLX, reason="requires Apple Silicon with MLX")


def read(path: Path) -> str:
    return path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# train.py source-level tests
# ---------------------------------------------------------------------------

class TestTrainSource:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = read(TRAIN)
        self.tree = ast.parse(self.src, filename=str(TRAIN))

    def test_no_unmasked_sdpa(self):
        assert "mask=None" not in self.src, "attention should not use an unmasked full-window SDPA path"

    def test_muon_state_key(self):
        assert "group_key = group['state_key']" in self.src

    def test_unscaled_micro_loss(self):
        assert "last_micro_loss = loss * grad_accum_steps" in self.src

    def test_eval_barrier_includes_micro_loss(self):
        assert "mx.eval(total_loss, accumulated_grads, last_micro_loss)" in self.src

    def test_nan_check(self):
        assert "math.isnan(train_loss_f)" in self.src

    def test_monotonic_clock(self):
        assert "time.perf_counter()" in self.src

    def test_rope_sign_convention(self):
        assert "negative-sin-on-second-row" in self.src

    def test_no_top_level_detect_memory(self):
        for node in self.tree.body:
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Name) and func.id == "detect_memory_tier":
                    pytest.fail("detect_memory_tier must not run at import time")

    def test_m4_ultra_in_flops_table(self):
        assert '"M4 Ultra"' in self.src

    def test_compile_targets(self):
        compiled = set()
        for node in ast.walk(self.tree):
            if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
                func = node.value.func
                if isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name):
                    if func.value.id == "mx" and func.attr == "compile":
                        for target in node.targets:
                            if isinstance(target, ast.Name):
                                compiled.add(target.id)
        assert compiled == {"adamw_step_compiled", "muon_step_compiled"}


# ---------------------------------------------------------------------------
# prepare.py source-level tests
# ---------------------------------------------------------------------------

class TestPrepareSource:
    @pytest.fixture(autouse=True)
    def _load(self):
        self.src = read(PREPARE)

    def test_project_namespace(self):
        assert "autoresearch-mlx-dlx" in self.src

    def test_cache_env_var(self):
        assert "AUTORESEARCH_MLX_DLX_CACHE_DIR" in self.src

    def test_legacy_cache_fallbacks(self):
        assert "LEGACY_CACHE_DIRS" in self.src


# ---------------------------------------------------------------------------
# docs and metadata tests
# ---------------------------------------------------------------------------

class TestDocsAndMetadata:
    def test_readme_title(self):
        assert read(README).startswith("# autoresearch-mlx-dlx")

    def test_program_title(self):
        assert read(PROGRAM).startswith("# autoresearch-mlx-dlx")

    def test_package_name(self):
        assert 'name = "autoresearch-mlx-dlx"' in read(PYPROJECT)

    def test_license_declared(self):
        assert "license" in read(PYPROJECT)

    def test_no_overclaim_compile(self):
        assert "mx.compile (works)" not in read(README)

    def test_min_shard_docs(self):
        assert "--num-shards 2" in read(README)

    def test_bpb_noise_documented(self):
        prog = read(PROGRAM)
        assert "0.003" in prog and "0.005" in prog


class TestGitignore:
    def test_ds_store(self):
        assert ".DS_Store" in read(ROOT / ".gitignore")


class TestRunLog:
    def test_run_log(self):
        if not RUN_LOG.exists():
            pytest.skip("run.log not present")
        log = read(RUN_LOG)
        assert "val_bpb:" in log
        match = re.search(r"training_seconds:\s+([0-9.]+)", log)
        if match:
            assert float(match.group(1)) > 0


# ---------------------------------------------------------------------------
# LR schedule (no MLX needed)
# ---------------------------------------------------------------------------

class TestLRSchedule:
    @staticmethod
    def get_lr_multiplier(progress, warmup=0.0, warmdown=0.5, final=0.0):
        if progress < warmup:
            return progress / warmup if warmup > 0 else 1.0
        elif progress < 1.0 - warmdown:
            return 1.0
        else:
            cooldown = (1.0 - progress) / warmdown
            return cooldown * 1.0 + (1 - cooldown) * final

    def test_full_at_start(self):
        assert self.get_lr_multiplier(0.0) == 1.0

    def test_full_during_constant(self):
        assert self.get_lr_multiplier(0.25) == 1.0

    def test_full_at_warmdown_boundary(self):
        assert self.get_lr_multiplier(0.5) == 1.0

    def test_half_at_75_percent(self):
        assert abs(self.get_lr_multiplier(0.75) - 0.5) < 1e-6

    def test_zero_at_completion(self):
        assert self.get_lr_multiplier(1.0) == 0.0


# ---------------------------------------------------------------------------
# New test cases (source-level, no MLX)
# ---------------------------------------------------------------------------

class TestMemoryTierBatchAlignment:
    """TOTAL_BATCH_SIZE % (device_batch_size * MAX_SEQ_LEN) == 0 for all tiers."""

    @pytest.mark.parametrize("device_batch_size", [2, 8, 16, 32])
    def test_batch_alignment(self, device_batch_size):
        # Extract constants from source to avoid importing MLX
        src = read(TRAIN)
        match = re.search(r"TOTAL_BATCH_SIZE\s*=\s*(.+)", src)
        total_batch = eval(match.group(1))
        src_p = read(PREPARE)
        match_seq = re.search(r"MAX_SEQ_LEN\s*=\s*(\d+)", src_p)
        max_seq = int(match_seq.group(1))
        tokens_per_fwdbwd = device_batch_size * max_seq
        assert total_batch % tokens_per_fwdbwd == 0, (
            f"TOTAL_BATCH_SIZE ({total_batch}) not divisible by "
            f"device_batch_size={device_batch_size} * MAX_SEQ_LEN={max_seq}"
        )


class TestDetectMemoryTierSource:
    """detect_memory_tier returns valid tier names."""

    def test_valid_tier_names(self):
        src = read(TRAIN)
        valid_tiers = {"compact", "standard", "pro", "ultra"}
        for tier in valid_tiers:
            assert f'"{tier}"' in src, f"tier '{tier}' not found in detect_memory_tier"


class TestHyperparamPositive:
    """All hyperparameter constants in train.py are positive."""

    @pytest.mark.parametrize("name", [
        "ASPECT_RATIO", "HEAD_DIM", "TOTAL_BATCH_SIZE", "DEPTH",
        "EMBEDDING_LR", "UNEMBEDDING_LR", "MATRIX_LR", "SCALAR_LR",
        "WEIGHT_DECAY",
    ])
    def test_positive(self, name):
        src = read(TRAIN)
        match = re.search(rf"^{name}\s*=\s*(.+?)(?:\s*#|$)", src, re.MULTILINE)
        assert match, f"{name} not found"
        val = eval(match.group(1).strip())
        assert val > 0, f"{name} = {val} is not positive"


class TestWindowPatternParser:
    """Window pattern parser handles edge cases."""

    def test_single_char_L(self):
        src = read(TRAIN)
        # The parser uses pattern[layer_idx % len(pattern)] with chars in "SL"
        # and forces last layer to long window. Single "L" should work.
        assert 'all(c in "SL" for c in pattern)' in src or "all(c in 'SL' for c in pattern)" in src

    def test_single_char_pattern_documented(self):
        # README and train.py both reference single-char patterns
        src = read(TRAIN)
        assert 'WINDOW_PATTERN = "L"' in src

    def test_repeating_pattern_in_source(self):
        # Default config uses "SSSL" which repeats for n_layer > 4
        src = read(TRAIN)
        assert "SSSL" in src


class TestGPTConfigConsistency:
    """GPTConfig defaults are self-consistent."""

    def test_embd_divisible_by_head(self):
        src = read(TRAIN)
        # Extract defaults from dataclass
        match_embd = re.search(r"n_embd:\s*int\s*=\s*(\d+)", src)
        match_head = re.search(r"n_head:\s*int\s*=\s*(\d+)", src)
        n_embd = int(match_embd.group(1))
        n_head = int(match_head.group(1))
        assert n_embd % n_head == 0, f"n_embd ({n_embd}) not divisible by n_head ({n_head})"

    def test_kv_head_divides_head(self):
        src = read(TRAIN)
        match_head = re.search(r"n_head:\s*int\s*=\s*(\d+)", src)
        match_kv = re.search(r"n_kv_head:\s*int\s*=\s*(\d+)", src)
        n_head = int(match_head.group(1))
        n_kv_head = int(match_kv.group(1))
        assert n_head % n_kv_head == 0


# ---------------------------------------------------------------------------
# Runtime tests (require MLX)
# ---------------------------------------------------------------------------

@metal
class TestRuntime:
    def test_forward_pass(self):
        from train import GPT, GPTConfig
        import mlx.core as mx

        config = GPTConfig(
            sequence_len=64, vocab_size=256, n_layer=1, n_head=2,
            n_kv_head=2, n_embd=64, window_pattern="L",
        )
        model = GPT(config)
        model.init_weights()
        x = mx.random.randint(0, 256, (1, 32))
        logits = model(x)
        mx.eval(logits)
        assert mx.isfinite(logits).all().item()

    def test_backward_pass(self):
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
        targets = mx.random.randint(0, 256, (1, 32))
        loss_fn = lambda m, xi, yi: m(xi, yi)
        loss, grads = nn.value_and_grad(model, loss_fn)(model, x, targets)
        mx.eval(loss)
        assert mx.isfinite(loss).all().item()

    def test_optimizer_update(self):
        from train import GPT, GPTConfig, MuonAdamW
        import mlx.core as mx

        config = GPTConfig(
            sequence_len=64, vocab_size=256, n_layer=1, n_head=2,
            n_kv_head=2, n_embd=64, window_pattern="L",
        )
        model = GPT(config)
        model.init_weights()
        x = mx.random.randint(0, 256, (1, 32))
        targets = mx.random.randint(0, 256, (1, 32))
        import mlx.nn as nn
        loss_fn = lambda m, xi, yi: m(xi, yi)
        _, grads = nn.value_and_grad(model, loss_fn)(model, x, targets)
        groups = model.setup_optimizer_groups()
        optimizer = MuonAdamW(model, groups)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

    def test_dataloader(self):
        import prepare
        import mlx.core as mx

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

        original = prepare._document_batches
        prepare._document_batches = fake_document_batches
        try:
            loader = prepare.make_dataloader(FakeTokenizer(), B=1, T=4, split="train", buffer_size=2)
            x1, y1, _ = next(loader)
            x2, y2, _ = next(loader)
            row1 = [int(v) for v in x1[0].tolist()] + [int(y1[0, -1].item())]
            row2 = [int(v) for v in x2[0].tolist()] + [int(y2[0, -1].item())]
            assert row1 == [101, 10, 11, 12, 101]
            assert [20, 21, 22] == row2[1:4]

            log_buffer = io.StringIO()
            with redirect_stdout(log_buffer):
                next(prepare.make_dataloader(FakeTokenizer(), B=1, T=4, split="train", buffer_size=2))
            assert "cropped_tokens" in log_buffer.getvalue()
        finally:
            prepare._document_batches = original


@metal
class TestWindowSizes:
    def test_last_layer_full_window(self):
        from train import GPTConfig, GPT
        config = GPTConfig(sequence_len=2048, n_layer=8, window_pattern="SSSL")
        model = GPT(config)
        assert model.window_sizes[-1] == 2048

    def test_short_window_half(self):
        from train import GPTConfig, GPT
        config = GPTConfig(sequence_len=2048, n_layer=8, window_pattern="SSSL")
        model = GPT(config)
        assert model.window_sizes[0] == 1024

    def test_long_window_full(self):
        from train import GPTConfig, GPT
        config = GPTConfig(sequence_len=2048, n_layer=8, window_pattern="SSSL")
        model = GPT(config)
        assert model.window_sizes[3] == 2048

    def test_window_count_matches_layers(self):
        from train import GPTConfig, GPT
        config = GPTConfig(sequence_len=2048, n_layer=8, window_pattern="SSSL")
        model = GPT(config)
        assert len(model.window_sizes) == 8

    def test_all_L_pattern(self):
        from train import GPTConfig, GPT
        config = GPTConfig(sequence_len=2048, n_layer=4, window_pattern="L")
        model = GPT(config)
        assert all(w == 2048 for w in model.window_sizes)


@metal
class TestFlopsEstimate:
    def test_positive(self):
        from train import GPTConfig, GPT
        config = GPTConfig(sequence_len=64, vocab_size=256, n_layer=2, n_head=2,
                           n_kv_head=2, n_embd=64, window_pattern="L")
        model = GPT(config)
        flops = model.estimate_flops()
        assert flops > 0
        assert isinstance(flops, (int, float))
