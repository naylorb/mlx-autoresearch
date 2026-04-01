"""
Lightweight source-level smoke tests for autoresearch-mlx-dlx.

This avoids importing MLX so it can run in environments where Metal device
initialization is unavailable, while still catching the regressions found in
review.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path


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
    require("math.isnan(train_loss_f)" in src, "training fast-fail checks NaN loss")
    require("time.perf_counter()" in src, "training timers use a monotonic clock")

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
    require("mx.compile (works)" not in readme, "README no longer overclaims whole-model mx.compile support")


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


def main() -> None:
    test_train_source()
    test_prepare_source()
    test_docs_and_metadata()
    test_run_log()
    print("OK: smoke tests passed")


if __name__ == "__main__":
    main()
