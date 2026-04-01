"""
Source-driven run simulator for autoresearch-mlx-dlx.

This does not import MLX. It reads the current training/data constants, derives
the per-tier batch semantics, and estimates what a compact-tier run looks like
using the existing run.log timings when available.
"""

from __future__ import annotations

import ast
import re
import statistics
from pathlib import Path


ROOT = Path(__file__).resolve().parent
TRAIN = ROOT / "train.py"
PREPARE = ROOT / "prepare.py"
RUN_LOG = ROOT / "run.log"


def parse_constants(path: Path, names: set[str]) -> dict[str, object]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    values: dict[str, object] = {}
    for node in tree.body:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and target.id in names:
                    values[target.id] = eval_literal_expr(node.value)
    missing = names - values.keys()
    if missing:
        raise SystemExit(f"Missing constants in {path.name}: {sorted(missing)}")
    return values


def eval_literal_expr(node: ast.AST) -> object:
    if isinstance(node, ast.Constant):
        return node.value
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
        value = eval_literal_expr(node.operand)
        return value if isinstance(node.op, ast.UAdd) else -value
    if isinstance(node, ast.BinOp) and isinstance(
        node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Pow, ast.Mod)
    ):
        left = eval_literal_expr(node.left)
        right = eval_literal_expr(node.right)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.FloorDiv):
            return left // right
        if isinstance(node.op, ast.Pow):
            return left ** right
        return left % right
    raise ValueError(f"Unsupported literal expression: {ast.dump(node)}")


def parse_dt_ms_from_log(path: Path) -> list[float]:
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    return [float(x) for x in re.findall(r"dt:\s+([0-9]+)ms", text)]


def detect_tiers(total_batch_size: int, max_seq_len: int) -> list[dict[str, object]]:
    tiers = [
        ("compact", 8, 2),
        ("standard", 16, 8),
        ("pro", 36, 16),
        ("ultra", 96, 32),
    ]
    out = []
    for name, example_gb, device_batch_size in tiers:
        tokens_per_fwdbwd = device_batch_size * max_seq_len
        grad_accum_steps = total_batch_size // tokens_per_fwdbwd
        eval_steps = (40 * 524288) // tokens_per_fwdbwd
        out.append(
            {
                "tier": name,
                "example_ram_gb": example_gb,
                "device_batch_size": device_batch_size,
                "tokens_per_fwdbwd": tokens_per_fwdbwd,
                "grad_accum_steps": grad_accum_steps,
                "eval_steps": eval_steps,
            }
        )
    return out


def main() -> None:
    train_consts = parse_constants(
        TRAIN,
        {
            "ASPECT_RATIO",
            "HEAD_DIM",
            "TOTAL_BATCH_SIZE",
            "DEPTH",
            "WINDOW_PATTERN",
        },
    )
    prepare_consts = parse_constants(PREPARE, {"MAX_SEQ_LEN", "TIME_BUDGET", "EVAL_TOKENS"})

    depth = int(train_consts["DEPTH"])
    aspect_ratio = int(train_consts["ASPECT_RATIO"])
    head_dim = int(train_consts["HEAD_DIM"])
    max_seq_len = int(prepare_consts["MAX_SEQ_LEN"])
    total_batch_size = int(train_consts["TOTAL_BATCH_SIZE"])
    time_budget = int(prepare_consts["TIME_BUDGET"])
    eval_tokens = int(prepare_consts["EVAL_TOKENS"])

    base_dim = depth * aspect_ratio
    model_dim = ((base_dim + head_dim - 1) // head_dim) * head_dim
    num_heads = model_dim // head_dim
    tiers = detect_tiers(total_batch_size, max_seq_len)
    dt_ms = parse_dt_ms_from_log(RUN_LOG)

    print("Simulated run")
    print("-------------")
    print(f"sequence_len:      {max_seq_len}")
    print(f"time_budget_s:     {time_budget}")
    print(f"eval_tokens:       {eval_tokens:,}")
    print(f"depth:             {depth}")
    print(f"window_pattern:    {train_consts['WINDOW_PATTERN']}")
    print(f"model_dim:         {model_dim}")
    print(f"num_heads:         {num_heads}")
    print(f"head_dim:          {head_dim}")
    print(f"total_batch_size:  {total_batch_size:,} tokens/step")
    print()

    print("Per-tier semantics")
    for tier in tiers:
        print(
            f"- {tier['tier']:8s} device_batch={tier['device_batch_size']:2d} "
            f"grad_accum={tier['grad_accum_steps']:2d} "
            f"tokens/fwdbwd={tier['tokens_per_fwdbwd']:5d} "
            f"eval_steps={tier['eval_steps']:4d}"
        )
    print()

    compact = next(t for t in tiers if t["tier"] == "compact")
    print("Compact-tier step trace")
    print(
        f"1. Load tokenizer, build a {depth}-layer model at d_model={model_dim}, "
        f"then prefetch one training batch of shape [{compact['device_batch_size']}, {max_seq_len}]."
    )
    print(
        f"2. Each optimizer step performs {compact['grad_accum_steps']} micro-steps; "
        f"each micro-step processes {compact['tokens_per_fwdbwd']:,} tokens and calls mx.eval() immediately."
    )
    print(
        f"3. One optimizer step therefore still represents {total_batch_size:,} tokens, "
        "matching larger-memory tiers mathematically."
    )
    print(
        f"4. Final validation uses the same batch size, so compact tier must execute "
        f"{compact['eval_steps']:,} validation batches to cover the fixed {eval_tokens:,} eval tokens."
    )

    if dt_ms:
        filtered = [x for x in dt_ms if x < 60_000]
        if filtered:
            median_dt_ms = statistics.median(filtered)
            estimated_steps = max(0, int(time_budget * 1000 // median_dt_ms))
            estimated_eval_s = compact["eval_steps"] * (median_dt_ms / compact["grad_accum_steps"]) / 1000
            print()
            print("Observed-timing estimate from run.log")
            print(f"- median_step_dt_ms (filtered): {median_dt_ms:.0f}")
            print(f"- estimated_optimizer_steps_in_300s: {estimated_steps}")
            print(f"- rough_eval_seconds_if_val_step_cost~=micro_step_cost: {estimated_eval_s:.0f}")
            print("- note: eval estimate is crude, but it shows why compact-tier wall clock can greatly exceed the 300s training budget")


if __name__ == "__main__":
    main()
