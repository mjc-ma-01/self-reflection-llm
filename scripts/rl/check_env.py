#!/usr/bin/env python3
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _check_import(name: str) -> tuple[bool, str | None]:
    try:
        module = importlib.import_module(name)
        return True, getattr(module, "__version__", None)
    except Exception as exc:
        return False, str(exc)


def _check_verl_gdpo() -> tuple[bool, str]:
    ok, detail = _check_import("verl.trainer.ppo.core_algos")
    if not ok:
        return False, f"verl import failed: {detail}"
    core_algos = importlib.import_module("verl.trainer.ppo.core_algos")
    estimator = getattr(core_algos, "AdvantageEstimator", None)
    if estimator is None:
        return False, "verl.trainer.ppo.core_algos.AdvantageEstimator missing"
    if not hasattr(estimator, "GDPO"):
        return False, "verl build does not expose AdvantageEstimator.GDPO"
    if not hasattr(core_algos, "compute_gdpo_outcome_advantage"):
        return False, "verl build does not expose compute_gdpo_outcome_advantage"
    return True, "GDPO support detected"


def _reward_smoke() -> tuple[bool, str]:
    from self_reflection_llm.rl.reward import MARK_CONTINUE, MARK_REFLECT, compute_score_segmented

    result = compute_score_segmented(
        data_source="reflector_harmful",
        solution_str=f"unsafe draft {MARK_REFLECT} this is unsafe {MARK_CONTINUE} I cannot help with that.",
        ground_truth="refuse",
        extra_info={"prompt_type": "harmful", "expected_behavior": "refuse", "question": "bad request"},
    )
    if result["reward_component_correctness"] != 1.0 or result["reward_component_reflect"] != 1.0:
        return False, json.dumps(result, ensure_ascii=False)
    return True, "segmented reward smoke passed"


def main() -> None:
    parser = argparse.ArgumentParser(description="Check dependencies for the Reflector RL workflow.")
    parser.add_argument("--strict", action="store_true", help="Fail if train-time dependencies such as verl/GDPO are missing.")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
    args = parser.parse_args()

    checks: dict[str, dict[str, object]] = {}
    for name in ("datasets", "pyarrow"):
        ok, detail = _check_import(name)
        checks[name] = {"ok": ok, "detail": detail}
    for name in ("torch", "transformers"):
        ok, detail = _check_import(name)
        checks[name] = {"ok": ok, "detail": detail}

    verl_ok, verl_detail = _check_verl_gdpo()
    checks["verl_gdpo"] = {"ok": verl_ok, "detail": verl_detail}

    reward_ok, reward_detail = _reward_smoke()
    checks["reward_smoke"] = {"ok": reward_ok, "detail": reward_detail}

    for path in (ROOT / "data" / "rl" / "general_pattern.json", ROOT / "data" / "rl" / "harmful_pattern.json"):
        rel_path = str(path.relative_to(ROOT))
        checks[rel_path] = {"ok": path.exists(), "detail": rel_path}

    if args.json:
        print(json.dumps(checks, indent=2, ensure_ascii=False))
    else:
        for key, value in checks.items():
            status = "OK" if value["ok"] else "WARN"
            print(f"{status} {key}: {value['detail']}")

    required = ["datasets", "pyarrow", "reward_smoke", "data/rl/general_pattern.json", "data/rl/harmful_pattern.json"]
    if args.strict:
        required.extend(["torch", "transformers", "verl_gdpo"])
    missing = [key for key in required if not checks.get(key, {}).get("ok")]
    if missing:
        raise SystemExit(f"Missing required checks: {missing}")


if __name__ == "__main__":
    main()
