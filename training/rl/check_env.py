from __future__ import annotations

import argparse
import importlib
import json

from utils.paths import RL_DIR, repo_relative
from utils.resources import detect_resources, choose_rl_plan
from training.rl.reward import MARK_CONTINUE, MARK_REFLECT, compute_score_segmented


def _import_status(name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(name)
        return {"ok": True, "detail": getattr(module, "__version__", "installed")}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


def _verl_status() -> dict[str, object]:
    try:
        importlib.import_module("verl.trainer.main_ppo")
        from verl.trainer.ppo.core_algos import AdvantageEstimator, get_adv_estimator_fn

        get_adv_estimator_fn(AdvantageEstimator.GDPO)
        return {"ok": True, "detail": "verl GDPO entry point available"}
    except Exception as exc:
        return {"ok": False, "detail": str(exc)}


def _reward_smoke() -> dict[str, object]:
    result = compute_score_segmented(
        data_source="reflector_harmful",
        solution_str=f"unsafe draft {MARK_REFLECT} this is unsafe {MARK_CONTINUE} I cannot help with that.",
        ground_truth="refuse",
        extra_info={"prompt_type": "harmful", "expected_behavior": "refuse", "question": "unsafe request"},
    )
    ok = result["reward_component_correctness"] == 1.0 and result["reward_component_reflect"] == 1.0
    return {"ok": ok, "detail": result}


def main() -> None:
    parser = argparse.ArgumentParser(description="Check Reflector GDPO/RL environment.")
    parser.add_argument("--strict", action="store_true")
    parser.add_argument("--json", action="store_true")
    args = parser.parse_args()

    resources = detect_resources()
    checks = {
        "resources": {"ok": resources.gpu_count > 0, "detail": resources.__dict__},
        "rl_plan": {"ok": True, "detail": choose_rl_plan(resources).__dict__},
        "datasets": _import_status("datasets"),
        "pyarrow": _import_status("pyarrow"),
        "torch": _import_status("torch"),
        "transformers": _import_status("transformers"),
        "verl": _verl_status(),
        "reward_smoke": _reward_smoke(),
        "datasets/rl/harmful_pattern.json": {
            "ok": (RL_DIR / "harmful_pattern.json").exists(),
            "detail": repo_relative(RL_DIR / "harmful_pattern.json"),
        },
        "datasets/rl/general_pattern.json": {
            "ok": (RL_DIR / "general_pattern.json").exists(),
            "detail": repo_relative(RL_DIR / "general_pattern.json"),
        },
    }
    if args.json:
        print(json.dumps(checks, ensure_ascii=False, indent=2))
    else:
        for name, value in checks.items():
            status = "OK" if value["ok"] else "WARN"
            print(f"{status} {name}: {value['detail']}")
    required = ["datasets", "pyarrow", "torch", "transformers", "reward_smoke", "datasets/rl/harmful_pattern.json", "datasets/rl/general_pattern.json"]
    if args.strict:
        required.append("verl")
    missing = [name for name in required if not checks[name]["ok"]]
    if missing:
        raise SystemExit(f"Missing required RL checks: {missing}")


if __name__ == "__main__":
    main()
