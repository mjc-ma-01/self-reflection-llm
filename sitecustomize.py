"""Process-local compatibility hooks for the research pipeline.

The hooks are disabled by default. Training launchers opt in with environment
variables so normal Python commands keep the upstream library behavior.
"""

from __future__ import annotations

import os


def _enable_verl_vllm_llama_key_compat() -> None:
    try:
        from verl.utils import model as verl_model_utils
        from verl.workers import fsdp_workers
    except Exception:
        return

    original = verl_model_utils.convert_weight_keys
    if getattr(original, "_self_reflection_compat", False):
        return

    def convert_weight_keys_compat(state_dict, model):
        converted = original(state_dict, model)
        fixed = {}
        changed = False
        for key, value in converted.items():
            if key.startswith(("embed_tokens.", "layers.", "norm.")):
                fixed[f"model.{key}"] = value
                changed = True
            else:
                fixed[key] = value
        return fixed if changed else converted

    convert_weight_keys_compat._self_reflection_compat = True
    verl_model_utils.convert_weight_keys = convert_weight_keys_compat
    fsdp_workers.convert_weight_keys = convert_weight_keys_compat


if os.environ.get("SELF_REFLECTION_VERL_VLLM_LLAMA_KEY_COMPAT") == "1":
    _enable_verl_vllm_llama_key_compat()
