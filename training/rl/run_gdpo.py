from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

from utils.paths import CHECKPOINTS_DIR, OUTPUTS_DIR, set_hf_cache_env
from utils.resources import choose_rl_plan, detect_resources

DEFAULT_MODEL = "checkpoints/sft/merged"
DEFAULT_DATA_DIR = OUTPUTS_DIR / "processed" / "rl"


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Launch Reflector GDPO with verl and the original segmented reward.")
    parser.add_argument("--model_path", default=os.environ.get("MODEL_PATH", DEFAULT_MODEL))
    parser.add_argument("--train_file", type=Path, default=DEFAULT_DATA_DIR / "train.parquet")
    parser.add_argument("--val_file", type=Path, default=DEFAULT_DATA_DIR / "test.parquet")
    parser.add_argument("--output_dir", type=Path, default=CHECKPOINTS_DIR / "rl")
    parser.add_argument("--experiment_name", default="llama3_8b_reflector_gdpo")
    parser.add_argument("--project_name", default="reflector")
    parser.add_argument("--epochs", type=float, default=None)
    parser.add_argument("--rollout_n", type=int, default=8)
    parser.add_argument("--max_prompt_length", type=int, default=2048)
    parser.add_argument("--max_response_length", type=int, default=512)
    parser.add_argument("--train_batch_size", type=int, default=None)
    parser.add_argument("--ppo_micro_batch_size_per_gpu", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    parser.add_argument("--extra_verl_args", nargs=argparse.REMAINDER, default=[])
    return parser


def main() -> None:
    set_hf_cache_env()
    args = build_arg_parser().parse_args()
    model_path = args.model_path
    resources = detect_resources()
    plan = choose_rl_plan(resources)
    if plan.gpu_count <= 0:
        raise RuntimeError("GDPO training requires at least one CUDA GPU.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    total_epochs = max(1, int(round(args.epochs if args.epochs is not None else plan.num_train_epochs)))
    train_batch_size = args.train_batch_size or max(plan.effective_batch_size, plan.gpu_count * args.rollout_n)
    micro_batch = args.ppo_micro_batch_size_per_gpu or plan.per_device_train_batch_size
    lr = args.learning_rate or plan.learning_rate

    report = {
        "algorithm": "gdpo",
        "resources": resources.__dict__,
        "plan": plan.__dict__,
        "model_path": model_path,
        "train_file": str(args.train_file),
        "val_file": str(args.val_file),
        "train_batch_size": train_batch_size,
        "rollout_n": args.rollout_n,
        "ppo_micro_batch_size_per_gpu": micro_batch,
        "learning_rate": lr,
        "gdpo_reward_keys": ["reward_component_correctness", "reward_component_reflect"],
        "gdpo_reward_weights": [1.0, 0.3],
        "hf_cache": os.environ.get("HUGGINGFACE_HUB_CACHE"),
    }
    (args.output_dir / "resource_plan.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    reward_path = Path(__file__).with_name("reward.py").resolve()
    cmd = [
        "python",
        "-m",
        "verl.trainer.main_ppo",
        f"actor_rollout_ref.model.path={model_path}",
        f"data.train_files=['{args.train_file}']",
        f"data.val_files=['{args.val_file}']",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"data.train_batch_size={train_batch_size}",
        "actor_rollout_ref.rollout.name=vllm",
        "actor_rollout_ref.rollout.load_format=auto",
        f"actor_rollout_ref.rollout.max_model_len={args.max_prompt_length + args.max_response_length}",
        f"actor_rollout_ref.rollout.max_num_seqs={max(train_batch_size * args.rollout_n, 1)}",
        f"actor_rollout_ref.rollout.n={args.rollout_n}",
        "actor_rollout_ref.rollout.tensor_model_parallel_size=1",
        "actor_rollout_ref.rollout.pipeline_model_parallel_size=1",
        f"actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu={micro_batch}",
        f"actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu={micro_batch}",
        f"actor_rollout_ref.actor.ppo_mini_batch_size={train_batch_size}",
        f"actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu={micro_batch}",
        "actor_rollout_ref.actor.use_torch_compile=false",
        "actor_rollout_ref.ref.use_torch_compile=false",
        f"actor_rollout_ref.actor.optim.lr={lr}",
        "+actor_rollout_ref.model.override_config.attn_implementation=sdpa",
        "algorithm.adv_estimator=gdpo",
        "+algorithm.gdpo_reward_keys=[reward_component_correctness,reward_component_reflect]",
        "+algorithm.gdpo_reward_weights=[1.0,0.3]",
        f"custom_reward_function.path={reward_path}",
        "custom_reward_function.name=compute_score_segmented",
        "+custom_reward_function.reward_kwargs.reflect_bonus_correct=0.3",
        "+custom_reward_function.reward_kwargs.reflect_penalty_wrong=-0.3",
        "+custom_reward_function.reward_kwargs.reflect_reward_prompt_types=harmful",
        f"trainer.project_name={args.project_name}",
        f"trainer.experiment_name={args.experiment_name}",
        "trainer.nnodes=1",
        f"trainer.n_gpus_per_node={plan.gpu_count}",
        f"trainer.total_epochs={total_epochs}",
        f"trainer.default_local_dir={args.output_dir}",
        "trainer.logger=[console]",
        '+ray_kwargs.ray_init.runtime_env.env_vars.SELF_REFLECTION_VERL_VLLM_LLAMA_KEY_COMPAT="1"',
        f"+ray_kwargs.ray_init.runtime_env.env_vars.PYTHONPATH={Path.cwd()}",
    ]
    cmd.extend(args.extra_verl_args)
    env = os.environ.copy()
    env.setdefault("PYTHONPATH", str(Path.cwd()))
    env["PYTHONPATH"] = f"{Path.cwd()}:{env['PYTHONPATH']}"
    env.setdefault("WANDB_DISABLED", "true")
    env.setdefault("HYDRA_FULL_ERROR", "1")
    env["SELF_REFLECTION_VERL_VLLM_LLAMA_KEY_COMPAT"] = "1"
    print(" ".join(cmd))
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
