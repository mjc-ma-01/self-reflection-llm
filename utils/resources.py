from __future__ import annotations

import json
import os
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class ResourceInfo:
    gpu_count: int
    gpu_names: list[str]
    total_memory_gb: list[float]
    free_memory_gb: list[float]
    bf16: bool


@dataclass
class TrainPlan:
    stage: str
    strategy: str
    gpu_count: int
    precision: str
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    per_device_eval_batch_size: int
    num_train_epochs: float
    learning_rate: float
    deepspeed_config: str | None
    use_peft: bool
    load_in_4bit: bool

    @property
    def effective_batch_size(self) -> int:
        return max(1, self.gpu_count) * self.per_device_train_batch_size * self.gradient_accumulation_steps


def _nvidia_smi() -> tuple[list[str], list[float], list[float]]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=name,memory.total,memory.free",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL)
    except Exception:
        return [], [], []
    names: list[str] = []
    total: list[float] = []
    free: list[float] = []
    for line in output.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) != 3:
            continue
        names.append(parts[0])
        total.append(round(float(parts[1]) / 1024, 1))
        free.append(round(float(parts[2]) / 1024, 1))
    return names, total, free


def detect_resources() -> ResourceInfo:
    names, total, free = _nvidia_smi()
    bf16 = False
    try:
        import torch

        bf16 = bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported())
        if not names and torch.cuda.is_available():
            names = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
            total = [round(torch.cuda.get_device_properties(i).total_memory / 1024**3, 1) for i in range(torch.cuda.device_count())]
            free = total[:]
    except Exception:
        pass
    return ResourceInfo(
        gpu_count=len(names),
        gpu_names=names,
        total_memory_gb=total,
        free_memory_gb=free,
        bf16=bf16,
    )


def choose_sft_plan(resources: ResourceInfo | None = None) -> TrainPlan:
    resources = resources or detect_resources()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_count = len([item for item in visible.split(",") if item.strip()]) if visible else resources.gpu_count
    free = resources.free_memory_gb[:gpu_count] if gpu_count else []
    min_free = min(free) if free else 0.0
    precision = "bf16" if resources.bf16 else "fp16"

    if gpu_count >= 4 and min_free >= 60:
        return TrainPlan("sft", "full_deepspeed_zero2", gpu_count, precision, 1, 4, 1, 3.0, 5e-6, "configs/deepspeed/zero2.json", False, False)
    if gpu_count >= 1 and min_free >= 28:
        return TrainPlan("sft", "lora_single_or_small_multi_gpu", gpu_count, precision, 1, 16, 1, 3.0, 1e-4, None, True, False)
    if gpu_count >= 1:
        return TrainPlan("sft", "qlora_low_memory", gpu_count, precision, 1, 32, 1, 1.0, 1e-4, None, True, True)
    return TrainPlan("sft", "cpu_debug", 0, "fp32", 1, 1, 1, 0.01, 1e-5, None, True, False)


def choose_rl_plan(resources: ResourceInfo | None = None) -> TrainPlan:
    resources = resources or detect_resources()
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    gpu_count = len([item for item in visible.split(",") if item.strip()]) if visible else resources.gpu_count
    free = resources.free_memory_gb[:gpu_count] if gpu_count else []
    min_free = min(free) if free else 0.0
    precision = "bf16" if resources.bf16 else "fp16"
    if gpu_count >= 4 and min_free >= 60:
        return TrainPlan("rl", "verl_gdpo_full", gpu_count, precision, 1, 8, 1, 1.0, 1e-6, None, False, False)
    if gpu_count >= 1 and min_free >= 80:
        return TrainPlan("rl", "verl_gdpo_single_gpu_full", gpu_count, precision, 1, 16, 1, 1.0, 1e-6, None, False, False)
    if gpu_count >= 1:
        return TrainPlan("rl", "insufficient_memory_for_full_gdpo", gpu_count, precision, 1, 32, 1, 0.0, 0.0, None, False, False)
    return TrainPlan("rl", "cpu_unavailable", 0, "fp32", 1, 1, 1, 0.0, 0.0, None, False, False)


def write_resource_report(path: str | Path) -> dict[str, object]:
    resources = detect_resources()
    report = {
        "resources": asdict(resources),
        "sft_plan": asdict(choose_sft_plan(resources)),
        "rl_plan": asdict(choose_rl_plan(resources)),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    report = {
        "resources": asdict(detect_resources()),
        "sft_plan": asdict(choose_sft_plan()),
        "rl_plan": asdict(choose_rl_plan()),
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
