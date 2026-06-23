---
license: llama3.1
base_model: meta-llama/Llama-3.1-8B-Instruct
library_name: transformers
pipeline_tag: text-generation
tags:
  - safety
  - reinforcement-learning
  - reflection
  - self-correction
  - gdpo
  - llama
  - llama-3.1
  - reflector
language:
  - en
---

# Reflector Internalizing Safety Llama 3.1 8B RL

`krystal7/Reflector-Internalizing-Safety-Llama-3.1-8B-RL` is the RL release from the Reflector project. It is a Llama 3.1 8B instruction model trained with the Reflector GDPO reinforcement-learning pipeline to strengthen step-wise reflection before the final answer.

Reflector targets a practical failure mode in safety alignment: a model may handle direct unsafe prompts, but still struggle with indirect jailbreaks, multi-step risky reasoning, or ambiguous dual-use requests. The RL checkpoint reinforces reflection quality, harmful-intent recognition, and safe redirection so the final response is safer and still useful.

Paper: [REFLECTOR: Internalizing Step-wise Reflection against Indirect Jailbreak](https://arxiv.org/abs/2605.20654)

Code: https://github.com/mjc-ma-01/self-reflection-llm

## Model Highlights

- RL-trained Reflector checkpoint for Llama 3.1 8B style chat generation.
- Trained with the repository's GDPO pipeline after reflection-oriented alignment.
- Designed for indirect jailbreak resistance, risk-aware generation, and safer final answers.
- Compatible with local deployment through `transformers` and OpenAI-compatible vLLM serving.

## Intended Use

This model is intended for research and application prototyping around:

- safety-aware chat assistants
- reflective reasoning studies
- indirect jailbreak and harmful-intent evaluation
- RL-based safety alignment experiments
- local Llama-style deployment tests

It is not a replacement for a full production safety stack. Deployments should still use policy filters, monitoring, rate limits, and domain-specific review.

## Quick Start with Transformers

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "krystal7/Reflector-Internalizing-Safety-Llama-3.1-8B-RL"

os.environ.setdefault("HF_HOME", "./hf_cache")
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", os.path.join(os.environ["HF_HOME"], "hub"))

tokenizer = AutoTokenizer.from_pretrained(
    model_id,
    cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    cache_dir=os.environ["HUGGINGFACE_HUB_CACHE"],
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
)

messages = [
    {"role": "system", "content": "You are a helpful and harmless assistant."},
    {"role": "user", "content": "How can I handle an ambiguous dual-use request safely?"},
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

with torch.no_grad():
    output_ids = model.generate(
        **inputs,
        max_new_tokens=512,
        do_sample=False,
        temperature=None,
        pad_token_id=tokenizer.eos_token_id,
    )

answer = tokenizer.decode(output_ids[0, inputs["input_ids"].shape[-1]:], skip_special_tokens=True)
print(answer.strip())
```

## vLLM Serving

```bash
pip install vllm

export HF_HOME=./hf_cache
export HUGGINGFACE_HUB_CACHE=$HF_HOME/hub

vllm serve krystal7/Reflector-Internalizing-Safety-Llama-3.1-8B-RL \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --served-model-name reflector-rl
```

OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="reflector-rl",
    messages=[
        {"role": "system", "content": "You are a helpful and harmless assistant."},
        {"role": "user", "content": "Explain how to answer an indirect harmful request responsibly."},
    ],
    temperature=0,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

## Training Summary

This checkpoint was trained with the Reflector GDPO RL pipeline.

| Item | Description |
|---|---|
| Base family | Llama 3.1 8B instruction model |
| Training stage | Reinforcement learning |
| RL method | GDPO |
| Data schema | harmful pattern + general pattern |
| Objective | reinforce step-wise reflection, harmful-intent recognition, and safe final answers |
| Output format | standard HuggingFace causal LM checkpoint |

## Evaluation

Use the repository evaluation scripts to reproduce local benchmark exports for this checkpoint:

```bash
export MODEL_PATH=krystal7/Reflector-Internalizing-Safety-Llama-3.1-8B-RL
NUM_SAMPLES=50 bash scripts/evaluation/eval_general.sh
```

The harmful pattern training data is not used as a public benchmark.

## Limitations

- The model can still make factual errors or produce incomplete refusals.
- Safety behavior should be evaluated in the target deployment domain before release.
- Reflection-style behavior may vary with decoding settings, system prompts, and prompt formatting.
- This model is a research checkpoint, not a comprehensive safety certification.

## Citation

```bibtex
@article{ma2026reflector,
  title={REFLECTOR: Internalizing Step-wise Reflection against Indirect Jailbreak},
  author={Ma, Jiachen and Zhang, Jiawen and Li, Xiangtian and Zou, Bo and Lu, Chaochao and Yang, Chao},
  journal={arXiv preprint arXiv:2605.20654},
  year={2026}
}
```
