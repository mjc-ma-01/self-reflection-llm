---
license: llama3
library_name: transformers
pipeline_tag: text-generation
base_model:
  - dphn/Dolphin3.0-Llama3.1-8B
tags:
  - llama
  - llama-3
  - llama-3.1
  - safety
  - reflection
  - self-correction
  - sft
  - reflector
language:
  - en
---

# Llama 8B Reflect SFT

`krystal7/llama-8b-reflect-sft` is the SFT release from the Reflector project. It is a Llama 3-family 8B instruction model trained to use reflection-oriented behavior before producing a final answer, with an emphasis on risk-aware generation and safety-aware self-correction.

Reflector targets a practical failure mode in safety alignment: a model may recognize surface-level unsafe prompts, but still struggle with indirect harmful requests, multi-step risky reasoning, jailbreak-style framing, or ambiguous dual-use questions. The core idea is to internalize a self-reflection step so the model can inspect intent, identify potential harm, and redirect toward a safe and useful response at reasoning time.

Paper: [Reflector: Internalizing Self-Reflection into Language Models](https://arxiv.org/abs/2605.20654)

Code: https://github.com/mjc-ma-01/self-reflection-llm

## Model Highlights

- Reflection-based safety alignment for Llama 3-family 8B models.
- Trained with the Reflector SFT pipeline on harmful and general pattern data.
- Designed for risk-aware generation, harmful-intent recognition, and safer multi-step reasoning.
- Stronger handling of indirect harmful requests and jailbreak-style framing than plain instruction following behavior.
- Compatible with local deployment through `transformers` and OpenAI-compatible vLLM serving.

## Intended Use

This model is intended for research and application prototyping around:

- safety-aware chat assistants
- reflective reasoning studies
- harmful-intent detection and safe redirection
- local Llama-style deployment experiments
- evaluation of reasoning-time alignment methods

It is not a replacement for a full production safety stack. Deployments should still use policy filters, monitoring, rate limits, and domain-specific review.

## Quick Start with Transformers

```python
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "krystal7/llama-8b-reflect-sft"

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
    {"role": "user", "content": "How can I learn about lock mechanisms for a security class without doing anything illegal?"},
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

vllm serve krystal7/llama-8b-reflect-sft \
  --dtype bfloat16 \
  --max-model-len 4096 \
  --served-model-name reflector-sft
```

OpenAI-compatible client:

```python
from openai import OpenAI

client = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY")

response = client.chat.completions.create(
    model="reflector-sft",
    messages=[
        {"role": "system", "content": "You are a helpful and harmless assistant."},
        {"role": "user", "content": "Explain how to handle an ambiguous dual-use safety question responsibly."},
    ],
    temperature=0,
    max_tokens=512,
)

print(response.choices[0].message.content)
```

## Training Summary

This checkpoint was trained with the Reflector SFT pipeline.

| Item | Description |
|---|---|
| Base family | Llama 3-family 8B instruction model |
| Training stage | Supervised fine-tuning |
| Data schema | harmful pattern + general pattern |
| Objective | learn reflection-oriented, safety-aware responses |
| Output format | standard HuggingFace causal LM checkpoint |

The repository also includes a GDPO RL pipeline. This SFT model card only describes the released SFT checkpoint.

## Evaluation

The following 50-sample benchmark export was produced with the current Reflector evaluation pipeline. The harmful pattern training data is not used as a public benchmark, and SimpleQA loading is retained only for future data reuse.

| model | benchmark | dataset key | metric | score | n |
|---|---|---|---|---:|---:|
| krystal7/llama-8b-reflect-sft | Do-Not-Answer | donot | refusal_rate | 0.7600 | 50 |
| krystal7/llama-8b-reflect-sft | General Benchmark | general | helpfulness_proxy | 1.0000 | 50 |
| krystal7/llama-8b-reflect-sft | GSM8K | gsm8k | accuracy | 0.7000 | 50 |
| krystal7/llama-8b-reflect-sft | StrongREJECT | strongreject | refusal_rate | 1.0000 | 50 |
| krystal7/llama-8b-reflect-sft | ToxicityPrompts | wildchat | refusal_rate | 0.9400 | 50 |
| krystal7/llama-8b-reflect-sft | XSTest | xstest | helpfulness_proxy | 0.6154 | 26 |
| krystal7/llama-8b-reflect-sft | XSTest | xstest | refusal_rate | 1.0000 | 24 |

## Limitations

- The model can still make factual errors or produce incomplete refusals.
- Safety behavior should be evaluated in the target deployment domain before release.
- The benchmark table is a lightweight reproducibility export, not a comprehensive safety certification.
- Reflection-style behavior may vary with decoding settings, system prompts, and prompt formatting.

## Citation

```bibtex
@misc{reflector2026,
  title         = {Reflector: Internalizing Self-Reflection into Language Models},
  year          = {2026},
  eprint        = {2605.20654},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL}
}
```
