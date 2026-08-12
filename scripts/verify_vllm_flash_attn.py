"""Minimal end-to-end smoke test for external FlashAttention and vLLM."""

import importlib.metadata as metadata
import os

import flash_attn
import torch
from flash_attn import flash_attn_func
from vllm import LLM, SamplingParams


def run_flash_attn() -> None:
    q = torch.randn(1, 16, 2, 64, device="cuda", dtype=torch.float16)
    out = flash_attn_func(q, q, q, dropout_p=0.0, causal=False)
    torch.cuda.synchronize()
    assert out.shape == q.shape


def main() -> None:
    assert torch.cuda.is_available(), "CUDA is not available"
    before = (metadata.version("flash-attn"), flash_attn.__file__)

    run_flash_attn()

    model = os.getenv("VLLM_TEST_MODEL", "Qwen/Qwen2.5-0.5B-Instruct")
    llm = LLM(
        model=model,
        dtype="float16",
        max_model_len=128,
        gpu_memory_utilization=0.2,
    )
    result = llm.generate(
        ["Say OK."], SamplingParams(temperature=0.0, max_tokens=4)
    )
    assert result[0].outputs[0].text

    run_flash_attn()
    after = (metadata.version("flash-attn"), flash_attn.__file__)
    assert before == after, f"flash-attn changed: {before} -> {after}"

    print("PASS: vLLM and external flash-attn both ran successfully.")
    print(f"torch={torch.__version__}, vllm={metadata.version('vllm')}")
    print(f"flash-attn={after[0]}, path={after[1]}")


if __name__ == "__main__":
    main()
