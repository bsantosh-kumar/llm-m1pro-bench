# llm-m1pro-bench

Measuring how fast small LLMs run on Apple Silicon, comparing **llama.cpp** and **MLX** across precisions (F16, Q8_0, Q4_K_M).

## Setup

| Item | Value |
|---|---|
| Machine | Apple M1 Pro, 16 GB unified memory |
| OS | macOS 26.6.2 (build 25G83) |
| llama.cpp | 0.7.0, build 11669, commit 620d2225c (AppleClang 21.0.0) |
| MLX | TODO |
| Model | [ggml-org/gemma-3-1b-it-GGUF](https://huggingface.co/ggml-org/gemma-3-1b-it-GGUF) (F16, Q8_0, Q4_K_M) |

## How to run

```bash
# download a precision
llama download --hf-repo ggml-org/gemma-3-1b-it-GGUF:F16

# run with verbose logs (confirms Metal backend)
llama cli -hf ggml-org/gemma-3-1b-it-GGUF:F16 -v --log-file first-llama.cpp.log
grep -i metal first-llama.cpp.log
```

TODO: fix the prompt and the number of generated tokens so runs are comparable.

## Results

| Date | Runtime | Model | Precision | Size | Backend | Prompt (t/s) | Generation (t/s) |
|---|---|---|---|---|---|---|---|
| 2026-10-20 | llama.cpp 0.7.0 | gemma-3-1b-it | F16 | 2.01 GB | Metal | 519.1 | 72.0 |
| | llama.cpp 0.7.0 | gemma-3-1b-it | Q8_0 | 1.07 GB | Metal | TODO | TODO |
| | llama.cpp 0.7.0 | gemma-3-1b-it | Q4_K_M | 806 MB | Metal | TODO | TODO |
| | MLX | gemma-3-1b-it | TODO | | Metal | TODO | TODO |

Single runs so far; final numbers will be the median of 5+ runs after a warm-up.

## Notes

- **Generation is memory-bandwidth-bound.** Each generated token reads every weight once. M1 Pro bandwidth is 200 GB/s, so for the 2.01 GB F16 model the ceiling is about 200 / 2.01 ≈ 100 t/s. Measured 72.0 t/s ≈ 72% of the ceiling.
- **Prompt processing is compute-bound.** The prompt is processed as a batch (matrix × matrix kernels, `kernel_mul_mm_*`), so it runs much faster than generation (matrix × vector, `kernel_mul_mv_*`).
