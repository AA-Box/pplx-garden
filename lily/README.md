# lily

A small Metal inference server for one checkpoint: Qwen3.6-35B-A3B converted
to MLX affine 4-bit weights. Lily exposes a minimal subset of the OpenAI chat
completions API and always decodes greedily.

This AA-Box fork adds Apple GPU family 9 compatibility for M4-class Macs while
retaining the upstream M5+ implementation. The inference engine remains Rust +
Metal; this port does not route inference through MLX.

Performance reports from upstream include the measurement contract and
reproduction steps:

- [2026-09-01: MLX 0.31.2](docs/2026-09-01-performance-mlx-0.31.2.md)
- [2026-09-02: MLX 0.32.2](docs/2026-09-02-performance-mlx-0.32.2.md)

Those reports were measured on M5 Max and should not be treated as M4
performance numbers.

The Metal kernels compile from source at runtime; there is no offline shader
build step.

## Requirements

- Apple GPU family 9 or later
  - M4 / M4 Pro / M4 Max: supported by the AA-Box compatibility path
  - M5 and newer: supported with the upstream native tensor-accelerated path
  - M3 is also Apple GPU family 9 and is accepted by the capability check, but
    this fork is specifically targeting and documenting M4
- macOS 26 or later for Metal 4 tensor operations
- Rust 1.92, pinned by `rust-toolchain.toml`
- A local Qwen3.6-35B-A3B MLX affine 4-bit checkpoint with group size 64

## Apple M4 support

Upstream Lily deliberately rejects devices below Apple GPU family 10 even
though its device code already identifies family 9 as the M3/M4 generation.
The AA-Box port changes the minimum accepted family to 9.

Lily's production prefill path uses Metal 4 tensors and Metal Performance
Primitives for BF16 GEMM, grouped Q4 GEMM, and prefill attention. On M5 /
Apple10, these operations can use the Neural Accelerator present in each GPU
core. On M4 / Apple9, the same Metal 4 programming model is used without the
M5 hardware Neural Accelerators.

This means M4 support is a compatibility port, not an attempt to claim M5
performance on M4. Tensor-heavy prefill is expected to be slower than M5;
decode remains much more sensitive to memory bandwidth and the existing
non-tensor kernels.

The port intentionally does **not**:

- convert Lily to MLX-LM;
- change the Qwen checkpoint format;
- add a CPU fallback;
- pretend that M4 has M5 Neural Accelerators;
- alter the OpenAI-compatible HTTP API.

Apple's public Metal feature tables identify M3 and M4 as Apple GPU family 9
and M5 as family 10. Metal 4 itself is available on these Apple Silicon
systems; family 10 adds the GPU Neural Accelerator hardware used to accelerate
tensor operations.

### Validate on an M4 Mac

From the `lily` directory:

```sh
cargo build --release --locked
cargo test --locked
```

The shader test compiles every shipped Metal source at MSL 3.1 and then at
MSL 4.0. With the family-9 gate enabled, an M4 reaches the production Metal 4
shader compilation instead of failing immediately during `MetalContext`
creation.

For an end-to-end model test:

```sh
LILY_MODEL_DIR_35B=/path/to/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_e2e_35b -- --ignored --test-threads=1
```

For performance measurements, use `lily-bench` and report the exact M4 SKU,
GPU core count, unified-memory size, macOS version, model revision, prompt
length, and decode length. Do not compare an M4 result directly with the
upstream M5 Max benchmark without accounting for the hardware difference.

Lily validates the exact 35B-A3B architecture and quantization layout at load
time. Dense Qwen checkpoints, smaller Qwen checkpoints, BF16 checkpoints,
GGUF, AWQ, GPTQ, int8 and fp8 are not supported.

The release benchmark and tests use the immutable
`mlx-community/Qwen3.6-35B-A3B-4bit` revision
`38740b847e4cb78f352aba30aa41c76e08e6eb46`. Download that exact checkpoint
with the Hugging Face CLI:

```sh
hf download mlx-community/Qwen3.6-35B-A3B-4bit \
  --revision 38740b847e4cb78f352aba30aa41c76e08e6eb46 \
  --local-dir /path/to/Qwen3.6-35B-A3B-4bit
```

## Run

```sh
cargo build --release --locked

./target/release/lily \
  --model /path/to/Qwen3.6-35B-A3B-4bit \
  --bind 127.0.0.1:8000 \
  --max-seq 4096
```

The server provides:

- `POST /v1/chat/completions`
- `GET /v1/models`
- `GET /health`

```sh
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "Qwen3.6-35B-A3B",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 64,
    "prompt_cache_key": "conversation-1"
  }'
```

The request surface is intentionally strict. It accepts text-only
system/user/assistant messages, `max_tokens`, `stream: false`, and the optional
`prompt_cache_key`. The final message must have role `user`, and the server
always renders the checkpoint template with thinking disabled. Sampling
parameters, streaming responses, tools, response formats, multimodal content,
and speculative decoding are rejected.

`--max-seq` is the combined prompt-plus-completion capacity. It cannot exceed
the kernel limit of 262,144 tokens and is clamped to the checkpoint's declared
`max_position_embeddings` when that value is smaller.

## Session prefix cache

The server keeps a fixed two-entry LRU cache of decode states. A state is reused
only when its token sequence is a strict prefix of the new prompt. A matching
`prompt_cache_key` prefers the corresponding entry but never bypasses token
equality. The response reports reused tokens in
`usage.prompt_tokens_details.cached_tokens`.

## Tests

```sh
cargo test --locked
```

This runs the CPU-reference kernel tests, shader compilation test, API surface
tests, and prefix-cache tests. Tests that need the 35B checkpoint are explicitly
ignored by default:

```sh
LILY_MODEL_DIR_35B=/path/to/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_tokenizer -- --ignored --test-threads=1

LILY_MODEL_DIR_35B=/path/to/Qwen3.6-35B-A3B-4bit \
  cargo test --test test_e2e_35b -- --ignored --test-threads=1
```

## Source layout

```text
src/config.rs       strict 35B-A3B checkpoint validation
src/weights.rs      MLX affine Q4 weight loading
src/model.rs        prefill and single-token greedy model graph
src/generate.rs     tokenizer-backed greedy decode loop
src/serve.rs        minimal OpenAI-compatible HTTP server
src/serve/session.rs strict token-prefix session cache
src/kernels/        Rust dispatch and Metal shader sources
tests/              kernel, API, tokenizer, shader, and 35B golden tests
benchmarks/         Lily/MLX harnesses and the fail-closed matrix runner
```

## Upstream

This fork is based on [perplexityai/pplx-garden](https://github.com/perplexityai/pplx-garden).
The M4 compatibility changes are maintained by AA-Box and are not claims about
upstream support policy.

## License

Apache-2.0. See `LICENSE` and `NOTICE`.
