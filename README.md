pplx-garden
===========

Perplexity AI open source garden for inference technology

> [!NOTE]
> This AA-Box fork extends **Lily** to Apple GPU family 9 so it can run on
> **M4, M4 Pro, and M4 Max** systems on macOS 26+. Upstream Lily intentionally
> gates the production path to Apple GPU family 10 / M5+. The AA-Box port keeps
> the same Rust + Metal engine and Metal 4 tensor kernels while allowing the
> family-9 compatibility path. M5+ keeps its native GPU neural-accelerator path.
>
> See [lily/README.md](lily/README.md#apple-m4-support) for requirements,
> limitations, validation commands, and performance expectations.

## Projects

### fabric-lib

RDMA TransferEngine, P2P MoE dispatch/combine kernel

* Docs: [docs/fabric-lib.md](docs/fabric-lib.md)
* MLSys'26 paper: [fabric-lib: RDMA Point-to-Point Communication for LLM Systems](https://arxiv.org/abs/2510.27656)
* Blog Post: [RDMA Point-to-Point Communication for LLM Systems](https://research.perplexity.ai/articles/rdma-point-to-point-communication-for-llm-systems)
* Blog Post: [Enabling Trillion-Parameter Models on AWS EFA](https://research.perplexity.ai/articles/enabling-trillion-parameter-models-on-aws-efa)
* Blog Post: [Weight Transfer for RL Post-Training in under 2 seconds](https://research.perplexity.ai/articles/weight-transfer-for-rl-post-training-in-under-2-seconds)
* Blog Post: [Disaggregated Prefill and Decode](https://research.perplexity.ai/articles/disaggregated-prefill-and-decode)

### pplx-unigram

Unigram tokenizer encoder

* Docs: [docs/unigram.md](docs/unigram.md)
* Blog Post: [Improving Unigram Tokenizer CPU Performance](https://research.perplexity.ai/articles/improving-unigram-tokenizer-cpu-performance)

### lily

Rust and Metal inference server for Qwen3.6-35B-A3B on Apple Silicon. Lily
provides greedy text generation through a minimal OpenAI-compatible HTTP API.

The AA-Box fork adds Apple GPU family 9 support for M4-class Macs while
preserving the upstream M5+ path.

* Docs: [lily/README.md](lily/README.md)
* Upstream Blog Post: [Optimizing On-Device Inference for Apple Silicon](https://www.perplexity.ai/hub/blog/optimizing-on-device-inference-for-apple-silicon)
* Upstream: [perplexityai/pplx-garden](https://github.com/perplexityai/pplx-garden)
* License: [Apache-2.0](lily/LICENSE), with third-party notices in [lily/NOTICE](lily/NOTICE)

## Directory Structure

* `fabric-lib/`: RDMA TransferEngine library
* `lily/`: Metal LLM inference for Apple Silicon
* `p2p-all-to-all/`: P2P MoE All-to-All implementation
* `pplx-unigram/`: Unigram tokenizer encoder
* `python-ext/`: Python extension module from Rust code
* `python/pplx_garden/`: Python code for the `pplx_garden` package
* `rust/`: Rust utility libraries
