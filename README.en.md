# QEdge

An experimental Qwen3.5-2B multimodal inference engine from a DNDX tuning project, targeting a T-Head PPU / SAIL-compatible environment. [中文说明](README.md).

QEdge separates scheduling, cache management, and model execution. It includes paged KV management, hybrid recurrent-state handling, multimodal prefix identity, fused operations, and a benchmark-compatible wrapper.

## Status and requirements

CI covers host-side logic; accelerator compatibility and performance require validation in the target environment. The dependency file lists required packages without a validated version lock. Follow the [installation notes](docs/installation.md) and use compatible vendor builds. Full prefix caching and FLA prefill are disabled by default. Default maximum concurrency is one request.

## Host-only checks

```bash
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
python scripts/check_source.py
```

These checks exercise real host-side modules without importing the accelerator-dependent package initializer. They do not validate full inference or kernels.

## Public development benchmark

Obtain the model and public MMBench development data separately; see [data sources](docs/data.md). In a prepared accelerator environment:

```bash
python benchmark_public.py --dataset-path ./datasets/mmbench/mmbench_dev_en.tsv --model-path ./Qwen3.5-2B --backend transformers --num-samples 20 --output result_dev_en.json
```

Use the explicit `transformers` backend name to fail on loading errors. This compatibility name loads the custom engine, not Transformers `generate`. `auto` may fall back to dummy output and is unsuitable for performance evidence.

The archived 2,000-sample result reports 80.65% accuracy, 60.449 ms mean TTFT, and 680.489 tokens/s mean per-sample decode throughput. This is historical, not an independently reproduced result or a system-throughput claim. Hardware and runtime revisions were not recorded.

Before public release, resolve [license status](LICENSE-STATUS.md) and retain required third-party notices. No new open-source license has been asserted for code of unverified provenance. [Contributing](CONTRIBUTING.md).
