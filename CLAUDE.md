# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Databricks AI Runtime (AIR) demo for DAIS: fine-tune `unsloth/Qwen3-4B-Instruct-2507` with Unsloth LoRA on IBM TabFormer credit-card fraud transactions, register the merged model to Unity Catalog as a custom LLM, deploy it to Mosaic AI Model Serving (vLLM), and load test the endpoint at high QPS. The deliverables are Databricks notebooks plus YAML configs — there is no test suite, linter, or build step.

## Commands

```bash
# Local environment (only needed for Databricks Connect execution)
python -m venv .venv && source .venv/bin/activate
pip install -r train/requirements.txt

# Verify CLI auth (profile DEFAULT; workspace host is in databricks.yml)
databricks auth profiles

# Run the ingestion notebook locally via Databricks Connect (serverless)
.venv/bin/python setup/01_load_tabformer_dataset.py

# Submit training via the AI Runtime CLI (alternative to the notebook's @distributed cell).
# COPYFILE_DISABLE=1 is required on macOS: without it bsdtar synthesizes AppleDouble
# (._*) entries in the code snapshot tarball from file xattrs; the remote launcher
# resolves $CODE_SOURCE_PATH from the tarball's first entry and picks `._train`,
# failing with "can't open file '/databricks/code_source/._train/train.py'".
# The CLI's own --exclude='._*' can't catch these (they're synthesized at archive
# time, not read from disk), and bsdtar hides them from `tar -tzf` listings —
# inspect snapshots with Python tarfile.getnames() instead.
cd train && COPYFILE_DISABLE=1 air run --file train.yaml --watch

# Bundle operations (Declarative Automation Bundle, target: dev)
databricks bundle validate
```

The training runner notebook (`train/runner.py`) **cannot run locally** — it requires Databricks Serverless GPU with the AI Runtime `AI v5` environment (`serverless_gpu.distributed`, H100/A10 GPUs). Run it as a workspace notebook. The load-test notebook runs on Databricks serverless compute.

## Databricks Notebook Source Format

Every `.py` file in `setup/`, `train/`, and `load_test/` is a Databricks notebook **except `train/train.py`**, which is a plain Python module holding the training implementation (imported by the training notebook and executed directly by the AI Runtime CLI per `train/train.yaml`). When editing notebooks, preserve the format: `# Databricks notebook source` header, `# COMMAND ----------` cell separators, `# MAGIC %md` markdown cells.

- Every consumer (the three notebooks, `train.py`, local scripts) gets the shared helpers (`load_yaml_config`, `config_str`/`config_int`/`config_float`/`config_bool`, `quote_identifier`, `full_name`, `get_spark_session`, `load_training_config`, `init_training_workspace`) the same way: insert the `train/` directory into `sys.path`, then `from training_utils import ...`. Never `%run` it, and never add the `# Databricks notebook source` header to it — notebook-formatted files cannot be imported in the workspace (`NotebookImportException: Importing notebooks directly is not supported`), and `train.py` must be able to import the module under the runner notebook, the AIR CLI, and local runs alike.
- The helpers module must **never** be renamed back to `utils.py`: the Databricks AI GPU base environment's `nvidia_cutlass_dsl` package registers a top-level `utils` module once torch/CUDA loads (e.g. when `train.py` imports unsloth), poisoning `sys.modules` so a local `from utils import ...` resolves to cutlass's vendored module and fails with `ImportError: cannot import name ... from 'utils' (...nvidia_cutlass_dsl/...)`.
- Training config flows through `training_utils.load_training_config()`, which returns a flat dict that the runner notebook and `train.py` each bind with `globals().update(...)` — so both see identical constants (`MODEL_NAME`, `TRAINING_OUTPUT_DIR`, `sft_table_q`, ...). It reads `parameters.training_config` from the YAML file at `$HYPERPARAMETERS_PATH` when AIR sets it (so `air run --override` values reach the script), falling back to `train.yaml` next to `training_utils.py`. It is deliberately a function, not top-level module code, so importing `training_utils` has no side effects (no config parse, no Spark, no schema creation). `load_yaml_config` resolves paths against the module's own directory by default; callers in other folders (the load-test notebook) pass `base_dir=Path.cwd()` for their own YAML.
- Notebooks support dual execution: inside Databricks (serverless notebooks, where `Path.cwd()` is the notebook's folder) or locally via Databricks Connect. `get_spark_session()` in `training_utils.py` attaches a serverless Connect session when none exists; `setup/01_load_tabformer_dataset.py` resolves its own directory via `__file__` with a `dbutils` notebook-context fallback before importing the helpers from `train/`.

## Architecture: Three-Stage Pipeline Coupled by YAML Configs

Each stage is a notebook driven by a sibling YAML file, connected to the next stage only through Unity Catalog objects:

1. **Ingest** — `setup/01_load_tabformer_dataset.py` + `setup/setup.yaml`: downloads TabFormer into a UC staging volume, writes the cleaned transaction table (`fraud_dataset`) and the SFT table (`fraud_sft_dataset`) with `prompt`, `assistant_response`, and `shard_id` columns, then exports the SFT records to the `training_data` volume as Parquet partitioned by `shard_id` (the AIR data-loading pattern for large Delta tables). **Overwrites the tables and the export on every run.** Creates the schema and volumes if missing.
2. **Train / register / deploy** — `train/runner.py` + `train/train.yaml` (one file: top-level AIR CLI workload schema with the demo's own configuration nested under `parameters.training_config`, the schema's designated field for custom structured config — the AIR CLI rejects unknown top-level keys): reads the SFT table, fine-tunes with Unsloth LoRA, saves rank-0 adapters to a UC volume, then (in later cells) merges the adapter, registers to UC, and creates/updates the serving endpoint. The training implementation is `train/train.py` (`run_rank_training()` trains one rank's shard slice; rank/world size are resolved launcher-agnostically); the notebook's `@distributed(gpus=N, gpu_type="h100")` cell is a thin wrapper that inserts `NOTEBOOK_DIR` into `sys.path` on each worker and calls it, and the AI Runtime CLI runs the same file standalone (`air run --file train.yaml`). The registration cell imports `load_unsloth_model` from the same module. Creates the schema and the checkpoint volume if they don't exist (the SFT table itself still comes from setup).
3. **Load test** — `load_test/load_test_serving_endpoint.py` + `load_test/serving_load_test.yaml`: samples prompts from the SFT table, smoke-tests the endpoint, then generates paced async HTTP traffic from Spark tasks via `mapInPandas` + `aiohttp`. **Appends** a summary row to the results Delta table.

### Cross-file contracts (keep these in sync)

- **Config coupling**: `catalog`, `schema`, and `sft_table` must agree across all three YAML files (the training values sit under `train.yaml`'s `parameters.training_config` key); `endpoint_name` must match between `train.yaml` and `serving_load_test.yaml`; `setup.yaml`'s `table` must equal `train.yaml`'s `source_table`; `sft_volume` must match between `setup.yaml` and `train.yaml` (it locates the Parquet export training reads). `train.yaml`'s `experiment_name` must stay alphanumeric/`-`/`_` (it becomes a Jobs API task key); AIR resolves it to `/Users/<user>/<experiment_name>` in the workspace.
- **Shard contract**: ingestion computes `shard_id` as `pmod(xxhash64(...), sft_shards)` (128 shards) and exports the SFT records as Parquet partitioned by `shard_id` into `/Volumes/<catalog>/<schema>/<sft_volume>/<sft_table>/shard_id=N/`. `run_rank_training()` claims the directories where `N % world_size == rank` and loads them with Hugging Face `datasets` (`load_dataset("parquet", ...)`) — no Spark session on the GPU workers. Sampling is two-level: for `sample_fraction < 1` only a seeded-random subset of the rank's shard directories is loaded (hash shards are uniform, so this is equivalent to row sampling) and row-level sampling within the loaded shards lands on the exact fraction — this keeps the HF `datasets` Arrow conversion ("Generating train split") proportional to the fraction instead of always materializing the full slice, per https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-volumes. Parquet was chosen because Unsloth consumes HF `datasets`, which memory-maps Parquet natively.
- **Prompt contract**: the prompt text is built once in ingestion (Spark `concat` expressions) and the assistant response is compact JSON with keys `risk`/`action`/`reason`. Training, the sample serving payload, and the load test all reuse this exact shape; if you change the prompt template in setup, the sample payload in the training notebook must match.
- **Serving contract**: registration wraps merged HF weights in an MLflow pyfunc whose `metadata` carries `task: llm/v1/chat` and a vLLM OpenAI-server `entrypoint` listening on port 8080 (the port Model Serving expects). The entrypoint launches from the model's `artifacts/` folder, so the `--model` path must be the bare artifact name — an `artifacts/` prefix makes vLLM treat it as a Hugging Face repo id and fail with a 401. Entrypoint models must be registered with `env_pack="databricks_model_serving"` (express deployment), which needs `databricks-sdk>=0.102.0` to avoid a 5-minute timeout uploading the env-pack tarball. The pyfunc placeholder class is deliberately defined inline in the registration cell so cloudpickle serializes it by value — no `code_paths` are logged, and none are needed as long as the logged model imports nothing from the repo (the vLLM entrypoint, not the pyfunc, serves traffic). The serving container builds its environment from the `pip_requirements` in `log_model`, not from `requirements.txt`; it pins `vllm==0.11.0` + `transformers>=4.56,<5` + `opencv-python-headless==4.12.0.88` (see the FIPS constraint below; transformers must be 4.x because transformers 5 removed tokenizer attributes like `all_special_tokens_extended` that this vLLM reads at startup), so the base model's architecture must be in that vLLM's supported list. The training env runs transformers 5 while the serving env runs 4.x — the saved checkpoint and tokenizer must stay loadable by both. vLLM settings (`vllm_dtype`, `vllm_max_model_len`, `vllm_gpu_memory_utilization`) come from `train.yaml`'s `parameters.training_config`.

### Ordering dependencies inside the training notebook

The registration cell requires `TRAINED_ADAPTER_OUTPUT_DIR`/`TRAINING_RUN_ID` set by the training cell; the deployment cell requires `REGISTERED_MODEL_VERSION` from registration (`register_model`/`deploy_endpoint` flags in `train.yaml`'s `parameters.training_config` gate these). Registration is deliberately separate from training so a failed registration or deployment can be rerun without re-training. Scaling is done by editing only the `gpus=` value in the `@distributed` decorator on the single training cell.

### Other constraints

- Qwen3.5 cannot currently be served via Custom LLM Serving (the reason this demo uses Qwen3): its architecture needs vLLM ≥ 0.17, but vLLM ≥ 0.15 requires `opencv-python-headless>=4.13`, whose bundled OpenSSL crashes with `FATAL FIPS SELFTEST FAILURE` on Model Serving's FIPS-enabled serverless pods (known platform issue as of June 2026). vLLM 0.11.0's opencv floor is 4.11, which is why the FIPS-safe `opencv-python-headless==4.12.0.88` pin resolves there and the serving stack works.
- The base model must be a **non-thinking** variant (hence `Qwen3-4B-Instruct-2507`, not base `Qwen3-4B`). Base Qwen3 is a hybrid reasoning model: vLLM's serving-time chat template defaults to `enable_thinking=True`, so the endpoint emits long reasoning before the JSON, truncating responses at `max_tokens` — and the training renders (which use `enable_thinking=False`) don't match the serving template, so the fine-tune can't suppress it.
- Custom LLM serving beta does not support scale-to-zero with `GPU_XLARGE` — the deploy cell raises if both are set.
- Capacity is **fixed** during beta (no autoscaling between replicas): `serving_workload_size` sets the replica count and requests beyond provisioned capacity are rejected with instant 429s (the threshold is not user-configurable). Measured on this workload: one `GPU_MEDIUM` (A10) replica sustains ~150 successful QPS at ~1.1-1.4s latency with `max_tokens: 64-128`. Size for peak: 10K QPS needs `Large` + `GPU_XLARGE`-class capacity or multiple endpoints sharded client-side. The load generator itself caps out at (cluster task slots) × (per-worker rate) — `load_generator_workers` must not exceed available Spark task slots or achieved QPS silently falls short with zero 429s.
- The serving container has no `ninja`/`nvcc`, so FlashInfer (present in the Databricks AI base env) cannot JIT-compile kernels — vLLM crashes at sampler warmup with `FileNotFoundError: 'ninja'` unless the served entity sets `VLLM_USE_FLASHINFER_SAMPLER=0` (done via `environment_vars` in the deploy cell; this is runtime config, no re-registration needed).
- Unsloth's `use_gradient_checkpointing="unsloth"` mode is reentrant and crashes multi-GPU DDP runs under `@distributed` ("Expected to mark a variable ready only once"). The training code uses HF non-reentrant checkpointing instead (`gradient_checkpointing_kwargs={"use_reentrant": False}` in `SFTConfig`, `use_gradient_checkpointing=False` in the PEFT kwargs).
- The MLflow experiment path is hardcoded in the training notebook (`/Users/ben.doan@databricks.com/unsloth_qwen3_4b_training`); update it when running as a different user.
- Only `README.md`, `CLAUDE.md`, `databricks.yml`, `setup/`, `train/`, and `load_test/` are version-controlled; `demo_script/`, `.claude/`, `.github/`, `.vscode/`, and other agent/IDE artifacts are gitignored.
