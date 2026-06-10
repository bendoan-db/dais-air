# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What This Is

A Databricks AI Runtime (AIR) demo for DAIS: fine-tune `unsloth/Qwen3-4B` with Unsloth LoRA on IBM TabFormer credit-card fraud transactions, register the merged model to Unity Catalog as a custom LLM, deploy it to Mosaic AI Model Serving (vLLM), and load test the endpoint at high QPS. The deliverables are Databricks notebooks plus YAML configs — there is no test suite, linter, or build step.

## Commands

```bash
# Local environment (only needed for Databricks Connect execution)
python -m venv .venv && source .venv/bin/activate
pip install -r air/requirements.txt

# Verify CLI auth (profile DEFAULT; workspace host is in databricks.yml)
databricks auth profiles

# Run the ingestion notebook locally via Databricks Connect (serverless)
.venv/bin/python setup/01_load_tabformer_dataset.py

# Bundle operations (Declarative Automation Bundle, target: dev)
databricks bundle validate
```

The training notebook (`air/train_qwen3_4b_unsloth.py`) **cannot run locally** — it requires Databricks Serverless GPU with the AI Runtime `AI v5` environment (`serverless_gpu.distributed`, H100/A10 GPUs). Run it as a workspace notebook. The load-test notebook runs on Databricks serverless compute.

## Databricks Notebook Source Format

Every `.py` file in `setup/` and `air/` is a Databricks notebook, not a plain module. When editing, preserve the format: `# Databricks notebook source` header, `# COMMAND ----------` cell separators, `# MAGIC %md` markdown cells.

- The `air/` notebooks load `air/utils.py` via `%run ./utils`, **not** an import. Its helpers (`load_yaml_config`, `config_str`/`config_int`/`config_float`/`config_bool`, `quote_identifier`, `full_name`, `get_spark_session`) become notebook-session globals. Do not add `from utils import ...`.
- Notebooks support dual execution: inside Databricks (global `spark`, `dbutils`) or locally via Databricks Connect. `get_notebook_dir()` and `get_spark_session()` in `utils.py` handle both paths; `setup/01_load_tabformer_dataset.py` duplicates this logic inline.

## Architecture: Three-Stage Pipeline Coupled by YAML Configs

Each stage is a notebook driven by a sibling YAML file, connected to the next stage only through Unity Catalog objects:

1. **Ingest** — `setup/01_load_tabformer_dataset.py` + `setup/setup.yaml`: downloads TabFormer into a UC staging volume, writes the cleaned transaction table (`fraud_dataset`) and the SFT table (`fraud_sft_dataset`) with `prompt`, `assistant_response`, and `shard_id` columns. **Overwrites both tables on every run.** Creates the schema and volume if missing.
2. **Train / register / deploy** — `air/train_qwen3_4b_unsloth.py` + `air/training.yaml`: reads the SFT table, fine-tunes with Unsloth LoRA inside one `@distributed(gpus=N, gpu_type="h100")` cell, saves rank-0 adapters to a UC volume, then (in later cells) merges the adapter, registers to UC, and creates/updates the serving endpoint. Does **not** create the schema/volume — run setup first.
3. **Load test** — `air/load_test_serving_endpoint.py` + `air/serving_load_test.yaml`: samples prompts from the SFT table, smoke-tests the endpoint, then generates paced async HTTP traffic from Spark tasks via `mapInPandas` + `aiohttp`. **Appends** a summary row to the results Delta table.

### Cross-file contracts (keep these in sync)

- **Config coupling**: `catalog`, `schema`, and `sft_table` must agree across all three YAML files; `endpoint_name` must match between `training.yaml` and `serving_load_test.yaml`; `setup.yaml`'s `table` must equal `training.yaml`'s `source_table`.
- **Shard contract**: ingestion computes `shard_id` as `pmod(xxhash64(...), sft_shards)` (128 shards). The distributed training function assigns rows per GPU worker with `WHERE pmod(shard_id, world_size) = rank`, so each worker reads its own Delta slice and nothing large ships from the driver.
- **Prompt contract**: the prompt text is built once in ingestion (Spark `concat` expressions) and the assistant response is compact JSON with keys `risk`/`action`/`reason`. Training, the sample serving payload, and the load test all reuse this exact shape; if you change the prompt template in setup, the sample payload in the training notebook must match.
- **Serving contract**: registration wraps merged HF weights in an MLflow pyfunc whose `metadata` carries `task: llm/v1/chat` and a vLLM OpenAI-server `entrypoint` listening on port 8080 (the port Model Serving expects). The entrypoint launches from the model's `artifacts/` folder, so the `--model` path must be the bare artifact name — an `artifacts/` prefix makes vLLM treat it as a Hugging Face repo id and fail with a 401. Entrypoint models must be registered with `env_pack="databricks_model_serving"` (express deployment), which needs `databricks-sdk>=0.102.0` to avoid a 5-minute timeout uploading the env-pack tarball. The serving container builds its environment from the `pip_requirements` in `log_model`, not from `requirements.txt`; it pins `vllm==0.15.0` with `transformers>=4.56,<5` (the newest FIPS-safe vLLM; vLLM ≤ 0.15 needs transformers 4.x because transformers 5 removed tokenizer attributes like `all_special_tokens_extended` that it reads at startup), so the base model's architecture must be in that vLLM's supported list. The training env runs transformers 5 while the serving env runs 4.x — the saved checkpoint and tokenizer must stay loadable by both. vLLM settings (`vllm_dtype`, `vllm_max_model_len`, `vllm_gpu_memory_utilization`) come from `training.yaml`.

### Ordering dependencies inside the training notebook

The registration cell requires `TRAINED_ADAPTER_OUTPUT_DIR`/`TRAINING_RUN_ID` set by the training cell; the deployment cell requires `REGISTERED_MODEL_VERSION` from registration (`register_model`/`deploy_endpoint` flags in `training.yaml` gate these). Registration is deliberately separate from training so a failed registration or deployment can be rerun without re-training. Scaling is done by editing only the `gpus=` value in the `@distributed` decorator on the single training cell.

### Other constraints

- Qwen3.5 cannot currently be served via Custom LLM Serving (the reason this demo uses Qwen3): its architecture needs vLLM ≥ 0.17, but every vLLM > 0.15 depends on `opencv-python-headless>=4.13`, whose bundled OpenSSL crashes with `FATAL FIPS SELFTEST FAILURE` on Model Serving's FIPS-enabled serverless pods. Known platform issue as of June 2026; the internal workaround (`opencv-python-headless==4.12.0.88`) conflicts with vLLM's dependency pin and cannot be expressed in `pip_requirements`.
- Custom LLM serving beta does not support scale-to-zero with `GPU_XLARGE` — the deploy cell raises if both are set.
- Unsloth's `use_gradient_checkpointing="unsloth"` mode is reentrant and crashes multi-GPU DDP runs under `@distributed` ("Expected to mark a variable ready only once"). The training code uses HF non-reentrant checkpointing instead (`gradient_checkpointing_kwargs={"use_reentrant": False}` in `SFTConfig`, `use_gradient_checkpointing=False` in the PEFT kwargs).
- The MLflow experiment path is hardcoded in the training notebook (`/Users/ben.doan@databricks.com/unsloth_qwen3_4b_training`); update it when running as a different user.
- Only `README.md`, `databricks.yml`, `setup/`, and `air/` are version-controlled; `demo_script/`, `.claude/`, `.github/`, `.vscode/`, and other agent/IDE artifacts are gitignored.
