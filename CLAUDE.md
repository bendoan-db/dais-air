# CLAUDE.md

## Repository Purpose

This repository is a Databricks AI Runtime pipeline for preparing fraud SFT
data, fine-tuning open-source LLMs, registering merged models in Unity
Catalog, deploying it through vLLM, load testing, and monitoring inference
traffic.

There is no conventional test suite or build. CI runs:

```bash
python -m compileall -q setup train load_test monitor
```

## Training Projects

Training is intentionally split into independent directories:

- `train/train_qwen_unsloth/`: Unsloth LoRA + DDP for Qwen3-4B.
- `train/train_phi_4_unsloth/`: Unsloth LoRA + DDP for Microsoft Phi-4.
- `train/train_gpt_oss_fsdp/`: TRL + PEFT + FSDP2 for GPT-OSS 120B.

Each project contains `01_runner.py`, `train.py`, `train.yaml`,
`project_config.py`, and `requirements.txt`. Its YAML owns all runtime inputs:
catalog, schema, experiment, compute, model-weight path, prepared train/eval
paths, output path, and trainer parameters. Trainers must not load
`global.yaml`, `setup.yaml`, a pipeline stage's `utils.py`, or another
project's config.

Both launch paths call the same project-local `run_rank_training()`:

```bash
cd train/train_qwen_unsloth && COPYFILE_DISABLE=1 air run --file train.yaml --watch
cd train/train_phi_4_unsloth && COPYFILE_DISABLE=1 air run --file train.yaml --watch
cd train/train_gpt_oss_fsdp && COPYFILE_DISABLE=1 air run --file train.yaml --watch
```

The workload snapshot must remain rooted at `.` and execute
`$CODE_SOURCE_PATH/train.py`. `$HYPERPARAMETERS_PATH` may contain either the
full workload or only `parameters`; project loaders support both shapes.

## Data and Artifact Contracts

`train_data_path` and `eval_data_path` point directly at separate directories
containing `shard_id=N/*.parquet`. Each rank claims shards where
`N % world_size == rank`; no Spark session runs on GPU workers.

`convert_sft: false` requires `prompt` and `assistant_response` columns.
`convert_sft: true` requires the raw fraud columns defined in each project's
`sft_conversion.py` and converts each rank's loaded sample once before trainer
construction. Keep the three project-local converters and setup's shared prompt
contract synchronized when changing the worked example.

`ignore_partitions: false` preserves rank-to-shard assignment.
`ignore_partitions: true` recursively enumerates all Parquet files for every
rank; row sampling happens only after full materialization. Use it only when
the entire split fits comfortably in each rank's host memory.

Model and adapter files under `/Volumes` must be staged to node-local disk
before loading because safetensors mmap reads are slow through FUSE. Trainer
outputs are written locally first and copied sequentially to the configured UC
output directory. Saved adapter metadata must restore the durable
`model_weights_path`, never the ephemeral staged path.

Unsloth custom gradient checkpointing is reentrant and unsafe under DDP here.
Keep HF non-reentrant checkpointing (`use_reentrant=False`). FSDP saves are
collective; every rank must call `trainer.save_model`.

## Setup and Shared Helpers

`setup/01_load_dataset.py` ingests TabFormer. `setup/02_stage_training_data.py`
creates deterministic raw train/eval shards. `setup/03_prepare_sft.py` renders
model-agnostic prompts/responses. `setup/04_download_base_model_weights.py`
optionally snapshots configured models.

Setup, load test, and monitoring each own a plain, import-safe `utils.py` in
their stage directory. Keep only functions used by that stage. The setup copy
owns the canonical fraud prompt/response renderers; keep the load-test and
trainer copies synchronized with it. Trainer projects must not import these
top-level `utils` modules because GPU packages may register a conflicting
module with that name.

## Notebook Format

Preserve `# Databricks notebook source`, `# COMMAND ----------`, and
`# MAGIC` markdown cells in notebook files. Plain modules include each
project's `train.py`/`project_config.py`/`sft_conversion.py`, each pipeline
stage's `utils.py`, `monitor/monitoring_utils.py`, and
`monitor/example_inputs.py`.

Workspace notebooks normally use their folder as `Path.cwd()`. Setup notebooks
also support local Databricks Connect and resolve their directory through
`__file__` with a `dbutils` context fallback.

## Deployment Constraints

Each training project owns `02_register_and_deploy.py` and the corresponding
`deploy_config` in its `train.yaml`. It selects a FINISHED rank-zero run that
logged `adapter_output_dir`, uses the project trainer's
`load_model_for_merge`, and deploys a custom `llm/v1/chat` vLLM entrypoint.
Qwen and Phi-4 merge with Unsloth; GPT-OSS merges through PEFT and requires 120B-scale
registration capacity.

Keep these serving requirements unless the platform constraints are retested:

- `transformers==4.57.6`
- `vllm==0.11.2`
- `mlflow==3.12.0`
- `openai==2.17.0`
- `databricks-sdk>=0.102.0`
- `opencv-python-headless==4.12.0.88`
- `VLLM_USE_FLASHINFER_SAMPLER=0`

The entrypoint listens on port 8080 and receives the bare MLflow artifact name,
not an `artifacts/`-prefixed path. Custom LLM registration uses
`env_pack="databricks_model_serving"`. During beta, `GPU_XLARGE` requires
enrollment, is AWS `us-west-2` only, and cannot use scale-to-zero.
Qwen3.5 is not supported by the pinned FIPS-safe serving stack; the worked
example uses the non-thinking Qwen3 Instruct variant.

## Configuration Ownership

`global.yaml` still owns catalog/schema for setup, load test, and monitor.
Training YAMLs intentionally duplicate those values so each project is
self-contained. Each YAML owns its deployment settings. Keep the load-test and
monitor endpoint contracts aligned with the Qwen deployment they currently
target.
