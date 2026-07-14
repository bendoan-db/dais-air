# AI Runtime End to End

A deployable Databricks AI Runtime (AIR) pipeline for fine-tuning an open-source language model, registering it to Unity Catalog as a custom LLM, deploying it to Mosaic AI Model Serving (vLLM), and load testing the deployed endpoint.

The repository ships with a complete worked example — classifying IBM TabFormer credit-card transactions for fraud with a fine-tuned Qwen3-4B — that exercises every stage end to end. To adapt it, point the configuration at your workspace and swap in your own dataset and base model (see [Bring Your Own Data and Model](#bring-your-own-data-and-model)).

## Project Layout

| Path | Purpose |
| --- | --- |
| `global.yaml` | **Global configuration** — the Unity Catalog identity (`catalog`, `schema`) shared by every module. |
| `setup/01_load_dataset.py` | Databricks notebook that downloads TabFormer, cleans transaction data, and overwrites the transaction Delta table. |
| `setup/02_stage_training_data.py` | Databricks notebook that builds the prompt/response SFT Delta table and stages it in a Unity Catalog volume as Parquet shards for AIR training. |
| `setup/03_download_base_model_weights.py` | Databricks notebook that snapshots the Hugging Face models listed in `setup/setup.yaml` into Unity Catalog volumes. |
| `setup/setup.yaml` | Stage-specific setup configuration: dataset source URL, staging volume, SFT shard settings, and the Hugging Face models to mirror into volumes. |
| `train/01_runner.py` | Databricks notebook for AIR fine-tuning with Unsloth and MLflow experiment logging. |
| `train/02_register_and_deploy.py` | Databricks notebook that selects a training run (explicit `run_id` or best run by metric), merges its adapter, registers the model to Unity Catalog, and deploys the serving endpoint with inference logging. |
| `train/train.py` | Standalone training module: imported by the notebook's `@distributed` cell and runnable directly via the AI Runtime CLI. |
| `train/train.yaml` | AI Runtime CLI workload definition (`air run --file train.yaml`) plus the stage-specific configuration: training settings (`parameters.training_config`) and registration/serving settings (`parameters.deploy_config`). |
| `load_test/load_test_serving_endpoint.py` | Databricks notebook that simulates high-QPS traffic against the deployed serving endpoint. |
| `load_test/serving_load_test.yaml` | Stage-specific load-test configuration: request rate, workers/concurrency, and the results table. |
| `monitor/01_unpack_inference_table.py` | Databricks notebook that incrementally unpacks the endpoint's inference table (raw JSON payloads) into an analysis-ready Delta table for monitoring. |
| `monitor/monitor.yaml` | Stage-specific monitoring configuration: unpacked table name, checkpoint volume, and structured-output field extraction. |
| `train/training_utils.py` | Shared utilities: YAML/config loading, Unity Catalog name handling, and model staging. |
| `train/requirements.txt` | Consolidated dependencies for the stage: installed by the notebooks, referenced by `train.yaml`'s environment (`-r requirements.txt`), and reused as the serving container's environment (`deploy_config`'s `serving_requirements_file`). `transformers` is pinned `<5` so the one file satisfies both training and the FIPS-safe vLLM serving stack. |
| `scripts/validate_config.py` | Offline validator for the cross-file configuration contracts (also run by CI). |
| `databricks.yml` | Databricks bundle definition, including a serverless job that runs the setup notebooks. |
| `extras/` | Experimental, unsupported training variants (Ray Train, custom launch configs) kept for reference. |

## Prerequisites

- Databricks workspace with Unity Catalog enabled.
- A Unity Catalog catalog that already exists.
- Unity Catalog privileges on that catalog: `USE CATALOG` plus `CREATE SCHEMA`, `CREATE VOLUME`, `CREATE TABLE`, and `CREATE MODEL` (or pre-created objects to point the config at), and permission to create serving endpoints.
- Databricks serverless compute for ingestion and load testing.
- Databricks Serverless GPU with AI Runtime enabled for training. Check accelerator availability in your region — a single A10 suffices for the 4B example; H100s are needed only for scaled multi-GPU runs.
- Model Serving with GPU workloads enabled for custom LLM serving (beta — confirm your workspace is enrolled).
- Local Databricks CLI authentication if running notebooks or scripts from this repository with Databricks Connect.

## Configuration

The repo-root **`global.yaml`** holds the Unity Catalog identity every module shares — `catalog` and `schema` — loaded by every notebook through `training_utils.load_global_config()`, so the one thing that must never diverge is configured exactly once (`scripts/validate_config.py` fails if a stage YAML re-introduces either key). Everything else is stage-owned in each module's YAML; the values shared between stages (table/volume names, `endpoint_name`, the inference-table name) are duplicated by design and every agreement is machine-checked by `scripts/validate_config.py`.

Update before running:

- `train/train.yaml` (`parameters.training_config` + `parameters.deploy_config`; the top-level fields configure the AI Runtime CLI workload)
  - `catalog`, `schema`, `source_table`, `sft_table`, and `sft_volume` (shared with the other stages), plus `checkpoint_volume` and `uc_model_name`
  - `model_name` and `model_volume_path` (the base model and its optional volume snapshot)
  - training parameters: `max_steps`, batch size, learning rate, `training_sample_fraction`, and the LoRA settings (`lora_r`, `lora_alpha`, `lora_dropout`, `lora_target_modules`)
  - `response_instruction_part` / `response_part` — chat-template markers that must match the base model's template
  - `notebook_gpus` / `notebook_gpu_type` — compute for the notebook training cell (the AIR CLI path uses the top-level `compute` block)
  - deployment (`deploy_config` section): `run_id` (empty = auto-select) with `best_run_metric` / `best_run_metric_goal`, registration names (`uc_model_name`, `served_model_name`), vLLM settings, endpoint sizing (`serving_workload_type`, `serving_provisioned_concurrency`, `serving_scale_to_zero`), `serving_requirements_file` (the serving container's pinned environment), `inference_table_prefix` (inference logging is always enabled at deployment), and `endpoint_name`
- `setup/setup.yaml` — `catalog`, `schema`, `table`, `sft_table`, and `sft_volume` (shared with `train.yaml`), plus stage-specific keys: dataset source URL and staging volume, SFT shard settings (`sft_shards`, `sft_shard_key_columns`), the Hugging Face `models` list to snapshot into volumes, and an optional secret reference for gated-model tokens
- `load_test/serving_load_test.yaml` — `catalog`, `schema`, `sft_table` (shared), and `endpoint_name` (must match `train.yaml`'s `deploy_config`), plus load-generator settings: `target_qps`, `duration_seconds`, worker/concurrency settings, and the results table name
- `monitor/monitor.yaml` — `catalog` / `schema` (shared) and `inference_table` (must equal `deploy_config`'s `inference_table_prefix` + `_payload`), plus `unpacked_table`, `checkpoint_volume`, and `response_json_fields` for structured-output extraction

After editing, validate the cross-file contracts locally (no workspace connection needed):

```bash
python scripts/validate_config.py
```

## End-to-End Workflow

1. Ingest the dataset.

   Run `setup/01_load_dataset.py` on Databricks serverless compute. The notebook:

   - Creates the configured schema and staging volume if they do not exist.
   - Downloads and extracts the IBM TabFormer transactions archive.
   - Standardizes transaction columns and data types.
   - Adds prompt-ready transaction fields and fraud labels.
   - Overwrites the cleaned transaction Delta table on each run.

2. Stage the training data.

   Run `setup/02_stage_training_data.py` on Databricks serverless compute. The notebook:

   - Builds prompt/response SFT records with stable `shard_id` columns from the transaction table, using Spark expressions (see [How the Data Sharding Works](#how-the-data-sharding-works)).
   - Overwrites the SFT Delta table on each run.
   - Exports the SFT records to a Unity Catalog volume as Parquet files partitioned by `shard_id`, per the [AI Runtime data-loading guidance](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-volumes).
   - Verifies the export by reading it back: shard count and row count must match the SFT table.

3. Download the base model weights (optional but recommended).

   Run `setup/03_download_base_model_weights.py` as a Databricks workspace notebook. It snapshots each Hugging Face model listed under `models:` in `setup/setup.yaml` into its configured Unity Catalog volume path, so training loads workspace-local weights instead of downloading from Hugging Face on every GPU worker. Point `train/train.yaml`'s `model_volume_path` at the snapshot the fine-tune should load (leave it empty to download from Hugging Face at training time).

4. Fine-tune with AI Runtime.

   Run `train/01_runner.py` on Databricks Serverless GPU with AI Runtime. The notebook:

   - Installs `train/requirements.txt`.
   - Reads the rank-sharded SFT Parquet files from the Unity Catalog volume with Hugging Face `datasets` (no Spark on the GPU workers).
   - Fine-tunes `unsloth/Qwen3-4B-Instruct-2507` with Unsloth LoRA.
   - Uses the `@distributed` decorator so the same training cell can run on one GPU or multiple GPUs by changing the `gpus` parameter.
   - Saves rank-0 adapter artifacts to a Unity Catalog volume.
   - Logs training metrics to MLflow, including the adapter location (`adapter_output_dir`) the deployment stage resolves.
   - Scores the fine-tuned model as a binary fraud classifier on a stratified holdout (excluded from training) and logs `eval_fraud_accuracy`, `eval_fraud_precision`, `eval_fraud_recall`, and `eval_fraud_f1` — set `best_run_metric: eval_fraud_f1` (maximize) in `deploy_config` to deploy the best classifier instead of the lowest loss.

   The training implementation lives in `train/train.py` and can also run without the notebook through the AI Runtime CLI — see [Training via the AI Runtime CLI](#training-via-the-ai-runtime-cli).

5. Register and deploy the model.

   Run `train/02_register_and_deploy.py` on Databricks Serverless GPU with AI Runtime. The notebook:

   - Selects the training run to deploy: the `run_id` from `train.yaml`'s `deploy_config`, or — when `run_id` is empty — the best FINISHED run in the configured experiment by `best_run_metric` / `best_run_metric_goal`.
   - Resolves the adapter location from the run's `adapter_output_dir` parameter and merges the adapter into the base model.
   - Packages the merged weights with a vLLM OpenAI-compatible server entrypoint (`llm/v1/chat`) and registers the model to Unity Catalog using the Databricks Model Serving environment pack.
   - Creates or updates the configured Model Serving endpoint and routes 100% of traffic to the new version.
   - Always enables AI Gateway inference logging as part of the deployment, so every request/response lands in the payload table the monitoring stage consumes.

6. Load test the endpoint.

   Run `load_test/load_test_serving_endpoint.py` after the endpoint is ready. The notebook:

   - Samples prompts from the SFT Delta table.
   - Runs a smoke test against the endpoint.
   - Generates asynchronous HTTP traffic from Spark tasks.
   - Records achieved throughput, status counts, latency samples, and summary metrics to a Delta table.

7. Monitor the endpoint.

   Run `monitor/01_unpack_inference_table.py` on Databricks serverless compute — on demand or as a scheduled job. The notebook incrementally unpacks the endpoint's inference table (raw JSON request/response payloads captured by AI Gateway inference logging) into an analysis-ready Delta table with prompts, completions, token usage, latency, and any configured structured-output fields, created with change data feed enabled for downstream data-quality monitoring. Logs are delivered to the payload table within about an hour of endpoint traffic. (A follow-up module creates the data-quality monitor over the unpacked table.)

## Bring Your Own Data and Model

The TabFormer ingestion (`setup/01`) and the fraud prompt construction in `setup/02` are the example half of the pipeline. Everything downstream depends only on two contracts:

**The SFT table contract.** Training consumes a Delta table with two string columns — `prompt` and `assistant_response` — plus the `shard_id` column added at staging. To train on your own data:

1. Produce a Delta table with `prompt`/`assistant_response` columns in the configured catalog and schema: either replace `setup/01` and the record-building cells of `setup/02` with your own logic, or point `sft_table` (in `setup/setup.yaml`, `train/train.yaml`, `load_test/serving_load_test.yaml`, and `monitor/monitor.yaml`) at a table you already maintain and keep only `setup/02`'s generic half (shard assignment, Parquet export, verification).
2. Set `sft_shard_key_columns` in `setup/setup.yaml` to columns that identify your rows — they drive the deterministic shard hash (see [How the Data Sharding Works](#how-the-data-sharding-works)).
3. Keep the response shape you fine-tune on identical to what clients will request at serving time; the serving smoke-test payload and the load test both sample prompts directly from the SFT table.

**The model contract.** To fine-tune a different open-source model:

1. Change `model_name` in `train/train.yaml`'s `training_config`, add a matching entry to `setup/setup.yaml`'s `models` list, point `model_volume_path` at its `volume_path`, and rerun `setup/03` (set the HF token secret keys in `setup.yaml` for gated models such as Llama).
2. Update `response_instruction_part` / `response_part` in `train/train.yaml` to the new model's chat-template markers — response-only loss masking silently degrades if these don't match the template.
3. Confirm the model's architecture is supported by the vLLM version pinned in `train/requirements.txt` (the registration notebook prints the architecture as a preflight), and prefer a non-reasoning variant — see `CLAUDE.md`'s serving constraints for why.

## How the Data Sharding Works

`setup/02_stage_training_data.py` exports the SFT Delta table to a Unity Catalog volume as Parquet files partitioned into 128 `shard_id=N/` directories (`sft_shards` in `setup/setup.yaml`), where each row's shard is a stable hash of its key columns (`sft_shard_key_columns`; the example uses the transaction fields):

```sql
shard_id = pmod(xxhash64(user_id_text, card_id_text, transaction_ts_text, amount_usd,
                         merchant_city_text, merchant_state_text, mcc_text), 128)
```

This layout is the contract between setup and training: setup promises 128 uniform, stable, directory-addressable slices of the dataset, and `run_rank_training()` in `train/train.py` turns `(rank, world_size)` into a file list against that promise.

### The problem it solves

Training runs as DDP (data parallel): every GPU holds a full model replica, trains on different rows, and the gradients are averaged each step. That only speeds anything up if the ranks read **disjoint** data — two GPUs training on the same examples burn compute on duplicates. Normally a `DistributedSampler` handles the split, but that assumes every worker can see and index the whole dataset. On AI Runtime the GPU workers have no Spark session, and pulling the full multi-GB export into every worker just to keep 1/Nth of it would be wasted I/O.

The shard layout replaces the coordinator. Each rank independently claims the directories where `N % world_size == rank`:

```python
rank_shard_dirs = [
    shard_dir for shard_dir in shard_dirs
    if int(shard_dir.name.split("=", 1)[1]) % world_size == rank
]
```

With 8 GPUs, rank 3 reads `shard_id ∈ {3, 11, 19, ..., 123}` — 16 directories. Every shard maps to exactly one rank, so the slices are disjoint and complete by construction: no communication between workers, no shared sampler state, and no scan over rows a rank won't train on. Because `partitionBy("shard_id")` encodes shard membership in the directory structure, claiming a slice is a filesystem glob rather than a filtered read of the data.

### Why 128 shards instead of one per GPU

The export is written once, at setup time, before anyone knows how many GPUs training will use. 128 divides evenly by 1, 2, 4, 8, and 16, so the same export serves the single-GPU validation run and the multi-GPU demo run — scaling is just the `gpus=` value in the `@distributed` decorator, with no data re-prep. Exporting exactly `world_size` pieces instead would force a re-export on every change in GPU count.

### Why hash-based assignment

- **Uniform and label-agnostic.** Hashing transaction fields is effectively random with respect to fraud labels, merchants, and amounts, so every shard is an unbiased ~1/128 sample of the full dataset. Each rank gets the same class balance and the same amount of work, keeping the synchronized DDP steps free of stragglers.
- **Deterministic.** A row's shard is a pure function of its values — not of Spark partitioning, task order, or a random seed at write time — so rerunning ingestion reproduces the same assignment and training runs stay comparable across data refreshes.

### The sampling bonus

Because each shard is statistically interchangeable with any other, loading a random subset of shard directories is equivalent to row-level sampling. Training exploits this with two-level sampling: for `training_sample_fraction < 1`, each rank loads only a seeded-random subset of its shard directories, then row-samples within them to land on the exact fraction. Hugging Face `datasets`' Arrow conversion cost scales with the bytes loaded, so a 0.1% demo run pays roughly 0.1% of the load cost instead of materializing the rank's full slice and discarding the rest.

### When you would not need this

Sharding earns its keep past two independent thresholds: **multiple GPUs** (someone must divide the data) and **data too large to fully materialize per worker** (the [AI Runtime data-loading guidance](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/dataloading#load-large-delta-tables-using-volumes) threshold for exporting large Delta tables to volumes). A single-GPU run on a small Hub-hosted dataset needs neither — one process already owns all the data, and loading it costs seconds — which is why simpler examples skip partitioning entirely. This demo sits past both thresholds at once (a ~24M-row SFT table with fat prompt strings, trained with multi-GPU DDP), which is why the export/shard contract exists.

## Training via the AI Runtime CLI

The same training code that the notebook's `@distributed` cell runs can be submitted from a laptop with the [AI Runtime CLI](https://docs.databricks.com/aws/en/machine-learning/ai-runtime/cli/), without opening a notebook. `train/train.yaml` is the single configuration file for both paths: its top-level fields define the CLI workload (experiment, environment, compute, code snapshot, command) and its `parameters.training_config` section holds the demo's own training/registration/serving settings.

1. Install the CLI (requires Python 3.10+ and [uv](https://docs.astral.sh/uv/)):

   ```bash
   uv tool install --force databricks-air --python 3.12
   air --version
   ```

2. Authenticate. The CLI reuses Databricks CLI profiles from `~/.databrickscfg`:

   ```bash
   databricks auth login --host https://<your-workspace>.cloud.databricks.com
   ```

3. Submit the training workload (run the setup notebooks first — training reads the Parquet shard export `setup/02_stage_training_data.py` produces):

   ```bash
   cd train && COPYFILE_DISABLE=1 air run --file train.yaml --watch -p <profile>
   ```

   `--watch` streams the job state and node logs until the run finishes. Validate the file without submitting using `--dry-run`, and override config values per run without editing the file, for example:

   ```bash
   COPYFILE_DISABLE=1 air run --file train.yaml \
     --override parameters.training_config.max_steps=50 --watch
   ```

   `COPYFILE_DISABLE=1` is required on macOS: without it, bsdtar embeds AppleDouble (`._*`) metadata entries in the code-snapshot tarball, the remote launcher resolves the code directory from the archive's first entry, and the job fails before user code with `can't open file '/databricks/code_source/._<snapshot-dir>/train/train.py'` (the snapshot roots at the repo so `global.yaml` ships with the code). Two related CLI workarounds are already baked into this repo: `$HYPERPARAMETERS_PATH` shape handling in `training_utils.py`, and the `DATABRICKS_RUNTIME_VERSION` entry under `env_variables` in `train.yaml`.

4. Monitor and manage runs:

   ```bash
   air list runs --limit 10        # recent runs (--active for running only)
   air get run <run-id>            # status and configuration for one run
   air logs <run-id>               # stream logs (defaults to node 0)
   air cancel <run-id>             # stop a run (do this on failures — max_retries
                                   # otherwise reruns the same broken workload)
   ```

Runs land in the same MLflow experiment as notebook runs (AIR resolves `experiment_name` to `/Users/<you>/<experiment_name>`), with two markers distinguishing the launch path: the run name carries an `-air-cli` suffix and the run is tagged `submitted_via: air-cli` (notebook runs are tagged `submitted_via: notebook`). Filter with `tags.submitted_via = 'air-cli'` in the MLflow UI.

To scale up, edit `compute` in `train.yaml` (for example `num_accelerators: 8` with `accelerator_type: GPU_8xH100`) — `train.py` resolves rank and world size from the runtime, and each rank loads only its own `shard_id` directories from the Parquet export. The CLI path runs training only; model registration and endpoint deployment live in `train/02_register_and_deploy.py`, which resolves the adapter from the MLflow run — CLI runs land in the same experiment, so best-run auto-selection covers them too.

## Local Development

Create and activate a virtual environment if needed:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install local dependencies:

```bash
.venv/bin/python -m pip install -r train/requirements.txt
```

The ingestion and training-data staging notebooks can run with Databricks Connect when authentication is configured:

```bash
databricks auth profiles
.venv/bin/python setup/01_load_dataset.py
.venv/bin/python setup/02_stage_training_data.py
```

Validate the configuration contracts at any time without a workspace connection:

```bash
python scripts/validate_config.py
```

The training notebook is intended to run on Databricks Serverless GPU because it depends on AI Runtime, GPU hardware, and the `serverless_gpu` distributed runtime.

## References

- Databricks AI Runtime: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/
- Databricks custom LLM serving: https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms
- IBM TabFormer dataset: https://github.com/IBM/TabFormer
