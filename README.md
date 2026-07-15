# AI Runtime End to End

A Databricks AI Runtime pipeline for preparing supervised fine-tuning data,
training open-source LLMs, registering a merged model in Unity Catalog,
deploying it with vLLM, load testing, and monitoring serving traffic.

The worked example classifies IBM TabFormer credit-card transactions with a
fine-tuned Qwen3 model. Training is split into three independent projects:

- `train/train_qwen_unsloth/`: Qwen3-4B with Unsloth LoRA and DDP.
- `train/train_phi_4_unsloth/`: Microsoft Phi-4 with Unsloth LoRA and DDP.
- `train/train_gpt_oss_fsdp/`: GPT-OSS 120B with TRL, PEFT, and FSDP2.

Each project owns its runner notebook, trainer, requirements, workload YAML,
and config loader. Its YAML contains the catalog, schema, model-weight path,
prepared train/eval paths, output path, compute, explicit MLflow experiment
path, and trainer parameters. No trainer reads `setup.yaml` or another
training project.

## Project Layout

| Path | Purpose |
| --- | --- |
| `setup/01_load_dataset.py` | Download and clean TabFormer transactions. |
| `setup/02_stage_training_data.py` | Create deterministic train/eval splits and raw Parquet shards. |
| `setup/04_download_base_model_weights.py` | Snapshot configured Hugging Face models into UC volumes. |
| `setup/utils.py` | Setup config, catalog, and Spark helpers. |
| `train/prep_sft.py` | Convert raw Parquet shards into prepared SFT Parquet in a separate UC volume. |
| `train/train_qwen_unsloth/` | Standalone Qwen training and deployment project. |
| `train/train_phi_4_unsloth/` | Standalone Phi-4 training and deployment project. |
| `train/train_gpt_oss_fsdp/` | Standalone GPT-OSS FSDP training and deployment project. |
| `load_test/` | Paced asynchronous serving load test with stage-local `utils.py`. |
| `monitor/01_unpack_inference_table.py` | Incrementally unpack AI Gateway inference payloads. |
| `monitor/02_create_quality_monitor.py` | Build the training baseline and data quality monitor. |
| `monitor/03_create_drift_sql_alert.py` | Provision the scheduled baseline-drift SQL alert. |
| `monitor/04_trigger_retraining.py` | Trigger a configured retraining job after drift breaches. |

## Prerequisites

- A Databricks workspace with Unity Catalog, serverless compute, and Serverless
  GPU AI Runtime enabled.
- Catalog privileges to create schemas, volumes, tables, and models.
- GPU Model Serving enabled for custom LLM serving.
- Databricks CLI authentication for bundle, Connect, or AIR CLI commands.

## Configuration

Edit the selected project's `train.yaml` before training. The important paths
under `parameters.training_config` are:

- `model_weights_path`: a populated `/Volumes/...` model snapshot.
- `train_data_path`: the training split's Parquet root.
- `eval_data_path`: the separate held-out split's Parquet root.
- `convert_sft`: `false` for rows that already contain `prompt` and
  `assistant_response`; `true` to render those fields from raw fraud records
  once per rank inside the trainer.
- `ignore_partitions`: `false` assigns `shard_id=N` directories by rank;
  `true` recursively loads every Parquet file under the split path on every
  rank. Any `training_sample_fraction` is applied only after those files load.
- `suspicious_amount_threshold`: labeling threshold used only by inline fraud
  conversion.
- `output_dir`: the UC volume directory that receives adapters.
- `experiment_path`: the absolute workspace path used by notebook training and
  deployment run selection.

The workload-level `compute` block sizes both the notebook's `@distributed`
call and the AIR CLI run. AIR still requires its workload-level
`experiment_name` task key; `mlflow_experiment_directory` and that key must
resolve to `training_config.experiment_path`. Each project YAML also owns
`deploy_config` for its local `02_register_and_deploy.py` notebook.

`setup/setup.yaml` separately controls the worked-example data pipeline and
optional model downloads. The GPT-OSS model path is not downloaded by default
because the snapshot is very large; populate that configured path separately.
Setup, load testing, and monitoring each define their own `catalog` and
`schema` at the top of their sibling YAML; there is no repository-wide config.

Compile-check the local Python sources without a workspace connection:

```bash
python -m compileall -q setup train load_test monitor
```

## End-to-End Workflow

1. Run `setup/01_load_dataset.py` on Databricks serverless compute.
2. Run `setup/02_stage_training_data.py` to create raw train/eval shards.
3. With `convert_sft: false`, run `train/prep_sft.py` to write `prompt`,
   `assistant_response`, and `is_fraud` records. With `convert_sft: true`, skip
   this step and point the project paths at setup/02's raw split directories.
4. Run `setup/04_download_base_model_weights.py` when the selected model is
   listed in `setup.yaml`, or populate the project's `model_weights_path`
   through another controlled process.
5. Run one training project's `01_runner.py` on Serverless GPU with AI v5, or
   submit that directory's workload through the AIR CLI.
6. Run the selected project's `02_register_and_deploy.py` to merge its
   adapter, register it, and update its serving endpoint.
7. Run `load_test/load_test_serving_endpoint.py`, then monitoring notebooks
   01-03. Schedule notebook 04 after each quality-monitor refresh.

The setup stages overwrite their data and exports. Load-test results append.
Inference unpacking is incremental through its Structured Streaming checkpoint.

## Running Training

Notebook entrypoints:

- `train/train_qwen_unsloth/01_runner.py`
- `train/train_phi_4_unsloth/01_runner.py`
- `train/train_gpt_oss_fsdp/01_runner.py`

AIR CLI entrypoints:

```bash
cd train/train_qwen_unsloth
COPYFILE_DISABLE=1 air run --file train.yaml --watch

cd ../train_phi_4_unsloth
COPYFILE_DISABLE=1 air run --file train.yaml --watch

cd ../train_gpt_oss_fsdp
COPYFILE_DISABLE=1 air run --file train.yaml --watch
```

Use `--override parameters.training_config.max_steps=50` for per-run changes.
`COPYFILE_DISABLE=1` prevents macOS AppleDouble files from corrupting the AIR
snapshot's resolved code path.

Both launch paths call the same project-local `run_rank_training()` function.
By default, each rank claims `shard_id=N` directories where
`N % world_size == rank`. With `ignore_partitions: true`, every rank loads the
complete file set and the Trainer's distributed sampler handles batches. This
mode supports unpartitioned inputs but multiplies CPU memory and I/O by the
world size. GPU workers require no Spark session in either mode.

## Data Contract

With `ignore_partitions: false`, each configured path must contain this layout:

```text
<split-path>/
  shard_id=0/*.parquet
  shard_id=1/*.parquet
  ...
```

With `ignore_partitions: true`, files may use any nested layout; all
`*.parquet` files below each configured split path are loaded recursively.

With `convert_sft: false`, Parquet records must include string columns `prompt`
and `assistant_response`. Qwen evaluation also expects `is_fraud`.

With `convert_sft: true`, records must instead contain the raw fraud fields:
`user_id_text`, `card_id_text`, `transaction_ts_text`, `amount_usd`,
`use_chip_text`, `merchant_city_text`, `merchant_state_text`, `mcc_text`,
`errors_text`, `is_fraud`, and `has_error_signal`. Point the paths at the raw
export from `setup/02_stage_training_data.py`; each rank converts its loaded
sample once before constructing the Hugging Face/TRL dataset.

The setup pipeline creates 128 deterministic hash shards by default; keep the
shard count divisible by every intended GPU count.

For custom data, use pre-converted SFT rows or replace the project-local
`sft_conversion.py` with a renderer for the raw schema. The projects do not
require the worked-example setup notebooks.

## Registration and Serving

Training rank zero logs `adapter_output_dir`; each project's deployment
notebook selects an explicit `run_id` or the best finished run in that
project's experiment. Qwen and Phi-4 merge through Unsloth. GPT-OSS loads the saved PEFT
adapter with `AutoPeftModelForCausalLM` before merging; its 120B packaging step
requires correspondingly large GPU, host-memory, and local-disk capacity.
All three register a custom `llm/v1/chat` model, deploy the vLLM entrypoint, and
enable AI Gateway inference logging.

Each project's serving environment is its local `requirements.txt`. The
`transformers==4.57.6`, `vllm==0.11.2`, `mlflow==3.12.0`, and
`opencv-python-headless==4.12.0.88` pins are required for the current
Databricks custom LLM serving environment. Do not replace them without
rechecking the target model architecture and serving image.

## Local Development

The data setup and SFT preparation notebooks can run through Databricks
Connect after installing their lightweight local dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install "databricks-connect>=17.0.0" "pyyaml>=6.0.2" "pandas>=2.2.0"
.venv/bin/python setup/01_load_dataset.py
.venv/bin/python setup/02_stage_training_data.py
.venv/bin/python train/prep_sft.py
```

The training runners require Databricks Serverless GPU and cannot run locally.
The model-download notebook requires the workspace `/Volumes` FUSE mount.
