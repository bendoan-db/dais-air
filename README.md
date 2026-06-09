# DAIS AI Runtime Demo

This project contains a Databricks AI Runtime demo for fine-tuning a small language model on credit-card fraud transactions, registering it as a custom LLM, deploying it to Mosaic AI Model Serving, and load testing the deployed endpoint.

The demo uses the IBM TabFormer credit-card dataset and prepares a supervised fine-tuning table where each row contains a transaction prompt and target assistant response.

## Project Layout

| Path | Purpose |
| --- | --- |
| `setup/01_load_tabformer_dataset.py` | Databricks notebook that downloads TabFormer, cleans transaction data, and overwrites Delta tables. |
| `setup/setup.yaml` | Ingestion configuration: catalog, schema, table names, staging volume, source URL, and SFT shard count. |
| `air/train_qwen3_4b_unsloth.py` | Databricks notebook for AIR fine-tuning with Unsloth, MLflow registration, and Model Serving deployment. |
| `air/training.yaml` | Training, registration, and serving configuration. |
| `air/load_test_serving_endpoint.py` | Databricks notebook that simulates high-QPS traffic against the deployed serving endpoint. |
| `air/serving_load_test.yaml` | Load-test configuration. |
| `air/utils.py` | Shared notebook utilities for YAML config loading and Unity Catalog name handling. |
| `air/requirements.txt` | Python dependencies used by the AIR training notebook. |
| `databricks.yml` | Databricks bundle metadata used by the Databricks extension/CLI. |
| `demo_script/` | Demo script materials. |

## Prerequisites

- Databricks workspace with Unity Catalog enabled.
- A Unity Catalog catalog that already exists.
- Permission to create schemas, volumes, tables, registered models, and serving endpoints in the target catalog/schema.
- Databricks serverless compute for ingestion and load testing.
- Databricks Serverless GPU with AI Runtime for training.
- Model Serving access with GPU workloads enabled for custom LLM serving.
- Local Databricks CLI authentication if running notebooks or scripts from this repository with Databricks Connect.

## Configuration

Update these files before running the demo:

- `setup/setup.yaml`
  - `catalog` and `schema`
  - `table` and `sft_table`
  - `staging_volume`
  - `source_url`

- `air/training.yaml`
  - `catalog`, `schema`, `source_table`, and `sft_table`
  - `checkpoint_volume`
  - `uc_model_name`
  - `endpoint_name`
  - training parameters such as `max_steps`, batch size, and learning rate
  - serving parameters such as `serving_workload_type`, `serving_workload_size`, and `serving_scale_to_zero`

- `air/serving_load_test.yaml`
  - `endpoint_name`
  - `target_qps`
  - `duration_seconds`
  - load-generator worker and concurrency settings

## Demo Flow

1. Ingest and prepare the dataset.

   Run `setup/01_load_tabformer_dataset.py` on Databricks serverless compute. The notebook:

   - Creates the configured schema if it does not exist.
   - Creates the configured staging volume if it does not exist.
   - Downloads and extracts the IBM TabFormer transactions archive.
   - Standardizes transaction columns and data types.
   - Adds prompt-ready transaction fields and fraud labels.
   - Writes the cleaned transaction Delta table.
   - Writes the prepared SFT Delta table with prompt, response, and shard columns.
   - Overwrites target tables on each run.

2. Fine-tune with AI Runtime.

   Run `air/train_qwen3_4b_unsloth.py` on Databricks Serverless GPU with AI Runtime. The notebook:

   - Installs `air/requirements.txt`.
   - Reads the prepared SFT Delta table.
   - Fine-tunes `unsloth/Qwen3-4B` with Unsloth LoRA.
   - Uses the `@distributed` decorator so the same training cell can run on one GPU or multiple GPUs by changing the `gpus` parameter.
   - Saves rank-0 adapter artifacts to a Unity Catalog volume.
   - Logs training metrics to MLflow.

3. Register the custom LLM.

   The training notebook includes a separate registration section that:

   - Loads the saved adapter artifacts.
   - Merges the adapter into the base model.
   - Saves merged Hugging Face weights into an MLflow artifact.
   - Configures a vLLM OpenAI-compatible server entrypoint for `llm/v1/chat`.
   - Registers the MLflow model to Unity Catalog using the Databricks Model Serving environment pack.

4. Deploy the serving endpoint.

   If `deploy_endpoint: true` in `air/training.yaml`, the training notebook creates or updates the configured Model Serving endpoint and routes 100% of traffic to the registered model version.

5. Load test the endpoint.

   Run `air/load_test_serving_endpoint.py` after the endpoint is ready. The notebook:

   - Samples prompts from the SFT Delta table.
   - Runs a smoke test against the endpoint.
   - Generates asynchronous HTTP traffic from Spark tasks.
   - Records achieved throughput, status counts, latency samples, and summary metrics to a Delta table.

## Local Development

Create and activate a virtual environment if needed:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install local dependencies:

```bash
.venv/bin/python -m pip install -r air/requirements.txt
```

The ingestion notebook can run with Databricks Connect when authentication is configured:

```bash
databricks auth profiles
.venv/bin/python setup/01_load_tabformer_dataset.py
```

The training notebook is intended to run on Databricks Serverless GPU because it depends on AI Runtime, GPU hardware, and the `serverless_gpu` distributed runtime.

## References

- Databricks AI Runtime: https://docs.databricks.com/aws/en/machine-learning/ai-runtime/
- Databricks custom LLM serving: https://docs.databricks.com/aws/en/machine-learning/model-serving/serve-custom-llms
- IBM TabFormer dataset: https://github.com/IBM/TabFormer
