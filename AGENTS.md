# Repository Guidelines

## Project Structure & Module Organization

This repository implements a five-stage Databricks AI Runtime pipeline. `setup/` ingests, shards, and renders SFT data. `train/train_qwen_unsloth/`, `train/train_phi_4_unsloth/`, and `train/train_gpt_oss_fsdp/` are standalone projects; each owns training and deployment notebooks, trainer, YAML, config helper, and requirements. Root `train/` contains shared assets only. `load_test/` drives serving traffic, while `monitor/` unpacks inference logs and configures quality monitoring. Experimental implementations live in `extras/`.

Most stage `.py` files are Databricks notebook sources. Plain modules include each project's `train.py`, `project_config.py`, and `sft_conversion.py`; stage-local `setup/utils.py`, `load_test/utils.py`, and `monitor/utils.py`; and `monitor/monitoring_utils.py`.

## Build, Test, and Development Commands

- `python -m compileall -q setup train load_test monitor`: compile-check all maintained Python sources, matching CI.
- `.venv/bin/python setup/01_load_dataset.py` and `.venv/bin/python setup/02_stage_training_data.py`: run setup through Databricks Connect.
- `cd train/train_qwen_unsloth && COPYFILE_DISABLE=1 air run --file train.yaml --watch`: submit Qwen training.
- `cd train/train_phi_4_unsloth && COPYFILE_DISABLE=1 air run --file train.yaml --watch`: submit Phi-4 training.
- `cd train/train_gpt_oss_fsdp && COPYFILE_DISABLE=1 air run --file train.yaml --watch`: submit GPT-OSS FSDP training.
- `databricks bundle validate`: validate the Databricks bundle after configuring its workspace host.

There is no build step or unit-test suite. Training, deployment, load testing, and monitoring require the documented Databricks compute environments.

## Coding Style & Naming Conventions

Use Python with four-space indentation, `snake_case` functions and variables, and `UPPER_SNAKE_CASE` runtime constants. Keep config helpers side-effect free. Preserve Databricks notebook markers (`# Databricks notebook source`, `# COMMAND ----------`, and `# MAGIC`). Trainers must use their project-local config and SFT conversion modules; do not make them depend on root pipeline config or the other project.

## Testing Guidelines

Before every pull request, run the compile check above. When changing YAML, verify every duplicated stage contract remains aligned. Workspace-dependent changes should be exercised in the appropriate notebook or AIR workload and documented in the PR.

## Commit & Pull Request Guidelines

Recent commits use short, lowercase, imperative summaries, for example `add sft prep notebook` or `update deployment params`. Keep commits focused. PRs should explain the affected pipeline stage, configuration or schema changes, validation performed, and required rerun order. Link relevant issues; include screenshots only for notebook output or serving/monitoring UI changes.
