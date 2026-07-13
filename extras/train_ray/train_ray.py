#!/usr/bin/env python3
"""Distributed data-parallel fine-tuning with Ray Train on a single 8x H100 node.

The workload `command` starts a Ray head with 8 GPUs and runs this script. Ray Train's
TorchTrainer launches one worker per GPU (8 total), wraps the model in DDP, shards
the dataset across workers, and aggregates metrics. Each worker runs `train_func`.

Uses a public model (no Hugging Face token required) so the example runs as-is.
"""

import os

import mlflow
import ray
import ray.train
import torch
from datasets import load_dataset
from ray.train import RunConfig, ScalingConfig
from ray.train.torch import TorchTrainer, prepare_data_loader, prepare_model
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-3B"
DATASET_NAME = "tatsu-lab/alpaca"
MAX_SEQ_LEN = 1024


def build_dataset(tokenizer):
    raw = load_dataset(DATASET_NAME, split="train[:8000]")

    def format_example(row):
        prompt = f"### Instruction:\n{row['instruction']}\n\n"
        if row.get("input"):
            prompt += f"### Input:\n{row['input']}\n\n"
        text = f"{prompt}### Response:\n{row['output']}{tokenizer.eos_token}"
        out = tokenizer(text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length")
        out["labels"] = out["input_ids"].copy()
        return out

    return raw.map(format_example, remove_columns=raw.column_names)


def train_func(config: dict):
    """Runs on every Ray Train worker (one per GPU)."""
    rank = ray.train.get_context().get_world_rank()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.bfloat16)
    model.config.use_cache = False
    # prepare_model moves the model to this worker's GPU and wraps it in DDP.
    model = prepare_model(model)

    dataset = build_dataset(tokenizer).with_format("torch")
    loader = DataLoader(dataset, batch_size=config["batch_size"], shuffle=True, drop_last=True)
    # prepare_data_loader injects a DistributedSampler and moves batches to the GPU.
    loader = prepare_data_loader(loader)

    optimizer = torch.optim.AdamW(model.parameters(), lr=config["lr"])

    # AI Runtime injects MLFLOW_RUN_ID and configures the databricks tracking URI on
    # the node, so logging works without DATABRICKS_HOST/TOKEN. Gate on MLFLOW_RUN_ID
    # so the script also runs cleanly off-platform (e.g. locally) where it is unset.
    use_mlflow = rank == 0 and bool(os.environ.get("MLFLOW_RUN_ID"))
    if use_mlflow:
        mlflow.start_run(run_id=os.environ.get("MLFLOW_RUN_ID"))
        mlflow.log_params({"model": MODEL_NAME, "lr": config["lr"], "batch_size": config["batch_size"]})

    model.train()
    step = 0
    for batch in loader:
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
        )
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        step += 1

        ray.train.report({"loss": out.loss.item(), "step": step})
        if use_mlflow:
            mlflow.log_metric("train_loss", out.loss.item(), step=step)
        if step >= config["max_steps"]:
            break

    if use_mlflow:
        mlflow.end_run()


def main():
    ray.init(address="auto")
    total_gpus = int(ray.cluster_resources().get("GPU", 0))
    print(f"Ray cluster ready: {total_gpus} GPU(s)", flush=True)

    trainer = TorchTrainer(
        train_func,
        train_loop_config={"lr": 2e-5, "batch_size": 4, "max_steps": 100},
        scaling_config=ScalingConfig(num_workers=total_gpus, use_gpu=True),
        run_config=RunConfig(storage_path="/tmp/ray_results", name="qwen-sft"),
    )
    result = trainer.fit()
    print(f"Training finished. Final metrics: {result.metrics}", flush=True)

    ray.shutdown()


if __name__ == "__main__":
    main()
