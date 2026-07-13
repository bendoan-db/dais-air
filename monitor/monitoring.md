# Monitoring & Drift Detection Design — Fraud Decision SLM

Design for the second module of the monitoring stage: a **Databricks data quality
monitor** (formerly Lakehouse Monitoring,
[docs](https://learn.microsoft.com/en-us/azure/databricks/data-governance/unity-catalog/data-quality-monitoring/))
over the unpacked requests table that `monitor/01_unpack_inference_table.py`
produces. The model is treated as a **classifier**: every request/response pair
follows the shape in `monitor/example_inputs.py` — a single user message whose
prompt always ends with a fixed transaction block:

```
Transaction:
- user_id: 492
- card_id: 3
- timestamp: 2026-06-08 13:45:00
- amount_usd: 2499.99
- use_chip: Online Transaction
- merchant_city: Miami
- merchant_state: FL
- merchant_category_code: 5732
- errors: Bad PIN
```

and a compact JSON completion `{"risk": ..., "action": ..., "reason": ...}`
where `risk` ∈ `{legitimate, suspicious, likely_fraud}` is the predicted label.
The stable `- key: value` shape means the transaction **features** can be parsed
out of every prompt, so this design monitors true input-feature drift, not just
prediction drift.

## 1. Goals

1. **Detect input feature drift** — shifts in the transaction fields embedded in
   every prompt (`amount_usd`, `use_chip`, `merchant_state`, ...), both over
   time and against the training data (covariate shift).
2. **Detect prediction drift** — shifts in the distribution of `risk`/`action`
   over time, against the training data, and across model versions.
3. **Detect contract breakage** — the SLM-specific failure modes: invalid JSON,
   out-of-vocabulary labels, truncated completions, verbosity creep, and a
   changed prompt template (which silently breaks feature extraction).
4. **Track operational health** — volume, latency, token usage per time window
   and per served model version.
5. **Enable model-quality metrics later** — a labeled path (Phase 2) that joins
   late-arriving fraud ground truth so the monitor computes real
   precision/recall instead of proxies.

Non-goal: scoring the free-text `reason` field. Distribution profiling says
nothing useful about prose; that is a sampled LLM-as-judge concern (§10),
layered separately if needed.

## 2. Architecture

```
serving endpoint (AI Gateway inference logging — enabled at deploy time)
        │  raw JSON request/response strings, best-effort delivery ≤ ~1h
        ▼
<catalog>.<schema>.qwen3_4b_instruct_finetuned_payload          (inference table)
        │  monitor/01_unpack_inference_table.py                 [exists; extended §4]
        │  Structured Streaming, trigger(availableNow=True)
        ▼
<catalog>.<schema>.qwen3_4b_instruct_finetuned_requests         (unpacked table)
        │  CDF enabled, partitioned by request_date
        │  response_risk / response_action extracted from the completion JSON
        │  txn_* feature columns extracted from the prompt      [new, §4]
        │
        │  data quality monitor — InferenceLog profile           [this design]
        │       ▲ baseline table: features + labels parsed from the SFT dataset
        ▼
qwen3_4b_instruct_finetuned_requests_profile_metrics             (generated)
qwen3_4b_instruct_finetuned_requests_drift_metrics               (generated)
        │
        ├── auto-generated AI/BI monitoring dashboard
        └── SQL alerts on the metrics tables
```

Module 01 already satisfies every precondition the monitor has: change data feed
is enabled (incremental refreshes read CDF), the table is partitioned by
`request_date`, `served_entity_id` is preserved as the model-version dimension,
and `response_json_fields: [risk, action]` in `monitor.yaml` surfaces the
prediction as the categorical columns `response_risk` and `response_action`.
The one change this design makes to module 01 is the prompt-feature extraction
in §4.

## 3. Signal model — what drift means for this SLM

The model is generative but wears a classifier's contract, so monitoring
decomposes into four signal families. All are computed from columns of the
unpacked table.

| Family | Columns | What a shift means |
|---|---|---|
| **Input feature drift** | `txn_amount_usd`, `txn_use_chip`, `txn_merchant_city`, `txn_merchant_state`, `txn_merchant_category_code`, `txn_errors` (§4) | The transaction mix reaching the model changed — new merchant categories, an amount distribution shift, a spike in `errors` values the model rarely saw in training. Baseline drift here is classic covariate shift vs. the training set. |
| **Prediction drift** | `response_risk`, `response_action` | The model's decisions changed. Read together with feature drift: feature drift + prediction drift = the world changed; prediction drift *without* feature drift = the model changed (regression after redeploy — disambiguate by slicing on `served_entity_id`). |
| **Contract integrity** | `response_risk` null rate, out-of-vocabulary `risk` values, `finish_reason`, `completion_tokens`, `response_chars`, `txn_*` null rates | The model or the prompt template is breaking rather than drifting: JSON parse failures, new labels, truncation at `max_tokens` (the reasoning-leakage failure mode), verbosity creep beyond the trained ~25–35 tokens, or a template change that nulls out feature extraction. |
| **Operational** | row volume per window, `execution_duration_ms`, `prompt_tokens`/`total_tokens`, `status_code` | Traffic, latency, and cost. Caveat: AI Gateway logging is best-effort and 4xx/5xx rows may never land, so 429 storms must be watched client-side (the load-test stage), not here. |

## 4. Prompt feature extraction (change to module 01)

Every prompt ends with the fixed transaction block, so module 01 parses the
`- key: value` lines into typed columns the same way it already parses the
response JSON — config-driven, nulling out on mismatch rather than failing the
stream.

### Configuration

A new `prompt_fields` list in `monitor.yaml`, each entry a field name (as it
appears in the prompt) and a Spark cast type:

```yaml
prompt_fields:
  - name: amount_usd
    type: double
  - name: use_chip
    type: string
  - name: merchant_city
    type: string
  - name: merchant_state
    type: string
  - name: merchant_category_code
    type: string        # a code, not a quantity — categorical on purpose
  - name: errors
    type: string
```

Each entry becomes a column `txn_<name>` (prefixed to mirror `response_<field>`
and avoid collisions). Extraction per field:

```python
raw = F.regexp_extract(F.col("prompt"), rf"(?m)^-\s*{re.escape(name)}:\s*(.*?)\s*$", 1)
col = F.when(raw == "", None).otherwise(raw).cast(cast_type)
```

`regexp_extract` returns `""` on no match; that is normalized to null, and a
failed cast also yields null — so a renamed field, a reordered template, or a
malformed value produces nulls, never a stream failure. Rising null rates on
`txn_*` columns are themselves a first-class signal (template drift, §7).

### Deliberately excluded fields

- **`user_id`, `card_id`** — high-cardinality identifiers. Chi-square/JS drift
  on them is noise and `frequent_items` bloats the metrics tables. If breadth
  of user mix matters, add a `distinct_count`-style custom metric instead of a
  column.
- **`timestamp`** (the transaction's own event time) — adds little as a
  profiled column. Optional later derivation: `request_time − txn event time`
  as a staleness/replay signal.

`merchant_city` is borderline (hundreds of categories): included by default
since `frequent_items` tracks top-k and baseline JS distance still behaves, but
it is the first field to drop from `prompt_fields` if metrics cost matters.

### Type discipline

Numeric fields (`amount_usd`) get `ks_test`/`wasserstein_distance` drift tests;
string fields get `chi_squared_test`/`tv_distance`/`l_infinity_distance`/
`js_distance`. `merchant_category_code` stays a string: treating a categorical
code as a number would produce meaningless numeric drift.

### Shared extraction code

The extraction-expression builder lives in a new **plain Python module**
`monitor/monitoring_utils.py` (no `# Databricks notebook source` header — same
precedent as `train/train.py`), imported by both module 01 (serving traffic) and
module 02 (baseline build, §6) so the two tables are guaranteed to parse
identically. CLAUDE.md's notebook-format rule gains this file as a documented
exception.

### Reprocessing

Adding or changing `prompt_fields` changes the unpacked table's schema. As
module 01 already documents for `response_json_fields`: delete the streaming
checkpoint directory and the unpacked table, then rerun to reprocess the full
payload table.

## 5. Monitor specification

One monitor (UC allows exactly one per table) on the unpacked table, using the
**InferenceLog** profile with `problem_type = classification`. The `txn_*`
feature columns need no monitor-side configuration — every column of the table
is profiled and drift-tested automatically; extraction (§4) is what makes the
features visible to it.

| Parameter | Value | Rationale |
|---|---|---|
| `table_name` | `<catalog>.<schema>.<unpacked_table>` | The analysis-ready table from module 01. |
| Profile type | `InferenceLog` | Prediction-aware profiling: per-label distributions, model-version series, and (Phase 2) quality metrics from labels. |
| `problem_type` | `PROBLEM_TYPE_CLASSIFICATION` | `risk` is a 3-class label. |
| `prediction_col` | `response_risk` | The extracted predicted label. |
| `timestamp_col` | `request_time` | Request arrival time (already a timestamp). |
| `model_id_col` | `served_entity_id` | Each redeploy creates a new served entity → per-version metric series for free; champion-vs-previous comparison after every deploy. |
| `label_col` | *(unset in Phase 1)* | Wired to `label_risk` in Phase 2 (§9). |
| `granularities` | `["1 day"]` (config-driven; add `"1 hour"` for high-traffic endpoints) | Each granularity multiplies refresh cost; daily matches the recommended weekly-to-daily unpack cadence. |
| `slicing_exprs` | `["response_action"]` (config-driven) | Risk distribution conditioned on the action taken. `txn_*` columns are valid slice targets too (e.g. `txn_use_chip` to split online vs. chip transactions) — but slices multiply cost, so the default stays minimal. |
| `baseline_table_name` | `<unpacked_table>_baseline` | Drift vs. the training distribution — features *and* labels (§6). |
| `output_schema_name` | `<catalog>.<schema>` (same schema by default, configurable) | Metrics tables land next to the tables they describe. |
| `assets_dir` | `/Workspace/Users/<current user>/quality_monitoring/<unpacked_table>` | Auto-generated dashboard location; derived from the current user like the MLflow experiment path — nothing hardcoded. |
| `schedule` | *(unset)* | Refresh is orchestrated by the job that runs the unpack notebook (§8), so metrics never lag the data. |

Created via the SDK (`WorkspaceClient().quality_monitors.create(...)`) from a new
notebook, `monitor/02_create_quality_monitor.py`, which is idempotent: it
`get`s the monitor and `update`s it if it exists, creates it otherwise, then
triggers an initial `run_refresh` and prints the dashboard link.

### Generated assets

- `<unpacked_table>_profile_metrics` — per window × granularity × model version
  × slice × column: counts, `percent_null`, `distinct_count`, `frequent_items`
  (categorical distributions for `response_*` and string `txn_*` columns),
  numeric stats (for `txn_amount_usd`, token counts, latency), plus the custom
  metrics in §7.
- `<unpacked_table>_drift_metrics` — pairwise comparisons, both **consecutive**
  (window vs. previous window) and **baseline** (window vs. training baseline):
  `chi_squared_test`, `tv_distance`, `l_infinity_distance`, `js_distance` for
  categoricals; `ks_test` / `wasserstein_distance` for numerics.
- An AI/BI dashboard over both tables, auto-created in `assets_dir`.

## 6. Baseline table — features and labels from the training set

Window-over-window drift misses slow drift and can't answer "is serving traffic
still what we trained on?". The baseline anchors both — and because the SFT
prompts use the **same transaction template** as serving traffic, the baseline
carries the training-time *feature* distribution, not just the label
distribution.

`monitor/02_create_quality_monitor.py` builds `<unpacked_table>_baseline` from
the SFT table (`fraud_sft_dataset`, the same table training and the load test
read):

1. Parse the `prompt` column with the **same** `prompt_fields` extraction from
   `monitoring_utils.py` (§4) → `txn_amount_usd`, `txn_use_chip`, ... columns.
   Baseline feature drift is therefore covariate shift measured against exactly
   what the model saw in training.
2. Parse `assistant_response` with `from_json` using the same
   `response_json_fields` schema module 01 uses → `response_risk`,
   `response_action` columns.
3. Add `served_entity_id = 'training_baseline'` — the baseline must carry the
   `model_id_col`. Baseline drift is computed only for columns present in both
   tables; columns with no training-time analogue (`prompt_tokens`,
   `execution_duration_ms`, `finish_reason`, ...) are simply absent from the
   baseline and skipped.
4. Overwrite the baseline table on every run of module 02, mirroring the
   pipeline's overwrite-on-rerun convention — after a retrain on new data,
   rerunning module 02 refreshes the baseline to match.

## 7. Custom metrics — contract-integrity signals

Built-in profiling covers distributions; the SLM-specific breakage signals are
**aggregate custom metrics** (table-scoped, computed per window / model version
/ slice, and therefore alertable and drift-comparable like any other metric):

| Metric | Definition (SQL aggregate) | Catches |
|---|---|---|
| `json_contract_failure_rate` | `avg(CASE WHEN response_text IS NOT NULL AND response_risk IS NULL THEN 1.0 ELSE 0.0 END)` | Completions that are not valid JSON or lost the `risk` key — the model breaking its output format. |
| `prompt_parse_failure_rate` | `avg(CASE WHEN prompt IS NOT NULL AND txn_amount_usd IS NULL AND txn_use_chip IS NULL AND txn_merchant_state IS NULL THEN 1.0 ELSE 0.0 END)` *(all-fields-null test generated from `prompt_fields`)* | The prompt template changed or clients send free-form prompts — feature extraction (and therefore feature-drift monitoring) has gone blind. Per-field `percent_null` catches partial template changes (one renamed field). |
| `invalid_risk_rate` | `avg(CASE WHEN response_risk IS NOT NULL AND response_risk NOT IN ('legitimate','suspicious','likely_fraud') THEN 1.0 ELSE 0.0 END)` | Out-of-vocabulary labels. The value list comes from `expected_prediction_values` in `monitor.yaml`, not hardcoded. |
| `truncation_rate` | `avg(CASE WHEN finish_reason = 'length' THEN 1.0 ELSE 0.0 END)` | Hitting `max_tokens` — reasoning leakage or verbosity regressions (the example payload finishes with `stop` at 28 completion tokens against `max_tokens: 64`; healthy traffic should stay there). |
| `likely_fraud_rate` | `avg(CASE WHEN response_risk = 'likely_fraud' THEN 1.0 ELSE 0.0 END)` | The business-level block rate as a single scalar time series — the number an on-call person checks first. |

`completion_tokens` and `response_chars` need no custom metric: numeric profile
stats plus drift tests already flag verbosity creep.

## 8. Refresh orchestration & scheduling

A single scheduled Lakeflow job (serverless) with two sequential tasks, so
metrics are computed immediately after new payload rows are unpacked:

1. **Unpack** — run `monitor/01_unpack_inference_table.py` (drains new payload
   rows via `availableNow`, including prompt-feature extraction).
2. **Refresh monitor** — `quality_monitors.run_refresh(table_name=...)`. The
   refresh reads the unpacked table's CDF, so it processes only changed
   partitions/rows.

Cadence: daily by default (matches the `1 day` granularity); hourly for
high-traffic endpoints, with the `1 hour` granularity enabled to match.
Rebuilding the baseline (module 02) is **not** part of this job — it reruns only
after a retrain changes the SFT table.

## 9. Phase 2 — late-arriving ground truth

Fraud labels arrive days later (chargebacks, investigations). The InferenceLog
profile is designed for exactly this: once a label column exists, refreshes
recompute windows and emit classification quality metrics (accuracy, per-class
precision/recall/F1, confusion matrix) for windows where labels are present.

1. Fraud ops maintain a `ground_truth_labels` table:
   `(client_request_id, label_risk, label_time)`.
2. A third task in the monitoring job MERGEs labels into the unpacked table as a
   nullable `label_risk` column (schema evolution; runs after the unpack task so
   the append stream and the MERGE never write concurrently). CDF propagates the
   updates to the next monitor refresh.
3. Update the monitor with `label_col="label_risk"`.

With labels joined, the `txn_*` columns pay off twice: quality metrics sliced by
feature (e.g. recall on `txn_use_chip = 'Online Transaction'`) localize *where*
the model degrades, not just *that* it degrades.

**Prerequisite:** callers must send a `client-request-id` header — it is the
only join key the inference table captures. The load-test generator does not set
one today; wiring it (e.g. to a transaction id) is a required change for an
end-to-end labeled demo.

## 10. Out of scope / known gaps

- **`reason` quality** — free text; if needed, a sampled daily LLM-as-judge pass
  (MLflow GenAI scorers or `ai_query`) scoring whether the reason is grounded in
  the prompt's transaction fields. Separate module, separate design.
- **Full-population feature drift** — the `txn_*` columns observe only
  transactions that reach the endpoint (a selection effect if upstream routing
  changes). A plain time-series monitor on the upstream transaction table covers
  the full population if that distinction matters.
- **Failed-request visibility** — 4xx/5xx (including 429s beyond provisioned
  concurrency) may never reach the inference table; error-rate alerting stays
  client-side.
- **Delivery lag** — inference logging is best-effort within ~1 hour; the most
  recent window is always partially filled. Alerts should exclude the current
  in-progress window.

## 11. Alerting

SQL alerts (Databricks SQL, scheduled after the monitoring job) over the
generated tables. Thresholds start conservative and are tuned against the first
weeks of profile history. Suggested set:

```sql
-- 1. Input feature drift vs. training baseline (covariate shift)
SELECT window.start, column_name, js_distance, chi_squared_test.pvalue,
       ks_test.pvalue AS ks_pvalue, wasserstein_distance
FROM <catalog>.<schema>.<unpacked_table>_drift_metrics
WHERE drift_type = 'BASELINE'
  AND column_name IN ('txn_amount_usd', 'txn_use_chip', 'txn_merchant_state',
                      'txn_merchant_category_code', 'txn_errors')
  AND window.start >= current_date() - INTERVAL 2 DAYS
  AND window.end   <= current_timestamp()          -- skip the in-progress window
  AND (js_distance > 0.15                          -- categorical features
       OR chi_squared_test.pvalue < 0.01
       OR ks_test.pvalue < 0.01);                  -- numeric features (amount_usd)

-- 2. Prediction drift vs. training baseline (categorical label)
SELECT window.start, column_name, js_distance, chi_squared_test.pvalue
FROM <catalog>.<schema>.<unpacked_table>_drift_metrics
WHERE column_name = 'response_risk'
  AND drift_type = 'BASELINE'
  AND window.start >= current_date() - INTERVAL 2 DAYS
  AND window.end   <= current_timestamp()
  AND (js_distance > 0.15 OR chi_squared_test.pvalue < 0.01);

-- 3. Contract breakage (custom metrics live in the profile table)
SELECT window.start, json_contract_failure_rate, prompt_parse_failure_rate,
       invalid_risk_rate, truncation_rate
FROM <catalog>.<schema>.<unpacked_table>_profile_metrics
WHERE column_name = ':table'
  AND window.start >= current_date() - INTERVAL 2 DAYS
  AND (json_contract_failure_rate > 0.02
       OR prompt_parse_failure_rate > 0.02
       OR invalid_risk_rate > 0.001
       OR truncation_rate > 0.01);

-- 4. Volume collapse (no traffic reached the table)
SELECT window.start, count
FROM <catalog>.<schema>.<unpacked_table>_profile_metrics
WHERE column_name = ':table'
  AND window.start >= current_date() - INTERVAL 2 DAYS
  AND count = 0;
```

A fifth alert on `likely_fraud_rate` (both absolute bounds and
consecutive-window drift) guards the business metric directly. Reading alerts 1
and 2 together disambiguates cause: both firing points at the world changing;
alert 2 alone points at the model or its serving stack.

## 12. Configuration & repo changes

### `monitor/monitor.yaml` additions

```yaml
# ---- 01_unpack_inference_table.py (extension) ----
# Transaction fields parsed from the prompt's fixed "- key: value" block into
# typed txn_<name> columns. user_id/card_id are deliberately excluded
# (high-cardinality IDs — drift stats on them are noise).
prompt_fields:
  - name: amount_usd
    type: double
  - name: use_chip
    type: string
  - name: merchant_city
    type: string
  - name: merchant_state
    type: string
  - name: merchant_category_code
    type: string
  - name: errors
    type: string

# ---- 02_create_quality_monitor.py ----
# Prediction field: must be one of response_json_fields above.
prediction_field: risk
# Closed vocabulary for the prediction; drives the invalid-value custom metric.
expected_prediction_values: [legitimate, suspicious, likely_fraud]
# Extracted response fields to slice profile metrics by (response_<field>).
slicing_fields: [action]
# Monitor windows; each granularity adds refresh cost.
granularities: ["1 day"]
# Baseline drift source: the SFT table training read. Both its prompt column
# (feature baseline) and assistant_response (label baseline) are parsed.
# Must match sft_table in setup/train/load-test configs (validate_config.py).
sft_table: fraud_sft_dataset
# Derived baseline table name (overwritten on each run of module 02).
baseline_table: qwen3_4b_instruct_finetuned_requests_baseline
# Where metrics tables land; empty = same catalog.schema as the monitored table.
monitor_output_schema: ""
# Phase 2: ground-truth column joined into the unpacked table; empty = unset.
label_field: ""
```

### New / changed files

- `monitor/monitoring_utils.py` — **plain Python module** (no notebook header;
  same precedent as `train/train.py`): builds the `txn_*` extraction expressions
  from `prompt_fields`, shared by modules 01 and 02 so serving and baseline
  tables parse identically. Documented as a notebook-format exception in
  CLAUDE.md.
- `monitor/01_unpack_inference_table.py` — extended with the prompt-feature
  extraction (§4). Schema change ⇒ existing deployments delete the checkpoint
  and unpacked table and reprocess.
- `monitor/02_create_quality_monitor.py` — new Databricks notebook (serverless
  CPU): builds the baseline table (features + labels) from the SFT table,
  creates-or-updates the monitor, triggers the first refresh, prints the
  dashboard link. Imports shared helpers from `train/training_utils.py` via
  `sys.path`, like module 01.
- *(Phase 2)* `monitor/03_join_ground_truth.py` — label MERGE task.

### Cross-file contract updates (`scripts/validate_config.py`)

- `monitor.yaml`'s `sft_table` must equal `sft_table` in `setup.yaml`,
  `train.yaml`, and `serving_load_test.yaml` (the baseline must come from the
  same table training read).
- `prompt_fields` entries must have a non-empty `name` and a `type` in
  `{string, int, double, timestamp}`; names must be unique and must not collide
  with `response_json_fields` or the built-in unpacked columns.
- `prompt_fields` names should be a subset of `setup.yaml`'s prompt-template
  fields (the columns `setup/02_stage_training_data.py` concatenates) — warn on
  mismatch, since a field absent from the template will be 100% null.
- `prediction_field` must be a member of `response_json_fields`.
- `slicing_fields` must reference extracted columns (`response_*` or `txn_*`).
- `expected_prediction_values` must be non-empty when `prediction_field` is set.

### Permissions

The identity creating the monitor needs `USE CATALOG`/`USE SCHEMA`, `SELECT` on
the unpacked and baseline tables, and `CREATE TABLE` on the output schema.
Refreshes run on Databricks-managed serverless compute.
