input = {'messages': [{'role': 'user',
   'content': 'You are a fraud decision model for a credit-card transaction stream. Classify the transaction as legitimate, suspicious, or likely_fraud. Return only compact JSON with keys risk, action, and reason.\n\nTransaction:\n- user_id: 492\n- card_id: 3\n- timestamp: 2026-06-08 13:45:00\n- amount_usd: 2499.99\n- use_chip: Online Transaction\n- merchant_city: Miami\n- merchant_state: FL\n- merchant_category_code: 5732\n- errors: Bad PIN'}],
 'max_tokens': 64,
 'temperature': 0.0}

output = {
  "id": "chatcmpl-6d190d10-a0d9-474c-9cac-ba404a5368a7",
  "object": "chat.completion",
  "created": 1783977960,
  "model": "qwen3_4b_instruct_finetuned",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "{\"risk\":\"likely_fraud\",\"action\":\"block_transaction\",\"reason\":\"Bad PIN detected during online transaction, indicating potential fraudulent activity.\"}",
        "refusal": null,
        "annotations": null,
        "audio": null,
        "function_call": null,
        "tool_calls": [],
        "reasoning_content": null
      },
      "logprobs": null,
      "finish_reason": "stop",
      "stop_reason": null,
      "token_ids": null
    }
  ],
  "service_tier": null,
  "system_fingerprint": null,
  "usage": {
    "prompt_tokens": 139,
    "total_tokens": 167,
    "completion_tokens": 28,
    "prompt_tokens_details": null
  },
  "prompt_logprobs": null,
  "prompt_token_ids": null,
  "kv_transfer_params": null
}