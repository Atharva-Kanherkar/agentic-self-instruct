# codex/sft-export-preset — Test Contract

## Functional Behavior

- `datasmith export --from <dir> --format sft --chat-template sharegpt --to local --output sft.jsonl` reads `<dir>/accepted.jsonl` and writes ShareGPT-style records.
- `datasmith export --from <accepted.jsonl> --format sft --chat-template chatml --to local --output sft.jsonl` accepts a direct JSONL path and writes ChatML-style records.
- `--chat-template` supports `sharegpt` and `chatml`; the default is `sharegpt`.
- ShareGPT records contain `conversations` with optional `system`, then `human`, then `gpt` turns using `from`/`value` keys.
- ChatML records contain `messages` with optional `system`, then `user`, then `assistant` turns using `role`/`content` keys.
- The optional system turn comes from `example.metadata["system"]` when present and non-empty.
- The user turn is `render_prompt(example)`.
- The assistant turn is `render_completion(example)` when `example.expected` is present, otherwise the first strong-solver attempt output.
- Examples with no `expected` and no usable strong attempt are skipped and counted as `no assistant output`.
- SFT records include a `metadata` sidecar with lineage fields: `gap`, `weak_score`, `strong_score`, `quality`, `tags`, and `generator` when present.
- Dict inputs without known prompt keys use the existing JSON fallback and report a prompt-fallback notice count in the CLI summary.
- `--chat-template chatml` is rejected for non-SFT formats with a clear error.

## Unit Tests

- `tests/test_export.py` covers the `sft` registry entry and unknown format error text including `sft`.
- `tests/test_export.py` covers ShareGPT output with an expected answer.
- `tests/test_export.py` covers ChatML output with a strong-attempt fallback answer.
- `tests/test_export.py` covers optional system turns and SFT metadata lineage.
- `tests/test_export.py` covers the skip case when no assistant turn can be emitted.
- `tests/test_export.py` covers the prompt JSON-fallback notice count.
- `tests/test_export.py` covers chat-template validation for SDK and CLI calls.

## Integration / Functional Tests

- CLI tests cover `datasmith export --format sft --chat-template sharegpt --to local` from a run directory.
- CLI tests cover `datasmith export --format sft --chat-template chatml --to local` from a direct JSONL path.
- CLI tests cover skipped SFT records and prompt-fallback notice summary output.

## Smoke Tests

- Run `.venv/bin/python -m pytest`.
- Run `.venv/bin/python -m ruff check .`.
- Run a local `datasmith export --format sft --chat-template sharegpt` command against a temporary accepted JSONL and inspect the output shape.
- Run a local `datasmith export --format sft --chat-template chatml` command against a temporary accepted JSONL and inspect the output shape.

## E2E Tests

- N/A — this is a local CLI export preset with no external service dependency.

## Manual / cURL Tests

- N/A — no HTTP API is changed.
