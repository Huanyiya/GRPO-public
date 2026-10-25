# ALFWorld evaluation

This evaluates an already converted Hugging Face checkpoint on the same strict
`valid_seen` TextWorld task pool used by SDA. The launcher starts a local
SGLang server, runs one episode for each eligible game, then stops that server.
No checkpoint conversion or training takes place.

Run from the repository root in an environment with the project's SGLang,
Transformers, ALFWorld and TextWorld dependencies installed:

```bash
MODEL_DIR=/path/to/converted_hf_model \
ALFWORLD_DATA=/path/to/data/alfworld \
CUDA_VISIBLE_DEVICES=0 \
bash scripts/test/run_alfworld.sh
```

The default settings are 140 sorted, solvable `valid_seen` games, one episode
per game, 50 environment steps, two history steps, 2048 prompt tokens, 512
response tokens per step, `temperature=1.0`, `top_p=1.0`, and one sampled
response per turn. The launcher uses the nonthinking chat template. The prompt,
action parsing, environment reward, and task category rules come from the
existing `slime_plugins.alfworld` adapter, which mirrors SDA. Request seeds
follow SDA's `per_task_turn` derivation; `TASK_SEED` selects the environment
seed and `EVAL_SEED` selects the sampling seed.

Useful overrides:

```bash
MODEL_DIR=/path/to/converted_hf_model \
ALFWORLD_DATA=/path/to/data/alfworld \
PYTHON_BIN=python3 TP_SIZE=1 PORT=30000 \
TASK_SEED=0 EVAL_SEED=1 TEMPERATURE=1.0 \
OUTPUT_DIR=/path/to/output \
bash scripts/test/run_alfworld.sh
```

For a short integration run, append `--limit 2`. Its summary marks
`strict_eval=false`. To use an already running SGLang server, set
`START_SERVER=0` and `SERVER_URL=http://host:port`; ensure that server serves
the same model as `MODEL_DIR`.

Results are written to `rollouts.jsonl` (one record per game, including every
turn) and `summary.json` (overall and per-category success rates). The script
refuses to overwrite either result file. A failed run can leave a partial
`rollouts.jsonl`, but it never writes a completed summary.

This matches SDA's evaluation task selection, environment interaction,
sampling parameters and metric definitions. SGLang and SDA's vLLM backend can
produce different sampled tokens for the same request seed.
