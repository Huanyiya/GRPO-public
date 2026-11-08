cd /mnt/cpfs/users/wxh/github-source/Self-Distillation-AgentRL

OPENBLAS_NUM_THREADS=1 \
OMP_NUM_THREADS=4 \
MKL_NUM_THREADS=4 \
NUMEXPR_NUM_THREADS=4 \
TOKENIZERS_PARALLELISM=false \
python examples/search/retriever/retrieval_server.py \
  --index_path /mnt/cpfs/users/wxh/github-source/data/SearchQA/searchR1/e5_Flat.index \
  --corpus_path /mnt/cpfs/users/wxh/github-source/data/SearchQA/searchR1/wiki-18.jsonl \
  --retriever_model /mnt/cpfs/users/wxh/github-source/models/e5-base-v2 \
  --retriever_name e5 \
  --topk 3 \
  --port 8000 \
  --metrics_log /mnt/cpfs/users/wxh/github-source/logs/searchqa/retriever-server.log


SEARCH_URL=http://127.0.0.1:8000/retrieve \
SEARCH_API_STYLE=single \
SEARCH_MAX_CONCURRENCY=64 \
bash /mnt/cpfs/users/wxh/github-source/GRPO/scripts/train/run-grpo-searchqa.sh --seed 2026 --rollout-seed 2026
