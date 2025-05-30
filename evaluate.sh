export CRAG_CACHE_DIR=/srv/scratch/CRUISE/z5544297/.cache

# Single-turn evaluation
python local_evaluation.py     --dataset-type single-turn     --split validation     --num-conversations 100     --display-conversations 5 --suppress-web-search-api   --eval-model gpt-4o-mini --output-dir result/
# python local_evaluation.py     --dataset-type single-turn     --split public_test --num-conversations 100     --display-conversations 5 --suppress-web-search-api   --eval-model gpt-4o-mini --output-dir result/

# Multi-turn evaluation
# python local_evaluation.py --dataset-type single-turn --split validation --num-conversations 20 --display-conversations 5 --eval-model gpt-4o-mini --revision v0.1.2 --output-dir result/
