test_type=text

PROVIDER=hybrid
LLM=deepseekv3
vlm=qwen2vl

# PROVIDER=openai
# LLM=4o
# vlm=4o

OUTPUT=results_val_${test_type}_${LLM}_${vlm}
log=$WORKDIR/datasets/gaia/${OUTPUT}.log

python run_gaia_roleplaying.py \
--model_provider $PROVIDER \
--test_type $test_type \
--save_to $OUTPUT | tee $log
