PROVIDER=hybrid
LLM=deepseekv3
vlm=qwen2vl

# PROVIDER=openai
# LLM=4o
# vlm=4o

OUTPUT=results_val_image_${LLM}_${vlm}
log=$WORKDIR/datasets/gaia/${OUTPUT}.log

python run_gaia_roleplaying.py \
--model_provider $PROVIDER \
--test_type image \
--save_to $OUTPUT | tee $log
