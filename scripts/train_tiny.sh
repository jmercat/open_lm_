# TORCH_DISTRIBUTED_DEBUG="DETAIL" torchrun --nproc-per-node 3 -m open_lm.main \
# TORCH_LOGS="+dynamo" TORCHDYNAMO_VERBOSE=1 \
TORCHINDUCTOR_VERBOSE=1 CUDA_VISIBLE_DEVICE=1 CUDA_LAUNCH_BLOCKING=1 python open_lm/main.py \
 --model open_lm_tiny_flex.json \
 --dataset-manifest "s3://tri-ml-datasets/openlm/dcnlp/datasets/tri-hero-run1_cc_v4_resiliparse_rw_v2_bff_minngram13_10shards_all_fasttext_OH_eli5_vs_rw_v2_bigram_200k_train_0.11-starcoder-math_datasets/manifest.jsonl" \
 --data-key "json.gz" \
 --train-num-samples 10_000_000 \
 --precision "amp_bfloat16" \
 --fsdp-amp \
 --fsdp-use-orig-params \
 --workers 1 \
 --global-batch-size 4 \
 --log-every-n-steps 100 \
 --grad-clip-norm 1 \
 --lr 3e-4 \
 --accum-freq 4 \
 --warmup 10 \
 --wd 0.1 \
 --beta2 0.98 \
 --epochs 1 \
 --report-to wandb \
 --wandb-project-name open_lm \
 --name open_lm_ex_$RANDOM \
 --resume latest \
 --logs logs \
 --z-loss-coefficient 1e-4 \
 --torchcompile 


    