# CUDA_VISIBLE_DEVICE=1 CUDA_LAUNCH_BLOCKING=1 python open_lm/main.py \
CUDA_VISIBLE_DEVICE=0,1 torchrun --nproc-per-node 2 open_lm/main.py -- \
 --model open_lm_long_context_test \
 --dataset-manifest "s3://tri-ml-datasets/openlm/dcnlp/datasets/rw_original_mistral_130k_small/manifest.jsonl" \
 --data-key "json.gz" \
 --train-num-samples 1_000_000 \
 --precision "amp_bfloat16" \
 --fsdp-amp \
 --fsdp-pure-bf16 \
 --workers 1 \
 --global-batch-size 2 \
 --log-every-n-steps 1 \
 --grad-clip-norm 1 \
 --lr 3e-4 \
 --accum-freq 1 \
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

    