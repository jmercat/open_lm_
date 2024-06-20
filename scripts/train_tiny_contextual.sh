# CUDA_VISIBLE_DEVICE=1 CUDA_LAUNCH_BLOCKING=1 python open_lm/main.py \
TORCH_DISTRIBUTED_DEBUG="DETAIL" torchrun --nproc-per-node 8 -m open_lm.main \
 --model open_lm_79m_contextual \
 --dataset-manifest s3://tri-ml-datasets/openlm/dcnlp/datasets/refined_web_tokenized/manifest.jsonl \
 --train-num-samples 157_845_504 \
 --precision "amp_bfloat16" \
 --fsdp-amp \
 --fsdp-pure-bf16 \
 --workers 1 \
 --global-batch-size 512 \
 --log-every-n-steps 100 \
 --grad-clip-norm 1 \
 --data-key json.gz \
 --lr 3e-3 \
 --accum-freq 4 \
 --warmup 400 \
 --wd 0.1 \
 --beta2 0.95 \
 --epochs 10 \
 --report-to wandb \
 --wandb-project-name open_lm \
 --name RW_79M_CC1_lr3e3_bs512_contextual_or_rotary_step32_$RANDOM \
 --resume latest \
 --logs logs \
 --z-loss-coefficient 1e-4 
#  --pretrained logs/open_lm_ex_6597/checkpoints/epoch_40.pt \

 
