#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=2

b=4
df=4
s=1024

cp="/data2/ficstamas/MILab/electra_charformer/block-${b}_ds-${df}_seq-${s}/checkpoint-500000/pytorch_model.bin"

data="https://raw.githubusercontent.com/xerevity/my-little-datasets/main/OpinHuBank_20130106.csv"

freeze=("-1" 2 4 6 8 10 12)
learning_rates=("5e-5")

for lr in ${learning_rates[*]}
do
  for f in ${freeze[*]}
  do
    cp="/data2/ficstamas/MILab/electra_charformer/block-${b}_ds-${df}_seq-${s}/checkpoint-500000/pytorch_model.bin"
    output="/data2/ficstamas/MILab/MSZNY/finetuning-dev/opinhubank/electra_charformer/block-${b}_ds-${df}_seq-${s}-lr${lr}-f${f}-upsample/"
    python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --charformer --max_block_size "$b" --downsample_factor "$df" --max_length "$s" --freeze "$f" --lr "$lr" --weight_decay 0.1 --adam_eps "1e-8" --score_consensus_attn --upsample_output
  done
done