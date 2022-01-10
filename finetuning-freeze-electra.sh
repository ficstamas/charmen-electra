#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

b=4
df=4
s=1024

data="https://raw.githubusercontent.com/xerevity/my-little-datasets/main/OpinHuBank_20130106.csv"

freeze=("-1" 2 4 6 8 10 12)
learning_rates=("5e-5")

for lr in ${learning_rates[*]}
do
  for f in ${freeze[*]}
  do
    cp="/data2/ficstamas/MILab/electra_hubert-token/trainer-experiments-500K_loss-1-50-AdamW-lr8e-5/checkpoint-500000/pytorch_model.bin"
    output="/data2/ficstamas/MILab/MSZNY/finetuning-dev/opinhubank/electra/lr${lr}-f${f}/"
    python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --max_length 256 --freeze "$f" --lr "$lr" --weight_decay 0.1 --adam_eps "1e-8"
  done
done