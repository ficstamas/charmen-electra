#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=1

s=256

data="/data2/ficstamas/MILab/data/OpinHuBank_20130106_updated.xls"

cp="/data2/ficstamas/MILab/electra_hubert-token/trainer-experiments-500K_loss-1-50-AdamW-lr8e-5/checkpoint-500000/pytorch_model.bin"

output="/data2/ficstamas/MILab/finetuning/opinhubank/electra/"
python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --max_length "$s"
output="/data2/ficstamas/MILab/finetuning/opinhubank-binary/electra/"
python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --max_length "$s" --binary