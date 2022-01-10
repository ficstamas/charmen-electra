#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES="0,1,2,3"

blocks=(4 6)
downsample_factor=(2 4)
seql=(512 1024)

data="https://raw.githubusercontent.com/xerevity/my-little-datasets/main/OpinHuBank_20130106.csv"

for b in ${blocks[*]}
do
  for df in ${downsample_factor[*]}
  do
    for s in ${seql[*]}
    do
      cp="/data2/ficstamas/MILab/electra_charformer/block-${b}_ds-${df}_seq-${s}/checkpoint-500000/pytorch_model.bin"
      if [ ! -f "${cp}" ]; then
        continue
      fi
      output="/data2/ficstamas/MILab/MSZNY/finetuning/opinhubank/electra_charformer/block-${b}_ds-${df}_seq-${s}/"
      python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --charformer --max_block_size "$b" --downsample_factor "$df" --max_length "$s" --score_consensus_attn --freeze 4 --lr "5e-5" --weight_decay 0.1 --adam_eps "1e-8"
      output="/data2/ficstamas/MILab/MSZNY/finetuning/opinhubank/electra_charformer/block-${b}_ds-${df}_seq-${s}-upsample/"
      python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --charformer --max_block_size "$b" --downsample_factor "$df" --max_length "$s" --score_consensus_attn --upsample_output --freeze 4 --lr "5e-5" --weight_decay 0.1 --adam_eps "1e-8"

      output="/data2/ficstamas/MILab/MSZNY/finetuning/opinhubank-binary/electra_charformer/block-${b}_ds-${df}_seq-${s}/"
      python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --charformer --max_block_size "$b" --downsample_factor "$df" --max_length "$s" --score_consensus_attn --binary --freeze 4 --lr "5e-5" --weight_decay 0.1 --adam_eps "1e-8"
      output="/data2/ficstamas/MILab/MSZNY/finetuning/opinhubank-binary/electra_charformer/block-${b}_ds-${df}_seq-${s}-upsample/"
      python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --charformer --max_block_size "$b" --downsample_factor "$df" --max_length "$s" --score_consensus_attn --binary --upsample_output --freeze 4 --lr "5e-5" --weight_decay 0.1 --adam_eps "1e-8"
    done
  done
done

cp="/data2/ficstamas/MILab/electra_hubert-token/trainer-experiments-500K_loss-1-50-AdamW-lr8e-5/checkpoint-500000/pytorch_model.bin"

output="/data2/ficstamas/MILab/MSZNY/finetuning/opinhubank/electra/"
python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --max_length 256 --freeze 12 --lr "5e-5" --weight_decay 0.1 --adam_eps "1e-8"
output="/data2/ficstamas/MILab/MSZNY/finetuning/opinhubank-binary/electra/"
python finetuning.py --checkpoint "${cp}" --output "${output}" --data ${data} --max_length 256 --binary --freeze 12 --lr "5e-5" --weight_decay 0.1 --adam_eps "1e-8"