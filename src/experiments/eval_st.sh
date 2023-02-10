#!/bin/bash

experiment_path=$1
data_root=$2
lang_pair=$3
split=$4

task=st

n_avg=10
ckpt_name=avg_best_${n_avg}_checkpoint.pt
path_to_ckpt=${experiment_path}/ckpts/${ckpt_name}

python ${SEGAUGMENT_ROOT}/src/utils/find_best_ckpts.py \
    -ckpt ${experiment_path}/ckpts -n $n_avg
python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
    --inputs $(head -n 1 ${experiment_path}/ckpts/best_${n_avg}.txt) \
    --output $path_to_ckpt

fairseq-generate ${data_root}/${lang_pair} \
--config-yaml config_${task}.yaml \
--gen-subset ${split}_${task} \
--task speech_to_text \
--path $path_to_ckpt \
--max-tokens 200_000 \
--batch-size 200 \
--beam 5 \
--scoring sacrebleu \
--seed 42 \
--max-source-positions 12_000 \
--results-path ${experiment_path}/results