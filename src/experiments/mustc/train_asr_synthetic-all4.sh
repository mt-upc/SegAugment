#!/bin/bash

original_data_root=$MUSTCv2_ROOT

task=asr
model_size=s
src_lang=en
max_tokens=54000
batch_size=114
base_update_freq=6

dataset_name=$(basename $original_data_root)
experiment_name=${task}_${model_size}_${dataset_name}_${src_lang}_synthetic-all4

synthetic_data_root=${SEGAUGMENT_ROOT}/synthetic_data/${dataset_name}

original_tsv=${original_data_root}/${lang_pair}/train_${task}.tsv
short_tsv=${synthetic_data_root}/${lang_pair}/0.4-3/train/train_0.4-3_${task}.tsv
med_tsv=${synthetic_data_root}/${lang_pair}/3-10/train/train_3-10_${task}.tsv
long_tsv=${synthetic_data_root}/${lang_pair}/10-20/train/train_10-20_${task}.tsv
xlong_tsv=${synthetic_data_root}/${lang_pair}/20-30/train/train_20-30_${task}.tsv

python ${SEGAUGMENT_ROOT}/src/utils/concat_tsv.py \
    -input ${original_tsv},${short_tsv},${med_tsv},${long_tsv},${xlong_tsv} \
    -output ${original_data_root}/${lang_pair}/train_original_synthetic-all4_${task}.tsv

experiment_path=${OUTPUT_ROOT}/${task}_models/${experiment_name}
mkdir -p ${experiment_path}/ckpts

n_cpus=$(eval nproc)
n_gpus=$(nvidia-smi --list-gpus | wc -l)

fairseq-train ${original_data_root}/${src_lang}-de \
--save-dir ${experiment_path}/ckpts/ \
--log-interval 100 \
--config-yaml config_${task}.yaml \
--train-subset train_original_synthetic-all4_${task} \
--valid-subset dev_${task} \
--num-workers $n_cpus \
--data-buffer-size 20 \
--dataset-impl mmap \
--max-tokens $max_tokens \
--max-update 200_000 \
--batch-size $batch_size \
--task speech_to_text \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--scoring wer \
--wer-tokenizer 13a \
--wer-lowercase \
--wer-remove-punct \
--keep-best-checkpoints 10 \
--patience 10 \
--no-last-checkpoints \
--early-stop-metric nll_loss \
--max-tokens-valid $((max_tokens*2)) \
--batch-size-valid $((batch_size*2)) \
--no-epoch-checkpoints \
--best-checkpoint-metric wer \
--arch s2t_transformer_${model_size} \
--optimizer adam \
--lr 0.001 \
--lr-scheduler inverse_sqrt \
--warmup-updates 5000 \
--clip-norm 10.0 \
--seed 42 \
--fp16 \
--no-scale-embedding \
--activation-fn gelu \
--update-freq $((base_update_freq / n_gpus))