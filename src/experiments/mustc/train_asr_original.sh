#!/bin/bash

original_data_root=$MUSTCv2_ROOT
task=asr
model_size=s
src_lang=en
max_tokens=54000
batch_size=114
base_update_freq=6

dataset_name=$(basename $original_data_root)
experiment_name=${task}_${model_size}_${dataset_name}_${src_lang}_original

experiment_path=${OUTPUT_ROOT}/${task}_models/${experiment_name}
mkdir -p ${experiment_path}/ckpts

n_cpus=$(eval nproc)
n_gpus=$(nvidia-smi --list-gpus | wc -l)

fairseq-train ${original_data_root}/${src_lang}-de \
--save-dir ${experiment_path}/ckpts/ \
--log-interval 100 \
--config-yaml config_${task}.yaml \
--train-subset train_${task} \
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