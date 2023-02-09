#!/bin/bash

lang=$1
model_size=$2

n_cpus=$(eval nproc)
n_gpus=$(nvidia-smi --list-gpus | wc -l)

task=asr
original_data_root=$MTEDX_ROOT
lang_pair=${lang}-${lang}

dataset_name=$(basename $original_data_root)
experiment_name=${task}_${model_size}_${dataset_name}_${lang}_original

experiment_path=${OUTPUT_ROOT}/asr_models/${experiment_name}
mkdir -p ${experiment_path}/ckpts

if [ $model_size == s ]; then
    max_tokens=54000
    batch_size=114
    base_update_freq=3
    dropout=0.15
elif [ $model_size == xs ]; then
    max_tokens=96000
    batch_size=160
    base_update_freq=2
    dropout=0.1
fi


fairseq-train ${original_data_root}/${lang_pair} \
--save-dir ${experiment_path}/ckpts/ \
--log-interval 100 \
--config-yaml config_${task}.yaml \
--train-subset train_${task} \
--valid-subset valid_${task} \
--num-workers $n_cpus \
--max-tokens $max_tokens \
--batch-size $batch_size \
--max-tokens-valid $((max_tokens*2)) \
--batch-size-valid $((batch_size*2)) \
--max-update 200_000 \
--task speech_to_text \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--scoring wer \
--wer-tokenizer 13a \
--wer-lowercase \
--wer-remove-punct \
--keep-best-checkpoints 10 \
--patience 10 \
--no-epoch-checkpoints \
--no-last-checkpoints \
--best-checkpoint-metric wer \
--early-stop-metric nll_loss \
--arch s2t_transformer_${model_size} \
--optimizer adam \
--lr 0.001 \
--lr-scheduler inverse_sqrt \
--warmup-updates 2000 \
--clip-norm 10.0 \
--seed 42 \
--dropout $dropout \
--fp16 \
--no-scale-embedding \
--activation-fn gelu \
--skip-invalid-size-inputs-valid-test \
--share-decoder-input-output-embed \
--data-buffer-size 20 \
--dataset-impl mmap \
--update-freq $(($base_update_freq / $n_gpus))