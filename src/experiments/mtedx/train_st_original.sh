#!/bin/bash

lang_pair=$1
model_size=$2

n_cpus=$(eval nproc)
n_gpus=$(nvidia-smi --list-gpus | wc -l)

task=st
original_data_root=$MTEDX_ROOT
src_lang=$(echo "$lang_pair" | cut -d'-' -f1)

dataset_name=$(basename $original_data_root)
experiment_name=${task}_${model_size}_${dataset_name}_${lang_pair}_original

experiment_path=${OUTPUT_ROOT}/${task}_models/${experiment_name}
mkdir -p ${experiment_path}/ckpts

n_avg=10
asr_ckpt_path=${OUTPUT_ROOT}/asr_models/asr_${model_size}_${dataset_name}_${src_lang}_original/ckpts
asr_checkpoint=${asr_ckpt_path}/avg_best_${n_avg}_checkpoint.pt

python ${SEGAUGMENT_ROOT}/src/utils/find_best_ckpts.py \
    -ckpt $asr_ckpt_path -n $n_avg -min
python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
    --inputs $(head -n 1 ${asr_ckpt_path}/best_${n_avg}.txt) \
    --output $asr_checkpoint

if [ $model_size == s ]; then
    max_tokens=54000
    batch_size=114
    base_update_freq=3
    dropout=0.15
    max_update=200_000
    warmup_updates=1000
    interval=1
    patience=20
elif [ $model_size == xs ]; then
    max_tokens=96000
    batch_size=160
    base_update_freq=2
    dropout=0.2
    max_update=10_000
    warmup_updates=500
    interval=5
    patience=-1
fi

fairseq-train ${original_data_root}/${lang_pair} \
--save-dir ${experiment_path}/ckpts/ \
--log-interval 100 \
--config-yaml config_${task}.yaml \
--train-subset train_${task} \
--valid-subset valid_${task} \
--num-workers $n_cpus \
--max-update $max_update \
--max-tokens $max_tokens \
--batch-size $batch_size \
--max-tokens-valid $((max_tokens*2)) \
--batch-size-valid $((batch_size*2)) \
--task speech_to_text \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--keep-best-checkpoints 10 \
--patience $patience \
--no-epoch-checkpoints \
--no-last-checkpoints \
--best-checkpoint-metric sacrebleu \
--maximize-best-checkpoint-metric \
--scoring sacrebleu \
--early-stop-metric nll_loss \
--arch s2t_transformer_${model_size} \
--optimizer adam \
--lr-scheduler inverse_sqrt \
--clip-norm 10.0 \
--seed 42 \
--fp16 \
--no-scale-embedding \
--activation-fn gelu \
--load-pretrained-encoder-from $asr_checkpoint \
--skip-invalid-size-inputs-valid-test \
--data-buffer-size 20 \
--dataset-impl mmap \
--warmup-updates $warmup_updates \
--share-decoder-input-output-embed \
--lr 0.001 \
--dropout $dropout \
--save-interval $interval \
--validate-interval $interval \
--update-freq $(($base_update_freq / $n_gpus))