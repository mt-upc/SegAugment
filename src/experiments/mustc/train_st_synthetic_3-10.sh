#!/bin/bash

task=st
model_size=s
dataset=mustcv2
original_data_root=$MUSTCv2_ROOT
lang_pair=en-de
max_tokens=54000
batch_size=114
base_update_freq=6

experiment_name=${task}_${model_size}_${dataset}_${lang_pair}_med_COORECT

dataset_name=$(basename $original_data_root)
synthetic_data_root=${OUTPUT_ROOT}/synthetic_data/${dataset_name}

original_tsv=${original_data_root}/${lang_pair}/train_st.tsv
med_tsv=${synthetic_data_root}/${lang_pair}/3-10/train/train_3-10_st.tsv

python ${SEGAUGMENT_ROOT}/src/utils/concat_tsv.py \
    -input ${original_tsv},${med_tsv} \
    -output ${original_data_root}/${lang_pair}/train_med_st.tsv \
    -drop

experiment_path=${OUTPUT_ROOT}/${task}_models/${experiment_name}
mkdir -p ${experiment_path}/ckpts

n_cpus=$(eval nproc)
n_gpus=$(nvidia-smi --list-gpus | wc -l)

# n_avg=10
# asr_ckpt_path=${OUTPUT_ROOT}/asr_models/asr_${model_size}_${dataset}_en/ckpts
# asr_checkpoint=${asr_ckpt_path}/avg_best_${n_avg}_checkpoint.pt

# python ${SEGAUGMENT_ROOT}/src/utils/find_best_ckpts.py \
#     -ckpt $asr_ckpt_path -n $n_avg -min
# python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
#     --inputs $(head -n 1 ${asr_ckpt_path}/best_${n_avg}.txt) \
#     --output $asr_checkpoint

asr_checkpoint=${VEUSSD}/segm_augm/asr_models/asr_s_mustc_en/ckpts/avg_best_10_checkpoint.pt


fairseq-train ${original_data_root}/${lang_pair} \
--save-dir ${experiment_path}/ckpts/ \
--log-interval 100 \
--max-update 200_000 \
--config-yaml config_${task}.yaml \
--train-subset train_med_${task} \
--valid-subset dev_${task} \
--num-workers $n_cpus \
--max-tokens $max_tokens \
--batch-size $batch_size \
--max-tokens-valid $((max_tokens*2)) \
--batch-size-valid $((batch_size*2)) \
--task speech_to_text \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--keep-best-checkpoints 10 \
--patience 15 \
--no-epoch-checkpoints \
--no-last-checkpoints \
--best-checkpoint-metric sacrebleu \
--maximize-best-checkpoint-metric \
--scoring sacrebleu \
--early-stop-metric nll_loss \
--arch s2t_transformer_${model_size} \
--optimizer adam \
--lr 0.002 \
--lr-scheduler inverse_sqrt \
--warmup-updates 5000 \
--clip-norm 10.0 \
--seed 42 \
--fp16 \
--no-scale-embedding \
--activation-fn gelu \
--load-pretrained-encoder-from $asr_checkpoint \
--skip-invalid-size-inputs-valid-test \
--data-buffer-size 20 \
--dataset-impl mmap \
--update-freq $((base_update_freq / n_gpus))