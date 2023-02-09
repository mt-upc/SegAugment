#!/bin/bash

dataset_root=$1
src_lang=$2
tgt_lang=$3
ell=$4
mt_data=$5
valid_dir=$6

dataset_name=$(basename $dataset_root)
lang_pair=${src_lang}-${tgt_lang}
valid_split=train_20
ref_dir=${dataset_root}/${lang_pair}/data/${valid_split}/txt

experiment_name=alignment_${dataset_name}_${lang_pair}_${ell}

experiment_path=${OUTPUT_ROOT}/alignment_models/${experiment_name}
mkdir -p $experiment_path

n_cpus=$(eval nproc)
if [ $n_cpus -lt 4 ]; then
    num_workers=0
else
    num_workers=$(($n_cpus / 2))
fi

fairseq-train $mt_data/data-bin \
--save-dir ${experiment_path}/ckpts/ \
--fp16 \
--num-workers $num_workers \
--max-update 400_000 \
--seed 42 \
--max-tokens 14_000 \
--arch transformer_m \
--optimizer adam \
--adam-betas '(0.9, 0.98)' \
--clip-norm 0.0 \
--lr 0.002 \
--lr-scheduler inverse_sqrt \
--warmup-init-lr '1e-07' \
--warmup-updates 2500 \
--dropout 0.1 \
--attention-dropout 0.0 \
--activation-dropout 0.0 \
--criterion label_smoothed_cross_entropy \
--label-smoothing 0.1 \
--eval-bleu-document-level \
--eval-bleu-document-level-ref-segmentation $ref_dir/${valid_split}.yaml \
--eval-bleu-document-level-ref-text $ref_dir/${valid_split}.${tgt_lang} \
--eval-bleu-document-level-hyp-segmentation ${valid_dir}/new.post.yaml \
--eval-bleu-document-level-valid-path ${valid_dir}/data-bin \
--eval-bleu-args '{"beam": 5, "max_len_a": 1.2, "max_len_b": 10, "lenpen": 1}' \
--eval-bleu-detok moses \
--eval-bleu-remove-bpe sentencepiece \
--best-checkpoint-metric bleu \
--maximize-best-checkpoint-metric \
--early-stop-metric bleu \
--maximize-early-stop-metric \
--scoring sacrebleu \
--keep-best-checkpoints 10 \
--patience 20 \
--no-last-checkpoints \
--no-epoch-checkpoints \
--encoder-ffn-embed-dim 2048 \
--decoder-ffn-embed-dim 2048 \
--activation-fn gelu