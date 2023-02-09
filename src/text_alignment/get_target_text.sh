#!/bin/bash

dataset_root=$1
src_lang=$2
tgt_lang=$3
min=$4
max=$5

lang_pair=${src_lang}-${tgt_lang}
ell=${min}-${max}
dataset_name=$(basename $dataset_root)
split=train
vocab_size=8000

synthetic_data_dir=${OUTPUT_ROOT}/synthetic_data/${dataset_name}/${lang_pair}/${ell}/${split}
alternative_mt_data_dir=${OUTPUT_ROOT}/alternative_mt_data/${dataset_name}/${lang_pair}/${ell}

mt_model_name=alignment_${dataset_name}_${lang_pair}_${ell}
mt_ckpt_name=avg_best_10_checkpoint
path_to_mt_ckpt=${OUTPUT_ROOT}/alignment_models/${mt_model_name}/ckpts/${mt_ckpt_name}.pt

if [ $max -gt 9 ]; then
    mt_lenpen=1.5
else
    mt_lenpen=1
fi

# apply vocab
python ${SEGAUGMENT_ROOT}/src/text_alignment/apply_spm.py \
    -m ${alternative_mt_data_dir}/spm_unigram${vocab_size}_${src_lang}.model \
    -i ${synthetic_data_dir}/new.post.${src_lang} \
    -o ${synthetic_data_dir}/new.post_bpe.${src_lang}

# generate binary
fairseq-preprocess \
    --source-lang $src_lang \
    --target-lang $tgt_lang \
    --testpref ${synthetic_data_dir}/new.post_bpe \
    --destdir ${synthetic_data_dir}/data-bin \
    --thresholdsrc 0 \
    --srcdict ${alternative_mt_data_dir}/data-bin/dict.${src_lang}.txt \
    --workers 4 \
    --only-source

# copy the source dictionary
cp ${alternative_mt_data_dir}/data-bin/dict.${tgt_lang}.txt ${synthetic_data_dir}/data-bin/dict.${tgt_lang}.txt

# average 10 best ckpts
n_avg=10
ckpts_dir=$(dirname "$path_to_mt_ckpt")
python ${SEGAUGMENT_ROOT}/src/utils/find_best_ckpts.py -ckpt $ckpts_dir -n $n_avg
python ${FAIRSEQ_ROOT}/scripts/average_checkpoints.py \
    --inputs $(head -n 1 ${ckpts_dir}/best_${n_avg}.txt) \
    --output $path_to_mt_ckpt

# translate
fairseq-generate ${synthetic_data_dir}/data-bin \
    --path $path_to_mt_ckpt \
    --results-path $synthetic_data_dir \
    --batch-size 64 \
    --beam 8 \
    --max-len-a 1.2 \
    --max-len-b 10 \
    --scoring sacrebleu \
    --num-workers 0 \
    --max-tokens 24_000 \
    --gen-subset 'test' \
    --remove-bpe sentencepiece \
    --lenpen $mt_lenpen

# make targets from fairseq-generate
python ${SEGAUGMENT_ROOT}/src/text_alignment/modify_generation_output.py \
    ${synthetic_data_dir}/generate-test.txt \
    ${synthetic_data_dir}/new.post.${tgt_lang}