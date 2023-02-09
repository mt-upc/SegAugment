#!/bin/bash

dataset_root=$1
src_lang=$2
tgt_lang=$3
min=$4
max=$5
shas_ckpt=$6

lang_pair=${src_lang}-${tgt_lang}
ell=${min}-${max}
dataset_name=$(basename $dataset_root)
split=train
n_tiny=20
tiny_split=${split}_${n_tiny}

### create a tiny train split to monitor the training of the MT alignment model

python ${SEGAUGMENT_ROOT}/src/text_alignment/create_tiny_split.py \
    $src_lang \
    $tgt_lang \
    ${dataset_root}/${lang_pair}/data/${split}/wav \
    ${dataset_root}/${lang_pair}/data/${split}/txt \
    $n_tiny \
    0 \
    60

## apply SegAugment to the tiny split to get source text

bash ${SEGAUGMENT_ROOT}/src/audio_alignment/source_text_pipeline.sh \
    $dataset_root \
    $src_lang \
    $tgt_lang \
    $min \
    $max \
    $shas_ckpt \
    $tiny_split

## create modified MT data by concatenating or splitting the original data

python ${SEGAUGMENT_ROOT}/src/text_alignment/create_modified_mt_data.py \
    -root $dataset_root \
    -out ${OUTPUT_ROOT}/alternative_mt_data/${dataset_name} \
    -l $lang_pair \
    -s $split \
    -min $min \
    -max $max

### prepare the modified MT data for training

mt_data=${OUTPUT_ROOT}/alternative_mt_data/${dataset_name}/${lang_pair}/${ell}
valid_dir=${OUTPUT_ROOT}/synthetic_data/${dataset_name}/${lang_pair}/${ell}/${tiny_split}

bash ${SEGAUGMENT_ROOT}/src/text_alignment/prep_mt_data.sh \
    $mt_data \
    $valid_dir \
    $src_lang \
    $tgt_lang \
    $ell

### train aligment MT model on the modified MT data and monitor the training with the document-level performance on the tiny train set

bash ${SEGAUGMENT_ROOT}/src/text_alignment/train_model.sh \
    $dataset_root \
    $src_lang \
    $tgt_lang \
    $ell \
    $mt_data \
    $valid_dir