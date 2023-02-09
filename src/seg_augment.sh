#!/bin/bash

dataset_root=$1
src_lang=$2
tgt_lang=$3
min=$4
max=$5
shas_ckpt=$6
shas_alg=$7

lang_pair=${src_lang}-${tgt_lang}
ell=${min}-${max}
dataset_name=$(basename $dataset_root)
split=train

wav_dir=${dataset_root}/${lang_pair}/data/${split}/wav
original_yaml=${dataset_root}/${lang_pair}/data/${split}/txt/${split}.yaml
original_src=${dataset_root}/${lang_pair}/data/${split}/txt/${split}.${src_lang}

synthetic_data_dir=${OUTPUT_ROOT}/synthetic_data/${dataset_name}/${lang_pair}/${ell}/${split}

### SEGMENTATION

eval "$(conda shell.bash hook)"
conda activate shas

python $SHAS_ROOT/src/supervised_hybrid/segment.py \
    -wav $wav_dir \
    -ckpt $shas_ckpt \
    -max $max \
    -min $min \
    -alg $shas_alg \
    -cache ${OUTPUT_ROOT}/shas_probabilities/${dataset_name}/${src_lang} \
    -yaml $synthetic_data_dir/new.yaml

eval "$(conda shell.bash hook)"
conda activate seg_augment

### AUDIO ALIGNMENT

forced_alignment_dir=${OUTPUT_ROOT}/forced_alignment/${dataset_name}/${src_lang}

python ${SEGAUGMENT_ROOT}/src/audio_alignment/get_word_segments.py \
    -lang $src_lang \
    -wav $wav_dir \
    -txt $original_src \
    -yaml $original_yaml \
    -out $forced_alignment_dir

### TEXT ALIGNMENT

bash ${SEGAUGMENT_ROOT}/src/text_alignment/get_alignment_model.sh \
  $dataset_root \
  $src_lang \
  $tgt_lang \
  $min \
  $max \
  $shas_ckpt

### GET SOURCE TEXT

python $SEGAUGMENT_ROOT/src/audio_alignment/get_source_text.py \
    -new_yaml $synthetic_data_dir/new.yaml \
    -align $forced_alignment_dir \
    -lang $src_lang

### GET TARGET TEXT

bash ${SEGAUGMENT_ROOT}/src/text_alignment/get_target_text.sh \
  $dataset_root \
  $src_lang \
  $tgt_lang \
  $min \
  $max