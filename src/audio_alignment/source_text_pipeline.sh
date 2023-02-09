#!/bin/bash

dataset_root=$1
src_lang=$2
tgt_lang=$3
min=$4
max=$5
shas_ckpt=$6
split=$7

lang_pair=${src_lang}-${tgt_lang}
ell=${min}-${max}
dataset_name=$(basename $dataset_root)

wav_dir=${dataset_root}/${lang_pair}/data/${split}/wav
original_yaml=${dataset_root}/${lang_pair}/data/${split}/txt/${split}.yaml
original_src=${dataset_root}/${lang_pair}/data/${split}/txt/${split}.${src_lang}

synthetic_data_dir=${OUTPUT_ROOT}/synthetic_data/${dataset_name}/${lang_pair}/${ell}/${split}

### SHAS SEGMENTATION

eval "$(conda shell.bash hook)"
conda activate shas

shas_probabilities_dir=${OUTPUT_ROOT}/shas_probabilities/${dataset_name}/${src_lang}

if [ $max -gt 20 ]; then
    alg=pstrm
else
    alg=pdac
fi

python $SHAS_ROOT/src/supervised_hybrid/segment.py \
    -wav $wav_dir \
    -ckpt $shas_ckpt \
    -max $max \
    -min $min \
    -alg $alg \
    -cache $shas_probabilities_dir \
    -yaml $synthetic_data_dir/new.yaml

eval "$(conda shell.bash hook)"
conda activate seg_augment

### FORCED ALIGNMENT

forced_alignment_dir=${OUTPUT_ROOT}/forced_alignment/${dataset_name}/${src_lang}

python ${SEGAUGMENT_ROOT}/src/audio_alignment/get_word_segments.py \
    -lang $src_lang \
    -wav $wav_dir \
    -txt $original_src \
    -yaml $original_yaml \
    -out $forced_alignment_dir

### SOURCE TEXT

python $SEGAUGMENT_ROOT/src/audio_alignment/get_source_text.py \
    -new_yaml $synthetic_data_dir/new.yaml \
    -align $forced_alignment_dir \
    -lang $src_lang