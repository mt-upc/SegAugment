#!/bin/bash

train_root=$1
valid_root=$2
src_lang=$3
tgt_lang=$4
length_tag=$5

vocab_size=8000

for lang in $src_lang $tgt_lang; do
    python ${SEGAUGMENT_ROOT}/src/text_alignment/learn_spm.py \
        $train_root/train_${length_tag}.${lang} \
        $train_root/spm_unigram${vocab_size}_${lang} \
        unigram \
        $vocab_size
done

# apply vocab
for lang in $src_lang $tgt_lang; do
    python ${SEGAUGMENT_ROOT}/src/text_alignment/apply_spm.py \
        -m $train_root/spm_unigram${vocab_size}_${lang}.model \
        -i $train_root/train_${length_tag}.${lang} \
        -o $train_root/train_${length_tag}_bpe.${lang}
done

# generate binary
fairseq-preprocess \
    --source-lang $src_lang \
    --target-lang $tgt_lang \
    --trainpref $train_root/train_${length_tag}_bpe \
    --destdir $train_root/data-bin \
    --thresholdsrc 0 \
    --thresholdtgt 0 \
    --workers 4

# apply vocab
python ${SEGAUGMENT_ROOT}/src/text_alignment/apply_spm.py \
    -m $train_root/spm_unigram${vocab_size}_${src_lang}.model \
    -i ${valid_root}/new.post.${src_lang} \
    -o ${valid_root}/new.post_bpe.${src_lang}

# generate binary
fairseq-preprocess \
    --source-lang $src_lang \
    --target-lang $tgt_lang \
    --validpref $valid_root/new.post_bpe \
    --destdir $valid_root/data-bin \
    --thresholdsrc 0 \
    --srcdict $train_root/data-bin/dict.${src_lang}.txt \
    --workers 4 \
    --only-source

cp $train_root/data-bin/dict.${tgt_lang}.txt ${valid_root}/data-bin/dict.${tgt_lang}.txt