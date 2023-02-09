# SegAugment: Maximizing the Utility of Speech Translation Data with Segmentation-based Augmentations

The pre-print of this research is available [here](https://arxiv.org/abs/2212.09699).

<em>
Data scarcity is one of the main issues with the end-to-end approach for Speech Translation, as compared to the cascaded one. Although most data resources for Speech Translation are originally document-level, they offer a sentence-level view, which can be directly used during training. But this sentence-level view is single and static, potentially limiting the utility of the data. Our proposed data augmentation method SegAugment challenges this idea and aims to increase data availability by providing multiple alternative sentence-level views of a dataset. Our method heavily relies on an Audio Segmentation system to re-segment the speech of each document, after which we obtain the target text with alignment methods. The Audio Segmentation system can be parameterized with different length constraints, thus giving us access to multiple and diverse sentence-level views for each document. Experiments in MuST-C show consistent gains across 8 language pairs, with an average increase of 2.2 BLEU points, and up to 4.7 BLEU for lower-resource scenarios in mTEDx. Additionally, we find that SegAugment is also applicable to purely sentence-level data, as in CoVoST, and that it enables Speech Translation models to completely close the gap between the gold and automatic segmentation at inference time.
</em>

## Contents
- [Download Synthetic Datasets](#download-synthetic-datasets)
- [Usage](#usage)
  - [Environment](#setting-up-the-environment)
  - [Download and Prepare the Original Data](#original-data)
    - [MuST-C](#must-c)
    - [mTEDx](#mtedx)
  - [Create Synthetic Data with SegAugment](#create-synthetic-data-with-segaugment)
    - [Segmentation](#step-1-segmentation)
    - [Audio Alignment](#step-2-audio-alignment)
    - [Text Alignment](#step-3-text-alignment)
    - [Source and Target text](#combining-the-previous-steps)
  - [Experiments](#train-with-synthetic-data-from-segaugment)
- [Citation](#citation)

## Download Synthetic Datasets

Here you can download the generated data from SegAugment for MuST-C, mTEDx and CoVoST.

The format is similar to the one found in MuST-C and mTEDx:

* .src: A text file with the transcription for each example
* .tgt: A text file with the translation for each example
* .yaml: A yaml file with the offset, duration and corresponding audio file for each example

MuST-C

|En-De (v2.0)|[short](https://drive.google.com/uc?export=download&id=1v5B_974Xp_UXjy8VH5eXZgzqHHSGduRZ)|[medium](https://drive.google.com/uc?export=download&id=1v4wQyhH6laOgCCVW_WS-5Jy9w-KnCQY9)|[long](https://drive.google.com/uc?export=download&id=1uyKf0E6b8NhCEfJzrUxtTrmHViKq5vDC)|[extra-long](https://drive.google.com/uc?export=download&id=1usaybFjJC-gu4ARxosBzZjNlJNCCC2WX)|
|---|---|---|---|---|
|En-Es|[short](https://drive.google.com/uc?export=download&id=1sbNhHpfR_IIfUWvbb7g2FxMP_sn0DKkl)|[medium](https://drive.google.com/uc?export=download&id=1s_ZGShc1WODNgrt1UT87DkhdGKXvweVx)|[long](https://drive.google.com/uc?export=download&id=1sQO0iYEfbNUdS11rDMCb9YGTOZMIl6DH)|[extra-long](https://drive.google.com/uc?export=download&id=1sLPjnbpaMnmvOmYMnCIRviuIWtiuZCNG)|
|En-Fr|[short](https://drive.google.com/uc?export=download&id=1szYcZfP9pFrAA6KN4oTpc5HouxAqxb8u)|[medium](https://drive.google.com/uc?export=download&id=1snpZjKGYRPJ3zu0f2HNWHgihByAQBklk)|[long](https://drive.google.com/uc?export=download&id=1sesgB_MUNnR5pHtuULn3kBLilHuZtctF)|[extra-long](https://drive.google.com/uc?export=download&id=1sgb9Am1doggPsVfeDrMCs6Y6qvpN4uiz)|
|En-It|[short](https://drive.google.com/uc?export=download&id=1tEDm93gCyAJUU0-UnsomVRGPJeXu0nxU)|[medium](https://drive.google.com/uc?export=download&id=1t8H1Bvid0wO4deWbm0YUNmnRsycCnSbl)|[long](https://drive.google.com/uc?export=download&id=1tD6N5wzFH9EgLCAA6MvAxXjvnd6YevQ9)|[extra-long](https://drive.google.com/uc?export=download&id=1snzkp73RPjZ8nGTcPdomiay0lQWoKoep)|
|En-Nl|[short](https://drive.google.com/uc?export=download&id=1tUVRKhhl7saQGsVDIeHfbMpzgTcWK-xq)|[medium](https://drive.google.com/uc?export=download&id=1tOnmjaaerpvNi3dd0UqYgUD35pOYWCAp)|[long](https://drive.google.com/uc?export=download&id=1tP4cAyDFXBj6ixDHff1CoOcxF2epaebS)|[extra-long](https://drive.google.com/uc?export=download&id=1tKnGKjOk6l4IlJjH22pHb7CTlk3_h68w)|
|En-Pt|[short](https://drive.google.com/uc?export=download&id=1ts_GRYUbLSGJcWwMTCMORJqRgUjDJLnD)|[medium](https://drive.google.com/uc?export=download&id=1tv8RKkC9BSHo4OyWmtrzLSux1eVlqUWo)|[long](https://drive.google.com/uc?export=download&id=1tuevLRQDSaJ0qTsP2kCla8uybMz2FKPk)|[extra-long](https://drive.google.com/uc?export=download&id=1taZXQ-dY_4aMjVr4higdkbebaT-uBI66)|
|En-Ro|[short](https://drive.google.com/uc?export=download&id=1uJNAqL4Cg5EE9Nup_eHCl7-WFYJ4qC8M)|[medium](https://drive.google.com/uc?export=download&id=1uC7rE6YiHimgJKgVBgZ0_N0t-YDzwuBR)|[long](https://drive.google.com/uc?export=download&id=1uAHFxmvP5Nyxgqy5ivyeFiBzJeDQd03Z)|[extra-long](https://drive.google.com/uc?export=download&id=1u-80RjJy3fpf_1Sm8inKOW-gIknpdvep)|
|En-Ru|[short](https://drive.google.com/uc?export=download&id=1us_0aZab16a_Lz_EH9fDEYVjTCJeuRkg)|[medium](https://drive.google.com/uc?export=download&id=1uWWe5rjFpK019ISi3oS18UWsJ7WodH2Z)|[long](https://drive.google.com/uc?export=download&id=1uSk9V9W0R-bsygzKxEXx_c6C-qeInHGR)|[extra-long](https://drive.google.com/uc?export=download&id=1uJrLlydTW_UjfydFuHqJ_2-8BUtOjm06)|

mTEDx

|Es-En|[short](https://drive.google.com/uc?export=download&id=1r_NltYeAoimyJIFgub8MJ9f6zESq-VCN)|[medium](https://drive.google.com/uc?export=download&id=1rXuli-c1bdE0pnF0k2e1Rmgo73kxKLA3)|[long](https://drive.google.com/uc?export=download&id=1rZt9-PpbsWkZxFGaImqzJkvuzZ4E8Ehw)|[extra-long](https://drive.google.com/uc?export=download&id=1rTRtSS9HMx6Sb_uupGN4cshdNQ142JWC)|
|---|---|---|---|---|
|Es-Fr|[short](https://drive.google.com/uc?export=download&id=1rtC4gGn1AeNPNjKY-ckvheS4Yo06h-NU)|[medium](https://drive.google.com/uc?export=download&id=1rTRtSS9HMx6Sb_uupGN4cshdNQ142JWC)|[long](https://drive.google.com/uc?export=download&id=1rlYgpBIE0zECVWCcc0SWxdhV2Phe_M18)|[extra-long](https://drive.google.com/uc?export=download&id=1rozUGTFvWg4EmCNDUOwXuWeR5FnNdCLn)|
|Pt-En|[short](https://drive.google.com/uc?export=download&id=1sB0haJofpIvtcXAuEqnEHnEjuSEcMcJI)|[medium](https://drive.google.com/uc?export=download&id=1sIc9nbQYaN-TaJNCXeZlexs2xYNZjQCb)|[long](https://drive.google.com/uc?export=download&id=1sB27P4r4VjCtHFnEI7Jf1LhgI5jaQpc4)|[extra-long](https://drive.google.com/uc?export=download&id=1rq4n_wlMijHXZ6zmxlc7o4ypVf_dUMzu)|
|Es-Es|[short](https://drive.google.com/uc?export=download&id=1rxez5cbcRjHVPVktvtIfagIL8P-9XCw2)|[medium](https://drive.google.com/uc?export=download&id=1rq4GPYtTzWX_-BfwMAQr1miCNLmCycYa)|[long](https://drive.google.com/uc?export=download&id=1reTKhYAleALZZRVOdC9YG0JXEl_nvavT)|[extra-long](https://drive.google.com/uc?export=download&id=1rd0pYjijNWUlJVrFVri8O3ODsuEo3U0R)|
|Pt-Pt|[short](https://drive.google.com/uc?export=download&id=1sXhFDtj508ABBYA8YV2zjcTo5m1KnsH6)|[medium](https://drive.google.com/uc?export=download&id=1sNM5qomeguAO89aNK8Yudw2R6v0Ggr87)|[long](https://drive.google.com/uc?export=download&id=1sPjUUEyo17dZpOtTIem-F7O5q1OZEM5I)|[extra-long](https://drive.google.com/uc?export=download&id=1sJa-rERFPChwUAMKsONGMY1-w3w3-_73)|

CoVoST

|En-De|[short](https://drive.google.com/uc?export=download&id=1v98N0eUS5cKDkvcEmN3I9tAAinaylKqasharing)|[medium](https://drive.google.com/uc?export=download&id=1v9hvRhoJWfi7ASBtvYzVQRN1wcecXMgf)|
|---|---|---|

To use the data for Speech Translation you will have to also download the original audio files for each dataset.

## Usage

### Setting up the environment

Set the environment variables:

```bash
export SEGAUGMENT_ROOT=...          # the path to this repo
export OUTPUT_ROOT=...              # the path to save the outputs of SegAugment, including synthetic data, alignments, and models
export FAIRSEQ_ROOT=...             # the path to our fairseq fork
export SHAS_ROOT=...                # the path to the SHAS repo
export SHAS_CKPTS=...               # the path to the pre-trained SHAS classifier checkpoints
export MUSTCv2_ROOT=...             # the path to save MuST-C v2.0
export MUSTCv1_ROOT=...             # the path to save MuST-C v1.0
export MTEDX_ROOT=...               # the path to save mTEDx
```

Clone this repository to `$SEGAUGMENT_ROOT`:

```bash
git clone https://github.com/mt-upc/SegAugment.git ${SEGAUGMENT_ROOT}    
```

Create a conda environment using the `environment.yml` file and activate it:

```bash
conda env create -f ${SEGAUGMENT_ROOT}/environment.yml && \
conda activate seg_augment
```

Install our fork of fairseq:

```bash
git clone -b SegAugment https://github.com/mt-upc/fairseq-internal.git ${FAIRSEQ_ROOT}
pip install --editable ${FAIRSEQ_ROOT}
```

Clone the SHAS repository to `$SHAS_ROOT`:

```bash
git clone -b experimental https://github.com/mt-upc/SHAS.git ${SHAS_ROOT}    
```

Create a second conda environment for SHAS (no need to activate it for now):

```bash
conda env create -f ${SHAS_ROOT}/environment.yml
```

Download the English and Multilingual pre-trained SHAS classifier and save at `$SHAS_CKPTS`:

|[English](https://drive.google.com/u/0/uc?export=download&confirm=DOjP&id=1Y7frjVkB_85snZYHTn0PQQG_kC5afoYN)|[Multilingual](https://drive.google.com/u/0/uc?export=download&confirm=x9hB&id=1GzwhzbHBFtwDmQPKoDOdAfESvWBrv_wB)|
|---|---|


### Original Data

For our main experiments we used MuST-C and mTEDx. Follow the instructions here to download and prepare the original data.

#### MuST-C

Download MuST-C v2.0 En-De to `$MUSTCv2_ROOT` and the v1.0 En-X to `$MUSTCv1_ROOT`:\
The dataset is available [here](https://ict.fbk.eu/must-c/). Press the bottom ”click here to download the corpus”, and select version V1 and V2 accordingly.

To prepare the data for training, run the following processing scripts. (We are also using the ASR data from v2.0 for pre-training.)

```bash
python ${FAIRSEQ_ROOT}/examples/speech_to_text/prep_mustc_data.py \
  --data-root ${MUSTCv2_ROOT} --task asr --vocab-type unigram --vocab-size 5000

for root in $MUSTCv2_ROOT $MUSTCv1_ROOT; do
  python ${FAIRSEQ_ROOT}/examples/speech_to_text/prep_mustc_data.py \
    --data-root $root --task st --vocab-type unigram --vocab-size 8000
done
```

#### mTEDx

Download the mTEDx Es-En, Es-Pt, Es-Fr and Es, Pt ASR data to `$MTEDX_ROOT` and run the processing scripts to prepare them:

```bash
mkdir -p ${MTEDX_ROOT}/log_dir
for lang_pair in {es-en,pt-en,es-fr,es-es,pt-pt}; do
  wget https://www.openslr.org/resources/100/mtedx_${lang_pair}.tgz -o ${MTEDX_ROOT}/log_dir/${lang_pair} -c -b -O - | tar -xz -C ${MTEDX_ROOT}
done
```

```bash
python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task asr --vocab-type unigram --vocab-size 5000 --lang-pairs es-es,pt-pt

python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task st --vocab-type unigram --vocab-size 8000 --lang-pairs es-en,pt-en

python examples/speech_to_text/prep_mtedx_data.py \
  --data-root ${MTEDX_ROOT} --task st --vocab-type unigram --vocab-size 1000 --lang-pairs es-fr
```

### Create Synthetic Data with SegAugment

Set up some useful parameters:

```bash
dataset_root=...      # the path to the dataset you want to augment
src_lang=...          # the source language id (eg. "en")
tgt_lang=...          # the target language id (eg. "de")
min=...               # the minimum segment length in seconds
max=...               # the maximum segment length in seconds
shas_ckpt=...         # the path to the pre-trained SHAS classifier ckpt (English/Multilingual)
shas_alg=...          # the type of segmentation algorithm (use "pdac" in general, and "pstrm" for max > 20)
```

The following script will execute all steps of SegAugment in sequence and create the synthetic data for a given dataset.

```bash
bash ${SEGAUGMENT_ROOT}/src/seg_augment.sh \
  $dataset_root $src_lang $tgt_lang $min $max $shas_ckpt $shas_alg
```

However since most steps can be done on parallel it is not very efficient. It is advisable to run the above command only after you have completed one round of augmentation with `$min`-`$max` since intermediate results will be cached.

The following steps can be run in parallel:

#### Step 1: Segmentation

Get an alternative segmentation for each document in the training set with SHAS.

```bash
conda activate shas

synthetic_data_dir=${OUTPUT_ROOT}/synthetic_data/${dataset_name}/${lang_pair}/${ell}/${split}

python $SHAS_ROOT/src/supervised_hybrid/segment.py \
    -wav ${dataset_root}/${lang_pair}/data/${split}/wav \
    -ckpt $shas_ckpt \
    -max $max \
    -min $min \
    -alg $alg \
    -cache ${OUTPUT_ROOT}/shas_probabilities/${dataset_name}/${src_lang} \
    -yaml $synthetic_data_dir/new.yaml

conda activate seg_augment
```

#### Step 2: Audio Alignment

Get the word segments for each document in the training set with CTC-based forced-alignment.

```bash
forced_alignment_dir=${OUTPUT_ROOT}/forced_alignment/${dataset_name}/${src_lang}

python ${SEGAUGMENT_ROOT}/src/audio_alignment/get_word_segments.py \
    -lang $src_lang \
    -wav ${dataset_root}/${lang_pair}/data/${split}/wav \
    -txt ${dataset_root}/${lang_pair}/data/${split}/txt/${split}.${src_lang} \
    -yaml ${dataset_root}/${lang_pair}/data/${split}/txt/${split}.yaml \
    -out $forced_alignment_dir
```

#### Step 3: Text Alignment

Learn the text alignment in the training set with an MT model.

```bash
bash ${SEGAUGMENT_ROOT}/src/text_alignment/get_alignment_model.sh \
  $dataset_root $src_lang $tgt_lang $min $max $shas_ckpt
```

#### Combining the previous steps

When all three steps are completed, get the synthetic transcriptions and translations:

```bash
python ${SEGAUGMENT_ROOT}/src/audio_alignment/get_source_text.py \
  -new_yaml $synthetic_data_dir/new.yaml -align $forced_alignment_dir -lang $src_lang

bash ${SEGAUGMENT_ROOT}/src/text_alignment/get_target_text.sh \
  $dataset_root $src_lang $tgt_lang $min $max
```

* The output is stored at `$OUTPUT_ROOT/synthetic_data/<dataset_name>/<lang_pair>/<ell>/train` and is the same as the files available to download at [this section](#synthetic-datasets).;
* The process can be repeated for different `$min`-`$max`. Several intermediate steps are cached, so that another augmentation is faster.;
* Segmentation and audio alignments do not have to be repeated for the same dataset but with a different target language.;
* The above scripts would work for any dataset that has the same file structure as MuST-C or mTEDx. This is `$DATASET_ROOT/<lang_pair>/data/<split>/txt/<split>.{src,tgt,yaml}`. Modifications would be required for other structures.

To use the synthetic data for training ST models, you need to run a processing script that creates a tsv file, similar to the one for the [orginal data](#data). The process is much faster when more than 8 CPU cores are available.

```bash
bash ${SEGAUGMENT_ROOT}/src/utils/prep_synthetic_tsv.sh \
  -data $dataset_root -src $src_lang -tgt $tgt_lang -ell ${min}-${max}
```

### Train with Synthetic Data from SegAugment

Example for MuST-C v2.0 En-De.

ASR pre-training on the original data:

```bash
bash $SEGAUGMENT_ROOT/src/experiments/mustc/train_asr_original.sh
```

ST training with the original and synthetic data (short, medium, long, xlong):

```bash
bash $SEGAUGMENT_ROOT/src/experiments/mustc/train_st_synthetic-all4.sh en-de
```

Example for mTEDx Es-En. (Use the "xs" model for Es-Fr)

For the low-resource pairs of mTEDx we found that ASR pre-training with the synthetic data was very beneficial:

```bash
bash $SEGAUGMENT_ROOT/src/experiments/mtedx/train_asr_synthetic-all4.sh es-es s
```

ST training with the original and synthetic data (short, medium, long, xlong):

```bash
bash $SEGAUGMENT_ROOT/src/experiments/mtedx/train_st_synthetic-all4.sh es-en s
```

## Citation

```bash
@misc{https://doi.org/10.48550/arxiv.2212.09699,
  doi = {10.48550/ARXIV.2212.09699},
  url = {https://arxiv.org/abs/2212.09699},
  author = {Tsiamas, Ioannis and Fonollosa, José A. R. and Costa-jussà, Marta R.},
  keywords = {Computation and Language (cs.CL), Sound (cs.SD), Audio and Speech Processing (eess.AS)},
  title = {{SegAugment: Maximizing the Utility of Speech Translation Data with Segmentation-based Augmentations}},
  publisher = {arXiv},
  year = {2022},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
```
