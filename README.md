# SegAugment: Maximizing the Utility of Speech Translation Data with Segmentation-based Augmentations

The pre-print of this research is available [here](https://arxiv.org/abs/2212.09699).

<em>
Data scarcity is one of the main issues with the end-to-end approach for Speech Translation, as compared to the cascaded one. Although most data resources for Speech Translation are originally document-level, they offer a sentence-level view, which can be directly used during training. But this sentence-level view is single and static, potentially limiting the utility of the data. Our proposed data augmentation method SegAugment challenges this idea and aims to increase data availability by providing multiple alternative sentence-level views of a dataset. Our method heavily relies on an Audio Segmentation system to re-segment the speech of each document, after which we obtain the target text with alignment methods. The Audio Segmentation system can be parameterized with different length constraints, thus giving us access to multiple and diverse sentence-level views for each document. Experiments in MuST-C show consistent gains across 8 language pairs, with an average increase of 2.2 BLEU points, and up to 4.7 BLEU for lower-resource scenarios in mTEDx. Additionally, we find that SegAugment is also applicable to purely sentence-level data, as in CoVoST, and that it enables Speech Translation models to completely close the gap between the gold and automatic segmentation at inference time.
</em>

![](figures/introduction_solid.jpg)

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

## Augmented Datasets

Here you can download the generated data from SegAugment for MuST-C, mTEDx and CoVoST.

The format is similar to the one found in MuST-C and mTEDx:

* .src: A text file with the transcription for each example
* .tgt: A text file with the translation for each example
* .yaml: A yaml file with the offset, duration and corresponding audio file for each example

MuST-C

|En-De (v2.0)|[short](...)|[medium](...)|[long](...)|[extra-long](...)|
|En-Es (v1.0)|[short](...)|[medium](...)|[long](...)|[extra-long](...)|
|En-Fr (v1.0)|[short](...)|[medium](...)|[long](...)|[extra-long](...)|
|En-It (v1.0)|[short](...)|[medium](...)|[long](...)|[extra-long](...)|
|En-Nl (v1.0)|[short](...)|[medium](...)|[long](...)|[extra-long](...)|
|En-Pt (v1.0)|[short](...)|[medium](...)|[long](...)|[extra-long](...)|
|En-Ro (v1.0)|[short](...)|[medium](...)|[long](...)|[extra-long](...)|
|En-Ru (v1.0)|[short](...)|[medium](...)|[long](...)|[extra-long](...)|
|---|---|---|---|---|---|

mTEDx

|Es-En|[short](https://drive.google.com/drive/folders/1czkMyCoLsVDrvpB3zR_YfG_5-vhtRbAX?usp=sharing)|[medium](https://drive.google.com/drive/folders/1cxUr8rGbtDXlcdxXDIwSzeLYTJ2EWHAR?usp=sharing)|[long](https://drive.google.com/drive/folders/1caJZ6eaVCRz6VA_TXdwb9lM0TZL-mCRP?usp=sharing)|[extra-long](https://drive.google.com/drive/folders/1coS_PzjINObjA9w4O5BPICvTvHQjaogr?usp=sharing)|
|Es-Fr|[short](https://drive.google.com/drive/folders/1cEDr4mgnciG5UCO39fQtKc9J3DNiBfGA?usp=sharing)|[medium](https://drive.google.com/drive/folders/1cEklKZeqepA1I91ee7UCrt6687u1milz?usp=sharing)|[long](https://drive.google.com/drive/folders/1cGfMDQH_r8myKULKUEH9wjNy-lQJen95?usp=sharing)|[extra-long](https://drive.google.com/drive/folders/1cEoUe5N-ncBEDdDDLZIeDh4_j3hJpIYy?usp=sharing)|
|Pt-En|[short](https://drive.google.com/drive/folders/1cEoUe5N-ncBEDdDDLZIeDh4_j3hJpIYy?usp=sharing)|[medium](https://drive.google.com/drive/folders/1cHeh6bZMKrsArawLvKrKK3s4-CjU6pWF?usp=sharing)|[long](https://drive.google.com/drive/folders/1cVlWUH71uw8yxq4oU9wlEyVba5eGJZUt?usp=sharing)|[extra-long](https://drive.google.com/drive/folders/1cHOfSzkhwBnEii_joyLThCTNKDdDaN_P?usp=share_link)|
|Es-Es|[short](https://drive.google.com/drive/folders/1dC2jJaGr5eH9bb80xZinR6ULFZqFO7hh?usp=sharing)|[medium](https://drive.google.com/drive/folders/1d22z7H5kAblhPrqZK3i3RnFEqQU2cCX2?usp=sharing)|[long](https://drive.google.com/drive/folders/1d4XI-fF6HjKFMirEpyY8Nno6fTGJJdLY?usp=sharing)|[extra-long](https://drive.google.com/drive/folders/1d9erH935eVuQ-vkGLee4TWY-Xdrdbwwq?usp=sharing)|
|Pt-Pt|[short](https://drive.google.com/drive/folders/1h0Eu9rWpcP5XCXJGbWSeYvggzZHo_D2b?usp=sharing)|[medium](https://drive.google.com/drive/folders/1h0NWUob8sTlyAyAQw3SQqMTBIGKJQ5nJ?usp=share_link)|[long](https://drive.google.com/drive/folders/1gz9UuB6QohsnVgsKIRRrSAF-CEFoKlWk?usp=share_link)|[extra-long](https://drive.google.com/drive/folders/1gxZBne-x_tM7wHqOoFrOoSbMJ_74vsGO?usp=share_link)|
|---|---|---|---|---|---|

CoVoST

|En-De|[short](https://drive.google.com/drive/folders/1j2qQvjTCQRquEx9JkvC7tbEOsXdJtChe?usp=sharing)|[medium](https://drive.google.com/drive/folders/1j3wTFCvxH5noBkja0gVT0WzTcl3XWsW9?usp=sharing)|-|-|
|---|---|---|---|---|---|

## Code and Instructions for using SegAugment

Under construction...