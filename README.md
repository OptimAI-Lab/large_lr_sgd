# Revisiting the Adam-SGD Gap in LLM Pre-Training: The Role of Large Effective Learning Rates

<a href="https://arxiv.org/abs/2605.17787">
  <img src="https://img.shields.io/static/v1?label=arXiv&message=2605.17787&color=b31b1b" />
</a>

Preliminary code release for our paper "Revisiting the Adam-SGD Gap in LLM Pre-Training: The Role of Large Effective Learning Rates", by Athanasios Glentis, Dawei Li, Chung-Yiu Yau and Mingyi Hong.


## Dataset

Follow `c4.sh` to download the C4 dataset.

## Ray

To activate ray run:

`ray start --head`

## Scripts

We provide the scripts to reproduce our LLaMA results in the `scripts` folder: 
- `train_{model_size}_SGD_LL.sh` for 130M, 350M and 1B model sizes.