---
license: apache-2.0
metrics:
- mse
- mae
tags:
- time series
- forecasting
- foundation models
- pretrained models
- generative models
- time series foundation models
library_name: transformers
---

# SymTime NeurIPS 2025

This code is the official PyTorch implementation of our NeurIPS'25 paper: **Synthetic Series-Symbol Data Generation for Time Series Foundation Models**.

<div align="center">

[![NeurIPS](https://img.shields.io/badge/NeurIPS'25-SymTime-orange)](https://neurips.cc/virtual/2025/poster/115260) [![PyPI version](https://badge.fury.io/py/s2generator.svg)](https://pypi.org/project/s2generator/) [![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/) [![PyTorch](https://img.shields.io/badge/PyTorch-2.0.1-blue)](https://pytorch.org/)

[Paper](https://arxiv.org/abs/2510.08445) | [Poster](https://github.com/wwhenxuan/wwhenxuan.github.io/blob/main/assets/img/poster_neurips_2025_115260_synthetic_series-symbol_data_generation.jpg) | [Blog](https://mp.weixin.qq.com/s/D6O5SBl2RYHdkiinV6UM8w) | [Video](https://www.bilibili.com/video/BV1RT4QzXECt/?spm_id_from=333.337.search-card.all.click) | [PPT](https://github.com/wwhenxuan/wwhenxuan.github.io/blob/main/assets/files/NeurIPS_2025_SymTime_video_en.pptx) | [Citation](#Citation) | [HF 🤗](https://huggingface.co/FlowVortex/SymTime)

</div>

This repository contains the official Hugging Face / PyTorch implementation of **SymTime** from our NeurIPS 2025 paper, *Synthetic Series-Symbol Data Generation for Time Series Foundation Models*.

## Overview

SymTime is a lightweight time series foundation model designed to learn strong temporal representations from patch-based inputs. It is built for practical downstream use and supports easy loading through the Hugging Face `AutoModel` interface.

<div style="text-align: center;">
    <img src="https://raw.githubusercontent.com/wwhenxuan/SymTime/main/configs/images/S2Generator_SymTime.png" alt="SymTime" style="zoom:80%;" />
</div>

The model takes a univariate time series, splits it into patches, and encodes the patch sequence with a transformer backbone. The repository includes the configuration, model definition, and a runnable example for inference.

## Quick start

### Install dependencies

```bash
pip install -r requirements.txt
```

### Load the model

```python
from transformers import AutoModel

model = AutoModel.from_pretrained("FlowVortex/SymTime", trust_remote_code=True)
```

### Run inference

```python
import torch

x = torch.randn(16, 256)
out = model(x)
out_no_cls = model(x, return_cls_token=False)
```

## Model summary

- Input: `Tensor` with shape `[batch_size, seq_length]`
- Output: patch embeddings, optionally with a CLS token output
- Backend: patch-based transformer encoder

## Citation <a id="Citation"></a>

If you find this code useful, please cite our paper.

```
@misc{wang2025syntheticseriessymboldatageneration,
      title={Synthetic Series-Symbol Data Generation for Time Series Foundation Models}, 
      author={Wenxuan Wang and Kai Wu and Yujian Betterest Li and Dan Wang and Xiaoyu Zhang},
      year={2025},
      eprint={2510.08445},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2510.08445}, 
}
```

## Contact

If you have any questions or are interested in our view on the complex dynamics of time series, feel free to contact:

- [Whenxuan Wang](https://wwhenxuan.github.io/) (whenxuanwang@stu.xidian.edu.cn)
- [Kai Wu](https://sparsel.github.io/index.html) (kwu@xidian.edu.cn)
- [Dan Wang](https://web.xidian.edu.cn/danwang/) (danwang@xidian.edu.cn)

## Acknowledgement

We appreciate the following GitHub repos a lot for their valuable code and efforts.

- Time-Series-Library (https://github.com/thuml/Time-Series-Library)
- PySDKit (https://github.com/wwhenxuan/PySDKit)
- ALBEF (https://github.com/salesforce/ALBEF)
- PatchTST (https://github.com/yuqinie98/PatchTST)
- Short-term Forecasting (https://github.com/ServiceNow/N-BEATS)
