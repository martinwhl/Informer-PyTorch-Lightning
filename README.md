# Informer-PyTorch-Lightning

[![GitHub stars](https://img.shields.io/github/stars/martinwhl/Informer-PyTorch-Lightning?label=stars&maxAge=2592000)](https://gitHub.com/martinwhl/Informer-PyTorch-Lightning/stargazers/) [![issues](https://img.shields.io/github/issues/martinwhl/Informer-PyTorch-Lightning)](https://github.com/martinwhl/Informer-PyTorch-Lightning/issues) [![License](https://img.shields.io/github/license/martinwhl/Informer-PyTorch-Lightning)](./LICENSE) [![Maintenance](https://img.shields.io/badge/Maintained%3F-yes-green.svg)](https://GitHub.com/martinwhl/Informer-PyTorch-Lightning/graphs/commit-activity) [![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black) ![Codefactor](https://www.codefactor.io/repository/github/martinwhl/Informer-PyTorch-Lightning/badge)

This is a **reorganized** implementation of [Informer](https://arxiv.org/2012.07436) based on [the official implementation](https://github.com/zhouhaoyi/Informer2020) and ⚡ Lightning.

## Requirements

* numpy
* pandas
* scikit-learn
* torch
* lightning>=2.0
* torchmetrics>=0.11

⚠️ The repository is currently based on Lightning 2.0. To use PyTorch Lightning v1.x, please switch to the `pl_v1` branch.

## Model Training

```bash
# template
python main.py --config configs/{DATASET_NAME}/{multi/uni}variate/pred_len_{PRED_LEN}.yaml

# for example
python main.py --config configs/ETTh1/multivariate/pred_len_24.yaml
```
