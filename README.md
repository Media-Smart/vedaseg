## Installation
### Requirements

- Linux
- Python 3.7+
- PyTorch 1.1.0 or higher
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 9.0
- Python 3.7.3

### Install vedaseg

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedaseg python=3.7 -y
conda activate vedaseg
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), e.g.,

```shell
conda install pytorch torchvision -c pytorch
```

c. Clone the vedaseg repository.

```shell
git clone https://github.com/Media-Smart/vedaseg.git
cd vedaseg
vedaseg_root=${PWD}
```

d. Install dependencies.

```shell
pip install -r requirements.txt
```

## Prepare data
It is recommended to symlink the dataset root to `${vedaseg_root}/data`, and write your own dataset class in `vedaseg/datasets`

## Train

a. Config

Modify some configuration accordingly in the config file like `configs/unet.py`

b. Run

```shell
python tools/trainval.py configs/unet.py
```

## Performance
Note: All models are evaluated on PASCAL VOC 2012 val set.

| Architecture | backbone | OS | MS & Flip | mIOU |
|:---:|:---:|:---:|:---:|:---:|
| deeplabv3plus | resnet101 | 16 | True | 79.80% |
| deeplabv3plus | resnet101 | 16 | False | 78.19% |
| deeplabv3 | resnet101 | 16 | True | 78.94% |
| deeplabv3 | resnet101 | 16 | False | 77.07% |
| FPN | resnet101 | 2 | True | 75.42% |
| FPN | resnet101 | 2 | False | 73.65% |
| PSPNet | resnet101 | 8 | True | 74.68% |
| PSPNet | resnet101 | 8 | False | 73.71% |
| unet | resnet101 | 1 | True | 73.09% |
| unet | resnet101 | 1 | False | 70.98% |

OS: Output stride used during evaluation\
MS: Multi-scale inputs during evaluation\
Flip: Adding left-right flipped inputs during evaluation

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks to [open-mmlab](https://github.com/open-mmlab).
