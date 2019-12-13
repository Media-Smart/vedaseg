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
git clone https://github.com/mileistone/vedaseg.git
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

## Credits
I got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks to [open-mmlab](https://github.com/open-mmlab).
