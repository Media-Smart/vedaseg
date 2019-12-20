## Introduction
vedaseg is an open source semantic segmentation framework based on PyTorch.

## Features

- **Modular Design**

  We decompose the semantic segmentation framework into different components. The flexible and extensible design make it easy to implement a customized semantic segmentation project by combining different modules like building Lego.

- **Support of several popular frameworks**

  The toolbox supports several popular and semantic segmentation frameworks out of box, *e.g.* DeepLabv3+, DeepLabv3, UNet, PSPNet, FPN, etc.

## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

Note: All models are evaluated on PASCAL VOC 2012 val set.

| Architecture | backbone | OS | MS & Flip | mIOU|
|:---:|:---:|:---:|:---:|:---:|
| DeepLabv3plus | ResNet-101 | 16 | True | 79.80% |
| DeepLabv3plus | ResNet-101 | 16 | False | 78.19% |
| DeepLabv3 | ResNet-101 | 16 | True | 78.94% |
| DeepLabv3 | ResNet101 | 16 | False | 77.07% |
| FPN | ResNet-101 | 2 | True | 75.42% |
| FPN | ResNet-101 | 2 | False | 73.65% |
| PSPNet | ResNet-101 | 8 | True | 74.68% |
| PSPNet | ResNet-101 | 8 | False | 73.71% |
| U-Net | ResNet-101 | 1 | True | 73.09% |
| U-Net | ResNet-101 | 1 | False | 70.98% |

OS: Output stride used during evaluation\
MS: Multi-scale inputs during evaluation\
Flip: Adding left-right flipped inputs during evaluation

Models above are available in the [GoogleDrive](https://drive.google.com/drive/folders/1ooIOX5Aeu-0aHJYT1eZgzkSnZUvPi2by).

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
Download [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [Pascal VOC 2012 augmented](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz), resulting in 10,582 training images(trainaug), 1,449 validatation images.

```
cd ${vedaseg_root}
mkdir ${vedaseg_root}/data
cd ${vedaseg_root}/data

wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
wget http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz

tar xf VOCtrainval_11-May-2012.tar
tar xf benchmark.tgz

python ../tools/encode_voc12_aug.py
python ../tools/encode_voc12.py

mkdir VOCdevkit/VOC2012/EncodeSegmentationClass
cp benchmark_RELEASE/dataset/encode_cls/* VOCdevkit/VOC2012/EncodeSegmentationClass
cp VOCdevkit/VOC2012/EncodeSegmentationClassPart/* VOCdevkit/VOC2012/EncodeSegmentationClass
```

## Train

a. Config

Modify some configuration accordingly in the config file like `configs/unet.py`

b. Run

```shell
python tools/trainval.py configs/unet.py
```

## Test

a. Config

Modify some configuration accordingly in the config file like `configs/unet.py`

b. Run

```shell
python tools/test.py configs/unet.py path_to_unet_weights
```

## Contact

This repo is currently maintained by Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/yhcao6)).

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks to [open-mmlab](https://github.com/open-mmlab).
