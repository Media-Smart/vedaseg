## Introduction
vedaseg is an open source semantic segmentation toolbox based on PyTorch.

## Features

- **Modular Design**

  We decompose the semantic segmentation framework into different components. The flexible and extensible design make it easy to implement a customized semantic segmentation project by combining different modules like building Lego.

- **Support of several popular frameworks**

  The toolbox supports several popular and semantic segmentation frameworks out of box, *e.g.* DeepLabv3+, DeepLabv3, U-Net, PSPNet, FPN, etc.

- **Deployment and acceleration**

  The toolbox can automatically transform and accelerate PyTorch, Onnx and Tensorflow models with TensorRT, can also automatically generate benchmark with given model.

- **Different training modes**
    
  The toolbox supports both single-label training and multi-label training.
  
## License

This project is released under the [Apache 2.0 license](LICENSE).

## Benchmark and model zoo

Note: All models are trained only on PASCAL VOC 2012 trainaug dataset and evaluated on PASCAL VOC 2012 val dataset.

| Architecture | backbone | OS | MS & Flip | mIOU |
|:---:|:---:|:---:|:---:|:---:|
| DeepLabv3plus | ResNet-101 | 16 | True | 79.46% |
| DeepLabv3plus | ResNet-101 | 16 | False | 77.90% |
| DeepLabv3 | ResNet-101 | 16 | True | 79.22% |
| DeepLabv3 | ResNet101 | 16 | False | 77.08% |
| FPN | ResNet-101 | 2 | True | 76.19% |
| FPN | ResNet-101 | 2 | False | 74.26% |
| PSPNet | ResNet-101 | 8 | True | 74.83% |
| PSPNet | ResNet-101 | 8 | False | 73.28% |
| U-Net | ResNet-101 | 1 | True | 73.89% |
| U-Net | ResNet-101 | 1 | False | 72.21% |

OS: Output stride used during evaluation\
MS: Multi-scale inputs during evaluation\
Flip: Adding left-right flipped inputs during evaluation

Models above are available in the [GoogleDrive](https://drive.google.com/drive/folders/1ooIOX5Aeu-0aHJYT1eZgzkSnZUvPi2by).

## Installation
### Requirements

- Linux
- Python 3.6+
- PyTorch 1.4.0 or higher
- CUDA 9.0 or higher

We have tested the following versions of OS and softwares:

- OS: Ubuntu 16.04.6 LTS
- CUDA: 10.2
- PyTorch 1.4.0
- Python 3.6.9

### Install vedaseg

a. Create a conda virtual environment and activate it.

```shell
conda create -n vedaseg python=3.6.9 -y
conda activate vedaseg
```

b. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

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
### VOC data
Download [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [Pascal VOC 2012 augmented](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) (you can get details at [Semantic Boundaries Dataset and Benchmark](http://home.bharathh.info/pubs/codes/SBD/download.html)), resulting in 10,582 training images(trainaug), 1,449 validatation images.

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
#cp benchmark_RELEASE/dataset/encode_cls/* VOCdevkit/VOC2012/EncodeSegmentationClass
(cd benchmark_RELEASE/dataset/encode_cls; cp * ${vedaseg_root}/data/VOCdevkit/VOC2012/EncodeSegmentationClass)
#cp VOCdevkit/VOC2012/EncodeSegmentationClassPart/* VOCdevkit/VOC2012/EncodeSegmentationClass
(cd VOCdevkit/VOC2012/EncodeSegmentationClassPart; cp * ${vedaseg_root}/data/VOCdevkit/VOC2012/EncodeSegmentationClass)

comm -23 <(cat benchmark_RELEASE/dataset/{train,val}.txt VOCdevkit/VOC2012/ImageSets/Segmentation/train.txt | sort -u) <(cat VOCdevkit/VOC2012/ImageSets/Segmentation/val.txt | sort -u) > VOCdevkit/VOC2012/ImageSets/Segmentation/trainaug.txt
```
To avoid tedious operations, you could save the above linux commands as a shell file and execute it.
### COCO data
Download the COCO-2017 dataset.
```
cd ${vedaseg_root}
mkdir ${vedaseg_root}/data
cd ${vedaseg_root}/data
mkdir COCO2017 && cd COCO2017
wget -c http://images.cocodataset.org/zips/train2017.zip
unzip train2017.zip && rm train2017.zip
wget -c http://images.cocodataset.org/zips/val2017.zip
unzip val2017.zip &&  rm val2017.zip
wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
unzip annotations_trainval2017.zip && rm annotations_trainval2017.zip
```
### Folder structure
The folder structure should similar as following:
 ```
data
├── COCO2017
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   ├── instances_val2017.json
│   ├── train2017
│   ├── val2017
│── VOCdevkit
│   │   ├── VOC2012
│   │   │   ├── JPEGImages
│   │   │   ├── SegmentationClass
│   │   │   ├── ImageSets
│   │   │   │   ├── Segmentation
│   │   │   │   │   ├── trainaug.txt
│   │   │   │   │   ├── val.txt

```
## Train

a. Config

Modify some configuration accordingly in the config file like `configs/voc_unet.py`
* for multi-label training use config file `configs/coco_multilabel_unet.py` and modify some configuration, the difference between single-label and multi-label training are mainly in following parameter in config file: `nclasses`,`multi_label`, `metrics` and `criterion`. Currently multi-label training is only supported in coco data format.

b. Run

```shell
python tools/trainval.py configs/voc_unet.py
```

Snapshots and logs will be generated at `${vedaseg_root}/workdir`.

## Test

a. Config

Modify some configuration accordingly in the config file like `configs/voc_unet.py`

b. Run

```shell
python tools/test.py configs/voc_unet.py checkpoint_path
```

## Inference

a. Config

Modify some configuration accordingly in the config file like `configs/voc_unet.py`

b. Run
```shell
# visualize the results in a new window
python tools/inference.py configs/voc_unet.py checkpoint_path image_path --show

# save the visualization results in folder which named with image prefix, default under folder './result/'
python tools/inference.py configs/voc_unet.py checkpoint_path image_path --out folder_name
```

## Deploy
### Known issues
1. Currently only pspnet model is not supported due to `AdaptiveAvgPool2d`.
2. Default onnx opset version is 9, PyTorch Upsample operation is only supported with specified size, nearest mode and align_corners being None. Set `opset_version=11` in file `volksdep/converters/torch2onnx.py`, if bilinear mode and align_corners are wanted. 
### Usage
a. Install volksdep following the [official instructions](https://github.com/Media-Smart/volksdep)

b. Benchmark(optional)
```shell
python tools/deploy/benchmark.py configs/voc_unet.py checkpoint_path image_path
```
More available arguments are detailed in [tools/deploy/benchmark.py](https://github.com/Media-Smart/vedacls/blob/master/tools/deploy/benchmark.py)

c. Export model
```shell
python tools/deploy/export.py configs/voc_unet.py checkpoint_path image_path out_model_path
```

More available arguments are detailed in [tools/deploy/export.py](https://github.com/Media-Smart/vedacls/blob/master/tools/deploy/export.py)

## Contact

This repository is currently maintained by Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).

## Credits
We got a lot of code from [mmcv](https://github.com/open-mmlab/mmcv) and [mmdetection](https://github.com/open-mmlab/mmdetection), thanks to [open-mmlab](https://github.com/open-mmlab).
