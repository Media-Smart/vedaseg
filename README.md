## Introduction

vedaseg is an open source semantic segmentation toolbox based on PyTorch.

## Features

- **Modular Design**

  We decompose the semantic segmentation framework into different components. The flexible and extensible design make it easy to implement a customized semantic segmentation project by combining different modules like building Lego.

- **Support of several popular frameworks**

  The toolbox supports several popular semantic segmentation frameworks out of the box, *e.g.* DeepLabv3+, DeepLabv3, U-Net, PSPNet, FPN, etc.

- **High efficiency**
    
  Multi-GPU data parallelism & distributed training.
  
- **Multi-Class/Multi-Label segmentation**

  We implement multi-class and multi-label segmentation(where a pixel can belong to multiple classes).
 
- **Acceleration and deployment**

  Models can be accelerated and deployed with TensorRT.

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
| FPN | ResNet-101 | 4 | True | 77.05% |
| FPN | ResNet-101 | 4 | False | 75.64% |
| PSPNet | ResNet-101 | 8 | True | 78.39% |
| PSPNet | ResNet-101 | 8 | False | 77.30% |
| PSPNet | ResNet_v1c-101 | 8 | True | 79.88% |
| PSPNet | ResNet_v1c-101 | 8 | False | 78.85% |
| U-Net | ResNet-101 | 1 | True | 74.58% |
| U-Net | ResNet-101 | 1 | False | 72.59% |

OS: Output stride used during evaluation.\
MS: Multi-scale inputs during evaluation.\
Flip: Adding horizontal flipped inputs during evaluation.\
ResNet_v1c: Modified stem from original ResNet, as shown in [Figure 2(b) in 
this paper](https://arxiv.org/pdf/1812.01187.pdf).

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

1. Create a conda virtual environment and activate it.

```shell
conda create -n vedaseg python=3.6.9 -y
conda activate vedaseg
```

2. Install PyTorch and torchvision following the [official instructions](https://pytorch.org/), *e.g.*,

```shell
conda install pytorch torchvision -c pytorch
```

3. Clone the vedaseg repository.

```shell
git clone https://github.com/Media-Smart/vedaseg.git
cd vedaseg
vedaseg_root=${PWD}
```

4. Install dependencies.

```shell
pip install -r requirements.txt
```

## Prepare data
### VOC data
Download [Pascal VOC 2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar) and [Pascal VOC 2012 augmented](http://www.eecs.berkeley.edu/Research/Projects/CS/vision/grouping/semantic_contours/benchmark.tgz) (you can get details at [Semantic Boundaries Dataset and Benchmark](http://home.bharathh.info/pubs/codes/SBD/download.html)), resulting in 10,582 training images(trainaug), 1,449 validatation images.

```shell
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
```shell
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

1. Config

Modify configuration files in [configs/](configs) according to your needs(e.g. [configs/voc_unet.py](configs/voc_unet.py)).

The major configuration difference between single-label and multi-label training lies in: `nclasses`, `multi_label`, `metrics`and `criterion`. You can take [configs/coco_multilabel_unet.py](configs/coco_multilabel_unet.py) as a reference. Currently, multi-label training is only supported in COCO data format.

2. Ditributed training
```shell
# train pspnet using GPUs with gpu_id 0, 1, 2, 3
./tools/dist_train.sh configs/voc_pspnet.py "0, 1, 2, 3" 
```

3. Non-distributed training
```shell
python tools/train.py configs/voc_unet.py
```

Snapshots and logs by default will be generated at `${vedaseg_root}/workdir/name_of_config_file`(you can specify workdir in config files).

## Test

1. Config

Modify configuration as you wish(e.g. [configs/voc_unet.py](configs/voc_unet.py)).

2. Ditributed testing
```shell
# test pspnet using GPUs with gpu_id 0, 1, 2, 3
./tools/dist_test.sh configs/voc_pspnet.py path/to/checkpoint.pth "0, 1, 2, 3" 
```

3. Non-distributed testing
```shell
python tools/test.py configs/voc_unet.py path/to/checkpoint.pth
```

## Inference

1. Config

Modify configuration as you wish(e.g. [configs/voc_unet.py](configs/voc_unet.py)).

2. Run

```shell
# visualize the results in a new window
python tools/inference.py configs/voc_unet.py checkpoint_path image_file_path --show

# save the visualization results in folder which named with image prefix, default under folder './result/'
python tools/inference.py configs/voc_unet.py checkpoint_path image_file_path --out folder_name
```

## Deploy

1. Convert to ONNX

Firstly, install volksdep following the [official instructions](https://github.com/Media-Smart/volksdep).

Then, run the following code to convert PyTorch to ONNX. The input shape format is `CxHxW`. 
If you need the ONNX model with dynamic input shape, please add `--dynamic_shape` in the end.

```shell
python tools/torch2onnx.py configs/voc_unet.py weight_path out_path --dummy_input_shape 3,513,513 --opset_version 11
```

Here are some known issues:
- Currently PSPNet model is not supported because of the unsupported operation `AdaptiveAvgPool2d`.
- Default ONNX opset version is 9 and PyTorch Upsample operation is only supported 
with specified size, nearest mode and align_corners being None. 
If bilinear mode and align_corners are wanted, please add `--opset_version 11` when using `torch2onnx.py`.

2. Inference SDK

Firstly, install [flexinfer](https://github.com/Media-Smart/flexinfer) and see the [example](https://github.com/Media-Smart/flexinfer/tree/master/examples/segmentation) for details.

## Contact

This repository is currently maintained by Yuxin Zou ([@YuxinZou](https://github.com/YuxinZou)), Tianhe Wang([@DarthThomas](https://github.com/DarthThomas)), Hongxiang Cai ([@hxcai](http://github.com/hxcai)), Yichao Xiong ([@mileistone](https://github.com/mileistone)).
