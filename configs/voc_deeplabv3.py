import cv2

# 1. configuration for inference
nclasses = 21
ignore_label = 255
image_pad_value = (123.675, 116.280, 103.530)
size_h = 513
size_w = 513
img_norm_cfg = dict(mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    max_pixel_value=255.0)
multi_label = False

inference = dict(
    gpu_id='0,1',
    multi_label=multi_label,
    transforms=[
        dict(type='PadIfNeeded', min_height=size_h, min_width=size_w,
             value=image_pad_value, mask_value=ignore_label),
        dict(type='Normalize', **img_norm_cfg),
        dict(type='ToTensor'),
    ],
    model=dict(
        # model/encoder
        encoder=dict(
            backbone=dict(
                type='ResNet',
                arch='resnet101',
                replace_stride_with_dilation=[False, False, True],
                multi_grid=[1, 2, 4],
            ),
            enhance=dict(
                type='ASPP',
                from_layer='c5',
                to_layer='enhance',
                in_channels=2048,
                out_channels=256,
                atrous_rates=[6, 12, 18],
                mode='bilinear',
                align_corners=True,
                dropout=0.1,
            ),
        ),
        collect=dict(type='CollectBlock', from_layer='enhance'),
        # model/head
        head=dict(
            type='Head',
            in_channels=256,
            inter_channels=256,
            out_channels=nclasses,
            num_convs=1,
            upsample=dict(
                type='Upsample',
                size=(size_h, size_w),
                mode='bilinear',
                align_corners=True
            ),
        )
    )
)

# 2. configuration for train/test
root_workdir = 'workdir'
dataset_type = 'VOCDataset'
dataset_root = 'data/VOCdevkit/VOC2012/'

common = dict(
    seed=0,
    logger=dict(
        handlers=(
            dict(type='StreamHandler', level='INFO'),
            dict(type='FileHandler', level='INFO'),
        ),
    ),
    cudnn_deterministic=False,
    cudnn_benchmark=True,
    metrics=[
        dict(type='IoU', num_classes=nclasses),
        dict(type='MIoU', num_classes=nclasses, average='equal'),
    ],
)

## 2.1 configuration for test
test = dict(
    data=dict(
        dataset=dict(
            type=dataset_type,
            root=dataset_root,
            imglist_name='val.txt',
            multi_label=multi_label,
        ),
        transforms=inference['transforms'],
        dataloader=dict(
            type='DataLoader',
            batch_size=16,
            num_workers=4,
            shuffle=False,
            drop_last=False,
            pin_memory=True,
        ),
    ),
    # tta=dict(
    #     scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    #     biases=[0.5, 0.25, 0.0, -0.25, -0.5, -0.75],
    #     flip=True,
    # ),
)

## 2.2 configuration for train
max_epochs = 50

train = dict(
    data=dict(
        train=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                imglist_name='trainaug.txt',
                multi_label=multi_label,
            ),
            transforms=[
                dict(type='RandomScale', scale_limit=(0.5, 2), scale_step=0.25,
                     interpolation=cv2.INTER_LINEAR),
                dict(type='PadIfNeeded', min_height=size_h, min_width=size_w,
                     value=image_pad_value, mask_value=ignore_label),
                dict(type='RandomCrop', height=size_h, width=size_w),
                dict(type='HorizontalFlip', p=0.5),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ToTensor'),
            ],
            dataloader=dict(
                type='DataLoader',
                batch_size=16,
                num_workers=4,
                shuffle=True,
                drop_last=True,
                pin_memory=True,
            ),
        ),
        val=dict(
            dataset=dict(
                type=dataset_type,
                root=dataset_root,
                imglist_name='val.txt',
                multi_label=multi_label,
            ),
            transforms=inference['transforms'],
            dataloader=dict(
                type='DataLoader',
                batch_size=16,
                num_workers=4,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            ),
        ),
    ),
    resume=None,
    criterion=dict(type='CrossEntropyLoss', ignore_index=ignore_label),
    optimizer=dict(type='SGD', lr=0.007, momentum=0.9, weight_decay=0.0001),
    lr_scheduler=dict(type='PolyLR', max_epochs=max_epochs),
    max_epochs=max_epochs,
    trainval_ratio=1,
    log_interval=10,
    snapshot_interval=5,
    save_best=True,
)
