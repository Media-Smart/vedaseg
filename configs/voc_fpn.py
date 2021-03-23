import cv2

# 1. configuration for inference
nclasses = 21
ignore_label = 255
multi_label = False

crop_size_h, crop_size_w = 513, 513
test_size_h, test_size_w = 513, 513
image_pad_value = (123.675, 116.280, 103.530)

img_norm_cfg = dict(
    max_pixel_value=255.0,
    std=(0.229, 0.224, 0.225),
    mean=(0.485, 0.456, 0.406),
)
norm_cfg = dict(type='BN')
fpn_post = dict(
    type='ConvModule',
    in_channels=256,
    out_channels=256,
    kernel_size=3,
    padding=1,
    norm_cfg=None,
    act_cfg=None,
)
fpn_upsample_2x = dict(
    type='Upsample',
    scale_factor=2,
    scale_bias=-1,
    mode='bilinear',
    align_corners=True,
)

inference = dict(
    gpu_id='0, 1',
    transforms=[
        dict(type='PadIfNeeded', min_height=test_size_h, min_width=test_size_w,
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
            ),
        ),
        # model/decoder
        decoder=dict(
            type='GFPN',
            # model/decoder/blocks
            neck=[
                # FPN - connection to m5
                dict(
                    type='JunctionBlock',
                    laterals=[
                        dict(
                            from_layer='c5',
                            type='ConvModule',
                            in_channels=2048,
                            out_channels=256,
                            kernel_size=1,
                            norm_cfg=None,
                            act_cfg=None,
                        ),
                    ],
                    to_layer='m5',
                ),
                # FPN - connections to m4~m2
                *[
                    dict(
                        type='JunctionBlock',
                        fusion_method='add',
                        verticals=[
                            {'from_layer': f'm{idx + 1}', **fpn_upsample_2x}
                        ],
                        laterals=[
                            dict(
                                from_layer=f'c{idx}',
                                type='ConvModule',
                                in_channels=2 ** (idx + 6),
                                out_channels=256,
                                kernel_size=1,
                                norm_cfg=None,
                                act_cfg=None,
                            ),
                        ],
                        to_layer=f'm{idx}',
                    ) for idx in range(4, 1, -1)
                ],
                # FPN - connections from m to p
                *[
                    dict(
                        type='JunctionBlock',
                        laterals=[{'from_layer': f'm{idx}', **fpn_post}],
                        to_layer=f'p{idx}',
                    ) for idx in range(5, 1, -1)
                ],  # 4

            ],
            fusion=dict(
                type='FusionBlock',
                method='add',
                from_layers=['p2', 'p3', 'p4', 'p5'],
                feat_strides=[4, 8, 16, 32],
                in_channels_list=[256, 256, 256, 256],
                out_channels_list=[128, 128, 128, 128],
                norm_cfg=dict(type='BN'),
                act_cfg=dict(type='Relu', inplace=True),
                common_stride=4,
                upsample=dict(
                    type='Upsample',
                    scale_factor=2,
                    scale_bias=-1,
                    mode='bilinear',
                    align_corners=True,
                ),
            ),  # 4
        ),
        # model/decoer/head
        head=dict(
            type='Head',
            in_channels=128,
            inter_channels=128,
            out_channels=nclasses,
            num_convs=3,
            upsample=dict(
                type='Upsample',
                scale_factor=4,
                scale_bias=-3,
                mode='bilinear',
                align_corners=True,
            ),
        ),
    ),
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
    cudnn_deterministic=True,
    cudnn_benchmark=False,
    metrics=[
        dict(type='IoU', num_classes=nclasses),
        dict(type='MIoU', num_classes=nclasses, average='equal'),
    ],
    dist_params=dict(backend='nccl'),
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
        sampler=dict(
            type='DefaultSampler',
        ),
        dataloader=dict(
            type='DataLoader',
            samples_per_gpu=4,
            workers_per_gpu=4,
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
                dict(type='RandomScale', scale_limit=(0.5, 2),
                     interpolation=cv2.INTER_LINEAR),
                dict(type='PadIfNeeded', min_height=crop_size_h,
                     min_width=crop_size_w, value=image_pad_value,
                     mask_value=ignore_label),
                dict(type='RandomCrop', height=crop_size_h, width=crop_size_w),
                dict(type='Rotate', limit=10, interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT, value=image_pad_value,
                     mask_value=ignore_label, p=0.5),
                dict(type='GaussianBlur', blur_limit=7, p=0.5),
                dict(type='HorizontalFlip', p=0.5),
                dict(type='Normalize', **img_norm_cfg),
                dict(type='ToTensor'),
            ],
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=8,
                workers_per_gpu=4,
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
            sampler=dict(
                type='DefaultSampler',
            ),
            dataloader=dict(
                type='DataLoader',
                samples_per_gpu=8,
                workers_per_gpu=4,
                shuffle=False,
                drop_last=False,
                pin_memory=True,
            ),
        ),
    ),
    resume=None,
    criterion=dict(type='CrossEntropyLoss', ignore_index=ignore_label),
    optimizer=dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001),
    lr_scheduler=dict(type='PolyLR', max_epochs=max_epochs),
    max_epochs=max_epochs,
    trainval_ratio=1,
    log_interval=10,
    snapshot_interval=5,
    save_best=True,
)
