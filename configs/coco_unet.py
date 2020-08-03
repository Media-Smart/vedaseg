import cv2

# 1. configuration for inference
nclasses = 81
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
        dict(type='LongestMaxSize', h_max=size_h, w_max=size_w,
             interpolation=cv2.INTER_LINEAR),
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
                pretrain=True,
            ),
        ),
        # model/decoder
        decoder=dict(
            type='GFPN',
            # model/decoder/blocks
            neck=[
                # model/decoder/blocks/block1
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='c5',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=dict(from_layer='c4'),
                    post=dict(
                        type='ConvModules',
                        in_channels=3072,  # 2048 + 1024 resnet101
                        # in_channels=768,  # 512 + 256 resnet18
                        out_channels=256,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p4',
                ),  # 16
                # model/decoder/blocks/block2
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='p4',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=dict(from_layer='c3'),
                    post=dict(
                        type='ConvModules',
                        in_channels=768,  # 256 + 512
                        # in_channels=384,  # 256 + 128
                        out_channels=128,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p3',
                ),  # 8
                # model/decoder/blocks/block3
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='p3',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=dict(from_layer='c2'),
                    post=dict(
                        type='ConvModules',
                        in_channels=384,  # 128 + 256
                        # in_channels=192,  # 128 + 64
                        out_channels=64,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p2',
                ),  # 4
                # model/decoder/blocks/block4
                dict(
                    type='JunctionBlock',
                    fusion_method='concat',
                    top_down=dict(
                        from_layer='p2',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=dict(from_layer='c1'),
                    post=dict(
                        type='ConvModules',
                        in_channels=128,  # 64 + 64 same as resnet18
                        out_channels=32,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p1',
                ),  # 2
                # model/decoder/blocks/block5
                dict(
                    type='JunctionBlock',
                    top_down=dict(
                        from_layer='p1',
                        upsample=dict(
                            type='Upsample',
                            scale_factor=2,
                            scale_bias=-1,
                            mode='bilinear',
                            align_corners=True,
                        ),
                    ),
                    lateral=None,
                    post=dict(
                        type='ConvModules',
                        in_channels=32,
                        out_channels=16,
                        kernel_size=3,
                        padding=1,
                        norm_cfg=dict(type='BN'),
                        act_cfg=dict(type='Relu', inplace=True),
                        num_convs=2,
                    ),
                    to_layer='p0',
                ),  # 1
            ]),
        # model/decoer/head
        head=dict(
            type='Head',
            in_channels=16,
            out_channels=nclasses,
            num_convs=0,
            upsample=dict(
                type='Upsample',
                size=(size_h, size_w),
                mode='bilinear',
                align_corners=True,
            ),
        )
    )
)

# 2. configuration for train/test
root_workdir = 'workdir'
dataset_type = 'CocoDataset'
dataset_root = 'data/COCO2017'

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
    ]
)

## 2.1 configuration for test
test = dict(
    data=dict(
        dataset=dict(
            type=dataset_type,
            root=dataset_root,
            ann_file='instances_val2017.json',
            img_prefix='val2017',
            multi_label=multi_label,
        ),
        transforms=inference['transforms'],
        dataloader=dict(
            type='DataLoader',
            batch_size=8,
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
                ann_file='instances_train2017.json',
                img_prefix='train2017',
                multi_label=multi_label,
            ),
            transforms=[
                dict(type='RandomScale', scale_limit=(0.5, 2),
                     interpolation=cv2.INTER_LINEAR),
                dict(type='PadIfNeeded', min_height=size_h, min_width=size_w,
                     value=image_pad_value, mask_value=ignore_label),
                dict(type='RandomCrop', height=size_h, width=size_w),
                dict(type='Rotate', limit=10, interpolation=cv2.INTER_LINEAR,
                     border_mode=cv2.BORDER_CONSTANT,
                     value=image_pad_value, mask_value=ignore_label, p=0.5
                     ),
                dict(type='GaussianBlur', blur_limit=7, p=0.5),
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
                ann_file='instances_val2014.json',
                img_prefix='val2014',
                multi_label=multi_label,
            ),
            transforms=inference['transforms'],
            dataloader=dict(
                type='DataLoader',
                batch_size=8,
                num_workers=4,
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
    log_interval=5,
    snapshot_interval=5,
    save_best=True,
)
