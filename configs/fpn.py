# work dir
root_workdir = 'workdir'

# seed
seed = 0

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='INFO'),
        # dict(type='FileHandler', level='INFO'),
    ),
)

# 2. data
test_cfg = dict(
    scales=[0.5, 0.75, 1.0, 1.25, 1.5, 1.75],
    bias=[0.5, 0.25, 0.0, -0.25, -0.5, -0.75],
    flip=True,
)
img_norm_cfg = dict(mean=(123.675, 116.280, 103.530), std=(58.395, 57.120, 57.375))
ignore_label = 255

dataset_type = 'VOCDataset'
dataset_root = 'data/VOCdevkit/VOC2012'
data = dict(
    train=dict(
        dataset=dict(
            type=dataset_type,
            root=dataset_root,
            imglist_name='trainaug.txt',
        ),
        transforms=[
            dict(type='RandomScale', min_scale=0.5, max_scale=2.0, mode='bilinear'),
            dict(type='RandomCrop', height=513, width=513, image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='RandomRotate', p=0.5, degrees=10, mode='bilinear', border_mode='constant', image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='GaussianBlur', p=0.5, ksize=7),
            dict(type='HorizontalFlip', p=0.5),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
        loader=dict(
            type='DataLoader',
            batch_size=16,
            num_workers=4,
            shuffle=True,
            drop_last=True,
        ),
    ),
    val=dict(
        dataset=dict(
            type=dataset_type,
            root=dataset_root,
            imglist_name='val.txt',
        ),
        transforms=[
            dict(type='PadIfNeeded', height=513, width=513, image_value=img_norm_cfg['mean'], mask_value=ignore_label),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='ToTensor'),
        ],
        loader=dict(
            type='DataLoader',
            batch_size=8,
            num_workers=4,
            shuffle=False,
            drop_last=False,
        ),
    ),
)

# 3. model
nclasses = 21
model = dict(
    # model/encoder
    encoder=dict(
        backbone=dict(
            type='ResNet',
            arch='resnet101'
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
                top_down=None,
                lateral=dict(
                    from_layer='c5',
                    type='ConvModule',
                    in_channels=2048,
                    out_channels=256,
                    kernel_size=1,
                    norm_cfg=None,
                    act_cfg=None,
                ),
                post=dict(
                    type='ConvModule',
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=None,
                    act_cfg=None,
                ),
                to_layer='p5',
            ),  # 32
            # model/decoder/blocks/block2
            dict(
                type='JunctionBlock',
                fusion_method='add',
                top_down=dict(
                    from_layer='p5',
                    upsample=dict(
                        type='Upsample',
                        scale_factor=2,
                        scale_bias=-1,
                        mode='bilinear',
                        align_corners=True,
                    ),
                ),
                lateral=dict(
                    from_layer='c4',
                    type='ConvModule',
                    in_channels=1024,
                    out_channels=256,
                    kernel_size=1,
                    norm_cfg=None,
                    act_cfg=None
                ),
                post=dict(
                    type='ConvModule',
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=None,
                    act_cfg=None,
                ),
                to_layer='p4',
            ),  # 16
            # model/decoder/blocks/block3
            dict(
                type='JunctionBlock',
                fusion_method='add',
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
                lateral=dict(
                    from_layer='c3',
                    type='ConvModule',
                    in_channels=512,
                    out_channels=256,
                    kernel_size=1,
                    norm_cfg=None,
                    act_cfg=None,
                ),
                post=dict(
                    type='ConvModule',
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=None,
                    act_cfg=None
                ),
                to_layer='p3',
            ),  # 8
            # model/decoder/blocks/block2
            dict(
                type='JunctionBlock',
                fusion_method='add',
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
                lateral=dict(
                    from_layer='c2',
                    type='ConvModule',
                    in_channels=256,
                    out_channels=256,
                    kernel_size=1,
                    norm_cfg=None,
                    act_cfg=None,
                ),
                post=dict(
                    type='ConvModule',
                    in_channels=256,
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    norm_cfg=None,
                    act_cfg=None,
                ),
                to_layer='p2',
            ),  # 4
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
            size=(513, 513),
            mode='bilinear',
            align_corners=True,
        ),
    )
)

## 3.1 resume
resume = None

# 4. criterion
criterion = dict(type='CrossEntropyLoss', ignore_index=ignore_label)

# 5. optim
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# 6. lr scheduler
max_epochs = 50
lr_scheduler = dict(type='PolyLR', max_epochs=max_epochs)

# 7. runner
runner = dict(
    type='Runner',
    max_epochs=max_epochs,
    trainval_ratio=1,
    snapshot_interval=5,
)

# 8. device
gpu_id = '0,1'
