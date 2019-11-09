# work dir
root_workdir = 'workdir'

# 1. logging
logger = dict(
    handlers=(
        dict(type='StreamHandler', level='DEBUG'),
        #dict(type='FileHandler', level='DEBUG'),
    ), )

# 2. data
data = dict(
    train=dict(
        dataset=dict(
            type='SteelDataset',
            data_folder='data/steel',
            filename='train.csv',
            phase='train'),
        transforms=[
            #dict(type='RandomSizedCrop', min_max_height=(256, 256), height=256, width=256),
            dict(type='HorizontalFlip'),
        ],
        loader=dict(type='DataLoader',
                    batch_size=8,
                    num_workers=4,
                    shuffle=True,
                    drop_last=True),
    ),
    val=dict(
        dataset=dict(
            type='SteelDataset',
            data_folder='data/steel',
            filename='train.csv',
            phase='val'),
        transforms=[],
        loader=dict(type='DataLoader',
                    batch_size=8,
                    num_workers=4,
                    shuffle=False,
                    drop_last=True),
    ),
)

# 3. model
model = dict(
    # model/encoder
    encoder=dict(backbone=dict(type='ResNet', arch='resnet50')),
    # model/decoder
    decoder=dict(
        type='GFPN',
        # model/decoder/blocks
        neck=[
            # model/decoder/blocks/block1
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(from_layer='c5',
                              upsample=dict(type='Upsample',
                                            scale_factor=2,
                                            mode='nearest')),
                lateral=dict(from_layer='c4'),
                post=dict(
                    type='ConvModules',
                    in_channels=3072,  # 2048 + 1024
                    out_channels=256,
                    kernel_size=3,
                    padding=1,
                    activation='relu',
                    norm_cfg=dict(type='BN'),
                    num_convs=2,
                ),
                to_layer='p4',
            ),  # 16
            # model/decoder/blocks/block2
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(from_layer='p4',
                              upsample=dict(type='Upsample',
                                            scale_factor=2,
                                            mode='nearest')),
                lateral=dict(from_layer='c3'),
                post=dict(
                    type='ConvModules',
                    in_channels=768,  # 256 + 512
                    out_channels=128,
                    kernel_size=3,
                    padding=1,
                    activation='relu',
                    norm_cfg=dict(type='BN'),
                    num_convs=2,
                ),
                to_layer='p3',
            ),  # 8
            # model/decoder/blocks/block3
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(from_layer='p3',
                              upsample=dict(type='Upsample',
                                            scale_factor=2,
                                            mode='nearest')),
                lateral=dict(from_layer='c2'),
                post=dict(
                    type='ConvModules',
                    in_channels=384,  # 128 + 256
                    out_channels=64,
                    kernel_size=3,
                    padding=1,
                    activation='relu',
                    norm_cfg=dict(type='BN'),
                    num_convs=2,
                ),
                to_layer='p2',
            ),  # 4
            # model/decoder/blocks/block4
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(from_layer='p2',
                              upsample=dict(type='Upsample',
                                            scale_factor=2,
                                            mode='nearest')),
                lateral=dict(from_layer='c1'),
                post=dict(
                    type='ConvModules',
                    in_channels=128,  # 64 + 64
                    out_channels=32,
                    kernel_size=3,
                    padding=1,
                    activation='relu',
                    norm_cfg=dict(type='BN'),
                    num_convs=2,
                ),
                to_layer='p1',
            ),  # 2
            # model/decoder/blocks/block5
            dict(
                type='JunctionBlock',
                top_down=dict(from_layer='p1',
                              upsample=dict(type='Upsample',
                                            scale_factor=2,
                                            mode='nearest')),
                lateral=None,
                post=dict(
                    type='ConvModules',
                    in_channels=32,
                    out_channels=16,
                    kernel_size=3,
                    padding=1,
                    activation='relu',
                    norm_cfg=dict(type='BN'),
                    num_convs=2,
                ),
                to_layer='p0',
            ),  # 1
        ]),
    # model/decoer/head
    head=dict(
        type='Head',
        in_channels=16,
        out_channels=4,
        num_convs=0,
    ))

## 3.1 resume
resume = None

# 4. criterion
criterion = dict(type='BCEWithLogitsLoss')

# 5. optim
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

# 6. lr scheduler
lr_scheduler = dict(type='MultiStepLR', milestones=[12, 16])

# 7. runner
runner = dict(type='Runner',
              max_epochs=20,
              trainval_ratio=1,
              snapshot_interval=5)
# 8. device
gpu_id = '0'
