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
        transforms=[
            #dict(type='RandomSizedCrop', min_max_height=(256, 256), height=256, width=256),
        ],
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
    encoder=dict(backbone=dict(
        type='ResNet',
        arch='resnet50',
        replace_stride_with_dilation=[False, False, True],
    ),
                 enhance=dict(type='ASPP',
                              from_layer='c5',
                              to_layer='enhance',
                              in_channels=2048,
                              out_channels=256,
                              atrous_rates=[6, 12, 18])),
    # model/decoder
    decoder=dict(
        type='GFPN',
        # model/decoder/blocks
        neck=[
            # model/decoder/blocks/block1
            dict(
                type='JunctionBlock',
                fusion_method='concat',
                top_down=dict(from_layer='enhance',
                              trans=dict(type='ConvModule',
                                         in_channels=256,
                                         out_channels=256,
                                         kernel_size=1,
                                         norm_cfg=dict(type='BN'),
                                         activation='relu',
                                         dropout=0.5),
                              upsample=dict(type='Upsample',
                                            scale_factor=4,
                                            mode='bilinear')),
                lateral=dict(from_layer='c2',
                             type='ConvModule',
                             in_channels=256,
                             out_channels=48,
                             kernel_size=1,
                             norm_cfg=dict(type='BN'),
                             activation='relu'),
                post=None,
                to_layer='p5',
            ),  # 4
        ]),
    # model/head
    head=dict(
        type='Head',
        in_channels=304,
        inter_channels=256,
        out_channels=4,
        num_convs=2,
        dropouts=[0.5, 0.1],
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
              max_epochs=3,
              trainval_ratio=1,
              snapshot_interval=100000)
# 8. device
gpu_id = '0'
