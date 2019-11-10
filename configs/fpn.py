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
    encoder=dict(backbone=dict(type='ResNet', arch='resnet50')),
    # model/decoder
    decoder=dict(
        type='GFPN',
        # model/decoder/blocks
        neck=[
            # model/decoder/blocks/block1
            dict(
                type='JunctionBlock',
                top_down=None,
                lateral=dict(from_layer='c5',
                             type='ConvModule',
                             in_channels=2048,
                             out_channels=256,
                             kernel_size=1,
                             norm_cfg=None,
                             activation=None),
                post=dict(type='ConvModule',
                          in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          padding=1,
                          norm_cfg=None,
                          activation=None),
                to_layer='p5',
            ),  # 32
            # model/decoder/blocks/block2
            dict(
                type='JunctionBlock',
                fusion_method='add',
                top_down=dict(from_layer='p5',
                              upsample=dict(type='Upsample', scale_factor=2)),
                lateral=dict(from_layer='c4',
                             type='ConvModule',
                             in_channels=1024,
                             out_channels=256,
                             kernel_size=1,
                             norm_cfg=None,
                             activation=None),
                post=dict(type='ConvModule',
                          in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          padding=1,
                          norm_cfg=None,
                          activation=None),
                to_layer='p4',
            ),  # 16
            # model/decoder/blocks/block3
            dict(
                type='JunctionBlock',
                fusion_method='add',
                top_down=dict(from_layer='p4',
                              upsample=dict(type='Upsample', scale_factor=2)),
                lateral=dict(from_layer='c3',
                             type='ConvModule',
                             in_channels=512,
                             out_channels=256,
                             kernel_size=1,
                             norm_cfg=None,
                             activation=None),
                post=dict(type='ConvModule',
                          in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          padding=1,
                          norm_cfg=None,
                          activation=None),
                to_layer='p3',
            ),  # 8
            # model/decoder/blocks/block2
            dict(
                type='JunctionBlock',
                fusion_method='add',
                top_down=dict(from_layer='p3',
                              upsample=dict(type='Upsample', scale_factor=2)),
                lateral=dict(from_layer='c2',
                             type='ConvModule',
                             in_channels=256,
                             out_channels=256,
                             kernel_size=1,
                             norm_cfg=None,
                             activation=None),
                post=dict(type='ConvModule',
                          in_channels=256,
                          out_channels=256,
                          kernel_size=3,
                          padding=1,
                          norm_cfg=None,
                          activation=None),
                to_layer='p2',
            ),  # 4
        ],
        fusion=dict(type='FusionBlock',
                    method='add',
                    from_layers=['p2', 'p3', 'p4', 'p5'],
                    feat_strides=[4, 8, 16, 32],
                    in_channels_list=[256, 256, 256, 256],
                    out_channels_list=[128, 128, 128, 128],
                    norm_cfg=dict(type='BN'),
                    activation='relu',
                    common_stride=4),  # 4
    ),
    # model/decoer/head
    head=dict(
        type='Head',
        in_channels=128,
        inter_channels=128,
        out_channels=4,
        num_convs=3,
        norm_cfg=dict(type='BN'),
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
