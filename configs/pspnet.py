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
    encoder=dict(backbone=dict(
        type='ResNet',
        arch='resnet50',
        replace_stride_with_dilation=[False, True, True]),
                 enhance=dict(type='PPM',
                              from_layer='c5',
                              to_layer='enhance',
                              in_channels=2048,
                              out_channels=512,
                              bins=[1, 2, 4, 6])),
    collect=dict(type='CollectBlock', from_layer='enhance'),
    # model/head
    head=dict(
        type='Head',
        in_channels=4096,
        inter_channels=512,
        dropouts=[0.1],
        num_convs=1,
        out_channels=4,
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
