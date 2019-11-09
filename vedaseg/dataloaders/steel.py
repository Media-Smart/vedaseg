def build_dataloader(
        data_folder,
        df_path,
        phases=['train', 'val'],
        mean=None,
        std=None,
        batch_sizes={
            'train': 8,
            'val': 4
        },
        num_workers=4,
):
    '''Returns dataloader for the model training'''
    df = pd.read_csv(df_path)
    # some preprocessing
    # https://www.kaggle.com/amanooo/defect-detection-starter-u-net
    #df['ImageId'], df['ClassId'] = zip(*df['ImageId_ClassId'].str.split('_'))
    df['ImageId'], df['ClassId'] = df['ImageId_ClassId'].str.slice(
        0, -2), df['ImageId_ClassId'].str.slice(-1)
    df['ClassId'] = df['ClassId'].astype(int)
    df = df.pivot(index='ImageId', columns='ClassId', values='EncodedPixels')
    df['defects'] = df.count(axis=1)

    train_df, val_df = train_test_split(df,
                                        test_size=0.2,
                                        stratify=df["defects"])
    #train_df = df
    print(train_df.head())
    print(val_df.head())
    dataloaders = {}
    for phase in phases:
        df = train_df if phase == 'train' else val_df
        image_dataset = SteelDataset(df, data_folder, phase)
        dataloader = DataLoader(
            image_dataset,
            batch_size=batch_sizes[phase],
            num_workers=num_workers,
            pin_memory=True,
            shuffle=True,
            #worker_init_fn=_init_fn
        )
        dataloaders[phase] = dataloader

    return dataloaders
