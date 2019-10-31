import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from skimage import io, transform


class CombinationDataset(Dataset):
    """combined dataset of map block data and image."""

    def __init__(self, csv_file, image_root_dir,target_label, transform_image=None, transform_data=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations. #todo: fill this
            image_root_dir (string): Directory with all the images. #todo: fill this
            transform (callable, optional): Optional transform to be applied #todo: fill this
                on a sample.
        """
        self.target_label = target_label
        self.accidents_df = pd.read_csv(csv_file, index_col=0)
        self.accidents_y_df = self.accidents_df.filter([target_label])
        self.accidents_df.drop(columns=[target_label], inplace=True)
        self.image_root_dir = image_root_dir
        self.transform_image = transform_image
        self.transform_data = transform_data

    def __len__(self):
        return len(self.accidents_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        block_data = self.accidents_df.iloc[idx, :]

        id = self.accidents_df.index[idx]
        value = self.accidents_y_df.loc[id, self.target_label]
        img_name = os.path.join(self.image_root_dir, str(id) + '.jpg')
        image = Image.open(img_name)
        block_data = np.array([block_data])
        block_data = block_data.astype('float')  # .reshape(-1, 2)

        if self.transform_image:
            image = self.transform_image(image)

        if self.transform_data:
            block_data = self.transform_data(block_data)

        sample = {'block_image': image, 'block_data': block_data, 'block_value': value}

        return sample
