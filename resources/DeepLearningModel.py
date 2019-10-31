from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
from matplotlib import pyplot as plt
import torchvision

from torch.utils.data import DataLoader

from resources.DeepLearningMain import Experiment


class Regressor(nn.Module):
    def __init__(self, num_block_data_features, experiment=Experiment.IMAGE_128):
        """
        :param num_class: number of different labels in dataset.
        """
        super().__init__()
        # ====== YOUR CODE: ======

        self.num_block_data_features = num_block_data_features
        self.image_features = 128
        kernel_size = 5
        stride = 1
        padding = 0

        modules = [nn.Conv2d(3, 64, kernel_size, padding=padding, stride=stride),
                   nn.ReLU(inplace=True),
                   nn.MaxPool2d(2),

                   nn.Conv2d(64, 64, kernel_size, padding=padding, stride=stride),
                   nn.ReLU(inplace=True),
                   nn.MaxPool2d(2),

                   nn.Conv2d(64, 128, 3, padding=1, stride=stride),
                   nn.ReLU(inplace=True)]

        if experiment != Experiment.IMAGE_32:
            modules.append(nn.MaxPool2d(2))
        modules.append(nn.Conv2d(128, 64, 3, padding=1, stride=stride))
        if experiment == Experiment.IMAGE_128:
            modules.append(nn.MaxPool2d(2))
        modules.append(nn.ReLU(inplace=True))
        modules.append(nn.Conv2d(64, self.image_features, kernel_size=kernel_size, padding=padding,
                                 stride=stride))
        if experiment != Experiment.IMAGE_32:
            modules.append(nn.MaxPool2d(2))
        modules.append(nn.ReLU(inplace=True))
        self.feature_extractor = nn.Sequential(*modules)

        self.regressor = nn.Sequential(
            nn.Linear(self.image_features + num_block_data_features, self.image_features + num_block_data_features),
            nn.ReLU(inplace=True),
            nn.Linear(self.image_features + num_block_data_features, 1),
            nn.ReLU(inplace=True),
        )
        # ========================

    def forward(self, x):
        """
        :param x: Input feature vector.
        :return: regression value for feature vector
        """
        # ====== YOUR CODE: ======
        image_features = self.feature_extractor(x["block_image"])
        x = torch.cat(
            (image_features.view(-1, self.image_features), x["block_data"].view(-1, self.num_block_data_features)), 1)

        return self.regressor(x)
        # ========================


class ImagelessRegressor(nn.Module):
    def __init__(self, num_block_data_features):
        """
        :param num_class: number of different labels in dataset.
        """
        super().__init__()
        # ====== YOUR CODE: ======

        self.num_block_data_features = num_block_data_features
        self.regressor = nn.Sequential(
            nn.Linear(num_block_data_features, num_block_data_features),
            nn.ReLU(inplace=True),
            nn.Linear(num_block_data_features, 1),
            nn.ReLU(inplace=True),
        )
        # ========================

    def forward(self, x):
        """
        :param x: Input feature vector.
        :return: regression value for feature vector
        """
        # ====== YOUR CODE: ======
        x = x["block_data"].view(-1, self.num_block_data_features)

        return self.regressor(x)
        # ========================


def regressor_loss_fn(predicted_y, y):
    """ #todo doc
    :param predicted_y: Embedded vectors classifications of instances sampled from the source dataset, shape (N,).
    :param y: class labels of source data instances.
    :return: The classifier loss.
    """
    # ====== YOUR CODE: ======
    mse_loss = nn.MSELoss()
    loss = mse_loss(predicted_y, y.view(-1, 1))
    # ========================
    return loss


def train_batch(reg_model: Regressor, reg_optimizer: Optimizer,
                dl: DataLoader):
    """
    Trains model for one batch
    :return: The loss.
    """

    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    block_image = dl["block_image"]
    block_data = dl["block_data"]
    block_value = dl["block_value"]

    block_image = block_image.type(torch.FloatTensor).to(device=device)
    block_data = block_data.type(torch.FloatTensor).to(device=device)
    block_value = block_value.type(torch.FloatTensor).to(device=device)

    block_input = {'block_image': block_image, 'block_data': block_data}

    # ========================
    # C network update

    reg_optimizer.zero_grad()
    y = reg_model(block_input)
    reg_loss = regressor_loss_fn(y, block_value)

    # Backward pass
    reg_loss.backward()

    # Optimization Step
    reg_optimizer.step()
    # ========================
    return reg_loss.item()


def train_imageless_batch(reg_model: Regressor, reg_optimizer: Optimizer,
                          dl: DataLoader):
    """
    Trains model for one batch
    :return: The loss.
    """

    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    block_data = dl["block_data"]
    block_value = dl["block_value"]

    block_data = block_data.type(torch.FloatTensor).to(device=device)
    block_value = block_value.type(torch.FloatTensor).to(device=device)

    block_input = {'block_data': block_data}

    # ========================
    # C network update

    reg_optimizer.zero_grad()
    y = reg_model(block_input)
    reg_loss = regressor_loss_fn(y, block_value)

    # Backward pass
    reg_loss.backward()

    # Optimization Step
    reg_optimizer.step()
    # ========================
    return reg_loss.item()
