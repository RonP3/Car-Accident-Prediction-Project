import itertools
import os
import IPython.display
import numpy as np
import torch
import time

import tqdm
from torch import optim
from torchvision.transforms import transforms
import torchvision.datasets as dataset

from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

from resources import DeepLearningModel, DeepLearningMain
from resources.DeepLearningMain import Experiment


def train(num_epochs, reg, reg_optimizer,
          training_dl, validation_dl, test_dl, device, imageless=False, experiment=Experiment.IMAGE_128):
    checkpoint_file_final = "final_model"

    test_accuracies = []
    test_avg_losses = []
    curr_best = np.inf
    for epoch_idx in range(num_epochs):
        # We'll accumulate batch losses and show an average once per epoch.
        reg_losses = []
        print(f'--- EPOCH {epoch_idx + 1}/{num_epochs} ---')
        start = time.time()

        for i, source_data in enumerate(training_dl):
            reg.train()

            if imageless:
                reg_loss = DeepLearningModel.train_imageless_batch(reg, reg_optimizer, source_data)
            else:
                reg_loss = DeepLearningModel.train_batch(reg, reg_optimizer, source_data)

            reg_losses.append(reg_loss)
        experiment_name = str(experiment.value) if experiment != Experiment.NO_IMAGE else "no_image"

        torch.save(reg, f'{checkpoint_file_final}_r_{experiment_name}.pt')

        end = time.time()
        print(end - start)
        test_acc, test_avg_loss = DeepLearningMain.test(reg, validation_dl, device, name="Validation")
        test_accuracies.append(test_acc)
        test_avg_losses.append(test_avg_loss)

        if test_avg_losses[-1] <= curr_best:
            curr_best = test_avg_losses[-1]
            torch.save(reg, f'{checkpoint_file_final}_best_r.pt')

        print("best loss: {:.5f} ".format(curr_best))

        lst_iter = range(1, len(test_avg_losses) + 1, 1)

        plt.plot(lst_iter, test_avg_losses, '-r', label='Validation Loss')

        title = "Validation Regression MSE Loss of Model By Training Epoch"
        plt.xlabel("Epoch")
        plt.ylabel("MSE Loss")
        plt.legend(loc='upper left')
        plt.title(title)
        if len(test_avg_losses) < 25:
            plt.xticks(np.arange(1, len(test_avg_losses) + 1, 1.0))
        else:
            plt.xticks(np.arange(1, len(test_avg_losses) + 1, 5.0))

        plt.show()

        reg_avg_loss = np.mean(reg_losses)
        print('Regression loss: {:.5f}'.format(reg_avg_loss))
