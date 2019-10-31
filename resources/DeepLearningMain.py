import csv
import os

import torch
import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, explained_variance_score, r2_score
from torch import optim
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchvision import datasets
from enum import Enum
from resources import DeepLearningModel, DeepLearningTraining
from resources.CombinationDataset import CombinationDataset
from resources.DeepLearningModel import regressor_loss_fn


class Experiment(Enum):
    NO_IMAGE = 0
    IMAGE_32 = 32
    IMAGE_64 = 64
    IMAGE_128 = 128


def main():
    num_epochs = 100
    batch_size = 50
    checkpoint_file_final = "final_model"
    target_label = "accidents_num"

    load = False
    train = True
    experiment = Experiment.IMAGE_128
    imageless = True if experiment == Experiment.NO_IMAGE else False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if experiment == Experiment.NO_IMAGE:
        num_block_data_features = 15
    else:
        num_block_data_features = 11

    if imageless:
        reg = DeepLearningModel.ImagelessRegressor(num_block_data_features=num_block_data_features).to(device)
    else:
        reg = DeepLearningModel.Regressor(num_block_data_features=num_block_data_features, experiment=experiment).to(
            device)
    experiment_name = str(experiment.value) if experiment != Experiment.NO_IMAGE else "no_image"
    file_to_load = f'{checkpoint_file_final}_r_{experiment_name}.pt'
    if load and (os.path.isfile(file_to_load)):
        print(f'*** Loading final checkpoint file {checkpoint_file_final} instead of training')
        reg = torch.load(file_to_load, map_location=device)

    data_train_root = os.path.join('data', 'train_processed.csv')
    data_val_root = os.path.join('data', 'validate_processed.csv')
    data_test_root = os.path.join('data', 'test_processed.csv')

    image_train_root = os.path.join('data', 'map_screenshots', 'train')
    image_val_root = os.path.join('data', 'map_screenshots', 'validate')
    image_test_root = os.path.join('data', 'map_screenshots', 'test')

    transform = transforms.Compose(
        [transforms.CenterCrop(240), transforms.Resize(experiment.value), transforms.ToTensor()])

    train_dataset = CombinationDataset(csv_file=data_train_root, image_root_dir=image_train_root,
                                       target_label=target_label,
                                       transform_image=transform)
    validate_dataset = CombinationDataset(csv_file=data_val_root, image_root_dir=image_val_root,
                                          target_label=target_label, transform_image=transform)
    test_dataset = CombinationDataset(csv_file=data_test_root, image_root_dir=image_test_root,
                                      target_label=target_label,
                                      transform_image=transform)

    training_dl = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                             drop_last=True)

    validation_dl = DataLoader(validate_dataset, batch_size=batch_size, num_workers=2)

    test_dl = DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=2,
                         drop_last=True)

    reg_optimizer = optim.Adam(reg.parameters(), betas=(0.9, 0.999), lr=0.001, weight_decay=1e-5)

    cudnn.benchmark = True

    if train:
        DeepLearningTraining.train(num_epochs, reg, reg_optimizer,
                                   training_dl, validation_dl, test_dl, device, imageless, experiment)

    test(reg, test_dl, device, experiment=experiment_name)


def calc_metrics():
    experiment_names = ["32", "64", "128", "no_image", "128_2017", "no_image_2017"]

    for name in experiment_names:
        res = pd.read_csv("deep_learning_test_result_" + name + ".csv")
        y_pred = res["y_pred"].values
        y = res["y"].values
        print("### experiment: " + name + " ####")
        print("mse: " + str(mean_squared_error(y, y_pred)))
        print("mae: " + str(mean_absolute_error(y, y_pred)))
        print("explained_variance_score: " + str(explained_variance_score(y, y_pred)))
        print("r2: " + str(r2_score(y, y_pred)))


def test(reg, test_dl, device, name="test", experiment=""):
    reg.eval()
    if len(experiment) > 0:
        experiment = '_' + experiment
    total = 0
    correct = 0
    losses = []
    with open('deep_learning_test_result' + experiment + '.csv', 'w', newline='') as abc:
        csv_writer = csv.writer(abc, delimiter=',')
        csv_writer.writerow(['y_pred', 'y'])

        for data in test_dl:
            block_image = data["block_image"]
            block_data = data["block_data"]
            block_value = data["block_value"]

            block_image = block_image.type(torch.FloatTensor).to(device=device)
            block_data = block_data.type(torch.FloatTensor).to(device=device)
            block_value = block_value.type(torch.FloatTensor).to(device=device)

            block_input = {'block_image': block_image, 'block_data': block_data}

            y = reg(block_input)

            total += block_value.size(0)
            truth_threshold = 5
            correct += (torch.le(torch.abs(y - block_value.view(-1, 1)), truth_threshold)).sum()
            np.savetxt(abc, torch.cat((y, block_value.view(-1, 1)), dim=1).cpu().detach().numpy(), delimiter=",")
            losses.append(regressor_loss_fn(y, block_value).detach().item())

    accuracy = float((100.0 * float(correct)) / float(total))
    avg_loss = np.average(losses)
    print(name + " acc: %.2f%%" % accuracy)
    return accuracy, avg_loss


if __name__ == '__main__':
    main()
    calc_metrics()
