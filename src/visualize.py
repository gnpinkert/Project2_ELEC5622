import argparse
import logging
import sys
import time
import os
import json
import copy

import torch
from torch import nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
import numpy as np
import random
from pathlib import Path
import bidict

from sklearn.metrics import confusion_matrix
import pandas as pd
import seaborn as sn

from network import AlexNet
from NetworkDetails import TrainingDetails
from loaders import create_data_loader, LoaderType
from extras import get_repo_root_dir

from loaders import ProjectData, _get_image_filepaths, _load_label_csv

def show_images(ims, gt_labels, pred_labels=None, lookup_names = None, save_path = None):
    fig, ax = plt.subplots(1, len(ims), figsize=(12,6))
    for id in range(len(ims)):
        ax[id].imshow(ims[id])
        ax[id].axis('off')

        if lookup_names is None:
            if pred_labels is None:
                im_title = f'GT: {gt_labels[id]}'
            else:
                im_title = f'GT: {gt_labels[id]}   Pred: {pred_labels[id]}'
            ax[id].set_title(im_title)

        else:
            if pred_labels is None:
                im_title = f'GT: {lookup_names[gt_labels[id]]}'
            else:
                im_title = f'GT: {lookup_names[gt_labels[id]]}   Pred: {lookup_names[pred_labels[id]]}'
            ax[id].set_title(im_title)

    # plt.show()
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight', dpi=600)

def plot_data(file_path: Path, savePath):

    with open(file_path, 'r') as json_file:
        data = json.load(json_file)

    validation_accuracy, validation_loss, training_accuracy, training_loss = data['validation_accuracy'], data['validation_loss'], data['training_accuracy'], data['training_loss']
    epochs = range(1, len(validation_accuracy) + 1)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(21, 7))

    ax1.plot(epochs, training_loss, label="Training")
    ax1.plot(epochs, validation_loss, label="Validation")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss [-]")
    ax1.grid("on")
    ax1.legend()

    ax2.plot(epochs, training_accuracy, label="Training")
    ax2.plot(epochs, validation_accuracy, label="Validation")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy [%]")
    ax2.grid("on")
    ax2.legend()

    # plt.tight_layout()

    plt.savefig(savePath, dpi=600,  bbox_inches='tight')
    plt.show()

# https://christianbernecker.medium.com/how-to-create-a-confusion-matrix-in-pytorch-38d06a7f04b7
def getConfusionMatrix(net, testloader, logging, name):
    y_pred = []
    y_true = []

    # iterate over test data
    for inputs, labels in testloader:
            output = net(inputs) # Feed Network

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ["Homogeneous","Speckled","Nucleolar","Centromere","NuMem","Golgi"]

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,8))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(name, bbox_inches = 'tight', dpi=600)

    if logging is not None:
        logging.info('Saved Confusion Matrix as {name}')

def runTestSetForConfusionMatrix():

    training_data = TrainingDetails(batch_size=4,
                                    learning_rate=0.0004,
                                    momentum=0.9,
                                    epochs=1,
                                    output_dir=Path())
    
    # target_dir = get_repo_root_dir() / "models" / str(training_data) / "14_30_50"

    # if not target_dir.exists():
    #     raise FileNotFoundError(f"Can't find existing model since target dir: \"{target_dir}\" does not exist")

    test_loader = create_data_loader(LoaderType.TEST)
    network = AlexNet()
    weights_path = 'project2.pth'
    network.load_state_dict(torch.load(weights_path, map_location='cpu'))
    getConfusionMatrix(net=network,
                       testloader=test_loader,
                       logging=None,
                       name='results/confusionMatrix.png')
    
def plotRandomImages():
    plt.ion()

    ############################################
    # Visualize Random Images and Transformations

    # Define Transforms
    no_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    # transforms.Normalize((0.5), (0.5))
    ])

    train_transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.RandomResizedCrop(size=224, scale=(0.8, 0.8)),
        transforms.RandomRotation(degrees=180,expand=False),
        transforms.Resize((224,224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5), (0.5))
    ])

    # Load dataset
    data_dir = Path('training_data/training')
    image_path = _get_image_filepaths(data_dir)
    dataset_tf = ProjectData(data=image_path, labels = _load_label_csv(), transform=train_transform)
    dataset_og = ProjectData(data=image_path, labels = _load_label_csv(), transform=no_transform)

    lookup_names = ["Homogeneous","Speckled","Nucleolar","Centromere","NuMem","Golgi"]

    # Get images to show
    n_ims = 6
    random_integers = random.sample(range(0, len(dataset_og) + 1), n_ims)
    og_ims = []
    og_labels = []
    aug_ims = []
    aug_labels = []
    for i in random_integers:
        img = dataset_og[i][0].permute(1, 2, 0)
        label = dataset_og[i][1]
        og_ims.append(img)
        og_labels.append(label)

        img = dataset_tf[i][0].permute(1, 2, 0)
        label = dataset_tf[i][1]
        aug_ims.append(img)
        aug_labels.append(label)

    show_images(ims=og_ims,gt_labels=og_labels, lookup_names = lookup_names, save_path='original_images.png')
    show_images(ims=aug_ims,gt_labels=aug_labels, lookup_names = lookup_names, save_path='modified_images.png')

    plt.show(block=True)

if __name__=="__main__":

    # plotRandomImages()
    runTestSetForConfusionMatrix()
