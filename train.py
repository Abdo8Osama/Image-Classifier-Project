# file name : train.py file  
# Author    : Abdo Osama

# The,train.py,file
# will train a new network on a dataset and save the model as a checkpoint

import time
import torch
import numpy as np
import seaborn as sb
from torch import nn
from torch import optim
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import datasets, transforms
import torchvision.models as models
from collections import OrderedDict
import argparse


train_parser = argparse.ArgumentParser(
    description='This is image classifier application')

    # Non-optional argument - must be input (not as -- just the direct name i.e. python train.py flowers)
train_parser.add_argument('data_dir', action="store", nargs='*', default="/home/workspace/ImageClassifier/flowers/")
    # Choose where to save the checkpoint
train_parser.add_argument('--save_dir', action="store", dest="save_dir", default="/home/workspace/ImageClassifier/checkpoint.pth")
    # Choose model architecture
train_parser.add_argument('--arch', action="store", dest="model", default="vgg16")
    # Choose learning rate
train_parser.add_argument('--learning_rate', action="store", dest="learning_rate", default=0.001)
    # Choose number of epochs
train_parser.add_argument('--epochs', action="store", dest="epochs", type=int, default=1)
    # Choose number of hidden units
train_parser.add_argument('--hidden_units', action="store", dest="hidden_units", type=int, default=4096)
    # Choose processor
train_parser.add_argument('--processor', action="store", dest="processor", default="GPU")

train_args = train_parser.parse_args()
print("Image Directory: ", train_args.data_dir, "  Save Directory: ", train_args.save_dir, "  Model: ", train_args.model, "Learning Rate: ", train_args.learning_rate, "Epochs: ", train_args.epochs, "Hidden units: ", train_args.hidden_units, "Processor :", train_args.processor)
    
from classifier_func import trainer

def main():
    print(train_args.model)
    trainer(train_args.model, train_args.data_dir, train_args.save_dir, train_args.learning_rate, train_args.epochs, train_args.hidden_units, train_args.processor)
    
if __name__ == "__main__":
    main()
