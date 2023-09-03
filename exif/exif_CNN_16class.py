import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision import models
from torchinfo import summary

from sakamoto_CNN_exif import CNN_v4
from pythonlibs.my_torch_lib import (
    evaluate_history,
    fit_for_exif,
    show_images_labels_for_exif,
    torch_seed,
)


from exif_ImageFolder import exif_ImageFolder

warnings.simplefilter("ignore")
plt.rcParams["axes.grid"] = True
np.set_printoptions(suppress=True, precision=5)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# remove augumentation in train
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(0.5, 0.5),
    ]
)

which_data = "data1"
data_dir = os.path.join("coins_data_sin", which_data)


train_dir = os.path.join(data_dir, "train")
test_dir = os.path.join(data_dir, "val")

which_data_ = os.path.join(which_data, "CNN", "EXIF")

classes = [
    str(i1) + str(i2) + str(i3) + str(i4)
    for i1 in range(0, 2)
    for i2 in range(0, 2)
    for i3 in range(0, 2)
    for i4 in range(0, 2)
]

# train_data = datasets.ImageFolder(train_dir, transform=train_transform)
train_data = exif_ImageFolder(train_dir, transform=train_transform)

# train_data2 = datasets.ImageFolder(train_dir, transform=train_transform)
train_data2 = exif_ImageFolder(train_dir, transform=train_transform)

# test_data = datasets.ImageFolder(test_dir, transform=test_transform)
test_data = exif_ImageFolder(test_dir, transform=test_transform)

batch_size = 10

train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)

test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

train_loader2 = DataLoader(train_data2, batch_size=50, shuffle=True)
test_loader2 = DataLoader(test_data, batch_size=50, shuffle=True)


show_images_labels_for_exif(test_loader2, classes, None, None, which_data_)


torch_seed()

n_output = len(classes)

net = CNN_v4(n_output)

net = net.to(device)

lr = 0.001

criterion = nn.CrossEntropyLoss()

optimizer = optim.Adam(net.parameters())

history = np.zeros((0, 5))

num_epochs = 500
history = fit_for_exif(
    net, optimizer, criterion, num_epochs, train_loader, test_loader, device, history
)

evaluate_history(history, which_data_)

show_images_labels_for_exif(test_loader2, classes, net, device, which_data_)
