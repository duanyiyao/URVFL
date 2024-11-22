from torch import nn, optim
import torch.nn.functional as F
from data_dis import Distribute_MNIST
from torchvision import datasets, transforms
import torch
import tqdm
from torch.utils.data import DataLoader,random_split,Subset
import os

import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random

class TinyImageNetDataset(Dataset):
    def __init__(self, root_dir, transform=None, train=True):
        self.root_dir = root_dir
        self.transform = transform
        self.train = train

        if self.train:
            self.data = []
            self.labels = []
            classes = os.listdir(os.path.join(root_dir, 'train'))
            for label, cls in enumerate(classes):
                cls_dir = os.path.join(root_dir, 'train', cls, 'images')
                for img_name in os.listdir(cls_dir):
                    self.data.append(os.path.join(cls_dir, img_name))
                    self.labels.append(label)
        else:
            self.data = []
            self.labels = []
            val_dir = os.path.join(root_dir, 'val', 'images')
            val_annotations = pd.read_csv(os.path.join(root_dir, 'val', 'val_annotations.txt'), 
                                          sep='\t', header=None, 
                                          names=['file_name', 'class', 'x1', 'y1', 'x2', 'y2'])
            class_to_idx = {cls: idx for idx, cls in enumerate(sorted(os.listdir(os.path.join(root_dir, 'train'))))}
            for _, row in val_annotations.iterrows():
                self.data.append(os.path.join(val_dir, row['file_name']))
                self.labels.append(class_to_idx[row['class']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path = self.data[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
class FilteredTinyImageNetDataset(TinyImageNetDataset):
    def __init__(self, root_dir, transform=None, train=True, selected_classes=None, class_to_new_label=None):
        super().__init__(root_dir, transform, train)
        self.selected_classes = selected_classes
        self.class_to_new_label = class_to_new_label

        # Filter the data and labels
        if selected_classes and class_to_new_label:
            self.data, self.labels = self._filter_data()

    def _filter_data(self):
        filtered_data = []
        filtered_labels = []
        for img_path, label in zip(self.data, self.labels):
            class_name = os.path.basename(os.path.dirname(os.path.dirname(img_path)))
            if class_name in self.selected_classes:
                filtered_data.append(img_path)
                filtered_labels.append(self.class_to_new_label[class_name])
        return filtered_data, filtered_labels


def tiny_data_pre(data_owners, device, train_batchsize = 64, test_batchsize = 1, small_batchsize = 32, num_samples = 12):

    
    # Defining the transformations
    transform = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
        # transforms.Normalize(mean=(0.485, 0.456, 0.406),
        #         std=(0.229, 0.224, 0.225))
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])
    # Loading the dataset
    data_dir = '/home/dyao/tiny-imagenet-200'
    

    # Randomly select 10 class indices
    
    all_classes = sorted(os.listdir(os.path.join(data_dir, 'train')))

    # Randomly select 10 classes
    selected_classes = random.sample(all_classes, 10)
    print("Selected Classes:", selected_classes)

    # Create a mapping from the selected class names to new labels 0-9
    class_to_new_label = {cls: i for i, cls in enumerate(selected_classes)}
    print("Class to New Label Mapping:", class_to_new_label)
    filtered_dataset = FilteredTinyImageNetDataset(
    root_dir=data_dir, 
    transform=transform, 
    train=True, 
    selected_classes=selected_classes, 
    class_to_new_label=class_to_new_label
)

    train_dataset, val_dataset = torch.utils.data.random_split(filtered_dataset, [4000, 1000])

    remaining_samples = len(train_dataset) - num_samples
    small_dataset, remaining_dataset = random_split(train_dataset, [num_samples, remaining_samples])
    
    trainloader = DataLoader(remaining_dataset, shuffle=True, batch_size=train_batchsize)
    testloader = DataLoader(val_dataset, batch_size= test_batchsize, shuffle=False)
    small_dataloader = DataLoader(small_dataset, batch_size=small_batchsize, shuffle=True)
    
    
    small_dis = Distribute_MNIST(data_owners=data_owners, data_loader=small_dataloader, device=device)
    distributed_trainloader = Distribute_MNIST(data_owners=data_owners, data_loader=trainloader, device=device)
    distributed_testloader = Distribute_MNIST(data_owners=data_owners, data_loader=testloader, device=device)
    return distributed_trainloader, distributed_testloader, small_dis