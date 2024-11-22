from torch import nn, optim
import torch.nn.functional as F
from data_dis import *
from torchvision import datasets, transforms
import torch
import tqdm
from torch.utils.data import DataLoader,random_split,Subset


def data_pre(data_owners, device, train_batchsize = 64, test_batchsize = 1, small_batchsize = 32, num_samples = 100):

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),])

   
    trainset = datasets.CIFAR10(root='/home/duanyi/data', train=True,download=True, transform=transform_train)
    testset = datasets.CIFAR10(root='/home/duanyi/data', train=False, download=True, transform=transform_test)


    remaining_samples = len(trainset) - num_samples
    small_dataset, remaining_dataset = random_split(trainset, [num_samples, remaining_samples])
   

    trainloader = torch.utils.data.DataLoader(remaining_dataset, shuffle=True, batch_size=train_batchsize)
    testloader = torch.utils.data.DataLoader(testset, batch_size= test_batchsize, shuffle=True)
    small_dataloader = DataLoader(small_dataset, batch_size=small_batchsize, shuffle=True)
    

 
    small_dis = Distribute_MNIST(data_owners=data_owners, data_loader=small_dataloader, device=device)
    distributed_trainloader = Distribute_MNIST(data_owners=data_owners, data_loader=trainloader, device=device)
    distributed_testloader = Distribute_MNIST(data_owners=data_owners, data_loader=testloader, device=device)
    return distributed_trainloader, distributed_testloader, small_dis

  
