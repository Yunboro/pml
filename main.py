# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:48:40 2023

@author: NHS-Notebook
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import os
import csv_dateien_lesen as lib

class MyDataset(Dataset):
    def __init__(self, is_train=True, transform=None):
        super().__init__()
        self.is_train = is_train
        self.transform = transform  
        ###################################
        # load all data if it fits into memory, otherwise put references (e. g. file paths) in a list
        if self.is_train:
            self.data = lib.load()
        else:
            self.data = [(11.5,1), (12.5,1), (9.,1), (14.5,0), (17.,0)]
        ###################################

    def __getitem__(self, idx):
        ###################################
        # get the sample with index idx, it is self.data[idx] if all data is in memory, 
        # otherwise acquire the data associated with the reference self.data[idx]
        sample = self.data[idx]
        ###################################
        if self.transform:
            sample = self.transform(sample)                  # modify raw data if necessary
        return sample

    def __len__(self):
        return len(self.data)


def demo1(dl):
    ''' Using iterators and next with an iterable such as a Dataset or a DataLoader, 
        a StopIteration exception may be thrown if no further data is available.
    ''' 
    it = iter(dl)
    sample = next(it)
    print(sample)
    sample = next(it)
    print(sample)

    
def demo2(dl):
    ''' Using a while loop with an iterable such as a Dataset or a DataLoader
    ''' 
    for sample in dl:
        print(sample)


if __name__ == '__main__':
    # Test your dataset first in a pure Python/Numpy environment, you do not need to know
    # much about Torch for it
    train_ds = MyDataset()
    demo1(train_ds)
    demo2(train_ds)
    
    #train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    #demo1(train_dl)
    #demo2(train_dl)
    img = imread('1/'+train_ds[0][0])
    print(img)
    # There are also well-known datasets already prepared.
    #root_dir = os.path.dirname(__file__)
    #print(root_dir)
    #train_ds = torchvision.datasets.FashionMNIST(root_dir+'/fashion_mnist_data', download=True)
    #demo1(train_ds)

# also try:
#   train_ds = torchvision.datasets.MNIST(root_dir+'/mnist_data', download=True)
# for the MNIST data. If the download is not successful, search and copy the files 
#   t10k-labels-idx1-ubyte.gz   and   train-labels-idx1-ubyte.gz
# from the internet and copy it into the mnist_data/MNIST/raw directory.

# Remark: The MNIST server rejects standard Python requests. You may change the requesting user by adding:
# -----------------------------------------------------
# import urllib
# opener = urllib.request.build_opener()
# opener.addheaders = [('User-agent', 'Mozilla/5.0')]
# urllib.request.install_opener(opener)
# -----------------------------------------------------
