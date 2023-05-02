# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 13:48:40 2023

@author: NHS-Notebook
"""
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision

from torchvision import transforms

import os
import csv_dateien_lesen as lib
import cv2
import matplotlib.pyplot as plt

class MyDataset(Dataset):
    def __init__(self, is_train=True, transform=None):
        super().__init__()
        self.is_train = is_train
        self.transform = transform  
        
        
        ###################################
        # load all data if it fits into memory, otherwise put references (e. g. file paths) in a list
        if self.is_train:
            self.data = lib.load()[0] # train_dataset_final
        else:
            self.data = lib.load()[1] # val_dataset_final
        ###################################
        
        
    def __getitem__(self, idx):
        ###################################
        # get the sample with index idx, it is self.data[idx] if all data is in memory, 
        # otherwise acquire the data associated with the reference self.data[idx]
        img_array, label = self.data[idx]
        
        # resize the image using opencv
        # we might have to experiment with different sizes to see which works best for your specific task and dataset.
        img_array = cv2.resize(img_array, (26, 26), interpolation=cv2.INTER_CUBIC)
        ###################################
        
        
        # convert the image to a PyTorch tensor and also scale the pixel values to be between 0 and 1
        # This is necessary because the pixel values in the original image are usually in the range of 0 to 255, 
        # which is too large for most neural network models.
        img_tensor = transforms.functional.to_tensor(img_array)
        
        
        if self.transform:
            
            # apply additional transformations if provided
            
            img_tensor = self.transform(img_tensor)                  # modify raw data if necessary
        
        
        # By returning the image as a tensor, we ensure that the model will receive the input data in the correct format. 
        # If we returned the image as a list of pixels, 
        # the model would not be able to process the input data correctly.
        
        sample = img_tensor, label

        return sample

    def __len__(self):
        return len(self.data)    
        

"""
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
"""


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
    
    #demo1(train_ds)
    #demo2(train_ds)
    
    train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    demo1(train_dl)
    demo2(train_dl)

    # There are also well-known datasets already prepared.
    #root_dir = os.path.dirname(__file__)
    #print(root_dir)
    #train_ds = torchvision.datasets.FashionMNIST(root_dir+'/fashion_mnist_data', download=True)
    #demo1(train_ds)

    # also try:
    #train_ds = torchvision.datasets.MNIST(root_dir+'/mnist_data', download=True)
    #train_dl = DataLoader(train_ds, batch_size=1, shuffle=True)
    #demo1(train_dl)
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








    
    #img = cv2.imread('1/'+train_ds[0][0])
    
    #img = train_ds[0][0]
   
    # for i in range(6,21):
    #     img[i][6] = (0,0,0)
    #     img[i][22] = (0,0,0)
    # for i in range(6,22):
    #     img[6][i] = (0,0,0)
    #     img[21][i] = (0,0,0)
    
    
    #cropped_img = img[5:40,6:43]
    # der andere slice mit zwei klammern funktioniert nicht
    #cv2.rectangle(img, (5,5),(20,21),(0,0,0),-1)
    
    #print (img.shape)
    #plt.imshow(cropped_img)
    
    


