import os    
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import numpy as np # import but unused
import torch
import torch.utils.data.dataloader as dataloader
from torch.utils.data import TensorDataset # import but unused
from torch import nn
import torchvision
from torchvision.transforms import ToTensor, Compose, Lambda
import matplotlib.pyplot as plt

from process_set_data import MyDataset # import but not YET used. We have to use it!!!!

root_dir = os.path.dirname(__file__)

batch_size = 128






#####################################################################################
# The data: This data is for experimantation
train_ds = torchvision.datasets.MNIST(root_dir+'/mnist_data', train=True, download=True, 
                                      transform=Compose([ToTensor(), Lambda(torch.flatten)]))
test_ds = torchvision.datasets.MNIST(root_dir+'/mnist_data', train=False, download=True, 
                                      transform=Compose([ToTensor(), Lambda(torch.flatten)]))
train_dl = dataloader.DataLoader(train_ds, shuffle=True, batch_size=batch_size, num_workers=0, drop_last=True)
test_dl = dataloader.DataLoader(test_ds, shuffle=False, batch_size=batch_size, num_workers=0, drop_last=True)
#####################################################################################






#####################################################################################
# THE DATA consists of images of traffic signals, sourced from our dataset. With this dataset we will trained our modell. 




train_ds = MyDataset()
 

train_dl = dataloader.DataLoader(train_ds, batch_size=1, shuffle=True)   # batch_size??? Regarding the batch size, we need to decide on a suitable value. What would be a reasonable size?


                                                                         # Do we have to flatten the data?: --> From N dimensions to one dimension
 
#####################################################################################




#####################################################################################
# THE MODEL:
    
    
    
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 80)
        self.lin2 = nn.Linear(80, 30)
        self.lin3 = nn.Linear(30, 10)
        
    def forward(self, X0):
        X1 = torch.relu(self.lin1(X0))
        X2 = torch.relu(self.lin2(X1))
        X3 = torch.softmax(self.lin3(X2), dim=-1)
        return X3

    def loss(self, Y_true, Y_pred):                                       # we put the loss fct in the class, this is not required
        Y_true_oh = torch.nn.functional.one_hot(Y_true, num_classes=10)   # recode to a one-hot tensor
        sample_loss = -(Y_true_oh*torch.log(Y_pred+1e-7)).sum(axis=1)
        return sample_loss.mean()
####################################################################################




####################################################################################
# THE TRAINER



def train_step(X, Y_true, mdl, opt):
    Y_pred = mdl(X)                        # predict
    L = mdl.loss(Y_true, Y_pred)           # compute the loss
    L.backward()                           # compute the gradients for the optimizer
    opt.step()                             # call the optimizer, modifies the weights
    opt.zero_grad()  
    return L.detach().numpy()              # we dont do further computations with L, the numpy value is sufficient for reporting
    
def train(train_dl, mdl, alpha, n_epochs):
    opt = torch.optim.Adam(mdl.parameters(), lr=alpha)                        # choose the optimizer
    hist = { 'loss': [] }
    for epoch in range(n_epochs):                                             # repeat for n epochs
        for step, (X, Y_true) in enumerate(train_dl):                         # repeat for all mini-batches
            L = train_step(X, Y_true, mdl, opt)
            hist['loss'].append(L)                                            # logging
            if step % 100 == 0:
                print(f'Epoch: {epoch}, step {step*batch_size:5d}/{len(train_dl.dataset)}:  loss: {L:.6f}')
    return hist
####################################################################################




if __name__ == '__main__':
    
    
    mdl = Model()
    hist = train(train_dl, mdl, alpha=0.001, n_epochs=3)            # We have to choose how many epochs???
    
    plt.plot(hist['loss'])
    plt.grid()
    plt.show(block=True)
    
    it = iter(test_dl)
    X_test, Y_test = next(it)
    Y_pred = mdl(X_test)
    print(Y_pred.detach().numpy())