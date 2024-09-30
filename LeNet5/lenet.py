#!/usr/bin/env python
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np

if torch.cuda.is_available(): 
    dev = "cuda:0"
else: 
    dev = "cpu"

class LeNet5(nn.Module):  
    def __init__(self):
        self.batch_size = 64
        self.train_set = datasets.MNIST(root='./',
                                        train=True,
                                        download=True,
                                        transform=transforms.Compose([transforms.Resize((32,32)),
                                                                    transforms.ToTensor()]))
        self.test_set  = datasets.MNIST(root='./',
                                        train=False,
                                        download=True,
                                        transform=transforms.Compose([transforms.Resize((32,32)),
                                                                    transforms.ToTensor()]))
        self.train_loader = torch.utils.data.DataLoader(dataset=self.train_set,
                                                        batch_size=self.batch_size,
                                                        shuffle=True)        
        self.test_loader  = torch.utils.data.DataLoader(dataset = self.test_set,
                                                        batch_size = self.batch_size,
                                                        shuffle = True)
        super(LeNet5, self).__init__()      
        self.convolution_layer1 = nn.Sequential(nn.Conv2d(1, 6, kernel_size=5, stride=1, padding=0),
                                                nn.Tanh(),
                                                nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.convolution_layer2 = nn.Sequential(nn.Conv2d(6, 16, kernel_size=5, stride=1, padding=0),
                                                nn.Tanh(),
                                                nn.MaxPool2d(kernel_size = 2, stride = 2))        
        self.convolution_layer3 = nn.Sequential(nn.Conv2d(16, 120, kernel_size=5, stride=1, padding=0),
                                                nn.Tanh())        
        self.fully_connected1 = nn.Linear(120, 64)
        self.fully_connected2 = nn.Linear(64, 10)

    def forward(self, x):
            out = self.convolution_layer1(x)
            out = self.convolution_layer2(out)
            out = self.convolution_layer3(out)
            out = out.reshape(out.size(0), -1)
            out = self.fully_connected1(out)
            out = self.fully_connected2(out)
            return out
    
class Learning_classrom():
    def __init__(self, model):
        self.model = model
        self.epochs = 5
        self.device = torch.device(dev)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)

    def train(self):
        x_train = []
        y_train = []
        iteration = 0
        for epoch in range(self.epochs):
            for i, (images, labels) in enumerate(self.model.train_loader):     
                images = images.to(self.device)
                labels = labels.to(self.device)          
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()     
                x_train.append(iteration)
                y_train.append(loss.item())
                iteration += 1
        plt.plot(np.array(x_train), np.array(y_train))
        plt.show()

    def test(self):
        x_test = []
        y_test = []
        iteration = 0
        with torch.no_grad():
            for images, labels in self.model.test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_func(outputs, labels)
                    x_test.append(iteration)
                    y_test.append(loss.item())
                    iteration += 1
        plt.plot(np.array(x_test), np.array(y_test))
        plt.show()

def main():    
    model = LeNet5()
    model.to(torch.device(dev))
    learning = Learning_classrom(model)
    learning.train()
    learning.test()

if __name__ == '__main__':
    main()