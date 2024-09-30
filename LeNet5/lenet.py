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
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize(mean = (0.1307,), std = (0.3081,))]))
        self.test_set  = datasets.MNIST(root='./',
                                        train=False,
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor(),
                                                                      transforms.Normalize(mean = (0.1307,), std = (0.3081,))]))
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
        self.convolution_layer3 = nn.Sequential(nn.Conv2d(16, 120, kernel_size=4, stride=1, padding=0))    
        self.fully_connected1 = nn.Sequential(nn.Flatten(),
                                              nn.Linear(120*1*1, 64),
                                              nn.Tanh())           
        self.fully_connected2 = nn.Sequential(nn.Linear(64, 10),
                                              nn.Tanh())                  

    def forward(self, x):
            out = self.convolution_layer1(x)
            out = self.convolution_layer2(out)
            out = self.convolution_layer3(out)
            out = self.fully_connected1(out)
            out = self.fully_connected2(out)
            return out
    
class Learning_classrom():
    def __init__(self, model):
        self.model = model
        self.epochs = 10
        self.device = torch.device(dev)
        self.loss_func = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)

    def train(self):
        x_train = []
        y_train = []
        for epoch in range(self.epochs):
            running_loss = 0.0
            count = 0
            for i, (images, labels) in enumerate(self.model.train_loader):     
                images = images.to(self.device)
                labels = labels.to(self.device)          
                self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)
                loss.backward()
                self.optimizer.step()
                with torch.no_grad():
                    running_loss += loss.item()  
                if i % 100 == 99:
                    count += 1
                    if (count == 9):
                        print('epoch: %d trainning loss: %.3f' %(epoch + 1, running_loss/100))
                        x_train.append(epoch)
                        y_train.append(running_loss)
                    running_loss = 0.0
        plt.plot(np.array(x_train), np.array(y_train))
        plt.show()       

    def test(self):
        x_test = []
        y_test = []
        with torch.no_grad():
            for epoch in range(self.epochs):
                running_loss = 0.0
                count = 0
                for i, (images, labels) in enumerate(self.model.test_loader):     
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_func(outputs, labels)
                    running_loss += loss.item() 
                    if i % 100 == 99:
                        count += 1
                        if (count == 9):
                            print('epoch: %d testing loss: %.3f' %(epoch + 1, running_loss/100))
                            x_test.append(epoch)
                            y_test.append(running_loss)
                        running_loss = 0.0
            plt.plot(np.array(x_test), np.array(y_test))
            plt.show()

def main():    
    model = LeNet5()
    print(model)
    model.to(torch.device(dev))
    learning = Learning_classrom(model)
    learning.train()
    learning.test()

if __name__ == '__main__':
    main()