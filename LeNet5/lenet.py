#!/usr/bin/env python
import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import torchvision.utils as utils
import matplotlib.pyplot as plt
from torchmetrics import Accuracy
from torchinfo import summary

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
                                        transform=transforms.Compose([transforms.ToTensor()]))
        self.test_set  = datasets.MNIST(root='./',
                                        train=False,
                                        download=True,
                                        transform=transforms.Compose([transforms.ToTensor()]))
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
        self.fully_connected2 = nn.Sequential(nn.Linear(64, 10))                  

    def forward(self, x):
            out = self.convolution_layer1(x)
            out = self.convolution_layer2(out)
            out = self.convolution_layer3(out)
            out = self.fully_connected1(out)
            out = self.fully_connected2(out)
            return out

class Learning_class():
    def __init__(self, model):
        self.epochs = 5
        self.device = torch.device(dev)
        self.model = model.to(self.device)
        self.loss_func = nn.CrossEntropyLoss()
        self.accuracy = Accuracy(task='multiclass', num_classes=10)
        self.accuracy = self.accuracy.to(dev)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.train_loss_hist = []
        self.train_acc_hist = []
        self.test_loss_hist = []
        self.test_acc_hist = []
        self.history = []

    def train(self, history = False):
        for epoch in range(self.epochs):
            train_loss, train_acc = 0.0, 0.0
            for images, labels in self.model.train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                self.model.train()
                outputs = self.model(images)
                loss = self.loss_func(outputs, labels)
                with torch.no_grad():
                    train_loss += loss.item()
                acc = self.accuracy(outputs, labels)
                train_acc += acc
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            train_loss /= len(self.model.train_loader)
            train_acc /= len(self.model.train_loader)

            test_loss, test_acc = 0.0, 0.0
            self.model.eval()
            with torch.inference_mode():
                for images, labels in self.model.test_loader:
                    images = images.to(self.device)
                    labels = labels.to(self.device)
                    outputs = self.model(images)
                    loss = self.loss_func(outputs,labels)
                    acc = self.accuracy(outputs,labels)
                    with torch.no_grad():
                        test_loss += loss.item()
                        test_acc += acc
                test_loss /= len(self.model.test_loader)
                test_acc /= len(self.model.test_loader)

            print(f"Epoch: {epoch+1} Train loss: {train_loss: .5f} Train acc: {train_acc: .5f} Test loss: {test_loss: .5f} Test acc: {test_acc: .5f}")
            if (history):
                self.train_loss_hist.append(train_loss)
                self.test_loss_hist.append(test_loss)
                self.train_acc_hist.append(train_acc.tolist())
                self.test_acc_hist.append(test_acc.tolist())

        self.history.append(self.train_loss_hist)
        self.history.append(self.test_loss_hist)
        self.history.append(self.train_acc_hist)
        self.history.append(self.test_acc_hist)
        return self.model, self.history

    def plot_loss(self, history):
        plt.figure(figsize=(5, 5))
        plt.plot(range(1,self.epochs+1),history[0], label='Train', color='red')
        plt.plot(range(1,self.epochs+1),history[1], label='Test', color='green')
        plt.title('Loss history')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.show()

    def plot_accuracy(self, history):
        plt.figure(figsize=(5, 5))
        plt.plot(range(1,self.epochs+1),history[2], label='Train', color='red')
        plt.plot(range(1,self.epochs+1),history[3], label='Test', color='green')
        plt.title('Accuracy history')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.show()

def main():    
    model = LeNet5()
    summary(model=model, input_size=(1, 1, 28, 28), col_width=20,
            col_names=['input_size', 'kernel_size', 'output_size'],
            row_settings=['ascii_only'])

    learning_model = Learning_class(model)
    trained_model, history = learning_model.train(history=True)
    learning_model.plot_accuracy(history)
    learning_model.plot_loss(history)

    kernels3 = trained_model.convolution_layer3[0].weight.detach().clone()
    kernels3 = kernels3 - kernels3.min()
    kernels3 = kernels3 / kernels3.max()
    kernels3,_ = torch.max(kernels3, 1, keepdim=True)
    filter_img = utils.make_grid(kernels3, nrow = 12)
    img = transforms.ToPILImage()(filter_img) 
    img = utils.save_image(filter_img, 'encoder_conv1_filters.png' ,nrow = 12)   

if __name__ == '__main__':
    main()