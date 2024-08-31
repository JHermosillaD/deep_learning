#!/bin/python3
from mpl_toolkits.mplot3d import axes3d
import matplotlib.pyplot as plt
import numpy as np 

class gradient_descent_family:
    def  __init__(self):
        self.x = 0
        self.y = 0
        self.init_x = 0
        self.init_y = 0
        self.learning_rate = 0
        self.steps = 0
        self.x_min_lst = []
        self.y_min_lst = []
        self.z_lst = []

    def set_data(self, upper_limit, bottom_limit, points, init_x, init_y):
        self.x = np.linspace(bottom_limit, upper_limit, points)
        self.y = np.linspace(bottom_limit, upper_limit, points)
        self.init_x = init_x
        self.init_y = init_y

    def set_hyperparameters(self, steps, learning_rate):
        self.steps = steps
        self.learning_rate = learning_rate

    def evaluate_function(self, x, y):
        return x**2 + y**2
    
    def evaluate_gradient(self, x, y):
        return 2*x, 2*y

    def gradient_descent(self):
        x_min = self.init_x
        y_min = self.init_y
        for step in range(int(self.steps)):
            z_fun = self.evaluate_function(x_min, y_min)
            self.x_min_lst.append(x_min)
            self.y_min_lst.append(y_min)
            self.z_lst.append(z_fun)
            x_grad, y_grad = self.evaluate_gradient(x_min, y_min)
            x_min -= self.learning_rate*x_grad
            y_min -= self.learning_rate*y_grad
    
    def momentum_gradient_descent(self, rho):
        vx = 0
        vy = 0
        x_min = self.init_x
        y_min = self.init_y
        for step in range(int(self.steps)):
            z_fun = self.evaluate_function(x_min, y_min)
            self.x_min_lst.append(x_min)
            self.y_min_lst.append(y_min)
            self.z_lst.append(z_fun)
            x_grad, y_grad = self.evaluate_gradient(x_min, y_min)
            vx = rho * vx + x_grad
            vy = rho * vy + y_grad
            x_min -= self.learning_rate*vx
            y_min -= self.learning_rate*vy

    def plot_results(self):
        X, Y = np.meshgrid(self.x, self.y)
        Z = X**2 + Y**2
        ax = plt.figure().add_subplot(projection='3d')
        ax.plot_surface(X, Y, Z, edgecolor='royalblue', lw=0.5, rstride=8, cstride=8, alpha=0.3)
        ax.scatter(self.x_min_lst, self.y_min_lst, self.z_lst, color='red', s=15, alpha=1.0)
        ax.plot(self.x_min_lst, self.y_min_lst, self.z_lst, color='black', linewidth=1)
        ax.grid(False)
        plt.show()