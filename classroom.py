#!/bin/python3
from deep_modules import __version__
from deep_modules.gradient_descent import gradient_descent_family

def main():
    gd_fam = gradient_descent_family()
    gd_fam.set_data(-10, 10, 100, 0, 10)
    gd_fam.set_hyperparameters(10, 0.9)
    #gd_fam.gradient_descent()
    #gd_fam.momentum_gradient_descent(0.9)
    #gd_fam.plot_results()

if __name__ == "__main__":
    main()