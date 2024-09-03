# Tensor Basis Machine Learning (TBML)

## Basic Overview
The TBML package is a collection of machine learning frameworks for predicting turbulence variables using scalar invariant basis and basis integrity tensors.

![image](https://github.com/user-attachments/assets/f475e160-cfce-41b0-a813-76982b5fc425)

**Figure 1** Example of models in the TBML package. The tensor basis neural network is shown on the left and the turbulent kinetic energy neural network is shown on the right.

## Installation
This package was developed with Python 3.10 and the following libraries:
- numpy 1.22
- torch 1.13
- pandas 1.4
- scikit-learn 1.2
- matplotlib 3.7
- seaborn 0.12

## Frameworks
TBML currently contains four frameworks: tensor basis neural network (TBNN), turbulent kinetic energy neural network (TKENN), zonal neural networks and tensor basis mixture density network (TBmix).

### Tensor Basis Neural Network (TBNN)
As the original TBNN developed by Ling *et al.* (2016) was implemented in Theano, this package offers a modern implementation of TBNN with PyTorch. Users should run TBNN/frontend.py with tbnn_main argument version="v1" to train, validate, and test TBNN.

Ling, J., Kurzawski, A. & Templeton, J. (2016) Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics* **807**, 155–166.

## Framework Feature Analysis



## References
The tensor basis neural network originally developed by Ling et al. (2016) is implemented with PyTorch in this package. The zonal modification to the tensor basis neural network detailed in Man et al. (2023) 



