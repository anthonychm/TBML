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

### 1. Tensor Basis Neural Network (TBNN)
As the original TBNN developed by Ling *et al.* (2016) was implemented with Theano, this package offers a modern implementation of TBNN with PyTorch. Users should run `TBNN/frontend.py` to train, validate, and test TBNN.

Ling, J., Kurzawski, A. & Templeton, J. (2016) Reynolds averaged turbulence modelling using deep neural networks with embedded invariance. *Journal of Fluid Mechanics* **807**, 155–166. https://doi.org/10.1017/jfm.2016.615

### 2. Turbulent Kinetic Energy Neural Network (TKENN)
The TKENN proposed in Man *et al.* (2023) uses scalar invariant basis inputs to predict turbulent kinetic energy. Users should run `TKE_NN/tke_frontend.py` to train, validate, and test TKENN.

Man, A., Jadidi, M., Keshmiri, A., Yin, H. & Mahmoudi, Y. (2023) A divide-and-conquer machine learning approach for modeling turbulent flows. *Physics of Fluids* **35**, 055110. https://doi.org/10.1063/5.0149750

Preprint: https://arxiv.org/abs/2408.13568

### 3. Zonal Neural Networks
Based on the divide-and-conquer technique, Man *et al.* (2023) proposed that multiple instances of TBNN and TKENN can be trained to give more accurate anisotropy and turbulent kinetic energy predictions, respectively. This approach involves using scalar indicators to partition engineering flow domains into regions of flow physics called zones. For each zone, a model instance is trained on data from the zone. Then the model instances are validated and tested on data from the same type of zone in the validation and test cases. Users should run `Driver/zonal_driver_v2.py` to train, validate, and test zonal TBNNs and zonal TKENNs.

Man, A., Jadidi, M., Keshmiri, A., Yin, H. & Mahmoudi, Y. (2023) A divide-and-conquer machine learning approach for modeling turbulent flows. *Physics of Fluids* **35**, 055110. https://doi.org/10.1063/5.0149750

Preprint: https://arxiv.org/abs/2408.13568

### 4. Tensor Basis Mixture Density Network (TBMix)
The TBMix is a modified version of TBNN with mixture density outputs to give probabilistic predictions of anisotropy. This model was developed to show that it can overcome non-unique mapping in ML models based on the tensor basis representation. A beta version has been released on this remote repository while the research paper proposing this model is currently in progress/under review.

## Framework Mapping Analyser
A mapping analyser tool based on unsupervised clustering was developed to assess the relationship (i.e., mapping) between the inputs and outputs of these models. With this tool, it was shown for the first time that non-unique mapping can occur between the conventional scalar invariant inputs and anisotropy components in TBNN for two-dimensional flows (Man *et al.*, 2024). This tool can be used for any type and number of inputs for the frameworks in this package. Users should run `Driver/mapping_analyser.py` to evaluate the input-output relationship of their chosen framework.

Man, A., Jadidi, M., Keshmiri, A., Yin, H. & Mahmoudi, Y. (2024) Non-unique machine learning mapping in data-driven Reynolds averaged turbulence models. *Physics of Fluids* **36**, accepted/in press.

Preprint: https://arxiv.org/abs/2312.13005
