
# **AutoEncoder based computational framework for tumor microenvironment decomposition and biomarker identification in metastatic melanoma**
the implementation of EPMDA
## 1. Required python packages
```
python 3.8
pytorch
sklearn
pandas
numpy
```
## 2. How to use
### 2.1 train the autoencoder model
```
autoencoder_skcm.py
```
### 2.2 calculate the signature for new sample
```
autoencoder_to_feature.py
```
### 2.3 calculate the gene contribution to the signature
```
autoencoder_weights.py
```
## 3. the trained model on TCGA data set

```
autoencoder_metastatic_lymphocyte_high_20_1e-05.pth
autoencoder_metastatic_lymphocyte_low_20_1e-05.pth
```
