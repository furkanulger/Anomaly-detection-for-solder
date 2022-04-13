# Anomaly Detection for Solder Joints Using *β*-VAE

This repository contains the implementation for the paper titled "Anomaly Detection for Solder Joints Using *β*-VAE".

Ulger, Furkan, Seniha Esen Yuksel, and Atila Yilmaz. "Anomaly Detection for Solder Joints Using *β*-VAE." IEEE Transactions on Components, Packaging and Manufacturing Technology 11.12 (2021): 2214-2221.

[[IEEE Xplore]](https://ieeexplore.ieee.org/document/9579423) [[arXiv]](https://arxiv.org/abs/2104.11927)

## Abstract
In the assembly process of printed circuit boards
(PCBs), most of the errors are caused by solder joints in
surface mount devices (SMDs). In the literature, traditional
feature extraction-based methods require designing hand-crafted
features and rely on the tiered red green blue (RGB) illumination
to detect solder joint errors, whereas the supervised convolutional
neural network (CNN)-based approaches require a lot of
labeled abnormal samples (defective solder joints) to achieve high
accuracy. To solve the optical inspection problem in unrestricted
environments with no special lighting and without the existence
of error-free reference boards, we propose a new beta-variational
autoencoder (beta-VAE) architecture for anomaly detection that
can work on both integrated circuit (IC) and non-IC components.
We show that the proposed model learns disentangled
representation of data, leading to more independent features and
improved latent space representations.We compare the activation
and gradient-based representations that are used to characterize
anomalies and observe the effect of different beta parameters on
accuracy and untwining the feature representations in beta-VAE.
Finally, we show that anomalies on solder joints can be detected
with high accuracy via a model trained directly on normal
samples without designated hardware or feature engineering.

<img width="643" alt="recons" src="https://user-images.githubusercontent.com/50952046/163036885-5e835ae1-b749-42a2-808a-9d104e7b271c.PNG">

## Installation
```
git clone https://github.com/furkanulger/Anomaly-detection-for-solder.git
cd Anomaly-detection-for-solder/
pip install -r requirements.txt
```

## Training Model
Type -h to get description of the parameters for each script ".py" as follows:
```
cd Anomaly-detection-for-solder/
python train.py -h 
```
![image](https://user-images.githubusercontent.com/50952046/163218586-b08e9e4e-7e0d-4faf-b7c3-5c9701d5a244.png)


To train Convolutional Autoencoder with reconstruction loss:
```
cd Anomaly-detection-for-solder/
python train.py -m CAE -bs(optional) 8 -lr(optional) 1e-4 -e(optional) 100
```

To train beta-Variational Autoencoder with ELBO objective:
```
python train.py -m VAE -b(optional) 3 -bs(optional) 8 -lr(optional) 1e-4 -e(optional) 100
```

To train Convolutional Autoencoder with gradient constraint:
```
python train_Gradloss.py -m CAE -bs(optional) 8 -lr(optional) 1e-4 -e(optional) 100
```

To train beta-Variational Autoencoder with gradient constraint:
```
python train_Gradloss.py -m VAE -b(optional) 3 -bs(optional) 8 -lr(optional) 1e-4 -e(optional) 100
```

## Testing Model
Type -h to get description of the parameters for each script ".py" as follows:
```
cd Anomaly-detection-for-solder/
python test.py -h 
```
![image](https://user-images.githubusercontent.com/50952046/163220891-6040cd4c-e2a6-4ad0-b480-fed2bf2123fe.png)


To train Convolutional Autoencoder with reconstruction loss:
```
python test.py -m CAE -p .\Results\Model_checkpoints\model_lowest_val_loss.pt -a Recon -th THRESHOLD_VALUE 
```
