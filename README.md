# Anomaly detection for solder joints using beta-VAE

This repository is the implementation for the paper titled "Anomaly Detection for Solder Joints using beta-VAE".

Ulger, Furkan, Seniha Esen Yuksel, and Atila Yilmaz. "Anomaly Detection for Solder Joints Using β-VAE." IEEE Transactions on Components, Packaging and Manufacturing Technology 11.12 (2021): 2214-2221.

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


