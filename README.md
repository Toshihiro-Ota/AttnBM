# Attention in a family of Boltzmann machines emerging from modern Hopfield networks

This notebook provides a Python implementation of the *attentional Boltzmann machine* (AttnBM) presented in the paper "Attention in a family of Boltzmann machines emerging from modern Hopfield networks," [arXiv:2212.04692](https://arxiv.org/abs/2212.04692).

We give a simple numerical demonstration in PyTorch. The results of Figures 1 & 2 in the paper can be reproduced by the following three steps:

1. Pre-processing the data (ZCA whitening)
1. Define and train AttnBM
1. Image reconstruction and visualization of the receptive fields

In this notebook we consider only the case of P=200 for the MNIST dataset, while the cases of P=50000 and the van Hateren natural images can easily be obtained by slightly modifying the Step 1 below. For more details, see Sec. 3.5 of the paper.
