# Attentional Boltzmann machine

This repository provides a Python implementation of the *attentional Boltzmann machine* (AttnBM) presented in the paper "Attention in a family of Boltzmann machines emerging from modern Hopfield networks," [arXiv:2212.04692](https://arxiv.org/abs/2212.04692).

## Usage

To train the AttnBM model for a data, e.g. MNIST dataset, run the following:

```bash
python attnbm.py \
    --data_source mnist \
    --n_sample 200 \
    --batch_size 5 \
    --n_hidden 900 \
    --epoch 10000 \
    --lr 0.01
```

We also give an illustrative demonstration in the [`attnbm.ipynb`](./attnbm.ipynb) notebook. In this notebook and the source code, we consider only the case of P=200 for the MNIST dataset, while the cases of P=50000 and the van Hateren natural images can easily be obtained by slightly modifying the Step 1. For more details, see Sec. 3.5 of the paper.

## Citation

If you use our code, or otherwise find our work useful, please cite the accompanying paper:

```bibtex
@article{ota2023attention,
    title={Attention in a family of Boltzmann machines emerging from modern Hopfield networks},
    author={Ota, Toshihiro and Karakida, Ryo},
    journal={Neural Computation},
    volume={35},
    number={8},
    pages={1463--1480},
    year={2023}
}
```
