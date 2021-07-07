# Differentiable Perturb-and-Parse operator

A PyTorch implementation of the continuous relaxation of the Eisner algorithm as described in the paper [DIFFERENTIABLE PERTURB-AND-PARSE:SEMI-SUPERVISED PARSING WITH A STRUCTURED VARIATIONAL AUTOENCODER](https://arxiv.org/pdf/1807.09875.pdf).

## Requirements
* Python 3.7
* PyTorch 1.8.1

### Arguments
1) `arc_scores` (Tensor): The arc-factored weights of dependencies.
2) `hard` (bool): `True` for hard (valid) non-differentiable dependency trees, `False` for soft differentiable dependency trees.
