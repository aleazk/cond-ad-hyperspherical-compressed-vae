# Method

A standard VAE maps an image `x` to Cartesian latent parameters `mu(x)` and `logvar(x)`.
In high dimensions, Gaussian samples concentrate near a thin hyperspherical shell and are
predominantly separated along vast equatorial regions. The compressed VAE keeps the
reparameterization in Cartesian coordinates but evaluates a KLD-like regularizer after a
differentiable conversion to hyperspherical coordinates.

For conditional OOD detection, each CIFAR-10 class is assigned a latent Cartesian axis.
Before computing angular statistics, class `k` is rotated by `k * class_axis_stride` latent
coordinates. Full compression places targets on all compressible angular cosine coordinates;
the vMF-like ablation targets only the first angular coordinate.

The sampled latent is normalized to radius `sqrt(d)`. After training on CIFAR-10, the mean
vectors of the CIFAR-10 training set form the reference bank. The anomaly score for a query
is the mean Euclidean distance to its three nearest reference means.

This repository preserves the historical loss constants and piecewise annealing schedule in
the reference configuration. See `src/compressed_vae/losses.py` for the exact implementation.
