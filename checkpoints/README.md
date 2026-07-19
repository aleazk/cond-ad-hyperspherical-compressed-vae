# Reference checkpoint

`cifar10_full_compression_z128_ec126.pt` is the checkpoint supplied with the historical
CIFAR-10 versus CIFAR-100 run. It uses:

- latent dimension: 128;
- full angular compression setting: `end_compress=126`;
- target compression gain: 2500;
- class-axis stride: 10;
- normalized reparameterized samples on radius `sqrt(128)`.

The original run loaded a 300-epoch CIFAR-10 checkpoint and then performed one additional
epoch before evaluation. The converted checkpoint retains the supplied weights and adds
machine-readable metadata.
