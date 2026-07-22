#!/usr/bin/env bash
set -euo pipefail
python -m compressed_vae.evaluate \
  --config configs/cifar10_cifar100_full.yaml \
  --checkpoint checkpoints/cifar10_full_compression_z128_ec126.pt \
  --protocol standard \
  --output-dir runs/cifar10_vs_cifar100_standard
