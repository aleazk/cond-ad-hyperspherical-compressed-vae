# Reproducing CIFAR-10 vs CIFAR-100

## Exact archived metric verification

This does not require a GPU or dataset download:

```bash
python scripts/verify_archived_result.py
```

Expected output:

```text
AUROC  = 0.8954305
FPR95  = 0.2323
```

## Re-evaluate the supplied checkpoint

```bash
pip install -e .
bash scripts/reproduce_cifar10_vs_cifar100.sh
```

The script downloads CIFAR-10 and CIFAR-100, builds a CIFAR-10 training embedding bank,
evaluates all CIFAR-10 and CIFAR-100 test images, and writes metrics, scores, a ROC/score
figure, and a 3D coordinate-averaged latent visualization.

The old research script used random horizontal flips while building the training reference
bank. The public configuration defaults to deterministic evaluation without augmentation.
Set `evaluation.legacy_train_augmentation: true` to emulate the old pipeline more closely;
small numerical deviations from the archived metric are expected because the historical run's
random state had already advanced during training.

## Train from scratch

```bash
compvae-train --config configs/cifar10_cifar100_full.yaml
```

The attached checkpoint was produced from a 300-epoch pretrained CIFAR-10 run followed by one
additional epoch. The OOD dataset is used only for evaluation.
