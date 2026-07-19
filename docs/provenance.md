# Code provenance

The original research archive contained a compact VAE implementation derived from the
Soft-IntroVAE architecture and subsequently modified by Olivier Salvado and Alejandro
Ascarate. The upstream Soft-IntroVAE repository is licensed under Apache-2.0.

The public package retains the checkpoint-compatible residual encoder/decoder naming while
rewriting the experiment orchestration, configuration handling, metric computation, tests,
and documentation. The hyperspherical coordinate transformation and compression objective
are presented as first-class modules.
