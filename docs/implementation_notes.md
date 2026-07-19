# Implementation notes

- The public code removes hard-coded cluster paths and all import-time experiment execution.
- The checkpoint loader accepts both the original `{"epoch", "model"}` layout and the new
  `{"epoch", "state_dict", "metadata"}` layout.
- The historical angular loss excludes the final signed hyperspherical angle from the mean
  compression term. This behavior is retained.
- The class-conditioned compression assumes ten classes and a class-axis stride of ten for
  the reference CIFAR configuration.
- A mini-batch must contain each CIFAR-10 class because the loss uses per-class batch
  statistics. Batch size 200 makes missing classes unlikely; a balanced sampler is a reasonable
  future improvement.
- The historical checkpoint was evaluated after loading a 300-epoch model and running one
  additional epoch. The exact archived scores are included separately from the deterministic
  public evaluator.

- The current manuscript prose states a 100-epoch schedule, whereas the attached CIFAR checkpoint
  metadata records a 300-epoch pretrained run plus one additional epoch. The public reproduction
  configuration follows the attached artifact; this discrepancy should be resolved before release.
