from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class WarmupStableDecayLR(LRScheduler):
    def __init__(
        self,
        optimizer: Optimizer,
        max_lr: float,
        warmup_iters: float,
        initial_div_factor: float,
        decay_iters: float = -1,
        final_div_factor: float = -1,
        epochs: int = -1,
        steps_per_epoch: int = -1,
        last_epoch: int = -1,
    ):
        """
        Scheduler with warmup, stable, and decay phases.

        Args:
            optimizer (Optimizer): Wrapped optimizer.
            max_lr (float): The target learning rate at the peak of the warmup phase.
            warmup_iters (float): The number of iterations for the warmup phase.
                                If <= 1, it represents the fraction of total iterations.
            initial_div_factor (float): Initial learning rate is max_lr / initial_div_factor.
                                If <= 0, the initial learning rate is 0.
            decay_iters (float): The number of iterations for the decay phase.
                                If <= 1, it represents the fraction of total iterations. Default: -1 (no decay).
            final_div_factor (float): Final learning rate is max_lr / final_div_factor. Default: -1.
                                If <= 0, the final learning rate is 0.
            epochs (int): Total number of epochs for training. Must be provided if warmup_iters or decay_iters <= 1.
            steps_per_epoch (int): Number of steps per epoch. Must be provided if warmup_iters or decay_iters <= 1.
            last_epoch (int): The index of the last epoch. Default: -1.

        Raises:
            AssertionError: If epochs and steps_per_epoch are not provided when required.
            AssertionError: If total iterations are less than the sum of warmup_iters and decay_iters.
        """

        if warmup_iters <= 1 or decay_iters <= 1 or decay_iters != 0:
            assert (
                epochs > 0 and steps_per_epoch > 0
            ), "Epochs and steps_per_epoch must be provided if warmup_iters / decay_iters <= 1 or decay_iters is set."

        if warmup_iters <= 0:
            self.warmup_iters = 0
        elif warmup_iters <= 1:
            self.warmup_iters = int(warmup_iters * epochs * steps_per_epoch)
        else:
            self.warmup_iters = int(warmup_iters)

        if decay_iters <= 0:
            self.decay_iters = 0
        elif decay_iters <= 1:
            self.decay_iters = int(decay_iters * epochs * steps_per_epoch)
        else:
            self.decay_iters = int(decay_iters)

        total_iters = epochs * steps_per_epoch
        self.stable_iters = max(0, total_iters - self.warmup_iters - self.decay_iters)

        assert (
            total_iters >= self.warmup_iters + self.decay_iters
        ), "Total iterations must be greater than or equal to warmup_iters + decay_iters."

        self.target_lr = max_lr
        if initial_div_factor <= 0:
            self.initial_lr = 0
        else:
            self.initial_lr = max_lr / initial_div_factor

        if final_div_factor <= 0:
            self.final_lr = 0
        else:
            self.final_lr = max_lr / final_div_factor
        super(WarmupStableDecayLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch < self.warmup_iters:
            # Warm-up phase
            warmup_lr = (
                self.initial_lr
                + (self.target_lr - self.initial_lr)
                * (self.last_epoch + 1)
                / self.warmup_iters
            )
            if self.verbose:
                print(f"Warm-up phase: Epoch {self.last_epoch}, LR {warmup_lr}")
            return [warmup_lr for _ in self.base_lrs]
        elif self.last_epoch < self.warmup_iters + self.stable_iters:
            # Stable phase
            if self.verbose:
                print(f"Stable phase: Epoch {self.last_epoch}, LR {self.target_lr}")
            return [self.target_lr for _ in self.base_lrs]
        else:
            # Decay phase
            if self.decay_iters > 0:
                decay_start = self.warmup_iters + self.stable_iters
                decay_progress = (self.last_epoch - decay_start + 1) / self.decay_iters
                decay_lr = (
                    self.target_lr - (self.target_lr - self.final_lr) * decay_progress
                )
                if self.verbose:
                    print(f"Decay phase: Epoch {self.last_epoch}, LR {decay_lr}")
                return [max(self.final_lr, decay_lr) for _ in self.base_lrs]
            else:
                # No decay phase, keep stable LR
                if self.verbose:
                    print(
                        f"No Decay phase: Epoch {self.last_epoch}, LR {self.target_lr}"
                    )
                return [self.target_lr for _ in self.base_lrs]
