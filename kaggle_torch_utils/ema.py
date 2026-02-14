"""Exponential Moving Average (EMA) for any PyTorch model."""

import warnings
from typing import Dict

import torch
import torch.nn as nn


class EMA:
    """
    Maintains exponential moving average of model parameters.

    Usage:
        ema = EMA(model, decay=0.9998)

        # During training (after optimizer.step()):
        ema.update()

        # For evaluation:
        ema.apply_shadow()
        # ... evaluate ...
        ema.restore()

        # To get EMA state dict:
        ema_state = ema.get_shadow_state_dict()
    """

    def __init__(self, model: nn.Module, decay: float = 0.9998):
        """
        Args:
            model: The model to track. Pass the raw model (not DataParallel wrapped).
            decay: EMA decay factor. Higher = slower update. Typical: 0.999-0.9999.
        """
        self.model = model
        self.decay = decay
        self.shadow: Dict[str, torch.Tensor] = {}
        self.backup: Dict[str, torch.Tensor] = {}
        for name, p in model.named_parameters():
            if p.requires_grad:
                self.shadow[name] = p.data.clone()

    def update(self) -> None:
        """Update shadow parameters with current model parameters."""
        for name, p in self.model.named_parameters():
            if p.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = p.data.clone()
                else:
                    self.shadow[name] = (
                        (1.0 - self.decay) * p.data + self.decay * self.shadow[name]
                    )

    def apply_shadow(self) -> None:
        """Replace model parameters with EMA shadow parameters. Call restore() to undo."""
        self.backup = {}
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.shadow:
                self.backup[name] = p.data.clone()
                p.data = self.shadow[name].clone()

    def restore(self) -> None:
        """Restore original model parameters after apply_shadow()."""
        if not self.backup:
            warnings.warn(
                "EMA.restore() called but backup is empty. Did you call apply_shadow()?"
            )
            return
        for name, p in self.model.named_parameters():
            if p.requires_grad and name in self.backup:
                p.data = self.backup[name].clone()
        self.backup = {}

    def get_shadow_state_dict(self) -> Dict[str, torch.Tensor]:
        """Return the EMA shadow weights as a state dict."""
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_shadow_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> None:
        """Load EMA shadow weights from a state dict."""
        for k, v in state_dict.items():
            self.shadow[k] = v.clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        """Return shadow weights for checkpoint saving."""
        return {k: v.detach().clone().cpu() for k, v in self.shadow.items()}
