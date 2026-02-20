"""
Abstract interfaces for Hyper-MVE model components.

Ensures Phase 5 upgrades (e.g. hypnettorch) require zero changes to upper-level code.
"""
from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
import torch.nn as nn


class WeightGenerator(ABC, nn.Module):
    """Interface for hypernetworks that generate target network weights."""

    @abstractmethod
    def forward(self, context: torch.Tensor) -> torch.Tensor:
        """
        Generate flattened weight vector from context.

        Args:
            context: (batch_size, context_dim)
        Returns:
            flat_params: (batch_size, num_weights)
        """
        pass

    @abstractmethod
    def get_output_dim(self) -> int:
        """Return total number of generated weights."""
        pass


class ContextEncoder(ABC, nn.Module):
    """Interface for context encoders (MLP for Exp2, GRU for Exp3)."""

    @abstractmethod
    def forward(self, *args, **kwargs):
        """Encode input into context vector."""
        pass

    @abstractmethod
    def get_context_dim(self) -> int:
        """Return dimension of output context vector."""
        pass


class FunctionalNetwork(ABC, nn.Module):
    """Interface for networks that use externally provided weights."""

    @abstractmethod
    def forward(self, x: torch.Tensor, flat_params: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with external weights.

        Args:
            x: input tensor
            flat_params: flattened parameters from hypernetwork
        """
        pass

    @property
    @abstractmethod
    def total_params(self) -> int:
        """Total number of parameters needed."""
        pass

    @property
    @abstractmethod
    def param_shapes(self) -> List[Tuple[int, ...]]:
        """Shapes of all parameter tensors (compatible with hypnettorch)."""
        pass
