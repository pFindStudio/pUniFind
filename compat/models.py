# Copyright (c) 2024, pUniFind Team
# Model framework for registering and building models

import argparse
import torch
import torch.nn as nn
from typing import Dict, Any, Optional, Type, Callable

# Global model registry
MODEL_REGISTRY: Dict[str, Type["BaseUnicoreModel"]] = {}
ARCH_MODEL_REGISTRY: Dict[str, Dict[str, Callable]] = {}
MODEL_DATACLASS_REGISTRY: Dict[str, Any] = {}


def register_model(name: str) -> Callable:
    """
    Decorator to register a model.

    Args:
        name: Name of the model

    Returns:
        Decorator function
    """
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        if name not in ARCH_MODEL_REGISTRY:
            ARCH_MODEL_REGISTRY[name] = {}
        return cls
    return decorator


def register_model_architecture(model_name: str, arch_name: str) -> Callable:
    """
    Decorator to register a model architecture.

    Args:
        model_name: Name of the model
        arch_name: Name of the architecture

    Returns:
        Decorator function
    """
    def decorator(fn):
        if model_name not in ARCH_MODEL_REGISTRY:
            ARCH_MODEL_REGISTRY[model_name] = {}
        ARCH_MODEL_REGISTRY[model_name][arch_name] = fn
        return fn
    return decorator


def build_model(
    args: argparse.Namespace,
    task: Optional[Any] = None,
) -> nn.Module:
    """
    Build a model from arguments.

    Args:
        args: Parsed arguments containing model configuration
        task: Optional task instance

    Returns:
        Model instance
    """
    # Ensure model registries are initialized
    if len(MODEL_REGISTRY) == 0:
        try:
            from PepMS.models import ms_pep
        except ImportError:
            pass

    model_name = getattr(args, 'arch', None)
    if model_name is None:
        raise ValueError("No architecture specified")

    if model_name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_REGISTRY.keys())}")

    model_cls = MODEL_REGISTRY[model_name]

    # Apply architecture defaults if available
    if model_name in ARCH_MODEL_REGISTRY:
        arch_name = model_name  # Use model name as default arch
        if arch_name in ARCH_MODEL_REGISTRY[model_name]:
            ARCH_MODEL_REGISTRY[model_name][arch_name](args)

    # Build model
    if hasattr(model_cls, 'build_model'):
        model = model_cls.build_model(args, task)
    else:
        model = model_cls(args)

    return model


class BaseUnicoreModel(nn.Module):
    """
    Base class for all models.
    """

    def __init__(self):
        super().__init__()
        self._num_updates = 0

    @staticmethod
    def add_args(parser: argparse.ArgumentParser) -> None:
        """
        Add model-specific arguments to the parser.

        Args:
            parser: ArgumentParser instance
        """
        pass

    @classmethod
    def build_model(
        cls,
        args: argparse.Namespace,
        task: Optional[Any] = None,
    ) -> "BaseUnicoreModel":
        """
        Build a new model instance.

        Args:
            args: Parsed arguments
            task: Optional task instance

        Returns:
            Model instance
        """
        return cls(args)

    def set_num_updates(self, num_updates: int) -> None:
        """
        Set the number of training updates.

        Args:
            num_updates: Number of updates
        """
        self._num_updates = num_updates
        for module in self.modules():
            if hasattr(module, 'set_num_updates') and module is not self:
                module.set_num_updates(num_updates)

    def get_num_updates(self) -> int:
        """
        Get the number of training updates.

        Returns:
            Number of updates
        """
        return self._num_updates

    def get_targets(self, sample: Dict[str, Any]) -> torch.Tensor:
        """
        Get targets from the sample.

        Args:
            sample: Sample dictionary

        Returns:
            Targets tensor
        """
        return sample.get("targets", sample.get("target", None))

    def prepare_for_inference(self) -> None:
        """
        Prepare the model for inference mode.
        """
        self.eval()

    def upgrade_state_dict(
        self,
        state_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Upgrade old state dicts to new format.

        Args:
            state_dict: Old state dict

        Returns:
            Upgraded state dict
        """
        return state_dict
