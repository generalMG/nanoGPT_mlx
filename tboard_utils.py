"""
TensorBoard utilities for training visualization.

This module provides a simple global writer pattern for TensorBoard logging
during model training. Initialize once with init_tensorboard() and retrieve
the writer anywhere with get_tensorboard().
"""

from typing import Optional

from tensorboardX import SummaryWriter

# Global TensorBoard writer instance
_TENSORBOARD_WRITER: Optional[SummaryWriter] = None


def init_tensorboard(logdir: str, **kwargs) -> SummaryWriter:
    """
    Initialize the global TensorBoard writer.

    Args:
        logdir: Directory path for TensorBoard log files.
        **kwargs: Additional arguments passed to SummaryWriter.

    Returns:
        The initialized SummaryWriter instance.

    Example:
        >>> init_tensorboard("./logs/experiment1")
        >>> writer = get_tensorboard()
        >>> writer.add_scalar("loss", 0.5, step=100)
    """
    global _TENSORBOARD_WRITER
    _TENSORBOARD_WRITER = SummaryWriter(logdir, **kwargs)
    return _TENSORBOARD_WRITER


def get_tensorboard() -> SummaryWriter:
    """
    Get the global TensorBoard writer.

    Returns:
        The initialized SummaryWriter instance.

    Raises:
        RuntimeError: If init_tensorboard() has not been called first.

    Example:
        >>> writer = get_tensorboard()
        >>> writer.add_scalar("train/loss", loss_value, global_step)
    """
    global _TENSORBOARD_WRITER
    if _TENSORBOARD_WRITER is None:
        raise RuntimeError(
            "get_tensorboard() called before init_tensorboard(). "
            "Please call init_tensorboard(logdir) first."
        )
    return _TENSORBOARD_WRITER


def close_tensorboard() -> None:
    """
    Close the global TensorBoard writer and release resources.

    Safe to call even if writer was never initialized.
    """
    global _TENSORBOARD_WRITER
    if _TENSORBOARD_WRITER is not None:
        _TENSORBOARD_WRITER.close()
        _TENSORBOARD_WRITER = None
