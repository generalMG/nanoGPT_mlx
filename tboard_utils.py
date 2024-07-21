from tensorboardX import SummaryWriter

_TENSORBOARD_WRITER = None

def init_tensorboard(logdir: str, **kwargs):
    global _TENSORBOARD_WRITER
    _TENSORBOARD_WRITER = SummaryWriter(logdir, **kwargs)


def get_tensorboard() -> SummaryWriter:
    global _TENSORBOARD_WRITER
    assert _TENSORBOARD_WRITER is not None, (
        "get_tensorboard() called before init_tensorboard(); please specify "
        "a logdir to init_tensorboard() first."
    )
    return _TENSORBOARD_WRITER