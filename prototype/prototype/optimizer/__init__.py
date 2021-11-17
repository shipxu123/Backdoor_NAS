from torch.optim import SGD, RMSprop, Adadelta, Adagrad, Adam  # noqa F401
from .lars import LARS  # noqa F401
# from .fp16_optim import FP16SGD, FP16RMSprop  # noqa F401

from .larc import LARC
from .lars_simclr import LARS_simclr

def optim_entry(config):
    rank = 0
    return globals()[config['type']](**config['kwargs'])
