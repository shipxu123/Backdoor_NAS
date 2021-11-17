from .resnet import (  # noqa: F401
    resnet18, resnet26, resnet34, resnet50,
    resnet101, resnet152, resnet_custom
)
from .resnet_official import *

import pdb

def model_entry(config):
    if config['type'] not in globals():
        if config['type'].startswith('spring_'):
            from prototype.spring.models import SPRING_MODELS_REGISTRY
            #try:
            #    from prototype.spring.models import SPRING_MODELS_REGISTRY
            #except ImportError:
            #    print('Please install Spring2 first!')
            model_name = config['type'][len('spring_'):]
            config['type'] = model_name
            return SPRING_MODELS_REGISTRY.build(config)
        else:
            from prototype.spring import PrototypeHelper
            return PrototypeHelper.external_model_builder[config['type']](**config['kwargs'])
    return globals()[config['type']](**config['kwargs'])
