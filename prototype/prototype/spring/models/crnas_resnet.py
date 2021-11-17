import torch
from .resnet import resnet_custom
from prototype.spring.models.utils.modify import modify_state_dict


__all__ = ['crnas_resnet18c', 'crnas_resnet50c', 'crnas_resnet101c']

model_urls = {
    'crnas_resnet18c': 'http://spring.sensetime.com/drop/$/cEidS.pth',
    'crnas_resnet50c': 'http://spring.sensetime.com/drop/$/M2SIn.pth',
    'crnas_resnet101c': 'http://spring.sensetime.com/drop/$/hhqwa.pth'
}

model_performances = {
    'crnas_resnet18c': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 5.961,
            'input_size': (3, 224, 224), 'accuracy': 72.302},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 48.213,
            'input_size': (3, 224, 224), 'accuracy': 72.302},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 375.048,
            'input_size': (3, 224, 224), 'accuracy': 72.302},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 1.341,
            'input_size': (3, 224, 224), 'accuracy': 72.276},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 3.516,
            'input_size': (3, 224, 224), 'accuracy': 72.276},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 15.569,
            'input_size': (3, 224, 224), 'accuracy': 72.276},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 51.983,
            'input_size': (3, 224, 224), 'accuracy': 72.462},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 394.917,
            'input_size': (3, 224, 224), 'accuracy': 72.462},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 3094.992,
            'input_size': (3, 224, 224), 'accuracy': 72.462},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 3.419,
            'input_size': (3, 224, 224), 'accuracy': 71.932},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 26.319,
            'input_size': (3, 224, 224), 'accuracy': 71.932},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 1.647,
            'input_size': (3, 224, 224), 'accuracy': 72.462},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 5.741,
            'input_size': (3, 224, 224), 'accuracy': 72.462},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 79.038,
            'input_size': (3, 224, 224), 'accuracy': 72.462},
    ],
    'crnas_resnet50c': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 14.378,
            'input_size': (3, 224, 224), 'accuracy': 76.938},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 115.103,
            'input_size': (3, 224, 224), 'accuracy': 76.938},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 901.291,
            'input_size': (3, 224, 224), 'accuracy': 76.938},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 2.382,
            'input_size': (3, 224, 224), 'accuracy': 77.026},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 7.117,
            'input_size': (3, 224, 224), 'accuracy': 77.026},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 36.430,
            'input_size': (3, 224, 224), 'accuracy': 77.026},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 107.989,
            'input_size': (3, 224, 224), 'accuracy': 77.032},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 831.789,
            'input_size': (3, 224, 224), 'accuracy': 77.032},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 6664.900,
            'input_size': (3, 224, 224), 'accuracy': 77.032},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 5.425,
            'input_size': (3, 224, 224), 'accuracy': 76.848},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 42.982,
            'input_size': (3, 224, 224), 'accuracy': 76.848},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 2.959,
            'input_size': (3, 224, 224), 'accuracy': 77.052},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 13.082,
            'input_size': (3, 224, 224), 'accuracy': 77.052},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 169.762,
            'input_size': (3, 224, 224), 'accuracy': 77.052},
    ],
    'crnas_resnet101c': [
        {'hardware': 'hisvp-nnie11-int8', 'batch': 1, 'latency': 24.670,
            'input_size': (3, 224, 224), 'accuracy': 77.16},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 8, 'latency': 220.082,
            'input_size': (3, 224, 224), 'accuracy': 77.16},
        {'hardware': 'hisvp-nnie11-int8', 'batch': 64, 'latency': 1568.149,
            'input_size': (3, 224, 224), 'accuracy': 77.16},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 1, 'latency': 3.545,
            'input_size': (3, 224, 224), 'accuracy': 77.168},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 8, 'latency': 12.470,
            'input_size': (3, 224, 224), 'accuracy': 77.168},
        {'hardware': 'cuda11.0-trt7.1-int8-P4', 'batch': 64, 'latency': 65.168,
            'input_size': (3, 224, 224), 'accuracy': 77.168},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 1, 'latency': 197.911,
            'input_size': (3, 224, 224), 'accuracy': 77.33},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 8, 'latency': 1498.931,
            'input_size': (3, 224, 224), 'accuracy': 77.33},
        {'hardware': 'cpu-ppl2-fp32', 'batch': 64, 'latency': 12522.792,
            'input_size': (3, 224, 224), 'accuracy': 77.33},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 1, 'latency': 8.024,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'halnn0.4-stpu-int8', 'batch': 8, 'latency': 63.105,
            'input_size': (3, 224, 224), 'accuracy': None},
        {'hardware': 'acl-ascend310-fp16', 'batch': 1, 'latency': 4.916,
            'input_size': (3, 224, 224), 'accuracy': 77.334},
        {'hardware': 'acl-ascend310-fp16', 'batch': 8, 'latency': 21.721,
            'input_size': (3, 224, 224), 'accuracy': 77.334},
        {'hardware': 'acl-ascend310-fp16', 'batch': 64, 'latency': 274.669,
            'input_size': (3, 224, 224), 'accuracy': 77.334},
    ]
}


def crnas_resnet18c(pretrained=False, pretrained_type='imagenet', **kwargs):
    kwargs['ceil_mode'] = True
    model = resnet_custom(block='basic', layers=[1, 1, 2, 4], **kwargs)
    model.performance = model_performances['crnas_resnet18c']
    if pretrained:
        model_url = model_urls['crnas_resnet18c']
        state_dict = torch.hub.load_state_dict_from_url(model_url, map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def crnas_resnet50c(pretrained=False, **kwargs):
    kwargs['ceil_mode'] = True
    model = resnet_custom(block='bottleneck', layers=[1, 3, 5, 7], **kwargs)
    model.performance = model_performances['crnas_resnet50c']
    if pretrained:
        model_url = model_urls['crnas_resnet50c']
        state_dict = torch.hub.load_state_dict_from_url(model_url, map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model


def crnas_resnet101c(pretrained=False, **kwargs):
    kwargs['ceil_mode'] = True
    model = resnet_custom(block='bottleneck', layers=[2, 3, 17, 11], **kwargs)
    model.performance = model_performances['crnas_resnet101c']
    if pretrained:
        model_url = model_urls['crnas_resnet101c']
        state_dict = torch.hub.load_state_dict_from_url(model_url, map_location='cpu')
        state_dict = modify_state_dict(model, state_dict)
        model.load_state_dict(state_dict, strict=False)
    return model
