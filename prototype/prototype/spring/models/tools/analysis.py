import argparse

from spring.models import SPRING_MODELS_REGISTRY
from spring.models.tools.profile import get_model_profile
from spring.utils.log_helper import default_logger as logger


# supported hardware list
hardware_list = [
    'cpu', 'T4', 'P4', '3559A', '3519A',
]
# supported backend list
backend_list = [
    'nart', 'ppl2', 'cuda10.0-trt5.0', 'cuda10.0-trt7.0', 'cuda11.0-trt7.1',
    'cuda11.0-nart', 'hisvp-nnie11', 'hisvp-nnie12',
]
# supported data type list
data_type_list = [
    'fp32', 'fp16', 'int16', 'int8', 'int4',
]
# supported fused type list
supported_list = [
    'cpu-nart-fp32', 'cpu-ppl2-fp32',
    'T4-cuda10.0-trt7.0-fp32', 'P4-cuda10.0-trt7.0-fp32',
    'T4-cuda10.0-trt7.0-fp16', 'P4-cuda10.0-trt7.0-int8',
    'T4-cuda11.0-trt7.1-fp32', 'P4-cuda11.0-trt7.1-fp32',
    'T4-cuda11.0-trt7.1-fp16', 'T4-cuda11.0-trt7.1-int8', 'P4-cuda11.0-trt7.1-int8',
    '3559A-hisvp-nnie11-int16', '3559A-hisvp-nnie11-int8',
    '3519A-hisvp-nnie12-int16', '3519A-hisvp-nnie12-int8',
]
# platform needed to analysis
wanted_list = [
    'cpu-nart-fp32', 'cpu-ppl2-fp32',
    'T4-cuda11.0-trt7.1-int8', 'T4-cuda11.0-trt7.1-fp16',
    'P4-cuda11.0-trt7.1-int8', 'P4-cuda11.0-trt7.1-fp32',
    '3559A-hisvp-nnie11-int8', '3519A-hisvp-nnie12-int8',
]
# batch size list
batch_size_list = [1, 8, 64]
# input size list
input_size_list = [224]


def test_all():
    # hardware
    for hardware in hardware_list:
        # backend
        for backend in backend_list:
            # data type
            for data_type in data_type_list:
                platform = hardware + '-' + backend + '-' + data_type
                if (platform in supported_list) and (platform in wanted_list):
                    # batch size
                    for batch_size in batch_size_list:
                        with open('{}-batch{}.txt'.format(platform, batch_size), 'w') as f:
                            for model_name in model_list:
                                results = get_model_profile(
                                    model=model_name,
                                    input_size=224,
                                    input_channel=3,
                                    batch_size=batch_size,
                                    hardware=hardware,
                                    backend=backend,
                                    data_type=data_type,
                                    force_test=True
                                )
                                f.write('model: {}, params: {:.3f}M, FLOPs: {:.3f}M, latency: {:.3f}ms \n'.format(
                                    model_name, results['total_param'], results['flops'], results['latency']
                                ))
                                logger.info('model: {}, params: {:.3f}M, FLOPs: {:.3f}M, latency: {:.3f}ms \n'.format(
                                    model_name, results['total_param'], results['flops'], results['latency']
                                ))
                            logger.info('{}-batch{} has done!'.format(platform, batch_size))


def test_part(model_list=[], input_size=224, hardware='cpu', backend='nart',
              data_type='fp32', batch_size=1, force_test=False):
    platform = hardware + '-' + backend + '-' + data_type
    assert platform in supported_list
    with open('{}-batch{}.txt'.format(platform, batch_size), 'a') as f:
        for model_name in model_list:
            results = get_model_profile(
                model=model_name,
                input_size=input_size,
                input_channel=3,
                batch_size=batch_size,
                hardware=hardware,
                backend=backend,
                data_type=data_type,
                force_test=force_test,
            )
            f.write('model: {}, params: {:.3f}M, FLOPs: {:.3f}M, latency: {:.3f}ms \n'.format(
                model_name, results['total_param'], results['flops'], results['latency']))
            logger.info('model: {}, params: {:.3f}M, FLOPs: {:.3f}M, latency: {:.3f}ms \n'.format(
                model_name, results['total_param'], results['flops'], results['latency']))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model Analysis')
    parser.add_argument('--input-size', default=224, type=int, nargs='+', help='model input size')
    parser.add_argument('--batch-size', default=64, type=int, help='input batch size')
    parser.add_argument('--hardware', default='cpu', type=str, help='hardware type, e.g. [cpu, T4, P4, 3559A, 3519A ]')
    parser.add_argument('--backend', default='nart', type=str, help='backend type, e.g. [nart, ppl2, cuda10.0-trt5.0, \
                            cuda10.0-trt7.0, cuda11.0-trt7.1, cuda11.0-nart, hisvp-nnie11, hisvp-nnie12]')
    parser.add_argument('--data-type', default='fp32', type=str, help='[fp32, fp16, int8]')
    parser.add_argument('--force-test', action='store_true', default=False, help='force test without querying database')
    args = parser.parse_args()

    model_list = list(SPRING_MODELS_REGISTRY.query())

    resnet_list = ['resnet18c_x0_125', 'resnet18c_x0_25', 'resnet18c_x0_5', 'googlenet']
    regnet_list = [_model for _model in model_list if 'regnetx' in _model]
    bignas_list = [_model for _model in model_list if 'bignas' in _model]
    oneshot_list = [_model for _model in model_list if 'oneshot' in _model]
    crnas_list = [_model for _model in model_list if 'crnas' in _model]
    mobilenet_list = [_model for _model in model_list if 'mobilenet' in _model]

    test_model_list = resnet_list + regnet_list + bignas_list + oneshot_list + crnas_list + mobilenet_list

    test_part(model_list=test_model_list, input_size=args.input_size, hardware=args.hardware,
              backend=args.backend, data_type=args.data_type, batch_size=args.batch_size,
              force_test=args.force_test)
