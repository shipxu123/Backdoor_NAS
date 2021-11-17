import os
import time
import onnx
import torch
import argparse

from prototype.spring.utils.log_helper import default_logger as logger


class ModelProfile(object):
    def __init__(self, log_result=True):
        self.M = 1e6
        self.log_result = log_result

    def count_params(self, model):
        total_param = sum(p.numel() for p in model.parameters())
        conv_param = 0
        fc_param = 0
        others_param = 0
        for name, m in model.named_modules():
            # skip non-leaf modules
            if len(list(m.children())) > 0:
                continue
            # current params
            num = sum(p.numel() for p in m.parameters())
            if isinstance(m, torch.nn.Conv2d):
                conv_param += num
            elif isinstance(m, torch.nn.Linear):
                fc_param += num
            else:
                others_param += num

        total_param /= self.M
        conv_param /= self.M
        fc_param /= self.M
        others_param /= self.M

        if self.log_result:
            logger.info('Profiling information of model on Params.\n \
                Total param: {:.3f}M, conv: {:.3f}M, fc: {:.3f}M, others: {:.3f}M'.format(
                total_param, conv_param , fc_param, others_param))

        return total_param, conv_param, fc_param, others_param

    @torch.no_grad()
    def count_flops(self, model, input_size=(1, 3, 224, 224)):
        """
        args:
            input_size: for example (1, 3, 224, 224)
        """
        flops_dict = {}

        def make_conv2d_hook(name):

            def conv2d_hook(m, input):
                n, _, h, w = input[0].size(0), input[0].size(
                    1), input[0].size(2), input[0].size(3)
                flops = n * h * w * m.in_channels * m.out_channels * m.kernel_size[0] * m.kernel_size[1] \
                    / m.stride[1] / m.stride[1] / m.groups
                flops_dict[name] = int(flops)

            return conv2d_hook

        hooks = []
        for name, m in model.named_modules():
            if isinstance(m, torch.nn.Conv2d):
                h = m.register_forward_pre_hook(make_conv2d_hook(name))
                hooks.append(h)

        input = torch.zeros(*input_size)

        model.eval()
        _ = model(input)
        model.train()
        total_flops = 0
        for k, v in flops_dict.items():
            # logger.info('module {}: {}'.format(k, v))
            total_flops += v

        if self.log_result:
            logger.info('Profiling information of model on FLOPs.\n \
                Total FLOPS: {:.3f}M'.format(total_flops / self.M))

        for h in hooks:
            h.remove()

        return total_flops / self.M

    def test_latency(self,
                     model,
                     input_size=(1, 3, 224, 224),
                     save_path='./',
                     hardware='cpu',
                     backend='nart',
                     batch_size=64,
                     data_type='fp32',
                     graph_name='',
                     force_test=False):
        """ Convert model into ONNX, then test hardware-related latency.

        args:
            model: pytorch model "nn.Module"
            input_size: tuple of 4 int
            save_path: path to save ONNX model
            hardware: hardware type, e.g. ['cpu', 'T4', 'P4', '3559A', '3519A']
            backend: backend type, e.g.
                ['nart', 'ppl2', 'cuda10.0-trt5.0', 'cuda10.0-trt7.0', 'cuda11.0-trt7.1',
                 'cuda11.0-nart', 'hisvp-nnie11', 'hisvp-nnie12']
            batch_size: int
            data_type: ['fp32', 'fp16', 'int8']
            graph_name: tag for this model
            force_test: force to test latency no matter whether this model has beed tested

        """
        # unImp
        return 0

#        logger.info('Converting model into ONNX type...')
#        if get_rank() == 0:
#            save_prefix = os.path.join(save_path, 'ONNX/model')
#            if not os.path.exists(save_prefix):
#                os.makedirs(save_prefix)
#            with pytorch.convert_mode():
#                pytorch.export_onnx(
#                    model, [input_size],
#                    filename=save_prefix,
#                    input_names=['data'],
#                    output_names=['output'],
#                    verbose=False,
#                    cloze=False
#                )
#        # link.barrier()
#        logger.info('Merging BN of the ONNX model...')
#        onnx_model = onnx.load(save_prefix + '.onnx')
#        graph = OnnxDecoder().decode(onnx_model)
#        graph.update_tensor_shape()
#        graph.update_topology()
#        ConvFuser().run(graph)
#        GemmFuser().run(graph)
#        DeadCodeElimination().run(graph)
#        graph.update_topology()
#
#        onnx_model = Model.make_model(graph)
#        onnx_model = onnx_model.dump_to_onnx()
#        onnx_file_path = save_prefix + '_merged.onnx'
#        onnx.save(onnx_model, onnx_file_path)
#
#        logger.info('Test latency using ONNX model...')
#        latency_client = Latency()
#        if get_rank() == 0:
#            test_status_success = False
#            while not test_status_success:
#                ret = latency_client.call(
#                    hardware_name=hardware,
#                    backend_name=backend,
#                    data_type=data_type,
#                    batch_size=batch_size,
#                    onnx_file=onnx_file_path,
#                    graph_name=graph_name,
#                    force_test=force_test,
#                )
#                if ret is None:
#                    test_status_success = False
#                else:
#                    test_status_success = ret['ret']['status'] == 'success'
#                logger.info('Whether test succeed: {}'.format(test_status_success))
#                logger.info(ret)
#                if not test_status_success:
#                    time.sleep(10)
#
#            if self.log_result:
#                logger.info('Profiling information of model on Latency.\n \
#                    In the platform of {}-{}-{}-{}, the latency is {} ms.'.format(
#                    hardware, backend, data_type, batch_size, ret['cost_time']))
#            return ret['cost_time']


MODEL_PROFILER = ModelProfile()


def get_model_profile(model, input_size=224, input_channel: int = 3, batch_size: int = 1, hardware: str = 'T4',
                      backend: str = 'cuda11.0-trt7.1', data_type: str = 'int8', force_test: bool = False):
    '''
        args:
            model: model name, e.g. "resnet18c_x0_125"
                or pytorch model "nn.Module"
            input_size: int or tuple of two integers that height and width respectively,
                e.g. input_size=224, input_size=(224, 224)
            input_channel: int
            batch_size: int
            hardware: hardware type, e.g.
                ['cpu', 'T4', 'P4', '3559A', '3519A']
            backend: backend type, e.g.
                ['nart', 'ppl2', 'cuda10.0-trt5.0', 'cuda10.0-trt7.0', 'cuda11.0-trt7.1',
                 'cuda11.0-nart', 'hisvp-nnie11', 'hisvp-nnie12']
            data_type: ['fp32', 'fp16', 'int8']
            force_test: force to test latency no matter whether this model has beed tested

    '''
    from prototype.spring.models import SPRING_MODELS_REGISTRY
    MODEL_PROFILER = ModelProfile(log_result=False)

    if isinstance(model, str):
        try:
            graph_name = 'spring.models.' + model
            model = SPRING_MODELS_REGISTRY.get(model)(task='classification')
        except NotImplementedError:
            print('model "{}" not found in SPRING_MODELS_REGISTRY'.format(model))
    else:
        assert isinstance(model, torch.nn.Module), \
            'the argument must be model name or model instance!, but get {}'.format(type(model))
        graph_name = ''

    if isinstance(input_size, tuple) or isinstance(input_size, list):
        input_size = (1, input_channel, input_size[0], input_size[1])
    elif isinstance(input_size, int):
        input_size = (1, input_channel, input_size, input_size)
    else:
        raise ValueError('Expected input_size to be tuple/list or int, but got {} with type {}'
                         .format(input_size, type(input_size)))

    total_param, conv_param, fc_param, others_param = MODEL_PROFILER.count_params(model)
    flops = MODEL_PROFILER.count_flops(model, input_size=input_size)
    #latency = MODEL_PROFILER.test_latency(model, input_size=input_size, hardware=hardware,
                                          #backend=backend, batch_size=batch_size, data_type=data_type,
                                          #graph_name=graph_name, force_test=force_test)

    logger.info('Profiling information of model on Params.\n \
        Total param: {:.3f}M, conv: {:.3f}M, fc: {:.3f}M, others: {:.3f}M'.format(
        total_param, conv_param, fc_param, others_param))
    logger.info('Profiling information of model on FLOPs.\n \
        Total FLOPS: {:.3f}M'.format(flops))
    #logger.info('Profiling information of model on Latency.\n \
    #    In the platform of {}-{}-{}-b{}, the latency is {} ms.'.format(
    #    hardware, backend, data_type, batch_size, latency))

    return {'total_param': total_param, 'conv_param': conv_param, 'other_param': others_param,
            'flops': flops}#, 'latency': latency}


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Model profiling')
    parser.add_argument('--model', type=str, help='model name', required=True)
    parser.add_argument('--input-size', default=224, type=int, nargs='+', help='model input size')
    parser.add_argument('--input-channel', default=3, type=int, help='model input channel')
    parser.add_argument('--batch-size', default=64, type=int, help='input batch size')
    parser.add_argument('--hardware', default='cpu', type=str, help='hardware type, e.g. [cpu, T4, P4, 3559A, 3519A ]')
    parser.add_argument('--backend', default='nart', type=str, help='backend type, e.g. [nart, ppl2, cuda10.0-trt5.0, \
                            cuda10.0-trt7.0, cuda11.0-trt7.1, cuda11.0-nart, hisvp-nnie11, hisvp-nnie12]')
    parser.add_argument('--data-type', default='fp32', type=str, help='[fp32, fp16, int8]')
    parser.add_argument('--force-test', action='store_true', default=False, help='force test without querying database')
    args = parser.parse_args()

    get_model_profile(args.model, input_size=args.input_size, input_channel=args.input_channel,
                      batch_size=args.batch_size, hardware=args.hardware, backend=args.backend,
                      data_type=args.data_type, force_test=args.force_test)
