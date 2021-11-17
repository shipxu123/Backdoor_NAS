"""
Models in Spring aim to provide hardware-related latency-aware models.

Example1:
>>> from prototype.spring.models import SPRING_MODELS_REGISTRY
### QUERY SUPPORTED MODELS
>>> SPRING_MODELS_REGISTRY.query()
### TYPE1 TO BUILD MODELS
>>> cls_model = SPRING_MODELS_REGISTRY.get('resnet18')(task='classification', **kwargs)
>>> det_model = SPRING_MODELS_REGISTRY.get('resnet18')(task='detection', **kwargs)
### TYPE2 TO BUILD MODELS
>>> models = SPRING_MODELS_REGISTRY.build(config)  # config contains: type and kwargs

Example2:
>>> from prototype.spring.models import SEARCH_ENGINE
### FIND WANTED MODELS
>>> candidates = SEARCH_ENGINE.search(hardware='cuda11.0-trt7.1-int8-P4', target_latency=10, batch=8)
### VIEW KNOWN MODELS
>>> details = SEARCH_ENGINE.view_details(model_name='resnet18c_x0_125')
"""

# from .resnet import (  # noqa: F401
#     resnet18c_x0_125, resnet18c_x0_25, resnet18c_x0_5,
#     resnet18, resnet34, resnet50, resnet101, resnet152,
#     resnext50_32x4d, resnext101_32x8d, wide_resnet50_2, wide_resnet101_2,
#     resnet5400, resnet_custom
# )
# from .regnet import (  # noqa: F401
#     regnetx_200m, regnetx_400m, regnetx_600m, regnetx_800m,
#     regnetx_1600m, regnetx_3200m, regnetx_4000m, regnetx_6400m,
#     regnety_200m, regnety_400m, regnety_600m, regnety_800m,
#     regnety_1600m, regnety_3200m, regnety_4000m, regnety_6400m,
# )
# from .bignas_resnet_basicblock import (  # noqa: F401
#     bignas_resnet18_9M, bignas_resnet18_37M, bignas_resnet18_50M,
#     bignas_resnet18_49M, bignas_resnet18_65M,
#     bignas_resnet18_107M, bignas_resnet18_125M, bignas_resnet18_150M,
#     bignas_resnet18_312M, bignas_resnet18_403M, bignas_resnet18_492M,
#     bignas_resnet18_1555M,
#     bignas_det_resnet18_1930M
# )
# from .bignas_resnet_bottleneck import (  # noqa: F401
#     bignas_resnet50_2954M, bignas_resnet50_3145M, bignas_resnet50_3811M
# )
# from .dmcp_resnet import (  # noqa: F401
#     dmcp_resnet18_47M, dmna_resnet18_1800M
# )
# from .shufflenet_v2 import (  # noqa: F401
#     shufflenet_v2_x0_5, shufflenet_v2_x1_0, shufflenet_v2_x1_5, shufflenet_v2_x2_0,
#     shufflenet_v2_scale
# )
# from .mobilenet_v2 import (  # noqa: F401
#     mobilenet_v2_x0_5, mobilenet_v2_x0_75, mobilenet_v2_x1_0, mobilenet_v2_x1_4
# )
# from .oneshot_supcell import (  # noqa: F401
#     oneshot_supcell_9M, oneshot_supcell_27M, oneshot_supcell_37M, oneshot_supcell_55M,
#     oneshot_supcell_70M, oneshot_supcell_91M, oneshot_supcell_96M, oneshot_supcell_113M,
#     oneshot_supcell_168M, oneshot_supcell_304M, oneshot_supcell_1710M, oneshot_supcell_3072M
# )
# from .crnas_resnet import (  # noqa: F401
#     crnas_resnet18c, crnas_resnet50c, crnas_resnet101c
# )
# from .efficientnet import (  # noqa: F401
#     efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
#     efficientnet_b4, efficientnet_b5, efficientnet_b6, efficientnet_b7
# )
# from .mobilenet_v3 import (  # noqa: F401
#     mobilenet_v3_small_x0_35, mobilenet_v3_small_x0_5, mobilenet_v3_small_x0_75,
#     mobilenet_v3_small_x1_0, mobilenet_v3_small_x1_4,
#     mobilenet_v3_large_x0_35, mobilenet_v3_large_x0_5, mobilenet_v3_large_x0_75,
#     mobilenet_v3_large_x1_0, mobilenet_v3_large_x1_4
# )
# from .googlenet import googlenet  # noqa: F401
# from .vision_transformer import (  # noqa: F401
#     vit_base_patch32_224, vit_base_patch16_224, vit_large_patch16_224, vit_huge_patch14_224,
#     deit_tiny_patch16_224, deit_small_patch16_224, deit_base_patch16_224
# )

from prototype.spring.models.tools import SEARCH_ENGINE #, MODEL_PROFILER  # noqa: F401

from prototype.spring.models.utils.register import Registry


SPRING_MODELS_REGISTRY = Registry()

imported_vars = list(globals().items())

for var_name, var in imported_vars:
    if callable(var):
        SPRING_MODELS_REGISTRY.register(var_name, var)
