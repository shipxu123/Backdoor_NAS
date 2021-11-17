# Standard Library
import numpy as np
import copy
# Import from third library
import torch
# Import from prototype.spring
# import linklink as link


class FrozenBatchNorm2d(torch.nn.BatchNorm2d):
    def __init__(self, *args, **kwargs):
        super(FrozenBatchNorm2d, self).__init__(*args, **kwargs)
        self.training = False

    def train(self, mode=False):
        self.training = False
        for module in self.children():
            module.train(False)
        return self


_norm_cfg = {
    'solo_bn': torch.nn.BatchNorm2d,
    'freeze_bn': FrozenBatchNorm2d,
    # 'gn': torch.nn.GroupNorm,
}


#def simple_group_split(world_size, rank, num_groups):
#    groups = []
#    rank_list = np.split(np.arange(world_size), num_groups)
#    rank_list = [list(map(int, x)) for x in rank_list]
#    for i in range(num_groups):
#        groups.append(link.new_group(rank_list[i]))
#    group_size = world_size // num_groups
#    return groups[rank // group_size]


def build_norm_layer(config):
    """
    Build normalization layer according to configurations.

    solo_bn (original bn): torch.nn.BatchNorm2d
    sync_bn (synchronous bn): link.nn.SyncBatchNorm2d
    freeze_bn (frozen bn): torch.nn.BatchNorm2d with training type of False
    gn (group normalization): torch.nn.GroupNorm
    """
    assert isinstance(config, dict) and 'type' in config
    config = copy.deepcopy(config)
    norm_type = config.pop('type')
    config_kwargs = config.get('kwargs', {})

    if norm_type == 'sync_bn':
        group_size = config_kwargs.get('group_size', 1)
        var_mode = config_kwargs.get('var_mode', 'L2')
        bn_group = None
       
        del config_kwargs['group_size']
        config_kwargs['group'] = bn_group
        config_kwargs['var_mode'] = (
            link.syncbnVarMode_t.L1 if var_mode == 'L1' else link.syncbnVarMode_t.L2
        )

    def NormLayer(*args, **kwargs):
        return _norm_cfg[norm_type](*args, **kwargs, **config_kwargs)

    return NormLayer
