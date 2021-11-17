from collections import OrderedDict
from prototype.spring.utils.log_helper import default_logger as logger


def modify_state_dict(model, state_dict):
    new_state_dict = OrderedDict()
    keys = list(state_dict.keys())
    dist_type = True if 'module' in keys[0] else False
    fc_keys_list = []
    if dist_type:
        for key in keys:
            new_key = '.'.join(key.split('.')[1:])
            new_state_dict[new_key] = state_dict[key]
            if ('fc' in new_key) or ('classifier' in new_key):
                fc_keys_list.append(new_key)
                if 'weight' in new_key:
                    # dimension of fc layer in ckpt
                    fc_dim = state_dict[key].size(0)

    if model.task != 'classification' or model.num_classes != fc_dim:
        for key in fc_keys_list:
            _ = new_state_dict.pop(key)
            logger.info('Pop the extra key in state_dict: {}'.format(key))
    return new_state_dict
