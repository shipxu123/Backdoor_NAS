from prototype.spring.models.resnet import model_performances as resnet_performances
from prototype.spring.models.regnet import model_performances as regnet_performances
from prototype.spring.models.bignas_resnet_basicblock import model_performances as bignas_basic_performances
from prototype.spring.models.bignas_resnet_bottleneck import model_performances as bignas_bottle_performances
from prototype.spring.models.dmcp_resnet import model_performances as dmcp_resnet_performances
from prototype.spring.models.shufflenet_v2 import model_performances as shufflenet_performances
from prototype.spring.models.mobilenet_v2 import model_performances as mbv2_performances
from prototype.spring.models.oneshot_supcell import model_performances as oneshot_supcell_performances
from prototype.spring.models.crnas_resnet import model_performances as crnas_resnet_performances
from prototype.spring.models.efficientnet import model_performances as efficientnet_performances
from prototype.spring.models.mobilenet_v3 import model_performances as mbv3_performances


def reform_dict2list(dict):
    tmp_list = []
    for model_name in dict.keys():
        for item in dict[model_name]:
            item.update({'model_name': model_name})
            tmp_list.append(item)
    return tmp_list


class SearchModels(object):
    def __init__(self, model_performances_list=None):
        self.model_performances_list = model_performances_list
        hardware_list = []
        model_name_list = []
        batch_list = []
        input_size_list = []
        for info in self.model_performances_list:
            hardware_list.append(info['hardware'])
            model_name_list.append(info['model_name'])
            batch_list.append(info['batch'])
            input_size_list.append(info['input_size'])

        self.hardware_set = set(hardware_list)
        self.model_name_set = set(model_name_list)
        self.batch_set = set(batch_list)
        self.input_size_set = set(input_size_list)
        self.num_candidates = len(self.model_performances_list)

    def search(self, hardware, target_latency, batch=1, input_size=(3, 224, 224)):
        """
        Search for models that satisfy requirements.

        Arguments:
        - hardware (:obj:`str`): name of supported hardware
        - target_latency (:obj:`float`): target latency
        - batch (:obj:`int`): deployment batch, e.g. 1, 8, 64
        - input_size (:obj:`int`): deployment input size

        Returns:
            candidates: list of details of candidate models
        """
        candidates = []
        for model_info in self.model_performances_list:
            if (model_info['latency'] is not None) and (model_info['accuracy'] is not None):
                if model_info['hardware'] == hardware and model_info['latency'] < target_latency and \
                        model_info['batch'] == batch and model_info['input_size'] == input_size:
                    candidates.append(model_info)
                    print('Satisfied candidate: {}'.format(model_info))
        print({'name': "spring.models", 'action': 'search', 'hardware': hardware,
                        'target_latency': target_latency, 'batch': batch, 'input_size': input_size})
        return candidates

    def view_details(self, model_name):
        """
        View model details in different hardwares.

        Arguments:
        - model_name (:obj:`str`): model_name for view

        Returns:
            model_details: list of model information in different hardwares
        """
        assert model_name in self.model_name_set, 'models are not included.'
        model_details = []
        for model_info in self.model_performances_list:
            if model_info['model_name'] == model_name:
                model_details.append(model_info)
                print('Model performance: {}'.format(model_info))
        print({'name': "spring.models", 'action': 'view', 'model_name': model_name})
        return model_details


series_dict = [resnet_performances, regnet_performances, bignas_basic_performances, bignas_bottle_performances,
               dmcp_resnet_performances, shufflenet_performances, mbv2_performances, oneshot_supcell_performances,
               crnas_resnet_performances, efficientnet_performances, mbv3_performances]

model_performances_list = []
for curr_dict in series_dict:
    model_performances_list.extend(reform_dict2list(curr_dict))

SEARCH_ENGINE = SearchModels(model_performances_list)


if __name__ == '__main__':
    candidates = SEARCH_ENGINE.search(hardware='cuda11.0-trt7.1-int8-P4', target_latency=10, batch=8)
    model_details = SEARCH_ENGINE.view_details(model_name='resnet18c_x0_125')
