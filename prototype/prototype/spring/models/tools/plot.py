import matplotlib.pyplot as plt
import random
import matplotlib.colors as mcolors
from adjustText import adjust_text
import math
import os

from .search import series_dict, reform_dict2list


model_types = ['resnet', 'regnet', 'bignas', 'dmcp', 'shufflenet_v2',
               'mobilenet_v2', 'oneshot_supcell', 'crnas_resnet', 'efficientnet', 'mobilenet_v3']


all_hardwares = [
    'cpu-ppl2-fp32',
    'hisvp-nnie11-int8',
    'cuda11.0-trt7.1-int8-P4',
    'acl-ascend310-fp16',
    'halnn0.4-stpu-int8'
]
all_batch_size = [1, 8, 64]
model_performances_list = []
for curr_dict in series_dict:
    model_performances_list.extend(reform_dict2list(curr_dict))


def _decouple_list(model_list):
    name, acc, latency = [], [], []
    for model in model_list:
        name.append(model['model_name'])
        acc.append(model['accuracy'])
        latency.append(model['latency'])
    return name, acc, latency


def _filter_hardware_list(performances_list, hardware, batch_size):
    filterd_list = []
    for tuples in performances_list:
        if tuples['hardware'].replace(' ', '').startswith(hardware):
            if tuples['batch'] == batch_size:
                if not ((tuples['accuracy'] is None) or (tuples['latency'] is None)):
                    if tuples['accuracy'] > 10:  # filter out bad models
                        filterd_list.append(tuples)
    return filterd_list


def _sort_by_latency(latency, acc, names):
    def take_first(lis):
        return lis[0]

    pairs = list(zip(latency, acc, names))
    pairs.sort(key=take_first)
    return zip(*pairs)


def _matching_model_type(name):
    for n in model_types:
        if name.startswith(n):
            return n
    raise ValueError('Unknown model type: {}'.format(name))


def _plot(latency, acc, names, plot_type='pareto'):
    # sort models by latency
    latency, acc, names = _sort_by_latency(latency, acc, names)

    if plot_type == 'pareto':
        # sample dot colors
        if len(names) > len(list(mcolors.TABLEAU_COLORS)):
            candidate_colors = mcolors.CSS4_COLORS
        else:
            candidate_colors = mcolors.TABLEAU_COLORS
        colors = random.sample(list(candidate_colors), len(names))
        # dot plot
        plt.scatter(latency, acc, marker='o', linewidth=8, c=colors)
        # annotations
        texts = []
        for i in range(len(names)):
            texts.append(plt.text(latency[i], acc[i], names[i], fontsize=14))
        # resolve overlaped annotations
        adjust_text(texts, va="top", ha="left")

    elif plot_type == 'normal':
        # dot plots with label
        label = _matching_model_type(names[0])
        # adjust x-axis
        latency = [math.log(l, 10) for l in latency]  # noqa
        # plot dot chart
        handle = plt.scatter(latency, acc, marker='o', linewidth=4, label=label)
        # annotations
        texts = []
        for i in range(len(names)):
            texts.append(plt.text(latency[i], acc[i], names[i], fontsize=14))
        return texts, handle
    else:
        raise ValueError


def _sample_pareto_models(all_latency, all_acc, all_name):
    pareto_models = []
    for l, a, n in zip(all_latency, all_acc, all_name):  # noqa
        is_pareto = True
        for i in range(len(all_latency)):
            if l > all_latency[i] - 1e-5 and a < all_acc[i] - 1e-5:
                is_pareto = False
                break
        if is_pareto:
            pareto_models.append((l, a, n))
    latency, acc, name = zip(*pareto_models)
    return latency, acc, name


def _adjust_pareto_texts(texts, pareto_names):
    for text in texts:
        if text.get_text() in pareto_names:
            text.set_color('red')


def plot(hardware, batch_size, pareto, save_name):

    assert hardware in all_hardwares
    assert batch_size in all_batch_size

    fig = plt.gcf()
    fig.set_size_inches(30.5, 15.5)

    if pareto:
        performances_list = _filter_hardware_list(model_performances_list, hardware, batch_size)
        name, acc, latency = _decouple_list(performances_list)
        latency, acc, name = _sample_pareto_models(latency, acc, name)
        texts = _plot(latency, acc, name, plot_type='pareto')
        plt.xlabel('Latency(ms)')
    else:
        handles = []
        texts = []
        all_latency, all_acc, all_name = [], [], []

        for single_model in series_dict:
            reformed = reform_dict2list(single_model)
            filtered_model = _filter_hardware_list(reformed, hardware, batch_size)
            name, acc, latency = _decouple_list(filtered_model)
            all_latency.extend(latency)
            all_acc.extend(acc)
            all_name.extend(name)
            if len(name) > 0:
                text, handle = _plot(latency, acc, name, plot_type='normal')
                handles.append(handle)
                texts.extend(text)
        adjust_text(texts, va="top", ha="left")

        # draw pareto lines
        latency, acc, name = _sample_pareto_models(all_latency, all_acc, all_name)
        latency, acc, name = _sort_by_latency(latency, acc, name)
        _adjust_pareto_texts(texts, name)
        latency = [math.log(l, 10) for l in latency]  # noqa
        handle, = plt.plot(latency, acc, marker=',', linestyle='--',
                           linewidth=1, label='pareto', color='red')
        handles.append(handle)
        plt.legend(handles=handles, loc='lower right', fontsize=24)
        plt.xlabel('Log-Latency(ms)')

    plt.ylabel('Top-1 Accuracy(%)')
    plt.savefig(save_name + '.png', dpi=200,
                bbox_inches='tight', pad_inches=0.3)
    plt.clf()


if __name__ == '__main__':
    folder_name = 'modelx_all_plots'
    if not os.path.exists(folder_name):
        os.makedirs(folder_name)

    for hardware in all_hardwares:
        curr_batch_list = [1, 8] if hardware == 'halnn0.4-stpu-int8' else all_batch_size
        for batch in curr_batch_list:
            save_name = os.path.join(folder_name, '{}_batch{}'.format(hardware, batch))
            plot(hardware=hardware, batch_size=batch, pareto=False, save_name=save_name)
