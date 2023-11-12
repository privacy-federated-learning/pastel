import copy
from torch.nn.functional import normalize
import torch


def pastel(target_model, layer_type='bn', layers=['bn']):
    model_dict = copy.deepcopy(target_model)
    for key in target_model:
        for layer in layers:
            if layer in key:
                if layer_type == 'batchnorm':
                    model_dict[key] = normalize(abs(torch.randn(list(target_model[key].shape))), p=1.0, dim=0)
                else:
                    model_dict[key] = normalize(torch.randn(list(target_model[key].shape)), dim=0)


                print('Key', key, 'altered')
    return model_dict


def aggregation_pastel(local_weights, layers):
    w_avg = copy.deepcopy(local_weights[0])
    for key in w_avg.keys():
        for i in range(1, len(local_weights)):
            w_avg[key] += local_weights[i][key]
        w_avg[key] = torch.div(w_avg[key], len(local_weights))
        for layer in layers:
            if layer not in key:
                for idx, _ in enumerate(local_weights):
                    local_weights[idx][key] = w_avg[key]
            else:
                print(key)
    return w_avg, local_weights