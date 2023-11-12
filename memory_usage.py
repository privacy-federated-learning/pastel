import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import sys
import torch.nn.functional as F
from vgg import VGG

class SizeEstimator(object):

    def __init__(self, model, model_name, dataset, gnl, anl, loader, bits=32):
        '''
        Estimates the size of PyTorch models in memory
        for a given input size
        '''
        self.model = model
        self.model_name = model_name
        self.bits = bits
        self.dataset = dataset
        self.loader = loader
        self.input_size = loader.dataset[0][0].shape
        self.gnl = gnl
        self.anl = anl
        return

    def get_parameter_sizes(self):
        '''Get sizes of all parameters in `model`'''
        mods = list(self.model.modules())
        sizes = []

        for i in range(1, len(mods)):
            m = mods[i]
            p = list(m.parameters())
            for j in range(len(p)):
                sizes.append(np.array(p[j].size()))

        self.param_sizes = sizes
        return

    def get_output_sizes(self):
        '''Run sample input through each layer to get output sizes'''

        def delete_multiple_element(list_object, indices):
            indices = sorted(indices, reverse=True)
            for idx in indices:
                if idx < len(list_object):
                    list_object.pop(idx)

        for _, (images, labels) in enumerate(self.loader):
            input_ = images.to('cuda')
            break

        mods = list(self.model.modules())
        out_sizes = []

        to_remove = []
        for i in range(0, len(mods)):
            m = mods[i]

            if self.model_name == 'resnet' and hasattr(m, 'adv'):
                to_remove.append(i)


        delete_multiple_element(mods, to_remove)

        for i in range(0, len(mods)):
            m = mods[i]
            if not hasattr(m, "in_features") and not hasattr(m, "weight") and not hasattr(m, "kernel_size"):
                continue

            if self.dataset == 'speech_commands' :

                if self.anl:
                    m_index = 39
                elif self.gnl:
                    m_index = len(mods) - 2
                else:
                    m_index = len(mods) -1
                if  i == m_index:
                    input_ = F.avg_pool1d(input_, input_.shape[-1])
                    input_ = input_.permute(0, 2, 1)

            elif self.model_name in ['vgg'] and i ==(len(mods) - (2 if self.gnl else 1)):
                input_ = input_.view(input_.size(0), -1)

            elif self.model_name == 'resnet':

                if ((26<=i<32) or (48<=i<54)):
                    continue

                elif i ==(len(mods) - (1 if self.gnl else 1)):

                    input_ = input_.view(input_.size(0), -1)

            out = m(input_)
            out_sizes.append(np.array(out.size()))


            if self.dataset == 'speech_commands' :
                if self.anl and i<39 and hasattr(m, 'running_var'):
                    out_sizes.append(np.array(out.size()))
                elif self.gnl and i==38:
                    out_sizes.append(np.array(out.size()))

            elif self.model_name in ['vgg', 'resnet']:
                if self.anl and hasattr(m, 'running_var'):
                    out_sizes.append(np.array(out.size()))
                elif self.gnl and (i ==(len(mods) - (2 if self.gnl and self.model_name =='vgg' else 1))):
                    out_sizes.append(np.array(out.size()))

            else:
                if self.anl and hasattr(m, 'running_var'):
                    out_sizes.append(np.array(out.size()))
                elif self.gnl and i==(len(mods)-2):
                    out_sizes.append(np.array(out.size()))

            input_ = out

        self.out_sizes = out_sizes
        return

    def calc_param_bits(self):
        '''Calculate total number of bits to store `model` parameters'''
        total_bits = 0
        for i in range(len(self.param_sizes)):
            s = self.param_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        self.param_bits = total_bits
        return


    def calc_param_size(self):
        '''Calculate total number of bits to store `model` parameters'''

        parameters_memory_usage = 0
        mods = list(self.model.modules())

        aggregated_layers = 0
        for m in mods:
            if not hasattr(m, "in_features") and not hasattr(m, "weight") and not hasattr(m, "kernel_size")\
                    and not hasattr(m, "adv") and not hasattr(m, "stddev"):
                continue
            p = list(m.parameters())


            aggregated_layers += 1

            for y in p:
                parameters_memory_usage += y.element_size() * y.nelement()

        self.param_bytes = parameters_memory_usage

        return

    def calc_forward_backward_bits(self):
        '''Calculate bits to store forward and backward pass'''
        total_bits = 0
        for i in range(len(self.out_sizes)):
            s = self.out_sizes[i]
            bits = np.prod(np.array(s)) * self.bits
            total_bits += bits
        # multiply by 2 for both forward AND backward
        self.forward_backward_bits = (total_bits * 2)
        return

    def calc_input_bits(self):
        '''Calculate bits to store input'''
        self.input_bits = np.prod(np.array(self.input_size)) * self.bits
        return

    def estimate_size(self):
        '''Estimate model size in memory in megabytes and bits'''
        self.get_parameter_sizes()
        self.get_output_sizes()
        self.calc_param_bits()
        self.calc_param_size()
        self.calc_forward_backward_bits()
        self.calc_input_bits()
        total = self.param_bits + self.forward_backward_bits + self.input_bits

        total_megabytes = (total / 8) / (1024 ** 2)
        return total_megabytes, total



def get_size(obj, seen=None):
    """Recursively finds size of objects"""
    size = sys.getsizeof(obj)
    if seen is None:
        seen = set()
    obj_id = id(obj)
    if obj_id in seen:
        return 0
    # Important mark as seen *before* entering recursion to gracefully handle
    # self-referential objects
    seen.add(obj_id)
    if isinstance(obj, dict):
        size += sum([get_size(v, seen) for v in obj.values()])
        size += sum([get_size(k, seen) for k in obj.keys()])
    elif hasattr(obj, '__dict__'):
        size += get_size(obj.__dict__, seen)
    elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
        size += sum([get_size(i, seen) for i in obj])
    return size