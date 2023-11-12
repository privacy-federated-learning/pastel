import torch

# Gradient Compression
class TensorPruner:
    def __init__(self, zip_percent):
        self.thresh_hold = 0.
        self.zip_percent = zip_percent

    def update_thresh_hold(self, tensor):
        tensor_copy = tensor.clone().detach()
        tensor_copy = torch.abs(tensor_copy)
        survivial_values = torch.topk(tensor_copy.reshape(1, -1),
                                      int(tensor_copy.reshape(1, -1).shape[1] * self.zip_percent))
        self.thresh_hold = survivial_values[0][0][-1]

    def prune_tensor(self, tensor):
        # whether the tensor to process is on cuda devices
        background_tensor = torch.zeros(tensor.shape).to(torch.float)
        if 'cuda' in str(tensor.device):
            background_tensor = background_tensor.cuda()
        # print("background_tensor", background_tensor)
        tensor = torch.where(abs(tensor) > self.thresh_hold, tensor, background_tensor)
        # print("tensor:", tensor)
        return tensor

