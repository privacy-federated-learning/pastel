import os
import numpy as np
import scipy.io
import torch

from data.motionsense.dataset_utils import GeneralPurposeDataset, load_AF_stats_item


def load_motion_sense_item(fileid, path, ext):
    file = fileid + ext
    user_id = fileid.split("/")[-1].split('_')[0]
    class_id = fileid.split("/")[-1].split('_')[-1]

    data = scipy.io.loadmat(file)['x']

    data = np.squeeze(data)
    return data, class_id, fileid.split("/")


def create_dataset(root, load_item_fn=load_AF_stats_item, ext_audio=".f0", **kwargs):
    return GeneralPurposeDataset(root=root, load_item_fn=load_item_fn, ext_audio=ext_audio)


def motionsense_collate_fn(batch):
    images = []
    targets = []
    for i, j, k in batch:
        images.append(i)
        targets.append(int(j))

    return torch.tensor(images, dtype=torch.float), torch.tensor(targets, dtype=torch.long)