import torch

def purchase_collate_fn(batch):
    images = []
    targets = []

    for i in batch:
        images.append(i[0])
        targets.append(i[1])
    print("OK")
    return torch.cat(images), torch.cat(targets)