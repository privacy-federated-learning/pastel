import torch

def celeba_collate_fn(batch):

    images = []
    targets = []
    for data in batch:
        target = int("".join(str(x) for x in [data[1].tolist()[y] for y in [20, 13, 15, 26, 19]]), 2)
        targets.append(target)
        images.append(data[0])

    return torch.stack(images), torch.tensor(targets, dtype=torch.long)


