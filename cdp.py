import torch

from tensorflow_privacy.privacy.analysis.compute_noise_from_budget_lib import compute_noise

def gaussian_noise(data_shape, s, sigma, device=None):
    """
    Gaussian noise
    """
    return torch.normal(0, sigma * s, data_shape).to(device)


def update_cdp(global_model, train_dataset, args):

    clipped_grads = {name: torch.zeros_like(param) for name, param in global_model.named_parameters()}
    torch.nn.utils.clip_grad_norm_(global_model.parameters(), max_norm=args.clip)

    for name, param in global_model.named_parameters():
            sigma = compute_noise(len(train_dataset), args.batch_size, args.eps, args.E * args.tot_T, 0.00001, 0.1)

            clipped_grads[name] += gaussian_noise(clipped_grads[name].shape, args.clip, sigma,
                                                  device='cuda')

            # scale back
    for name, param in global_model.named_parameters():
        clipped_grads[name] /= (len(train_dataset) * 0.5)

    global_model.load_state_dict(clipped_grads)

    return global_model