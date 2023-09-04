import numpy as np
import torch


_GPU_ID = 0
_USE_GPU = False
_DEVICE = None


def set_gpu_mode(mode, gpu_id=0):
    global _GPU_ID
    global _USE_GPU
    global _DEVICE
    _GPU_ID = gpu_id
    _USE_GPU = mode
    _DEVICE = torch.device(("cuda:" + str(_GPU_ID)) if _USE_GPU else "cpu")
    torch.set_default_tensor_type(
        torch.cuda.FloatTensor if _USE_GPU else torch.FloatTensor
    )


def get_device():
    global _DEVICE
    return _DEVICE


def to_torch(x, dtype=None, device=None):
    if device is None:
        device = get_device()
    return torch.as_tensor(x, dtype=dtype, device=device)


def to_np(x):
    return x.detach().cpu().numpy()


def soft_update(target, source, tau):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)


def hard_update(target, source):
    for target_param, param in zip(target.parameters(), source.parameters()):
        target_param.data.copy_(param.data)


class FreezeParameters:
    def __init__(self, params):
        self.params = params
        self.param_states = [p.requires_grad for p in self.params]

    def __enter__(self):
        for param in self.params:
            param.requires_grad = False

    def __exit__(self, exc_type, exc_val, exc_tb):
        for i, param in enumerate(self.params):
            param.requires_grad = self.param_states[i]


def lambda_return(rewards, values, discounts, bootstrap, lambda_=0.95):
    next_values = torch.cat([values[1:], bootstrap[None]], 0)
    inputs = rewards + discounts * next_values * (1 - lambda_)
    last = bootstrap
    outputs = []
    for t in reversed(range(len(inputs))):
        last = inputs[t] + discounts[t] * lambda_ * last
        outputs.append(last)
    outputs = list(reversed(outputs))
    returns = torch.stack(outputs, 0)
    return returns


def preprocess(obs):
    # Preprocess a batch of observations
    ndims = len(obs.shape)
    assert ndims == 2 or ndims == 4, "preprocess accepts a batch of observations"
    if ndims == 4:
        obs = ((obs.astype(np.float32) / 255) * 2) - 1.0
    return obs


def postprocess(obs):
    # Postprocess a batch of observations
    ndims = len(obs.shape)
    assert ndims == 2 or ndims == 4, "postprocess accepts a batch of observations"
    if ndims == 4:
        obs = np.floor((obs + 1.0) / 2 * 255).clip(0, 255).astype(np.uint8)
    return obs
