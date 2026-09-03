import torch


def allocate_host_tensor(*size, pin_memory: bool = True, **kwargs) -> torch.Tensor:
    return torch.empty(*size, device="cpu", pin_memory=pin_memory, **kwargs)
