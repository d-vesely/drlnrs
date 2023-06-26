import torch


def get_huber_loss(td_error, k=1.0):
    huber_loss = torch.where(
        td_error.abs() <= k,
        0.5 * td_error.pow(2),
        k * (td_error.abs() - 0.5 * k)
    )
    return huber_loss
