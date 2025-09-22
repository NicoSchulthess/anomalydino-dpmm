import math
import torch


def positionalencoding2d(D, H, W):
    """
    :param D: dimension of the model
    :param H: H of the positions
    :param W: W of the positions
    :return: DxHxW position matrix
    """
    if D % 4 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with odd dimension (got dim={:d})".format(D))
    P = torch.zeros(D, H, W)
    # Each dimension use half of D
    D = D // 2
    div_term = torch.exp(torch.arange(0.0, D, 2) * -(math.log(1e4) / D))
    pos_w = torch.arange(0.0, W).unsqueeze(1)
    pos_h = torch.arange(0.0, H).unsqueeze(1)
    P[0:D:2, :, :] = torch.sin(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[1:D:2, :, :] = torch.cos(pos_w * div_term).transpose(0, 1).unsqueeze(1).repeat(1, H, 1)
    P[D::2,  :, :] = torch.sin(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    P[D+1::2,:, :] = torch.cos(pos_h * div_term).transpose(0, 1).unsqueeze(2).repeat(1, 1, W)
    return P


def add_position_encoding(features, height, width, batch_size, device):
    xx, yy = torch.meshgrid([
        torch.linspace(-1, 1, width, device=device),
        torch.linspace(-1, 1, height, device=device),
    ], indexing="xy")
    position_encoding = torch.stack([xx, yy], dim=-1)
    position_encoding = position_encoding.reshape(1, height * width, 2)
    position_encoding = position_encoding.repeat(batch_size, 1, 1)
    features = torch.cat([features, position_encoding], dim=2)
    return features


def get_positional_encoding(batch_size, dimension, height, width, device):
    positional_encoding = positionalencoding2d(dimension, height, width)
    positional_encoding = positional_encoding.unsqueeze(0).repeat(batch_size, 1, 1, 1)          # NCHW
    positional_encoding = positional_encoding.permute(0, 2, 3, 1)                               # NHWC
    positional_encoding = positional_encoding.reshape(batch_size, height * width, dimension)    # NSD
    positional_encoding = positional_encoding.to(device)
    return positional_encoding
