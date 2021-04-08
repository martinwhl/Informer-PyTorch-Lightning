import torch


def triangular_causal_mask(B, L):
    mask_shape = [B, 1, L, L]
    with torch.no_grad():
        mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1)
    return mask


def prob_mask(B, H, L, index, scores):
    mask = torch.ones(L, scores.shape[-1], dtype=torch.bool).triu(1)
    mask_ex = mask[None, None, :].expand(B, H, L, scores.shape[-1])
    indicator = mask_ex[torch.arange(B)[:, None, None],
                        torch.arange(H)[None, :, None],
                        index, :]
    mask = indicator.view(scores.shape)
    return mask
