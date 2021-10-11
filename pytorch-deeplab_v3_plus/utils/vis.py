import torch

def get_edges(seg_map):
    edges = torch.zeros_like(seg_map) == 1
    edges[..., :-1, :] |= seg_map[..., :-1, :] != seg_map[..., 1:, :]
    edges[..., :, :-1] |= seg_map[..., :, :-1] != seg_map[..., :, 1:]
    edges[..., 1:, :]  |= seg_map[..., :-1, :] != seg_map[..., 1:, :]
    edges[..., :, 1:]  |= seg_map[..., :, :-1] != seg_map[..., :, 1:]
    return edges
