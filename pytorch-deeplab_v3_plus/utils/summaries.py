import os
import torch
import numpy as np
import scipy.ndimage
from torchvision.utils import make_grid
from tensorboardX import SummaryWriter
from dataloaders.utils import decode_seg_map_sequence
from utils import vis

class TensorboardSummary(object):
    def __init__(self, directory):
        self.directory = directory

    def create_summary(self):
        writer = SummaryWriter(log_dir=os.path.join(self.directory))
        return writer

    def visualize_image(self, writer, dataset, image, target, output, global_step, prefix=''):
        image = image[:9].clone().cpu()
        grid_image = make_grid(image.data, 3, normalize=True)
        writer.add_image(prefix + 'Image', grid_image, global_step)

        seg_map = torch.max(output[:9], 1)[1].detach().cpu()
        grid_image = make_grid(decode_seg_map_sequence(seg_map.numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image(prefix + 'Predicted label', grid_image, global_step)

        edges = vis.get_edges(seg_map)
        mx = image.max()
        for i in range(image.shape[0]):
            image[i, :, edges[i]] = mx
            # image[i, :, scipy.ndimage.binary_dilation(edges[i], iterations=5)] = mx
        grid_image = make_grid(image, 3, normalize=True)
        writer.add_image(prefix + 'Image_Edges', grid_image, global_step)

        grid_image = make_grid(decode_seg_map_sequence(torch.squeeze(target[:9], 1).detach().cpu().numpy(),
                                                       dataset=dataset), 3, normalize=False, range=(0, 255))
        writer.add_image(prefix + 'Groundtruth label', grid_image, global_step)
