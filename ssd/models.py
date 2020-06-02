import torch
import torch.nn as nn
import math
from itertools import product


class SSDHead(nn.Module):

    def __init__(self, in_channels, num_classes, num_anchors, img_size, feature_size, min_size, max_size):
        super(SSDHead, self).__init__()
        self.location = nn.Conv2d(in_channels, num_anchors * 4, kernel_size=3, padding=1)
        self.conference = nn.Conv2d(in_channels, num_anchors * (num_classes + 1), kernel_size=3, padding=1)
        self.img_size = img_size
        self.feature_size = feature_size
        self.min_size = min_size
        self.max_size = max_size
        if num_anchors == 6:
            self.aspect_ratios = [2, 3]
        elif num_anchors == 4:
            self.aspect_ratios = [2]
        else:
            raise KeyError

    def _prior_boxes(self, device, dtype):
        mean = []
        for i, j in product(range(self.feature_size), repeat=2):
            # unit center x,y
            cx = (j + 0.5) / self.feature_size
            cy = (i + 0.5) / self.feature_size

            # aspect_ratio: 1
            # rel size: min_size
            s_k = self.min_sizes / self.image_size
            mean += [cx, cy, s_k, s_k]

            # aspect_ratio: 1
            # rel size: sqrt(s_k * s_(k+1))
            s_k_prime = math.sqrt(s_k * (self.max_sizes / self.image_size))
            mean += [cx, cy, s_k_prime, s_k_prime]

            # rest of aspect ratios
            for ar in self.aspect_ratios:
                mean += [cx, cy, s_k * math.sqrt(ar), s_k / math.sqrt(ar)]
                mean += [cx, cy, s_k / math.sqrt(ar), s_k * math.sqrt(ar)]

        out = torch.tensor(mean, dtype=dtype, device=device).view(-1, 4)
        out.clamp_(min=0, max=1)
        return out

    def forward(self, input):
        prior = self._prior_boxes(input.device, input.dtype)
        loc = self.location(input)
        conf = self.conference(input)

        loc = loc.permute(0, 2, 3, 1).contiguous()
        conf = conf.permute(0, 2, 3, 1).contiguous()

        out = (loc, prior, conf)

        return out
