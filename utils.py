import time
import os
import cv2
import matplotlib.pyplot as plt

import visdom

from math import exp

import numpy as np
import torch
import torch.nn.functional as F
from torch.autograd import Variable

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def ssim(image1, image2, K, window_size):
    _, channel1, _, _ = image1.size()
    _, channel2, _, _ = image2.size()
    channel = min(channel1, channel2)

    # gaussian window generation
    sigma = 1.5  # default
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    _1D_window = (gauss / gauss.sum()).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
        window = window.to(device)

    # define constants
    # * L = 255 for constants doesn't produce meaningful results; thus L = 1
    # C1 = (K[0]*L)**2;
    # C2 = (K[1]*L)**2;
    C1 = K[0] ** 2;
    C2 = K[1] ** 2;

    mu1 = F.conv2d(image1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(image2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(image1 * image1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(image2 * image2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(image1 * image2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    return ssim_map.mean()




class Visualizer(object):
    def __init__(self, env='default', **kwargs):
        self.vis = visdom.Visdom(env=env, **kwargs)
        self.index = {}

    def plot_many_stack(self, d):
        '''
        self.plot('loss',1.00)
        '''
        name = list(d.keys())
        name_total = " ".join(name)
        x = self.index.get(name_total, 0)
        val = list(d.values())
        if len(val) == 1:
            y = np.array(val)
        else:
            y = np.array(val).reshape(-1, len(val))
        # print(x)
        self.vis.line(Y=y, X=np.ones(y.shape) * x,
                      win=str(name_total),  # unicode
                      opts=dict(legend=name,
                                title=name_total),
                      update=None if x == 0 else 'append'
                      )
        self.index[name_total] = x + 1

    def plot_heatmap(self, tensor):
        self.vis.images(
            tensor,
            nrow=8,
            win=str('heatmap'),
            opts={'title':'heatmap'}
        )

def compute_loss(preds, targets):
    ''' Modified focal loss. Exactly the same as CornerNet.
        Runs faster and costs a little bit more memory
        Arguments:
        preds (C x H/r x W/r)
        targets (C x H/r x W/r)
    '''
    pos_inds = targets.eq(1).float()
    neg_inds = targets.lt(1).float()

    # beta=4
    neg_weights = torch.pow(1 - targets, 4).float()

    loss = 0
    for pred in preds:
        pred = torch.clamp(torch.sigmoid(pred), min=1e-4, max=1 - 1e-4)
        pos_loss = torch.log(pred) * torch.pow(1 - pred, 2) * pos_inds # 正样本

        # alpha=2
        # print(type(neg_weights))

        neg_loss = torch.log(1 - pred) * torch.pow(pred,2) * neg_weights * neg_inds # 负样本

        num_pos = pos_inds.float().sum()

        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()

        if num_pos == 0:
            loss = loss - neg_loss
        else:
            loss = loss - (pos_loss + neg_loss) / num_pos

    return loss / len(preds)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.) / 2. for ss in shape]
    y, x = np.ogrid[-m:m + 1, -n:n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    # 限制最小的值
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian1 = gaussian2D((diameter, diameter), sigma=diameter / 10)
    gaussian2 = gaussian2D((diameter, diameter), sigma=diameter / 10)

    #gaussian4 = gaussian2D((diameter, diameter), sigma=diameter / 12)
    gaussian=gaussian1+gaussian2

    # 一个圆对应内切正方形的高斯分布

    x, y = int(center[0]), int(center[1])

    height, width = heatmap.shape

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top:y + bottom, x - left:x + right]
    masked_gaussian = gaussian[radius - top:radius +
                               bottom, radius - left:radius + right]

    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)


        # 将高斯分布覆盖到heatmap上，取最大，而不是叠加
    return heatmap


if __name__ == "__main__":
    # h w0
    heatmap = np.zeros((256, 256))
    a=draw_umich_gaussian(heatmap, (94, 130), 3)
    b = draw_umich_gaussian(a, (50, 100), 3) * 255
    #b = draw_umich_gaussian(heatmap, (94, 131), 3) * 255
    cv2.imwrite("./output1/%d.bmp" % 6, b )

    print(heatmap.shape)

    plt.figure()
    img = plt.imshow(heatmap)
    img.set_cmap('hot')
    plt.savefig("heatmap.jpg")
