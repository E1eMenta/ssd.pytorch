import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

class Inception(nn.Module):
    def __init__(self, in_channels, ch1_1, ch1_2, ch2_1, ch2_2, ch2_3, ch2_4):
        super(Inception, self).__init__()
        self.l1x1 = nn.Conv2d(in_channels, ch2_1, kernel_size=1)

        self.l3x3_reduce = nn.Conv2d(in_channels, ch1_1, kernel_size=1)
        self.l3x3 = nn.Conv2d(ch1_1, ch2_2, kernel_size=3, padding=1)

        self.l5x5_reduce = nn.Conv2d(in_channels, ch1_2, kernel_size=1)
        self.l5x5 = nn.Conv2d(ch1_2, ch2_3, kernel_size=5, padding=2)

        self.pool_ploj = nn.Conv2d(in_channels, ch2_4, kernel_size=1)

        self.out_channels_num = ch2_1 + ch2_2 + ch2_3 + ch2_4

    def channels_num(self):
        return self.out_channels_num

    def forward(self, x):
        branch1x1 = F.relu(self.l1x1(x), inplace=True)

        branch3x3 = F.relu(self.l3x3_reduce(x), inplace=True)
        branch3x3 = F.relu(self.l3x3(branch3x3), inplace=True)

        branch5x5 = F.relu(self.l5x5_reduce(x), inplace=True)
        branch5x5 = F.relu(self.l5x5(branch5x5), inplace=True)

        branch_pool = F.max_pool2d(x, kernel_size=3, stride=1, padding=1)
        branch_pool = F.relu(self.pool_ploj(branch_pool), inplace=True)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)