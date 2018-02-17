import torch.nn.functional as F
import torch.nn as nn
import torch
from torch.autograd import Variable

from models.layers import Inception
from utils.box_utils import match, log_sum_exp, bboxes2offsets, one_hot, smooth_l1


class VGGSSD(nn.Module):
    base = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'C', 512, 512, 512, 'M',
                512, 512, 512]
    extras_list = [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256]
    mbox = [4, 6, 6, 6, 4, 4]

    feature_maps= [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]

    min_sizes = [30, 60, 111, 162, 213, 264]

    max_sizes = [60, 111, 162, 213, 264, 315]
    image_size = 300
    aspect_ratios = [[2.0, 1/2.0], [2.0, 1/2.0, 3.0, 1/3.0], [2.0, 1/2.0, 3.0, 1/3.0],
                       [2.0, 1/2.0, 3.0, 1/3.0], [2.0, 1/2.0], [2.0, 1/2.0]]
    # aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

    def create_defaults(self):
        mean = []
        from math import sqrt
        from itertools import product
        for k, f in enumerate(self.feature_maps):
            # print(f)
            # print(self.min_sizes[k], self.max_sizes[k])
            s_k = self.min_sizes[k] / self.image_size
            f_k = self.image_size / self.steps[k]
            s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
            for i, j in product(range(f), repeat=2):
                # print(self.steps[k])

                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                # s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                # s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
        # input("epogne")
        output = torch.Tensor(mean).view(-1, 4)
        return output
    def vgg_build(self, cfg, i, batch_norm=False):
        layers = []
        in_channels = i
        for v in cfg:
            if v == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif v == 'C':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)]
            else:
                conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
                if batch_norm:
                    layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
                else:
                    layers += [conv2d, nn.ReLU(inplace=True)]
                in_channels = v
        pool5 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        conv6 = nn.Conv2d(512, 1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(1024, 1024, kernel_size=1)
        layers += [pool5, conv6,
                   nn.ReLU(inplace=True), conv7, nn.ReLU(inplace=True)]
        return layers

    def add_extras(self, cfg, i, batch_norm=False):
        # Extra layers added to VGG for feature scaling
        layers = []
        in_channels = i
        flag = False
        for k, v in enumerate(cfg):
            if in_channels != 'S':
                if v == 'S':
                    layers += [nn.Conv2d(in_channels, cfg[k + 1],
                                         kernel_size=(1, 3)[flag], stride=2, padding=1)]
                else:
                    layers += [nn.Conv2d(in_channels, v, kernel_size=(1, 3)[flag])]
                flag = not flag
            in_channels = v
        return layers

    def multibox(self, vgg, extra_layers, cfg, num_classes):
        loc_layers = []
        conf_layers = []
        vgg_source = [24, -2]
        for k, v in enumerate(vgg_source):
            loc_layers += [nn.Conv2d(vgg[v].out_channels,
                                     cfg[k] * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(vgg[v].out_channels,
                                      cfg[k] * num_classes, kernel_size=3, padding=1)]
        for k, v in enumerate(extra_layers[1::2], 2):
            loc_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                     * 4, kernel_size=3, padding=1)]
            conf_layers += [nn.Conv2d(v.out_channels, cfg[k]
                                      * num_classes, kernel_size=3, padding=1)]
        return vgg, extra_layers, (loc_layers, conf_layers)

    def __init__(self, num_classes, use_cuda=True):
        super(VGGSSD, self).__init__()
        self.num_classes = num_classes
        self.use_cuda = use_cuda

        self.priors = self.create_defaults()

        base, extras, head = self.multibox(self.vgg_build(self.base, 3),
                                           self.add_extras(self.extras_list, 1024),
                                           self.mbox, self.num_classes)
        self.vgg = nn.ModuleList(base)
        self.extras = nn.ModuleList(extras)
        self.loc = nn.ModuleList(head[0])
        self.conf = nn.ModuleList(head[1])

    def forward(self, x):
        sources = list()
        loc = list()
        conf = list()

        # apply vgg up to conv4_3 relu
        for k in range(23):
            x = self.vgg[k](x)

        sources.append(x)

        # apply vgg up to fc7
        for k in range(23, len(self.vgg)):
            x = self.vgg[k](x)
        sources.append(x)

        # apply extra layers and cache source layer outputs
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if k % 2 == 1:
                sources.append(x)

        # apply multibox head to source layers
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        loc = torch.cat([o.view(o.size(0), -1, 4) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1, self.num_classes) for o in conf], 1)
        # print(loc.size())
        # print(conf.size())
        # print(self.priors.size())
        #
        # import numpy as np
        # anchors = np.load("anchors.npy")
        # priors = self.priors.cpu().numpy()
        # print(np.mean(anchors-priors)/(np.mean(anchors) + np.mean(priors)))
        # print(self.priors.size())
        # input("ef[omw")

        # if self.phase == "test":
        #     output = self.detect(
        #         loc.view(loc.size(0), -1, 4),  # loc preds
        #         self.softmax(conf.view(-1, self.num_classes)),  # conf preds
        #         self.priors.type(type(x.data))  # default boxes
        #     )
        # else:
        #     output = (
        #         loc.view(loc.size(0), -1, 4),
        #         conf.view(conf.size(0), -1, self.num_classes),
        #         self.priors
        #     )
        self.priors = self.priors.type(type(x.data))
        # if self.use_cuda:
        #     self.priors = Variable(self.priors.cuda(), volatile=True)
        # else:
        #     self.priors = Variable(self.priors, volatile=True)
        return loc, conf, self.priors

class SSDBase(nn.Module):
    def __init__(self, class_num):
        super(SSDBase, self).__init__()
        # ssd parameters
        self.anchors = [12, 12, 12, 12, 12]
        self.aspect_ratios = [1.0, 3.0, 4.0, 5.0]
        self.class_num = class_num
        self.offsets_num = 4
        self.old_height = None
        self.old_width = None

        # cnn weights initialization
        self.conv1 = nn.Conv2d(3, 21, 3, padding=1, stride=2)
        self.conv2 = nn.Conv2d(21, 12, 1)
        self.conv3 = nn.Conv2d(12, 20, 3, padding=1)

        self.ins1 = Inception(20, 12, 8, 6, 16, 8, 6)
        self.ins3 = Inception(36, 4, 8, 18, 24, 10, 12)
        ins3_channels = self.ins3.channels_num()
        self.ins4 = Inception(64, 12, 8, 6, 16, 8, 6)
        self.ins5 = Inception(36, 16, 8, 16, 28, 12, 8)
        ins5_channels = self.ins5.channels_num()
        self.ins6 = Inception(64, 12, 8, 6, 16, 8, 6)
        ins6_channels = self.ins6.channels_num()
        self.ins8 = Inception(36, 16, 8, 16, 28, 12, 8)
        ins8_channels = self.ins8.channels_num()
        self.ins9 = Inception(64, 16, 8, 16, 28, 12, 8)
        ins9_channels = self.ins9.channels_num()

        self.out0 = nn.Conv2d(ins3_channels, self.anchors[0] * (self.class_num + self.offsets_num), 3, padding=1)
        self.out1 = nn.Conv2d(ins5_channels, self.anchors[1] * (self.class_num + self.offsets_num), 3, padding=1)
        self.out2 = nn.Conv2d(ins6_channels, self.anchors[2] * (self.class_num + self.offsets_num), 3, padding=1)
        self.out3 = nn.Conv2d(ins8_channels, self.anchors[3] * (self.class_num + self.offsets_num), 3, padding=1)
        self.out4 = nn.Conv2d(ins9_channels, self.anchors[4] * (self.class_num + self.offsets_num), 3, padding=1)

        self.softmax = nn.Softmax()

    def create_defaults(self, out_sizes, box_ratios):
        import numpy as np
        boxes = []
        for out_size in out_sizes:
            dy = 1.0 / out_size[0]
            dx = 1.0 / out_size[1]
            for i in range(out_size[0]):
                for j in range(out_size[1]):  # coordinate loop
                    for ratio in box_ratios:  # aspect ratio loop
                        ny = int(np.ceil(1 * np.sqrt(ratio))) + 1
                        nx = int(np.ceil(1 / np.sqrt(ratio)))
                        for ax in range(nx):
                            for ay in range(ny):
                                cy = dy * (i + 1.0 * (ay + 1) / (ny + 1))
                                cx = dx * (j + 1.0 * (ax + 1) / (nx + 1))
                                h = 1.15 * dy / np.sqrt(ratio)
                                w = 1.15 * dx * np.sqrt(ratio)
                                newbox = np.array([cx, cy, w, h])
                                boxes.append(newbox)
        boxes = np.stack(boxes)
        # np.savetxt("default_boxes.csv", boxes, delimiter="\n")
        return torch.from_numpy(boxes).type(torch.FloatTensor)

    def detection_head(self, layers, conv_ops, batch_size, change_defaults):
        outs = []
        out_sizes = []
        for layer, conv_op in zip(layers, conv_ops):
            # print(layer.size, "\n\n")
            out_sizes.append(layer.size()[2:])
            outs.append(conv_op(layer).permute(0, 2, 3, 1).contiguous())

        r_outs = [out.view(batch_size, -1, self.class_num + 1 + self.offsets_num) for out in outs]
        prep_out = torch.cat(r_outs, 1)

        loc = prep_out[:, :, :4]
        conf = prep_out[:, :, 4:]

        if change_defaults:
            print("create new defaults")
            defaults = self.create_defaults(out_sizes, self.aspect_ratios).cuda()
            self.defaults = Variable(defaults, volatile=True)

        return loc, conf, self.defaults

    def forward(self, x):
        batch_size = x.size(0)
        change_defaults = False
        if self.old_height != x.size(2) or self.old_width != x.size(3):
            change_defaults = True
            self.old_height = x.size(2)
            self.old_width = x.size(3)

        x = x * 0.00392156862
        conv1 = F.relu(self.conv1(x), inplace=True)
        pool1 = F.max_pool2d(conv1, 3, stride=2, ceil_mode=True)

        conv2 = F.relu(self.conv2(pool1), inplace=True)
        conv3 = F.relu(self.conv3(conv2), inplace=True)
        pool2 = F.max_pool2d(conv3, 3, stride=2, ceil_mode=True)

        ins1 = self.ins1(pool2)
        ins3 = self.ins3(ins1)
        pool3 = F.max_pool2d(ins3, 3, stride=2, ceil_mode=True)

        ins4 = self.ins4(pool3)
        ins5 = self.ins5(ins4)
        pool4 = F.max_pool2d(ins5, 3, stride=2, ceil_mode=True)

        ins6 = self.ins6(pool4)
        pool5 = F.max_pool2d(ins6, 3, stride=2, ceil_mode=True)

        ins8 = self.ins8(pool5)
        pool6 = F.max_pool2d(ins8, 3, stride=2, ceil_mode=True)

        ins9 = self.ins9(pool6)

        layers = [ins3, ins5, ins6, ins8, ins9]
        conv_ops = [self.out0, self.out1, self.out2, self.out3, self.out4]

        loc, conf, defauts = self.detection_head(layers, conv_ops, batch_size, change_defaults)

        return loc, conf, defauts


class MultiBoxLoss(nn.Module):
    """SSD Weighted Loss Function
    Compute Targets:
        1) Produce Confidence Target Indices by matching  ground truth boxes
           with (default) 'priorboxes' that have jaccard index > threshold parameter
           (default threshold: 0.5).
        2) Produce localization target by 'encoding' variance into offsets of ground
           truth boxes and their matched  'priorboxes'.
        3) Hard negative mining to filter the excessive number of negative examples
           that comes with using a large number of default bounding boxes.
           (default negative:positive ratio 3:1)
    Objective Loss:
        L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
        Where, Lconf is the CrossEntropy Loss and Lloc is the SmoothL1 Loss
        weighted by α which is set to 1 by cross val.
        Args:
            c: class confidences,
            l: predicted boxes,
            g: ground truth boxes
            N: number of matched default boxes
        See: https://arxiv.org/pdf/1512.02325.pdf for more details.
    """

    def __init__(self, num_classes, overlap_thresh, bkg_label=0, neg_pos=3, alfa=1, use_gpu=True):
        super().__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.negpos_ratio = neg_pos
        self.alfa = alfa

    def cross_entropy_loss(self, x, y):
        '''Cross entropy loss w/o averaging across all samples.
        Args:
          x: (tensor) sized [N,D].
          y: (tensor) sized [N,].
        Return:
          (tensor) cross entroy loss, sized [N,].
        '''
        xmax = x.data.max()
        log_sum_exp = torch.log(torch.sum(torch.exp(x - xmax), 1)) + xmax
        return log_sum_exp - torch.squeeze(x.gather(1, y.view(-1, 1)), 1)

    def hard_negative_mining(self, conf_loss, pos):
        '''Return negative indices that is 3x the number as postive indices.
        Args:
          conf_loss: (tensor) cross entroy loss between conf_preds and conf_targets, sized [N*8732,].
          pos: (tensor) positive(matched) box indices, sized [N,8732].
        Return:
          (tensor) negative indices, sized [N,8732].
        '''
        batch_size, num_boxes = pos.size()
        # print(conf_loss.size())
        # print(pos.size())
        conf_loss[pos] = 0  # set pos boxes = 0, the rest are neg conf_loss
        # print(conf_loss.size())
        # conf_loss = conf_loss.view(batch_size, -1)  # [N,8732]

        _, idx = conf_loss.sort(1, descending=True)  # sort by neg conf_loss
        _, rank = idx.sort(1)  # [N,8732]
        num_pos = pos.long().sum(1)  # [N,1]
        num_neg = torch.clamp(3 * num_pos, max=num_boxes - 1)  # [N,1]
        num_neg = num_neg.unsqueeze(1)
        # print("num_neg", num_neg.size())
        # print("rank", rank.size())
        # print("num_neg.expand_as(rank)", num_neg.expand_as(rank).size())

        neg = rank < num_neg.expand_as(rank)  # [N,8732]
        return neg

    # def forward(self, predictions, conf_targets, loc_targets):
    #     loc_data, conf_data, priors = predictions
    #     conf_t = conf_targets
    #     loc_t = loc_targets
    #
    #     num = loc_data.size(0)
    #     priors = priors[:loc_data.size(1), :]
    #     num_priors = (priors.size(0))
    #     num_classes = self.num_classes
    #
    #     pos = conf_t > 0
    #     # num_pos = pos.sum(keepdim=True)
    #
    #     # Localization Loss (Smooth L1)
    #     # Shape: [batch,num_priors,4]
    #     pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
    #     loc_p = loc_data[pos_idx].view(-1, 4)
    #     loc_t = loc_t[pos_idx].view(-1, 4)
    #     loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)
    #
    #     # Compute max conf across batch for hard negative mining
    #     batch_conf = conf_data.view(-1, self.num_classes)
    #
    #     loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))
    #
    #     # Hard Negative Mining
    #     loss_c[pos] = 0  # filter out pos boxes for now
    #     loss_c = loss_c.view(num, -1)
    #     _, loss_idx = loss_c.sort(1, descending=True)
    #     _, idx_rank = loss_idx.sort(1)
    #     num_pos = pos.long().sum(1, keepdim=True)
    #     num_neg = torch.clamp(self.negpos_ratio * num_pos, max=pos.size(1) - 1)
    #     neg = idx_rank < num_neg.expand_as(idx_rank)
    #
    #     # Confidence Loss Including Positive and Negative Examples
    #     pos_idx = pos.unsqueeze(2).expand_as(conf_data)
    #     neg_idx = neg.unsqueeze(2).expand_as(conf_data)
    #     conf_p = conf_data[(pos_idx + neg_idx).gt(0)].view(-1, self.num_classes)
    #     targets_weighted = conf_t[(pos + neg).gt(0)]
    #     loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)
    #
    #     # Sum of losses: L(x,c,l,g) = (Lconf(x, c) + αLloc(x,l,g)) / N
    #
    #     N = num_pos.data.sum()
    #     loss_l /= N
    #     loss_c /= N
    #
    #     conf_loss = loss_c
    #     loc_loss = loss_l
    #     total_loss = conf_loss + loc_loss
    #
    #     #
    #     # batch_size, num_boxes, _ = loc_preds.size()
    #     # pos = conf_targets > 0  # [N,8732], pos means the box matched.
    #     # num_matched_boxes = pos.data.long().sum()
    #     # if num_matched_boxes == 0:
    #     #     return Variable(torch.Tensor([0]))
    #     #
    #     # ################################################################
    #     # # loc_loss = SmoothL1Loss(pos_loc_preds, pos_loc_targets)
    #     # ################################################################
    #     # pos_mask = pos.unsqueeze(2).expand_as(loc_preds)    # [N,8732,4]
    #     # pos_loc_preds = loc_preds[pos_mask].view(-1,4)      # [#pos,4]
    #     # pos_loc_targets = loc_targets[pos_mask].view(-1,4)  # [#pos,4]
    #     # loc_loss = F.smooth_l1_loss(pos_loc_preds, pos_loc_targets, size_average=False)
    #     # ################################################################
    #     # # conf_loss = CrossEntropyLoss(pos_conf_preds, pos_conf_targets)
    #     # #           + CrossEntropyLoss(neg_conf_preds, neg_conf_targets)
    #     # ################################################################
    #     # conf_loss = self.cross_entropy_loss(conf_preds.view(-1, self.num_classes),
    #     #                                     conf_targets.view(-1))  # [N*8732,]
    #     # conf_loss = conf_loss.view(batch_size, -1)
    #     #
    #     # neg = self.hard_negative_mining(conf_loss, pos)  # [N,8732]
    #     #
    #     # pos_mask = pos.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
    #     # neg_mask = neg.unsqueeze(2).expand_as(conf_preds)  # [N,8732,21]
    #     # mask = (pos_mask + neg_mask).gt(0)
    #     # pos_and_neg = (pos + neg).gt(0)
    #     # preds = conf_preds[mask].view(-1, self.num_classes)  # [#pos+#neg,21]
    #     # targets = conf_targets[pos_and_neg]  # [#pos+#neg,]
    #     # conf_loss = F.cross_entropy(preds, targets, size_average=False)
    #     #
    #     # loc_loss /= num_matched_boxes
    #     # conf_loss /= num_matched_boxes
    #     #
    #     # total_loss = conf_loss + self.alfa * loc_loss
    #     #
    #     return total_loss, conf_loss, loc_loss, 1

    def forward(self, predictions, conf_targets, loc_targets):
        pred_loc, pred_conf, priors = predictions
        true_labels = conf_targets
        true_loc = loc_targets
        num_classes = pred_conf.size()[2]

        isPositive = true_labels != 0
        positive_num = isPositive.data.sum()

        pred_conf = pred_conf.view(-1, num_classes)
        pred_conf = F.log_softmax(pred_conf, dim=-1)
        class_loss = F.nll_loss(pred_conf, true_labels.view(-1), size_average=False)

        isPositive = isPositive.unsqueeze(isPositive.dim()).expand_as(pred_loc)

        loc_loss = F.smooth_l1_loss(pred_loc[isPositive], true_loc[isPositive], size_average=False)

        total_loss = (class_loss + self.alfa * loc_loss) / positive_num

        return total_loss, class_loss, loc_loss