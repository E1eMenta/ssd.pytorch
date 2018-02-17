from torch.utils.data import DataLoader
from torch.autograd import Variable
import torch
import cv2
import argparse
import numpy as np
from pathlib import Path
import time

from data.augmentation import SSDAugmentation, BaseTransform
from data.coco import TransformedCocoDataset, COCO_CLASSES
from data.voc import AnnotationTransform, VOC_CLASSES, VOCDetection
from data import collate_fn
from utils.init import model_init
from utils import draw

from models import ssd
from utils.box_utils import bboxes2offsets

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training')
parser.add_argument('--resume', default=None, help='path to pretrained model')
args = parser.parse_args()

dataset = "VOC"
cuda = True
num_workers = 2
max_iterations = 100000
report_steps = 300
lr = 0.00002
match_threshold = 0.5

batch_size = 16
height, width = 300, 300
if dataset == "COCO":
    num_classes = len(COCO_CLASSES) + 1
elif dataset == "VOC":
    num_classes = len(VOC_CLASSES) + 1
elif dataset == "LPD":
    num_classes = 3



#Train data loader
if dataset == "COCO":
    cocoRoot = "/mnt/media/users/renat/mscoco/"
    datase_t = TransformedCocoDataset(cocoRoot + "train2017/train2017",
                                      cocoRoot + "annotations_trainval2017/annotations/instances_train2017.json",
                                      SSDAugmentation(size=(height, width), mean=(0,0,0)))

    #Validation data loader
    datase_v = TransformedCocoDataset(cocoRoot + "val2017/val2017",
                                      cocoRoot + "annotations_trainval2017/annotations/instances_val2017.json")

    dataLoader_v = DataLoader(datase_v, batch_size, num_workers=num_workers,
                              shuffle=False, collate_fn=collate_fn, pin_memory=cuda)
elif dataset == "VOC":
    voc_root = "/home/renatkhiz/data/VOCdevkit/"
    train_sets = [('2007', 'trainval'), ('2012', 'trainval')]
    datase_t = VOCDetection(voc_root, train_sets, SSDAugmentation(
        size=(height, width), mean=(0, 0, 0)), AnnotationTransform())

dataLoader_t = DataLoader(datase_t, batch_size, num_workers=num_workers,
                          shuffle=True, collate_fn=collate_fn, pin_memory=cuda)

model_cpu = ssd.VGGSSD(num_classes=num_classes)
model_init(model_cpu)
if cuda:
    model = torch.nn.DataParallel(model_cpu).cuda()
else:
    model = model_cpu

if args.resume != None:
    # model.load_state_dict(torch.load(args.pretrained))
    model_cpu.load_state_dict(torch.load(args.resume, map_location=lambda storage, loc: storage))


criterion = ssd.MultiBoxLoss(num_classes, overlap_thresh=0.5, neg_pos=3, use_gpu=cuda)
# optimizer = torch.optim.Adam(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=3e-4,
                      momentum=0.9, weight_decay=5e-4)

iter_idx = 0
epoch = 0
loss_list = []
loss_c_list = []
loss_l_list = []
model.train()
t = time.time()
while True:
    for images, targets in dataLoader_t:
        iter_idx += 1

        if cuda:
            images = images.cuda()
            targets = [anno.cuda() for anno in targets]
        images = Variable(images)


        # targets = [Variable(anno, volatile=True) for anno in targets]
        output = model(images)
        priors = output[2]
        conf_targets, loc_targets = bboxes2offsets(targets, priors, threshold=match_threshold)
        conf_targets = Variable(conf_targets)
        loc_targets = Variable(loc_targets)



        defaults = output[2].cpu().numpy()
        loc = output[0].data.cpu().numpy()
        conf = torch.nn.functional.softmax(output[1], dim=2)
        conf = conf.data.cpu().numpy()
        # conf_targets, loc_targets = bboxes2offsets(targets, output[2])
        # for image, c,l, target in zip(images.data.cpu().numpy(), conf, loc, targets):
        #     # print(c.shape)
        #     image = image.astype(np.uint8)
        #     image = np.transpose(image, (1, 2, 0))
        #     target = target.cpu().numpy()
        #     img = draw.draw_target(image, target, color=(0,255,0))
        #     img = draw.draw_pred_defaults(img,c,defaults)
        #     # img = draw.draw_pred_offsets(img,c,l,defaults)
        #
        #     cv2.imshow("img", img)
        #     cv2.waitKey()

        for conf_target, target, image, loc_target in zip(conf_targets.data.cpu().numpy(), targets, images.data.cpu().numpy(), loc_targets.data.cpu().numpy()):
            image = image.astype(np.uint8)
            image = np.transpose(image, (1, 2, 0))
            img = draw.draw_true_defaults(image, conf_target, defaults)
            img = draw.draw_true_offsets(img, conf_target, loc_target, defaults, color=(0,255,0))
            cv2.imshow("img", img)
            cv2.waitKey()



        loss, loss_c, loss_l  = criterion(output, conf_targets, loc_targets)
        # print(N // batch_size)
        # print(loss.data.cpu()[0], loss_l.data.cpu()[0], loss_c.data.cpu()[0])
        # loss = loss_c + loss_l
        loss_list.append(loss.data.cpu()[0])
        loss_c_list.append(loss_c.data[0])
        loss_l_list.append(loss_l.data[0])

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm(model.parameters(), 100)
        optimizer.step()

        if iter_idx % 100 == 0 and iter_idx != 0:
            loss = np.mean(np.array(loss_list))
            loss_c = np.mean(np.array(loss_c_list))
            loss_l = np.mean(np.array(loss_l_list))
            delta = time.time() - t
            t = time.time()
            print("{0} total loss: {1:.4f}. Class: {2:.4f}, loc {3:.4f}, time: {4}".format(iter_idx, loss, loss_c, loss_l, delta))
            loss, loss_c, loss_l = [], [], []
            # print("B {0}:  total loss: {1:.4f}.".format(iter_idx, loss))
            # loss_list = []
        if iter_idx % 2000 == 0 and iter_idx != 0:
            torch.save(model.state_dict(), 'weights/voc_{}.pth'.format(iter_idx))

    epoch += 1