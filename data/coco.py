import torch
from torchvision.datasets import CocoDetection
import numpy as np
from random import randint


COCO_CLASSES = ("person","bicycle","car","motorcycle","airplane","bus","train","truck","boat",
                "traffic light","fire hydrant","None","stop sign","parking meter","bench","bird",
                "cat","dog","horse","sheep","cow","elephant","bear","zebra","giraffe","None",
                "backpack","umbrella","None","None","handbag","tie","suitcase","frisbee","skis",
                "snowboard","sports ball","kite","baseball bat","baseball glove","skateboard","surfboard"
                ,"tennis racket","bottle","None","wine glass","cup","fork","knife","spoon","bowl","banana"
                ,"apple","sandwich","orange","broccoli","carrot","hot dog","pizza","donut","cake","chair",
                "couch","potted plant","bed","None","dining table","None","None","toilet","None","tv","laptop"
                ,"mouse","remote","keyboard","cell phone","microwave","oven","toaster","sink","refrigerator",
                "None","book","clock","vase","scissors","teddy bear","hair drier","toothbrush")



class TransformedCocoDataset(CocoDetection):
    def __init__(self, root, annFile, ann_transform=None, max_ims=None):
        super(TransformedCocoDataset, self).__init__(root, annFile)
        self.ann_transform = ann_transform
        self.name = "coco2017"
        self.max_ims = max_ims
        if self.max_ims is None:
            self.length = super(TransformedCocoDataset, self).__len__()
        else:
            self.length = self.max_ims


    def __getitem__(self, idx):
        while True:
            image, ann = super(TransformedCocoDataset, self).__getitem__(idx)
            import cv2
            import numpy as np



            image = image.convert("RGB")
            image = np.asarray(image, dtype=np.uint8)
            image = image[:, :, :3][...,::-1] # RGB -> BRG(opencv format)


            target = []
            for object in ann:
                x = object["bbox"][0] / float(image.shape[1])
                y = object["bbox"][1] / float(image.shape[0])
                w = object["bbox"][2] / float(image.shape[1])
                h = object["bbox"][3] / float(image.shape[0])
                id = object["category_id"] - 1
                target.append(np.array([x, y, x+w,y+h, id], dtype=np.float32))
            if target == []:
                idx = randint(0, self.length - 1)
                continue
            target = np.stack(target)
            if self.ann_transform != None:
                bboxes = target[:, :4]
                labels = target[:, 4]
                image, boxes, labels = self.ann_transform(image, bboxes, labels)
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            # img = np.transpose(image, (2, 0, 1)).astype(np.uint8)
            #
            # print(target)
            # w = img.shape[2]
            # h = img.shape[1]
            #
            # target[:, 0] *= w
            # target[:, 1] *= h
            # target[:, 2] *= w
            # target[:, 3] *= h
            # target = target.astype(np.int32)
            #
            # img = np.transpose(img, (1, 2, 0)).copy()
            # for bbox in target:
            #     cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (255, 0, 0), 2)
            #
            # cv2.imshow("img", img)
            # cv2.waitKey()

            return torch.from_numpy(image).permute(2, 0, 1), target
    def __len__(self):
        return self.length

    def pull_item(self, idx):
        while True:
            image, ann = super(TransformedCocoDataset, self).__getitem__(idx)
            image = image.convert("RGB")
            image = np.asarray(image, dtype=np.uint8)
            image = image[:, :, :3][...,::-1]

            target = []
            for object in ann:
                x = object["bbox"][0] / image.shape[1]
                y = object["bbox"][1] / image.shape[0]
                w = object["bbox"][2] / image.shape[1]
                h = object["bbox"][3] / image.shape[0]
                id = object["category_id"] - 1
                target.append(np.array([x, y, x+w,y+h, id], dtype=np.float32))
            if target == []:
                length = super(TransformedCocoDataset, self).__len__()
                idx = randint(0, length - 1)
                continue
            target = np.stack(target)
            if self.transform != None:
                image, boxes, labels = self.transform(image, target[:, :4], target[:, 4])
                target = np.hstack((boxes, np.expand_dims(labels, axis=1)))

            return torch.from_numpy(image).permute(2, 0, 1), target, image.shape[0], image.shape[1]

