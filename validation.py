import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

import cv2

def detect(loc_data, conf_data, prior_data, background_label=0, top_k=100, conf_thresh=0.5, nms_threshold=0.4, variance=(0.1, 0.2)):
    """
    Args:
        loc_data: (tensor) Loc preds from loc layers
            Shape: [batch,num_priors, 4]
        conf_data: (tensor) Shape: Conf preds from conf layers
            Shape: [batch, num_priors, num_classes]
        prior_data: (tensor) Prior boxes and variances from priorbox layers
            Shape: [1,num_priors,4]
    """
    batch_size = loc_data.shape[0]  # batch size
    # masks for removing background bboxes
    conf_data = np.delete(conf_data, (background_label), 2)
    mask_all = conf_data.max(axis=2) > conf_thresh

    boxes_list  = []
    labels_list = []
    scores_list = []

    # Decode predictions into bboxes.
    for i in range(batch_size):
        conf_data_i = conf_data[i]
        loc_data_i  = loc_data[i]

        conf_data_i = conf_data_i[mask_all[i]]

        decoded_boxes = decode_np(loc_data_i[mask_all[i]], prior_data[mask_all[i]], variance)
        bboxes, labels, scores = nms_np(decoded_boxes, conf_data_i, nms_threshold, top_k)

        boxes_list.append(bboxes)
        labels_list.append(labels)
        scores_list.append(scores)
    return boxes_list, labels_list, scores_list

class Validator():
    def __init__(self, data_loader, model, num_classes, criterion=None,
                 max_ims=None, conf_thresh=0.5, nms_thresh=0.3, match_treshold=0.5, cuda=True):
        self.loader = data_loader
        self.gt = None
        self.model = model
        self.num_classes = num_classes - 1
        self.criterion = criterion
        self.nms_thresh = nms_thresh
        self.conf_thresh = conf_thresh
        self.match_treshold = match_treshold
        self.cuda = cuda
        if max_ims != None:
            self.max_ims = max_ims
        else:
            self.max_ims = len(self.loader)

    def validate(self):

        # precision, recall = self.precision_recall(self.conf_thresh)
        #
        # for idx in range(self.num_classes):
        #     print("{2:20}  Precision: {0:.4f}, Recall: {1:.4f}".format(precision[idx], recall[idx], self.classes_names[idx]))
        if self.criterion:
            AP, losses = self.AP()
        else:
            AP = self.AP()
        # for idx in range(self.num_classes):
        #     print("{0:20}  AP: {1:.4f}".format(self.classes_names[idx], AP[idx]))
        #
        # print("mAP: {}".format(np.mean(AP)))

        mAP = np.mean(AP)
        if self.criterion:
            return AP, mAP, losses
        else:
            return AP, mAP

    def get_pred(self, loc_data, conf_data, priors,
                 background_label=0,
                 top_k=100,
                 conf_thresh=0.5,
                 nms_threshold=0.5,
                 variance=(0.1, 0.2)):


        conf_data = F.softmax(conf_data, dim=2).data.cpu().numpy()
        loc_data = loc_data.data.cpu().numpy()
        priors = priors.data.cpu().numpy()

        pred_boxes, pred_labels, scores = detect(loc_data,
                                                 conf_data,
                                                 priors,
                                                 background_label=background_label,
                                                 top_k=top_k,
                                                 conf_thresh=conf_thresh,
                                                 nms_threshold=nms_threshold,
                                                 variance=variance)
        return pred_boxes, pred_labels, scores, priors

    def precision_recall(self, conf_thresh):
        positives = np.zeros((self.num_classes), dtype=np.int32)
        tp = np.zeros((self.num_classes), dtype=np.int32)
        fp = np.zeros((self.num_classes), dtype=np.int32)
        loss = []
        loss_loc = []
        loss_conf = []
        for imgs, gt_bboxes in self.loader:
            if self.cuda:
                imgs = imgs.cuda()
            imgs = Variable(imgs)
            gt_bboxes = [Variable(gt) for gt in gt_bboxes]
            loc_data, conf_data, priors = self.model(imgs)

            if self.criterion != None:
                loss_l, loss_c = self.criterion((loc_data, conf_data, priors), gt_bboxes)
                loss_loc  += [loss_l.data.cpu().numpy()]
                loss_conf += [loss_c.data.cpu().numpy()]
                loss += [(loss_l + loss_c).data.cpu().numpy()]


            pred_boxes, pred_labels, scores, priors = self.get_pred(loc_data, conf_data, priors,
                                                                    background_label=0,
                                                                    top_k=100,
                                                                    conf_thresh=conf_thresh,
                                                                    nms_threshold=self.nms_thresh,
                                                                    variance = (0.1, 0.2)
                                                                    )
            gt_bboxes = [gt.data.cpu().numpy() for gt in gt_bboxes]
            # for i in range(len(gt_bboxes)):
            #     img = np.transpose(imgs[i].data.cpu().numpy(), (1,2,0)).astype(np.uint8)[:, :, ::-1]
            #     pred_box = pred_boxes[i]
            #     img = draw.draw_target(img, pred_box)
            #     img = draw.draw_target(img, gt_bboxes[i], color=(0,0,255))
            #     cv2.imshow("img", img)
            #     cv2.waitKey()


            for i in range(len(gt_bboxes)):
                gt = gt_bboxes[i]
                pred_boxes_i = pred_boxes[i]
                pred_labels_i = pred_labels[i]
                positives_i, tp_i, fp_i = match_output(gt[:, :4], gt[:, 4],
                                                   pred_boxes_i, pred_labels_i,
                                                   self.num_classes, iou_match=self.match_treshold)

                positives += positives_i
                fp += fp_i
                tp += tp_i


        recall = tp / positives
        precision = tp / (tp + fp + 0.001)

        if self.criterion != None:
            loss = np.mean(np.stack(loss))
            loss_conf = np.mean(np.stack(loss_conf))
            loss_loc = np.mean(np.stack(loss_loc))
            losses = (loss, loss_conf, loss_loc)

            return precision, recall, losses
        else:
            return precision, recall


    def AP(self):

        gt_list = []
        pred_boxes_list, pred_labels_list, scores_list = [], [], []
        loss = []
        loss_loc = []
        loss_conf = []
        for imgs, gt_bboxes in self.loader:
            if self.cuda:
                imgs = imgs.cuda()
                gt_bboxes =  [ gt.cuda() for gt in gt_bboxes]
            imgs = Variable(imgs)
            gt_bboxes = [Variable(gt) for gt in gt_bboxes]
            loc_data, conf_data, priors = self.model(imgs)

            if self.criterion != None:
                loss_l, loss_c = self.criterion((loc_data, conf_data, priors), gt_bboxes)
                loss_loc  += [loss_l.data.cpu().numpy()]
                loss_conf += [loss_c.data.cpu().numpy()]
                loss += [(loss_l + loss_c).data.cpu().numpy()]

            pred_boxes, pred_labels, scores, priors = self.get_pred(loc_data, conf_data, priors,
                                                                    background_label=0,
                                                                    top_k=100,
                                                                    conf_thresh=0.01,
                                                                    nms_threshold=self.nms_thresh,
                                                                    variance=(0.1, 0.2)
                                                                    )



            gt_bboxes = [gt.data.cpu().numpy() for gt in gt_bboxes]
            gt_list += gt_bboxes
            pred_boxes_list += pred_boxes
            pred_labels_list += pred_labels
            scores_list += scores

        tp_list, pos_list = [], []

        for pred_boxes, pred_labels, scores, gt in zip(pred_boxes_list, pred_labels_list, scores_list, gt_list):
            pos, true_pos = mathching_mask(gt[:, :4], gt[:, 4],
                                                 pred_boxes, pred_labels,
                                                 self.num_classes, iou_match=self.match_treshold)
            tp_list.append(true_pos)
            pos_list.append(pos)

        print(len(pos_list))
        positives = np.stack(pos_list)
        positives = np.sum(positives, axis=0)
        print(positives)

        tp_list = np.concatenate(tp_list, axis=0)
        pred_labels_list = np.concatenate(pred_labels_list, axis=0)
        scores_list = np.concatenate(scores_list, axis=0)


        AP = np.zeros((self.num_classes))
        for class_idx in range(self.num_classes):
            idx = pred_labels_list == class_idx
            tp_list_i = tp_list[idx]
            scores_list_i = scores_list[idx]
            idx_sort = np.argsort(-scores_list_i)
            fa = 1 - tp_list_i[idx_sort]
            tp_list_i = np.cumsum(tp_list_i[idx_sort])
            fa_list_i = np.cumsum(fa)

            precision = tp_list_i / (tp_list_i + fa_list_i + 0.001)
            recall = tp_list_i / positives[class_idx]
            AP[class_idx] = np.sum((recall[1:] - recall[:-1]) * precision[1:])

        if self.criterion != None:
            loss = np.mean(np.stack(loss))
            loss_conf = np.mean(np.stack(loss_conf))
            loss_loc = np.mean(np.stack(loss_loc))
            losses = (loss, loss_conf, loss_loc)
            return AP, losses
        else:
            return AP




def mathching_mask(true_bboxes, true_labels, pred_bboxes, pred_labels, num_classes, iou_match = 0.5):
    '''
    Calculates per class matching between true and predicted bounding boxes
    true_bboxes: ground truth bboxes [x1, y1, x2, y2], Shape: [N, 4]
    true_labels: ground truth labels, Shape: [N]
    pred_bboxes: predicted bboxes [x1, y1, x2, y2],  Shape: [M, 4]
    pred_labels: predicted labels, Shape: [N]
    num_classes: number of classes
    iou_match: threshold for compare bboxes
    return:
    positives: Number of bboxes corresponding to all classes, Shape: [num_classes]
    is_matched: Vector of pred_bboxes length, is_matched[i] = 1 - pred_bboxes[i] is matched with some ground truth
                                              is_matched[i] = 0 - pred_bboxes[i] is false alarm, Shape: [M]
    '''
    positives = np.zeros((num_classes), dtype=np.int32)
    is_matched = np.zeros((pred_bboxes.shape[0]), dtype=np.int32)

    for class_idx in range(num_classes):
        true_bboxes_i = true_bboxes[true_labels == class_idx]
        pred_bboxes_i = pred_bboxes[pred_labels == class_idx]
        if len(pred_bboxes_i) == 0:
            continue

        true_matched = np.zeros((len(true_bboxes_i)), dtype=np.int32)
        pred_matched = np.zeros((len(pred_bboxes_i)), dtype=np.int32)
        for idx, true_bbox in enumerate(true_bboxes_i):
            iou = IoU_point_np(true_bbox, pred_bboxes_i)

            # exclude matched bboxes
            iou = iou * (1 - pred_matched)

            # find matching
            iou_max = np.max(iou)
            if iou_max > iou_match:
                true_matched[idx] = 1
                pred_matched[np.argmax(iou)] = 1
        is_matched[pred_labels == class_idx] = pred_matched
        positives[class_idx] = len(true_bboxes_i)

    return positives, is_matched

def match_output(true_bboxes, true_labels, pred_bboxes, pred_labels, num_classes, iou_match = 0.5):
    '''
    Calculates per class matching between true and predicted bounding boxes
    true_bboxes: ground truth bboxes [x1, y1, x2, y2], Shape: [N, 4]
    true_labels: ground truth labels, Shape: [N]
    pred_bboxes: predicted bboxes [x1, y1, x2, y2],  Shape: [M, 4]
    pred_labels: predicted labels, Shape: [N]
    num_classes: number of classes
    iou_match: threshold for compare bboxes
    return:
    positives: Number of bboxes corresponding to all classes, Shape: [num_classes]
    tp: Per class true positives number, Shape: [num_classes]
    fp: Per class false positives number, Shape: [num_classes]
    '''
    positives = np.zeros((num_classes), dtype=np.int32)
    tp = np.zeros((num_classes), dtype=np.int32)
    fp = np.zeros((num_classes), dtype=np.int32)
    for class_idx in range(num_classes):
        true_bboxes_i = true_bboxes[true_labels == class_idx]
        pred_bboxes_i = pred_bboxes[pred_labels == class_idx]
        true_matched = np.zeros((len(true_bboxes_i)), dtype=np.int32)
        pred_matched = np.zeros((len(pred_bboxes_i)), dtype=np.int32)
        if len(pred_bboxes_i) > 0:
            for idx, true_bbox in enumerate(true_bboxes_i):
                # print(true_bbox)
                # print(pred_bboxes_i)
                iou = IoU_point_np(true_bbox, pred_bboxes_i)
                # print(iou)
                # print()
                # exclude matched bboxes
                iou = iou * (1 - pred_matched)

                # find matching
                iou_max = np.max(iou)
                if iou_max > iou_match:
                    true_matched[idx] = 1
                    pred_matched[np.argmax(iou)] = 1
        positives[class_idx] = len(true_matched)
        tp[class_idx] = np.sum(true_matched)
        fp[class_idx] = np.sum(1 - pred_matched)
    # input("wsbjkwkbg")
    return positives, tp, fp

def decode_np(loc, priors, variances):
    """Decode locations from predictions using priors to undo
    the encoding we did for offset regression at train time.
    Args:
        loc (tensor): location predictions for loc layers,
            Shape: [num_priors,4]
        priors (tensor): Prior boxes in center-offset form.
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        decoded bounding box predictions
    """
    boxes = np.concatenate((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * np.exp(loc[:, 2:] * variances[1])), axis=1)

    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes

def perClass_iou_np(box, label, boxes, labels):
    iou = IoU_point_np(box, boxes)
    iou[labels != label] = 0.0
    return iou

def IoU_point_np(box, boxes):
    """Find intersection over union
        Args:
            box: (tensor) One box [xmin,ymin,xmax,ymax]; Shape: [4].
            boxes: (tensor)  Shape:[N, 4].
        Return:
            Intersection over union. Shape: [N]
        """
    A = np.maximum(box[:2], boxes[:, :2])
    B = np.minimum(box[2:], boxes[:, 2:])
    interArea = np.maximum(B[:, 0] - A[:, 0], 0) * np.maximum(B[:, 1] - A[:, 1], 0)
    boxArea = (box[2] - box[0]) * (box[3] - box[1])

    boxesArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # compute the intersection over union
    union = boxArea + boxesArea - interArea
    iou = interArea / union
    # return the intersection over union value
    return iou

def nms_np(boxes, scores_all, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [remaining_num, 4].
        scores_all: (tensor) The class predscores for the img, Shape:[remaining_num, classes].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """
    if boxes.shape[0] == 0:
        return np.zeros((0, 4)), np.zeros((0)), []

    scores = scores_all.max(axis=1)
    labels = np.argmax(scores_all, axis=1)

    idx = np.argsort(scores)[::-1]
    top_k = min(top_k, len(boxes))
    idx = idx[:top_k]

    boxes = boxes[idx]
    labels = labels[idx]
    scores = scores[idx]

    # box_x_min = np.minimum(boxes[:, 0], boxes[:, 2])[:, np.newaxis]
    # box_x_max = np.maximum(boxes[:, 0], boxes[:, 2])[:, np.newaxis]
    # box_y_min = np.minimum(boxes[:, 1], boxes[:, 3])[:, np.newaxis]
    # box_y_max = np.maximum(boxes[:, 1], boxes[:, 3])[:, np.newaxis]
    # fixed_boxes = np.concatenate((box_x_min, box_y_min, box_x_max, box_y_max), axis=1)


    keep_boxes  = np.zeros((top_k, 4))
    keep_labels = np.zeros((top_k))
    keep_scores = np.zeros((top_k))

    # first box
    keep_boxes[0]  = boxes[0]
    keep_labels[0] = labels[0]
    keep_scores[0] = scores[0]
    count = 1
    i = 1

    while i < top_k:
        iou_compare = perClass_iou_np(boxes[i], labels[i], keep_boxes[:count], keep_labels[:count])
        isIntersected = iou_compare > overlap
        if not np.any(isIntersected):
            keep_boxes[count] = boxes[i]
            keep_labels[count] = labels[i]
            keep_scores[count] = scores[i]
            count += 1

        i += 1

    return keep_boxes[:count], keep_labels[:count], keep_scores[:count]