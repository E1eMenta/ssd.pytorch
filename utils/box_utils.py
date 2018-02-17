import numpy as np

# def match_(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
#     """Match each prior box with the ground truth box of the highest jaccard
#     overlap, encode the bounding boxes, then return the matched indices
#     corresponding to both confidence and location preds.
#     Args:
#         threshold: (float) The overlap threshold used when mathing boxes.
#         truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
#         priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
#         variances: (tensor) Variances corresponding to each prior coord,
#             Shape: [num_priors, 4].
#         labels: (tensor) All the class labels for the image, Shape: [num_obj].
#         loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
#         conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
#         idx: (int) current batch index
#     Return:
#         The matched indices corresponding to 1)location and 2)confidence preds.
#     """
#     # jaccard index
#     overlaps = jaccard(
#         truths,
#         point_form(priors)
#     )
#     # (Bipartite Matching)
#     # [1,num_objects] best prior for each ground truth
#     best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
#     # [1,num_priors] best ground truth for each prior
#     best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
#     best_truth_idx.squeeze_(0)
#     best_truth_overlap.squeeze_(0)
#     best_prior_idx.squeeze_(1)
#     best_prior_overlap.squeeze_(1)
#     best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
#     # TODO refactor: index  best_prior_idx with long tensor
#     # ensure every gt matches with its prior of max overlap
#     for j in range(best_prior_idx.size(0)):
#         best_truth_idx[best_prior_idx[j]] = j
#     matches = truths[best_truth_idx]          # Shape: [num_priors,4]
#     conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
#     conf[best_truth_overlap < threshold] = 0  # label as background
#     loc = encode(matches, priors, variances)
#     loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
#     conf_t[idx] = conf  # [num_priors] top class label for each prior
#
# def IoU_point_np(box, boxes):
#     """Find intersection over union
#         Args:
#             box: (tensor) One box [xmin, ymin, xmax, ymax]; Shape: [4].
#             boxes: (tensor)  Shape:[N, 4].
#         Return:
#             Intersection over union. Shape: [N]
#         """
#     A = np.maximum(box[:2], boxes[:, :2])
#     B = np.minimum(box[2:], boxes[:, 2:])
#     # compute the area of intersection rectangle
#     interArea = np.maximum(B[:, 0] - A[:, 0], 0) * np.maximum(B[:, 1] - A[:, 1], 0)
#     # compute the area of all rectangles
#     boxArea = (box[2] - box[0]) * (box[3] - box[1])
#     boxesArea = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
#     # compute the intersection over union
#     union = boxArea + boxesArea - interArea
#     iou = interArea / union
#     # return the intersection over union value
#     return iou
#
# def point_form(boxes):
#     """ Convert center form(cx, cy, w, h) to point form(xmin, ymin, xmax, ymax)
#     representation for comparison to point form ground truth data.
#     Args:
#         boxes: (tensor) center-size boxes.
#     Return:
#         boxes: (tensor) Converted point form.
#     """
#     point_form = np.concatenate((boxes[:, :2] - boxes[:, 2:] / 2, boxes[:, :2] + boxes[:, 2:]/2), 1)
#     return point_form
#
# def center_form(boxes):
#     """ Convert center form(cx, cy, w, h) to point form(xmin, ymin, xmax, ymax)
#     representation for comparison to point form ground truth data.
#     Args:
#         boxes: (tensor) center-size boxes.
#     Return:
#         boxes: (tensor) Converted point form.
#     """
#     cx = (boxes[:, 2] - boxes[:, 0]) / 2
#     cy = (boxes[:, 3] - boxes[:, 1]) / 2
#     w = (boxes[:, 2] - boxes[:, 0])
#     h = (boxes[:, 3] - boxes[:, 1])
#     center_form = np.concatenate((cx, cy, w, h), 1)
#     return center_form
#
# def match_np(targets, priors, threshold, background_label=0):
#     """Match each prior box with the ground truth box of the highest jaccard
#         overlap, encode the bounding boxes, then return the matched indices
#         corresponding to both confidence and location preds.
#         Args:
#             threshold: (float) The overlap threshold used when mathing boxes.
#             targets: (tensor) Ground truth boxes, Shape: [num_obj, 5] (xmin, ymin, xmax, ymax, id).
#             priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
#             background_label (int) background class index
#         Return:
#             loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
#             conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
#             The matched indices corresponding to 1)location and 2)confidence preds.
#         """
#     batch_size = len(targets)
#     anchors_num = priors.shape[0]
#     loc  = np.zeros((batch_size, anchors_num, 4), dtype=np.float32)
#     labels = np.ones((batch_size, anchors_num), dtype=np.int32) * background_label
#     for idx, target in enumerate(targets):
#         bbox = target[:4]
#         class_id = target[4]
#         overlaps = IoU_point_np(bbox,
#                                 point_form(priors))
#
#         matched = overlaps > threshold
#         labels[matched] = class_id
#
#         c_bbox = center_form(bbox[np.newaxis, :])[0]
#
#         matched_priors = priors[matched]
#
#         g_cx =
#
#         loc[matched]

import torch

def one_hot(labels, depth):
    eye = torch.eye(depth)
    # print(list(labels.size()))
    size = list(labels.size())
    l = labels.view(-1)
    # print(l.size(), type(l.data))
    one_hot = eye[labels.data]
    new_size = size + [depth]
    one_hot.view(new_size)
    # print(list(one_hot.size()))
    return one_hot

def smooth_l1(tensor):
    l2 = 0.5 * (tensor ** 2.0)
    l1 = torch.abs(tensor) - 0.5

    smooth = torch.min(l2, l1)
    return smooth

def bboxes2offsets(targets, default_boxes, threshold=0.5):
    batch_size = len(targets)
    anchor_num = len(default_boxes)
    conf = []
    loc = []

    for batch_idx, target in enumerate(targets):
        boxes = target[:, :4]
        classes = target[:, 4].long()
        point_defaults = point_form(default_boxes)
        overlaps = iou(boxes, point_defaults)# [#obj,anchor_num]

        max_iou, max_idx = overlaps.max(0)  # [1,anchor_num]
        max_idx.squeeze_(0)                 # [anchor_num,]
        max_iou.squeeze_(0)                     # [anchor_num,]

        boxes = boxes[max_idx]  # [anchor_num,4]
        # variances = [0.1, 0.2]  #WTF????
        variances = [1, 1]  #WTF????
        cxcy = (boxes[:, :2] + boxes[:, 2:]) / 2 - default_boxes[:, :2]  # [anchor_num,2]
        cxcy /= variances[0] * default_boxes[:, 2:]
        wh = (boxes[:, 2:] - boxes[:, :2]) / default_boxes[:, 2:]  # [anchor_num,2]
        wh = torch.log(wh) / variances[1]
        loc_i = torch.cat([cxcy, wh], 1)  # [anchor_num,4]

        conf_i = 1 + classes[max_idx]  # [anchor_num,], background class = 0
        conf_i[max_iou < threshold] = 0

        conf.append(conf_i)
        loc.append(loc_i)

    conf = torch.stack(conf)
    loc = torch.stack(loc)


    return conf, loc

def point_form(boxes):
    """ Convert prior_boxes to (xmin, ymin, xmax, ymax)
    representation for comparison to point form ground truth data.
    Args:
        boxes: (tensor) center-size default boxes from priorbox layers.
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    """ Convert prior_boxes to (cx, cy, w, h)
    representation for comparison to center-size form ground truth data.
    Args:
        boxes: (tensor) point_form boxes
    Return:
        boxes: (tensor) Converted xmin, ymin, xmax, ymax form of boxes.
    """
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def intersect(box_a, box_b):
    """ We resize both tensors to [A,B,2] without new malloc:
    [A,2] -> [A,1,2] -> [A,B,2]
    [B,2] -> [1,B,2] -> [A,B,2]
    Then we compute the area of intersect between box_a and box_b.
    Args:
      box_a: (tensor) bounding boxes, Shape: [A,4].
      box_b: (tensor) bounding boxes, Shape: [B,4].
    Return:
      (tensor) intersection area, Shape: [A,B].
    """
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def jaccard(box_a, box_b):
    """Compute the jaccard overlap of two sets of boxes.  The jaccard overlap
    is simply the intersection over union of two boxes.  Here we operate on
    ground truth boxes and default boxes.
    E.g.:
        A ∩ B / A ∪ B = A ∩ B / (area(A) + area(B) - A ∩ B)
    Args:
        box_a: (tensor) Ground truth bounding boxes, Shape: [num_objects,4]
        box_b: (tensor) Prior boxes from priorbox layers, Shape: [num_priors,4]
    Return:
        jaccard overlap: (tensor) Shape: [box_a.size(0), box_b.size(0)]
    """
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):
    """Match each prior box with the ground truth box of the highest jaccard
    overlap, encode the bounding boxes, then return the matched indices
    corresponding to both confidence and location preds.
    Args:
        threshold: (float) The overlap threshold used when mathing boxes.
        truths: (tensor) Ground truth boxes, Shape: [num_obj, num_priors].
        priors: (tensor) Prior boxes from priorbox layers, Shape: [n_priors,4].
        variances: (tensor) Variances corresponding to each prior coord,
            Shape: [num_priors, 4].
        labels: (tensor) All the class labels for the image, Shape: [num_obj].
        loc_t: (tensor) Tensor to be filled w/ endcoded location targets.
        conf_t: (tensor) Tensor to be filled w/ matched indices for conf preds.
        idx: (int) current batch index
    Return:
        The matched indices corresponding to 1)location and 2)confidence preds.
    """
    # jaccard index
    overlaps = jaccard(
        truths,
        point_form(priors)
    )
    # (Bipartite Matching)
    # [1,num_objects] best prior for each ground truth
    best_prior_overlap, best_prior_idx = overlaps.max(1, keepdim=True)
    # [1,num_priors] best ground truth for each prior
    best_truth_overlap, best_truth_idx = overlaps.max(0, keepdim=True)
    best_truth_idx.squeeze_(0)
    best_truth_overlap.squeeze_(0)
    best_prior_idx.squeeze_(1)
    best_prior_overlap.squeeze_(1)
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    # ensure every gt matches with its prior of max overlap
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]
    conf[best_truth_overlap < threshold] = 0  # label as background
    loc = encode(matches, priors, variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

    # import numpy as np
    # import cv2
    #
    # img = np.zeros((300,300, 3))
    # print(priors[torch.nonzero(conf)].size())

    # print(truths.size())
    # print(torch.nonzero(conf).size())
    # input("sgegeg")
def iou(box1, box2):
    '''Compute the intersection over union of two set of boxes, each box is [x1,y1,x2,y2].
    Args:
      box1: (tensor) bounding boxes, sized [N,4].
      box2: (tensor) bounding boxes, sized [M,4].
    Return:
      (tensor) iou, sized [N,M].
    '''

    N = box1.size(0)
    M = box2.size(0)
    lt = torch.max(
        box1[:, :2].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, :2].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )
    rb = torch.min(
        box1[:, 2:].unsqueeze(1).expand(N, M, 2),  # [N,2] -> [N,1,2] -> [N,M,2]
        box2[:, 2:].unsqueeze(0).expand(N, M, 2),  # [M,2] -> [1,M,2] -> [N,M,2]
    )

    wh = rb - lt  # [N,M,2]
    wh[wh < 0] = 0  # clip at 0
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    area1 = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])  # [N,]
    area2 = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])  # [M,]
    area1 = area1.unsqueeze(1).expand_as(inter)  # [N,] -> [N,1] -> [N,M]
    area2 = area2.unsqueeze(0).expand_as(inter)  # [M,] -> [1,M] -> [N,M]

    iou = inter / (area1 + area2 - inter)
    return iou
def encode(matched, priors, variances):
    """Encode the variances from the priorbox layers into the ground truth boxes
    we have matched (based on jaccard overlap) with the prior boxes.
    Args:
        matched: (tensor) Coords of ground truth for each prior in point-form
            Shape: [num_priors, 4].
        priors: (tensor) Prior boxes in center-offset form
            Shape: [num_priors,4].
        variances: (list[float]) Variances of priorboxes
    Return:
        encoded boxes (tensor), Shape: [num_priors, 4]
    """

    # dist b/t match center and prior's center
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    # encode variance
    g_cxcy /= (variances[0] * priors[:, 2:])
    # match wh / prior wh
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    # return target for smooth_l1_loss
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]


# Adapted from https://github.com/Hakuyume/chainer-ssd
def decode(loc, priors, variances):
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

    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    """Utility function for computing log_sum_exp while determining
    This will be used to determine unaveraged confidence loss across
    all examples in a batch.
    Args:
        x (Variable(tensor)): conf_preds from conf layers
    """
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max


# Original author: Francisco Massa:
# https://github.com/fmassa/object-detection.torch
# Ported to PyTorch by Max deGroot (02/01/2017)
def nms(boxes, scores, overlap=0.5, top_k=200):
    """Apply non-maximum suppression at test time to avoid detecting too many
    overlapping bounding boxes for a given object.
    Args:
        boxes: (tensor) The location preds for the img, Shape: [num_priors,4].
        scores: (tensor) The class predscores for the img, Shape:[num_priors].
        overlap: (float) The overlap thresh for suppressing unnecessary boxes.
        top_k: (int) The Maximum number of box preds to consider.
    Return:
        The indices of the kept boxes with respect to num_priors.
    """

    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    # I = I[v >= 0.01]
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()

    # keep = torch.Tensor()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w*h
        # IoU = i / (area(a) + area(b) - i)
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter/union  # store result in iou
        # keep only elements with an IoU <= overlap
        idx = idx[IoU.le(overlap)]
    return keep, count

