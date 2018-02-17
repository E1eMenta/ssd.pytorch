import numpy as np
import cv2
def draw_target(image, target, color=(255,0,0),thickness=2):
    img = image.copy()

    w = img.shape[1]
    h = img.shape[0]

    target[:, 0] *= w
    target[:, 1] *= h
    target[:, 2] *= w
    target[:, 3] *= h

    target = target.astype(np.int32)

    for bbox in target:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, thickness)

    return img

def draw_true_defaults(image, conf_target, defaults, color=(255,0,0),thickness=1):
    img = image.copy()
    width = img.shape[0]
    height = img.shape[1]

    for i, bbox in enumerate(defaults):
        if (conf_target[i] != 0):
            x = bbox[0] * width
            y = bbox[1] * height
            w = bbox[2] * width
            h = bbox[3] * height
            cv2.rectangle(img,
                          (int(x - w / 2), int(y - h / 2)),
                          (int(x + w / 2), int(y + h / 2)),
                          color, thickness)

    return img

def draw_true_offsets(image, conf_target, loc_target, defaults, color=(255,0,0), thickness=1):
    img = image.copy()
    width = img.shape[0]
    height = img.shape[1]

    for i, bbox in enumerate(defaults):
        if (conf_target[i] != 0):
            o = loc_target[i]
            # print(o.shape)
            tx = o[0]
            ty = o[1]
            tw = o[2]
            th = o[3]
            x = bbox[0] * width
            y = bbox[1] * height
            w = bbox[2] * width
            h = bbox[3] * height
            x = tx * w + x
            y = ty * h + y
            w = w * np.exp(tw)
            h = h * np.exp(th)
            cv2.rectangle(img,
                          (int(x - w / 2), int(y - h / 2)),
                          (int(x + w / 2), int(y + h / 2)),
                          color, thickness)

    return img

def draw_pred_defaults(image, conf_pred, defaults, color=(255,0,0),thickness=1, threshold=0.5):
    img = image.copy()
    width = img.shape[0]
    height = img.shape[1]

    for i, bbox in enumerate(defaults):
        max_score = np.max(conf_pred[i, 1:])
        if (max_score > threshold):
            print("apngnpeg")
            x = bbox[0] * width
            y = bbox[1] * height
            w = bbox[2] * width
            h = bbox[3] * height
            cv2.rectangle(img,
                          (int(x - w / 2), int(y - h / 2)),
                          (int(x + w / 2), int(y + h / 2)),
                          color, thickness)

    return img

def draw_pred_offsets(image, conf_pred, loc_pred, defaults, color=(255,0,0), thickness=1, threshold=0.5):
    img = image.copy()
    width = img.shape[0]
    height = img.shape[1]

    for i, bbox in enumerate(defaults):
        max_score = np.max(conf_pred[i, 1:])
        if (max_score > threshold):
            o = loc_pred[i]
            # print(o.shape)
            tx = o[0]
            ty = o[1]
            tw = o[2]
            th = o[3]
            x = bbox[0] * width
            y = bbox[1] * height
            w = bbox[2] * width
            h = bbox[3] * height
            x = tx * w + x
            y = ty * h + y
            w = w * np.exp(tw)
            h = h * np.exp(th)
            cv2.rectangle(img,
                          (int(x - w / 2), int(y - h / 2)),
                          (int(x + w / 2), int(y + h / 2)),
                          color, thickness)

    return img
