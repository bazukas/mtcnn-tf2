import numpy as np
import tensorflow as tf

from .box_utils import calibrate_box, get_image_boxes, convert_to_square
from .stages import first_stage


def detect_faces(img,
                 pnet, rnet, onet,
                 min_face_size=20.0,
                 thresholds=[0.6, 0.7, 0.8],
                 nms_thresholds=[0.7, 0.7, 0.7],
                 max_output_size=100):

    # BUILD AN IMAGE PYRAMID
    height, width, _ = img.shape
    min_length = min(height, width)

    min_detection_size = 12
    factor = 0.707  # sqrt(0.5)

    # scales for scaling the image
    scales = []

    # scales the image so that
    # minimum size that we can detect equals to
    # minimum face size that we want to detect
    m = min_detection_size / min_face_size
    min_length *= m

    factor_count = 0
    while min_length > min_detection_size:
        scales.append(m * factor**factor_count)
        min_length *= factor
        factor_count += 1

    # STAGE 1

    # it will be returned
    bounding_boxes = []

    # run P-Net on different scales
    for s in scales:
        boxes = first_stage(img, pnet, scale=s, threshold=thresholds[0],
                            max_output_size=max_output_size)
        bounding_boxes.append(boxes)

    # collect boxes (and offsets, and scores) from different scales
    bounding_boxes = [i for i in bounding_boxes if i is not None]
    if not bounding_boxes:
        return [], []
    bounding_boxes = np.vstack(bounding_boxes)

    keep = tf.image.non_max_suppression(bounding_boxes[:, 0:4], bounding_boxes[:, 4],
                                        max_output_size, iou_threshold=nms_thresholds[0])
    bounding_boxes = tf.gather(bounding_boxes, keep)

    # use offsets predicted by pnet to transform bounding boxes
    bounding_boxes = calibrate_box(bounding_boxes[:, 0:5], bounding_boxes[:, 5:])
    # shape [n_boxes, 5]

    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 2

    img_boxes = get_image_boxes(bounding_boxes, img, size=24)
    offsets, probs = rnet(img_boxes)

    keep = np.where(probs[:, 1] > thresholds[1])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = tf.reshape(tf.gather(probs[:, 1], keep), [-1])
    offsets = tf.gather(offsets, keep)

    keep = tf.image.non_max_suppression(bounding_boxes[:, 0:4], bounding_boxes[:, 4],
                                        max_output_size, nms_thresholds[1])
    bounding_boxes = tf.gather(bounding_boxes, keep)
    bounding_boxes = calibrate_box(bounding_boxes, tf.gather(offsets, keep))
    bounding_boxes = convert_to_square(bounding_boxes)
    bounding_boxes[:, 0:4] = np.round(bounding_boxes[:, 0:4])

    # STAGE 3

    img_boxes = get_image_boxes(bounding_boxes, img, size=48)
    if len(img_boxes) == 0:
        return [], []
    landmarks, offsets, probs = onet(img_boxes)

    keep = np.where(probs[:, 1] > thresholds[2])[0]
    bounding_boxes = bounding_boxes[keep]
    bounding_boxes[:, 4] = tf.reshape(tf.gather(probs[:, 1], keep), [-1])
    offsets = tf.gather(offsets, keep)
    landmarks = tf.gather(landmarks, keep)

    # compute landmark points
    width = np.expand_dims(bounding_boxes[:, 2] - bounding_boxes[:, 0] + 1.0, 1)
    height = np.expand_dims(bounding_boxes[:, 3] - bounding_boxes[:, 1] + 1.0, 1)
    xmin = np.expand_dims(bounding_boxes[:, 0], 1)
    ymin = np.expand_dims(bounding_boxes[:, 1], 1)
    landmarks = tf.concat([xmin + width * landmarks[:, 0:5],
                           ymin + height * landmarks[:, 5:10]], 1)

    bounding_boxes = calibrate_box(bounding_boxes, offsets)
    keep = tf.image.non_max_suppression(bounding_boxes[:, 0:4], bounding_boxes[:, 4],
                                        max_output_size, nms_thresholds[2])
    bounding_boxes = tf.gather(bounding_boxes, keep)
    landmarks = tf.gather(landmarks, keep)

    return bounding_boxes, landmarks
