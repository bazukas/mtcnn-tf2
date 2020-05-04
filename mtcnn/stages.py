import math
import tensorflow as tf

from .box_utils import calibrate_box, convert_to_square, get_image_boxes


def stage_one(img, pnet, scales, threshold, nms_threshold, max_output_size):
    bboxes = []
    offsets = []
    scores = []

    # run P-Net on different scales
    for s in scales:
        bb, o, s = _run_pnet(img, pnet, scale=s, threshold=threshold,
                             max_output_size=max_output_size)
        bboxes.append(bb)
        offsets.append(o)
        scores.append(s)
    # collect boxes (and offsets, and scores) from different scales
    bboxes = tf.concat(bboxes, 0)
    offsets = tf.concat(offsets, 0)
    scores = tf.concat(scores, 0)
    if len(bboxes) == 0:
        return bboxes

    keep = tf.image.non_max_suppression(bboxes, scores, max_output_size, iou_threshold=nms_threshold)
    bboxes = tf.gather(bboxes, keep)
    offsets = tf.gather(offsets, keep)
    scores = tf.gather(scores, keep)

    # use offsets predicted by pnet to transform bounding boxes
    bboxes = calibrate_box(bboxes, offsets)
    bboxes = convert_to_square(bboxes)
    return bboxes


def stage_two(img, bboxes, rnet, threshold, nms_threshold, max_output_size):
    img_boxes = get_image_boxes(bboxes, img, size=24)
    offsets, probs = rnet(img_boxes)

    keep = tf.where(probs[:, 1] > threshold)[:, 0]
    bboxes = tf.gather(bboxes, keep)
    offsets = tf.gather(offsets, keep)
    scores = tf.reshape(tf.gather(probs[:, 1], keep), [-1])

    keep = tf.image.non_max_suppression(bboxes, scores, max_output_size, nms_threshold)
    bboxes = tf.gather(bboxes, keep)
    offsets = tf.gather(offsets, keep)
    bboxes = calibrate_box(bboxes, offsets)
    bboxes = convert_to_square(bboxes)
    return bboxes


def stage_three(img, bboxes, onet, threshold, nms_threshold, max_output_size):
    img_boxes = get_image_boxes(bboxes, img, size=48)
    landmarks, offsets, probs = onet(img_boxes)

    keep = tf.where(probs[:, 1] > threshold)[:, 0]
    bboxes = tf.gather(bboxes, keep)
    offsets = tf.gather(offsets, keep)
    scores = tf.reshape(tf.gather(probs[:, 1], keep), [-1])
    landmarks = tf.gather(landmarks, keep)

    # compute landmark points
    width = tf.expand_dims(bboxes[:, 2] - bboxes[:, 0] + 1.0, 1)
    height = tf.expand_dims(bboxes[:, 3] - bboxes[:, 1] + 1.0, 1)
    xmin = tf.expand_dims(bboxes[:, 0], 1)
    ymin = tf.expand_dims(bboxes[:, 1], 1)
    landmarks = tf.concat([xmin + width * landmarks[:, 0:5],
                           ymin + height * landmarks[:, 5:10]], 1)

    bboxes = calibrate_box(bboxes, offsets)
    keep = tf.image.non_max_suppression(bboxes, scores, max_output_size, nms_threshold)
    bboxes = tf.gather(bboxes, keep)
    landmarks = tf.gather(landmarks, keep)
    return bboxes, landmarks


def _run_pnet(img, pnet, scale, threshold, max_output_size=100):
    # scale the image and convert it to a float array
    height, width, _ = img.shape
    hs = int(math.ceil(height * scale))
    ws = int(math.ceil(width * scale))
    img = tf.image.resize(img, (hs, ws))

    offsets, probs = pnet(tf.expand_dims(img, 0))
    probs = probs[0, :, :, 1]
    # probs: probability of a face at each sliding window
    # offsets: transformations to true bounding boxes

    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return boxes[:, 0:4], boxes[:, 5:], boxes[:, 4]
    keep = tf.image.non_max_suppression(boxes[:, 0:4], boxes[:, 4], max_output_size, iou_threshold=0.5)
    boxes = tf.gather(boxes, keep)
    return boxes[:, 0:4], boxes[:, 5:], boxes[:, 4]


def _generate_bboxes(probs, offsets, scale, threshold):
    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = tf.where(probs > threshold)
    if inds.shape[0] == 0:
        return tf.zeros((0, 9))

    offsets = tf.gather_nd(offsets[0, :, :, :], inds)
    score = tf.expand_dims(tf.gather_nd(probs, inds), axis=1)

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    inds = tf.cast(inds, tf.float32)
    bounding_boxes = tf.concat([
        tf.expand_dims(tf.math.round((stride * inds[:, 1] + 1) / scale), 1),
        tf.expand_dims(tf.math.round((stride * inds[:, 0] + 1) / scale), 1),
        tf.expand_dims(tf.math.round((stride * inds[:, 1] + 1 + cell_size) / scale), 1),
        tf.expand_dims(tf.math.round((stride * inds[:, 0] + 1 + cell_size) / scale), 1),
        score, offsets
    ], 1)
    return bounding_boxes
