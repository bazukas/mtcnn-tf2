import numpy as np
import tensorflow as tf

from .box_utils import _preprocess


def first_stage(img, pnet, scale, threshold, max_output_size=100):
    # scale the image and convert it to a float array
    height, width, _ = img.shape
    hs = int(np.ceil(height * scale))
    ws = int(np.ceil(width * scale))
    img = tf.image.resize(img, (hs, ws))

    img = _preprocess(img)
    offsets, probs = pnet(img)
    probs = probs[0, :, :, 1]
    # probs: probability of a face at each sliding window
    # offsets: transformations to true bounding boxes

    boxes = _generate_bboxes(probs, offsets, scale, threshold)
    if len(boxes) == 0:
        return None
    keep = tf.image.non_max_suppression(boxes[:, 0:4], boxes[:, 4], max_output_size, iou_threshold=0.5)
    return tf.gather(boxes, keep)


def _generate_bboxes(probs, offsets, scale, threshold):
    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    # indices of boxes where there is probably a face
    inds = tf.where(probs > threshold)
    if inds.shape[0] == 0:
        return np.array([])

    offsets = tf.gather_nd(offsets[0, :, :, :], inds)
    score = tf.expand_dims(tf.gather_nd(probs, inds), axis=1)

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    inds = tf.cast(inds, tf.float32)
    bounding_boxes = np.hstack([
        np.expand_dims(np.round((stride * inds[:, 1] + 1) / scale), 1),
        np.expand_dims(np.round((stride * inds[:, 0] + 1) / scale), 1),
        np.expand_dims(np.round((stride * inds[:, 1] + 1 + cell_size) / scale), 1),
        np.expand_dims(np.round((stride * inds[:, 0] + 1 + cell_size) / scale), 1),
        score, offsets
    ])
    return bounding_boxes
