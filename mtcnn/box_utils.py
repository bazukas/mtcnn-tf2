import tensorflow as tf


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.

    Arguments:
        bboxes: a float numpy array of shape [n, 4].

    Returns:
        a float numpy array of shape [n, 4],
            squared bounding boxes.
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    h = y2 - y1 + 1.0
    w = x2 - x1 + 1.0
    max_side = tf.maximum(h, w)
    dx1 = x1 + w * 0.5 - max_side * 0.5
    dy1 = y1 + h * 0.5 - max_side * 0.5
    dx2 = dx1 + max_side - 1.0
    dy2 = dy1 + max_side - 1.0
    return tf.stack([
        tf.math.round(dx1),
        tf.math.round(dy1),
        tf.math.round(dx2),
        tf.math.round(dy2),
    ], 1)


def calibrate_box(bboxes, offsets):
    """Transform bounding boxes to be more like true bounding boxes.
    'offsets' is one of the outputs of the nets.

    Arguments:
        bboxes: a float numpy array of shape [n, 4].
        offsets: a float numpy array of shape [n, 4].

    Returns:
        an array of shape [n, 4], bboxes
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1 + 1.0
    h = y2 - y1 + 1.0
    w = tf.expand_dims(w, 1)
    h = tf.expand_dims(h, 1)

    translation = tf.concat([w, h, w, h], 1) * offsets
    return bboxes[:, 0:4] + translation


def get_image_boxes(bounding_boxes, img, size=24):
    """Cut out boxes from the image.

    Arguments:
        bounding_boxes: a float numpy array of shape [n, 4].
        img: image tensor
        size: an integer, size of cutouts.

    Returns:
        a float numpy array of shape [n, size, size, 3].
    """

    if bounding_boxes.shape[0] == 0:
        return tf.zeros((0, size, size, 3))
    height, width, _ = img.shape

    x1, y1, x2, y2 = _correct_bboxes(bounding_boxes, width, height)
    boxes = tf.stack([y1, x1, y2, x2], 1)
    img_boxes = tf.image.crop_and_resize(tf.expand_dims(img, 0), boxes,
                                         tf.zeros(boxes.shape[0], dtype=tf.int32),
                                         (size, size))
    return img_boxes


def _correct_bboxes(bboxes, width, height):
    """Crop boxes that are too big and get coordinates
    with respect to cutouts.

    Arguments:
        bboxes: a float numpy array of shape [n, 5],
            where each row is (xmin, ymin, xmax, ymax, score).
        width: a float number.
        height: a float number.

    Returns:
        x1, y1, x2, y2: a int numpy arrays of shape [n],
            corrected xmin, ymin, xmax, ymax.
    """
    x1 = tf.math.maximum(bboxes[:, 0], 0.0) / width
    y1 = tf.math.maximum(bboxes[:, 1], 0.0) / height
    x2 = tf.math.minimum(bboxes[:, 2], width - 1.0) / width
    y2 = tf.math.minimum(bboxes[:, 3], height - 1.0) / height
    return x1, y1, x2, y2
