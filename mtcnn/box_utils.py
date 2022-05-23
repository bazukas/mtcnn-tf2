from sklearn.preprocessing import scale
import tensorflow as tf


def convert_to_square(bboxes):
    """Convert bounding boxes to a square form.

    Parameters:
        bboxes: float tensor of shape [n, 4]

    Returns:
        float tensor of shape [n, 4]
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    print(x1, y1, x2, y2)
    h = y2 - y1
    w = x2 - x1
    max_side = tf.maximum(h, w)
    dx1 = x1 + w * 0.5 - max_side * 0.5
    dy1 = y1 + h * 0.5 - max_side * 0.5
    dx2 = dx1 + max_side
    dy2 = dy1 + max_side
    return tf.stack([
        tf.math.round(dx1),
        tf.math.round(dy1),
        tf.math.round(dx2),
        tf.math.round(dy2),
    ], 1)


def calibrate_box(bboxes, offsets):
    """Use offsets returned by a network to
    correct the bounding box coordinates.

    Parameters:
        bboxes: float tensor of shape [n, 4].
        offsets: float tensor of shape [n, 4].

    Returns:
        float tensor of shape [n, 4]
    """
    x1, y1, x2, y2 = [bboxes[:, i] for i in range(4)]
    w = x2 - x1
    h = y2 - y1

    translation = tf.stack([w, h, w, h], 1) * offsets
    return bboxes + translation


def get_image_boxes(bboxes, img, height, width, num_boxes, size=24):
    """Cut out boxes from the image.

    Parameters:
        bboxes: float tensor of shape [n, 4]
        img: image tensor
        height: float, image height
        width: float, image width
        num_boxes: int, number of rows in bboxes
        size: int, size of cutouts

    Returns:
        float tensor of shape [n, size, size, 3]
    """
    x1 = tf.math.maximum(bboxes[:, 0], 0.0) / width
    y1 = tf.math.maximum(bboxes[:, 1], 0.0) / height
    x2 = tf.math.minimum(bboxes[:, 2], width) / width
    y2 = tf.math.minimum(bboxes[:, 3], height) / height
    boxes = tf.stack([y1, x1, y2, x2], 1)
    img_boxes = tf.image.crop_and_resize(tf.expand_dims(img, 0), boxes,
                                         tf.zeros(num_boxes, dtype=tf.int32),
                                         (size, size))
    img_boxes = preprocess(img_boxes)
    print(img_boxes.shape)
    return img_boxes


def generate_bboxes(probs, offsets, scale, threshold):
    """Convert output of PNet to bouding boxes tensor.

    Parameters:
        probs: float tensor of shape [p, m, 2], output of PNet
        offsets: float tensor of shape [p, m, 4], output of PNet
        scale: float, scale of the input image
        threshold: float, confidence threshold

    Returns:
        float tensor of shape [n, 9]
    """

    # applying P-Net is equivalent, in some sense, to
    # moving 12x12 window with stride 2
    stride = 2
    cell_size = 12

    probs = probs[:, :, 1]
    # indices of boxes where there is probably a face
    # inds: N x 2
    inds = tf.where(probs > threshold)
    if inds.shape[0] == 0:
        return tf.zeros((0, 9))

    print('this is offsets:',offsets)
    # offsets: N x 4
    offsets = tf.gather_nd(offsets, inds)
    # score: N x 1
    score = tf.expand_dims(tf.gather_nd(probs, inds), axis=1)

    # P-Net is applied to scaled images
    # so we need to rescale bounding boxes back
    inds = tf.cast(inds, tf.float32)
    print('this is inds:',inds)
    # bounding_boxes: N x 9
    bounding_boxes = tf.concat([
        tf.expand_dims(tf.math.round((stride * inds[:, 1]) / scale), 1),
        tf.expand_dims(tf.math.round((stride * inds[:, 0]) / scale), 1),
        tf.expand_dims(tf.math.round((stride * inds[:, 1] + cell_size) / scale), 1),
        tf.expand_dims(tf.math.round((stride * inds[:, 0] + cell_size) / scale), 1),
        score, offsets
    ], 1)
    return bounding_boxes


def preprocess(img):
    """Preprocess image tensor before applying a network.

    Parameters:
        img: image tensor

    Returns:
        float tensor with shape of img
    """
    img = (img - 127.5) * 0.0078125
    return img

if __name__ == '__main__':
    # print(convert_to_square(tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])))

    # print(calibrate_box(tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]), 
    # tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])))

    #test get_image_boxes
    # boxes = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    # img = tf.ones((100, 100, 3))
    # height = 100
    # width = 100
    # num_boxes = 2
    # size = 24
    # get_image_boxes(boxes, img, height, width, num_boxes, size)

    # test generate_bboxes
    # probs = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    # offsets = tf.constant([[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]])
    # scale = 1.0
    # threshold = 0.5
    # generate_bboxes(probs, offsets, scale, threshold)

    #probs shape [p, m, 2]
    probs = tf.constant([[[0.1, 0.2], [0.5, 0.6], [0.7, 0.8]], [[0.1, 0.2], [0.5, 0.6], [0.7, 0.8]], [[0.1, 0.2], [0.5, 0.6], [0.7, 0.8]], [[0.1, 0.2], [0.5, 0.6], [0.7, 0.8]]])
    # print(probs)
    # probs = probs[:, :, 1]
    #offsets shape [p, m, 4]
    offsets = tf.constant([[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]], [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]])
    # scale = 1.0
    # threshold = 0.5
    # generate_bboxes(probs, offsets, scale, threshold)
    inds = tf.where(probs > 0.5)
    print(inds)
    print(inds[:,1])
    print(inds[:,0])
    print(inds[:,2])
    # # inds = tf.zeros((0, 9))
    # print(inds)
    # print(inds[:, 1])
    # tf.gather_nd(offsets, inds)


    