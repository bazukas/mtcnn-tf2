import tensorflow as tf
from .nets import PNet, RNet, ONet
from .box_utils import calibrate_box, convert_to_square, get_image_boxes, generate_bboxes


DEF_THRESHOLDS = [0.6, 0.7, 0.8]
DEF_NMS_THRESHOLDS = [0.7, 0.7, 0.7]


class MTCNN(object):
    def __init__(self, pnet_path, rnet_path, onet_path,
                 min_face_size=20.0,
                 thresholds=None,
                 nms_thresholds=None,
                 max_output_size=100):
        self.pnet = PNet.load(pnet_path)
        self.rnet = RNet.load(rnet_path)
        self.onet = ONet.load(onet_path)
        self.min_face_size = min_face_size
        self.thresholds = thresholds or DEF_THRESHOLDS
        self.nms_thresholds = nms_thresholds or DEF_NMS_THRESHOLDS
        self.max_output_size = max_output_size

    def detect(self, img):
        img = self.preprocess(img)
        bboxes = self.stage_one(img)
        if len(bboxes) == 0:
            return [], []
        bboxes = self.stage_two(img, bboxes, img.shape[0], img.shape[1], bboxes.shape[0])
        if len(bboxes) == 0:
            return [], []
        bboxes, landmarks = self.stage_three(img, bboxes, img.shape[0], img.shape[1], bboxes.shape[0])
        return bboxes, landmarks

    def get_scales(self, height, width):
        min_length = min(height, width)
        min_detection_size = 12
        factor = 0.707  # sqrt(0.5)
        # scales for scaling the image
        scales = []
        # scales the image so that
        # minimum size that we can detect equals to
        # minimum face size that we want to detect
        m = min_detection_size / self.min_face_size
        min_length *= m
        factor_count = 0
        while min_length > min_detection_size:
            scales.append(m * factor**factor_count)
            min_length *= factor
            factor_count += 1
        return scales

    def preprocess(self, img):
        img = (img - 127.5) * 0.0078125
        return img

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(1, None, None, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32)])
    def stage_one_scale(self, img, scale):

        offsets, probs = self.pnet(img)
        probs = probs[0, :, :, 1]
        # probs: probability of a face at each sliding window
        # offsets: transformations to true bounding boxes

        boxes = generate_bboxes(probs, offsets, scale, self.thresholds[0])
        if len(boxes) == 0:
            return boxes[:, 0:4], boxes[:, 5:], boxes[:, 4]
        keep = tf.image.non_max_suppression(boxes[:, 0:4], boxes[:, 4],
                                            self.max_output_size, iou_threshold=0.5)
        boxes = tf.gather(boxes, keep)
        return boxes[:, 0:4], boxes[:, 5:], boxes[:, 4]

    def stage_one(self, img):
        height, width, _ = img.shape
        scales = self.get_scales(height, width)

        bboxes = []
        offsets = []
        scores = []

        img = tf.expand_dims(img, 0)
        # run P-Net on different scales
        for s in scales:
            # scale the image and convert it to a float array
            hs = tf.math.ceil(height * s)
            ws = tf.math.ceil(width * s)
            bb, o, s = self.stage_one_scale(tf.image.resize(img, (hs, ws)), s)
            bboxes.append(bb)
            offsets.append(o)
            scores.append(s)
        # collect boxes (and offsets, and scores) from different scales
        bboxes = tf.concat(bboxes, 0)
        offsets = tf.concat(offsets, 0)
        scores = tf.concat(scores, 0)
        if len(bboxes) == 0:
            return bboxes

        keep = tf.image.non_max_suppression(bboxes, scores, self.max_output_size,
                                            iou_threshold=self.nms_thresholds[0])
        bboxes = tf.gather(bboxes, keep)
        offsets = tf.gather(offsets, keep)
        scores = tf.gather(scores, keep)

        # use offsets predicted by pnet to transform bounding boxes
        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)
        return bboxes

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.int32)])
    def stage_two(self, img, bboxes, height, width, num_boxes):
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=24)
        offsets, probs = self.rnet(img_boxes)

        keep = tf.where(probs[:, 1] > self.thresholds[1])[:, 0]
        bboxes = tf.gather(bboxes, keep)
        offsets = tf.gather(offsets, keep)
        scores = tf.reshape(tf.gather(probs[:, 1], keep), [-1])

        keep = tf.image.non_max_suppression(bboxes, scores,
                                            self.max_output_size, self.nms_thresholds[1])
        bboxes = tf.gather(bboxes, keep)
        offsets = tf.gather(offsets, keep)
        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)
        return bboxes

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.int32)])
    def stage_three(self, img, bboxes, height, width, num_boxes):
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=48)
        landmarks, offsets, probs = self.onet(img_boxes)

        keep = tf.where(probs[:, 1] > self.thresholds[2])[:, 0]
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
        keep = tf.image.non_max_suppression(bboxes, scores,
                                            self.max_output_size, self.nms_thresholds[2])
        bboxes = tf.gather(bboxes, keep)
        landmarks = tf.gather(landmarks, keep)
        return bboxes, landmarks
