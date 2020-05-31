import tensorflow as tf
from .nets import PNet, RNet, ONet
from .box_utils import calibrate_box, convert_to_square, get_image_boxes, generate_bboxes, preprocess


DEF_THRESHOLDS = [0.7, 0.8, 0.9]
DEF_NMS_THRESHOLDS = [0.6, 0.6, 0.6]


class MTCNN(object):
    """ Top level class for mtcnn detection """
    def __init__(self, pnet_path, rnet_path, onet_path,
                 min_face_size=20.0,
                 thresholds=None,
                 nms_thresholds=None,
                 max_output_size=300):
        self.pnet = PNet.load(pnet_path)
        self.rnet = RNet.load(rnet_path)
        self.onet = ONet.load(onet_path)
        self.min_face_size = min_face_size
        self.thresholds = thresholds or DEF_THRESHOLDS
        self.nms_thresholds = nms_thresholds or DEF_NMS_THRESHOLDS
        self.max_output_size = max_output_size

    def detect(self, img):
        """Detect faces and facial landmarks on an image

        Parameters:
            img: rgb image, numpy array of shape [h, w, 3]

        Returns:
            bboxes: float tensor of shape [n, 4], face bounding boxes
            landmarks: float tensor of shape [n, 10], 5 facial landmarks,
                        first 5 numbers of array are x coords, last are y coords
            scores: float tensor of shape [n], confidence scores
        """
        img = tf.convert_to_tensor(img, tf.float32)
        bboxes = self.stage_one(img)
        if len(bboxes) == 0:
            return [], [], []
        bboxes = self.stage_two(img, bboxes, img.shape[0], img.shape[1], bboxes.shape[0])
        if len(bboxes) == 0:
            return [], [], []
        bboxes, landmarks, scores = self.stage_three(img, bboxes,
                                                     img.shape[0], img.shape[1], bboxes.shape[0])
        return bboxes, landmarks, scores

    def get_scales(self, height, width):
        """Compute scaling factors for given image dimensions

        Parameters:
            height: float
            width: float

        Returns:
            list of floats, scaling factors
        """
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

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32)])
    def stage_one_scale(self, img, height, width, scale):
        """Perform stage one part with a given scaling factor

        Parameters:
            img: rgb image, float tensor of shape [h, w, 3]
            height: image height, float
            width: image width, float
            scale: scaling factor, float

        Returns:
            float tensor of shape [n, 9]
        """
        hs = tf.math.ceil(height * scale)
        ws = tf.math.ceil(width * scale)
        img_in = tf.image.resize(img, (hs, ws))
        img_in = preprocess(img_in)
        img_in = tf.expand_dims(img_in, 0)

        offsets, probs = self.pnet(img_in)
        boxes = generate_bboxes(probs[0], offsets[0], scale, self.thresholds[0])
        if len(boxes) == 0:
            return boxes
        keep = tf.image.non_max_suppression(boxes[:, 0:4], boxes[:, 4], self.max_output_size,
                                            iou_threshold=0.5)

        boxes = tf.gather(boxes, keep)
        return boxes

    def stage_one(self, img):
        """Run stage one on the input image

        Parameters:
            img: rgb image, float tensor of shape [h, w, 3]

        Returns:
            float tensor of shape [n, 4], predicted bounding boxes
        """
        height, width, _ = img.shape
        scales = self.get_scales(height, width)

        boxes = []

        # run P-Net on different scales
        for s in scales:
            boxes.append(self.stage_one_scale(img, height, width, s))
        # collect boxes (and offsets, and scores) from different scales
        boxes = tf.concat(boxes, 0)
        if boxes.shape[0] == 0:
            return tf.zeros((0, 4))

        bboxes, scores, offsets = boxes[:, :4], boxes[:, 4], boxes[:, 5:]
        # use offsets predicted by pnet to transform bounding boxes
        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)

        keep = tf.image.non_max_suppression(bboxes, scores, self.max_output_size,
                                            iou_threshold=self.nms_thresholds[0])
        bboxes = tf.gather(bboxes, keep)
        return bboxes

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.int32)])
    def stage_two(self, img, bboxes, height, width, num_boxes):
        """Run stage two on the input image

        Parameters:
            img: rgb image, float tensor of shape [h, w, 3]
            bboxes: bounding boxes from stage one, float tensor of shape [n, 4]
            height: image height, float
            width: image width, float
            num_boxes: number of rows in bboxes, int

        Returns:
            float tensor of shape [n, 4], predicted bounding boxes
        """
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=24)
        offsets, probs = self.rnet(img_boxes)

        keep = tf.where(probs[:, 1] > self.thresholds[1])[:, 0]
        bboxes = tf.gather(bboxes, keep)
        offsets = tf.gather(offsets, keep)
        scores = tf.gather(probs[:, 1], keep)

        bboxes = calibrate_box(bboxes, offsets)
        bboxes = convert_to_square(bboxes)

        keep = tf.image.non_max_suppression(bboxes, scores,
                                            self.max_output_size, self.nms_thresholds[1])
        bboxes = tf.gather(bboxes, keep)
        return bboxes

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None, 3), dtype=tf.float32),
                         tf.TensorSpec(shape=(None, 4), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.float32),
                         tf.TensorSpec(shape=(), dtype=tf.int32)])
    def stage_three(self, img, bboxes, height, width, num_boxes):
        """Run stage three on the input image

        Parameters:
            img: rgb image, float tensor of shape [h, w, 3]
            bboxes: bounding boxes from stage two, float tensor of shape [n, 4]
            height: image height, float
            width: image width, float
            num_boxes: number of rows in bboxes, int

        Returns:
            bboxes: float tensor of shape [n, 4], face bounding boxes
            landmarks: float tensor of shape [n, 10], 5 facial landmarks,
                        first 5 numbers of array are x coords, last are y coords
            scores: float tensor of shape [n], confidence scores
        """
        img_boxes = get_image_boxes(bboxes, img, height, width, num_boxes, size=48)
        landmarks, offsets, probs = self.onet(img_boxes)

        keep = tf.where(probs[:, 1] > self.thresholds[2])[:, 0]
        bboxes = tf.gather(bboxes, keep)
        offsets = tf.gather(offsets, keep)
        scores = tf.gather(probs[:, 1], keep)
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
        scores = tf.gather(scores, keep)
        return bboxes, landmarks, scores
