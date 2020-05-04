from .nets import PNet, RNet, ONet
from .stages import stage_one, stage_two, stage_three


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
        scales = self.get_scales(img.shape[0], img.shape[1])

        bboxes = stage_one(img, self.pnet, scales, self.thresholds[0],
                           self.nms_thresholds[0], self.max_output_size)
        if len(bboxes) == 0:
            return [], []
        bboxes = stage_two(img, bboxes, self.rnet, self.thresholds[1],
                           self.nms_thresholds[1], self.max_output_size)
        if len(bboxes) == 0:
            return [], []
        bboxes, landmarks = stage_three(img, bboxes, self.onet, self.thresholds[2],
                                        self.nms_thresholds[2], self.max_output_size)
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
