from .nets import PNet, RNet, ONet
from .detect import detect_faces


class MTCNN(object):
    def __init__(self, pnet_path, rnet_path, onet_path):
        self.pnet = PNet.load(pnet_path)
        self.rnet = RNet.load(rnet_path)
        self.onet = ONet.load(onet_path)

    def detect(self, img):
        return detect_faces(img, self.pnet, self.rnet, self.onet)
