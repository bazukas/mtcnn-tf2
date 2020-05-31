import tensorflow as tf
from tensorflow.keras import Model, layers


class PNet(Model):
    """ Proposal Network, receives an image and outputs
    bbox offset regressions and confidence scores for each sliding
    window of 12x12
    """
    def __init__(self):
        super(PNet, self).__init__()
        self.features = [
            # permute needed because of imported weights
            layers.Permute((2, 1, 3), name='permute'),
            layers.Conv2D(10, 3, 1, name='conv1'),
            layers.PReLU(shared_axes=[1, 2], name='prelu1'),
            layers.MaxPool2D(2, 2, padding='same', name='pool1'),

            layers.Conv2D(16, 3, 1, name='conv2'),
            layers.PReLU(shared_axes=[1, 2], name='prelu2'),

            layers.Conv2D(32, 3, 1, name='conv3'),
            layers.PReLU(shared_axes=[1, 2], name='prelu3'),
        ]

        self.conv4_1 = layers.Conv2D(2, 1, 1, name='conv4_1')
        self.conv4_2 = layers.Conv2D(4, 1, 1, name='conv4_2')

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, None, None, 3),
                                       dtype=tf.float32)])
    def call(self, x):
        for l in self.features:
            x = l(x)
        a = self.conv4_1(x)
        b = self.conv4_2(x)
        a = layers.Softmax()(a)
        # permute needed because of imported weights
        a = tf.transpose(a, (0, 2, 1, 3))
        b = tf.transpose(b, (0, 2, 1, 3))
        return b, a

    @classmethod
    def load(cls, path):
        m = cls()
        m.build((None, 12, 12, 3))
        m.load_weights(path)
        return m


class RNet(Model):
    """ Refine Network, receives image crops from PNet and outputs
    further offset refinements and confidence scores to filter out
    the predictions
    """
    def __init__(self):
        super(RNet, self).__init__()
        self.features = [
            # permute needed because of imported weights
            layers.Permute((2, 1, 3), name='permute'),
            layers.Conv2D(28, 3, 1, input_shape=(24, 24, 3), name='conv1'),
            layers.PReLU(shared_axes=[1, 2], name='prelu1'),
            layers.MaxPool2D(3, 2, padding='same', name='pool1'),

            layers.Conv2D(48, 3, 1, name='conv2'),
            layers.PReLU(shared_axes=[1, 2], name='prelu2'),
            layers.MaxPool2D(3, 2, name='pool2'),

            layers.Conv2D(64, 2, 1, name='conv3'),
            layers.PReLU(shared_axes=[1, 2], name='prelu3'),

            layers.Flatten(name='flatten'),
            layers.Dense(128, name='dense4'),
            layers.PReLU(name='prelu4'),
        ]

        self.dense5_1 = layers.Dense(2, name='dense5_1')
        self.dense5_2 = layers.Dense(4, name='dense5_2')

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 24, 24, 3),
                                       dtype=tf.float32)])
    def call(self, x):
        for l in self.features:
            x = l(x)
        a = self.dense5_1(x)
        b = self.dense5_2(x)
        a = layers.Softmax()(a)
        return b, a

    @classmethod
    def load(cls, path):
        m = cls()
        m.build((None, 24, 24, 3))
        m.load_weights(path)
        return m


class ONet(Model):
    """ Output Network, receives image crops from RNet and outputs
    final offset regressions, facial landmark positions and confidence scores
    """

    def __init__(self):
        super(ONet, self).__init__()
        self.features = [
            # permute needed because of imported weights
            layers.Permute((2, 1, 3), name='permute'),
            layers.Conv2D(32, 3, 1, input_shape=(48, 48, 3), name='conv1'),
            layers.PReLU(shared_axes=[1, 2], name='prelu1'),
            layers.MaxPool2D(3, 2, padding='same', name='pool1'),

            layers.Conv2D(64, 3, 1, name='conv2'),
            layers.PReLU(shared_axes=[1, 2], name='prelu2'),
            layers.MaxPool2D(3, 2, name='pool2'),

            layers.Conv2D(64, 3, 1, name='conv3'),
            layers.PReLU(shared_axes=[1, 2], name='prelu3'),
            layers.MaxPool2D(2, 2, padding='same', name='pool3'),

            layers.Conv2D(128, 2, 1, name='conv4'),
            layers.PReLU(shared_axes=[1, 2], name='prelu4'),

            layers.Flatten(name='flatten'),
            layers.Dense(256, name='dense5'),
            layers.PReLU(name='prelu5'),
        ]

        self.dense6_1 = layers.Dense(2, name='dense6_1')
        self.dense6_2 = layers.Dense(4, name='dense6_2')
        self.dense6_3 = layers.Dense(10, name='dense6_3')

    @tf.function(
        input_signature=[tf.TensorSpec(shape=(None, 48, 48, 3),
                                       dtype=tf.float32)])
    def call(self, x):
        for l in self.features:
            x = l(x)
        a = self.dense6_1(x)
        b = self.dense6_2(x)
        c = self.dense6_3(x)
        a = layers.Softmax()(a)
        return c, b, a

    @classmethod
    def load(cls, path):
        m = cls()
        m.build((None, 48, 48, 3))
        m.load_weights(path)
        return m
